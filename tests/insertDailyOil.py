"""
石油日度盘后数据抓取与清洗系统
功能: 从 yahoo-fin 抓取原油价格、宏观指标，经过清洗与特征计算后，Upsert 到 MySQL

数据流程:
    1. 数据采集 (Fetcher) - 调用 yahoo-fin API 获取 WTI, Brent, USD, VIX, TNX, CRAK
    2. 数据清洗 (Pipeline) - IQR 异常值剔除, 线性插值, 特征工程
    3. 数据持久化 (Storage) - 分块 Upsert 到 MySQL

作者: OpenCode
日期: 2026-06-09

代码结构概览
1. 数据采集模块 (Fetcher)
- fetch_single_ticker() - 获取单个 ticker 数据
- fetch_all_data() - 外连接合并所有数据源
- 数据源映射：WTI(CL=F), Brent(BZ=F), USD(DX-Y.NYB), VIX(^VIX), TNX(^TNX), CRAK
2. 数据清洗模块 (Pipeline)
- filter_invalid_values() - 剔除 ≤0 的物理错误值
- remove_outliers_iqr() - 20日滚动窗口 IQR 异常值检测
- fill_missing_values() - 线性插值 + 前向填充
- calculate_derived_features() - 计算 wti_60dma, brent_wti_spread, wti_rsi, term_structure
3. 持久化模块 (Storage)
- prepare_insert_data() - DataFrame 转元组列表
- upsert_to_mysql() - 分块批量 ON DUPLICATE KEY UPDATE
4. 主入口
- run_oil_pipeline_job(start_date, end_date) - 完整批处理流程
使用方式
# 命令行调用
python insertDailyOil.py --start 2024-01-01 --end 2024-12-31
python insertDailyOil.py -s 2024-06-01 -e 2024-06-30

# Python 调用
from insertDailyOil import run_oil_pipeline_job
result = run_oil_pipeline_job('2024-01-01', '2024-12-31')


"""

import pandas as pd
import numpy as np
import pymysql
from pymysql import MySQLError
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import time
import random

# yahoo-fin 用于获取金融数据
from yahoo_fin import stock_info as si

# 项目内部模块
import log4ak

# ============================================================
# 配置区域
# ============================================================

# 日志配置
log = log4ak.LogManager(log_level=log4ak.INFO)

# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'user': 'powerbi',
    'password': 'longyu',
    'database': 'akshare',
    'port': 3306,
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

# Yahoo-fin 数据源映射
TICKER_MAP = {
    'wti_close': 'CL=F',       # WTI 原油期货
    'brent_close': 'BZ=F',     # Brent 原油期货
    'usd_index': 'DX-Y.NYB',   # 美元指数
    'vix_index': '^VIX',       # VIX 恐慌指数
    'us_10y_yield': '^TNX',    # 美国10年期国债收益率
    'crack_spread': 'CRAK',    # 裂解价差 ETF
}

# IQR 异常值检测配置
IQR_WINDOW = 20       # 滚动窗口大小
IQR_MULTIPLIER = 1.5  # IQR 乘数

# RSI 计算周期
RSI_PERIOD = 14

# 移动平均周期
MA_PERIOD = 60

# 批量插入配置
BATCH_SIZE = 500


# ============================================================
# 1. 数据采集模块 (Fetcher)
# ============================================================

def fetch_single_ticker(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    从 yahoo-fin 获取单个 ticker 的历史数据
    
    Args:
        ticker: Yahoo Finance 股票/期货代码
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
    
    Returns:
        DataFrame 包含 trade_date 和 adjclose 列
    """
    try:
        log.info(f"正在获取 {ticker} 数据: {start_date} ~ {end_date}")
        
        # 调用 yahoo-fin API
        df = si.get_data(
            ticker,
            start_date=start_date,
            end_date=end_date,
            index_as_date=True
        )
        
        if df is None or df.empty:
            log.info(f"⚠️ {ticker} 返回空数据")
            return pd.DataFrame()
        
        # 提取复权收盘价
        df = df[['adjclose']].copy()
        df.index.name = 'trade_date'
        df.reset_index(inplace=True)
        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
        
        log.info(f"✅ {ticker} 获取成功，共 {len(df)} 条记录")
        
        # API 限流保护: 随机延迟 1-3 秒
        time.sleep(random.uniform(1, 3))
        
        return df
        
    except Exception as e:
        log.error(f"❌ 获取 {ticker} 失败: {e}")
        return pd.DataFrame()


def fetch_all_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取所有 ticker 数据并按 trade_date 外连接合并
    
    Args:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
    
    Returns:
        合并后的 DataFrame
    """
    log.info(f"===== 开始数据采集: {start_date} ~ {end_date} =====")
    
    merged_df = None
    
    for field_name, ticker in TICKER_MAP.items():
        df = fetch_single_ticker(ticker, start_date, end_date)
        
        if df.empty:
            continue
        
        # 重命名 adjclose 为目标字段名
        df.rename(columns={'adjclose': field_name}, inplace=True)
        
        if merged_df is None:
            merged_df = df
        else:
            # 外连接合并，保留所有交易日
            merged_df = pd.merge(
                merged_df, df,
                on='trade_date',
                how='outer'
            )
    
    if merged_df is None:
        log.error("❌ 所有数据源均获取失败")
        return pd.DataFrame()
    
    # 关键步骤: 按日期升序排列 (防止未来函数)
    merged_df['trade_date'] = pd.to_datetime(merged_df['trade_date'])
    merged_df.sort_values('trade_date', inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    
    log.info(f"✅ 数据采集完成，合并后共 {len(merged_df)} 条记录")
    
    return merged_df


# ============================================================
# 2. 数据清洗与特征转化模块 (Pipeline)
# ============================================================

def filter_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    步骤一: 基础过滤 - 剔除 <= 0 的物理错误值
    """
    log.info("执行基础过滤: 剔除 <= 0 的无效值")
    
    price_cols = ['wti_close', 'brent_close', 'usd_index', 'vix_index', 
                  'us_10y_yield', 'crack_spread']
    
    for col in price_cols:
        if col in df.columns:
            invalid_count = (df[col] <= 0).sum()
            if invalid_count > 0:
                log.info(f"  {col}: 发现 {invalid_count} 个无效值，置为 NaN")
                df.loc[df[col] <= 0, col] = np.nan
    
    # 特殊处理: us_10y_yield 可能返回百分比格式
    if 'us_10y_yield' in df.columns:
        # 若大于 10 则除以 10 (设计文档要求)
        df.loc[df['us_10y_yield'] > 10, 'us_10y_yield'] = \
            df.loc[df['us_10y_yield'] > 10, 'us_10y_yield'] / 10
    
    return df


def remove_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """
    步骤二: IQR 算法剔除闪崩异常值 (针对原油价格)
    """
    log.info("执行 IQR 异常值检测 (窗口=20, 乘数=1.5)")
    
    price_cols = ['wti_close', 'brent_close']
    
    for col in price_cols:
        if col not in df.columns:
            continue
        
        outlier_count = 0
        
        for i in range(len(df)):
            # 滑动窗口: 当前点之前的 IQR_WINDOW 个数据点
            start_idx = max(0, i - IQR_WINDOW + 1)
            window_data = df[col].iloc[start_idx:i+1].dropna()
            
            # 防错: 窗口内有效数据不足则跳过
            if len(window_data) < 5:
                continue
            
            q1 = window_data.quantile(0.25)
            q3 = window_data.quantile(0.75)
            iqr = q3 - q1
            
            # 防错: IQR 为 0 则跳过
            if iqr == 0:
                continue
            
            lower_bound = q1 - IQR_MULTIPLIER * iqr
            upper_bound = q3 + IQR_MULTIPLIER * iqr
            
            current_value = df[col].iloc[i]
            
            if pd.notna(current_value) and (current_value < lower_bound or current_value > upper_bound):
                df.loc[df.index[i], col] = np.nan
                outlier_count += 1
        
        if outlier_count > 0:
            log.info(f"  {col}: 剔除 {outlier_count} 个异常值")
    
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    步骤三: 缺失值填充 - 线性插值 + 前向填充
    """
    log.info("执行缺失值填充")
    
    # 线性插值: 价格和金融指标
    interpolate_cols = ['wti_close', 'brent_close', 'usd_index', 
                        'vix_index', 'us_10y_yield', 'crack_spread']
    
    for col in interpolate_cols:
        if col in df.columns:
            before_na = df[col].isna().sum()
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
            after_na = df[col].isna().sum()
            if before_na > after_na:
                log.info(f"  {col}: 线性插值填充 {before_na - after_na} 个缺失值")
    
    # 前向填充: 低频宏观数据 (如 gpr_index)
    ffill_cols = ['gpr_index']
    
    for col in ffill_cols:
        if col in df.columns:
            df[col] = df[col].ffill()
    
    return df


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    计算标准 RSI 指标
    """
    delta = series.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    步骤四: 衍生指标计算 (特征工程)
    """
    log.info("执行衍生指标计算")
    
    # 1. WTI 60日均线
    if 'wti_close' in df.columns:
        df['wti_60dma'] = df['wti_close'].rolling(
            window=MA_PERIOD, min_periods=1
        ).mean()
        log.info(f"  计算 wti_60dma (60日均线)")
    
    # 2. Brent-WTI 跨区价差
    if 'wti_close' in df.columns and 'brent_close' in df.columns:
        df['brent_wti_spread'] = df['brent_close'] - df['wti_close']
        log.info(f"  计算 brent_wti_spread (跨区价差)")
    
    # 3. WTI RSI 指标 (14日)
    if 'wti_close' in df.columns:
        df['wti_rsi'] = calculate_rsi(df['wti_close'], RSI_PERIOD)
        log.info(f"  计算 wti_rsi (14日RSI)")
    
    # 4. 期货曲线结构形态
    # 注: 由于未接入远月合约，使用 Brent-WTI 价差作为降级替代方案
    if 'brent_wti_spread' in df.columns:
        def determine_term_structure(spread):
            if pd.isna(spread):
                return None
            elif spread > 0.5:
                return 'Backwardation'
            elif spread < -0.5:
                return 'Contango'
            else:
                return 'Flat'
        
        df['term_structure'] = df['brent_wti_spread'].apply(determine_term_structure)
        log.info(f"  计算 term_structure (期限结构) - 注: 使用价差降级方案")
    
    # 5. GPR 指数 - 暂时填 NULL (akshare 可能无此接口)
    if 'gpr_index' not in df.columns:
        df['gpr_index'] = None
        log.info(f"  gpr_index: 暂无数据源，填 NULL")
    
    # 6. 估算风险溢价 - 暂时填 NULL
    df['risk_premium_est'] = None
    log.info(f"  risk_premium_est: 暂无计算逻辑，填 NULL")
    
    return df


def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    执行完整的数据清洗与特征转化流水线
    """
    log.info("===== 开始数据清洗流水线 =====")
    
    if df.empty:
        log.error("输入数据为空，跳过清洗")
        return df
    
    # 步骤一: 基础过滤
    df = filter_invalid_values(df)
    
    # 步骤二: IQR 异常值剔除
    df = remove_outliers_iqr(df)
    
    # 步骤三: 缺失值填充
    df = fill_missing_values(df)
    
    # 步骤四: 衍生指标计算
    df = calculate_derived_features(df)
    
    # 断言检查: wti_close 和 brent_close 不应有 NaN
    wti_na = df['wti_close'].isna().sum() if 'wti_close' in df.columns else 0
    brent_na = df['brent_close'].isna().sum() if 'brent_close' in df.columns else 0
    
    if wti_na > 0 or brent_na > 0:
        log.info(f"⚠️ 警告: 仍存在缺失值 (wti={wti_na}, brent={brent_na})，建议前推历史日期")
    
    log.info("✅ 数据清洗流水线完成")
    
    return df


# ============================================================
# 3. 持久化模块 (Storage)
# ============================================================

def prepare_insert_data(df: pd.DataFrame) -> List[Tuple]:
    """
    准备插入数据: DataFrame -> List of Tuples
    """
    # 数据库字段顺序
    columns = [
        'trade_date', 'wti_close', 'brent_close', 'wti_60dma', 'wti_rsi',
        'usd_index', 'us_10y_yield', 'vix_index',
        'brent_wti_spread', 'term_structure', 'crack_spread',
        'gpr_index', 'risk_premium_est'
    ]
    
    # 确保所有列存在
    for col in columns:
        if col not in df.columns:
            df[col] = None
    
    # 转换日期格式
    df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
    
    # 将 NaN 转换为 None (MySQL NULL)
    df = df.replace({np.nan: None})
    
    # 转换为元组列表
    data = [tuple(row) for row in df[columns].values]
    
    return data


def upsert_to_mysql(data: List[Tuple]) -> int:
    """
    Upsert 数据到 MySQL (ON DUPLICATE KEY UPDATE)
    
    Returns:
        成功插入/更新的记录数
    """
    if not data:
        log.info("无数据需要插入")
        return 0
    
    # Upsert SQL 语句
    insert_sql = """
    INSERT INTO `daily_oil_metrics` 
    (`trade_date`, `wti_close`, `brent_close`, `wti_60dma`, `wti_rsi`,
     `usd_index`, `us_10y_yield`, `vix_index`,
     `brent_wti_spread`, `term_structure`, `crack_spread`,
     `gpr_index`, `risk_premium_est`)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        `wti_close` = VALUES(`wti_close`),
        `brent_close` = VALUES(`brent_close`),
        `wti_60dma` = VALUES(`wti_60dma`),
        `wti_rsi` = VALUES(`wti_rsi`),
        `usd_index` = VALUES(`usd_index`),
        `us_10y_yield` = VALUES(`us_10y_yield`),
        `vix_index` = VALUES(`vix_index`),
        `brent_wti_spread` = VALUES(`brent_wti_spread`),
        `term_structure` = VALUES(`term_structure`),
        `crack_spread` = VALUES(`crack_spread`),
        `gpr_index` = VALUES(`gpr_index`),
        `risk_premium_est` = VALUES(`risk_premium_est`)
    """
    
    conn = None
    total_affected = 0
    
    try:
        conn = pymysql.connect(**DB_CONFIG)
        log.info(f"✅ 数据库连接成功 | MySQL版本: {conn.get_server_info()}")
        conn.autocommit(False)
        
        with conn.cursor() as cursor:
            # 分块提交
            for i in range(0, len(data), BATCH_SIZE):
                batch = data[i:i + BATCH_SIZE]
                cursor.executemany(insert_sql, batch)
                conn.commit()
                total_affected += cursor.rowcount
                log.info(f"  批次 {i//BATCH_SIZE + 1}: 写入 {len(batch)} 条")
        
        log.info(f"✅ 数据持久化完成，共影响 {total_affected} 条记录")
        return total_affected
        
    except MySQLError as e:
        log.error(f"❌ 数据库错误: {e}")
        if conn and conn.open:
            conn.rollback()
        raise
        
    finally:
        if conn and conn.open:
            conn.close()


# ============================================================
# 4. 主入口函数
# ============================================================

def run_oil_pipeline_job(start_date: str, end_date: str) -> dict:
    """
    批处理任务主入口
    
    Args:
        start_date: 起始日期 (YYYY-MM-DD 格式)
        end_date: 结束日期 (YYYY-MM-DD 格式)
    
    Returns:
        执行结果字典，包含 success, records_count, elapsed_time 等信息
    """
    result = {
        'success': False,
        'records_count': 0,
        'elapsed_time': 0,
        'message': ''
    }
    
    start_time = time.time()
    
    try:
        # 1. 输入校验
        log.info("=" * 60)
        log.info(f"原油日度数据批处理任务启动")
        log.info(f"日期范围: {start_date} ~ {end_date}")
        log.info("=" * 60)
        
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError as e:
            raise ValueError(f"日期格式错误，请使用 YYYY-MM-DD 格式: {e}")
        
        if start_dt > end_dt:
            raise ValueError(f"start_date ({start_date}) 不能大于 end_date ({end_date})")
        
        # 2. 数据采集
        raw_df = fetch_all_data(start_date, end_date)
        
        if raw_df.empty:
            result['message'] = '数据采集失败，无有效数据'
            log.error(result['message'])
            return result
        
        # 3. 数据清洗与特征转化
        clean_df = run_pipeline(raw_df)
        
        # 4. 数据持久化
        log.info("===== 开始数据持久化 =====")
        insert_data = prepare_insert_data(clean_df)
        records_count = upsert_to_mysql(insert_data)
        
        # 5. 完成
        elapsed_time = time.time() - start_time
        
        result['success'] = True
        result['records_count'] = records_count
        result['elapsed_time'] = round(elapsed_time, 2)
        result['message'] = f'成功处理 {len(clean_df)} 条记录，耗时 {result["elapsed_time"]} 秒'
        
        log.info("=" * 60)
        log.info(f"✅ 任务完成: {result['message']}")
        log.info("=" * 60)
        
        return result
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        result['elapsed_time'] = round(elapsed_time, 2)
        result['message'] = f'任务失败: {str(e)}'
        log.error(result['message'])
        raise


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='原油日度盘后数据抓取与清洗系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python insertDailyOil.py --start 2024-01-01 --end 2024-12-31
  python insertDailyOil.py -s 2024-06-01 -e 2024-06-30
        """
    )
    
    parser.add_argument(
        '-s', '--start',
        type=str,
        required=True,
        help='起始日期 (YYYY-MM-DD 格式)'
    )
    
    parser.add_argument(
        '-e', '--end',
        type=str,
        required=True,
        help='结束日期 (YYYY-MM-DD 格式)'
    )
    
    args = parser.parse_args()
    
    try:
        result = run_oil_pipeline_job(args.start, args.end)
        if result['success']:
            print(f"\n✅ {result['message']}")
        else:
            print(f"\n❌ {result['message']}")
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        exit(1)
