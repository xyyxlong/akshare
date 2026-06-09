"""
石油周度基本面数据抓取与清洗系统
功能: 从 EIA/CFTC/Baker Hughes 抓取周度供需数据，经过清洗与特征计算后，Upsert 到 MySQL

数据流程:
    1. 数据采集 (Ingestion) - 调用 EIA API v2, CFTC 持仓, Baker Hughes 钻井数
    2. 数据清洗 (ETL) - 物理值过滤, 52周IQR异常检测, 线性插值, Prophet预测填充
    3. 特征工程 (Feature) - 计算 cftc_net_long_chg (周环比变化)
    4. 数据持久化 (Storage) - 分块 Upsert 到 MySQL

作者: OpenCode
日期: 2026-06-09


代码结构概览
1. 数据采集模块 (Data Ingestion)
函数	数据源	说明
fetch_eia_series()	EIA API v2	获取单个 EIA 序列
fetch_all_eia_data()	EIA	合并 6 个 EIA 序列
fetch_cftc_data()	CFTC	持仓数据 (需配置)
fetch_baker_hughes_data()	Baker Hughes	钻井数 (需配置)
fetch_inventory_forecast()	财经日历	库存预测值 (需配置)
2. ETL 清洗模块
步骤	函数	说明
物理值过滤	filter_invalid_physical_values()	剔除 <=0 的绝对量 (库存变化允许负数)
IQR 异常检测	remove_outliers_iqr()	52周窗口, 乘数=2.0
线性插值	fill_missing_linear()	EIA 高频数据
Prophet 填充	fill_missing_prophet()	CFTC 低频数据 (连续缺失>=2周)
3. 特征工程模块
- calculate_features() - 计算 cftc_net_long_chg = diff(1)
4. 持久化模块
- upsert_to_mysql() - ON DUPLICATE KEY UPDATE 幂等写入
5. 主入口
- process_weekly_metrics(start_date, end_date) - 自动前推 52 周
设计文档要求的关键实现
要求	实现位置
历史前推 52 周	process_weekly_metrics() 第 2 步
时序正序排列	fetch_all_data() 末尾 sort_values()
物理值过滤 (<=0)	filter_invalid_physical_values()
IQR 窗口=52, 乘数=2.0	remove_outliers_iqr()
Prophet 预测填充	fill_missing_prophet()
cftc_net_long_chg = diff(1)	calculate_features()
裁剪到用户日期范围	process_weekly_metrics() 第 6 步
ON DUPLICATE KEY UPDATE	upsert_to_mysql() SQL 模板
数据溯源日志	IQR 和 Prophet 填充时记录详细日志
API 限流 1-2.5秒	fetch_eia_series() 末尾 time.sleep()
使用方式
# 命令行调用
python insertWeeklyOil.py --start 2024-01-01 --end 2024-12-31
python insertWeeklyOil.py -s 2024-06-01 -e 2024-06-30 --api-key YOUR_EIA_KEY

# Python 调用
from insertWeeklyOil import process_weekly_metrics
result = process_weekly_metrics('2024-01-01', '2024-12-31')
配置说明
需要配置以下数据源:
1. EIA API Key: 在 EIA_API_KEY 变量中设置，从 https://www.eia.gov/opendata/ 注册获取
2. CFTC 数据: 实现 fetch_cftc_data() 函数，可爬取 CFTC 官网 CSV
3. Baker Hughes: 实现 fetch_baker_hughes_data() 函数，可爬取官网 Excel
4. 库存预测值: 实现 fetch_inventory_forecast() 函数，可对接金十数据等财经日历 API
"""

import pandas as pd
import numpy as np
import pymysql
from pymysql import MySQLError
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict
import time
import random
import requests
import warnings

# Prophet 用于低频数据的时序预测填充
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("Prophet 未安装，CFTC 数据将使用前向填充替代")

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

# EIA API 配置
# 注意: 需要在 EIA 官网注册获取 API Key: https://www.eia.gov/opendata/
EIA_API_KEY = "YOUR_EIA_API_KEY"  # 请替换为实际的 API Key
EIA_BASE_URL = "https://api.eia.gov/v2"

# EIA 数据序列映射
EIA_SERIES_MAP = {
    'eia_crude_inventory_chg': {
        'endpoint': '/petroleum/stoc/wstk/data/',
        'series': 'WCESTUS1',
        'description': '商业原油库存变化'
    },
    'eia_gasoline_chg': {
        'endpoint': '/petroleum/stoc/wstk/data/',
        'series': 'WGTSTUS1',
        'description': '汽油库存变化'
    },
    'eia_distillates_chg': {
        'endpoint': '/petroleum/stoc/wstk/data/',
        'series': 'WDISTUS1',
        'description': '馏分油库存变化'
    },
    'eia_cushing_inventory': {
        'endpoint': '/petroleum/stoc/wstk/data/',
        'series': 'WCSSTUS1',
        'description': '库欣地区原油库存'
    },
    'us_crude_production': {
        'endpoint': '/petroleum/sum/sndw/data/',
        'series': 'WCRFPUS2',
        'description': '美国原油产量'
    },
    'refinery_utilization': {
        'endpoint': '/petroleum/pnp/wiup/data/',
        'series': 'WPULEUS3',
        'description': '炼厂开工率'
    },
}

# IQR 异常值检测配置 (周度数据使用52周窗口)
IQR_WINDOW = 52       # 52周 ≈ 1年
IQR_MULTIPLIER = 2.0  # 设计文档要求使用 2.0

# 历史前推周数 (用于滑动窗口和Prophet预测)
LOOKBACK_WEEKS = 52

# 批量插入配置
BATCH_SIZE = 100


# ============================================================
# 1. 数据采集模块 (Data Ingestion)
# ============================================================

def fetch_eia_series(series_id: str, endpoint: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    从 EIA API v2 获取单个数据序列
    
    Args:
        series_id: EIA 序列 ID (如 WCESTUS1)
        endpoint: API 端点路径
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
    
    Returns:
        DataFrame 包含 report_date 和 value 列
    """
    try:
        url = f"{EIA_BASE_URL}{endpoint}"
        params = {
            'api_key': EIA_API_KEY,
            'frequency': 'weekly',
            'data[0]': 'value',
            'facets[series][]': series_id,
            'start': start_date,
            'end': end_date,
            'sort[0][column]': 'period',
            'sort[0][direction]': 'asc'
        }
        
        log.info(f"正在获取 EIA 序列 {series_id}: {start_date} ~ {end_date}")
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if 'response' not in data or 'data' not in data['response']:
            log.info(f"⚠️ EIA {series_id} 返回空数据")
            return pd.DataFrame()
        
        records = data['response']['data']
        
        if not records:
            log.info(f"⚠️ EIA {series_id} 无记录")
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        df = df[['period', 'value']].copy()
        df.columns = ['report_date', 'value']
        df['report_date'] = pd.to_datetime(df['report_date']).dt.strftime('%Y-%m-%d')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        log.info(f"✅ EIA {series_id} 获取成功，共 {len(df)} 条记录")
        
        # API 限流保护
        time.sleep(random.uniform(1.0, 2.5))
        
        return df
        
    except requests.exceptions.RequestException as e:
        log.error(f"❌ EIA API 请求失败 ({series_id}): {e}")
        return pd.DataFrame()
    except Exception as e:
        log.error(f"❌ 获取 EIA {series_id} 失败: {e}")
        return pd.DataFrame()


def fetch_all_eia_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取所有 EIA 数据序列并按 report_date 外连接合并
    """
    log.info(f"===== 开始 EIA 数据采集: {start_date} ~ {end_date} =====")
    
    merged_df = None
    
    for field_name, config in EIA_SERIES_MAP.items():
        df = fetch_eia_series(
            config['series'],
            config['endpoint'],
            start_date,
            end_date
        )
        
        if df.empty:
            continue
        
        # 重命名 value 为目标字段名
        df.rename(columns={'value': field_name}, inplace=True)
        
        if merged_df is None:
            merged_df = df
        else:
            # 外连接合并
            merged_df = pd.merge(
                merged_df, df,
                on='report_date',
                how='outer'
            )
    
    if merged_df is not None:
        log.info(f"✅ EIA 数据采集完成，合并后共 {len(merged_df)} 条记录")
    else:
        log.info("⚠️ EIA 数据采集返回空")
        merged_df = pd.DataFrame()
    
    return merged_df


def fetch_cftc_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取 CFTC 持仓数据
    
    注意: CFTC 数据需要从官网下载 CSV 或使用第三方数据源
    此处提供模拟实现，实际生产中需要替换为真实数据源
    """
    log.info(f"正在获取 CFTC 持仓数据: {start_date} ~ {end_date}")
    
    # TODO: 实现真实的 CFTC 数据获取逻辑
    # 可选方案:
    # 1. 爬取 CFTC 官网 CSV: https://www.cftc.gov/MarketReports/CommitmentsofTraders/
    # 2. 使用 quandl/nasdaq 等第三方 API
    # 3. 从本地 CSV 文件读取
    
    # 返回空 DataFrame，由调用方处理
    log.info("⚠️ CFTC 数据源未配置，返回空数据 (请配置真实数据源)")
    return pd.DataFrame(columns=['report_date', 'cftc_net_long'])


def fetch_baker_hughes_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取 Baker Hughes 活跃钻井数数据
    
    注意: Baker Hughes 数据需要从官网下载 Excel
    此处提供模拟实现，实际生产中需要替换为真实数据源
    """
    log.info(f"正在获取 Baker Hughes 钻井数据: {start_date} ~ {end_date}")
    
    # TODO: 实现真实的 Baker Hughes 数据获取逻辑
    # 数据源: https://rigcount.bakerhughes.com/
    
    log.info("⚠️ Baker Hughes 数据源未配置，返回空数据 (请配置真实数据源)")
    return pd.DataFrame(columns=['report_date', 'baker_hughes_rig_count'])


def fetch_inventory_forecast(start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取 EIA 原油库存预测值 (市场预期)
    
    注意: 预测值需要从财经日历 API 获取 (如金十数据、Investing.com)
    """
    log.info(f"正在获取 EIA 库存预测值: {start_date} ~ {end_date}")
    
    # TODO: 实现真实的预测值数据获取
    # 可选数据源:
    # 1. Investing.com Economic Calendar API
    # 2. 金十数据 API
    
    log.info("⚠️ EIA 库存预测值数据源未配置，返回空数据")
    return pd.DataFrame(columns=['report_date', 'eia_crude_inventory_forecast'])


def calculate_ref_week_end(report_date: pd.Series) -> pd.Series:
    """
    计算数据所属周的结束日期 (周五)
    
    EIA 数据通常在周三发布，数据截至上周五
    """
    def get_previous_friday(dt):
        if pd.isna(dt):
            return None
        dt = pd.to_datetime(dt)
        # 找到上一个周五
        days_since_friday = (dt.weekday() - 4) % 7
        if days_since_friday == 0 and dt.weekday() != 4:
            days_since_friday = 7
        return (dt - timedelta(days=days_since_friday)).strftime('%Y-%m-%d')
    
    return report_date.apply(get_previous_friday)


def fetch_all_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取所有数据源并按 report_date 外连接合并
    """
    log.info("=" * 60)
    log.info(f"开始全量数据采集: {start_date} ~ {end_date}")
    log.info("=" * 60)
    
    # 1. EIA 数据
    eia_df = fetch_all_eia_data(start_date, end_date)
    
    # 2. CFTC 持仓数据
    cftc_df = fetch_cftc_data(start_date, end_date)
    
    # 3. Baker Hughes 钻井数
    bh_df = fetch_baker_hughes_data(start_date, end_date)
    
    # 4. EIA 库存预测值
    forecast_df = fetch_inventory_forecast(start_date, end_date)
    
    # 合并所有数据源
    merged_df = eia_df
    
    for df in [cftc_df, bh_df, forecast_df]:
        if not df.empty and 'report_date' in df.columns:
            if merged_df.empty:
                merged_df = df
            else:
                merged_df = pd.merge(
                    merged_df, df,
                    on='report_date',
                    how='outer'
                )
    
    if merged_df.empty:
        log.error("❌ 所有数据源均获取失败")
        return pd.DataFrame()
    
    # 按日期升序排列 (关键步骤: 防止未来函数)
    merged_df['report_date'] = pd.to_datetime(merged_df['report_date'])
    merged_df.sort_values('report_date', inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    
    # 计算 ref_week_end (数据所属周的周五)
    merged_df['ref_week_end'] = calculate_ref_week_end(merged_df['report_date'])
    
    log.info(f"✅ 全量数据采集完成，合并后共 {len(merged_df)} 条记录")
    
    return merged_df


# ============================================================
# 2. 数据清洗与预处理模块 (ETL & Cleaning)
# ============================================================

def filter_invalid_physical_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    步骤一: 绝对物理值过滤 - 剔除 <= 0 的无效值
    
    注意: 库存变化量 (_chg) 字段允许为负数，不执行此过滤
    """
    log.info("执行物理值过滤: 剔除 <= 0 的无效值")
    
    # 只过滤代表绝对物理量的字段
    physical_cols = [
        'eia_cushing_inventory',      # 库欣库存绝对量
        'us_crude_production',        # 产量
        'refinery_utilization',       # 开工率
        'baker_hughes_rig_count'      # 钻井数
    ]
    
    for col in physical_cols:
        if col in df.columns:
            invalid_count = (df[col] <= 0).sum()
            if invalid_count > 0:
                log.info(f"  {col}: 发现 {invalid_count} 个无效值 (<=0)，置为 NaN")
                df.loc[df[col] <= 0, col] = np.nan
    
    return df


def remove_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """
    步骤二: 滑动 IQR 算法过滤基本面噪点
    
    针对 eia_crude_inventory_chg 使用 52周窗口, 乘数=2.0
    """
    log.info(f"执行 IQR 异常值检测 (窗口={IQR_WINDOW}周, 乘数={IQR_MULTIPLIER})")
    
    # 仅对库存变化字段执行 IQR 过滤
    target_col = 'eia_crude_inventory_chg'
    
    if target_col not in df.columns:
        return df
    
    outlier_count = 0
    outlier_dates = []
    
    for i in range(len(df)):
        # 滑动窗口: 当前点之前的 IQR_WINDOW 个数据点
        start_idx = max(0, i - IQR_WINDOW + 1)
        window_data = df[target_col].iloc[start_idx:i+1].dropna()
        
        # 防错: 窗口内有效数据不足则跳过
        if len(window_data) < 10:
            continue
        
        q1 = window_data.quantile(0.25)
        q3 = window_data.quantile(0.75)
        iqr = q3 - q1
        
        # 防错: IQR 为 0 则跳过
        if iqr == 0:
            continue
        
        lower_bound = q1 - IQR_MULTIPLIER * iqr
        upper_bound = q3 + IQR_MULTIPLIER * iqr
        
        current_value = df[target_col].iloc[i]
        
        if pd.notna(current_value) and (current_value < lower_bound or current_value > upper_bound):
            df.loc[df.index[i], target_col] = np.nan
            outlier_count += 1
            outlier_dates.append(df['report_date'].iloc[i])
            # 数据溯源日志
            log.info(f"  IQR 异常: {df['report_date'].iloc[i]} 的 {target_col} = {current_value:.2f} 超出范围 [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    if outlier_count > 0:
        log.info(f"  {target_col}: 共剔除 {outlier_count} 个异常值")
    
    return df


def fill_missing_linear(df: pd.DataFrame) -> pd.DataFrame:
    """
    步骤三-A: 高频基本面数据线性插值
    """
    log.info("执行线性插值填充 (EIA 高频数据)")
    
    interpolate_cols = [
        'eia_crude_inventory_chg',
        'eia_gasoline_chg',
        'eia_distillates_chg',
        'eia_cushing_inventory',
        'us_crude_production',
        'refinery_utilization'
    ]
    
    for col in interpolate_cols:
        if col in df.columns:
            before_na = df[col].isna().sum()
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
            after_na = df[col].isna().sum()
            filled = before_na - after_na
            if filled > 0:
                log.info(f"  {col}: 线性插值填充 {filled} 个缺失值")
    
    return df


def fill_missing_prophet(df: pd.DataFrame) -> pd.DataFrame:
    """
    步骤三-B: 低频宏观数据 Prophet 预测填充 (CFTC 持仓)
    
    若 Prophet 不可用，退化为前向填充
    """
    target_col = 'cftc_net_long'
    
    if target_col not in df.columns:
        return df
    
    missing_count = df[target_col].isna().sum()
    
    if missing_count == 0:
        return df
    
    # 检查是否存在连续 2 周以上的缺失
    consecutive_na = 0
    max_consecutive = 0
    for val in df[target_col]:
        if pd.isna(val):
            consecutive_na += 1
            max_consecutive = max(max_consecutive, consecutive_na)
        else:
            consecutive_na = 0
    
    log.info(f"CFTC 持仓数据: {missing_count} 个缺失值, 最大连续缺失 {max_consecutive} 周")
    
    if max_consecutive >= 2 and PROPHET_AVAILABLE:
        log.info("启动 Prophet 时序预测填充")
        
        try:
            # 准备 Prophet 训练数据
            train_df = df[['report_date', target_col]].dropna().copy()
            train_df.columns = ['ds', 'y']
            train_df['ds'] = pd.to_datetime(train_df['ds'])
            
            if len(train_df) < 10:
                log.info("⚠️ Prophet 训练数据不足 (<10), 退化为前向填充")
                df[target_col] = df[target_col].ffill()
                return df
            
            # 训练 Prophet 模型
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            model.fit(train_df)
            
            # 预测缺失日期
            missing_mask = df[target_col].isna()
            future_df = pd.DataFrame({
                'ds': pd.to_datetime(df.loc[missing_mask, 'report_date'])
            })
            
            if not future_df.empty:
                forecast = model.predict(future_df)
                
                # 填充预测值
                for idx, pred_val in zip(df.index[missing_mask], forecast['yhat']):
                    df.loc[idx, target_col] = pred_val
                    # 数据溯源日志
                    log.info(f"  Prophet 填充: {df.loc[idx, 'report_date']} 的 {target_col} = {pred_val:.2f}")
            
            log.info(f"✅ Prophet 预测填充完成，共填充 {missing_count} 个值")
            
        except Exception as e:
            log.error(f"❌ Prophet 预测失败: {e}，退化为前向填充")
            df[target_col] = df[target_col].ffill()
    else:
        log.info("使用前向填充 (ffill) 处理 CFTC 缺失值")
        df[target_col] = df[target_col].ffill()
    
    return df


def run_etl_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    执行完整的 ETL 清洗流水线
    """
    log.info("===== 开始 ETL 清洗流水线 =====")
    
    if df.empty:
        log.error("输入数据为空，跳过清洗")
        return df
    
    # 步骤一: 物理值过滤
    df = filter_invalid_physical_values(df)
    
    # 步骤二: IQR 异常值剔除
    df = remove_outliers_iqr(df)
    
    # 步骤三-A: 线性插值
    df = fill_missing_linear(df)
    
    # 步骤三-B: Prophet/前向填充
    df = fill_missing_prophet(df)
    
    # Baker Hughes 钻井数使用前向填充
    if 'baker_hughes_rig_count' in df.columns:
        df['baker_hughes_rig_count'] = df['baker_hughes_rig_count'].ffill()
    
    # EIA 预测值使用前向填充
    if 'eia_crude_inventory_forecast' in df.columns:
        df['eia_crude_inventory_forecast'] = df['eia_crude_inventory_forecast'].ffill()
    
    log.info("✅ ETL 清洗流水线完成")
    
    return df


# ============================================================
# 3. 特征工程模块 (Feature Engineering)
# ============================================================

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算衍生特征
    
    主要计算 cftc_net_long_chg (CFTC 净多头持仓周环比变化)
    """
    log.info("===== 开始特征工程 =====")
    
    # CFTC 持仓周环比变化
    if 'cftc_net_long' in df.columns:
        df['cftc_net_long_chg'] = df['cftc_net_long'].diff(1)
        log.info("  计算 cftc_net_long_chg (周环比变化)")
    else:
        df['cftc_net_long_chg'] = None
    
    log.info("✅ 特征工程完成")
    
    return df


# ============================================================
# 4. 持久化模块 (Storage)
# ============================================================

def prepare_insert_data(df: pd.DataFrame) -> List[Tuple]:
    """
    准备插入数据: DataFrame -> List of Tuples
    """
    # 数据库字段顺序
    columns = [
        'report_date', 'ref_week_end',
        'eia_crude_inventory_chg', 'eia_crude_inventory_forecast',
        'eia_gasoline_chg', 'eia_distillates_chg', 'eia_cushing_inventory',
        'us_crude_production', 'refinery_utilization',
        'cftc_net_long', 'cftc_net_long_chg',
        'baker_hughes_rig_count'
    ]
    
    # 确保所有列存在
    for col in columns:
        if col not in df.columns:
            df[col] = None
    
    # 转换日期格式
    df['report_date'] = pd.to_datetime(df['report_date']).dt.strftime('%Y-%m-%d')
    if 'ref_week_end' in df.columns:
        df['ref_week_end'] = pd.to_datetime(df['ref_week_end']).dt.strftime('%Y-%m-%d')
    
    # 将 NaN 转换为 None (MySQL NULL)
    df = df.replace({np.nan: None})
    
    # 转换为元组列表
    data = [tuple(row) for row in df[columns].values]
    
    return data


def upsert_to_mysql(data: List[Tuple]) -> int:
    """
    Upsert 数据到 MySQL (ON DUPLICATE KEY UPDATE)
    """
    if not data:
        log.info("无数据需要插入")
        return 0
    
    # Upsert SQL 语句 (设计文档 5.2 模板)
    insert_sql = """
    INSERT INTO `weekly_oil_metrics` (
        `report_date`, `ref_week_end`,
        `eia_crude_inventory_chg`, `eia_crude_inventory_forecast`,
        `eia_gasoline_chg`, `eia_distillates_chg`, `eia_cushing_inventory`,
        `us_crude_production`, `refinery_utilization`,
        `cftc_net_long`, `cftc_net_long_chg`,
        `baker_hughes_rig_count`
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        `ref_week_end` = VALUES(`ref_week_end`),
        `eia_crude_inventory_chg` = VALUES(`eia_crude_inventory_chg`),
        `eia_crude_inventory_forecast` = VALUES(`eia_crude_inventory_forecast`),
        `eia_gasoline_chg` = VALUES(`eia_gasoline_chg`),
        `eia_distillates_chg` = VALUES(`eia_distillates_chg`),
        `eia_cushing_inventory` = VALUES(`eia_cushing_inventory`),
        `us_crude_production` = VALUES(`us_crude_production`),
        `refinery_utilization` = VALUES(`refinery_utilization`),
        `cftc_net_long` = VALUES(`cftc_net_long`),
        `cftc_net_long_chg` = VALUES(`cftc_net_long_chg`),
        `baker_hughes_rig_count` = VALUES(`baker_hughes_rig_count`)
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
# 5. 主入口函数
# ============================================================

def process_weekly_metrics(start_date: str, end_date: str) -> dict:
    """
    周度盘后基本面数据批处理主入口
    
    Args:
        start_date: 用户要求的入库开始日期 (YYYY-MM-DD 格式)
        end_date: 用户要求的入库结束日期 (YYYY-MM-DD 格式)
    
    Returns:
        执行结果字典
    
    【核心执行流程】:
    1. 内部自动将 start_date 向历史方向前推 52 周，得到 fetch_start_date
    2. 调用 Ingestion 模块抓取 [fetch_start_date, end_date] 的完整数据
    3. 将整个长序列送入 ETL 模块进行清洗、插值与时序填充
    4. 将清洗后的长序列送入 Feature 模块计算周环比变化
    5. 裁剪数据，仅保留用户原始要求的 [start_date, end_date] 范围
    6. 调用 Storage 模块批量 Upsert 入库
    """
    result = {
        'success': False,
        'records_count': 0,
        'elapsed_time': 0,
        'message': ''
    }
    
    start_time = time.time()
    
    try:
        log.info("=" * 60)
        log.info("原油周度基本面数据批处理任务启动")
        log.info(f"用户请求日期范围: {start_date} ~ {end_date}")
        log.info("=" * 60)
        
        # 1. 输入校验
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError as e:
            raise ValueError(f"日期格式错误，请使用 YYYY-MM-DD 格式: {e}")
        
        if start_dt > end_dt:
            raise ValueError(f"start_date ({start_date}) 不能大于 end_date ({end_date})")
        
        # 2. 历史前推 52 周 (用于滑动窗口和 Prophet)
        fetch_start_dt = start_dt - timedelta(weeks=LOOKBACK_WEEKS)
        fetch_start_date = fetch_start_dt.strftime('%Y-%m-%d')
        log.info(f"历史前推 {LOOKBACK_WEEKS} 周，实际抓取范围: {fetch_start_date} ~ {end_date}")
        
        # 3. 数据采集
        raw_df = fetch_all_data(fetch_start_date, end_date)
        
        if raw_df.empty:
            result['message'] = '数据采集失败，无有效数据'
            log.error(result['message'])
            return result
        
        # 4. ETL 清洗
        clean_df = run_etl_pipeline(raw_df)
        
        # 5. 特征工程
        feature_df = calculate_features(clean_df)
        
        # 6. 裁剪数据: 仅保留用户原始要求的 [start_date, end_date]
        feature_df['report_date'] = pd.to_datetime(feature_df['report_date'])
        final_df = feature_df[
            (feature_df['report_date'] >= start_dt) &
            (feature_df['report_date'] <= end_dt)
        ].copy()
        
        log.info(f"裁剪后数据: {len(final_df)} 条记录 (原始 {len(feature_df)} 条)")
        
        if final_df.empty:
            result['message'] = '裁剪后无数据'
            log.info(result['message'])
            return result
        
        # 7. Schema 校验
        log.info("===== 开始数据持久化 =====")
        insert_data = prepare_insert_data(final_df)
        
        # 8. 批量 Upsert
        records_count = upsert_to_mysql(insert_data)
        
        # 9. 完成
        elapsed_time = time.time() - start_time
        
        result['success'] = True
        result['records_count'] = records_count
        result['elapsed_time'] = round(elapsed_time, 2)
        result['message'] = f'成功处理 {len(final_df)} 条记录，耗时 {result["elapsed_time"]} 秒'
        
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
        description='原油周度基本面数据抓取与清洗系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python insertWeeklyOil.py --start 2024-01-01 --end 2024-12-31
  python insertWeeklyOil.py -s 2024-06-01 -e 2024-06-30

注意:
  - 需要配置 EIA_API_KEY (从 https://www.eia.gov/opendata/ 获取)
  - CFTC 和 Baker Hughes 数据源需要单独配置
  - 程序会自动前推 52 周获取历史数据用于滑动窗口计算
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
    
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='EIA API Key (可选，也可在代码中配置)'
    )
    
    args = parser.parse_args()
    
    # 如果命令行提供了 API Key，覆盖默认值
    if args.api_key:
        EIA_API_KEY = args.api_key
    
    try:
        result = process_weekly_metrics(args.start, args.end)
        if result['success']:
            print(f"\n✅ {result['message']}")
        else:
            print(f"\n❌ {result['message']}")
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        exit(1)
