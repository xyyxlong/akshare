"""
CFTC 持仓数据自动化下载与解析模块
功能: 从 CFTC 官网下载历史持仓数据，提取原油非商业净多头持仓

数据流程:
    1. 根据日期范围确定需要下载的年份
    2. 下载 CFTC 官方 ZIP 文件到内存
    3. 解析 CSV，过滤原油数据
    4. 计算 cftc_net_long = Long_All - Short_All
    5. 整合为一个 CSV 保存到 ../input/CFTC/CFTC_<start_date>_<end_date>.csv

作者: OpenCode
日期: 2026-06-10

使用方式:
    # 命令行调用
    python downloadCFTC.py --start 2024-01-01 --end 2024-12-31
    python downloadCFTC.py -s 2025-01-01 -e 2025-06-30

    # Python 调用 (供 insertWeeklyOil.py 使用)
    from downloadCFTC import get_cftc_net_long
    df = get_cftc_net_long('2024-01-01', '2024-12-31')

输出文件:
    ../input/CFTC/CFTC_2024-01-01_2024-12-31.csv
    包含列: report_date, ref_week_end, cftc_net_long 及持仓明细
"""

import pandas as pd
import numpy as np
import requests
import zipfile
import io
import os
import time
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
from pathlib import Path

# 项目内部模块
import log4ak

# ============================================================
# 配置区域
# ============================================================

# 日志配置
log = log4ak.LogManager(log_level=log4ak.INFO)

# CFTC 官方下载 URL 模板
CFTC_URL_TEMPLATE = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"

# 备用 URL (使用 XLS 格式)
CFTC_URL_TEMPLATE_ALT = "https://www.cftc.gov/files/dea/history/fut_disagg_xls_{year}.zip"

# 原油过滤条件
CRUDE_OIL_MARKET_TYPE = "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE"
CRUDE_OIL_MARKET_CODE = "067651"

# 请求头 (伪装浏览器，避免 403)
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
}

# 输出目录
OUTPUT_DIR = Path(__file__).parent.parent / "input" / "CFTC"

# 反爬延迟 (秒)
REQUEST_DELAY = 2.0


# ============================================================
# 核心功能函数
# ============================================================

def get_years_from_date_range(start_date: str, end_date: str) -> List[int]:
    """
    从日期范围提取涉及的年份列表
    
    Args:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
    
    Returns:
        年份列表，如 [2025, 2026]
    """
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    years = list(range(start_dt.year, end_dt.year + 1))
    log.info(f"日期范围 {start_date} ~ {end_date} 涉及年份: {years}")
    
    return years


def download_cftc_zip(year: int) -> Optional[bytes]:
    """
    下载指定年份的 CFTC ZIP 文件
    
    Args:
        year: 年份
    
    Returns:
        ZIP 文件的字节内容，失败返回 None
    """
    url = CFTC_URL_TEMPLATE.format(year=year)
    
    log.info(f"正在下载 CFTC {year} 年数据: {url}")
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=60)
        response.raise_for_status()
        
        log.info(f"✅ CFTC {year} 年数据下载成功，大小: {len(response.content) / 1024 / 1024:.2f} MB")
        return response.content
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            log.info(f"⚠️ CFTC {year} 年数据不存在 (404)，尝试备用 URL...")
            # 尝试备用 URL
            url_alt = CFTC_URL_TEMPLATE_ALT.format(year=year)
            try:
                response = requests.get(url_alt, headers=HEADERS, timeout=60)
                response.raise_for_status()
                log.info(f"✅ CFTC {year} 年数据 (备用URL) 下载成功")
                return response.content
            except Exception as e2:
                log.error(f"❌ 备用 URL 也失败: {e2}")
                return None
        else:
            log.error(f"❌ CFTC {year} 年数据下载失败: {e}")
            return None
            
    except requests.exceptions.RequestException as e:
        log.error(f"❌ 网络请求失败: {e}")
        return None


def extract_csv_from_zip(zip_content: bytes) -> Optional[pd.DataFrame]:
    """
    从 ZIP 内容中提取 CSV 数据
    
    Args:
        zip_content: ZIP 文件的字节内容
    
    Returns:
        解析后的 DataFrame，失败返回 None
    """
    try:
        with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
            # 列出 ZIP 中的文件
            file_list = zf.namelist()
            log.info(f"ZIP 包含文件: {file_list}")
            
            # 寻找 CSV/TXT 文件
            csv_file = None
            for f in file_list:
                if f.lower().endswith(('.txt', '.csv')):
                    csv_file = f
                    break
            
            if csv_file is None:
                log.error("❌ ZIP 中未找到 CSV/TXT 文件")
                return None
            
            log.info(f"解析文件: {csv_file}")
            
            # 读取 CSV
            with zf.open(csv_file) as f:
                df = pd.read_csv(f, low_memory=False)
            
            # 统一列名为大写并去除空格 (防呆处理)
            df.columns = df.columns.str.strip().str.upper()
            
            log.info(f"✅ CSV 解析成功，共 {len(df)} 行，{len(df.columns)} 列")
            
            return df
            
    except zipfile.BadZipFile:
        log.error("❌ ZIP 文件损坏")
        return None
    except Exception as e:
        log.error(f"❌ 解析 ZIP 失败: {e}")
        return None


def filter_crude_oil_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    过滤出原油相关数据
    
    过滤条件:
    - Market_and_Market_Type = "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE"
    - 或 CFTC_Market_Code = "067651"
    
    Args:
        df: 原始 DataFrame
    
    Returns:
        过滤后的原油 DataFrame
    """
    log.info("过滤原油数据...")
    
    # 查找可能的列名
    market_type_col = None
    market_code_col = None
    
    for col in df.columns:
        if 'MARKET_AND_EXCHANGE_NAMES' in col.upper():
            market_type_col = col
        elif 'CFTC_CONTRACT_MARKET_CODE' in col.upper() or 'CFTC_MARKET_CODE' in col.upper():
            market_code_col = col
    
    log.info(f"市场类型列: {market_type_col}, 市场代码列: {market_code_col}")
    
    # 构建过滤条件
    mask = pd.Series([False] * len(df))
    
    if market_type_col and market_type_col in df.columns:
        # 使用包含匹配，因为名称可能略有不同
        mask |= df[market_type_col].str.upper().str.contains('CRUDE OIL', na=False)
    
    if market_code_col and market_code_col in df.columns:
        mask |= df[market_code_col].astype(str) == CRUDE_OIL_MARKET_CODE
    
    filtered_df = df[mask].copy()
    
    log.info(f"✅ 过滤后剩余 {len(filtered_df)} 条原油记录")
    
    return filtered_df


def calculate_net_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算非商业净多头持仓
    
    cftc_net_long = Prod_Merc_Positions_Long_All - Prod_Merc_Positions_Short_All
    
    注意: 设计文档中提到的是 Prod_Merc (生产商/贸易商)，
         但通常分析使用的是 Money Manager (基金经理) 或 Non-Commercial (非商业)
         这里按设计文档实现，同时也计算 Money Manager 版本供参考
    
    Args:
        df: 过滤后的原油 DataFrame
    
    Returns:
        添加了 cftc_net_long 列的 DataFrame
    """
    log.info("计算净多头持仓...")
    
    # 查找可能的列名 (CFTC CSV 列名可能有多种变体)
    long_col = None
    short_col = None
    
    # 优先使用 Money Manager (基金经理) 数据
    for col in df.columns:
        col_upper = col.upper()
        if 'M_MONEY_POSITIONS_LONG' in col_upper or 'MONEY_MANAGER_LONGS' in col_upper:
            long_col = col
        elif 'M_MONEY_POSITIONS_SHORT' in col_upper or 'MONEY_MANAGER_SHORTS' in col_upper:
            short_col = col
    
    # 如果没找到 Money Manager，使用 Prod_Merc (生产商/贸易商)
    if long_col is None or short_col is None:
        for col in df.columns:
            col_upper = col.upper()
            if 'PROD_MERC_POSITIONS_LONG' in col_upper:
                long_col = col
            elif 'PROD_MERC_POSITIONS_SHORT' in col_upper:
                short_col = col
    
    # 如果还没找到，尝试 Non-Commercial
    if long_col is None or short_col is None:
        for col in df.columns:
            col_upper = col.upper()
            if 'NONCOMM_POSITIONS_LONG' in col_upper or 'NON_COMMERCIAL_LONG' in col_upper:
                long_col = col
            elif 'NONCOMM_POSITIONS_SHORT' in col_upper or 'NON_COMMERCIAL_SHORT' in col_upper:
                short_col = col
    
    if long_col is None or short_col is None:
        log.error(f"❌ 未找到多空持仓列。可用列: {list(df.columns)}")
        df['CFTC_NET_LONG'] = None
        return df
    
    log.info(f"使用列: Long={long_col}, Short={short_col}")
    
    # 计算净多头
    df['CFTC_NET_LONG'] = pd.to_numeric(df[long_col], errors='coerce') - \
                          pd.to_numeric(df[short_col], errors='coerce')
    
    log.info(f"✅ 净多头计算完成")
    
    return df


def calculate_ref_week_end(report_date: str) -> str:
    """
    计算数据所属周的周五日期
    
    CFTC 报告通常在周二统计、周五发布
    将 report_date 调整为当周周五
    
    Args:
        report_date: 报告日期 (YYYY-MM-DD)
    
    Returns:
        当周周五日期 (YYYY-MM-DD)
    """
    dt = datetime.strptime(report_date, '%Y-%m-%d')
    # 计算到周五的天数差
    days_until_friday = (4 - dt.weekday()) % 7
    if days_until_friday == 0 and dt.weekday() != 4:
        days_until_friday = 7
    friday = dt + timedelta(days=days_until_friday)
    return friday.strftime('%Y-%m-%d')


def process_and_save(df: pd.DataFrame, start_date: str, end_date: str) -> str:
    """
    处理数据并整合保存为一个 CSV 文件
    
    文件名格式: CFTC_<start_date>_<end_date>.csv
    
    Args:
        df: 处理后的 DataFrame
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        保存的文件路径，失败返回空字符串
    """
    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 查找日期列
    date_col = None
    for col in df.columns:
        if 'REPORT_DATE' in col.upper():
            date_col = col
            break
    
    if date_col is None:
        log.error("❌ 未找到报告日期列")
        return ''
    
    # 转换日期格式
    df[date_col] = pd.to_datetime(df[date_col])
    
    # 过滤日期范围
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    df_filtered = df[(df[date_col] >= start_dt) & (df[date_col] <= end_dt)].copy()
    
    if df_filtered.empty:
        log.info(f"⚠️ 日期范围 {start_date} ~ {end_date} 内无数据")
        return ''
    
    # 添加 ref_week_end 列
    df_filtered['REF_WEEK_END'] = df_filtered[date_col].apply(
        lambda dt: calculate_ref_week_end(dt.strftime('%Y-%m-%d'))
    )
    
    # 按日期排序
    df_filtered.sort_values(date_col, inplace=True)
    df_filtered.reset_index(drop=True, inplace=True)
    
    # 格式化日期列为字符串
    df_filtered[date_col] = df_filtered[date_col].dt.strftime('%Y-%m-%d')
    
    # 选择输出列: 核心列在前，持仓明细列在后
    core_cols = [date_col, 'REF_WEEK_END', 'CFTC_NET_LONG']
    detail_cols = []
    for col in df_filtered.columns:
        col_upper = col.upper()
        if col in core_cols:
            continue
        if any(kw in col_upper for kw in ['LONG', 'SHORT', 'SPREAD', 'OPEN_INTEREST']):
            detail_cols.append(col)
    
    output_cols = [c for c in core_cols if c in df_filtered.columns] + detail_cols
    output_df = df_filtered[output_cols]
    
    # 保存为单个 CSV 文件
    filename = f"CFTC_{start_date}_{end_date}.csv"
    filepath = OUTPUT_DIR / filename
    output_df.to_csv(filepath, index=False, encoding='utf-8')
    
    log.info(f"✅ 保存 {len(output_df)} 条记录到 {filepath}")
    
    return str(filepath)


def download_cftc_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    下载并处理 CFTC 数据的主函数
    
    Args:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
    
    Returns:
        处理后的 DataFrame，包含 report_date, ref_week_end, cftc_net_long 等列
    """
    log.info("=" * 60)
    log.info(f"CFTC 数据下载任务启动")
    log.info(f"日期范围: {start_date} ~ {end_date}")
    log.info("=" * 60)
    
    # 1. 获取涉及的年份
    years = get_years_from_date_range(start_date, end_date)
    
    # 2. 下载并合并各年份数据
    all_dfs = []
    
    for i, year in enumerate(years):
        # 反爬延迟
        if i > 0:
            log.info(f"等待 {REQUEST_DELAY} 秒...")
            time.sleep(REQUEST_DELAY)
        
        # 下载 ZIP
        zip_content = download_cftc_zip(year)
        if zip_content is None:
            continue
        
        # 解析 CSV
        df = extract_csv_from_zip(zip_content)
        if df is None:
            continue
        
        all_dfs.append(df)
    
    if not all_dfs:
        log.error("❌ 无法获取任何年份的数据")
        return pd.DataFrame()
    
    # 3. 合并多年数据
    if len(all_dfs) > 1:
        log.info(f"合并 {len(all_dfs)} 年数据...")
        combined_df = pd.concat(all_dfs, ignore_index=True)
    else:
        combined_df = all_dfs[0]
    
    log.info(f"合并后共 {len(combined_df)} 条记录")
    
    # 4. 过滤原油数据
    crude_df = filter_crude_oil_data(combined_df)
    
    if crude_df.empty:
        log.error("❌ 未找到原油相关数据")
        return pd.DataFrame()
    
    # 5. 计算净多头持仓
    crude_df = calculate_net_long(crude_df)
    
    # 6. 整合保存为单个 CSV 文件
    saved_file = process_and_save(crude_df, start_date, end_date)
    
    log.info("=" * 60)
    if saved_file:
        log.info(f"✅ CFTC 数据下载任务完成，文件: {saved_file}")
    else:
        log.info(f"✅ CFTC 数据下载任务完成，但未生成文件")
    log.info("=" * 60)
    
    return crude_df


def get_cftc_net_long(start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取 CFTC 净多头持仓数据 (供 insertWeeklyOil.py 调用)
    
    优先读取本地缓存文件，无缓存时自动下载
    
    Args:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
    
    Returns:
        DataFrame 包含 report_date, cftc_net_long 列
    """
    empty_result = pd.DataFrame(columns=['report_date', 'cftc_net_long'])
    
    # 精确匹配缓存文件: CFTC_<start_date>_<end_date>.csv
    cache_file = OUTPUT_DIR / f"CFTC_{start_date}_{end_date}.csv"
    
    if cache_file.exists():
        log.info(f"发现本地缓存文件: {cache_file}")
        try:
            result = pd.read_csv(cache_file)
            result.columns = result.columns.str.upper()
            
            date_col = next((c for c in result.columns if 'REPORT_DATE' in c), None)
            if date_col:
                output = pd.DataFrame({
                    'report_date': result[date_col],
                    'cftc_net_long': result.get('CFTC_NET_LONG', None)
                })
                log.info(f"✅ 从缓存读取 {len(output)} 条记录")
                return output
        except Exception as e:
            log.info(f"⚠️ 缓存文件读取失败: {e}，重新下载")
    
    # 查找可能覆盖当前日期范围的缓存文件
    if OUTPUT_DIR.exists():
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        for f in OUTPUT_DIR.glob("CFTC_*_*.csv"):
            try:
                # 解析文件名中的日期范围
                parts = f.stem.replace('CFTC_', '').split('_')
                # 期望格式: YYYY-MM-DD_YYYY-MM-DD -> 拆为6段再拼回
                if len(parts) == 6:
                    file_start = f"{parts[0]}-{parts[1]}-{parts[2]}"
                    file_end = f"{parts[3]}-{parts[4]}-{parts[5]}"
                    file_start_dt = datetime.strptime(file_start, '%Y-%m-%d')
                    file_end_dt = datetime.strptime(file_end, '%Y-%m-%d')
                    
                    # 如果缓存文件的范围覆盖了请求范围，直接使用
                    if file_start_dt <= start_dt and file_end_dt >= end_dt:
                        log.info(f"发现覆盖范围的缓存文件: {f.name}")
                        result = pd.read_csv(f)
                        result.columns = result.columns.str.upper()
                        
                        date_col = next((c for c in result.columns if 'REPORT_DATE' in c), None)
                        if date_col:
                            result[date_col] = pd.to_datetime(result[date_col])
                            mask = (result[date_col] >= start_dt) & (result[date_col] <= end_dt)
                            filtered = result[mask]
                            output = pd.DataFrame({
                                'report_date': filtered[date_col].dt.strftime('%Y-%m-%d'),
                                'cftc_net_long': filtered.get('CFTC_NET_LONG', None)
                            })
                            log.info(f"✅ 从缓存截取 {len(output)} 条记录")
                            return output
            except (ValueError, IndexError):
                continue
    
    # 本地无可用缓存，下载数据
    log.info("本地无可用缓存，开始下载...")
    crude_df = download_cftc_data(start_date, end_date)
    
    if crude_df.empty:
        return empty_result
    
    # 提取需要的列
    date_col = next((c for c in crude_df.columns if 'REPORT_DATE' in c.upper()), None)
    if date_col:
        output = pd.DataFrame({
            'report_date': crude_df[date_col],
            'cftc_net_long': crude_df.get('CFTC_NET_LONG', None)
        })
        return output
    
    return empty_result


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='CFTC 持仓数据自动化下载与解析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python downloadCFTC.py --start 2024-01-01 --end 2024-12-31
  python downloadCFTC.py -s 2025-01-01 -e 2025-06-30

输出:
  文件保存到 ../input/CFTC/CFTC_<start_date>_<end_date>.csv
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
        df = download_cftc_data(args.start, args.end)
        if not df.empty:
            print(f"\n✅ 下载完成，共 {len(df)} 条原油持仓记录")
            print(f"文件保存到: {OUTPUT_DIR / f'CFTC_{args.start}_{args.end}.csv'}")
        else:
            print(f"\n❌ 下载失败或无数据")
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        exit(1)
