"""
Baker Hughes 钻井数自动化下载与解析模块
功能: 从 Baker Hughes 官网下载北美钻井数据 Excel，提取美国原油活跃钻井数

数据流程:
    1. 请求 Baker Hughes 官网，动态提取最新 Excel 下载链接
    2. 下载 Excel 到内存，解析 Master/Data 工作表
    3. 提取 Date + U.S. Oil 列，转换为 report_date + baker_hughes_rig_count
    4. 若需要历史回溯，额外下载归档 Excel 并合并
    5. 按日期范围过滤，整合保存为单个 CSV

作者: OpenCode
日期: 2026-06-10

使用方式:
    # 命令行调用
    python downloadBakerHughes.py --start 2024-01-01 --end 2024-12-31

    # Python 调用 (供 insertWeeklyOil.py 使用)
    from downloadBakerHughes import get_baker_hughes_rig_count
    df = get_baker_hughes_rig_count('2024-01-01', '2024-12-31')

输出文件:
    ../input/BakerHughes/BakerHughes_2024-01-01_2024-12-31.csv
    
代码结构
核心函数
函数	功能
parse_download_links()	解析官网 HTML, 动态提取最新 + 归档 Excel 链接
download_excel()	下载 Excel 到内存
parse_excel_to_dataframe()	解析 Excel: 自动定位表头行、Sheet、Date 列和 US Oil 列
save_to_csv()	按日期范围过滤并保存为单个 CSV
download_baker_hughes_data()	主函数：下载 + 解析 + 合并 + 保存
get_baker_hughes_rig_count()	供 insertWeeklyOil.py 调用，优先读本地缓存
_parse_bh_csv()	解析本地缓存 CSV 文件
设计文档要求实现
要求	实现位置
动态提取 Excel URL (非硬编码)	parse_download_links() 用 BeautifulSoup 解析
伪装请求头 + Referer	HEADERS 配置
自动定位 Sheet (Master/Data)	parse_excel_to_dataframe() 按 TARGET_SHEETS 候选列表匹配
自动定位表头行 (找 Date)	parse_excel_to_dataframe() 扫描前 20 行
US Oil 列模糊匹配	OIL_RIG_COLUMN_CANDIDATES 候选列表
物理极限拦截 (<=0 置 None)	parse_excel_to_dataframe() 中过滤
ref_week_end = report_date	Baker Hughes 发布日即为周五
归档补数 + 分段合并	download_baker_hughes_data() 下载 latest + archive 后 concat + drop_duplicates
日期切片输出	save_to_csv()
反爬延迟	REQUEST_DELAY = 2.0
使用方式
# 命令行
python downloadBakerHughes.py --start 2024-01-01 --end 2024-12-31

# Python 调用
from downloadBakerHughes import get_baker_hughes_rig_count
df = get_baker_hughes_rig_count('2024-01-01', '2024-12-31')
输出文件
input/BakerHughes/
  BakerHughes_2024-01-01_2024-12-31.csv
CSV 包含列:
- report_date - 报告日期 (周五)
- ref_week_end - 等于 report_date
- baker_hughes_rig_count - 美国原油活跃钻井数 (INT)
    
"""

import pandas as pd
import numpy as np
import requests
import io
import re
import time
from datetime import datetime, timedelta
from typing import Optional, List
from pathlib import Path
from bs4 import BeautifulSoup

# 项目内部模块
import log4ak

# ============================================================
# 配置区域
# ============================================================

log = log4ak.LogManager(log_level=log4ak.INFO)

# Baker Hughes 官网
BH_PAGE_URL = "https://rigcount.bakerhughes.com/na-rig-count"

# 请求头 (伪装浏览器，设置 Referer)
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,'
              'image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://rigcount.bakerhughes.com/',
    'Connection': 'keep-alive',
}

# 输出目录
OUTPUT_DIR = Path(__file__).parent.parent / "input" / "BakerHughes"

# 反爬延迟 (秒)
REQUEST_DELAY = 2.0

# 目标 Sheet 名称候选列表 (按优先级)
TARGET_SHEETS = ['Master', 'Data', 'Rigs by Country', 'US Rig Count']

# 目标列名候选 (U.S. Oil 钻井数)
OIL_RIG_COLUMN_CANDIDATES = [
    'U.S. Oil', 'US Oil', 'US Oil Rig', 'Oil', 'United States - Oil'
]


# ============================================================
# 1. 网页解析 - 提取动态 Excel 下载链接
# ============================================================

def parse_download_links(page_url: str = BH_PAGE_URL) -> dict:
    """
    解析 Baker Hughes 官网页面，提取 Excel 下载链接

    Returns:
        dict 包含:
            'latest': 最新报告的 Excel URL
            'archive': 归档历史数据的 Excel URL 列表
    """
    log.info(f"正在解析 Baker Hughes 页面: {page_url}")

    links = {'latest': None, 'archive': []}

    try:
        resp = requests.get(page_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, 'html.parser')

        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            text = a_tag.get_text(strip=True).lower()

            # 跳过非 Excel 链接
            if not (href.endswith('.xlsx') or href.endswith('.xls')
                    or '.xlsx' in href or '.xls' in href
                    or 'download' in href.lower()):
                continue

            # 将相对路径转换为绝对路径
            if href.startswith('/'):
                href = f"https://rigcount.bakerhughes.com{href}"

            # 最新报告
            if 'new report' in text or 'current' in text:
                links['latest'] = href
                log.info(f"  [latest] {href}")
            # 归档
            elif 'archive' in text or 'rig count' in text:
                links['archive'].append(href)
                log.info(f"  [archive] {href}")

        # 如果没有精确匹配，收集所有 Excel 链接
        if links['latest'] is None and not links['archive']:
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                if href.endswith('.xlsx') or href.endswith('.xls'):
                    if href.startswith('/'):
                        href = f"https://rigcount.bakerhughes.com{href}"
                    links['archive'].append(href)
                    log.info(f"  [excel] {href}")

        log.info(f"找到链接: latest={links['latest'] is not None}, "
                 f"archive={len(links['archive'])}")

        return links

    except requests.exceptions.RequestException as e:
        log.error(f"页面请求失败: {e}")
        return links
    except Exception as e:
        log.error(f"页面解析失败: {e}")
        return links


# ============================================================
# 2. Excel 下载与解析
# ============================================================

def download_excel(url: str) -> Optional[bytes]:
    """
    下载 Excel 文件

    Returns:
        文件字节内容，失败返回 None
    """
    log.info(f"正在下载 Excel: {url}")

    try:
        resp = requests.get(url, headers=HEADERS, timeout=120, stream=True)
        resp.raise_for_status()

        content = resp.content
        log.info(f"下载成功，大小: {len(content) / 1024 / 1024:.2f} MB")
        return content

    except requests.exceptions.RequestException as e:
        log.error(f"Excel 下载失败: {e}")
        return None


def parse_excel_to_dataframe(content: bytes) -> pd.DataFrame:
    """
    解析 Excel 字节内容，提取 report_date 和 baker_hughes_rig_count

    Returns:
        DataFrame 包含 report_date, ref_week_end, baker_hughes_rig_count
    """
    log.info("解析 Excel 数据...")

    try:
        xls = pd.ExcelFile(io.BytesIO(content))
        sheet_names = xls.sheet_names
        log.info(f"  工作表列表: {sheet_names}")

        # 定位目标 Sheet
        target_sheet = None
        for candidate in TARGET_SHEETS:
            for name in sheet_names:
                if candidate.lower() in name.lower():
                    target_sheet = name
                    break
            if target_sheet:
                break

        # 如果没匹配到候选名，尝试用第一个 Sheet
        if target_sheet is None:
            target_sheet = sheet_names[0]
            log.info(f"  未匹配到候选 Sheet，使用第一个: {target_sheet}")
        else:
            log.info(f"  使用工作表: {target_sheet}")

        # 读取 Sheet
        df = pd.read_excel(
            io.BytesIO(content),
            sheet_name=target_sheet,
            header=None  # 先不指定表头，手动定位
        )

        log.info(f"  原始数据: {df.shape[0]} 行 x {df.shape[1]} 列")

        # 自动定位表头行: 查找包含 'Date' 的行
        header_row = None
        for i in range(min(20, len(df))):
            row_values = df.iloc[i].astype(str).str.strip().str.lower().tolist()
            if 'date' in row_values:
                header_row = i
                break

        if header_row is None:
            log.error("  未找到包含 'Date' 的表头行")
            return pd.DataFrame()

        log.info(f"  表头行: 第 {header_row} 行")

        # 重新设置表头
        df.columns = df.iloc[header_row].astype(str).str.strip()
        df = df.iloc[header_row + 1:].reset_index(drop=True)

        # 查找日期列
        date_col = None
        for col in df.columns:
            if 'date' in str(col).lower():
                date_col = col
                break

        if date_col is None:
            log.error(f"  未找到日期列。可用列: {list(df.columns[:10])}")
            return pd.DataFrame()

        # 查找 US Oil 钻井数列
        oil_col = None
        for candidate in OIL_RIG_COLUMN_CANDIDATES:
            for col in df.columns:
                if candidate.lower() in str(col).lower():
                    oil_col = col
                    break
            if oil_col:
                break

        if oil_col is None:
            log.error(f"  未找到 US Oil 钻井数列。可用列: {list(df.columns[:20])}")
            return pd.DataFrame()

        log.info(f"  日期列: '{date_col}', 钻井数列: '{oil_col}'")

        # 提取目标列
        result = df[[date_col, oil_col]].copy()
        result.columns = ['report_date', 'baker_hughes_rig_count']

        # 转换日期
        result['report_date'] = pd.to_datetime(result['report_date'], errors='coerce')
        result.dropna(subset=['report_date'], inplace=True)

        # 转换钻井数为整型 (无效值置 None)
        result['baker_hughes_rig_count'] = pd.to_numeric(
            result['baker_hughes_rig_count'], errors='coerce'
        )
        # 物理极限拦截: <= 0 置为 None
        result.loc[result['baker_hughes_rig_count'] <= 0,
                   'baker_hughes_rig_count'] = None

        # ref_week_end = report_date (Baker Hughes 发布日本身就是周五)
        result['ref_week_end'] = result['report_date']

        # 格式化日期
        result['report_date'] = result['report_date'].dt.strftime('%Y-%m-%d')
        result['ref_week_end'] = result['ref_week_end'].dt.strftime('%Y-%m-%d')

        # 按日期排序、去重
        result.sort_values('report_date', inplace=True)
        result.drop_duplicates(subset=['report_date'], keep='last', inplace=True)
        result.reset_index(drop=True, inplace=True)

        log.info(f"解析完成，共 {len(result)} 条有效记录")

        return result

    except Exception as e:
        log.error(f"Excel 解析失败: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


# ============================================================
# 3. 数据保存
# ============================================================

def save_to_csv(df: pd.DataFrame, start_date: str, end_date: str) -> str:
    """
    过滤日期范围并保存为 CSV

    Returns:
        保存的文件路径，失败返回空字符串
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 过滤日期范围
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    df['_dt'] = pd.to_datetime(df['report_date'])
    filtered = df[(df['_dt'] >= start_dt) & (df['_dt'] <= end_dt)].copy()
    filtered.drop(columns=['_dt'], inplace=True)

    if filtered.empty:
        log.info(f"日期范围 {start_date} ~ {end_date} 内无数据")
        return ''

    # 排序
    filtered.sort_values('report_date', inplace=True)
    filtered.reset_index(drop=True, inplace=True)

    # 保存
    filename = f"BakerHughes_{start_date}_{end_date}.csv"
    filepath = OUTPUT_DIR / filename
    filtered.to_csv(filepath, index=False, encoding='utf-8')

    log.info(f"保存 {len(filtered)} 条记录到 {filepath}")
    return str(filepath)


# ============================================================
# 4. 主入口函数
# ============================================================

def download_baker_hughes_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    下载并处理 Baker Hughes 数据的主函数

    Args:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)

    Returns:
        处理后的 DataFrame
    """
    log.info("=" * 60)
    log.info("Baker Hughes 数据下载任务启动")
    log.info(f"日期范围: {start_date} ~ {end_date}")
    log.info("=" * 60)

    all_dfs = []

    # 1. 解析页面获取下载链接
    links = parse_download_links()

    # 2. 下载并解析所有 Excel
    urls_to_download = []
    if links['latest']:
        urls_to_download.append(('latest', links['latest']))
    for i, url in enumerate(links['archive']):
        urls_to_download.append((f'archive_{i}', url))

    if not urls_to_download:
        log.error("未找到任何 Excel 下载链接")
        return pd.DataFrame()

    for label, url in urls_to_download:
        content = download_excel(url)
        if content is None:
            continue

        df = parse_excel_to_dataframe(content)
        if not df.empty:
            all_dfs.append(df)
            log.info(f"  [{label}] 解析得到 {len(df)} 条记录")

        # 反爬延迟
        if len(urls_to_download) > 1:
            time.sleep(REQUEST_DELAY)

    if not all_dfs:
        log.error("所有 Excel 均解析失败")
        return pd.DataFrame()

    # 3. 合并并去重
    if len(all_dfs) > 1:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.drop_duplicates(subset=['report_date'], keep='last', inplace=True)
        combined.sort_values('report_date', inplace=True)
        combined.reset_index(drop=True, inplace=True)
    else:
        combined = all_dfs[0]

    log.info(f"合并后共 {len(combined)} 条记录")

    # 4. 保存到 CSV
    saved_file = save_to_csv(combined, start_date, end_date)

    log.info("=" * 60)
    if saved_file:
        log.info(f"任务完成，文件: {saved_file}")
    else:
        log.info("任务完成，但未生成文件")
    log.info("=" * 60)

    return combined


def get_baker_hughes_rig_count(start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取 Baker Hughes 钻井数 (供 insertWeeklyOil.py 调用)

    优先读取本地缓存文件，无缓存时自动下载

    Returns:
        DataFrame 包含 report_date, baker_hughes_rig_count 列
    """
    empty_result = pd.DataFrame(columns=['report_date', 'baker_hughes_rig_count'])

    # 1. 精确匹配缓存文件
    cache_file = OUTPUT_DIR / f"BakerHughes_{start_date}_{end_date}.csv"
    if cache_file.exists():
        log.info(f"找到缓存文件: {cache_file.name}")
        return _parse_bh_csv(cache_file, start_date, end_date)

    # 2. 查找覆盖范围的缓存文件
    if OUTPUT_DIR.exists():
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        for f in OUTPUT_DIR.glob("BakerHughes_*_*.csv"):
            try:
                parts = f.stem.replace('BakerHughes_', '').split('_')
                if len(parts) == 2:
                    file_start = parts[0]
                    file_end = parts[1]
                    file_start_dt = datetime.strptime(file_start, '%Y-%m-%d')
                    file_end_dt = datetime.strptime(file_end, '%Y-%m-%d')

                    if file_start_dt <= start_dt and file_end_dt >= end_dt:
                        log.info(f"找到覆盖范围的缓存文件: {f.name}")
                        return _parse_bh_csv(f, start_date, end_date)
            except (ValueError, IndexError):
                continue

    # 3. 本地无缓存，在线下载
    log.info("本地无缓存，开始在线下载...")
    combined = download_baker_hughes_data(start_date, end_date)

    if combined.empty:
        return empty_result

    return combined[['report_date', 'baker_hughes_rig_count']].copy()


def _parse_bh_csv(filepath: Path, start_date: str, end_date: str) -> pd.DataFrame:
    """
    解析本地缓存 CSV 文件

    Returns:
        DataFrame 包含 report_date, baker_hughes_rig_count 列
    """
    try:
        df = pd.read_csv(filepath)

        # 标准化列名
        col_map = {}
        for col in df.columns:
            cl = col.lower().strip()
            if 'report_date' in cl or 'date' in cl:
                col_map[col] = 'report_date'
            elif 'rig_count' in cl or 'rig' in cl:
                col_map[col] = 'baker_hughes_rig_count'

        df.rename(columns=col_map, inplace=True)

        if 'report_date' not in df.columns:
            log.error("CSV 中未找到日期列")
            return pd.DataFrame(columns=['report_date', 'baker_hughes_rig_count'])

        # 过滤日期范围
        df['report_date'] = pd.to_datetime(df['report_date'])
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        df = df[(df['report_date'] >= start_dt) & (df['report_date'] <= end_dt)].copy()
        df['report_date'] = df['report_date'].dt.strftime('%Y-%m-%d')

        # 确保 baker_hughes_rig_count 列存在
        if 'baker_hughes_rig_count' not in df.columns:
            df['baker_hughes_rig_count'] = None

        result = df[['report_date', 'baker_hughes_rig_count']].copy()
        result.sort_values('report_date', inplace=True)
        result.reset_index(drop=True, inplace=True)

        log.info(f"从缓存 CSV 解析 {len(result)} 条记录")
        return result

    except Exception as e:
        log.error(f"解析缓存 CSV 失败: {e}")
        return pd.DataFrame(columns=['report_date', 'baker_hughes_rig_count'])


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Baker Hughes 钻井数自动化下载与解析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python downloadBakerHughes.py --start 2024-01-01 --end 2024-12-31
  python downloadBakerHughes.py -s 2023-01-01 -e 2024-06-30

output:
  ../input/BakerHughes/BakerHughes_<start_date>_<end_date>.csv
        """
    )

    parser.add_argument(
        '-s', '--start', type=str, required=True,
        help='start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '-e', '--end', type=str, required=True,
        help='end date (YYYY-MM-DD)'
    )

    args = parser.parse_args()

    try:
        df = download_baker_hughes_data(args.start, args.end)
        if not df.empty:
            filepath = OUTPUT_DIR / f"BakerHughes_{args.start}_{args.end}.csv"
            print(f"\n[OK] download complete, {len(df)} records")
            print(f"     file: {filepath}")
        else:
            print(f"\n[FAIL] no data")
    except Exception as e:
        print(f"\n[FAIL] execution error: {e}")
        exit(1)
