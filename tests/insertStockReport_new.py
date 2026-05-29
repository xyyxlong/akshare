import time
import random
import numpy as np
import akshare as ak
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import log4ak
from tqdm import tqdm
from getAllStock import get_all_stocks, get_select_stocks
import insert2Mysql as i2m

# 日志配置
log = log4ak.LogManager(log_level=log4ak.INFO)

@dataclass
class InsertStockReportConfig:
    """配置类，替代全局变量"""
    start_year: str = "2019"
    chunk_num: int = 10
    max_consecutive_errors: int = 30
    out_time: int = 5
    reconnect_time: int = 100

# ----------------- 数据库列名映射定义 -----------------
COLUMN_MAP = {
    'stock_code': 'stock_code',
    'stock_name': 'stock_name',
    '日期': 'report_date',
    '摊薄每股收益(元)': 'diluted_eps',
    '加权每股收益(元)': 'weighted_eps',
    '每股收益_调整后(元)': 'adjusted_eps',
    '扣除非经常性损益后的每股收益(元)': 'non_gaap_eps',
    '每股净资产_调整前(元)': 'net_asset_per_share',
    '每股净资产_调整后(元)': 'adjusted_net_asset',
    '每股经营性现金流(元)': 'operating_cash_flow_per_share',
    '每股资本公积金(元)': 'capital_reserve_per_share',
    '每股未分配利润(元)': 'retained_earnings_per_share',
    '调整后的每股净资产(元)': 'adjusted_net_asset_value',
    '总资产利润率(%)': 'roa',
    '主营业务利润率(%)': 'operating_profit_margin',
    '总资产净利润率(%)': 'roa_profit_margin',
    '成本费用利润率(%)': 'cost_profit_ratio',
    '营业利润率(%)': 'operating_profit_ratio',
    '主营业务成本率(%)': 'main_cost_ratio',
    '销售净利率(%)': 'net_profit_margin',
    '股本报酬率(%)': 'capital_return_ratio',
    '净资产报酬率(%)': 'roe_return_ratio',
    '资产报酬率(%)': 'asset_return_ratio',
    '销售毛利率(%)': 'gross_profit_margin',
    '三项费用比重': 'three_expense_ratio',
    '非主营比重': 'non_main_ratio',
    '主营利润比重': 'main_profit_ratio',
    '股息发放率(%)': 'dividend_payout_ratio',
    '投资收益率(%)': 'investment_return_ratio',
    '净资产收益率(%)':'roe',
    '加权净资产收益率(%)': 'weighted_roe',
    '主营业务收入增长率(%)': 'revenue_growth',
    '净利润增长率(%)': 'net_profit_growth',
    '净资产增长率(%)': 'net_asset_growth',
    '总资产增长率(%)': 'total_asset_growth',
    '应收账款周转率(次)': 'receivables_turnover',
    '应收账款周转天数(天)': 'receivables_days',
    '存货周转天数(天)': 'inventory_days',
    '存货周转率(次)': 'inventory_turnover',
    '固定资产周转率(次)': 'fixed_asset_turnover',
    '总资产周转率(次)': 'total_asset_turnover',
    '总资产周转天数(天)': 'total_asset_days',
    '流动资产周转率(次)': 'current_asset_turnover',
    '流动资产周转天数(天)': 'current_asset_days',
    '股东权益周转率(次)': 'equity_turnover',
    '流动比率': 'current_ratio',
    '速动比率': 'quick_ratio',
    '现金比率(%)': 'cash_ratio',
    '利息支付倍数': 'interest_coverage',
    '长期债务与营运资金比率(%)': 'long_term_debt_ratio',
    '股东权益比率(%)': 'equity_ratio',
    '长期负债比率(%)': 'long_term_liability_ratio',
    '股东权益与固定资产比率(%)': 'equity_to_fixed_assets',
    '负债与所有者权益比率(%)': 'debt_to_equity',
    '长期资产与长期资金比率(%)': 'long_term_assets_ratio',
    '资本化比率(%)': 'capitalization_ratio',
    '固定资产净值率(%)': 'fixed_asset_net_ratio',
    '资本固定化比率(%)': 'fixed_capitalization_ratio',
    '产权比率(%)': 'equity_multiplier',
    '清算价值比率(%)': 'liquidation_value_ratio',
    '固定资产比重(%)': 'fixed_asset_ratio',
    '资产负债率(%)': 'asset_liability_ratio',
    '经营现金净流量对销售收入比率(%)': 'cash_flow_to_sales',
    '资产的经营现金流量回报率(%)': 'cash_flow_return_on_assets',
    '经营现金净流量与净利润的比率(%)': 'cash_flow_to_net_income',
    '经营现金净流量对负债比率(%)': 'cash_flow_to_debt',
    '现金流量比率(%)': 'cash_flow_ratio',
    '总资产(元)': 'total_assets',
    '短期股票投资(元)': 'short_stock_invest',
    '短期债券投资(元)': 'short_bond_invest',
    '短期其它经营性投资(元)': 'short_other_invest',
    '长期股票投资(元)': 'long_stock_invest',
    '长期债券投资(元)': 'long_bond_invest',
    '长期其它经营性投资(元)': 'long_other_invest',
    '主营业务利润(元)': 'main_profit',
    '扣除非经常性损益后的净利润(元)': 'non_gaap_net_profit',
    '1年以内应收帐款(元)': 'receivables_1y',
    '1-2年以内应收帐款(元)': 'receivables_1_2y',
    '2-3年以内应收帐款(元)': 'receivables_2_3y',
    '3年以内应收帐款(元)': 'receivables_over_3y',
    '1年以内预付货款(元)': 'prepayment_1y',
    '1-2年以内预付货款(元)': 'prepayment_1_2y',
    '2-3年以内预付货款(元)': 'prepayment_2_3y',
    '3年以内预付货款(元)': 'prepayment_over_3y',
    '1年以内其它应收款(元)': 'other_receivables_1y',
    '1-2年以内其它应收款(元)': 'other_receivables_1_2y',
    '2-3年以内其它应收款(元)': 'other_receivables_2_3y',
    '3年以内其它应收款(元)': 'other_receivables_over_3y'
}

REVERSE_COLUMN_MAP = {v: k for k, v in COLUMN_MAP.items()}
# --------------------------------------------------------

class ConsecutiveErrorException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class FinancialReportFetcher:
    def __init__(self, config: InsertStockReportConfig):
        self.config = config

    def run_with_timeout(self, func, timeout: int, *args, **kwargs):
        """使用线程池包装超时调用"""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            except TimeoutError:
                executor.shutdown(wait=False, cancel_futures=True)
                raise TimeoutError("接口响应超时")

    def get_financial_report(self, stock_code: str) -> Optional[pd.DataFrame]:
        """获取单个股票财报数据"""
        for attempt in range(self.config.max_consecutive_errors):
            try:
                df = self.run_with_timeout(
                    ak.stock_financial_analysis_indicator,
                    self.config.out_time,
                    symbol=stock_code,
                    start_year=self.config.start_year
                )
                return df
            except TimeoutError:
                log.error(f"[{stock_code}] 接口调用超时 (尝试 {attempt+1}/{self.config.max_consecutive_errors})")
            except Exception as e:
                log.error(f"[{stock_code}] 接口调用失败: {e} (尝试 {attempt+1}/{self.config.max_consecutive_errors})")

            if attempt < self.config.max_consecutive_errors - 1:
                sleep_time = random.uniform(1, self.config.reconnect_time)
                time.sleep(sleep_time)
            else:
                log.error(f"[{stock_code}] 达到最大重试次数，跳过")
                return None
        return None

    def process_fin_data_batch(self, df: pd.DataFrame) -> Tuple[List[Tuple], str]:
        """向量化处理财报数据，转换为入库格式"""
        if df is None or df.empty:
            return [], ""

        processed_df = df.rename(columns=COLUMN_MAP).copy()
        
        # 确保日期格式
        if 'report_date' in processed_df.columns:
            processed_df['report_date'] = pd.to_datetime(processed_df['report_date']).dt.strftime('%Y-%m-%d')
            
        # 向量化处理缺失值，转换为 None
        processed_df = processed_df.astype(object).where(pd.notnull(processed_df), None)

        columns = ', '.join(processed_df.columns)
        placeholders = ', '.join(['%s'] * len(processed_df.columns))
        sql = f"""
            INSERT IGNORE INTO stock_financial_reports ({columns})
            VALUES ({placeholders})
        """
        
        # 快速转换为 tuple list
        batch_data = [tuple(x) for x in processed_df.to_numpy()]
        return batch_data, sql

    def execute_batch_insert(self, path: str = "all"):
        """主执行流程"""
        if path == "all":
            stock_df = get_all_stocks()
        else:
            stock_df = get_select_stocks()

        if stock_df is None or stock_df.empty:
            log.error("未获取到股票列表。")
            return

        stock_df = stock_df[['代码', '名称']]
        total_rows = len(stock_df)
        chunk_indices = np.array_split(np.arange(total_rows), self.config.chunk_num)
        
        log.info(f"总计 {total_rows} 只股票，分 {self.config.chunk_num} 块处理")
        error_count = 0

        for file_num, chunk_idx in enumerate(chunk_indices):
            chunk_df = stock_df.iloc[chunk_idx]
            log.info(f"==> 开始处理第 {file_num+1} 批，共 {len(chunk_df)} 只")
            
            # 使用 list 收集该批次的 df，避免 pd.concat 性能陷阱
            batch_dfs = []
            
            for check_count, (_, row) in enumerate(chunk_df.iterrows(), 1):
                r_code = row['代码']
                r_name = row['名称']
                
                try:
                
                    try:
                        dffin = self.get_financial_report(r_code)
                        if dffin is not None and not dffin.empty:
                            dffin['stock_code'] = r_code
                            dffin['stock_name'] = r_name
                            batch_dfs.append(dffin)
                            log.info(f"  [{r_code}] 获取成功，条数: {len(dffin)}")
                        else:
                            log.error(f"  [{r_code}] 无数据返回")

                        error_count = 0
                        time.sleep(1) # 礼貌休眠
                        
                    except Exception as e:
                        error_count += 1
                        log.error(f"  [{r_code}] 处理异常: {e}. 连续错误={error_count}")
                        sleep_time = random.uniform(1, self.config.reconnect_time)
                        time.sleep(sleep_time)
                        if error_count >= self.config.max_consecutive_errors:
                            raise ConsecutiveErrorException(f"连续 {error_count} 次异常，服务终止")
                        
                except KeyboardInterrupt:
                    # 捕获 Ctrl+C，执行优雅退出，保存当前已爬取的数据
                    log.error("检测到手动中断 (Ctrl+C)，正在紧急保存当前批次已获取的数据...")
                    if batch_dfs:
                        final_chunk_df = pd.concat(batch_dfs, ignore_index=True)
                        batch_data, sql = self.process_fin_data_batch(final_chunk_df)
                        if batch_data:
                            i2m.insert_to_mysql(batch_data, sql)
                            log.info(f"<== 紧急保存完成，入库 {len(batch_data)} 条明细。")
                    log.info("程序已安全退出。")
                    return  # 终止程序执行
            
            # 合并本批次数据并入库
            if batch_dfs:
                final_chunk_df = pd.concat(batch_dfs, ignore_index=True)
                batch_data, sql = self.process_fin_data_batch(final_chunk_df)
                if batch_data:
                    i2m.insert_to_mysql(batch_data, sql)
                    log.info(f"<== 第 {file_num+1} 批入库完成，共入库 {len(batch_data)} 条明细")
            else:
                log.info(f"<== 第 {file_num+1} 批无数据入库")

        log.info("🎉 所有分块处理并入库完成")
        return "所有分块处理完成"

def get_stockfin_data_from_mysql(stock_code: str, start_date: str = None) -> pd.DataFrame:
    """从数据库反向读取"""
    sql = f"""
        SELECT {', '.join(REVERSE_COLUMN_MAP.keys())}
        FROM stock_financial_reports
        WHERE stock_code = %s
    """
    columns, rows = i2m._execute_query(sql, (stock_code,))
    if not rows:
        return pd.DataFrame()
        
    df = pd.DataFrame(rows, columns=columns)
    return df.rename(columns=REVERSE_COLUMN_MAP)

if __name__ == "__main__":
    config = InsertStockReportConfig()
    fetcher = FinancialReportFetcher(config)
    fetcher.execute_batch_insert(path="all")#all or select
