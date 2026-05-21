import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Any

import akshare as ak
import log4ak
from getAllStock import get_all_stocks, get_select_stocks, ipodatefilter_stocks
import insertSelectStockPE as issp
import insert2Mysql as ins
import get_stockPE_his as gsh

# 日志配置
log = log4ak.LogManager(log_level=log4ak.INFO)

@dataclass
class SelectStockConfig:
    """选股配置类，替代全局变量"""
    is_my: bool = True               # 是否选取自选配置 False/True
    is_mysql: bool = True            # PE数据来源，True=数据库, False=Akshare
    chunk_num: int = 1               # 全市场数据分块处理数量
    start_year: str = "2019"         # 计算的起始年份
    roe_min: float = 15.0            # ROE最低要求 (%)
    pe_max: float = 25.0             # PE最大要求
    past_day: int = 30               # 获取PE数据的过去天数
    past_year: int = 5               # 获取财务数据的过去年数
    max_consecutive_errors: int = 3  # 最大允许连续错误次数
    out_time: int = 10               # 接口长时间无返回报错(秒)
    reconnect_time: int = 30         # 断线重连休眠时间(秒)
    base_path: Path = Path(__file__).parent.parent  # 项目根目录 (akshare/)

class ConsecutiveErrorException(Exception):
    """连续异常超过阈值时触发"""
    def __init__(self, error_code: int, message: str):
        self.error_code = error_code
        self.message = message
        super().__init__(self.message)

def run_with_timeout(func, timeout: int, *args, **kwargs):
    """使用线程池包装超时调用的通用辅助函数"""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            executor.shutdown(wait=False, cancel_futures=True)
            raise TimeoutError("任务执行超时")

def safe_divide(numerator: float, denominator: float, default: Optional[float] = None) -> Optional[float]:
    """安全的除法操作，防止除以 0 或 NaN"""
    if denominator is None or pd.isna(denominator) or denominator == 0:
        return default
    if numerator is None or pd.isna(numerator):
        return default
    return numerator / denominator

class StockSelector:
    def __init__(self, config: SelectStockConfig):
        self.config = config

    def check_roe_cash_ebit(self, stock_code: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        获取核心财务指标：ROE, 现金利润比, 净利增长率, 资产负债率, 应收账款周转天数
        """
        df = None
        for attempt in range(self.config.max_consecutive_errors):
            try:
                log.info(f"[{stock_code}] 获取 {self.config.start_year} 至今财报数据")
                df = run_with_timeout(
                    ak.stock_financial_analysis_indicator, 
                    self.config.out_time, 
                    symbol=stock_code, 
                    start_year=self.config.start_year
                )
                if df is not None and not df.empty:
                    break
            except TimeoutError:
                log.error(f"[{stock_code}] 财报接口调用超时 (尝试 {attempt+1}/{self.config.max_consecutive_errors})")
            except Exception as e:
                log.error(f"[{stock_code}] 财报接口调用失败: {e} (尝试 {attempt+1}/{self.config.max_consecutive_errors})")
            
            if attempt < self.config.max_consecutive_errors - 1:
                time.sleep(self.config.reconnect_time)
            else:
                log.error(f"[{stock_code}] 无法获取财报数据，跳过")
                return None, None, None, None, None
        
        if df is None or df.empty:
            return None, None, None, None, None

        # 数据清洗和字段处理
        clean_df = df.rename(columns={
            '资产负债率(%)': 'debt_ratio',
            '应收账款周转天数(天)': 'receivable_days'
        }).copy()
        
        # 提取12月31日的年报
        clean_df['日期'] = pd.to_datetime(clean_df['日期'], errors='coerce')
        year_end_mask = (clean_df['日期'].dt.month == 12) & (clean_df['日期'].dt.day == 31)
        clean_df = clean_df[year_end_mask].sort_values('日期', ascending=False)
        
        # 数值转换
        numeric_cols = [
            '净资产收益率(%)', 'debt_ratio', 'receivable_days', 
            '每股经营性现金流(元)', '扣除非经常性损益后的每股收益(元)', 
            '扣除非经常性损益后的净利润(元)'
        ]
        for col in numeric_cols:
            if col in clean_df.columns:
                clean_df[col] = pd.to_numeric(clean_df[col].replace('--', np.nan), errors='coerce')

        if clean_df.empty:
            return None, None, None, None, None

        # 1. 平均ROE (过去 N 年)
        roe_values = clean_df['净资产收益率(%)'].head(self.config.past_year).dropna()
        var1 = roe_values.mean() if not roe_values.empty else None
        
        # 2. 经营现金流/扣非净利润 比率
        cash_flow_mean = clean_df['每股经营性现金流(元)'].head(self.config.past_year).fillna(0).mean()
        profit_mean = clean_df['扣除非经常性损益后的每股收益(元)'].head(self.config.past_year).fillna(0).mean()
        var2 = safe_divide(cash_flow_mean, profit_mean)
        
        # 3. 最新扣非净利润 / 前 N 年平均扣非净利润
        profit_series = clean_df['扣除非经常性损益后的净利润(元)'].dropna()
        var3 = None
        if len(profit_series) >= 2:
            latest_profit = profit_series.iloc[0]
            historical_avg_profit = profit_series.iloc[1:self.config.past_year+1].mean()
            var3 = safe_divide(latest_profit, historical_avg_profit)
        
        # 4. 资产负债率均值
        debt_ratios = clean_df['debt_ratio'].head(self.config.past_year).dropna()
        var4 = debt_ratios.mean() if not debt_ratios.empty else None
        
        # 5. 应收账款周转天数均值
        receivable_values = clean_df['receivable_days'].head(self.config.past_year).dropna()
        var5 = receivable_values.mean() if not receivable_values.empty else None
        
        log.debug(f"[{stock_code}] 指标: ROE={var1}, 现金流比={var2}, 净利增长={var3}, 负债率={var4}, 周转天数={var5}")
        return var1, var2, var3, var4, var5

    def get_pe_from_akshare(self, stock_code: str, stock_name: str) -> Optional[pd.DataFrame]:
        """通过 Akshare 获取 PE 数据并自动入库"""
        df = None
        for attempt in range(self.config.max_consecutive_errors):
            try:
                log.info(f"[{stock_code}] 通过网络接口获取市盈率数据")
                df = run_with_timeout(
                    ak.stock_a_indicator_lg, 
                    self.config.out_time, 
                    symbol=stock_code
                )
                if df is not None and not df.empty:
                    break
            except TimeoutError:
                log.error(f"[{stock_code}] PE接口调用超时 (尝试 {attempt+1}/{self.config.max_consecutive_errors})")
            except Exception as e:
                log.error(f"[{stock_code}] PE接口调用失败: {e} (尝试 {attempt+1}/{self.config.max_consecutive_errors})")
            
            if attempt < self.config.max_consecutive_errors - 1:
                time.sleep(self.config.reconnect_time)
            else:
                log.error(f"[{stock_code}] 无法获取PE数据，跳过")
                return None

        # 存入本地数据库作为缓存
        try:
            df = df.assign(stock_code=stock_code, stock_name=stock_name)
            batch_data = issp.process_pe_data_batch(df)
            ins.insert_to_mysql(batch_data, issp.INSERT_SQL)
            log.info(f"[{stock_code}] PE数据成功存入本地数据库缓存")
        except Exception as e:
            log.error(f"[{stock_code}] PE数据入库失败: {e}")
            
        return df

    def check_pe_condition(self, stock_code: str, stock_name: str) -> Tuple[Optional[float], Optional[float]]:
        """获取并计算过去的 PE_TTM 均值 和 股息率 dv_ratio 均值"""
        df = pd.DataFrame()
        
        if self.config.is_mysql:
            df = gsh.get_stock_pe_his(stock_code)
            if df is not None and not df.empty:
                df = df.reset_index().rename(columns={'日期': 'trade_date'})
            else:
                df = self.get_pe_from_akshare(stock_code, stock_name)
        else:
            df = self.get_pe_from_akshare(stock_code, stock_name)

        if df is None or df.empty:
            log.error(f"[{stock_code}] 无有效市盈率数据")
            return None, None
            
        # 转换日期并筛选
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        now = datetime.datetime.now()
        date_threshold = (now - datetime.timedelta(days=self.config.past_day)).date()
        year_threshold = (now - datetime.timedelta(days=self.config.past_year * 365)).date()
        
        df_date_filtered = df[df['trade_date'].dt.date > date_threshold].copy()
        
        # PE_TTM 均值 (忽略空值)
        pe_ttm_avg = None
        if 'pe_ttm' in df_date_filtered.columns and not df_date_filtered['pe_ttm'].dropna().empty:
            pe_ttm_avg = df_date_filtered['pe_ttm'].astype(float).mean()
            
        # DV_RATIO 均值 (过去5年，空值按0计算)
        dv_ratio_avg = None
        df_year_filtered = df[df['trade_date'].dt.date > year_threshold].copy()
        if 'dv_ratio' in df_year_filtered.columns:
            dv_series = df_year_filtered['dv_ratio'].fillna(0).astype(float)
            dv_ratio_avg = dv_series.mean() if not dv_series.empty else 0.0

        pe_str = f"{pe_ttm_avg:.2f}" if pe_ttm_avg is not None else "0.00"
        dv_str = f"{dv_ratio_avg:.2f}" if dv_ratio_avg is not None else "0.00"
        log.info(f"[{stock_code}] PE_TTM(均值)={pe_str}, 股息率={dv_str}")
        return pe_ttm_avg, dv_ratio_avg

    def run(self):
        """执行选股主循环"""
        if self.config.is_my:
            stock_df = get_select_stocks()
        else:
            raw_df = get_all_stocks()
            stock_df = ipodatefilter_stocks(raw_df, f"{self.config.start_year}0101")
            
        if stock_df is None or stock_df.empty:
            log.error("未获取到股票列表，程序退出")
            return

        stock_df = stock_df[['代码', '名称']]
        total_rows = len(stock_df)
        chunk_indices = np.array_split(np.arange(total_rows), self.config.chunk_num)
        
        log.info(f"总计 {total_rows} 只股票，分 {self.config.chunk_num} 块处理")
        error_count = 0

        for file_num, chunk_idx in enumerate(chunk_indices):
            chunk_df = stock_df.iloc[chunk_idx]
            results: List[Dict[str, Any]] = []
            log.info(f"开始处理第 {file_num+1} 批数据，共 {len(chunk_df)} 只")
            
            for check_count, (_, row) in enumerate(chunk_df.iterrows(), 1):
                stock_code = row['代码']
                stock_name = row['名称']
                
                try:
                    log.info(f"进度: 批次 {file_num+1}, {check_count}/{len(chunk_df)} -> {stock_code}({stock_name})")
                    
                    var1, var2, var3, var4, var5 = self.check_roe_cash_ebit(stock_code)
                    pe_ttm, dv_ratio = self.check_pe_condition(stock_code, stock_name)
                    
                    results.append({
                        'stock': stock_code,
                        'name': stock_name,
                        'ROE': var1,
                        '现金': var2,
                        '净利': var3,
                        '负债': var4,
                        '回款': var5,
                        'pe_ttm': pe_ttm,
                        'ratio': dv_ratio
                    })
                    
                    error_count = 0  # 成功后重置错误计数
                    time.sleep(2)    # 防封禁强制休眠
                    
                except Exception as e:
                    error_count += 1
                    err_msg = f"处理 {stock_code} 时发生异常: {e}. 连续错误={error_count}"
                    log.error(err_msg)
                    time.sleep(self.config.reconnect_time)
                    
                    if error_count >= self.config.max_consecutive_errors:
                        log.critical(f"连续错误达到阈值 {self.config.max_consecutive_errors}，本批次中止。")
                        break
            
            # 将结果批量存为 Excel
            if results:
                res_df = pd.DataFrame(results)
                
                # 格式化浮点数列输出，避免在计算逻辑中混入字符串
                float_cols = ['ROE', '现金', '净利', '负债', '回款', 'pe_ttm', 'ratio']
                for col in float_cols:
                    res_df[col] = res_df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else None)
                
                output_dir = self.config.base_path / "output"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"select_result_{file_num}.xlsx"
                
                res_df.to_excel(output_file, index=False)
                log.info(f"第 {file_num+1} 批数据已保存至 {output_file}，共 {len(res_df)} 条")
                
        return "所有分块处理完成"

if __name__ == "__main__":
    config = SelectStockConfig()
    selector = StockSelector(config)
    selector.run()