import random
import time
from typing import List, Dict, Tuple, Optional
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm
import baostock as bs

import insert2Mysql as i2m
from getAllStock import get_select_stocks
import log4ak

# 日志配置
log = log4ak.LogManager(log_level=log4ak.INFO)

# 列名映射（中文列名 -> 数据库列名）
COLUMN_MAP = {
    "日期": "date",
    "开盘": "open", 
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude", 
    "涨跌幅": "change_percent",
    "涨跌额": "change_amount",
    "换手率": "turnover_rate",
    "股票代码": "stock_code"
}

# 创建反向映射（数据库列名 -> 中文列名）
REVERSE_COLUMN_MAP = {v: k for k, v in COLUMN_MAP.items()}

class BaostockClient:
    """Baostock 客户端封装"""
    
    def __init__(self):
        self.is_logged_in = False
        
    def login(self):
        """登录 Baostock"""
        if not self.is_logged_in:
            try:
                lg = bs.login()
                if lg.error_code == '0':
                    self.is_logged_in = True
                    log.info("✅ Baostock 登录成功")
                else:
                    log.error(f"❌ Baostock 登录失败: {lg.error_msg}")
            except Exception as e:
                log.error(f"❌ Baostock 登录异常: {e}")
        return self.is_logged_in
    
    def logout(self):
        """登出 Baostock"""
        if self.is_logged_in:
            bs.logout()
            self.is_logged_in = False
            log.info("✅ Baostock 已登出")
    
    def format_stock_code(self, stock_code: str) -> str:
        """
        格式化股票代码，添加市场前缀
        """
        code = str(stock_code).strip()
        if code.startswith('sh.') or code.startswith('sz.'):
            return code
        if code.endswith('.SH'):
            return f"sh.{code[:-3]}"
        elif code.endswith('.SZ'):
            return f"sz.{code[:-3]}"
        
        if code.startswith('6') or code.startswith('9'):
            return f"sh.{code}"
        elif code.startswith('0') or code.startswith('3'):
            return f"sz.{code}"
        else:
            return f"sh.{code}"
    
    def get_history_data(self, stock_code: str, start_date: str, end_date: str, 
                        frequency: str = "d", adjustflag: str = "3") -> Optional[pd.DataFrame]:
        """获取历史K线数据"""
        if not self.login():
            return None
            
        try:
            formatted_code = self.format_stock_code(stock_code)
            fields = "date,code,open,high,low,close,volume,amount,turn,pctChg"
            
            rs = bs.query_history_k_data_plus(
                code=formatted_code,
                fields=fields,
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
                adjustflag=adjustflag
            )
            
            if rs.error_code != '0':
                log.error(f"❌ 查询 {stock_code} 数据失败: {rs.error_msg}")
                return None
            
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            if not data_list:
                return pd.DataFrame()
            
            df = pd.DataFrame(data_list, columns=rs.fields)
            
            column_mapping = {
                'date': '日期', 'code': '股票代码', 'open': '开盘', 'high': '最高',
                'low': '最低', 'close': '收盘', 'volume': '成交量', 'amount': '成交额',
                'turn': '换手率', 'pctChg': '涨跌幅'
            }
            df = df.rename(columns=column_mapping)
            df['股票代码'] = df['股票代码'].str.replace(r'^(sh|sz)\.', '', regex=True)
            
            # Vectorized type casting
            numeric_cols = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '涨跌幅', '换手率']
            for col in numeric_cols:
                if col in df.columns:
                    # Treat empty strings as NaN
                    df[col] = pd.to_numeric(df[col].replace('', np.nan), errors='coerce')
            
            # Safely calculate amplitude and change amount
            if all(col in df.columns for col in ['最高', '最低', '收盘']):
                # Protect against Division by Zero where '最低' is 0 or NaN
                df['振幅'] = np.where(
                    df['最低'] > 0, 
                    ((df['最高'] - df['最低']) / df['最低'] * 100).round(4), 
                    None
                )
                df['涨跌额'] = (df['收盘'] - df['收盘'].shift(1)).round(4)
            
            df['日期'] = pd.to_datetime(df['日期']).dt.strftime('%Y-%m-%d')
            
            return df
            
        except Exception as e:
            log.error(f"❌ 获取 {stock_code} 数据异常: {e}")
            return None

class StockHistoricalData:
    def __init__(self):
        self.MAX_TRYTIMES = 3
        self.AK_TRY_FAILD_SLEEPTIME = 60
        self.WAITTIME = 1
        self.bs_client = BaostockClient()
        
    def fetch_stock_data(self, stock_code: str, period: str = "daily", 
                         adjust: str = "", start_date: str = None, 
                         end_date: str = None) -> pd.DataFrame:
        """带重试机制的数据获取"""
        for attempt in range(self.MAX_TRYTIMES):
            try:
                start_date_formatted = pd.to_datetime(start_date, format='%Y%m%d').strftime('%Y-%m-%d') if start_date else "1990-12-19"
                end_date_formatted = pd.to_datetime(end_date, format='%Y%m%d').strftime('%Y-%m-%d') if end_date else datetime.now().strftime('%Y-%m-%d')
                
                adjust_map = {"": "3", "qfq": "2", "hfq": "1"}
                adjustflag = adjust_map.get(adjust, "3")
                
                freq_map = {"daily": "d", "weekly": "w", "monthly": "m"}
                frequency = freq_map.get(period, "d")
                
                df = self.bs_client.get_history_data(
                    stock_code=stock_code,
                    start_date=start_date_formatted,
                    end_date=end_date_formatted,
                    frequency=frequency,
                    adjustflag=adjustflag
                )
                
                if df is not None:
                    return df
                    
            except Exception as e:
                log.error(f"❌ 获取{stock_code}({adjust})数据失败(尝试{attempt+1}/{self.MAX_TRYTIMES}): {e}")
                if attempt < self.MAX_TRYTIMES - 1:
                    sleep_time = random.uniform(1, self.AK_TRY_FAILD_SLEEPTIME)
                    time.sleep(sleep_time)
                else:
                    return pd.DataFrame()
        return pd.DataFrame()

    def save_to_mysql(self, df: pd.DataFrame, table_name: str = "stock_historical_data") -> int:
        """将数据保存到MySQL数据库 (Optimized Vectorized version)"""
        if df.empty:
            return 0
        
        # 建立副本防止警告
        df = df.copy()
        df.rename(columns=COLUMN_MAP, inplace=True)
        
        required_columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount', 
                        'amplitude', 'change_percent', 'change_amount', 'turnover_rate', 'stock_code']
        
        # Add missing columns
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        
        df = df[required_columns]
        
        # Replace NaN/NaT with None for MySQL compatibility using vectorized where
        df = df.astype(object).where(pd.notnull(df), None)
        
        try:
            columns = ', '.join(df.columns)
            placeholders = ', '.join(['%s'] * len(df.columns))
            sql = f"""
                INSERT IGNORE INTO {table_name} ({columns})
                VALUES ({placeholders})
            """
            
            # Efficiently convert dataframe to list of tuples
            data = [tuple(x) for x in df.to_numpy()]
            rowcount = i2m.insert_batch_insert(data, sql)
            return rowcount
        
        except Exception as e:
            log.error(f"❌ 数据库插入失败: {e}")
            return 0

    def batch_process_stocks(self, stock_codes: List[str] = None, period: str = "daily", 
                            start_date: str = None, end_date: str = None,
                            fetch_qfq: bool = True):
        """
        批量处理多只股票
        Optimized: Fetches both normal and qfq data in a single stock loop to reduce context switching and duplicate loop overhead.
        """
        if stock_codes is None:
            select_df = get_select_stocks()
            if select_df is None or select_df.empty:
                log.error("❌ 无法获取股票列表")
                return
            stock_codes_list = select_df['代码'].tolist()
        else:
            stock_codes_list = stock_codes

        total_inserted_normal = 0
        total_inserted_qfq = 0
        failed_stocks = []

        for idx, code in tqdm(enumerate(stock_codes_list), total=len(stock_codes_list), desc="处理股票进度"):
            log.info(f"📈 开始处理股票: {code} ({idx+1}/{len(stock_codes_list)})")
            
            success_normal = False
            
            # 1. Fetch Normal Data
            df_normal = self.fetch_stock_data(code, period, "", start_date, end_date)
            if not df_normal.empty:
                inserted = self.save_to_mysql(df_normal, 'stock_historical_data')
                total_inserted_normal += inserted
                success_normal = True
            
            # 2. Fetch QFQ Data (If requested)
            if fetch_qfq:
                df_qfq = self.fetch_stock_data(code, period, "qfq", start_date, end_date)
                if not df_qfq.empty:
                    inserted = self.save_to_mysql(df_qfq, 'stock_historical_data_qfq')
                    total_inserted_qfq += inserted
                elif success_normal:
                     # Failed to get QFQ but got normal, still count as a partial failure
                     failed_stocks.append(f"{code}(QFQ)")
                     
            if df_normal.empty and (not fetch_qfq or df_qfq.empty):
                 failed_stocks.append(code)

            # Polite sleep to avoid Baostock rate limits
            try:
                time.sleep(random.uniform(1, self.WAITTIME))
            except KeyboardInterrupt:
                log.error("⚠️ 程序被用户中断")
                break
        
        log.info("=" * 50)
        log.info(f"✅ 处理完成统计:")
        log.info(f"   总股票数: {len(stock_codes_list)}")
        log.info(f"   不复权插入记录: {total_inserted_normal} 条")
        if fetch_qfq:
            log.info(f"   前复权插入记录: {total_inserted_qfq} 条")
        
        if failed_stocks:
            log.info(f"   存在获取失败的股票: {', '.join(failed_stocks[:20])}")
        
        self.bs_client.logout()
        return total_inserted_normal + total_inserted_qfq

def test_baostock_connection():
    log.info("🔧 测试 Baostock 连接...")
    client = BaostockClient()
    if client.login():
        client.logout()
        return True
    return False

if __name__ == "__main__":
    if test_baostock_connection():
        log.info("🚀 开始执行主程序...")
        processor = StockHistoricalData()
        
        # 优化后：在一次遍历中同时拉取并存储不复权和前复权数据
        processor.batch_process_stocks(
            stock_codes=None,
            period="daily",
            start_date="20260501",
            end_date=None,
            fetch_qfq=True
        )
    else:
        log.error("❌ Baostock 连接测试失败，程序退出")