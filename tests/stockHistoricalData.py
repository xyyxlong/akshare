import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import insert2Mysql as i2m
from getAllStock import get_all_stocks, get_select_stocks
import log4ak
import baostock as bs

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
        返回格式: sh.600000 或 sz.000001
        """
        # 清理可能的空格和特殊字符
        code = str(stock_code).strip()
        
        # 移除已有的前缀
        if code.startswith('sh.') or code.startswith('sz.'):
            return code
            
        # 移除可能的后缀
        if code.endswith('.SH'):
            return f"sh.{code[:-3]}"
        elif code.endswith('.SZ'):
            return f"sz.{code[:-3]}"
        
        # 根据代码前缀判断市场
        if code.startswith('6') or code.startswith('9'):
            return f"sh.{code}"
        elif code.startswith('0') or code.startswith('3'):
            return f"sz.{code}"
        else:
            # 默认为上海市场
            return f"sh.{code}"
    
    def get_history_data(self, stock_code: str, start_date: str, end_date: str, 
                        frequency: str = "d", adjustflag: str = "3") -> Optional[pd.DataFrame]:
        """
        获取历史K线数据
        :param stock_code: 股票代码
        :param start_date: 开始日期 (YYYY-MM-DD)
        :param end_date: 结束日期 (YYYY-MM-DD)
        :param frequency: 频率，默认为d(日k线)
        :param adjustflag: 复权类型(1:后复权, 2:前复权, 3:不复权)
        :return: DataFrame
        """
        if not self.login():
            return None
            
        try:
            formatted_code = self.format_stock_code(stock_code)
            
            # 查询历史K线数据
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
                log.warning(f"⚠️ 股票 {stock_code} 在 {start_date} 到 {end_date} 期间无数据")
                return pd.DataFrame()
            
            df = pd.DataFrame(data_list, columns=rs.fields)
            
            # 重命名列以匹配原结构
            column_mapping = {
                'date': '日期',
                'code': '股票代码',
                'open': '开盘',
                'high': '最高',
                'low': '最低',
                'close': '收盘',
                'volume': '成交量',
                'amount': '成交额',
                'turn': '换手率',
                'pctChg': '涨跌幅'
            }
            df = df.rename(columns=column_mapping)
            
            # 清理股票代码，移除前缀
            df['股票代码'] = df['股票代码'].str.replace(r'^(sh|sz)\.', '', regex=True)
            
            # 数据类型转换
            numeric_cols = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '涨跌幅']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 计算其他字段
            if all(col in df.columns for col in ['最高', '最低', '收盘']):
                df['振幅'] = ((df['最高'] - df['最低']) / df['最低'] * 100).round(4)
                df['涨跌额'] = (df['收盘'] - df['收盘'].shift(1)).round(4)
            
            # 确保日期格式
            df['日期'] = pd.to_datetime(df['日期']).dt.strftime('%Y-%m-%d')
            
            return df
            
        except Exception as e:
            log.error(f"❌ 获取 {stock_code} 数据异常: {e}")
            return None

class StockHistoricalData:
    def __init__(self):
        # 接口连续失败调用的上限
        self.MAX_TRYTIMES = 3
        # 接口调用失败的休眠时间
        self.AK_TRY_FAILD_SLEEPTIME = 60
        self.WAITTIME = 5  # 请求间隔时间，避免过快请求
        
        # 初始化 Baostock 客户端
        self.bs_client = BaostockClient()
        
    def fetch_stock_data(self, stock_code: str, period: str = "daily", 
                         adjust: str = "", start_date: str = None, 
                         end_date: str = None) -> pd.DataFrame:
        """
        获取股票历史行情数据
        :param stock_code: 股票代码(不带市场前缀)
        :param period: 周期(daily, weekly, monthly)
        :param adjust: 复权类型(qfq: 前复权, hfq: 后复权, 空: 不复权)
        :param start_date: 开始日期(YYYYMMDD)
        :param end_date: 结束日期(YYYYMMDD)
        :return: 包含历史行情数据的DataFrame
        """
        for attempt in range(self.MAX_TRYTIMES):
            try:
                # 处理日期格式
                if start_date:
                    start_date_formatted = pd.to_datetime(start_date, format='%Y%m%d').strftime('%Y-%m-%d')
                else:
                    start_date_formatted = "1990-12-19"  # 中国股市开始日期
                
                if end_date:
                    end_date_formatted = pd.to_datetime(end_date, format='%Y%m%d').strftime('%Y-%m-%d')
                else:
                    end_date_formatted = datetime.now().strftime('%Y-%m-%d')
                
                # 处理复权类型
                adjust_map = {
                    "": "3",      # 不复权
                    "qfq": "2",   # 前复权
                    "hfq": "1"    # 后复权
                }
                adjustflag = adjust_map.get(adjust, "3")
                
                # 处理频率
                freq_map = {
                    "daily": "d",
                    "weekly": "w",
                    "monthly": "m"
                }
                frequency = freq_map.get(period, "d")
                
                # 使用 Baostock 获取数据
                df = self.bs_client.get_history_data(
                    stock_code=stock_code,
                    start_date=start_date_formatted,
                    end_date=end_date_formatted,
                    frequency=frequency,
                    adjustflag=adjustflag
                )
                
                if df is not None and not df.empty:
                    log.info(f"✅ 成功获取 {stock_code} 的 {len(df)} 条数据")
                    return df
                else:
                    log.warning(f"⚠️ 股票 {stock_code} 返回数据为空")
                    return pd.DataFrame()
                    
            except Exception as e:
                log.error(f"❌ 获取{stock_code}历史数据失败(尝试{attempt+1}/{self.MAX_TRYTIMES}): {e}")
                
                if attempt < self.MAX_TRYTIMES - 1:
                    try:
                        sleep_time = random.uniform(1, self.AK_TRY_FAILD_SLEEPTIME)
                        log.info(f"等待 {sleep_time:.1f} 秒后重试...")
                        time.sleep(sleep_time)
                    except KeyboardInterrupt:
                        log.error("程序被中断，返回已获取的数据")
                        return pd.DataFrame()
                else:
                    log.error(f"❌ 无法获取{stock_code}的历史数据，跳过")
                    return pd.DataFrame()
        
        return pd.DataFrame()

    def save_to_mysql(self, df: pd.DataFrame, table_name: str = "stock_historical_data") -> int:
        """
        将数据保存到MySQL数据库
        :param df: 包含股票历史数据的DataFrame
        :param table_name: 数据库表名
        :return: 插入的行数
        """
        if df.empty:
            log.warning("⚠️ 无数据可保存")
            return 0
        
        # 重命名列（中文 -> 英文）
        df.rename(columns=COLUMN_MAP, inplace=True)
        
        # 确保列顺序一致
        required_columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount', 
                        'amplitude', 'change_percent', 'change_amount', 'turnover_rate', 'stock_code']
        
        # 添加缺失的列
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        
        # 按顺序选择列
        df = df[required_columns]
        
        # 将 NaN 值替换为 None
        df = df.where(pd.notnull(df), None)
        
        # 调试：打印前几行数据查看
        log.debug(f"📊 保存前数据预览（前2行）:")
        for i, row in df.head(2).iterrows():
            log.debug(f"  第{i+1}行: {row.to_dict()}")
        
        try:
            # 准备插入语句
            columns = ', '.join(df.columns)
            placeholders = ', '.join(['%s'] * len(df.columns))
            sql = f"""
                INSERT IGNORE INTO {table_name} ({columns})
                VALUES ({placeholders})
            """
            
            # 批量插入数据
            data = []
            for row in df.itertuples(index=False, name=None):
                # 确保每个值都被正确处理
                processed_row = []
                for value in row:
                    if pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
                        processed_row.append(None)
                    else:
                        processed_row.append(value)
                data.append(tuple(processed_row))
            
            rowcount = i2m.insert_batch_insert(data, sql)
            
            log.info(f"✅ 成功插入 {rowcount} 条数据到 {table_name}")
            return rowcount
        
        except Exception as e:
            log.error(f"❌ 数据库插入失败: {e}")
            # 调试：打印一条样本数据
            if not df.empty:
                sample_row = df.iloc[0]
                log.error(f"📊 样本数据: {sample_row.to_dict()}")
            return 0

    def batch_process_stocks(self, stock_codes: List[str] = None, period: str = "daily", 
                            adjust: str = "", start_date: str = None, 
                            end_date: str = None):
        """
        批量处理多只股票
        :param stock_codes: 股票代码列表
        :param period: 数据周期
        :param adjust: 复权类型
        :param start_date: 开始日期
        :param end_date: 结束日期
        """
        if stock_codes is None:
            select_df = get_select_stocks()  # 对自选列表进行处理
            if select_df is None or select_df.empty:
                log.error("❌ 无法获取股票列表")
                return
            stock_codes_list = select_df['代码'].tolist()
        else:
            stock_codes_list = stock_codes

        total_inserted = 0
        processed_count = 0
        failed_stocks = []

        for idx, code in tqdm(enumerate(stock_codes_list), total=len(stock_codes_list), desc="处理股票进度"):
            log.info(f"📈 开始处理股票: {code} ({idx+1}/{len(stock_codes_list)})")
            
            # 获取数据
            df = self.fetch_stock_data(
                stock_code=code,
                period=period,
                adjust=adjust,
                start_date=start_date,
                end_date=end_date
            )
            
            if not df.empty:
                # 保存到数据库
                table_name = 'stock_historical_data'
                if adjust == "qfq":
                    table_name = 'stock_historical_data_qfq'
                
                inserted = self.save_to_mysql(df, table_name)
                total_inserted += inserted
                processed_count += 1
                
                # 避免请求过快
                try:
                    wait_time = random.uniform(1, self.WAITTIME)
                    time.sleep(wait_time)
                except KeyboardInterrupt:
                    log.error("⚠️ 程序被用户中断")
                    break
            else:
                failed_stocks.append(code)
        
        # 输出统计信息
        log.info("=" * 50)
        log.info(f"✅ 处理完成统计:")
        log.info(f"   总股票数: {len(stock_codes_list)}")
        log.info(f"   成功处理: {processed_count}")
        log.info(f"   失败股票: {len(failed_stocks)}")
        log.info(f"   总插入记录: {total_inserted} 条")
        
        if failed_stocks:
            log.info(f"   失败的股票代码: {', '.join(failed_stocks[:20])}{'...' if len(failed_stocks) > 20 else ''}")
        
        # 登出 Baostock
        self.bs_client.logout()
        
        return total_inserted

def get_stock_data_from_mysql(stock_code: str, adjust: str = None, 
                              start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    从MySQL数据库查询股票历史数据
    :param stock_code: 股票代码
    :param adjust: 复权类型
    :param start_date: 开始日期(YYYY-MM-DD)
    :param end_date: 结束日期(YYYY-MM-DD)
    :return: 包含历史数据的DataFrame(列名为中文)
    """
    # 构建查询语句
    table_name = 'stock_historical_data'
    if adjust == "qfq":
        table_name = 'stock_historical_data_qfq'
    
    sql = f"""
        SELECT {', '.join(REVERSE_COLUMN_MAP.keys())}
        FROM {table_name}
        WHERE stock_code = %s
    """
    
    params = [stock_code]
    
    if start_date:
        sql += " AND date >= %s"
        params.append(start_date)
    
    if end_date:
        sql += " AND date <= %s"
        params.append(end_date)
    
    sql += " ORDER BY date"
    
    columns, rows = i2m._execute_query(sql, tuple(params))
    df = convert_to_dataframe(columns, rows, REVERSE_COLUMN_MAP)
    
    # 转换数据类型
    if not df.empty:
        decimal_columns = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额']
        for col in decimal_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def convert_to_dataframe(columns: List[str], rows: List[Tuple], column_map: Dict[str, str]) -> pd.DataFrame:
    """将查询结果转换为DataFrame并重命名列"""
    if not rows:
        return pd.DataFrame()
        
    # 创建原始DataFrame（使用数据库列名）
    df = pd.DataFrame(rows, columns=columns)
        
    # 映射列名为中文
    return df.rename(columns=column_map)

def test_baostock_connection():
    """测试 Baostock 连接"""
    log.info("🔧 测试 Baostock 连接...")
    
    try:
        # 登录测试
        lg = bs.login()
        if lg.error_code == '0':
            log.info("✅ Baostock 连接测试成功")
            
            # 获取一只股票数据测试
            rs = bs.query_history_k_data_plus(
                "sh.600000",
                "date,code,open,high,low,close,volume,amount",
                start_date="2024-01-01",
                end_date="2024-01-10",
                frequency="d"
            )
            
            if rs.error_code == '0':
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())
                
                if data_list:
                    log.info(f"✅ 数据获取测试成功，获取到 {len(data_list)} 条记录")
                else:
                    log.warning("⚠️ 数据获取测试返回空数据")
            else:
                log.error(f"❌ 数据获取测试失败: {rs.error_msg}")
            
            bs.logout()
            return True
        else:
            log.error(f"❌ Baostock 连接测试失败: {lg.error_msg}")
            return False
            
    except Exception as e:
        log.error(f"❌ Baostock 连接测试异常: {e}")
        return False

if __name__ == "__main__":
    # 测试连接
    if test_baostock_connection():
        log.info("🚀 开始执行主程序...")
        
        # 创建处理器
        processor = StockHistoricalData()
        
        # 可以指定要处理的股票代码列表
        stock_codes = ['000333', '000858']
        
        # 或者使用 None 来自动获取自选股列表
        # stock_codes = None
        
        # 批量处理股票（不复权数据）
        processor.batch_process_stocks(
            stock_codes=stock_codes,
            period="daily",
            adjust="",  # 不复权
            start_date="20260101",  # 可选，指定开始日期
            end_date=None  # 可选，指定结束日期
        )
        
        # 如果需要前复权数据，取消注释下面的代码
        # processor.batch_process_stocks(
        #     stock_codes=stock_codes,
        #     period="daily",
        #     adjust="qfq", #前复权
        #     start_date="20260101",  # 可选，指定开始日期
        #     end_date=None  # 可选，指定结束日期
        # )
        
        # 从数据库查询示例
        # df = get_stock_data_from_mysql("000001", "", "2024-01-01", "2024-12-31")
        # if not df.empty:
        #     print(f"获取到 {len(df)} 条记录")
        #     print(df.head())
    else:
        log.error("❌ Baostock 连接测试失败，程序退出")