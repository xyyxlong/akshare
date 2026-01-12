import akshare as ak
import pandas as pd
from tqdm import tqdm
import time
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import insert2Mysql as i2m
from getAllStock import get_all_stocks, get_select_stocks
import log4ak

# 日志配置
log = log4ak.LogManager(log_level=log4ak.INFO)# 日志配置


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

class StockHistoricalData:
    def __init__(self):
        #akshare接口连续失败调用的上限以及失败次数记录
        self.MAX_TRYTIMES = 3
        #self.AK_TRYTIME = 0
        #akshare接口调用失败的休眠时间
        self.AK_TRY_FAILD_SLEEPTIME = 600
        self.WAITTIME = 10  # 请求间隔时间，避免过快请求导致被封IP
        

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
                # 获取股票历史数据
                df = ak.stock_zh_a_hist(
                    symbol=stock_code,
                    period=period,
                    start_date=start_date or "19900101",
                    end_date=end_date or datetime.now().strftime("%Y%m%d"),
                    adjust=adjust
                )
                
                # 添加股票代码列
                #df["股票代码"] = stock_code
                return df
                
            except Exception as e:
                log.error(f"获取{stock_code}历史数据失败(尝试{attempt+1}/{self.MAX_TRYTIMES}): {e}")
                if attempt < self.MAX_TRYTIMES - 1:
                    time.sleep(self.AK_TRY_FAILD_SLEEPTIME)
                else:
                    log.error(f"无法获取{stock_code}的历史数据，跳过")
                    return pd.DataFrame()

    def save_to_mysql(self, df: pd.DataFrame, table_name: str = "stock_historical_data") -> None:
        """
        将数据保存到MySQL数据库
        :param df: 包含股票历史数据的DataFrame
        :param table_name: 数据库表名
        """
        if df.empty:
            log.error("无数据可保存")
            return 0
        
        # 重命名列（中文 -> 英文）
        df.rename(columns=COLUMN_MAP, inplace=True)
            
        try:
            # 准备插入语句
            columns = ', '.join(df.columns)
            placeholders = ', '.join(['%s'] * len(df.columns))
            sql = f"""
                INSERT IGNORE INTO {table_name} ({columns})
                VALUES ({placeholders})
            """
                
            # 批量插入数据
            data = [tuple(row) for row in df.itertuples(index=False)]

            rowcount = i2m.insert_batch_insert(data,sql)
                
            log.info(f" 成功插入{rowcount}条数据到 {table_name}")
            return rowcount
        
        except Exception as e:
            log.error(f"{df['stock_code']}数据库插入失败: {e}")
            raise

    def batch_process_stocks(self, stock_codes: List[str], period: str = "daily", 
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
            stock_codes = get_select_stocks()#对自选列表进行处理
            #stock_codes = get_all_stocks()#对全列表进行处理


        total_inserted = 0

        for idx, code in tqdm(enumerate(stock_codes["代码"]), total=len(stock_codes)):
            log.info(f"开始处理股票: {code}")
            
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
                if adjust=="qfq": table_name = 'stock_historical_data_qfq'

                inserted = self.save_to_mysql(df,table_name)
                total_inserted += inserted
                
                # 避免请求过快
                time.sleep(self.WAITTIME)
        
        log.info(f"所有股票处理完成，共插入 {total_inserted} 条记录")

def get_stock_data_from_mysql(stock_code: str, adjust:str = None,start_date: str = None, 
                             end_date: str = None) -> pd.DataFrame:
    """
    从MySQL数据库查询股票历史数据
    :param stock_code: 股票代码
    :param start_date: 开始日期(YYYY-MM-DD)
    :param end_date: 结束日期(YYYY-MM-DD)
    :return: 包含历史数据的DataFrame(列名为中文)
    """
    # 构建查询语句
    table_name = 'stock_historical_data'
    if adjust=="qfq": table_name = 'stock_historical_data_qfq'

    sql = f"""
        SELECT {', '.join(REVERSE_COLUMN_MAP.keys())}
        FROM {table_name}
        WHERE stock_code = %s
    """
    columns, rows = i2m._execute_query(sql, (stock_code,))
    df = convert_to_dataframe(columns, rows, REVERSE_COLUMN_MAP)
    decimal_columns = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额']  
    df[decimal_columns]=df[decimal_columns].astype(float)
    return df

def convert_to_dataframe(columns: List[str], rows: List[Tuple], column_map: Dict[str, str]) -> pd.DataFrame:
    """将查询结果转换为DataFrame并重命名列"""
    if not rows:
        return pd.DataFrame()
        
    # 创建原始DataFrame（使用数据库列名）
    df = pd.DataFrame(rows, columns=columns)
        
    # 映射列名为中文[1](@ref)
    return df.rename(columns=column_map)

if __name__ == "__main__":
    # 示例用法
    processor = StockHistoricalData()

    stock_codes = None
    
    # 股票代码列表（示例）
    #stock_codes = pd.DataFrame(data=['600900'],columns=['代码'])
    # time.sleep(1800)
    
    # 批量处理股票
    processor.batch_process_stocks(
    stock_codes=stock_codes,
    period="daily",
    adjust=""
    )
    
    processor.batch_process_stocks(
        stock_codes=stock_codes,
        period="daily",
        adjust="qfq"
    )


    
    # 从数据库查询示例
    #df = get_stock_data_from_mysql("002466", "","20230101", "20231231")
    #if not df.empty:
    #    print(df.head())