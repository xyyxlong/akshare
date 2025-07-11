import akshare as ak
import pandas as pd
import insert2Mysql as i2m
from getAllStock import get_all_stocks, get_select_stocks
from datetime import datetime
from tqdm import tqdm
import time
import log4ak
from typing import Dict, List, Optional, Tuple

log = log4ak.LogManager(log_level=log4ak.INFO)# 日志配置

# 分红数据列名映射
DIVIDEND_COLUMNS_MAP = {
    "股票代码": "stock_code",
    "公告日期": "announcement_date",
    "送股": "bonus_share",
    "转增": "additional_shares",
    "派息": "cash_dividend",
    "进度": "progress",
    "除权除息日": "ex_dividend_date",
    "股权登记日": "equity_reg_date",
    "红股上市日": "bonus_listing_date",
}

# 配股数据列名映射（类似上述逻辑）
ALLOTMENT_COLUMNS_MAP = {
    "股票代码": "stock_code",
    "公告日期": "announcement_date",
    "配股方案": "allotment_plan",
    "配股价格": "allotment_price",
    "基准股本": "base_equity",
    "除权日": "ex_rights_date",
    "股权登记日": "equity_reg_date",
    "缴款起始日": "payment_start",
    "缴款终止日": "payment_end",
    "配股上市日": "allotment_listing",
    "募集资金合计": "total_funds"
}

# 创建反向映射（数据库列名 -> 中文列名）
REVERSE_DIVIDEND_MAP = {v: k for k, v in DIVIDEND_COLUMNS_MAP.items()}
REVERSE_ALLOTMENT_MAP = {v: k for k, v in ALLOTMENT_COLUMNS_MAP.items()}

class InsertDividendInfo:
    def __init__(self):
        #akshare接口连续失败调用的上限以及失败次数记录
        self.MAX_TRYTIMES = 3
        self.AK_TRYTIME = 0
        #akshare接口调用失败的休眠时间
        self.AK_TRY_FAILD_SLEEPTIME = 60

        

    def fetch_dividend_data(self,stock_code):
        """
        获取分红信息[7](@ref)
        """
        self.AK_TRYTIME += 1
        try:
            # 获取分红数据
            df = ak.stock_history_dividend_detail(symbol=stock_code, indicator="分红")

            # 添加股票代码列
            df['股票代码'] = stock_code
            return df
        except Exception as e:
                if self.AK_TRYTIME < self.MAX_TRYTIMES:
                    log.error(f"{stock_code}通过akshare获取分红失败{self.AK_TRYTIME} 次，休眠后重试")
                    time.sleep(self.AK_TRY_FAILD_SLEEPTIME)
                    #失败次数没到上限休眠后重新查询
                    return self.fetch_dividend_data(stock_code)
                else:
                    # 备用方法：使用模拟数据
                    log.error(f"{stock_code}通过akshare获取分红失败{self.AK_TRYTIME} 次，不在重试。")
                    log.error(f"错误信息: {str(e)}")
                    return pd.DataFrame()

    def fetch_allotment_data(self,stock_code):
        """
        获取配股信息[7](@ref)
        """
        self.AK_TRYTIME += 1
        try:
            # 获取配股数据
            df = ak.stock_history_dividend_detail(symbol=stock_code, indicator="配股")

            # 添加股票代码列
            df['股票代码'] = stock_code
            return df
        except Exception as e:
                if self.AK_TRYTIME < self.MAX_TRYTIMES:
                    log.error(f"{stock_code}通过akshare获取配股失败{self.AK_TRYTIME} 次，休眠后重试")
                    time.sleep(self.AK_TRY_FAILD_SLEEPTIME)
                    #失败次数没到上限休眠后重新查询
                    return self.fetch_allotment_data(stock_code)
                else:
                    # 备用方法：使用模拟数据
                    log.error(f"{stock_code}通过akshare获取配股失败{self.AK_TRYTIME} 次，不在重试。")
                    log.error(f"错误信息: {str(e)}")
                    return pd.DataFrame()

    def save_to_mysql(self, df, table_name):
        """
        将数据保存到MySQL[8](@ref)
        """
        if df.empty:
            log.error("无数据可保存")
            return

        # 列名映射（根据表类型选择）
        if table_name == "dividend_info":
            column_map = DIVIDEND_COLUMNS_MAP  # 分红列映射
        elif table_name == "allotment_info":
            column_map = ALLOTMENT_COLUMNS_MAP  # 配股列映射
    
        df.rename(columns=column_map, inplace=True)  # 重命名列
    
        try:        
            # 准备插入语句
            columns = ', '.join(df.columns)
            placeholders = ', '.join(['%s'] * len(df.columns))
            sql = f"INSERT IGNORE INTO {table_name} ({columns}) VALUES ({placeholders})"
        
            # 批量插入
            data = [tuple(x) for x in df.to_records(index=False)]
            #cursor.executemany(sql, data)
            #conn.commit()
            rowcount = i2m.insert_batch_insert(data,sql)
        
            log.info(f"成功插入{rowcount}条数据到 {table_name}")
            return rowcount
        except Exception as e:
            log.error(f"数据库插入失败: {e}")
            raise

def insert_selectStock(stock_list: pd.DataFrame) -> None:
    """
    存入自选股票的分红配股信息
    """
    if stock_list is None:
        stock_list = get_select_stocks()#对自选列表进行处理

    idi = InsertDividendInfo()
    for idx, code in tqdm(enumerate(stock_list["代码"]), total=len(stock_list)):
        log.info(f"\n处理股票: {code}")
            
        # 获取分红数据
        idi.AK_TRYTIME = 0
        dividend_df = idi.fetch_dividend_data(code)
        if not dividend_df.empty:
            idi.save_to_mysql(dividend_df, 'dividend_info')
            
        # 间隔避免请求过快
        time.sleep(1)
            
        # 获取配股数据
        idi.AK_TRYTIME = 0
        allotment_df = idi.fetch_allotment_data(code)
        if not allotment_df.empty:
            idi.save_to_mysql(allotment_df, 'allotment_info')
            
        # 间隔避免请求过快
        time.sleep(2)

def get_dividend_data_mysql(stock_code: str) -> pd.DataFrame:
    """查询分红数据（使用游标逐行获取）"""
    sql = """
        SELECT stock_code, announcement_date, bonus_share, additional_shares,
                cash_dividend, progress, ex_dividend_date, equity_reg_date, bonus_listing_date
        FROM dividend_info
        WHERE stock_code = %s
    """
    columns, rows = i2m._execute_query(sql, (stock_code,))
    return convert_to_dataframe(columns, rows, REVERSE_DIVIDEND_MAP)



def get_allotment_data_mysql(stock_code: str) -> pd.DataFrame:
    """查询配股数据（使用游标逐行获取）"""
    sql = """
        SELECT stock_code, announcement_date, allotment_plan, allotment_price,
                base_equity, ex_rights_date, equity_reg_date, payment_start,
                payment_end, allotment_listing, total_funds
        FROM allotment_info
        WHERE stock_code = %s
    """
    columns, rows = i2m._execute_query(sql, (stock_code,))
    return convert_to_dataframe(columns, rows, REVERSE_ALLOTMENT_MAP)

def convert_to_dataframe(columns: List[str], rows: List[Tuple], column_map: Dict[str, str]) -> pd.DataFrame:
    """将查询结果转换为DataFrame并重命名列"""
    if not rows:
        return pd.DataFrame()
        
    # 创建原始DataFrame（使用数据库列名）
    df = pd.DataFrame(rows, columns=columns)
        
    # 映射列名为中文[1](@ref)
    return df.rename(columns=column_map)

if __name__ == "__main__":
   
    stocklist = None

    #stocklist = pd.DataFrame(data=['300146','600183','600596','600598','600618','601336'],columns=['代码'])

    insert_selectStock(stocklist)#对自选列表进行处理
    
    #df = get_dividend_data_mysql('002466')
    #df = get_allotment_data_mysql('000408')

    #print(df)

