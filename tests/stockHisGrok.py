import random
import baostock as bs
import pandas as pd
from tqdm import tqdm
import time
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import insert2Mysql as i2m
from getAllStock import get_all_stocks, get_select_stocks
import log4ak
import numpy as np 
import atexit 

# 日志配置
log = log4ak.LogManager(log_level=log4ak.INFO)

# 列名映射（Baostock 返回的英文列名 -> 数据库列名）
# Baostock 常用字段：date, code, open, high, low, close, preclose, volume, amount, adjustflag, turn, pctChg 等
COLUMN_MAP = {
    "date": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "amount": "amount",
    "turn": "turnover_rate",      # 换手率
    "pctChg": "change_percent",   # 涨跌幅
    # 可根据需要补充其他字段（如 amplitude、change_amount 等，可后续计算）
    "code": "stock_code"
}

# 创建反向映射（数据库列名 -> 中文列名，用于查询时显示）
REVERSE_COLUMN_MAP = {v: k for k, v in COLUMN_MAP.items()}

class StockHistoricalDataGroker:
    def __init__(self):
        self.MAX_TRYTIMES = 3
        self.BS_TRY_FAILD_SLEEPTIME = 10
        self.WAITTIME = 10  # 请求间隔，避免被限流

        # 登录 Baostock（只需一次）
        lg = bs.login()
        if lg.error_code != '0':
            log.error(f"Baostock 登录失败: {lg.error_msg}")
        else:
            log.info("Baostock 登录成功")

    def _get_baostock_code(self, stock_code: str) -> str:
        """将纯6位代码转换为 Baostock 格式 (sh./sz./bj.)"""
        code = str(stock_code).zfill(6)
        if code.startswith(('600', '601', '603', '688', '689')):
            return f"sh.{code}"
        elif code.startswith(('000', '001', '002', '003', '300', '301')):
            return f"sz.{code}"
        elif code.startswith(('430', '831', '833', '870')):  # 北交所示例
            return f"bj.{code}"
        else:
            return f"sh.{code}"  # 默认沪市，实际可根据需要调整

    def fetch_stock_data(self, stock_code: str, period: str = "daily",
                         adjust: str = "", start_date: str = None,
                         end_date: str = None) -> pd.DataFrame:
        """
        使用 Baostock 获取股票历史行情数据
        """
        bs_code = self._get_baostock_code(stock_code)
        
        
        # 处理日期格式
        if start_date:
            start_date_formatted = pd.to_datetime(start_date, format='%Y%m%d').strftime('%Y-%m-%d')
        else:
            start_date_formatted = "1990-12-19"  # 中国股市开始日期
        
        if end_date:
            end_date_formatted = pd.to_datetime(end_date, format='%Y%m%d').strftime('%Y-%m-%d')
        else:
            end_date_formatted = datetime.now().strftime('%Y-%m-%d')

        # 频率映射
        frequency_map = {"daily": "d", "weekly": "w", "monthly": "m"}
        frequency = frequency_map.get(period, "d")

        # 复权映射
        adjustflag = "3"  # 默认不复权
        if adjust == "qfq":
            adjustflag = "2"  # 前复权
        elif adjust == "hfq":
            adjustflag = "1"  # 后复权

        fields = "date,code,open,high,low,close,volume,amount,turn,pctChg"

        for attempt in range(self.MAX_TRYTIMES):
            try:
                rs = bs.query_history_k_data_plus(
                    code=bs_code,
                    fields=fields,
                    start_date=start_date_formatted,
                    end_date=end_date_formatted,
                    frequency=frequency,
                    adjustflag=adjustflag
                )

                if rs.error_code != '0':
                    raise Exception(f"Baostock 查询错误: {rs.error_msg}")

                df = rs.get_data()  # 直接转为 DataFrame

                if df.empty:
                    log.warning(f"{stock_code} 返回空数据")
                    return pd.DataFrame()

                # 重命名列以兼容原有逻辑
                df.rename(columns=COLUMN_MAP, inplace=True)

                # 添加/确保 stock_code 列（使用原纯代码）
                df["stock_code"] = stock_code

                # === 关键修复：类型转换 + NaN 处理 ===
                numeric_cols = ["open", "high", "low", "close", "volume", "amount",
                                "turnover_rate", "change_percent"]

                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                # 计算振幅 (high - low)
                df["amplitude"] = (df["high"] - df["low"]).round(4)

                # 计算涨跌额 (close - preclose)，但 Baostock 默认不返回 preclose
                # 如果你需要，可在 fields 中增加 ",preclose"，然后再计算
                # 这里先用 0 或 None 占位，避免 nan 导致插入失败
                df["change_amount"] = None   # 或者 df["close"] - df.get("preclose", df["close"])

                # 把所有 NaN 替换为 None（最重要！防止 insert2Mysql 报 np 错误）
                df = df.replace({np.nan: None})

                log.info(f"✅ 获取成功 {stock_code} ({bs_code}) → {len(df)} 条记录")
                return df

            except Exception as e:
                log.error(f"获取{stock_code}历史数据失败(尝试{attempt+1}/{self.MAX_TRYTIMES}): {e}")
                if attempt < self.MAX_TRYTIMES - 1:
                    sleep_time = random.uniform(1, self.BS_TRY_FAILD_SLEEPTIME)
                    log.info(f"休眠 {sleep_time:.0f} 秒后重试...")
                    time.sleep(sleep_time)
                else:
                    log.error(f"无法获取{stock_code}的历史数据，跳过")
                    return pd.DataFrame()

    def save_to_mysql(self, df: pd.DataFrame, table_name: str = "stock_historical_data") -> int:
        """保存到 MySQL（逻辑与原版一致）"""
        if df.empty:
            log.error("无数据可保存")
            return 0

        try:
            columns = ', '.join(df.columns)
            placeholders = ', '.join(['%s'] * len(df.columns))
            sql = f"""
                INSERT IGNORE INTO {table_name} ({columns})
                VALUES ({placeholders})
            """

            data = [tuple(row) for row in df.itertuples(index=False)]
            rowcount = i2m.insert_batch_insert(data, sql)

            log.info(f"成功插入 {rowcount} 条数据到 {table_name}")
            return rowcount

        except Exception as e:
            log.error(f"数据库插入失败: {e}")
            raise

    def batch_process_stocks(self, stock_codes: List[str] = None, period: str = "daily",
                             adjust: str = "", start_date: str = None,
                             end_date: str = None):
        """批量处理多只股票"""
        if stock_codes is None:
            stock_codes = get_select_stocks()  # 或 get_all_stocks()

        if isinstance(stock_codes, pd.DataFrame):
            stock_list = stock_codes["代码"].tolist()
        else:
            stock_list = stock_codes

        total_inserted = 0
        for idx, code in tqdm(enumerate(stock_list), total=len(stock_list)):
            log.info(f"开始处理股票: {code}")

            df = self.fetch_stock_data(
                stock_code=code,
                period=period,
                adjust=adjust,
                start_date=start_date,
                end_date=end_date
            )

            if not df.empty:
                table_name = 'stock_historical_data'
                if adjust == "qfq":
                    table_name = 'stock_historical_data_qfq'

                inserted = self.save_to_mysql(df, table_name)
                total_inserted += inserted

            # 避免请求过快
            try:
                time.sleep(random.uniform(1, self.WAITTIME))
            except KeyboardInterrupt:
                log.error("程序被中断，返回已获取的数据")
                break

        log.info(f"所有股票处理完成，共插入 {total_inserted} 条记录")
        # 注册退出时自动登出（推荐方式）
        atexit.register(self._logout)

    def _logout(self):
        """安全登出 Baostock"""
        try:
            bs.logout()
            log.info("✅ Baostock 已正常登出")
        except Exception as e:
            # 防止登出本身出错时再抛异常
            log.debug(f"Baostock 登出时出现小问题（可忽略）: {e}")


# 如果需要独立的查询函数，可保留或稍作调整
def get_stock_data_from_mysql(stock_code: str, adjust: str = None, start_date: str = None,
                              end_date: str = None) -> pd.DataFrame:
    table_name = 'stock_historical_data'
    if adjust == "qfq":
        table_name = 'stock_historical_data_qfq'

    sql = f"""
        SELECT {', '.join(REVERSE_COLUMN_MAP.keys())}
        FROM {table_name}
        WHERE stock_code = %s
    """
    if start_date:
        sql += " AND date >= %s"
    if end_date:
        sql += " AND date <= %s"

    params = (stock_code,)
    if start_date and end_date:
        params = (stock_code, start_date, end_date)
    elif start_date:
        params = (stock_code, start_date)
    elif end_date:
        params = (stock_code, end_date)

    columns, rows = i2m._execute_query(sql, params)
    df = convert_to_dataframe(columns, rows, REVERSE_COLUMN_MAP)

    # 类型转换
    decimal_columns = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额']
    for col in decimal_columns:
        if col in df.columns:
            df[col] = df[col].astype(float)

    return df


def convert_to_dataframe(columns: List[str], rows: List[Tuple], column_map: Dict[str, str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=columns)
    return df.rename(columns=column_map)


if __name__ == "__main__":
    processor = StockHistoricalDataGroker()
    
    # 可以指定要处理的股票代码列表
    stock_codes = ['000333', '000858']
    # 使用 get_select_stocks() 获取自选股票列表
    # stock_codes = None 
    
    start_date= "20260101"
    end_date= "20260416"

    # 示例：处理自选股票（不复权）
    processor.batch_process_stocks(
        stock_codes=stock_codes,   # None 表示使用 get_select_stocks()
        period="daily",
        adjust="",        # ""=不复权, "qfq"=前复权
        start_date=start_date,
        end_date=end_date
    )

    # 处理前复权示例
    # processor.batch_process_stocks(..., adjust="qfq")

    # 查询示例
    # df = get_stock_data_from_mysql("002466", adjust="qfq", start_date="2023-01-01", end_date="2023-12-31")
    # print(df.head() if not df.empty else "无数据")