import akshare as ak
from datetime import datetime, timedelta
import pandas as pd

# 获取全量历史交易日数据（截止到2025年）

def get_last_trade_dates() -> str:

    last_trade_dates = ""
    trade_dates = ak.tool_trade_date_hist_sina()

    # 转换为日期格式并排序
    trade_dates['trade_date'] = pd.to_datetime(trade_dates['trade_date'])
    today = datetime.now().strftime("%Y%m%d") # 当前日期
    last_trade_date = trade_dates[trade_dates['trade_date'] < today].iloc[-1]['trade_date'].strftime("%Y%m%d")

    return last_trade_date


if __name__ == "__main__":
    print(f"上一个交易日：{get_last_trade_dates()}")  # 输出示例：20250611

