import baostock as bs
import pandas as pd

# 登录测试
lg = bs.login()
print(f"登录状态: {lg.error_code}, 信息: {lg.error_msg}")

# 获取数据测试
rs = bs.query_history_k_data_plus(
    "sh.600000",
    "date,code,open,high,low,close,volume",
    start_date='2024-01-01',
    end_date='2024-12-31',
    frequency="d"
)
df = rs.get_data()
print(df.head())

# 登出
bs.logout()