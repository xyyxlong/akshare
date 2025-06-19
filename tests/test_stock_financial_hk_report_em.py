import akshare as ak
import log4ak

# 第一次初始化


log1 = log4ak.LogManager(log_level=log4ak.MYLOGLEVEL)
log1.error("错误信息")  # 输出到info.error和console
log1.info("普通信息")   # 不会输出（因为级别是INFO）

# 后续调用（返回同一个实例）
log2 = log4ak.LogManager()
print(log1 is log2)  # 输出: True

# 日志记录
log1.info("普通信息")  # 输出到info.error和console

# 获取腾讯控股2023年利润表
#df_income = ak.stock_financial_hk_report_em(stock="00241", symbol="利润表",indicator="年度")

#df_income = df_income[(df_income["STD_ITEM_NAME"]=="经营收入总额")|(df_income["STD_ITEM_NAME"]=="经营溢利")]
#df_income = df_income[df_income["REPORT_DATE"]=="2024-12-31 00:00:00"]
#df_income = df_income[df_income["FISCAL_YEAR"]=="1231"]
# 定义可能存在的revenue列名集合
#revenue_columns = ["营业额", "经营收入总额", "营业收入"]
#revenue_NAME="营业额"
# 动态检测有效列
#for col in revenue_columns:
#        if  not df_income[df_income["STD_ITEM_NAME"]==col].empty:
#            revenue_NAME=col
#            break
#print(revenue_NAME)
#print(df_income["STD_ITEM_NAME"]=="营业额")
#print(df_income[["SECURITY_CODE","REPORT_DATE","STD_ITEM_NAME","AMOUNT"]])
#print(df_income[["SECURITY_CODE","SECURITY_NAME_ABBR","STD_ITEM_NAME","REPORT_DATE"]])
#输出为：    SECUCODE          REPORT_DATE  ...   STD_ITEM_NAME        AMOUNT
# 0   00700.HK  2024-12-31 00:00:00  ...             营业额  6.524980e+11
# 25  00700.HK  2024-12-31 00:00:00  ...  本公司拥有人应占全面收益总额  2.790090e+11

