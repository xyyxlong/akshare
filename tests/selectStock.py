import time
import numpy as np
import akshare as ak
import pandas as pd
import datetime
import log4ak
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from getAllStock import get_all_stocks, get_select_stocks, ipodatefilter_stocks
import insertSelectStockPE as issp
import insert2Mysql as ins
import get_stockPE_his as gsh

##运行配置：
log = log4ak.LogManager(log_level=log4ak.INFO)# 日志配置
MAX_CONSECUTIVE_ERRORS = 3  # 最大允许连续错误次数
OUTTIME = 5  # 接口长时间无返回报错
RECONNECT_TIME = 30 #断线重连休眠时间
CHUNK_NUM = 30# 全市场数据过多分10块处理
ISMY = False #是否选取自选配置False/True
IS_MYSQL = False  #PE数据来源，使用数据库速度快很多：数据库/Akshare  True/False

##选股参数设置：
STARTYEAR = "2019"  #计算的起始年份
ROE = 15 #过去几年来平均净资产收益率高于15%
PEMAX = 25 #过去几天平均市盈率低于25且大于0
PASTDAY = 30 #过去30天
PASTYEAR = 5 #计算过去5年的净资产收益率，负债率，应收账款周期
DEBT_RATIOS = 70 #负债率低于70%     风险
RECEIVABLE_DAYS = 30  #应收账款周期小于30  行业地位
CASH2PROFIT = 1.25 #经营性现金流/净利润比例>1.1  保证赚的是真钱


def selectStock():
    ## A 股上市公司列表

    if ISMY:
        stock_zh_a_spot_df = get_select_stocks()#对自选列表进行处理
    else:
        df = get_all_stocks()#对全市场数据进行处理
        stock_zh_a_spot_df = ipodatefilter_stocks(df,f"{STARTYEAR}0101") #对上市时间进行筛选
 
    log.info(f"获取到 A 股上市公司列表，是否只选取自选股：{ISMY}")
    df_stock = stock_zh_a_spot_df[['代码','名称']]#[339:]

    # 分块处理设置[2,3](@ref)
    total_rows = len(df_stock)
    chunk_num = CHUNK_NUM
    chunk_indices = np.array_split(np.arange(total_rows), chunk_num)
    log.info(f"分块处理设置总记录数total_rows={total_rows}；块数chunk_num={chunk_num}，每块记录数chunk_indices={len(chunk_indices[0])}")

    # 初始化错误计数器（放在循环体外层）
    error_count = 0  # 连续错误计数器    

    # 分批处理逻辑
    for file_num, chunk_idx in enumerate(chunk_indices):
        
        chunk_df = df_stock.iloc[chunk_idx]
        df_result = pd.DataFrame(columns=['stock','name','指标1-ROI','指标2-现金','指标3-净利','指标4-负债','指标5-回款','指标6-PE','ratio','综合评估'])
        log.info(f"开始处理第{file_num+1}批数据，包含{len(chunk_df)}条记录")
        checkcount = 0
        
        # 处理单个数据块
        for row_index, row in tqdm(chunk_df.iterrows(), total=len(chunk_df), desc=f"处理第{file_num+1}批\n"):
            try:
                r_code = row['代码']
                r_name = row['名称']
                checkcount += 1
                log.info(f"处理第{file_num+1}批第{checkcount}条记录：{r_code}")

                # 指标计算
                var1, var2, var3, var4, var5 = checkRoeCashEBIT(r_code, STARTYEAR)
                ratio , var6 = check_pe_condition(r_code, r_name)
                log.info(f"第{file_num+1}批第{checkcount}条记录处理结果var6={var6}")

                varAll = var1 and var2 and var3 and var4 and var5 and var6
                log.info(f"第{file_num+1}批第{checkcount}条记录处理结果varAll={varAll}")
                
                # 结果存储
                df_result.loc[row_index] = {
                    'stock': r_code,
                    'name': r_name,
                    '指标1-ROI': var1,
                    '指标2-现金': var2,
                    '指标3-净利': var3,
                    '指标4-负债': var4,
                    '指标5-回款': var5,
                    '指标6-PE': var6,
                    'ratio': ratio,
                    '综合评估': varAll
                }
                error_count = 0  # 成功执行后重置计数器[6](@ref)
                log.info(f"功执行后重置计数器error_count={error_count}")
                time.sleep(2)
            except AttributeError as e:
                error_count += 1  # 捕获特定异常时计数[6](@ref)
                errormsg=f"股票{r_code}解析失败: {str(e)}。连续次数{error_count}"
                handle_error(r_code, e, errormsg, error_count)  # 封装错误处理
                if error_count >= MAX_CONSECUTIVE_ERRORS:
                    break  # 达到阈值终止循环                
            except ValueError as e:
                error_count += 1  # 捕获特定异常时计数[6](@ref)
                errormsg=f"股票{r_code}表格缺失: {str(e)}。连续次数{error_count}"
                handle_error(r_code, e, errormsg, error_count)  # 封装错误处理
                if error_count >= MAX_CONSECUTIVE_ERRORS:
                    break  # 达到阈值终止循环
            except Exception as e:
                error_count += 1  # 捕获特定异常时计数[6](@ref)
                errormsg=f"处理{row['代码']}时出错：{str(e)}。连续次数{error_count}"
                handle_error(r_code, e, errormsg, error_count)  # 封装错误处理
                if error_count >= MAX_CONSECUTIVE_ERRORS:
                    break  # 达到阈值终止循环
        
        # 分块存储[1,5](@ref)
        df_result.to_excel(f'.\output\stock_result_{file_num}.xlsx', index=False)
        log.info(f"第{file_num+1}批数据已存储，包含{len(df_result)}条记录")
    
    return "所有分块处理完成"

def checkRoeCashEBIT(r_code="601398", startyear=STARTYEAR):
    """
    优化说明：
    1. 新增var4（5年资产负债率<=70%）和var5（应收账款周转天数<30天）指标
    2. 增强NaN值处理机制
    3. 优化数据校验逻辑
    """
    for attempt in range(MAX_CONSECUTIVE_ERRORS):
        try:
            log.info(f"{r_code} 获取 {startyear} 至今财报数据")
        
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(ak.stock_financial_analysis_indicator, symbol=r_code, start_year=startyear)
                df = future.result(timeout=OUTTIME)
            break
    
        except TimeoutError:
            log.error(f"checkRoeCashEBIT接口调用超时，次数{attempt+1} | 股票代码: {r_code}")
            if attempt < MAX_CONSECUTIVE_ERRORS - 1:
                time.sleep(RECONNECT_TIME)
            else:
                log.error(f"无法获取{r_code}的财报数据，跳过")
                return False, False, False, False, False, None
        except Exception as e:
            log.error(f"checkRoeCashEBIT接口调用失败，次数{attempt+1} | 股票代码: {r_code}")
            if attempt < MAX_CONSECUTIVE_ERRORS - 1:
                time.sleep(RECONNECT_TIME)
            else:
                log.error(f"无法获取{r_code}的财报数据，跳过")
                return False, False, False, False, False, None
        
    # 数据清洗和字段处理
    clean_df = df.rename(columns={
        '资产负债率(%)': 'debt_ratio',
        '应收账款周转天数(天)': 'receivable_days'
    }).copy()
    
    # 日期处理和过滤
    clean_df['日期'] = pd.to_datetime(clean_df['日期'], errors='coerce')
    year_end_mask = (clean_df['日期'].dt.month == 12) & (clean_df['日期'].dt.day == 31)
    clean_df = clean_df[year_end_mask].sort_values('日期', ascending=False)

    log.info(f"{r_code}获取年报信息: {format_dates(clean_df['日期'])}")
    
    # 数值型字段转换（增强NaN处理）
    numeric_cols = ['净资产收益率(%)', 'debt_ratio', 'receivable_days', '每股经营性现金流(元)']
    for col in numeric_cols:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col].replace('--', np.nan), errors='coerce')
    
    # ================= 原有指标 ================= 
    # 指标1：ROE平均高于ROE
    roe_values = clean_df['净资产收益率(%)'].head(PASTYEAR)
    var1 = len(roe_values) >= PASTYEAR and roe_values.mean() > ROE
    
    # 指标2：近几年经营现金流/收益>1。
    #cash_flow = clean_df['每股经营性现金流(元)'].head(1)
    #var2 = len(cash_flow) > 0 and cash_flow.iloc[0] > 0
    cash_flow  = clean_df['每股经营性现金流(元)'].head(PASTYEAR)
    cash_flow_pers = cash_flow.mean()
    profit_values_pers = clean_df['扣除非经常性损益后的每股收益(元)'].head(PASTYEAR).mean()
    var2 =  len(cash_flow) >= PASTYEAR and cash_flow.min() >0 and cash_flow_pers/profit_values_pers > CASH2PROFIT
    
    # 指标3：最新净利润 > 前5年最大值
    clean_df['扣非净利润'] = pd.to_numeric(clean_df['扣除非经常性损益后的净利润(元)'], errors='coerce') / 10000
    profit_values = clean_df['扣非净利润']
    if len(profit_values) >= 6:  # 确保有最新1期+前5年数据
        var3 = profit_values.iloc[0] > profit_values.iloc[1:PASTYEAR+1].max()
    else:
        var3 = False
    
    # ================= 新增指标 ================= 
    # 指标4：过去5年资产负债率 <= 70% (增强NaN处理)
    debt_ratios = clean_df['debt_ratio'].head(PASTYEAR).dropna()
    
    # 双重验证：足够年份+全部满足条件
    valid_debt_ratios = debt_ratios[debt_ratios.notna()]
    var4 = False
    
    if len(valid_debt_ratios) >= PASTYEAR:
        # 检查所有值都满足条件
        var4 = (valid_debt_ratios <= DEBT_RATIOS).all()
    
    # 指标5：应收账款周转天数 < 30天 (增强NaN处理)
    receivable_values = clean_df['receivable_days'].head(PASTYEAR).dropna()
    
    valid_receivable = receivable_values[receivable_values.notna()]
    var5 = False
    
    #if len(valid_receivable) >= PASTYEAR:#有些时候财报中没有这个值，所以不做时间判断
        # 检查所有值都满足条件
    var5 = (valid_receivable < RECEIVABLE_DAYS).all()
    
    # 日志记录（包含有效数值）
    log.info(f"""
        {r_code} 财务指标结果:
        var1(ROE>14%): {var1} | 数值: {roe_values.values}
        var2(现金流正): {var2} | 数值: {cash_flow_pers}/{profit_values_pers}
        var3(净利润增长): {var3} | 数值: {profit_values.values}
        var4(负债率<=70%): {var4} | 有效数值: {valid_debt_ratios.values}
        var5(周转天数<30天): {var5} | 有效数值: {valid_receivable.values}
    """)
    
    return var1, var2, var3, var4, var5


## 指标6- 市盈率低于PEMAX且大于0
def check_pe_condition(stock_code="601398",stock_name="", pastday=PASTDAY):
    """
    获取PE，得到指标6- 市盈率低于PEMAX且大于0
    可以通过数据库获取，也可以通过网路获取
    """
    if IS_MYSQL:
        df = gsh.get_stock_pe_his(stock_code)
        df = df.reset_index().copy()
        df = df.rename(columns={
            '日期': 'trade_date'
            }).copy()
    else:
        # 获取最新接口调用添加精确的超时控制
        for attempt in range(MAX_CONSECUTIVE_ERRORS):
            try:
                log.info(f"{stock_code}通过网络接口获取{pastday}天内有效市盈率数据")         
      
                # 添加超时控制（网页[1][1](@ref)推荐方法）
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(ak.stock_a_indicator_lg, symbol=stock_code)
                    df = future.result(timeout=OUTTIME)  # 设置5秒超时[1,8](@ref)
                    if df is not None : 
                        #如果是从网络接口调用获取PE，顺便存入本地数据库
                        batch_data = issp.patch_pe_data(stock_code, stock_name, df)
                        ins.insert_to_mysql(batch_data, issp.INSERT_SQL)
                        log.info(f"{stock_code}PE数据成功存入本地数据库")
                        break

            except TimeoutError:
                log.error(f"check_pe_condition接口调用超时，次数{attempt+1} | 股票代码: {stock_code}")
                if attempt < MAX_CONSECUTIVE_ERRORS - 1:
                    time.sleep(RECONNECT_TIME)
                else:
                    log.error(f"无法获取{stock_code}的有效市盈率数据，跳过")
                    return 0.0, False
            except Exception as e:
                log.error(f"check_pe_condition接口调用失败，次数{attempt+1} | 股票代码: {stock_code}")
                if attempt < MAX_CONSECUTIVE_ERRORS - 1:
                    time.sleep(RECONNECT_TIME)
                else:
                    log.error(f"无法获取{stock_code}的有效市盈率数据，跳过")
                    return 0.0,False  
    
    # 日期处理优化（网页[3][3](@ref)数据格式）
    date_threshold = datetime.datetime.now() - datetime.timedelta(pastday)
    date_threshold = date_threshold.date()
    
    # 数据清洗与计算（网页[1][1](@ref)字段说明）
    try:
        valid_df = df[
            (pd.to_datetime(df['trade_date']).dt.date > date_threshold) 
            & (df['pe'].notna())
        ]
    except KeyError as ke:
        log.error(f"数据字段缺失：{str(ke)} | 请确认接口返回格式")
        return 0.0, False
        
    if valid_df.empty:
        log.info(f"{stock_code}无近期有效市盈率数据")
        return 0.0, False
    
    # 计算逻辑优化（网页[1][1](@ref)数据处理建议）
    try:
        pe_mean = valid_df['pe'].astype(float).mean()
        var6 = 0 < pe_mean < PEMAX 
        dv_ratio = valid_df['dv_ratio'].astype(float).mean()

        log.info(f"{stock_code}近{pastday}内PE<{PEMAX}:var6={var6} | 数值: {pe_mean}")
        log.info(f"{stock_code}股息率 = {dv_ratio}")
        return dv_ratio, var6
    except ValueError as ve:
        log.error(f"市盈率数据类型错误：{str(ve)}")
        return 0.0, False


def format_dates(date_series, fmt='%Y%m%d'):
    """日期序列格式化工具"""
    return (
        pd.to_datetime(date_series, errors='coerce')
        .dt.strftime(fmt)
        .tolist()
    )


def handle_error(code: str, e: Exception, error_msg: str, counter: int):
    """统一处理错误日志和阈值判断"""
    log.error(error_msg)
    time.sleep(RECONNECT_TIME)  # 错误后延迟防止高频请求[6](@ref)
    
    # 触发连续错误异常
    if counter >= MAX_CONSECUTIVE_ERRORS:
        raise ConsecutiveErrorException(
            error_code=5001,
            message=f"连续{counter}次接口异常，服务终止"
        )

class ConsecutiveErrorException(Exception):
    """连续异常超过阈值时触发"""
    def __init__(self, error_code: int, message: str):
        self.error_code = error_code  # 如 5001
        self.message = message
        super().__init__(self.message)

if __name__ == "__main__":
    #time.sleep(600)
    df = selectStock()
    #df=ak.stock_financial_analysis_indicator("600519","2023")
    #cc = df.columns.values
    #print(df)
    #导出Excel并自动调整列宽[4](@ref)
    #df.to_excel(f'.\output\output.xlsx', index=False)
    #selectStock()