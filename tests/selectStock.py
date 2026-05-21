import os
from pathlib import Path
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
##运行配置：
base_path = Path(__file__).parent #系统绝对目录
log = log4ak.LogManager(log_level=log4ak.ERROR)# 日志配置

MAX_CONSECUTIVE_ERRORS = 3  # 最大允许连续错误次数
OUTTIME = 5  # 接口长时间无返回报错
RECONNECT_TIME = 30 #断线重连休眠时间
CHUNK_NUM = 1# 全市场数据过多分10块处理
ISMY = True #是否选取自选配置False/True
IS_MYSQL = True  #PE数据来源，使用数据库速度快很多：数据库/Akshare  True/False

##选股参数设置：
STARTYEAR = "2019"  #计算的起始年份
ROE = 15 #过去几年来平均净资产收益率高于15%
PEMAX = 25 #过去几天平均市盈率低于25且大于0
PASTDAY = 30 #过去30天
PASTYEAR = 5 #计算过去5年的净资产收益率，经营性现金流，负债率，应收账款周期
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
        df_result = pd.DataFrame(columns=['stock','name','ROE','现金','净利','负债','回款','pe_ttm','ratio'])
        log.info(f"开始处理第{file_num+1}批数据，包含{len(chunk_df)}条记录")
        checkcount = 0
        
        # 处理单个数据块
        for row_index, row in chunk_df.iterrows():
            try:
                r_code = row['代码']
                r_name = row['名称']
                checkcount += 1
                log.info(f"处理第{file_num+1}批第{checkcount}条记录：{r_code}")

                # 指标计算
                var1, var2, var3, var4, var5 = checkRoeCashEBIT(r_code, STARTYEAR)
                #varAll = var1 and var2 and var3 and var4 and var5
                #log.info(f"第{file_num+1}批第{checkcount}条记录处理结果varAll={varAll}")

                pe_ttm,ratio  = check_pe_condition(r_code, r_name)
                
                # 结果存储
                df_result.loc[row_index] = {
                    'stock': r_code,
                    'name': r_name,
                    'ROE': var1,
                    '现金': var2,
                    '净利': var3,
                    '负债': var4,
                    '回款': var5,
                    'pe_ttm': pe_ttm,
                    'ratio': ratio
                    #'综合评估': varAll
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
        df_result.to_excel(base_path / f'.\output\select_result_{file_num}.xlsx', index=False)
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

                try:
                    df = future.result(timeout=OUTTIME)
                except TimeoutError:
                    # 显式关闭线程池（强制取消未完成的任务）
                    executor.shutdown(wait=False, cancel_futures=True)  # Python 3.9+ 支持[7](@ref)
                    raise TimeoutError(f"任务超时，已强制终止")

            break
    
        except TimeoutError:
            log.error(f"checkRoeCashEBIT接口调用超时，次数{attempt+1} | 股票代码: {r_code}")
            if attempt < MAX_CONSECUTIVE_ERRORS - 1:
                time.sleep(RECONNECT_TIME)
            else:
                log.error(f"无法获取{r_code}的财报数据，跳过")
                return None, None, None, None, None
        except Exception as e:
            log.error(f"checkRoeCashEBIT接口调用失败，次数{attempt+1} | 股票代码: {r_code}，错误信息:{str(e)}")
            if attempt < MAX_CONSECUTIVE_ERRORS - 1:
                time.sleep(RECONNECT_TIME)
            else:
                log.error(f"无法获取{r_code}的财报数据，跳过")
                return None, None, None, None, None
        
    # 数据清洗和字段处理
    clean_df = df.rename(columns={
        '资产负债率(%)': 'debt_ratio',
        '应收账款周转天数(天)': 'receivable_days'
    }).copy()
    
    # 日期处理和过滤
    clean_df['日期'] = pd.to_datetime(clean_df['日期'], errors='coerce')
    year_end_mask = (clean_df['日期'].dt.month == 12) & (clean_df['日期'].dt.day == 31)
    clean_df = clean_df[year_end_mask].sort_values('日期', ascending=False)

    log.debug(f"{r_code}获取年报信息: {format_dates(clean_df['日期'])}")
    
    # 数值型字段转换（增强NaN处理）
    numeric_cols = ['净资产收益率(%)', 'debt_ratio', 'receivable_days', '每股经营性现金流(元)']
    for col in numeric_cols:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col].replace('--', np.nan), errors='coerce')

    # 指标1：平均ROE
    roe_values = clean_df['净资产收益率(%)'].head(PASTYEAR)
    var1 = '{:.2f}'.format(roe_values.mean()) if len(roe_values) >= PASTYEAR else None
    
    # 指标2：近几年经营现金流/收益
    #cash_flow = clean_df['每股经营性现金流(元)'].head(1)
    #var2 = len(cash_flow) > 0 and cash_flow.iloc[0] > 0
    cash_flow  = clean_df['每股经营性现金流(元)'].head(PASTYEAR)
    cash_flow_pers = cash_flow.fillna(0).mean()
    profit_values_pers = clean_df['扣除非经常性损益后的每股收益(元)'].head(PASTYEAR).fillna(0).mean()
    var2 =  '{:.2f}'.format(cash_flow_pers/profit_values_pers) 
    
    # 指标3：最新净利润/前5年平均
    clean_df = clean_df.rename(columns={'扣除非经常性损益后的净利润(元)': '扣非净利润'})
    profit_value_lastyear = clean_df['扣非净利润'].copy().fillna(0).iloc[0]
    var3 = '{:.2f}'.format(profit_value_lastyear / clean_df['扣非净利润'].dropna().iloc[1:PASTYEAR+1].mean())
    
    # 指标4：过去5年平均资产负债率(增强NaN处理)
    debt_ratios = clean_df['debt_ratio'].head(PASTYEAR).dropna()
    var4 = '{:.2f}'.format(debt_ratios.mean())
    
    # 指标5：应收账款周转天数(增强NaN处理)
    receivable_values = clean_df['receivable_days'].head(PASTYEAR).dropna()    
    var5 = '{:.2f}'.format(receivable_values.mean())
    
    # 日志记录（包含有效数值）
    log.debug(f"""
        {r_code} 财务指标结果:
        var1(ROE): {var1} 
        var2(现金流): {var2}
        var3(净利润): {var3}
        var4(负债率): {var4}
        var5(周转天数): {var5}
        """)
    
    return var1, var2, var3, var4, var5



def check_pe_condition(stock_code="601398",stock_name="", pastday=PASTDAY):
    """
    获取pe_ttm，ratio
    可以通过数据库获取，也可以通过网路获取
    """
    if IS_MYSQL:
        #通过数据库获取PE信息
        df = gsh.get_stock_pe_his(stock_code)
        df = df.reset_index().copy()
        df = df.rename(columns={
            '日期': 'trade_date'
            }).copy()   

        if len(df) == 0:
            getPEfromAkshare(stock_code, stock_name)
    else:
        #通过网络库获取PE信息
        df = getPEfromAkshare(stock_code, stock_name)

    #非空判断
    if len(df) == 0:
        log.error(f"{stock_code}数据库中无有效市盈率数据")
        return 0.0, False       
    
    # 日期处理优化（网页[3][3](@ref)数据格式）
    date_threshold = datetime.datetime.now() - datetime.timedelta(pastday)
    year_threshold = datetime.datetime.now() - datetime.timedelta(PASTYEAR * 365)
    date_threshold = date_threshold.date()
    year_threshold = year_threshold.date()

    pe_ttm = None
    dv_ratio = None
    
    # PE数据清洗与计算（网页[1][1](@ref)字段说明）
    valid_df = df[
        pd.to_datetime(df['trade_date']).dt.date > date_threshold
    ].copy()
        
    null_count = valid_df['pe_ttm'].isnull().sum()
    if null_count > 0:
        log.info(f"{stock_code}在{pastday}天之内有{null_count}条pe_ttm空值")

    # pe_ttm的空值不处理，直接求平均值（因为pe小是更好的）
    pe_ttm = '{:.2f}'.format(valid_df['pe_ttm'].astype(float).mean())

    # dv_ratio数据清洗与计算（网页[1][1](@ref)字段说明）
    valid_df = df[
        pd.to_datetime(df['trade_date']).dt.date > year_threshold
    ].copy()
        
    null_count = valid_df['dv_ratio'].isnull().sum()
    if null_count > 0:
        log.info(f"{stock_code}在{year_threshold}之后有{null_count}条股息率空值（已填充为0）")

    # 处理空值：将dv_ratio中的空值填充为0
    valid_df['dv_ratio'] = valid_df['dv_ratio'].fillna(0)
    dv_ratio = '{:.2f}'.format(valid_df['dv_ratio'].astype(float).mean())

    log.info(f"{stock_code}近{pastday}天内pe_ttm={pe_ttm}")
    log.info(f"{stock_code}近{PASTYEAR}年内股息率 dv_ratio={dv_ratio}")

    return pe_ttm, dv_ratio

def getPEfromAkshare(stock_code:str, stock_name:str) -> pd.DataFrame:
    """
    通过Akshare网路接口获取股票PE信息
    """
    # 获取最新接口调用添加精确的超时控制
    df = None
    for attempt in range(MAX_CONSECUTIVE_ERRORS):
        try:
            log.info(f"{stock_code}通过网络接口获取有效市盈率数据")         
      
            # 添加超时控制（网页[1][1](@ref)推荐方法）
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(ak.stock_a_indicator_lg, symbol=stock_code)
                df = future.result(timeout=OUTTIME)  # 设置5秒超时[1,8](@ref)
            if len(df) or df is None == 0:
                break
                
        except TimeoutError:
            log.error(f"check_pe_condition接口调用超时，次数{attempt+1} | 股票代码: {stock_code}")
            if attempt < MAX_CONSECUTIVE_ERRORS - 1:
                time.sleep(RECONNECT_TIME)
            else:
                log.error(f"无法获取{stock_code}的有效市盈率数据，跳过")
                return None
        except Exception as e:
            log.error(f"check_pe_condition接口调用失败，次数{attempt+1} | 股票代码: {stock_code}")
            if attempt < MAX_CONSECUTIVE_ERRORS - 1:
                time.sleep(RECONNECT_TIME)
            else:
                log.error(f"无法获取{stock_code}的有效市盈率数据，跳过")
                return None

        #如果是从网络接口调用获取PE，顺便存入本地数据库
        # 使用assign实现向量化赋值
        df = df.assign(**{
        'stock_code': stock_code,
        'stock_name': stock_name
        })
        #首先把装df数据到存储的列表中
        batch_data = issp.process_pe_data_batch(df)
        #调用数据库接口存储入库
        ins.insert_to_mysql(batch_data, issp.INSERT_SQL)
        log.info(f"{stock_code}PE数据成功存入本地数据库")
        return df
                    



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
