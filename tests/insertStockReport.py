import time
import numpy as np
import akshare as ak
import pandas as pd
import datetime
from typing import List, Dict, Tuple, Optional
import log4ak
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from getAllStock import get_all_stocks, get_select_stocks
import insert2Mysql as i2m

# 日志配置
log = log4ak.LogManager(log_level=log4ak.INFO)# 日志配置

MAX_CONSECUTIVE_ERRORS = 25  # 最大允许连续错误次数
OUTTIME = 5  # 接口长时间无返回报错
RECONNECT_TIME = 60 #断线重连休眠时间
CHUNK_NUM = 10# 分块数量处理设置

STARTYEAR = "2019"  #计算的起始年份

INSERT_SQL ="""
    INSERT INTO stock_financial_reports (
        stock_code, stock_name, report_date, diluted_eps, weighted_eps, adjusted_eps, non_gaap_eps,
        net_asset_per_share, adjusted_net_asset, operating_cash_flow_per_share,
        capital_reserve_per_share, retained_earnings_per_share, adjusted_net_asset_value,
        roa, operating_profit_margin, net_profit_margin, gross_profit_margin, roe, weighted_roe,
        receivables_turnover, inventory_turnover, total_asset_turnover,
        cash_flow_to_sales, cash_flow_to_net_income
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
    """

COLUMNS=['日期', '摊薄每股收益(元)', '加权每股收益(元)', '每股收益_调整后(元)', '扣除非经常性损益后的每股收益(元)',
       '每股净资产_调整前(元)', '每股净资产_调整后(元)', '每股经营性现金流(元)', '每股资本公积金(元)',
       '每股未分配利润(元)', '调整后的每股净资产(元)', '总资产利润率(%)', '主营业务利润率(%)', '总资产净利润率(%)',
       '成本费用利润率(%)', '营业利润率(%)', '主营业务成本率(%)', '销售净利率(%)', '股本报酬率(%)',
       '净资产报酬率(%)', '资产报酬率(%)', '销售毛利率(%)', '三项费用比重', '非主营比重', '主营利润比重',
       '股息发放率(%)', '投资收益率(%)', '主营业务利润(元)', '净资产收益率(%)', '加权净资产收益率(%)',
       '扣除非经常性损益后的净利润(元)', '主营业务收入增长率(%)', '净利润增长率(%)', '净资产增长率(%)',
       '总资产增长率(%)', '应收账款周转率(次)', '应收账款周转天数(天)', '存货周转天数(天)', '存货周转率(次)',
       '固定资产周转率(次)', '总资产周转率(次)', '总资产周转天数(天)', '流动资产周转率(次)', '流动资产周转天数(天)',
       '股东权益周转率(次)', '流动比率', '速动比率', '现金比率(%)', '利息支付倍数', '长期债务与营运资金比率(%)',
       '股东权益比率(%)', '长期负债比率(%)', '股东权益与固定资产比率(%)', '负债与所有者权益比率(%)',
       '长期资产与长期资金比率(%)', '资本化比率(%)', '固定资产净值率(%)', '资本固定化比率(%)', '产权比率(%)',
       '清算价值比率(%)', '固定资产比重(%)', '资产负债率(%)', '总资产(元)', '经营现金净流量对销售收入比率(%)',
       '资产的经营现金流量回报率(%)', '经营现金净流量与净利润的比率(%)', '经营现金净流量对负债比率(%)', '现金流量比率(%)',
       '短期股票投资(元)', '短期债券投资(元)', '短期其它经营性投资(元)', '长期股票投资(元)', '长期债券投资(元)',
       '长期其它经营性投资(元)', '1年以内应收帐款(元)', '1-2年以内应收帐款(元)', '2-3年以内应收帐款(元)',
       '3年以内应收帐款(元)', '1年以内预付货款(元)', '1-2年以内预付货款(元)', '2-3年以内预付货款(元)',
       '3年以内预付货款(元)', '1年以内其它应收款(元)', '1-2年以内其它应收款(元)', '2-3年以内其它应收款(元)',
       '3年以内其它应收款(元)']

# 列名映射（中文列名 -> 数据库列名）
COLUMN_MAP = {
    # 基础信息
    'stock_code': 'stock_code',
    'stock_name': 'stock_name',
    '日期': 'report_date',
    
    # 每股指标 (单位：元)
    '摊薄每股收益(元)': 'diluted_eps',
    '加权每股收益(元)': 'weighted_eps',
    '每股收益_调整后(元)': 'adjusted_eps',
    '扣除非经常性损益后的每股收益(元)': 'non_gaap_eps',
    '每股净资产_调整前(元)': 'net_asset_per_share',
    '每股净资产_调整后(元)': 'adjusted_net_asset',
    '每股经营性现金流(元)': 'operating_cash_flow_per_share',
    '每股资本公积金(元)': 'capital_reserve_per_share',
    '每股未分配利润(元)': 'retained_earnings_per_share',
    '调整后的每股净资产(元)': 'adjusted_net_asset_value',
    
    # 盈利能力 (%)
    '总资产利润率(%)': 'roa',
    '主营业务利润率(%)': 'operating_profit_margin',
    '总资产净利润率(%)': 'roa_profit_margin',
    '成本费用利润率(%)': 'cost_profit_ratio',
    '营业利润率(%)': 'operating_profit_ratio',
    '主营业务成本率(%)': 'main_cost_ratio',
    '销售净利率(%)': 'net_profit_margin',
    '股本报酬率(%)': 'capital_return_ratio',
    '净资产报酬率(%)': 'roe_return_ratio',
    '资产报酬率(%)': 'asset_return_ratio',
    '销售毛利率(%)': 'gross_profit_margin',
    '三项费用比重': 'three_expense_ratio',
    '非主营比重': 'non_main_ratio',
    '主营利润比重': 'main_profit_ratio',
    '股息发放率(%)': 'dividend_payout_ratio',
    '投资收益率(%)': 'investment_return_ratio',
    '净资产收益率(%)':'roe',
    '加权净资产收益率(%)': 'weighted_roe',
    
    # 成长能力 (%)
    '主营业务收入增长率(%)': 'revenue_growth',
    '净利润增长率(%)': 'net_profit_growth',
    '净资产增长率(%)': 'net_asset_growth',
    '总资产增长率(%)': 'total_asset_growth',
    
    # 营运能力
    '应收账款周转率(次)': 'receivables_turnover',
    '应收账款周转天数(天)': 'receivables_days',
    '存货周转天数(天)': 'inventory_days',
    '存货周转率(次)': 'inventory_turnover',
    '固定资产周转率(次)': 'fixed_asset_turnover',
    '总资产周转率(次)': 'total_asset_turnover',
    '总资产周转天数(天)': 'total_asset_days',
    '流动资产周转率(次)': 'current_asset_turnover',
    '流动资产周转天数(天)': 'current_asset_days',
    '股东权益周转率(次)': 'equity_turnover',
    
    # 偿债能力
    '流动比率': 'current_ratio',
    '速动比率': 'quick_ratio',
    '现金比率(%)': 'cash_ratio',
    '利息支付倍数': 'interest_coverage',
    '长期债务与营运资金比率(%)': 'long_term_debt_ratio',
    '股东权益比率(%)': 'equity_ratio',
    '长期负债比率(%)': 'long_term_liability_ratio',
    '股东权益与固定资产比率(%)': 'equity_to_fixed_assets',
    '负债与所有者权益比率(%)': 'debt_to_equity',
    '长期资产与长期资金比率(%)': 'long_term_assets_ratio',
    '资本化比率(%)': 'capitalization_ratio',
    '固定资产净值率(%)': 'fixed_asset_net_ratio',
    '资本固定化比率(%)': 'fixed_capitalization_ratio',
    '产权比率(%)': 'equity_multiplier',
    '清算价值比率(%)': 'liquidation_value_ratio',
    '固定资产比重(%)': 'fixed_asset_ratio',
    '资产负债率(%)': 'asset_liability_ratio',
    
    # 现金流指标 (%)
    '经营现金净流量对销售收入比率(%)': 'cash_flow_to_sales',
    '资产的经营现金流量回报率(%)': 'cash_flow_return_on_assets',
    '经营现金净流量与净利润的比率(%)': 'cash_flow_to_net_income',
    '经营现金净流量对负债比率(%)': 'cash_flow_to_debt',
    '现金流量比率(%)': 'cash_flow_ratio',
    
    # 资产与投资 (单位：元)
    '总资产(元)': 'total_assets',
    '短期股票投资(元)': 'short_stock_invest',
    '短期债券投资(元)': 'short_bond_invest',
    '短期其它经营性投资(元)': 'short_other_invest',
    '长期股票投资(元)': 'long_stock_invest',
    '长期债券投资(元)': 'long_bond_invest',
    '长期其它经营性投资(元)': 'long_other_invest',
    '主营业务利润(元)': 'main_profit',
    '扣除非经常性损益后的净利润(元)': 'non_gaap_net_profit',
    
    # 应收款项账龄 (单位：元)
    '1年以内应收帐款(元)': 'receivables_1y',
    '1-2年以内应收帐款(元)': 'receivables_1_2y',
    '2-3年以内应收帐款(元)': 'receivables_2_3y',
    '3年以内应收帐款(元)': 'receivables_over_3y',
    '1年以内预付货款(元)': 'prepayment_1y',
    '1-2年以内预付货款(元)': 'prepayment_1_2y',
    '2-3年以内预付货款(元)': 'prepayment_2_3y',
    '3年以内预付货款(元)': 'prepayment_over_3y',
    '1年以内其它应收款(元)': 'other_receivables_1y',
    '1-2年以内其它应收款(元)': 'other_receivables_1_2y',
    '2-3年以内其它应收款(元)': 'other_receivables_2_3y',
    '3年以内其它应收款(元)': 'other_receivables_over_3y'
}


# 创建反向映射（数据库列名 -> 中文列名）
REVERSE_COLUMN_MAP = {v: k for k, v in COLUMN_MAP.items()}

def insertStockReport(path:str):
    ## 存入所选的A股上市公司历史财报
    if path == "all":
        stock_zh_a_spot_df = get_all_stocks()
    else:
        stock_zh_a_spot_df = get_select_stocks()

    log.info("获取到所选的 A 股上市公司列表")
    df_stock = stock_zh_a_spot_df[['代码','名称']]#[2868:]

    # 分块处理设置[2,3](@ref)
    total_rows = len(df_stock)
    
    chunk_indices = np.array_split(np.arange(total_rows), CHUNK_NUM)
    log.info(f"分块处理设置总记录数total_rows={total_rows}；块数CHUNK_NUM={CHUNK_NUM}，每块记录数chunk_indices={len(chunk_indices[0])}")
      
    # 直接存储处理后的元组列表
    batch_data = []
    # 初始化错误计数器（放在循环体外层）
    error_count = 0  # 连续错误计数器   
    

    # 分批处理逻辑
    for file_num, chunk_idx in enumerate(chunk_indices):
        
        chunk_df = df_stock.iloc[chunk_idx]
        log.info(f"开始处理第{file_num+1}批数据，包含{len(chunk_df)}条记录")
        checkcount = 0
        df_result = pd.DataFrame(columns=COLUMNS)
        
        # 处理单个数据块
        for row_index, row in chunk_df.iterrows():
            try:
                r_code = row['代码']
                r_name = row['名称']
                checkcount += 1
                log.info(f"处理第{file_num+1}批第{checkcount}条记录：{r_code}")

                # 获取股票历史财报
                dffin=get_financial_report(r_code)
                log.info(f"获取到{r_code}历史财报，数据条数:{len(dffin)}")

                # 使用assign实现向量化赋值
                dffin = dffin.assign(**{
                    'stock_code': r_code,
                    'stock_name': r_name
                    })

                #df数据合并
                df_result = pd.concat([df_result, dffin], ignore_index=True)
                log.info(f"df_result数据块合并后大小为:{len(df_result)}")

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

        log.info(f"第{file_num+1}批数据已获取，包含{len(df_result)}条记录")

        #已经合并好的df数据进行入库数据封装：1，转list按行处理。2，Nan->None。
        batch_data,sql = process_fin_data_batch(df_result)
        #调用数据库接口存储入库



        i2m.insert_to_mysql(batch_data, sql)

    # 所有数据处理完成后插入数据库
    log.info(f"所有数据已处理完成，共{len(batch_data)}条记录")


    return "所有分块处理完成"

def get_financial_report(r_code: str = "600004", start_year: str = "1900") -> pd.DataFrame:
    """
    通过akshare接口获取财报数据
    :param r_code: 股票代码
    :param start_year: 开始年份

    :return: 新浪财经-财务分析-财务指标
    :rtype: pandas.DataFrame
    
    """
    
    for attempt in range(MAX_CONSECUTIVE_ERRORS):
        try:
            #log.info(f"{r_code} 获取 {STARTYEAR} 至今财报数据")
        
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(ak.stock_financial_analysis_indicator, symbol=r_code, start_year=STARTYEAR)

                try:
                    df = future.result(timeout=OUTTIME)
                except TimeoutError:
                    # 显式关闭线程池（强制取消未完成的任务）
                    executor.shutdown(wait=False, cancel_futures=True)  # Python 3.9+ 支持[7](@ref)
                    raise TimeoutError(f"任务超时，已强制终止")

            return df
    
        except TimeoutError:
            log.error(f"get_financial_report接口调用超时，次数{attempt+1} | 股票代码: {r_code}")
            if attempt < MAX_CONSECUTIVE_ERRORS - 1:
                time.sleep(RECONNECT_TIME)
            else:
                log.error(f"无法获取{r_code}的财报数据，跳过")
                return None
        except Exception as e:
            log.error(f"get_financial_report接口调用失败，次数{attempt+1} | 股票代码: {r_code}，错误信息:{str(e)}")
            if attempt < MAX_CONSECUTIVE_ERRORS - 1:
                time.sleep(RECONNECT_TIME)
            else:
                log.error(f"无法获取{r_code}的财报数据，跳过")
                return None
    

def process_fin_data_batch(df: pd.DataFrame):
    """
    将akshare查询到的财报数据转换为数据库插入格式
    
    参数:
        df: 从akshare获取的原始DataFrame(含中文列名)
        stock_code: 当前处理的股票代码
        
    返回:
        list: 包含所有记录元组的列表，可直接用于executemany批量插入
    """    
    
    # 2. 复制数据避免污染原始数据
    processed_df = df.copy()
    
    # 重命名列
    processed_df = processed_df.rename(columns=COLUMN_MAP)
           
    # 准备插入语句
    columns = ', '.join(processed_df.columns)
    placeholders = ', '.join(['%s'] * len(df.columns))
    sql = f"""
        INSERT IGNORE INTO stock_financial_reports ({columns})
        VALUES ({placeholders})
    """
    
    # 5. 处理特殊字段
    # 确保日期格式正确(转换为YYYY-MM-DD格式)
    if 'report_date' in processed_df:
        processed_df['report_date'] = pd.to_datetime(processed_df['report_date']).dt.strftime('%Y-%m-%d')
    
    # 6. 处理缺失值(将NaN转为None)
    processed_df = processed_df.replace({np.nan: None})
    
    # 7. 转换为元组列表
    batch_data = []
    for row in processed_df.itertuples(index=False):
        # 显式处理每个元素，确保没有NaN残留
        processed_row = tuple(
            None if pd.isna(item) else item
            for item in row
        )
        batch_data.append(processed_row)
    
    return batch_data,sql


def get_stockfin_data_from_mysql(stock_code: str,start_date: str = None) -> pd.DataFrame:
    """
    从MySQL数据库查询股票历史数据
    :param stock_code: 股票代码
    :param start_date: 开始日期(YYYY-MM-DD)
    :return: 包含历史财报数据的DataFrame(列名为中文)
    """
    # 构建查询语句

    sql = f"""
        SELECT {', '.join(REVERSE_COLUMN_MAP.keys())}
        FROM stock_financial_reports
        WHERE stock_code = %s
    """
    columns, rows = i2m._execute_query(sql, (stock_code,))
    df = convert_to_dataframe(columns, rows, REVERSE_COLUMN_MAP)
    decimal_columns = []  
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
    #df = insertSelectStockPE(SELECT_PATH)

    #查询所有股票PE并入库
    
    df = insertStockReport("all")#all

    #df = get_stockfin_data_from_mysql('600036')
    #print(df)


    #导出Excel并自动调整列宽[4](@ref)
    #with pd.ExcelWriter(".\output\output.xlsx") as writer:
    #    df.to_excel(writer, sheet_name="全量数据")
    #selectStock()