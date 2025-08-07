import os
from pathlib import Path
from typing import List, Tuple, Union
import time
import numpy as np
import akshare as ak
import pandas as pd
from tqdm import tqdm
from getAllStock import get_all_stocks, get_select_stocks
import get_industry_historyPE as gi
import get_stockPE_his as gs
import insertStockHist as ish
import insert_major_index_valuation as imiv
import commTools as ct
from datetime import datetime, timedelta
import log4ak

base_path = Path(__file__).parent #系统绝对目录
log = log4ak.LogManager(log_level=log4ak.INFO)# 日志配置
MAX_CONSECUTIVE_ERRORS = 3  # 最大允许连续错误次数
OUTTIME = 5  # 接口长时间无返回报错
ALL_PE_DOC_NUM = 1 #当输出ALLPEexcel文档时，分割为几个文档输出


KEEPDAY = 3
DODUP = 0.12 #连续5个交易日放量平均增长率15%
GROUNDVOLUME_PERCENTILE = 5 #地价地量阈值检测时间内5%分位
N_YEARS = 3  #回测N_YEARS年百分位
START_DATE = "20160701" #回测开始日期

ISMY_SELECT = False#是否检测自选True/False
ISALL = False#是否对所有检测标的输出结果True/False
IS_MYSQL = True #PE数据来源，使用数据库速度快很多：数据库/Akshare  True/False
IS_BUY = False#是否直接返回量价买点
BUY_WITH_PE_PERCENTLE = False#通过PE和成交量判断买点

PE_ROLLING_TIME = 3 #滚动PE百分位时间配置，默认5年
PE_PERCENTILE = 5 #滚动PE百分位阈值配置，默认5%
EQUAL_WEIGHT_BUY = True#是否等权买入，即每支股票只买入一次
IS_BUY_K = True#是否加入指数PE分位系数
PE300_PERCENTILE_YEAR = 3#PE分位回溯时长（年）
DF_HS300PETTM = imiv.get_index_pe_his('沪深300') if IS_BUY_K else None


def detect_price_volume_reversal(stock_list: pd.DataFrame, 
                          start_date: str = "20200101", 
                          n_years: int = N_YEARS) -> list:
    """
    地量地价反转信号检测函数
    
    参数：
        stock_list : DataFrame，包含列["代码","名称"]
        start_date : str，检测起始日期（格式：YYYYMMDD）
        n_years    : int，历史数据回溯年限
    
    返回：
        result     : list[DataFrame]，包含符合条件的股票及信号特征：
            ["代码","名称","地量日期","反转日期","量能变化率(%)"]
    """
    # 初始化结果容器
    result = []
    signals_list = []  # 存储检测到的信号[2,9](@ref)

    # 获取当前日期
    #today = datetime.now().strftime("%Y%m%d") # 当前日期
    
    for row in stock_list.itertuples(index=False):
        code = row[0]
        name = row[1]
        try:
            
            # 获取历史数据（需替换为实际数据接口）
            df = get_stock_data(code, start_date)  # 假设返回包含日期、成交额、收盘价的DataFrame
            # 添加数据有效性校验，避免停牌日零成交量的干扰
            hist_data = df[df['成交额'] > 0].copy()
            #按时间排序
            hist_data = hist_data.sort_index(ascending=True)
            
            # 计算n年历史分位（参考网页5/7的地量判断逻辑）
            rolling_window = n_years * 243  # 假设每年243个交易日

            #收盘价历史分位
            hist_data['price'] = hist_data['收盘'].rolling(rolling_window).apply(
                lambda x: x.rank(pct=True).iloc[-1], raw=False)
            # 收盘价检测地量条件
            hist_data['p_mask'] = hist_data['price'] < GROUNDVOLUME_PERCENTILE/100  # 收盘价处于近n年最低5%分位
            
            #成交额历史分位
            hist_data['volume'] = hist_data['成交额'].rolling(rolling_window).apply(
                lambda x: x.rank(pct=True).iloc[-1], raw=False)            
            # 成交额检测地量条件
            hist_data['v_mask'] = hist_data['volume'] < GROUNDVOLUME_PERCENTILE/100  # 成交额处于近n年最低5%分位
            
            # 检测连续量能递增（网页2/9的递增逻辑）
            hist_data['v_growth'] = hist_data['成交额'].pct_change() + 1
            #consecutive_growth = (hist_data['volume_growth']
            #                      .rolling(KEEPDAY).apply(lambda x: np.all(x >= (1+DODUP))))
            hist_data['vg_mask'] = hist_data['v_growth'].rolling(KEEPDAY).apply(lambda x: np.mean(x >= (1+DODUP)))==1
            
            result.append(hist_data)

            log.info(f"{code}量价分位分析完毕，已完成{len(result)}个分析处理。")

            if IS_BUY:

                signals_list.extend(test_buy_signals(code,name,hist_data))

                ## ==== 新增：检测连续5天量价信号 ====
                ## 创建同时满足条件的掩码
                #hist_data['both_mask'] = hist_data['p_mask'] & hist_data['v_mask']

                ## 生成连续区块标识
                #hist_data['block_id'] = (hist_data['both_mask'] != hist_data['both_mask'].shift(1)).cumsum()
            
                ## 检测连续5天满足条件[7](@ref)
                #consecutive_mask = (
                #    hist_data['both_mask']
                #    .rolling(window=5)
                #    .apply(lambda x: np.all(x), raw=False)
                #)
            
                ## 提取满足条件的日期和价格
                #consecutive_dates = consecutive_mask[consecutive_mask == True].index

                ## 按区块分组，只取每组第一个信号（关键去重逻辑）
                ## 重新获取不复权的价格
                #current_block = -1                
                #for date in consecutive_dates:
                #    block_id = hist_data.loc[date, 'block_id']
                #    # 新区块的第一个信号
                #    if block_id != current_block:  
                #        price = hist_data.loc[date, '收盘']
                #        signals_list.append({
                #            'A股代码': code,
                #            'buydate': date,
                #            'price': price
                #        })
                #        current_block = block_id  # 更新当前区块
                #        log.info(f"ok 新增信号 @ {date} (区块:{block_id})")
                #    else:
                #        log.info(f"pass 跳过连续信号 @ {date} (同区块:{block_id})")
                ## ===== 新增部分结束 =====

            time.sleep(0.3)        
        except Exception as e:
            log.error(f"股票{code}数据处理异常: {str(e)}")
            time.sleep(0.3) 
            continue


    # 如果检测买点，就返回买入信号列表
    if IS_BUY and signals_list:
        return signals_list
    else:
        #否则返回正常检测列表
        return result
    
def test_buy_signals(code:str, name:str,hist_data: pd.DataFrame) -> list: 
    """
    检测买点信号
    1，​区块化处理​：通过block_id将连续信号分组，避免重复检测
    ​2，终点标记法​：滚动窗口检测会标记连续区间的结束点（第5/6/7...天）
    ​3，首次触发原则​：每个连续信号区间只取第一个有效信号
    参数：
        code：str 股票代码
        hist_data : 股票历史价格和交易额数据
    
    返回：
        signals_list，满足买点信号条件的list[代码，日期，价格]
    """
   # 初始化信号存储和过滤条件
    signals_list = []
    last_valid_signal_date = None
    MIN_DAYS_BETWEEN_SIGNALS = 30
    excluded_early_signals = 0
    MAX_EARLY_SIGNALS_TO_EXCLUDE = 2 #第几次新低才发起信号，3波浪理论

    log.debug(f"历史数据最早日期：{hist_data.index[0]}")


    if BUY_WITH_PE_PERCENTLE:
        # 获取完整的个股历史PE百分位（尽可能早的数据）
        all_stock_pe_percentle = gs.calculate_pe_time_percentile(code,PE_ROLLING_TIME)
        hist_data  = ct.merge_on_date_str_index(hist_data, all_stock_pe_percentle)
        # pe_ttm百分位检测条件
        hist_data['pe_mask'] = hist_data['pettm_per'] < PE_PERCENTILE  # PE百分位低于阈值
        # 创建联合条件掩码
        hist_data['both_mask'] = hist_data['pe_mask'] & hist_data['v_mask']
    else:
        # 创建联合条件掩码
        hist_data['both_mask'] = hist_data['p_mask'] & hist_data['v_mask']

    # 生成连续区块标识
    hist_data['block_id'] = (hist_data['both_mask'] != hist_data['both_mask'].shift(1)).cumsum()

    # 检测连续5天满足条件
    consecutive_mask = (
        hist_data['both_mask']
        .rolling(window=KEEPDAY)
        .apply(lambda x: np.all(x), raw=False)
    )

    # 提取满足条件的日期和价格
    consecutive_dates = consecutive_mask[consecutive_mask == True].index
        
    # 按区块分组处理信号
    current_block = -1
    for date in consecutive_dates:
        block_id = hist_data.loc[date, 'block_id']
    
        # 条件1：跳过前两次信号
        if excluded_early_signals < MAX_EARLY_SIGNALS_TO_EXCLUDE:
            
            log.info(f"{code}排除早期信号 #{excluded_early_signals} @ {date}")
            excluded_early_signals += 1
            continue
        
        # 条件2：检查30天内是否已有有效信号
        if last_valid_signal_date is not None:
            days_since_last = (date - last_valid_signal_date).days
            if days_since_last < MIN_DAYS_BETWEEN_SIGNALS:
                log.info(f"{code}跳过30天内重复信号 @ {date} (上次信号:{last_valid_signal_date})")
                continue
    
        # 新区块的第一个信号
        if block_id != current_block:
            pp = hist_data.loc[date, 'pe_ttm'] if BUY_WITH_PE_PERCENTLE else hist_data.loc[date, '收盘']

            # 获取指定日期的PE
            onedayPE = gs.get_stock_pe(code,date.strftime('%Y%m%d'))            
            #判断指定日期股息率>
            dv_ratio =  onedayPE['dv_ratio'].iloc[0] if onedayPE is not None else -1

            hs300PEttm_percentile = imiv.get_pe_percentile(DF_HS300PETTM,date.strftime('%Y%m%d'), imiv.PE_TTM,PE300_PERCENTILE_YEAR) if IS_BUY_K else 0
        
            # 记录有效信号
            signals_list.append({
                'A股代码': code,
                'buydate': date,
                'price/pe': pp,
                'dv_ratio':dv_ratio,
                '300%':  '{:.2f}'.format(hs300PEttm_percentile),
                '名称': name,
            })
            last_valid_signal_date = date
            current_block = block_id

            #连续信号次数重置
            excluded_early_signals=0
        
            log.info(f"ok {code}有效信号 @ {date} (区块:{block_id})，重置连续信号次数{excluded_early_signals}")

            #如果等权买入，每支股票只录入第一个信号
            if EQUAL_WEIGHT_BUY:
                break

        else:
            log.info(f"pass {code}同区块跳过 @ {date}")
    return signals_list



def get_stock_data(code: str, start_date: str) -> pd.DataFrame:
    """
    使用akshare获取股票历史数据（前复权）
    
    参数：
        code : 股票代码（支持格式：'600519' 或 '000001.SZ'）
        start_date : 开始日期（格式：'YYYYMMDD'）
    
    返回：
        DataFrame，包含列：日期（索引）、成交额、收盘价
    """
    # 清洗代码格式（兼容带后缀的代码）
    code_clean = code.split('.')[0]
    
    df = pd.DataFrame()
    
    if IS_MYSQL:
        df = ish.get_stock_data_from_mysql(code_clean,'qfq')
        df = df[['日期', '收盘', '成交额']].copy()
        start_date_dt = pd.to_datetime(start_date, format='%Y%m%d')
        df['日期']= pd.to_datetime(df['日期'], format='%Y%m%d')

        df = df[df['日期'] > start_date_dt]

    else:
        df = ak.stock_zh_a_hist(
            symbol=code_clean,
            period="daily",
            adjust="qfq",
            start_date=start_date
        )
    
        # 字段处理
        df = df[['日期', '收盘', '成交额']].copy()
    
    # 严格日期处理
    df.loc[:, '日期'] = pd.to_datetime(
        df['日期'], 
        errors='coerce', 
        format='%Y%m%d'
    ).astype('datetime64[ns]')

    df = df.dropna(subset=['日期'])  # 删除无效日期行
    df.set_index('日期', inplace=True)
    return df

def get_stock_industry_pe(stock_code: str) -> pd.DataFrame:
    """
    使用akshare获取股票所数行业的当前估值
    
    参数：
        code : 股票代码（支持格式：'600519' ）
    
    返回：
        DataFrame ：股票所行业最新交易日的估值信息
    """
    result = []
    # 获取行业信息
    industry = gi.get_industry_info(stock_code)

    last_trade_date = ct.get_last_trade_dates()
    # 获取估值
    # 获取行业PE数据（网页2接口）
    pe_df = ak.stock_industry_pe_ratio_cninfo(
        symbol="证监会行业分类",
        date = last_trade_date)

    if industry is not None and pe_df is not None:
        # 数据筛选与格式化（网页2字段结构）
        result = pe_df[pe_df["行业编码"] == industry["行业编码"].values[0]].rename(columns={
            '变动日期': '日期',
            '静态市盈率-加权平均': 'PE静-加权',
            '静态市盈率-中位数': 'PE静-中位',
            '静态市盈率-算术平均': 'PE静-平均',
            '行业名称':'行业'
        })

    return result


def get_stock_pe(stock_code: str):
    """
    使用akshare获取股票最新PE信息
    
    参数：
        code : 股票代码（支持格式：'600519' ）
    
    返回：
        DataFrame ：股票最新交易日的pe，pe_ttm
    """
    stock_df = []
    pe = -1.0
    pe_ttm = -1.0
    # 获取行业    
    stock_df = ak.stock_a_indicator_lg(stock_code)
    last_index = stock_df.index[-1]
    pe=stock_df.loc[last_index,'pe']
    pe_ttm=stock_df.loc[last_index,'pe_ttm']

    return pe,pe_ttm

def getPE_after_detect(result: list, stock_list: pd.DataFrame)-> list:
    """
    检测完成后根据结果按股票代码分查询PE相关信息
    输入： result: list[DataFrame] 包含日期，价格，成交量，5%分位地价True标识，5%分位地量True标识
    """
    resultlist = []
    passNum = 0
   

    for idx, code in tqdm(enumerate(stock_list["代码"]), total=len(stock_list)):
        try:
            if idx >= len(result) or result[idx].empty:
                continue
            df = result[idx].copy()
            # 显式构造索引避免警告

            # 只获取最新的一天进行价格和交易量判断，达到阈值才查询PE信息进行展示
            last_row = df.iloc[-1]

            testTrue = ISALL #默认配置False，调测时改为True使用，

            if any([
                    testTrue,
                    last_row.get('p_mask', False),
                    last_row.get('v_mask', False),
                    last_row.get('vg_mask', False)
                    ]):
            
                # 1. 获取行业和个股的完整历史PE数据（不只是当前日期范围）
                all_industry_pe = None
                all_stock_pe = None
                df_industry_historyPE = []
                df_industry_historyPE = gi.get_stock_industry_pe_mysql(code)


                 # 获取行业信息
                df_industry_historyPE = gi.get_stock_industry_pe_mysql(code)
                if df_industry_historyPE is not None:
                    industry_info = df_industry_historyPE['industry_info']
                    industry_code = industry_info['行业编码'].values[0]
                    # 获取完整的行业历史PE（尽可能早的数据）
                    all_industry_pe = gi.get_industry_pe_mysql_new_conn(industry_code)
            
                # 获取完整的个股历史PE（尽可能早的数据）
                all_stock_pe = gs.get_stock_pe_his(code)
            
                # 2. 合并当前日期范围内的PE数据
                if all_industry_pe is not None:
                    df = pd.merge(df, all_industry_pe, how='left', on='日期', suffixes=('', '_industry'))
                if all_stock_pe is not None:
                    df = pd.merge(df, all_stock_pe, how='left', on='日期', suffixes=('', '_stock'))

                # 3. 初始化新列
                #df['industry_pe_percentile'] = np.nan
                df['stock_pe_percentile'] = np.nan
                #df['industry_pe_mask'] = False
                df['stock_pe_mask'] = False
            
                # 4. 5年滚动百分位计算（核心逻辑）
                if not df.empty:
                    # 确保日期为datetime类型并按时间排序
                    #df['日期'] = pd.to_datetime(df['日期'])
                    df = df.sort_values('日期')
                
                    # 获取完整的行业和个股历史PE数据（包含当前日期之前的所有数据）
                    #full_industry_pe = all_industry_pe.copy() if all_industry_pe is not None else pd.DataFrame()
                    full_stock_pe = all_stock_pe.copy() if all_stock_pe is not None else pd.DataFrame()
                
                    #if not full_industry_pe.empty:
                    #    full_industry_pe['日期'] = pd.to_datetime(full_industry_pe['日期'])
                    #    full_industry_pe = full_industry_pe.sort_values('日期').set_index('日期')
                
                    #if not full_stock_pe.empty:
                        #full_stock_pe['日期'] = pd.to_datetime(full_stock_pe['日期'])
                        #full_stock_pe = full_stock_pe.sort_values('日期').set_index('日期')
                
                    # 遍历每一行计算滚动百分位
                    for i, current_date in enumerate(df.index):
                        window_start = current_date - pd.DateOffset(years=PE_ROLLING_TIME)
                    
                        # 行业PE百分位计算
                        #if not full_industry_pe.empty:
                        #    window_data = full_industry_pe.loc[window_start:current_date - pd.DateOffset(days=1)]
                        #    if not window_data.empty:
                        #        current_pe = df.at[i, 'industry_pe']
                        #        if not pd.isna(current_pe):
                        #            # 计算百分位：小于等于当前值的比例
                        #            rank = np.sum(window_data <= current_pe) / len(window_data)
                        #            df.at[i, 'industry_pe_percentile'] = rank * 100
                        #            df.at[i, 'industry_pe_mask'] = (rank * 100) <= PE_PERCENTILE
                    
                        # 个股PE百分位计算
                        if not full_stock_pe.empty:
                            window_data = full_stock_pe.loc[window_start:current_date - pd.DateOffset(days=1)][ 'pe_ttm']
                            if not window_data.empty:
                                current_pe = df.at[current_date, 'pe_ttm']
                                if not pd.isna(current_pe):
                                    rank = np.sum(window_data <= current_pe) / len(window_data)
                                    df.at[current_date, 'stock_pe_percentile'] = rank * 100
                                    df.at[current_date, 'stock_pe_mask'] = (rank * 100) <= PE_PERCENTILE
            
                # 5. 设置最终索引格式
                df.index = pd.Index(
                    df.index.strftime('%Y%m%d'), 
                    dtype='object', 
                    name='日期'
                )
                resultlist.append((str(code), df))
                passNum += 1
                if not IS_MYSQL: 
                    time.sleep(0.3) 
            
        except Exception as e:
            log.error(f"{code}获取PE信息异常: {str(e)}")
            time.sleep(0.3) 
            continue

    log.info(f"今天检测通过数量：{passNum}")
    return resultlist

def save_to_excel_n(result: list, filename: str, nlist: int = 1) -> None:
    """
    优化版：将检测结果按股票代码分Sheet保存到Excel，支持分割为多个文件并均匀分配记录
    
    参数：
        result   : detect_volume_reversal返回的结果列表
        filename : 输出Excel文件名（如"volume_signals.xlsx"）
        nlist    : 将结果分割为多少份输出（默认1份）
    """
    if not result:
        log.info("结果列表为空，无需保存")
        return
    
    total_records = len(result)
    actual_n = min(nlist, total_records)
    
    # 计算基础块大小和余数（需多存一条记录的文件数）
    base_size = total_records // actual_n
    remainder = total_records % actual_n  # 余数决定前几个文件需多存一条
    
    # 动态分块：前remainder个文件存base_size+1条，其余存base_size条
    chunks = []
    start_index = 0
    for i in range(actual_n):
        # 计算当前块大小（前remainder个文件多1条）
        current_size = base_size + 1 if i < remainder else base_size
        end_index = start_index + current_size
        chunks.append(result[start_index:end_index])
        start_index = end_index
    
    # 写入Excel
    for i, chunk in enumerate(chunks):
        file_suffix = f"_{i+1}" if actual_n > 1 else ""
        output_file = str(filename).replace(".xlsx", f"{file_suffix}.xlsx")
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            for sheet_name, df in chunk:
                df.to_excel(writer, sheet_name=sheet_name, index=True)
        log.info(f"已保存: {output_file} (包含 {len(chunk)} 个Sheet)")

def save_to_excel(result: list, filename: str) -> None:
    """
    将检测结果按股票代码分Sheet保存到Excel
    
    参数：
        result      : detect_volume_reversal返回的结果列表
        filename    : 输出Excel文件名（如"volume_signals.xlsx"）
    """

    # 无有效数据时跳过写入
    if not result:
        log.error("警告：无符合条件的股票数据，跳过Excel生成")
        return
    passNum = 0

    # 写入Excel
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:

            if IS_BUY:
                df = pd.DataFrame(result,index = None)
                df['buydate'] = pd.to_datetime(df['buydate']).dt.strftime('%Y%m%d')
                df.to_excel(writer, sheet_name="信号列表", index=False)
                log.info(f"成功存储买点信号列表excel，数量{len(result)}")      
                return

            # 若所有Sheet均被过滤，添加默认Sheet
            #if len(writer.sheets) == 0:
            #    pd.DataFrame(["无符合条件的数据"]).to_excel(writer, sheet_name="Empty")
            for sheet_name, df in result:
                df.to_excel(writer, sheet_name=sheet_name, index=True)
                passNum += 1

            log.info(f"成功存储excel数量：{passNum}")           
        
    except PermissionError:
        log.error(f"错误：文件 {filename} 被其他程序占用，请关闭后重试")


    #with pd.ExcelWriter(filename, engine='openpyxl') as writer:
    #    # 遍历股票代码与对应的DataFrame
    #    for idx, code in enumerate(stock_list["代码"]):
    #        if idx < len(result) and not result[idx].empty:  # 确保索引不越界且数据非空
    #            # 提取当前股票的DataFrame
    #            df = result[idx].copy()
    #            df.index = df.index.strftime('%Y%m%d')  # 设置日期格式
    #            # 写入Excel，Sheet名使用股票代码（字符串格式避免特殊符号问题）
    #            df.to_excel(
    #                writer, 
    #                sheet_name=str(code), 
    #                index=True,  # 保留日期索引
    #                header=True
    #            )
    
def save_to_excel_filter(result: list, stock_list: pd.DataFrame, filename: str) -> None:
    """
    将检测结果按股票代码分Sheet保存到Excel
    优化后的存储函数（增加3个条件筛选）
    """
    sheets_to_write = []
    passNum = 0
   

    for idx, code in tqdm(enumerate(stock_list["代码"]), total=len(stock_list)):
        try:
            if idx >= len(result) or result[idx].empty:
                continue
            df = result[idx].copy()
            # 显式构造索引避免警告
            df.index = pd.Index(
                df.index.strftime('%Y%m%d'), 
                dtype='object', 
                name='日期'
            ).infer_objects()

            # 只获取最新的一天进行价格和交易量判断，达到阈值才查询PE信息进行展示
            last_row = df.iloc[-1]

            testTrue = ISALL #默认配置False，调测时改为True使用，

            if any([
                    testTrue,
                    last_row.get('p_mask', False),
                    last_row.get('v_mask', False),
                    last_row.get('vg_mask', False)
                    ]):
            
                #查询检测通过股票的行业PE
                df_industry = []
                df_industry = get_stock_industry_pe(code)
            
                # 先检查 df_industry 是否为 None
                if df_industry is not None and not df_industry.empty:
                    last_index = df_industry.index[-1]  # 获取最后一行索引
                    industry_id = df_industry.loc[last_index, "行业编码"]
                    industry_name = df_industry.loc[last_index, "行业"]
                    pe_weighted = df_industry.loc[last_index, "PE静-加权"]
                    pe_mean = df_industry.loc[last_index, "PE静-平均"]
                    pe_median = df_industry.loc[last_index, "PE静-中位"]
                else:
                    # 处理 df_industry 为 None 或 empty 的情况
                    industry_id = industry_name = pe_weighted = pe_mean = pe_median = None

                last_index = df.index[-1]  # 获取最后一行索引
                
                if IS_MYSQL:
                    #通过数据库查询PE
                    log.info(f"通过数据库库获取{code}最后一天的PE数据。")
                    stockPE = gs.get_stock_pe_percentile(code, N_YEARS)#回测N_YEARS年百分位，不带入df的最新日期last_index，而是使用数据库中最新的日期。
                    #stockPE = gs.get_stock_pe_percentile(code, N_YEARS,last_index)
                    stock_pe = stockPE['pe']
                    stock_pe_ttm = stockPE['pe_ttm']
                    stock_pe_percentile = stockPE['percentile']
                else:
                    #通过Akshare接口查询PE
                    stock_pe,stock_pe_ttm = get_stock_pe(code) #通过Akshare接口查询PE
                    stock_pe_percentile = None

                
                df.loc[last_index, 'industry_id'] = industry_id
                df.loc[last_index, 'industry_name'] = industry_name
                df.loc[last_index, 'pe_weighted'] = pe_weighted
                df.loc[last_index, 'pe_mean'] = pe_mean
                df.loc[last_index, 'pe_median'] = pe_median
                df.loc[last_index, 'stock_pe'] = stock_pe
                df.loc[last_index, 'stock_pe_ttm'] = stock_pe_ttm
                df.loc[last_index, 'percentile'] = stock_pe_percentile
                sheets_to_write.append((str(code), df))

                

                passNum += 1
                log.info(f"{code}数据成功存入excel，sheet{passNum}")
                time.sleep(0.3) 
            
        except Exception as e:
            log.error(f"{code}获取行业或股票估值异常: {str(e)}")
            time.sleep(0.3) 
            continue
    
    # 无有效数据时跳过写入
    if not sheets_to_write:
        log.error("警告：无符合条件的股票数据，跳过Excel生成")
        return
    
    # 写入Excel
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            for sheet_name, df in sheets_to_write:
                df.to_excel(writer, sheet_name=sheet_name, index=True)
            # 若所有Sheet均被过滤，添加默认Sheet
            if len(writer.sheets) == 0:
                pd.DataFrame(["无符合条件的数据"]).to_excel(writer, sheet_name="Empty")
        
    except PermissionError:
        log.error(f"错误：文件 {filename} 被其他程序占用，请关闭后重试")

    log.info(f"成功存储excel数量：{passNum}")


def detect_with_allPE(path: str):
    """
    检测所选输入文件，返回检测结果以及所有交易日标的以及所属行业的PE
    """
    my_select=path

    #PE数据来源，使用数据库速度快很多：数据库/Akshare  True/False
    log.info(f"是否检测自选标的：{ISMY_SELECT}")
    log.info(f"是否从数据库获取PE信息：{IS_MYSQL}")

    #选定标的
    if ISMY_SELECT:
        test_stocks = get_select_stocks(my_select)
    else:
        test_stocks = get_select_stocks()

    result = detect_price_volume_reversal(test_stocks, start_date = START_DATE, n_years = N_YEARS)
    write_df = getPE_after_detect(result,test_stocks)

    log.info(f"检查成功检测数：{len(write_df)}")

    end_date = datetime.now().strftime("%Y%m%d")


    if ISMY_SELECT:
        filename =base_path / f'..\output\detect\detect_allPE_{end_date}_my.xlsx'
    else:
        filename =base_path / f'..\output\detect\detect_allPE_{end_date}.xlsx'

    save_to_excel_n(write_df,filename,ALL_PE_DOC_NUM)


def detect_with_lastPE(path: str):
    """
    检测所选输入文件，返回检测结果以及最新交易日标的PE百分位以及所属行业的PE
    """
    my_select=path

    log.info(f"是否检测自选标的：{ISMY_SELECT}")
    log.info(f"是否从数据库获取PE信息：{IS_MYSQL}")
    
    #选定标的
    if ISMY_SELECT:
        test_stocks = get_select_stocks(my_select)
    else:
        test_stocks = get_select_stocks() 

    # 执行检测 选取start_date开始日期数据，n_year内通过股价，交易额分位进行情绪判断买点，并给出标的和行业的估值参考
    #result = detect_price_volume_reversal(test_stocks, start_date = "20230501", n_years=1) 

    result = detect_price_volume_reversal(test_stocks, start_date = START_DATE, n_years = N_YEARS)
    end_date = datetime.now().strftime("%Y%m%d")
    if ISMY_SELECT:
        filename =base_path / f'..\output\detect\detect_rev_lastPE_{end_date}_my.xlsx'
    else:
        filename =base_path / f'..\output\detect\detect_rev_lastPE_{end_date}.xlsx'
    
    log.info(f"检查成功检测数：{len(result)}")

    save_to_excel_filter(result,test_stocks,filename)

def detect_with_buy(path: str):
    """
    检测所选输入文件，返回买点检测信息
    """

    log.info(f"是否检测买点：{IS_BUY}")
    if not IS_BUY:
        log.info(f"不检测买点，直接结束返回")
        return

    my_select=path

    #PE数据来源，使用数据库速度快很多：数据库/Akshare  True/False
    log.info(f"是否检测自选标的：{ISMY_SELECT}")
    log.info(f"是否从数据库获取PE信息：{IS_MYSQL}")


    #选定标的
    if ISMY_SELECT:
        test_stocks = get_select_stocks(my_select)
    else:
        test_stocks = get_select_stocks()

    result = detect_price_volume_reversal(test_stocks, start_date = START_DATE, n_years = N_YEARS)

    end_date = datetime.now().strftime("%Y%m%d")
    if ISMY_SELECT:
        filename =base_path / f'..\output\detect\detect_rev_BUY_{end_date}_my.xlsx'
    else:
        filename =base_path / f'..\output\detect\detect_rev_BUY_{end_date}.xlsx'

    log.info(f"检查成功检测数：{len(result)}")

    save_to_excel(result,filename)

# 每天（有空）执行检测
if __name__ == "__main__":
    #自选文件
    my_select= base_path / r"..\input\selectlist_my.xlsx"
    #回测N_YEARS年百分位
    N_YEARS = 3 
    #回测开始日期
    START_DATE = "20210701" 
    #PE数据来源，使用数据库速度快很多：数据库/Akshare  True/False
    IS_MYSQL = True
    #是否检测自选True/False
    ISMY_SELECT = False
    #是否对所有检测标的输出结果True/False
    ISALL = True

    PE_ROLLING_TIME = 3 #滚动PE百分位时间配置，默认5年
    PE_PERCENTILE = 5 #滚动PE百分位阈值配置，默认5%

    #检测自选股买点，只展现有买点可能性的标的
    # detect_with_lastPE(my_select)

    
    #检测自选股买点，呈现所有自选股的数据
    ALL_PE_DOC_NUM = 1 #当输出ALLPEexcel文档时，分割为几个文档输出
    detect_with_allPE(my_select)#通过PE来判断，主要用于成长股。

    #直接生成买点订单（回测使用）
    #IS_BUY = True
    #BUY_WITH_PE_PERCENTLE = False#通过PE和成交量判断买点
    #detect_with_buy(my_select)


    






    #my_select=r"..\input\selectlist_my.xlsx"
    ##是否检测自选True/False
    #ISMY_SELECT = True
    ##PE数据来源，使用数据库速度快很多：数据库/Akshare  True/False
    #IS_MYSQL = True

    #print(f"是否检测自选标的：{ISMY_SELECT}")
    #print(f"是否从数据库获取PE信息：{IS_MYSQL}")
    
    ##选定标的
    #if ISMY_SELECT:
    #    test_stocks = get_select_stocks(my_select)
    #else:
    #    test_stocks = get_select_stocks() 

    ## 执行检测 选取start_date开始日期数据，n_year内通过股价，交易额分位进行情绪判断买点，并给出标的和行业的估值参考
    ##result = detect_price_volume_reversal(test_stocks, start_date = "20230501", n_years=1) 
    #N_YEARS = 3
    #result = detect_price_volume_reversal(test_stocks, start_date = "20160501", n_years=N_YEARS)
    #end_date = datetime.now().strftime("%Y%m%d")
    #if ISMY_SELECT:
    #    filename = f'.\output\detect\detect_volume_reversal{end_date}_my.xlsx'
    #else:
    #    filename = f'.\output\detect\detect_volume_reversal{end_date}.xlsx'
    
    #print(f"检查成功检测数：{len(result)}")
    #save_to_excel_filter(result,test_stocks,filename)




