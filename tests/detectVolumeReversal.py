import time
import numpy as np
import akshare as ak
import pandas as pd
import xlsxwriter
from tqdm import tqdm
from getAllStock import get_all_stocks, get_select_stocks
import commTools as ct
from datetime import datetime, timedelta
from get_industry_historyPE import get_industry_info

KEEPDAY = 3
DODUP = 0.15 #连续3个交易日放量增长率15%
GROUNDVOLUME = 0.05 #地量阈值检测时间内5%分位

MAX_CONSECUTIVE_ERRORS = 3  # 最大允许连续错误次数
OUTTIME = 5  # 接口长时间无返回报错
ISMY = False

def detect_price_volume_reversal(stock_list: pd.DataFrame, 
                          start_date: str = "20200101", 
                          n_years: int = 3) -> pd.DataFrame:
    """
    地量地价反转信号检测函数
    
    参数：
        stock_list : DataFrame，包含列["代码","名称"]
        start_date : str，检测起始日期（格式：YYYYMMDD）
        n_years    : int，历史数据回溯年限
    
    返回：
        DataFrame，包含符合条件的股票及信号特征：
            ["代码","名称","地量日期","反转日期","量能变化率(%)"]
    """
    # 初始化结果容器
    result = []
    
    # 获取当前日期
    #today = datetime.now().strftime("%Y%m%d") # 当前日期
    
    for idx, code in tqdm(enumerate(stock_list["代码"]), total=len(stock_list)):
        
        try:
            # 获取历史数据（需替换为实际数据接口）
            df = get_stock_data(code, start_date)  # 假设返回包含日期、成交额、收盘价的DataFrame
            # 添加数据有效性校验，避免停牌日零成交量的干扰
            hist_data = df[df['成交额'] > 0].copy()
            #按时间排序
            hist_data = hist_data.sort_index(ascending=True)
            
            # 计算n年历史分位（参考网页5/7的地量判断逻辑）
            rolling_window = n_years * 250  # 假设每年250个交易日

            #收盘价历史分位
            hist_data['price'] = hist_data['收盘'].rolling(rolling_window).apply(
                lambda x: x.rank(pct=True).iloc[-1], raw=False)
            # 收盘价检测地量条件
            hist_data['p_mask'] = hist_data['price'] < GROUNDVOLUME  # 收盘价处于近n年最低5%分位
            
            #成交额历史分位
            hist_data['volume'] = hist_data['成交额'].rolling(rolling_window).apply(
                lambda x: x.rank(pct=True).iloc[-1], raw=False)            
            # 成交额检测地量条件
            hist_data['v_mask'] = hist_data['volume'] < GROUNDVOLUME  # 成交额处于近n年最低5%分位
            
            # 检测连续量能递增（网页2/9的递增逻辑）
            hist_data['v_growth'] = hist_data['成交额'].pct_change() + 1
            #consecutive_growth = (hist_data['volume_growth']
            #                      .rolling(KEEPDAY).apply(lambda x: np.all(x >= (1+DODUP))))
            hist_data['vg_mask'] = hist_data['v_growth'].rolling(KEEPDAY).apply(lambda x: np.all(x >= (1+DODUP)))==1
            
            result.append(hist_data)
            #if len(result) == 0:
            #    result = hist_data.copy()
            #else:
            #    result.append(hist_data)

            # 生成信号（网页3/9的反转确认逻辑）
            #signal_dates = hist_data[hist_data['low_volume_mask'] & hist_data['consecutive_growth']].index
            
            #if len(signal_dates) > 0:
            #    first_signal = signal_dates[0]
            #    reversal_data = {
            #        "代码": code,
            #        "名称": name,
            #        "地量日期": hist_data[low_volume_mask].index[-1].strftime("%Y%m%d"),
            #        "反转日期": first_signal.strftime("%Y%m%d"),
            #        "量能变化率(%)": round((
            #            hist_data.loc[first_signal, '成交额'] / 
            #            hist_data[low_volume_mask]['成交额'].mean()) * 100, 2)
            #    }
            #    result.append(reversal_data)
            time.sleep(0.3)        
        except Exception as e:
            print(f"股票{code}数据处理异常: {str(e)}")
            time.sleep(0.3) 
            continue
            
    return result


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
    industry = get_industry_info(stock_code)

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

def save_to_excel(result: list, stock_list: pd.DataFrame, filename: str) -> None:
    """
    将检测结果按股票代码分Sheet保存到Excel
    
    参数：
        result      : detect_volume_reversal返回的结果列表
        stock_list  : 原始股票列表（含代码列）
        filename    : 输出Excel文件名（如"volume_signals.xlsx"）
    """
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # 遍历股票代码与对应的DataFrame
        for idx, code in enumerate(stock_list["代码"]):
            if idx < len(result) and not result[idx].empty:  # 确保索引不越界且数据非空
                # 提取当前股票的DataFrame
                df = result[idx].copy()
                df.index = df.index.strftime('%Y%m%d')  # 设置日期格式
                # 写入Excel，Sheet名使用股票代码（字符串格式避免特殊符号问题）
                df.to_excel(
                    writer, 
                    sheet_name=str(code), 
                    index=True,  # 保留日期索引
                    header=True
                )

    
def save_to_excel_filter(result: list, stock_list: pd.DataFrame, filename: str) -> None:
    """
    将检测结果按股票代码分Sheet保存到Excel
    优化后的存储函数（增加3个条件筛选）
    """
    sheets_to_write = []
    passNum = 0
    
    # 

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

            last_row = df.iloc[-1]

            testTrue = ISMY #默认配置False，调测时改为True使用，

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

                stock_pe,stock_pe_ttm = get_stock_pe(code)

                last_index = df.index[-1]  # 获取最后一行索引
                df.loc[last_index, 'industry_id'] = industry_id
                df.loc[last_index, 'industry_name'] = industry_name
                df.loc[last_index, 'pe_weighted'] = pe_weighted
                df.loc[last_index, 'pe_mean'] = pe_mean
                df.loc[last_index, 'pe_median'] = pe_median
                df.loc[last_index, 'stock_pe'] = stock_pe
                df.loc[last_index, 'stock_pe_ttm'] = stock_pe_ttm
                sheets_to_write.append((str(code), df))

                passNum += 1
                time.sleep(0.3) 
            
        except Exception as e:
            print(f"{code}获取行业或股票估值异常: {str(e)}")
            time.sleep(0.3) 
            continue
    
    # 无有效数据时跳过写入
    if not sheets_to_write:
        print("警告：无符合条件的股票数据，跳过Excel生成")
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
        print(f"错误：文件 {filename} 被其他程序占用，请关闭后重试")

    print(f"今天检测通过数量：{passNum}")


# 每天（有空）执行检测
if __name__ == "__main__":
    
    #print(get_stock_pe('600519'))

    my_select=r"..\input\selectlist_my.xlsx"
    #是否检测自选True/False
    ISMY = True

    print(f"是否检测自选标的：{ISMY}")
    
    #选定标的
    if ISMY:
        test_stocks = get_select_stocks(my_select)
    else:
        test_stocks = get_select_stocks() 

    # 执行检测 选取start_date开始日期数据，n_year内通过股价，交易额分位进行情绪判断买点，并给出标的和行业的估值参考
    #result = detect_price_volume_reversal(test_stocks, start_date = "20230501", n_years=1) 
    result = detect_price_volume_reversal(test_stocks, start_date = "20160501", n_years=3)
    end_date = datetime.now().strftime("%Y%m%d")
    if ISMY:
        filename = f'.\output\detect\detect_volume_reversal{end_date}_my.xlsx'
    else:
        filename = f'.\output\detect\detect_volume_reversal{end_date}.xlsx'
    
    print(f"检查成功检测数：{len(result)}")
    save_to_excel_filter(result,test_stocks,filename)




