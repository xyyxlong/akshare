import os
from pathlib import Path
import pandas as pd
import numpy as np
import akshare as ak
from tqdm import tqdm
import insert2Mysql as ins
import log4ak
from safeLgAPI import safe_stock_index_pe_lg, _stock_index_pe_lg_fixed

base_path = Path(__file__).parent #系统绝对目录
log = log4ak.LogManager(log_level=log4ak.INFO)# 日志配置

PE_STATIC = 'pe_static'
PE_TTM = 'pe_ttm'
PE_STATIC_MEDIAN = 'pe_static_median'
PE_TTM_MEDIAN = 'pe_ttm_median'
PE_EQUAL_WEIGHT_STATIC = 'pe_equal_weight_static'
PE_EQUAL_WEIGHT_STATIC = 'pe_equal_weight_ttm'


INSERT_SQL ="""
    INSERT IGNORE INTO `index_valuation_history` 
    (`index_code`, `index_name`, `trade_date`, `index_value`, 
    `pe_equal_weight_static`, `pe_static`, `pe_static_median`,
     `pe_equal_weight_ttm`, `pe_ttm`, `pe_ttm_median`)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

def get_major_index_valuation():
    """获取三大指数估值数据"""
    indices = {
        "上证50": "000016.SH",
        "沪深300": "000300.SH",
        "上证380": "000009.SH",
        "创业板50": "399673.SZ",
        "中证500": "000905.SH",
        "上证180": "000010.SH",
        "深证红利": "399324.SZ",
        "深证100": "399330.SZ",
        "中证1000": "000852.SH",
        "上证红利": "000015.SH",
        "中证100": "000903.SH",
        "中证800": "000906.SH"
               }
    for name,code in tqdm(indices.items(),desc=f"获取宽基指数估值……"):
        print(f"......查询{name} PE")
        try:
            # 获取基础估值数据[5,9](@ref)
            df = _stock_index_pe_lg_fixed(name)
            df["指数代码"]=code
            df["指数名称"]=name

            # 生成批量数据（需要根据具体存储表来修改代码）
            batch_data = [
                        (row['指数代码'],row['指数名称'],row['日期'],row['指数'], 
                         None if pd.isna(row['等权静态市盈率']) else float(row['等权静态市盈率']),
                         None if pd.isna(row['静态市盈率']) else float(row['静态市盈率']),
                         None if pd.isna(row['静态市盈率中位数']) else float(row['静态市盈率中位数']),                         
                         None if pd.isna(row['等权滚动市盈率']) else float(row['等权滚动市盈率']),
                         None if pd.isna(row['滚动市盈率']) else float(row['滚动市盈率']),
                         None if pd.isna(row['滚动市盈率中位数']) else float(row['滚动市盈率中位数']))
                        for _, row in df.iterrows()
                        ]
            ins.insert_to_mysql(batch_data,INSERT_SQL)
        except Exception as e:
            print(f"{name} 数据存储失败: {str(e)}")
    
    return "insert finished!"

def get_index_pe_his(index_name: str) -> pd.DataFrame:
    """
    查询指定股票所有历史PE数据
    
    返回格式:
    DataFrame包含列: 
        trade_date (str): 交易日期 (YYYY-MM-DD格式)
        pe (float): 静态市盈率
        pe_ttm (float): 滚动市盈率(TTM)
        
    无数据时返回空DataFrame
    """

    HISTORY_SQL = """
        SELECT 
            trade_date AS `日期`, 
            index_code,index_name,index_value,
            pe_equal_weight_static,pe_static,pe_static_median,
            pe_equal_weight_ttm,pe_ttm,pe_ttm_median  
        FROM index_valuation_history 
        WHERE index_name = %s 
        ORDER BY trade_date
    """
    df = ins.getdata_fetchall(HISTORY_SQL,(index_name,))
    return df

def get_pe_percentile(df: pd.DataFrame, testdate: str, pename: str,year_window: int) -> float:
    """
    计算指定日期指数PE在历史数据中的时间百分位
    
    参数:
        df (pd.DataFrame): get_index_pe_his()返回的历史PE数据
        testdate (str): 查询日期 (YYYY-MM-DD格式)
        pename (str): PE列名（如'pe_ttm'）
        year_window (int): 回测时间窗口(年数)
        
        
    返回:
        float: PE时间百分位值(0-100之间)
        
    异常处理:
        无数据时返回np.nan
    """
    # 1. 数据预处理
    # 复制数据避免修改原DataFrame
    df = df.copy()
    
    # 转换日期格式并过滤有效数据
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values('日期').dropna(subset=[pename])
    
    # 2. 确定时间窗口范围
    test_date = pd.to_datetime(testdate)
    start_date = test_date - pd.DateOffset(years=year_window)
    
    # 3. 提取窗口内数据
    window_mask = (df['日期'] >= start_date) & (df['日期'] <= test_date)
    if not window_mask.any():
        return np.nan
    
    window_df = df.loc[window_mask].copy()
    window_pe = window_df[pename].values
    
    # 4. 获取测试日PE值
    testday_pe = window_df.loc[window_df['日期'] == test_date, pename].values
    if len(testday_pe) == 0:
        # 如果测试日无数据，使用最近交易日的PE
        prev_days = window_df[window_df['日期'] < test_date]
        if prev_days.empty:
            return np.nan
        testday_pe = prev_days.iloc[-1][pename]
    
    # 5. 计算百分位[3,4](@ref)
    sorted_pe = np.sort(window_pe)
    # 计算小于等于当前PE的数据点数量[5](@ref)
    count_below = np.searchsorted(sorted_pe, testday_pe, side='right')
    
    # 6. 计算百分位值[7](@ref)
    percentile = (count_below / len(sorted_pe)) * 100
    
    return  percentile[0].astype(float) # 保留两位小数

def get_pe_percentile_list(df: pd.DataFrame, pename: str,year_window: int) -> list:
    """
    计算指定PE列在历史数据中的时间百分位列表
    参数:
        df (pd.DataFrame): get_index_pe_his()返回的历史PE数据
        pename (str): PE列名（如'pe_ttm'）
        year_window (int): 回测时间窗口(年数)   
    返回:
        list: 每个交易日的PE时间百分位值列表
    """
    percentile_list = []
    # 数据预处理
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values('日期').dropna(subset=[pename])
    dates = df['日期'].tolist()
    pes = df[pename].tolist()

    for i, test_date in enumerate(dates):
        start_date = test_date - pd.DateOffset(years=year_window)
        days = (test_date - start_date).days
        # 取窗口内数据
        window_mask = (df['日期'] >= start_date) & (df['日期'] <= test_date)
        window_df = df.loc[window_mask]
        window_pe = window_df[pename].values

        # 如果窗口长度小于year_window对应的天数，赋值为nan
        winlen= len(window_pe)
        if winlen < 240 * year_window:  # 250为一年交易日数
            percentile_list.append(np.nan)
            continue

        testday_pe = pes[i]
        sorted_pe = np.sort(window_pe)
        count_below = np.searchsorted(sorted_pe, testday_pe, side='right')
        percentile = (count_below / len(sorted_pe)) * 100
        percentile_list.append(round(percentile, 2))

    return percentile_list

def get_pe_percentile_list(df: pd.DataFrame, penamelist: list[str], year_window: int) -> pd.DataFrame:
    """
    计算指定PE列在历史数据中的时间百分位列表
    参数:
        df (pd.DataFrame): get_index_pe_his()返回的历史PE数据
        penamelist (list[str]): PE列名列表（如['pe_ttm', 'pe_static']）
        year_window (int): 回测时间窗口(年数)   
    返回:
        pd.DataFrame: 包含每个PE列的时间百分位列表
    """
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values('日期')
    result = pd.DataFrame({'日期': df['日期']})

    for pename in penamelist:
        if pename not in df.columns:
            result[pename + '_percentile'] = np.nan
            continue
        pes = df[pename].tolist()
        percentile_list = []
        for i, test_date in enumerate(df['日期']):
            start_date = test_date - pd.DateOffset(years=year_window)
            window_mask = (df['日期'] >= start_date) & (df['日期'] <= test_date)
            window_df = df.loc[window_mask]
            window_pe = window_df[pename].dropna().values

            # 如果窗口长度小于year_window对应的交易日数，赋值为nan
            if len(window_pe) < 240 * year_window:
                percentile_list.append(np.nan)
                continue

            testday_pe = pes[i]
            sorted_pe = np.sort(window_pe)
            count_below = np.searchsorted(sorted_pe, testday_pe, side='right')
            percentile = (count_below / len(sorted_pe)) * 100
            percentile_list.append(round(percentile, 2))
        result[pename + '_percentile'] = percentile_list

    return result

if __name__ == "__main__":
    # 使用示例
    df = get_major_index_valuation()
    
    # df = get_index_pe_his('沪深300')
    # year_window = 5 # 回测时间窗口为5年
    # pename = 'pe_ttm' #pe_static/pe_ttm/pe_static_median/pe_ttm_median/pe_equal_weight_static/pe_equal_weight_ttm
    
    # percent = '{:.2f}'.format(get_pe_percentile(df, "20251210",pename, year_window))
    # print(df.iloc[-1])
    # print(f"{pename} {year_window} year percent: {percent}%")
    
    # percentlist= get_pe_percentile_list(df, pename, year_window)
    # df['percentile'] = percentlist
    # filename=base_path / f'..\output/index_pe_{pename}_percentile.xlsx'
    # df.to_excel(filename, index=False)
    # print(df[['日期', pename, 'percentile']].tail(10))
    
    # penamelist = ['pe_static', 'pe_ttm', 'pe_static_median', 'pe_ttm_median', 'pe_equal_weight_static', 'pe_equal_weight_ttm']
    # percent_df = get_pe_percentile_list(df, penamelist, year_window)
    # # 合并原始df和百分位df
    # merged_df = pd.concat([df.reset_index(drop=True), percent_df.drop(columns=['日期']).reset_index(drop=True)], axis=1)
    # merged_df.to_excel(base_path / f'..\output/index_pe_with_percentile_{year_window}year.xlsx', index=False)
    # print(merged_df.tail(10))
    
