import pandas as pd
import numpy as np
import akshare as ak
from tqdm import tqdm
import insert2Mysql as ins


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
            df = ak.stock_index_pe_lg(name)
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

def get_pe_percentile(df: pd.DataFrame, testdate: str, year_window: int) -> float:
    """
    计算指定日期指数PE在历史数据中的时间百分位
    
    参数:
        df (pd.DataFrame): get_index_pe_his()返回的历史PE数据
        testdate (str): 查询日期 (YYYY-MM-DD格式)
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
    df = df.sort_values('日期').dropna(subset=['pe_ttm'])
    
    # 2. 确定时间窗口范围
    test_date = pd.to_datetime(testdate)
    start_date = test_date - pd.DateOffset(years=year_window)
    
    # 3. 提取窗口内数据
    window_mask = (df['日期'] >= start_date) & (df['日期'] <= test_date)
    if not window_mask.any():
        return np.nan
    
    window_df = df.loc[window_mask].copy()
    window_pe = window_df['pe_ttm'].values
    
    # 4. 获取测试日PE值
    testday_pe = window_df.loc[window_df['日期'] == test_date, 'pe_ttm'].values
    if len(testday_pe) == 0:
        # 如果测试日无数据，使用最近交易日的PE
        prev_days = window_df[window_df['日期'] < test_date]
        if prev_days.empty:
            return np.nan
        testday_pe = prev_days.iloc[-1]['pe_ttm']
    
    # 5. 计算百分位[3,4](@ref)
    sorted_pe = np.sort(window_pe)
    # 计算小于等于当前PE的数据点数量[5](@ref)
    count_below = np.searchsorted(sorted_pe, testday_pe, side='right')
    
    # 6. 计算百分位值[7](@ref)
    percentile = (count_below / len(sorted_pe)) * 100
    
    return  percentile[0].astype(float) # 保留两位小数

if __name__ == "__main__":
    # 使用示例
    #df = get_major_index_valuation()
    df = get_index_pe_his('沪深300')
    percent = get_pe_percentile(df, "20250704",3)
    #df = df.iloc[0:2]

    print(percent)
