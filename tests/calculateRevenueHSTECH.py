import akshare as ak
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import time

# ================== 全局配置 ==================
CACHE_EXPIRY = 3600  # 成分股缓存有效期1小时
RETRY_TIMES = 1  # 接口重试次数

HSTECH_STOCKS=[#恒生指数公司官网（https://www.hsi.com.hk）：查看“恒生科技指数”最新季检公告，通常在3月、6月、9月、12月发布。
    '01810',
    '09999',
    '00700',
    '09988',
    '01211',
    '03690',
    '09618',
    '00981',
    '01024',
    '02015',
    '09961',
    '09868',
    '09888',
    '00992',
    '06690',
    '02382',
    '09626',
    '06618',
    '00268',
    '00300',
    '00020',
    '03888',
    '00241',
    '00780',
    '00285',
    '01347',
    '00522',
    '09866',
    '09660',
    '01698',
    ]

# ================== 成分股获取优化 ==================
def get_hstech_components(use_cache=True)-> pd.DataFrame:
    """
    获取恒生科技指数成分股（动态接口+静态备份）
    :param use_cache: 是否使用缓存，默认为True
    :return: 成分股DataFrame
    :rtype: pandas.DataFrame  
    """
    # 缓存检查
    if use_cache and hasattr(get_hstech_components, 'cache'):
        time_diff = datetime.now() - get_hstech_components.cache_time
        if time_diff.total_seconds() < CACHE_EXPIRY:
            return get_hstech_components.cache.copy()
    
    try:
        # 动态获取最新成分股
        components = ak.index_stock_cons(symbol="HSTECH")
        # 字段标准化
        components = components.rename(columns={
            '品种代码': '代码',
            '成分股代码': '代码',
            '品种名称': '名称'
        })[['代码', '名称']]
        
        # 缓存更新
        get_hstech_components.cache = components
        get_hstech_components.cache_time = datetime.now()
        return components
    
    except Exception as e:
        print(f"接口获取失败: {e}, 启用静态数据")
        # 2025年5月最新成分股（含腾讯音乐、地平线机器人）
        return pd.DataFrame({'代码': HSTECH_STOCKS})

# ================== 财务数据获取优化 ==================
def fetch_financial_data(code, report_year)-> dict:
    """
    使用港股专用接口获取财务数据
    :param code: 港股代码
    :param report_year: 财报年份
    :return: 财务数据字典
    :rtype: dict    
    """
    for _ in range(RETRY_TIMES):
        try:
            # 使用港股财务接口
            df = ak.stock_financial_hk_report_em(stock=code,
                symbol="利润表",  # 利润表
                indicator="年度"        # 年度报告
            )
            
            # 筛选指定年份数据（港股年报实际披露日期为次年3-4月）
            report_date = f"{report_year}-12-31 00:00:00"  # 假设年报在次年3月31日披露
            
            # 获取revenue关键指标
            filtered_df = df[(df["STD_ITEM_NAME"].isin(["营业额","经营收入总额", "经营溢利"])) & (df["REPORT_DATE"].isin([f"{report_year}-12-31 00:00:00",f"{report_year}-03-31 00:00:00"]))]
                             
            # 定义可能存在的revenue列名集合
            revenue_columns = ["营业额", "经营收入总额", "营业收入"]
            revenue = 0

            # 动态检测有效列，并赋值
            for col in revenue_columns:
                    if  not filtered_df[filtered_df["STD_ITEM_NAME"] == col].empty:
                        revenue=filtered_df[filtered_df["STD_ITEM_NAME"] == col]["AMOUNT"].values[0]
                        break

            profit = filtered_df[filtered_df["STD_ITEM_NAME"] == "经营溢利"]["AMOUNT"].values[0]

            stockName = filtered_df[filtered_df["STD_ITEM_NAME"] == "经营溢利"]["SECURITY_NAME_ABBR"].values[0]

            return {
                    '简称': stockName,
                    '营业收入': revenue,
                    '净利润': profit
            }
            
        except Exception as e:
            print(f"重试{code}: {str(e)}")
            time.sleep(1)
    
    return {'简称': None,'营业收入': None, '净利润': None}

def get_hk_financials(report_year=2024) -> pd.DataFrame:
    """
    获取全成分股财务数据（含进度条和缓存）
    :param report_year: 财报年份，默认为2024
    :return: 成分股财务数据DataFrame 
    :rtype: pandas.DataFrame
    """
    # 报告期验证
    if not (2019 < report_year < datetime.now().year):
        raise ValueError("报告年份不合法")
    
    # 获取成分股
    components = get_hstech_components()
    
    # 并发获取数据（建议使用线程池优化）
    financials = []
    for code in tqdm(components['代码'], desc="获取财务数据"):
        data = fetch_financial_data(code, report_year)
        financials.append({
            '代码': code,
            '简称' : data['简称'],
            '报告期': f"{report_year}-12-31",
            '营收(亿元)': data['营业收入'] / 1e8 if data['营业收入'] else None,  # 转换为亿元
            '净利润(亿元)': data['净利润'] / 1e8 if data['净利润'] else None
        })
    
    # 合并数据
    result_df = components.merge(pd.DataFrame(financials), on='代码')
    
    # 数据校验
    missing = result_df[['营收(亿元)', '净利润(亿元)']].isnull().sum()
    print(f"\n缺失数据统计：\n{missing}")
    
    return result_df.dropna(subset=['简称','营收(亿元)', '净利润(亿元)'])

# ================== 分析功能增强 ==================
def analyze_hstech(report_years=[2023, 2024])-> pd.DataFrame:
    """
    多维度对比分析
    :param report_years: 报告年份列表，默认为[2023, 2024]
    :return: 分析结果DataFrame 
    :rtype: pandas.DataFrame
    """
    analysis = []
    for year in report_years:
        df = get_hk_financials(year)
        if not df.empty:
            # 核心指标计算
            total_rev = df['营收(亿元)'].sum()
            total_profit = df['净利润(亿元)'].sum()
            rev_growth = df['营收(亿元)'].pct_change().mean() * 100
            profit_margin = (df['净利润(亿元)'] / df['营收(亿元)']).mean() * 100
            
            # TOP5企业贡献度
            top5_rev = df.nlargest(5, '营收(亿元)')['营收(亿元)'].sum() / total_rev * 100
            
            analysis.append({
                '报告期': f"{year}年报",
                '总营收(亿元)': round(total_rev, 2),
                '营收增速(%)': round(rev_growth, 1),
                '净利率(%)': round(profit_margin, 1),
                'TOP5贡献度(%)': round(top5_rev, 1),
                '样本数': len(df)
            })
    
    return pd.DataFrame(analysis)

# ================== 执行主程序 ==================
if __name__ == "__main__":
    # 获取最新财务数据
    report_year=2024
    df = get_hk_financials(report_year)
    print("\n2024年数据样本：")
    print(df[['代码', '简称','营收(亿元)', '净利润(亿元)']].head())
    
    # 对比分析
    #comparison = analyze_hstech([2023, 2024])
    #print("\n年度对比分析：")
    #print(comparison)
    
    # 保存结果
    save_path = f".\output\HSTECH_核心财务数据_{report_year}.xlsx"
    with pd.ExcelWriter(save_path) as writer:
        df.to_excel(writer, sheet_name='明细数据', index=False)
        #comparison.to_excel(writer, sheet_name='对比分析', index=False)
    print(f"\n数据已保存至：{save_path}")