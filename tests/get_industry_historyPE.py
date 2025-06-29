import akshare as ak
import numpy as np
import pandas as pd
from check_historical_year_pe import get_historical_year_pe
from typing import Tuple, Dict, Any, Optional
import pymysql
from datetime import datetime
import log4ak


log = log4ak.LogManager(log_level=log4ak.INFO)# 日志配置


# 数据库配置（适配PyMySQL参数）
DB_CONFIG = {
    'host': 'localhost',
    'user': 'powerbi',
    'password': 'longyu',
    'database': 'akshare',
    'port': 3306,
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}


def get_industry_info(stock_code: str) -> pd.DataFrame:
    """获取指定股票的最新行业分类信息（申万标准）"""
    # 参数标准化
    stock_code = f"{stock_code:0>6}"
    
    try:
        # 获取行业变动全量数据（动态设置时间范围）
        end_date = datetime.now().strftime("%Y%m%d")
        df = ak.stock_industry_change_cninfo(
            symbol=stock_code,
            start_date="20000101",  # 足够早的起始日期
            end_date=end_date
        )
        
        # 字段验证（根据网页1接口返回结构）
        required_columns = ['证券代码', '行业编码', '行业门类', '行业大类', '分类标准编码', '变更日期']
        if not set(required_columns).issubset(df.columns):
            missing = set(required_columns) - set(df.columns)
            raise KeyError(f"缺失关键字段: {missing}")

        # 申银万国行业分类标准
        #df = ak.stock_industry_category_cninfo(symbol="申银万国行业分类标准")
        #"证监会行业分类标准": "008001",
        #"巨潮行业分类标准": "008002",
        #"申银万国行业分类标准": "008003",
        #"新财富行业分类标准": "008004",
        #"国资委行业分类标准": "008005",
        #"巨潮产业细分标准": "008006",
        #"天相行业分类标准": "008007",
        #"全球行业分类标准": "008008",
        # 筛选逻辑优化
        industry = df[
            (df['证券代码'] == stock_code) & 
            (df['分类标准编码'] == "008001")  # 008001对应 证监会行业分类标准
        ].sort_values('变更日期', ascending=False)  # 按变更日期倒序
        
        if industry.empty:
            raise ValueError(f"未找到股票{stock_code}证监会行业分类标准")

        # 取最新记录（根据网页1的变更日期字段）
        latest_record = industry.head(1)
        
        return latest_record[['证券代码', '行业编码', '行业门类', '行业大类']]
        
    except KeyError as e:
        log.error(f"字段缺失异常: {str(e)}")
        return pd.DataFrame()
    except IndexError:
        log.error(f"未找到股票代码 {stock_code} 对应的行业")
        return pd.DataFrame()
    except Exception as e:
        log.error(f"申银万国行业分类查询失败: {str(e)}")
        return pd.DataFrame()


def get_sw_industry_pe_history(symbol="I64", start_year="2023"):
    """获取历史年度PE估值数据（证监会行业分类）"""
    """只能获取2023年之后的数据"""
    try:
        # 获取当前年份
        current_year = datetime.now().year
        last_year = current_year - 1  # 截止去年[2](@ref)
        
        # 获取完整交易日历（参考网页3接口）
        trade_dates = ak.tool_trade_date_hist_sina()["trade_date"].tolist()
        
        results = []
        for year in range(int(start_year), last_year + 1):
            try:
                # 筛选年度最后交易日（参考网页6逻辑）
                year_dates = [d for d in trade_dates if int(d.year)==year]
                if not year_dates:
                    continue
                lastday = sorted(year_dates)[-1]  # 取年度最后交易日

                #print(f"{year}年最后交易日：{lastday}")
                
                # 获取行业PE数据（网页2接口）
                pe_df = ak.stock_industry_pe_ratio_cninfo(
                    symbol="证监会行业分类",
                    date=f"{lastday:%Y%m%d}" # 转换为YYYYMMDD格式
                )
                
                # 数据筛选与格式化（网页2字段结构）
                industry_data = pe_df[pe_df["行业编码"] == symbol].rename(columns={
                    '变动日期': '日期',
                    '静态市盈率-加权平均': 'PE静-加权',
                    '静态市盈率-中位数': 'PE静-中位',
                    '静态市盈率-算术平均': 'PE静-平均'
                })
                
                if not industry_data.empty:
                    results.append(industry_data[["日期","行业编码","PE静-加权","PE静-中位","PE静-平均"]])
                    
            except Exception as e:
                log.error(f"{year}年数据获取失败: {str(e)}")
                continue
        
        return pd.concat(results).reset_index(drop=True) if results else pd.DataFrame()
    
    except Exception as e:
        log.error(f"主程序异常: {str(e)}")
        return pd.DataFrame()

def get_stock_industry_valuation(stock_code, start_year="2023"):
    """一站式查询股票行业及历史估值"""
    # 获取行业
    industry = get_industry_info(stock_code)
    
    # 获取估值
    valuation_df = get_sw_industry_pe_history(industry["行业编码"].values[0],start_year)

    #stockPE=get_historical_year_pe(stock_code, start_year="2023")
    
    # 数据增强：添加百分位分析
    #if not valuation_df.empty:
    #    for metric in ['PE', 'PB']:
    #        valuation_df[f'{metric}_百分位'] = valuation_df[metric].rank(pct=True)
    
    return industry, valuation_df#, stockPE

def get_industry_pe_mysql_new_conn(industry_code: str) -> pd.DataFrame:
    """
    从MySQL数据库获取行业的历史PE数据
    
    参数:
        conn: 数据库连接对象
        industry_code: 行业代码
    
    返回:
        包含行业历史PE数据的DataFrame
    """

    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            sql = """
                SELECT 
                    trade_date AS `日期`,
                    pe_weighted AS `PE静-加权`,
                    pe_median AS `PE静-中位`,
                    pe_mean AS `PE静-平均`
                FROM industry_pe_history
                WHERE industry_code = %s
                ORDER BY trade_date ASC
            """
            cursor.execute(sql, (industry_code,))
            results = cursor.fetchall()
            
            if not results:
                log.error(f"未找到行业 {industry_code} 的PE历史数据")
                return pd.DataFrame()
            
            # 转换为DataFrame
            df = pd.DataFrame(results)
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
            log.info(f"找到行业 {industry_code} 的PE历史数据{len(df)}条")
            return df
    
    except Exception as e:
        log.error(f"获取行业PE历史数据失败: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def get_industry_pe_mysql(industry_code: str,conn) -> pd.DataFrame:
    """
    从MySQL数据库获取行业的历史PE数据
    
    参数:
        conn: 数据库连接对象
        industry_code: 行业代码
    
    返回:
        包含行业历史PE数据的DataFrame
    """
    try:
        with conn.cursor() as cursor:
            sql = """
                SELECT 
                    trade_date,
                    pe_weighted AS `PE静-加权`,
                    pe_median AS `PE静-中位`,
                    pe_mean AS `PE静-平均`
                FROM industry_pe_history
                WHERE industry_code = %s
                ORDER BY trade_date ASC
            """
            cursor.execute(sql, (industry_code,))
            results = cursor.fetchall()
            
            if not results:
                log.error(f"未找到行业 {industry_code} 的PE历史数据")
                return pd.DataFrame()
            
            # 转换为DataFrame
            df = pd.DataFrame(results)
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df.set_index('trade_date', inplace=True)
            
            return df
    
    except Exception as e:
        log.error(f"获取行业PE历史数据失败: {e}")
        return pd.DataFrame()

def get_stock_industry_pe_mysql(stock_code: str) -> Dict[str, Any]:
    """
    一站式获取股票对应的行业及其历史PE数据（数据库版）
    
    参数:
        stock_code: 6位股票代码
    
    返回:
        包含股票信息和行业历史PE数据的字典
    """
    # 标准化股票代码
    stock_code = f"{stock_code:0>6}"
    
    result = {
        "status": "success",
        "code": stock_code,
        "industry_info": None,
        "pe_history": pd.DataFrame(),
        "message": ""
    }
    
    try:
        # 连接数据库
        conn = pymysql.connect(**DB_CONFIG)
        
        # 1. 获取股票行业信息
        industry_info = get_industry_info(stock_code)
        if industry_info.empty:
            result.update({
                "status": "error",
                "message": "未找到行业分类信息"
            })
            return result
            
        result["industry_info"] = industry_info
        
        # 2. 获取行业历史PE
        pe_history = get_industry_pe_mysql(industry_info['行业编码'].values[0], conn)
        result["pe_history"] = pe_history
        
        # 添加统计信息
        if not pe_history.empty:
            last_record = pe_history.iloc[-1].to_dict()
            result.update({
                "latest_pe_weighted": last_record['PE静-加权'],
                "latest_pe_median": last_record['PE静-中位'],
                "years_of_data": pe_history.index.year.nunique(),
                "min_pe_median": pe_history['PE静-中位'].min(),
                "max_pe_median": pe_history['PE静-中位'].max(),
                "current_pe_percentile": (pe_history['PE静-中位'] <= last_record['PE静-中位']).mean()
            })
        
        conn.close()
        return result
        
    except Exception as e:
        log.error(f"一站式查询失败: {e}")
        return {
            "status": "error",
            "code": stock_code,
            "message": f"数据库操作失败: {str(e)}"
        }


def df_to_excel(df):

    # 导出Excel并自动调整列宽[4](@ref)
    with pd.ExcelWriter(".\output\output.xlsx") as writer:
        df.to_excel(writer, sheet_name="全量数据")
        #worksheet = writer.sheets["全量数据"]
        #if not df.columns.empty:
        #    for idx, col in enumerate(df.columns):
        #        worksheet.set_column(idx, idx, 20)  # 设置列宽为20字符
    return 

def test_get_stock_industry_pe_mysql() -> None:
    # 测试获取行业PE历史数据
    print("\n测试1: 获取贵州茅台行业历史PE")
    result_600519 = get_stock_industry_pe_mysql("600519")
    print(f"状态: {result_600519['status']}")
    print(f"行业信息: {result_600519['industry_info']}")
    print(f"PE历史数据样本:\n{result_600519['pe_history'].head(3) if not result_600519['pe_history'].empty else '无数据'}")
    
    print("\n测试2: 获取宁德时代行业历史PE")
    result_300750 = get_stock_industry_pe_mysql("300750")
    print(f"状态: {result_300750['status']}")
    print(f"行业信息: {result_300750['industry_info']}")
    print(f"PE历史数据条数: {len(result_300750['pe_history'])}")
    
    print("\n测试3: 无效股票代码")
    result_invalid = get_stock_industry_pe_mysql("999999")
    print(f"状态: {result_invalid['status']}")
    print(f"消息: {result_invalid['message']}")

if __name__ == "__main__":
    # 示例：查询贵州茅台行业
    #df = get_industry_info("000858")
    #df = ak.stock_industry_pe_ratio_cninfo("证监会行业分类","20241223")
    #df_to_excel(df)

    test_get_stock_industry_pe_mysql()


    #df = get_stock_industry_valuation("000858")
    
    #print(f"{df}")


    #industry, valuation_data = get_stock_industry_valuation("600519")
    #print(f"\n{industry}历史估值数据：")
    #print(valuation_data.tail())