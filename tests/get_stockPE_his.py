import pandas as pd
import akshare as ak
import pymysql
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from decimal import Decimal
import log4ak

# 配置日志
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

SELECT_SQL = """
        SELECT 
            target.pe,
            target.pe_ttm,
            SUM(IF(hist.pe_ttm <= target.pe_ttm, 1, 0)) / COUNT(*) AS percentile
        FROM stock_pe_history AS hist
        JOIN (
            SELECT pe, pe_ttm 
            FROM stock_pe_history
            WHERE stock_code = %s 
                AND trade_date = %s
            LIMIT 1
        ) AS target
        WHERE hist.stock_code = %s
            AND hist.trade_date BETWEEN %s AND %s
            AND hist.pe_ttm IS NOT NULL
        GROUP BY target.pe, target.pe_ttm
    """


def get_stock_pe_percentile(code: str, statyears: int, getdate: str = None) -> dict:
    """
    获取股票PE百分位(兼容MySQL严格模式)
     返回格式:
    {
        "status": "success" | "database_error" | "akshare_error",
        "code": "股票代码",
        "date": "YYYY-MM-DD",
        "pe": 0.00,              # 静态市盈率
        "pe_ttm": 0.00,          # 滚动市盈率(TTM)
        "percentile": 0.0000,     # PE_TTM历史百分位(0-1之间)
        "source": "database" | "akshare",
        "message": "操作描述信息"
    }
    """    
    try:
        with pymysql.connect(**DB_CONFIG) as conn:
            # 第一步：获取目标日期及PE值
            target_date = _get_target_date(conn, code, getdate)
            if not target_date:
                log.error(f"股票{code}在数据库中无有效数据,通过_get_from_akshare获取数据")
                return _get_from_akshare(code, statyears, getdate)
            log.info(f"数据库中存在{code}在{target_date}的PE数据")
                
            # 第二步：计算日期范围
            start_date = target_date - relativedelta(years=statyears)
            
            # 第三步：单次查询完成所有计算
            result = _fetch_pe_calculation(conn, code, target_date, start_date)
            log.info(f"{code}执行_fetch_pe_calculation成功,result={result}")

            if result:
                # 修复结果访问方式 - 使用字段名而非索引
                pe = result.get('pe') or result.get('current_pe')
                pe_ttm = result.get('pe_ttm') or result.get('current_pe_ttm')
                percentile = result.get('percentile')
                
            if percentile is not None:
                return {
                    'code': code,
                    'date': target_date.strftime('%Y-%m-%d'),
                    'pe': float(pe) if isinstance(pe, Decimal) else pe,
                    'pe_ttm': float(pe_ttm) if isinstance(pe_ttm, Decimal) else pe_ttm,
                    'percentile': round(float(percentile), 4) if isinstance(percentile, Decimal) else percentile,
                    'source': 'database'
                }
                
    except Exception as e:
        log.error(f"数据库查询失败: {e}")
    
    # 数据库无数据时调用akshare
    return _get_from_akshare(code, statyears, getdate)

def _get_target_date(conn, code, getdate):
    """获取目标查询日期"""
    cursor = conn.cursor()
    
    if getdate:
        # 处理YYMMDD格式日期
        target_date = datetime.strptime(getdate, "%Y%m%d").date()
        cursor.execute("""
            SELECT 1 FROM stock_pe_history 
            WHERE stock_code = %s AND trade_date = %s
            LIMIT 1
        """, (code, target_date))
        if cursor.fetchone():
            return target_date
        else:   
            log.info(f"数据库中无{code}在{target_date}的PE数据")
            return None
    
    # 未指定日期时获取最新日期
    log.info(f"{code}未指定日期时获取最新日期")
    cursor.execute("""
        SELECT MAX(trade_date) AS max_date 
        FROM stock_pe_history 
        WHERE stock_code = %s
    """, (code,))
    result = cursor.fetchone()
    max_date = result.get('max_date') if result else None
    log.info(f"{code}在数据库中最新日期：{max_date}")

    return max_date

def _fetch_pe_calculation(conn, code, target_date, start_date):
    """核心查询计算(MySQL严格模式兼容)"""
    cursor = conn.cursor()
    cursor.execute(SELECT_SQL, (code, target_date, code, start_date, target_date))
    log.info(f"成功执行SQL：{SELECT_SQL}")
    
    return cursor.fetchone()

def _get_from_akshare(code, statyears, getdate):
    """从akshare获取数据并计算"""

    # 初始化返回格式
    result_template = {
        "status": "success",
        "code": code,
        "date": "",
        "pe": None,
        "pe_ttm": None,
        "percentile": None,
        "source": "",
        "message": ""
    }

    result = result_template.copy()

    try:
        df = ak.stock_a_indicator_lg(symbol=code)

        if df.empty:
            result.update({
                "status": "akshare_error",
                "message": "AKShare返回空数据"
            })
            log.info("AKShare返回空数据")
            return result
            
        # 处理日期
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        log.info(f"通过AKShare获取到{len(df)}条估值数据")


        target_date = None
        if getdate in df["trade_date"]:
            target_date = datetime.strptime(getdate, "%Y%m%d").date()
            log.info(f"通过AKShare获取{code}在{getdate}的PE")
        else:
            target_date = df["trade_date"].max()
            log.info(f"getdate={getdate}无效，通过AKShare获取{code}最新{target_date}的PE")

        target_row = df[df["trade_date"] == target_date].iloc[0]
        
        # 计算历史范围
        start_date = target_date - relativedelta(years=statyears)
        history_df = df[
            (df["trade_date"] >= start_date) &
            (df["trade_date"] <= target_date) &
            (df["pe_ttm"].notnull())
        ]
        log.info(f"有效历史数据范围: {start_date} 至 {target_date}, 共{len(history_df)}条")
        
        # 计算百分位
        percentile = None
        if not history_df.empty:
            sorted_ttm = history_df["pe_ttm"].sort_values().values
            target_value = target_row["pe_ttm"]
            count_less_equal = sum(1 for x in sorted_ttm if x <= target_value)
            percentile = count_less_equal / len(sorted_ttm)
            
            log.info(f"AKShare计算完成 | PE_TTM: {target_value} | 百分位: {percentile}")
        else:
            log.info("历史数据不足，无法计算百分位")

        result = {
            'code': code,
            'date': target_date.strftime('%Y-%m-%d'),
            'pe': float(target_row["pe"]),
            'pe_ttm': float(target_row["pe_ttm"]),
            'percentile': round(percentile, 4) if percentile is not None else None,
            'source': 'akshare'
        }
        log.info(f"AKShare获取数据PE：{result}")
        
        return result
        
    except Exception as e:
        log.error(f"AKShare接口调用失败，获取{code}在{getdate}的PE计算失败: {e}")
        return None

# 测试用例
def test_pe_calculator():
    print("测试用例1: 指定日期查询")
    code="000333"
    getdate = "20240620"
    years=5
    result = get_stock_pe_percentile(code, years, getdate)
    print(f"{code}在{getdate}的{years}年PE百分位: {result}")

    print("测试用例2: 指定日期查询")
    code="000333"
    getdate = "20130620"
    years=5
    result = get_stock_pe_percentile(code, years, getdate)
    print(f"{code}在{getdate}的{years}年PE百分位: {result}")
    
    print("\n测试用例3: 最新日期查询")
    result = get_stock_pe_percentile("600519", 3)
    print(f"600519最新3年PE百分位: {result}")
    
    print("\n测试用例4: 无效股票代码")
    result = get_stock_pe_percentile("999999", 5)
    print(f"无效股票查询结果: {result}")

if __name__ == "__main__":
    test_pe_calculator()