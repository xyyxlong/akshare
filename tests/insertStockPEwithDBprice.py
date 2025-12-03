import os
from pathlib import Path
import pymysql
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from getAllStock import get_all_stocks, get_select_stocks

import log4ak

base_path = Path(__file__).parent #系统绝对目录
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

class StockValuationCalculator:
    def __init__(self, db_config: Dict):
        """
        初始化估值计算器
        Args:
            db_config: 数据库连接配置
        """
        self.db_config = db_config
        self.connection = None
        self.dividend_cache = {}  # 添加分红数据缓存
        
    def connect_to_database(self):
        """建立数据库连接"""
        try:
            self.connection = pymysql.connect(**self.db_config)
            log.debug(f"数据库相关信息：{self.db_config}")
            
            log.info(f"✅ 连接成功 | MySQL版本:{self.connection.get_server_info()}")
            log.info("成功连接到数据库")
        except Exception as e:
            log.error(f"数据库连接失败: {e}")
            raise
            
    def disconnect_from_database(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            log.info("数据库连接已关闭")
    
    def get_stock_codes(self) -> List[str]:
        """
        获取需要计算估值的股票代码列表
        Returns:
            股票代码列表
        """
        try:
            with self.connection.cursor() as cursor:
                sql = "SELECT DISTINCT stock_code FROM stock_historical_data ORDER BY stock_code"
                cursor.execute(sql)
                result = cursor.fetchall()
                stock_codes = [item['stock_code'] for item in result]
                log.info(f"获取到 {len(stock_codes)} 只股票代码")
                return stock_codes
        except Exception as e:
            log.error(f"获取股票代码失败: {e}")
            return []
    
    def get_historical_price_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取指定股票的历史价格数据
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
        Returns:
            包含历史价格数据的DataFrame
        """
        try:
            with self.connection.cursor() as cursor:
                sql = """
                SELECT date, close, stock_code 
                FROM stock_historical_data 
                WHERE stock_code = %s AND date BETWEEN %s AND %s 
                ORDER BY date
                """
                cursor.execute(sql, (stock_code, start_date, end_date))
                result = cursor.fetchall()
                
                if not result:
                    return pd.DataFrame()
                
                df = pd.DataFrame(result)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                return df
        except Exception as e:
            log.error(f"获取股票 {stock_code} 价格数据失败: {e}")
            return pd.DataFrame()
        
    def get_dividend_data(self, stock_code: str) -> pd.DataFrame:
        """
        获取指定股票的所有分红数据
        Args:
            stock_code: 股票代码
        Returns:
            包含分红数据的DataFrame
        """
        # 检查缓存
        if stock_code in self.dividend_cache:
            return self.dividend_cache[stock_code]
            
        try:
            with self.connection.cursor() as cursor:
                sql = """
                SELECT stock_code, equity_reg_date, cash_dividend, progress
                FROM dividend_info 
                WHERE stock_code = %s AND cash_dividend > 0 
                AND progress = '实施'
                ORDER BY equity_reg_date
                """
                cursor.execute(sql, (stock_code,))
                result = cursor.fetchall()
                
                if not result:
                    log.debug(f"股票 {stock_code} 没有分红数据")
                    return pd.DataFrame()
                
                df = pd.DataFrame(result)
                df['equity_reg_date'] = pd.to_datetime(df['equity_reg_date'])
                
                # 计算每股分红（将每10股分红转换为每股分红）
                df['dividend_per_share'] = df['cash_dividend'] / 10.0
                
                # 缓存数据
                self.dividend_cache[stock_code] = df
                
                return df
                
        except Exception as e:
            log.error(f"获取股票 {stock_code} 分红数据失败: {e}")
            return pd.DataFrame()
    
    def get_previous_year_dividend(self, stock_code: str, current_date: datetime) -> float:
        """
        获取指定日期所在财年的上一财年每股现金分红
        规则：下一年1月的分红属于当前财年
        Args:
            stock_code: 股票代码
            current_date: 当前日期
        Returns:
            上一财年每股分红金额，如果未找到返回0.0
        """
        try:
            current_year = current_date.year
            current_month = current_date.month
            
            # 确定财年：下一年1月的分红属于当前财年
            # 例如：2025年1月的分红属于2024财年
            fiscal_year = current_year - 1  # 默认上一财年就是去年
            
            with self.connection.cursor() as cursor:
                # 查找指定财年的分红数据，包括下一年1月的分红
                sql = """
                SELECT SUM(cash_dividend) as total_cash_dividend
                FROM dividend_info 
                WHERE stock_code = %s 
                AND progress = '实施'
                AND (
                    (YEAR(equity_reg_date) = %s AND MONTH(equity_reg_date) != 1) OR
                    (YEAR(equity_reg_date) = %s AND MONTH(equity_reg_date) = 1)
                )
                AND cash_dividend > 0 
                ORDER BY equity_reg_date DESC
                """
                # %s1: stock_code, %s2: fiscal_year, %s3: fiscal_year + 1
                cursor.execute(sql, (stock_code, fiscal_year, fiscal_year + 1))
                result = cursor.fetchone()
                
                if result and result['total_cash_dividend'] is not None:
                    # 将每10股分红转换为每股分红，并返回总和
                    return result['total_cash_dividend'] / 10.0
                else:
                    return 0.0
                    
        except Exception as e:
            log.error(f"获取股票 {stock_code} 上一财年分红数据失败: {e}")
            return 0.0
    
    def get_main_profit_data(self, stock_code: str, report_date: str) -> Optional[float]:
        """
        获取指定报告期的主营业务利润
        Args:
            stock_code: 股票代码
            report_date: 报告日期
        Returns:
            主营业务利润，如果未找到返回None
        """
        try:
            with self.connection.cursor() as cursor:
                sql = """
                SELECT main_profit FROM stock_financial_reports 
                WHERE stock_code = %s AND report_date = %s
                LIMIT 1
                """
                cursor.execute(sql, (stock_code, report_date))
                result = cursor.fetchone()
                return result['main_profit'] if result and result['main_profit'] else None
        except Exception as e:
            log.error(f"获取股票 {stock_code} 主营业务利润失败: {e}")
            return None
       
    def get_previous_fiscal_year_report(self, stock_code: str, as_of_date: str) -> Optional[Dict]:
        """
        获取指定日期前上一个财年（12月31日）的财报数据
        Args:
            stock_code: 股票代码
            as_of_date: 基准日期
        Returns:
            财报数据字典
        """
        try:
            with self.connection.cursor() as cursor:
                # 获取上一个财年12月31日的报告
                sql = """
                SELECT * FROM stock_financial_reports 
                WHERE stock_code = %s AND report_date <= %s 
                AND MONTH(report_date) = 12 AND DAY(report_date) = 31
                ORDER BY report_date DESC 
                LIMIT 1
                """
                cursor.execute(sql, (stock_code, as_of_date))
                result = cursor.fetchone()
                return result
        except Exception as e:
            log.error(f"获取股票 {stock_code} 上年财报数据失败: {e}")
            return None
    
    def get_latest_quarter_report(self, stock_code: str, as_of_date: str) -> Optional[Dict]:
        """
        获取指定日期前最新的季度财报数据
        Args:
            stock_code: 股票代码
            as_of_date: 基准日期
        Returns:
            财报数据字典
        """
        try:
            with self.connection.cursor() as cursor:
                sql = """
                SELECT * FROM stock_financial_reports 
                WHERE stock_code = %s AND report_date <= %s 
                ORDER BY report_date DESC 
                LIMIT 1
                """
                cursor.execute(sql, (stock_code, as_of_date))
                result = cursor.fetchone()
                return result
        except Exception as e:
            log.error(f"获取股票 {stock_code} 最新季报数据失败: {e}")
            return None
    
    def get_previous_quarter_report(self, stock_code: str, as_of_date: str, current_report_date: str) -> Optional[Dict]:
        """
        获取指定日期前上一个季度的财报数据
        Args:
            stock_code: 股票代码
            as_of_date: 基准日期
            current_report_date: 当前报告日期
        Returns:
            财报数据字典
        """
        try:
            with self.connection.cursor() as cursor:
                sql = """
                SELECT * FROM stock_financial_reports 
                WHERE stock_code = %s AND report_date < %s AND report_date <= %s
                ORDER BY report_date DESC 
                LIMIT 1
                """
                cursor.execute(sql, (stock_code, current_report_date, as_of_date))
                result = cursor.fetchone()
                return result
        except Exception as e:
            log.error(f"获取股票 {stock_code} 上季报数据失败: {e}")
            return None
    
    def get_financial_report_by_date(self, stock_code: str, report_date: str) -> Optional[Dict]:
        """
        根据精确报告日期获取指定股票的财报数据
        Args:
            stock_code: 股票代码
            report_date: 报告日期 (格式: YYYY-MM-DD)
        Returns:
            财报数据字典，如果未找到返回None
        """
        try:
            with self.connection.cursor() as cursor:
                sql = """
                SELECT * FROM stock_financial_reports 
                WHERE stock_code = %s AND report_date = %s
                LIMIT 1
                """
                cursor.execute(sql, (stock_code, report_date))
                result = cursor.fetchone()
                
                if result:
                    log.debug(f"找到股票 {stock_code} 在 {report_date} 的财报数据")
                else:
                    log.error(f"未找到股票 {stock_code} 在 {report_date} 的财报数据")
                    
                return result
                
        except Exception as e:
            log.error(f"查询股票 {stock_code} 在 {report_date} 的财报数据失败: {e}")
            return None
    
        
    def calculate_dividend_yields(self, stock_code: str, price_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        计算两种股息率：静态股息率(dv_ratio)和滚动股息率(dv_ttm)
        静态股息率使用上一财年分红数据（包含下一年1月的分红），滚动股息率使用最近12个月分红数据
        Args:
            stock_code: 股票代码
            price_df: 包含日期和收盘价的数据框
        Returns:
            (静态股息率序列, 滚动股息率序列)
        """
        dividend_df = self.get_dividend_data(stock_code)
        if dividend_df.empty:
            zero_series = pd.Series(index=price_df.index, data=0.0)
            return zero_series, zero_series
        
        # 确保ex_dividend_date是datetime类型
        if dividend_df['equity_reg_date'].dtype == 'object':
            dividend_df['equity_reg_date'] = pd.to_datetime(dividend_df['equity_reg_date'])
        
        date_range = price_df.index
        dv_ratio_series = pd.Series(index=date_range, data=0.0)  # 静态股息率
        dv_ttm_series = pd.Series(index=date_range, data=0.0)     # 滚动股息率
        
        for current_date in date_range:
            current_price = price_df.loc[current_date, 'close']
        
            if current_price <= 0:
                continue
            
            # 1. 计算静态股息率 (dv_ratio) - 使用上一财年分红（包含下一年1月的分红）
            previous_fiscal_year_dividend = self.get_previous_year_dividend(stock_code, current_date)
            if previous_fiscal_year_dividend > 0:
                from decimal import Decimal
                dv_ratio = (Decimal(str(previous_fiscal_year_dividend)) / Decimal(str(current_price))) * Decimal(100)
                dv_ratio_series.loc[current_date] = round(float(dv_ratio), 4)
            
            # 2. 计算滚动股息率 (dv_ttm) - 使用最近12个月分红
            one_year_ago = current_date - pd.DateOffset(years=1)
            recent_dividends = dividend_df[
                (dividend_df['equity_reg_date'] > one_year_ago) & 
                (dividend_df['equity_reg_date'] <= current_date)
            ]
            
            if not recent_dividends.empty:
                total_dividend = recent_dividends['dividend_per_share'].sum()
                from decimal import Decimal
                dv_ttm = (Decimal(str(total_dividend)) / Decimal(str(current_price))) * Decimal(100)
                dv_ttm_series.loc[current_date] = round(float(dv_ttm), 4)
        
        return dv_ratio_series, dv_ttm_series
     
    
    def calculate_pe_ratio(self, price: float, eps: float) -> Optional[float]:
        """
        计算市盈率
        Args:
            price: 股价
            eps: 每股收益
        Returns:
            市盈率，如果计算失败返回None
        """
        if eps is None or eps <= 0:
            return None
        return round(price / eps, 4)
    
    def calculate_pe_ttm(self, price: float, eps_ttm: float) -> Optional[float]:
        """
        计算滚动市盈率（TTM）
        Args:
            price: 股价
            eps_ttm: 滚动每股收益
        Returns:
            滚动市盈率，如果计算失败返回None
        """
        if eps_ttm is None or eps_ttm <= 0:
            return None
        return round(price / eps_ttm, 4)
    
    def calculate_pb_ratio(self, price: float, navps: float) -> Optional[float]:
        """
        计算市净率
        Args:
            price: 股价
            navps: 每股净资产
        Returns:
            市净率，如果计算失败返回None
        """
        if navps is None or navps <= 0:
            return None
        return round(price / navps, 4)
    
    def calculate_eps_ttm(self, stock_code: str, as_of_date: str, latest_report_date: str) -> Optional[float]:
        """
        计算滚动每股收益（TTM）
        逻辑：使用最新季度的累计EPS + 上年全年EPS - 上年同期的累计EPS
        Args:
            stock_code: 股票代码
            as_of_date: 计算日期
            latest_report_date: 最新报告日期
        Returns:
            滚动每股收益，如果计算失败返回None
        """
        try:
            # 1. 获取最新季度报告
            latest_report = self.get_latest_quarter_report(stock_code, as_of_date)
            if not latest_report:
                return None
            latest_eps = latest_report.get('weighted_eps') or latest_report.get('diluted_eps')
            if latest_eps is None:
                return None

            # 2. 获取上年全年EPS（基于当前日期as_of_date判断）
            # 解析当前日期as_of_date
            current_date_obj = datetime.strptime(as_of_date, '%Y-%m-%d')
            current_year = current_date_obj.year
            current_month = current_date_obj.month
            current_day = current_date_obj.day
            
            #获取上一年年报
            previous_year_end = f"{current_year - 1}-12-31"   
            previous_year_annual_report = self.get_previous_fiscal_year_report(stock_code, previous_year_end)
            annual_eps = None
            
            if not previous_year_annual_report:
                # 判断当前日期是否在1月1日至3月31日之间
                if current_month == 1 or (current_month == 2) or (current_month == 3 and current_day <= 31):
                    #如果在1-3月，无法获取上年年报，尝试获取上上年年报
                    previous_2year_annual_report = self.get_previous_fiscal_year_report(stock_code, f"{current_year - 2}-12-31")
                    if not previous_2year_annual_report:
                        return None
                    #过去一年的 annual_eps = 最新季报的eps + 上上年年报的eps - 上上年同期季报的eps，
                    #在这里获取 上上年年报的eps
                    annual_eps = previous_2year_annual_report.get('weighted_eps') or previous_2year_annual_report.get('diluted_eps')                      
                    
                else:
                    return None
            else:
                annual_eps = previous_year_annual_report.get('weighted_eps') or previous_year_annual_report.get('diluted_eps')

            if annual_eps is None:
                return None

            # 3. 获取上年同期累计EPS（关键修正：取去年相同季度的累计数据）
            # 构建上年同期的报告日期，例如最新报告是2023Q2，则上年同期为2022Q2
            # 假设report_date字段存储的格式能反映季度，例如 '2022-06-30' 代表Q2
            latest_date = datetime.strptime(latest_report_date, '%Y-%m-%d')
            previous_year_period_date = None
            
            # 判断当前日期是否在1月1日至3月31日之间
            if current_month == 1 or (current_month == 2) or (current_month == 3 and current_day <= 31):
                
                #如果没有获取到上年同期的报告，说明去年财报还没出来
                if not previous_year_annual_report:
                    #如果在1-3月，且无法获取上年年报，尝试获取上上年年报
                    previous_year_period_date = f"{latest_date.year - 1}-12-31"
                else:
                    #否则取上年年报
                    previous_year_period_date = previous_year_end
                                    
            else:
                previous_year_period_date = f"{latest_date.year - 1}-{latest_date.month:02d}-{latest_date.day:02d}"
            
            previous_year_period_report = self.get_financial_report_by_date(stock_code, previous_year_period_date)
            
            if not previous_year_period_report:
                # 如果没找到精确日期的报告，可以尝试获取该日期之前最近的一份报告（例如上年Q2结束日可能不是同一天）
                previous_year_period_report = self.get_previous_quarter_report(stock_code, previous_year_period_date)
            
            if not previous_year_period_report:
                return None
            previous_year_eps = previous_year_period_report.get('weighted_eps') or previous_year_period_report.get('diluted_eps')
            if previous_year_eps is None:
                return None

            # 4. 应用正确的TTM公式计算
            eps_ttm = latest_eps + annual_eps - previous_year_eps
            return round(max(eps_ttm, 0), 4)  # 确保非负并保留4位小数

        except Exception as e:
            log.error(f"计算股票 {stock_code} TTM EPS失败: {e}")
            return None
    
    def calculate_valuation_for_stock(self, stock_code: str, stock_name: str, 
                                    start_date: str, end_date: str) -> pd.DataFrame:
        """
        计算单只股票的每日估值数据
        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            start_date: 开始日期
            end_date: 结束日期
        Returns:
            包含估值数据的DataFrame
        """
        log.info(f"开始计算股票 {stock_code}({stock_name}) 的估值数据")
        
        # 获取历史价格数据
        price_df = self.get_historical_price_data(stock_code, start_date, end_date)
        if price_df.empty:
            log.error(f"股票 {stock_code} 没有价格数据")
            return pd.DataFrame()
        
        # 计算两种股息率
        dv_ratio_series, dv_ttm_series = self.calculate_dividend_yields(stock_code, price_df)
        
        valuation_data = []
        
        for date, row in price_df.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            close_price = row['close']
            
            # 获取上一个财年财报数据（用于计算静态PE）
            annual_financial_data = self.get_previous_fiscal_year_report(stock_code, date_str)
            if not annual_financial_data:
                log.error(f"股票 {stock_code} 在 {date_str} 没有上年财报数据")
                continue
            
            # 获取最新季度财报数据（用于计算TTM）
            latest_quarter_data = self.get_latest_quarter_report(stock_code, date_str)
            if not latest_quarter_data:
                log.error(f"股票 {stock_code} 在 {date_str} 没有季度财报数据")
                continue
            
            # 提取财务指标
            annual_eps = annual_financial_data.get('weighted_eps') or annual_financial_data.get('diluted_eps')
            navps = annual_financial_data.get('adjusted_net_asset') or annual_financial_data.get('net_asset_per_share')
            
            # 计算TTM EPS
            eps_ttm = self.calculate_eps_ttm(stock_code, date_str, latest_quarter_data['report_date'].strftime('%Y-%m-%d'))
            
            # 计算估值指标
            pe = self.calculate_pe_ratio(close_price, annual_eps) if annual_eps else None
            pe_ttm = self.calculate_pe_ttm(close_price, eps_ttm) if eps_ttm else None
            pb = self.calculate_pb_ratio(close_price, navps) if navps else None
            
           # 获取两种股息率
            dv_ratio = dv_ratio_series.loc[date] if date in dv_ratio_series.index else 0.0
            dv_ttm = dv_ttm_series.loc[date] if date in dv_ttm_series.index else 0.0
            
            # 估算总市值（需要根据实际情况实现）
            market_cap = self.estimate_market_cap(stock_code, date_str, close_price)
            
            valuation_data.append({
                'stock_code': stock_code,
                'stock_name': stock_name,
                'trade_date': date_str,
                'close_price': close_price,
                'pe': pe,
                'pe_ttm': pe_ttm,
                'pb': pb,
                'dv_ratio': dv_ratio,
                'dv_ttm': dv_ttm,
                'ps': None,       # 需要营收数据
                'ps_ttm': None,   # 需要营收数据
                'total_mv': market_cap,
                'annual_eps': annual_eps,
                'eps_ttm': eps_ttm,
                'navps': navps
            })
        
        log.info(f"完成股票 {stock_code} 的估值计算，共 {len(valuation_data)} 条记录")
        return pd.DataFrame(valuation_data)
    
    def calculate_main_profit_ttm(self, stock_code: str, as_of_date: str, latest_report_date: str) -> Optional[float]:
        """
        计算滚动主营业务利润（TTM）
        公式：TTM Main Profit = 最新累计Main Profit + 上年全年Main Profit - 上年同期累计Main Profit
        Args:
            stock_code: 股票代码
            as_of_date: 计算日期
            latest_report_date: 最新报告日期
        Returns:
            滚动主营业务利润，如果计算失败返回None
        """
        try:
            # 1. 获取最新季度报告的主营业务利润
            latest_report = self.get_latest_quarter_report(stock_code, as_of_date)
            if not latest_report:
                return None
            latest_main_profit = latest_report.get('main_profit')
            if latest_main_profit is None:
                return None

            # 2. 获取上年全年主营业务利润（上一年12月31日的报告）
            latest_date = datetime.strptime(latest_report_date, '%Y-%m-%d')
            previous_year_end = f"{latest_date.year - 1}-12-31"
            previous_year_annual_report = self.get_previous_fiscal_year_report(stock_code, previous_year_end)
            if not previous_year_annual_report:
                return None
            annual_main_profit = previous_year_annual_report.get('main_profit')
            if annual_main_profit is None:
                return None

            # 3. 获取上年同期累计主营业务利润
            previous_year_period_date = f"{latest_date.year - 1}-{latest_date.month:02d}-{latest_date.day:02d}"
            previous_year_period_report = self.get_financial_report_by_date(stock_code, previous_year_period_date)
            
            if not previous_year_period_report:
                previous_year_period_report = self.get_previous_quarter_report(stock_code, previous_year_period_date)
            
            if not previous_year_period_report:
                return None
            previous_year_main_profit = previous_year_period_report.get('main_profit')
            if previous_year_main_profit is None:
                return None

            # 4. 应用TTM公式计算
            main_profit_ttm = latest_main_profit + annual_main_profit - previous_year_main_profit
            return round(max(main_profit_ttm, 0), 4)  # 确保非负并保留4位小数

        except Exception as e:
            log.error(f"计算股票 {stock_code} TTM主营业务利润失败: {e}")
            return None

    def calculate_ps_ratio(self, price: float, main_profit_per_share: float) -> Optional[float]:
        """
        计算市销率
        Args:
            price: 股价
            main_profit_per_share: 每股主营业务利润
        Returns:
            市销率，如果计算失败返回None
        """
        if main_profit_per_share is None or main_profit_per_share <= 0:
            return None
        return round(price / main_profit_per_share, 4)

    def calculate_ps_ttm(self, price: float, main_profit_ttm_per_share: float) -> Optional[float]:
        """
        计算滚动市销率（PS_TTM）
        Args:
            price: 股价
            main_profit_ttm_per_share: 滚动每股主营业务利润
        Returns:
            滚动市销率，如果计算失败返回None
        """
        if main_profit_ttm_per_share is None or main_profit_ttm_per_share <= 0:
            return None
        return round(price / main_profit_ttm_per_share, 4)     
    
    def get_revenue_per_share(self, financial_data: Dict, total_shares: float) -> Optional[float]:
        """
        计算每股营业收入
        Args:
            financial_data: 财务数据
            total_shares: 总股本
        Returns:
            每股营业收入，如果计算失败返回None
        """
        # 这里需要根据实际财报字段调整
        # 假设主营业务利润可以代表营业收入
        if 'main_profit' in financial_data and financial_data['main_profit'] and total_shares > 0:
            return financial_data['main_profit'] / total_shares
        return None
    
    def estimate_total_shares(self, market_cap: float, price: float) -> Optional[float]:
        """
        估算总股本
        Args:
            market_cap: 总市值
            price: 股价
        Returns:
            总股本数，如果计算失败返回None
        """
        if price is None or price <= 0:
            return None
        return market_cap / price
    
    
    def estimate_market_cap(self, stock_code: str, date: str, price: float) -> Optional[float]:
        """
        估算总市值（这里需要根据实际情况实现）
        Args:
            stock_code: 股票代码
            date: 日期
            price: 股价
        Returns:
            总市值估算值
        """
        # 这是一个示例实现，实际中可能需要：
        # 1. 从数据库获取总股本数据
        # 2. 使用其他数据源
        # 3. 根据历史数据估算
        
        # 这里简单返回None，实际使用时需要完善这个函数
        return None
    
    def save_valuation_data(self, valuation_df: pd.DataFrame):
        """
        保存估值数据到数据库
        Args:
            valuation_df: 包含估值数据的DataFrame
        """
        if valuation_df.empty:
            return
        
        try:
            with self.connection.cursor() as cursor:
                # 准备插入SQL
                sql = """
                INSERT IGNORE INTO stock_pe_history (
                    stock_code, stock_name, trade_date, pe, pe_ttm, pb, 
                    dv_ratio, dv_ttm, ps, ps_ttm, total_mv
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                # 准备数据
                data_to_insert = []
                for _, row in valuation_df.iterrows():
                    data_to_insert.append((
                        row['stock_code'],
                        row['stock_name'],
                        row['trade_date'],
                        row['pe'],
                        row['pe_ttm'],
                        row['pb'],
                        row['dv_ratio'],
                        row['dv_ttm'],
                        row['ps'],
                        row['ps_ttm'],
                        row['total_mv']
                    ))
                
                # 批量插入
                cursor.executemany(sql, data_to_insert)
                self.connection.commit()
                
                log.info(f"成功保存 {len(data_to_insert)} 条估值记录到数据库")
                
        except Exception as e:
            self.connection.rollback()
            log.error(f"保存估值数据失败: {e}")
    
    def calculate_valuation_for_all_stocks(self, start_date: str, end_date: str, 
                                         batch_size: int = 10):
        """
        计算所有股票的估值数据
        Args:
            start_date: 开始日期
            end_date: 结束日期
            batch_size: 批量处理大小
        """
        log.info(f"开始计算所有股票从 {start_date} 到 {end_date} 的估值数据")
        
        # 获取所有股票代码
        # stock_codes = self.get_stock_codes()
        
        selectDF = get_select_stocks()
        if selectDF is None or selectDF.empty:
            log.error("没有获取到股票代码，退出计算")
            return        
        stock_codes = selectDF['代码'].tolist()
        stock_names = selectDF.set_index('代码')['名称'].to_dict()
        
        if not stock_codes:
            log.error("没有获取到股票代码，退出计算")
            return
        
        total_stocks = len(stock_codes)
        log.info(f"共需要处理 {total_stocks} 只股票")
        
        # 批量处理股票
        for i in range(0, total_stocks, batch_size):
            batch_codes = stock_codes[i:i + batch_size]
            log.info(f"处理批次 {i//batch_size + 1}/{(total_stocks-1)//batch_size + 1}")
                      
            for stock_code in batch_codes:
                try:
                    stock_name = stock_names.get(stock_code, stock_code)
                    
                    # 计算单只股票的估值
                    valuation_df = self.calculate_valuation_for_stock(
                        stock_code, stock_name, start_date, end_date
                    )
                    
                    if not valuation_df.empty:
                        # 保存到数据库
                        self.save_valuation_data(valuation_df)
                        
                except Exception as e:
                    log.error(f"处理股票 {stock_code} 时发生错误: {e}")
                    continue
        
        log.info("完成所有股票的估值计算")

def main():
    
    # 日期范围     start_date = '2025-07-01' 
    """
    在此以前的PE数据都是从akshare获取的ak.stock_a_indicator_lg接口获取的。
    从8月份无法再使用接口，写了这个脚本从数据库中获取历史价格和财报数据计算PE等估值指标
    但通过财报计算的PE等指标和akshare接口获取的会有差异，主要是财报数据的口径和时间点不一样。
    所以在该时间点之后的PE等估值指标会和以前的有断层。    
    未来可以考虑从数据库中获取历史数据，减少对akshare的依赖
    """
    start_date = '2025-9-20'
    end_date = '2025-12-02'
    
    # 创建估值计算器
    calculator = StockValuationCalculator(DB_CONFIG)
    
    try:
        # 连接数据库
        calculator.connect_to_database()
        
        # 计算所有股票的估值
        calculator.calculate_valuation_for_all_stocks(start_date, end_date,1)
        
    except Exception as e:
        log.error(f"程序执行失败: {e}")
    finally:
        # 关闭数据库连接
        calculator.disconnect_from_database()

if __name__ == "__main__":
    main()