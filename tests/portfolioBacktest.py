import akshare as ak
import pandas as pd
import numpy as np
import log4ak
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union, Any, Optional
import math

# 日志配置
log = log4ak.LogManager(log_level=log4ak.INFO)



class PositionTracker:
    """
    单只股票持仓跟踪器
    负责单只股票的成本计算、分红处理和收益计算
    """
    def __init__(
        self, 
        code: str, 
        buy_date: str, 
        shares: int, 
        buy_fee: float = 0.0017, 
        dividend_tax: float = 0.1,
        buy_price: float = 0.0
    ) -> None:
        """
        初始化单只股票持仓
        :param code: 股票代码
        :param buy_date: 买入日期 (YYYYMMDD)
        :param shares: 买入股数
        :param buy_fee: 买入费率 (默认0.17%)
        :param dividend_tax: 分红税率 (默认10%)
        :param buy_price: 买入价格
        """
        self.code = code
        self.buy_date = buy_date
        self.shares = shares
        self.buy_fee = buy_fee
        self.dividend_tax = dividend_tax
        self.dividends = []  # 分红记录: [(date, 每股分红, 税后金额)]
        
        # 获取买入价格(不复权)
        self.buy_price = buy_price
        log.info(f"初始化{code}持仓: {buy_date}买入{shares}股 @ {self.buy_price:.2f}元")
    
    def _get_actual_price(self, date: str) -> float:
        """获取含手续费的买入价格"""
        try:
            # 获取不复权收盘价
            price_df = ak.stock_zh_a_hist(
                symbol=self.code,
                period="daily",
                start_date=date,
                end_date=date,
                adjust=""
            )
            close_price = price_df.iloc[0]['收盘']
            
            # 计算含手续费的实际成本
            return close_price * (1 + self.buy_fee)
        except Exception as e:
            log.error(f"获取{self.code}买入价格失败: {str(e)}")
            raise
    
    def add_dividend(self, dividend_date: str, dividend_per_share: float) -> float:
        """记录分红信息并计算税后金额"""
        net_dividend = dividend_per_share * (1 - self.dividend_tax)
        total_net_dividend = net_dividend * self.shares
        self.dividends.append((dividend_date, dividend_per_share, total_net_dividend))
        log.info(f"{self.code}在{dividend_date}分红: 每股{dividend_per_share:.4f}元 -> 税后{net_dividend:.4f}元")
        return total_net_dividend
    
    def calculate_daily_positionvalues(
        self, 
        current_date: str, 
        current_price: float
    ) -> Tuple[float, float, float]:
        """
        计算单只股票在指定日期的价值
        :param current_date: 当前日期 (YYYYMMDD)
        :param current_price: 当前不复权价格
        :return: (市值, 累计收益，累计分红)
        """
        # 当前市值
        market_value = self.shares * current_price
        
        # 累计分红收益 (税后)
        dividend_income = sum(
            amount for date, _, amount in self.dividends 
            if  pd.to_datetime(date) <= pd.to_datetime(current_date)
        )
        
        # 总收益 = (当前市值 - 成本市值) + 分红收益
        cost_value = self.shares * self.buy_price
        total_return = (market_value - cost_value) + dividend_income
        
        return market_value, total_return, dividend_income

class PortfolioSimulator:
    """
    投资组合模拟器
    管理整个投资组合的持仓、现金和净值计算
    """
    
    def __init__(
        self, 
        initial_cash: float = 0, 
        start_date: str = '20230101', 
        buy_fee: float = 0.0017, 
        dividend_tax: float = 0.1,
        isSaveStock: bool = False
    ) -> None:
        """
        初始化投资组合
        :param initial_cash: 初始资金
        :param start_date: 开始日期 (YYYYMMDD)
        :param buy_fee: 买入费率 (默认0.17%)
        :param dividend_tax: 分红税率 (默认10%)
        """

        self.initial_cash = initial_cash #初始本金
        self.current_cash = initial_cash #初始现金
        self.start_date = start_date #组合回测开始时间
        self.buy_fee = buy_fee #买入费率 (默认0.17%)
        self.dividend_tax = dividend_tax  #分红税率 (默认10%)
        self.isSaveStock = isSaveStock #是否需要在excel存储每天的股票价格(默认False)
        self.positions = {}  # 单只股票持仓跟踪器组合中所有股票持仓的跟踪器 {股票代码: PositionTracker}
        self.dividend_cache = {}  # 单只股票期间所有分红数据的跟踪器 {股票代码: dividend}
        self.dividend_records = []  # 全部分红记录
        self.trade_dates = self._get_trading_calendar(start_date) #str类型的交易日List
        # 新增价格缓存字典 {股票代码: DataFrame}
        self.price_cache = {}
        # 新增回测结束日期存储
        self.backtest_end_date = None
        # 新增：待处理订单列表[3](@ref)
        self.pending_orders = [] 

        log.info(f"组合初始化: 起始资金{initial_cash:.2f}元, 开始日期{start_date}")

    def _cache_stock_data(
        self, 
        code: str, 
        start_date: str, 
        end_date: str
    ) -> None:
        """预加载并缓存单只股票历史数据"""
        if code not in self.price_cache:
            try:
                # 一次性获取股票全部历史数据
                df = ak.stock_zh_a_hist(
                    symbol=code,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust=""
                )
                # 设置日期索引加速查询[9](@ref)
                df['日期']=pd.to_datetime(df['日期']).dt.strftime('%Y%m%d')
                df = df.set_index('日期')
                self.price_cache[code] = df
                log.info(f"缓存{code}数据: {start_date}至{end_date}共{len(df)}条")
            except Exception as e:
                log.error(f"缓存{code}数据失败: {str(e)}")
                self.price_cache[code] = pd.DataFrame()

    def _precache_dividend_data(self, codes: List[str]) -> None:
        """预加载所有股票的分红数据"""
        for code in codes:
            if code not in self.dividend_cache:
                self.dividend_cache[code] = self._get_dividend_data(code)
                log.info(f"预加载{code}分红数据: {len(self.dividend_cache[code])}条记录")


    def _get_trading_calendar(self, start_date: str) -> List[str]:
        """获取交易日历"""
        trade_dates = ak.tool_trade_date_hist_sina()
        trade_dates['trade_date'] = pd.to_datetime(trade_dates['trade_date'])
        return trade_dates[trade_dates['trade_date'] >= pd.to_datetime(start_date)]['trade_date'].dt.strftime('%Y%m%d').tolist()
    
    def buy_stock(self, code: str, buy_date: str, shares: int) -> bool:
        """买入下单处理，仅记录订单，不立即扣款"""
        if any(order[0] == code for order in self.pending_orders):
            log.error(f"{code}已有待处理订单")
            return False
            
        self.pending_orders.append((code, buy_date, shares))

        ##初始化创建持仓
        position = PositionTracker(
            code, self.start_date, 0, # 初始数量设为0
            self.buy_fee, self.dividend_tax,
            0  # 初始价格设为0
        )
        self.positions[code] = position
        log.info(f"登记买入订单: {buy_date}买入{code} {shares}股")
        return True

    def process_pending_orders(self, current_date: str) -> None:
        """
        处理当日应执行的订单
        创建持仓PositionTracker
        创建持仓时需要根据买入日期查询股票价格
        """

        executed_orders = []
        #强制日期格式转换

        for order in self.pending_orders:
            code, buy_date, shares = order
            if current_date != buy_date:
                continue
                
            try:
                # 获取当日实际价格
                if code not in self.price_cache or current_date not in self.price_cache[code].index:
                    # 容错：自动补充缓存
                    self._cache_stock_data(code, self.start_date, self.backtest_end_date)
                
                # 从缓存获取价格（统一日期格式）
                price_df = self.price_cache[code]
                close_price = price_df.loc[current_date, '收盘']
                actual_price = close_price * (1 + self.buy_fee)
                
                # 计算成本
                cost = shares * actual_price
                
                if cost > self.current_cash:
                    log.error(f"{current_date}现金不足: 需要{cost:.2f}元, 可用{self.current_cash:.2f}元")
                    continue
                    
                # 扣减现金
                self.current_cash -= cost
                
                # 创建持仓
                position = PositionTracker(
                    code, current_date, shares,
                    self.buy_fee, self.dividend_tax,
                    actual_price  # 直接传入计算好的价格
                )
                self.positions[code] = position
                executed_orders.append(order)
                log.info(f"{current_date}执行买入: {code} {shares}股 @ {actual_price:.2f}元")
                # 容错：自动补充缓存
                if code not in self.price_cache:
                    self._cache_stock_data(code, self.start_date, self.backtest_end_date)
                    
            except Exception as e:
                log.error(f"{current_date}执行{code}买入失败: {str(e)}")
                continue
    
    def _get_dividend_data(self, code: str) -> pd.DataFrame:
        """获取股票分红数据"""
        try:
            # 实际使用时应替换为正确的分红接口
            dividend_df = ak.stock_history_dividend_detail(symbol=code)
            if not dividend_df.empty:
                dividend_df = dividend_df[['除权除息日', '派息']]
                dividend_df['每股分红'] = dividend_df['派息'] / 10
                return dividend_df[['除权除息日', '每股分红']]
            return pd.DataFrame()
        except:
            # 备用方法：使用模拟数据
            log.error(f"使用模拟分红数据: {code}")
            return pd.DataFrame({
                '除权除息日': ['20250115', '20250601'],
                '每股分红': [1.5, 2.0]
            })
    
    def process_dividends(self, current_date: str) -> None:
        """处理分红事件"""
        for code, position in self.positions.items():
            # 从缓存获取数据（不再实时调用API）
            dividend_df = self.dividend_cache.get(code, pd.DataFrame())

            #转换日期格式并筛选
            current_date_dt = pd.to_datetime(current_date, format="%Y%m%d")
            buy_date = pd.to_datetime(position.buy_date, format="%Y%m%d")

            # 将除权出席日期列转换为datetime类型[3,5,6](@ref)
            dividend_df['除权除息日'] = pd.to_datetime(dividend_df['除权除息日'], format="%Y%m%d", errors='coerce')
            
            
            # 筛选当前日期前的分红
            dividends = dividend_df[
                (dividend_df['除权除息日'] <= current_date_dt) & 
                (dividend_df['除权除息日'] >= buy_date)
            ]
            
            # 处理未记录的分红
            for _, row in dividends.iterrows():
                div_date = row['除权除息日']
                div_per_share = row['每股分红']
                
                # 检查是否已记录
                if not any(d[0] == div_date for d in position.dividends):
                    net_amount = position.add_dividend(div_date, div_per_share)
                    self.current_cash += net_amount
                    self.dividend_records.append({
                        'date': current_date,
                        'code': code,
                        'amount': net_amount
                    })
    
    def calculate_daily_totalvalues(self, current_date:str) -> Dict[str, Any]:
        """
        计算组合每日价值
        :return: {
            'date': 日期,
            'cash': 现金余额,
            'positions_value': 持仓市值,
            'total_value': 总资产,
            'net_value': 单位净值,
            'return': 累计收益,
        }
        """
        # 初始化结果
        result = {
            'date': current_date,
            'cash': self.current_cash,
            'positions_value': 0,
            'total_value': self.current_cash,
            'net_value': 1,
            'return': 0
        }

        # 统一日期格式比较[3](@ref)
        current_date_dt = pd.to_datetime(current_date)


        #1，遍历持仓并计算持仓市值
        for code, position in self.positions.items():        
            try:
                # 容错：自动补充缓存
                if code not in self.price_cache or self.price_cache[code].empty:
                    self._cache_stock_data(code, self.start_date, self.backtest_end_date)
            
                # 统一索引格式（关键改进）
                price_df = self.price_cache[code].reset_index()
                price_df['日期'] = pd.to_datetime(price_df['日期'])               
            
                # 获取最近有效价格（优化逻辑）a
                valid_prices = price_df[price_df['日期'] <= current_date_dt]
                if not valid_prices.empty:
                    current_price = valid_prices.iloc[-1]['收盘']
                else:
                    current_price = 0.0
                    log.error(f"{code}无有效价格数据")                 
            except Exception as e:
                current_price = 0.0
                log.error(f"价格获取失败: {str(e)}")

            #计算单支股票持仓市值
            market_value, one_position_return,dividend_income = position.calculate_daily_positionvalues(current_date, current_price)            
            result['positions_value'] += market_value
            result[f'{code}return'] = one_position_return
            result[f'{code}dividend'] = dividend_income
            if self.isSaveStock :
                result[f'{code}current_price'] = current_price #如果要看每天的价格可激活此行
        
        #2，计算总市值=现金+持仓市值
        result['total_value'] = self.current_cash +  result['positions_value']
        #3，计算净值和利润
        result['net_value'] = result['total_value']/self.initial_cash
        result['return'] = result['total_value'] - self.initial_cash

        return result
    
    def run_backtest(self, end_date: str = '20230101') -> pd.DataFrame:
        """运行回测"""

        #初始化回测结束日期
        self.backtest_end_date = end_date

        # 合并持仓股票与待处理订单股票
        pending_codes = {order[0] for order in self.pending_orders}
        existing_codes = set(self.positions.keys())
        all_codes = existing_codes.union(pending_codes)
        
        # 预缓存所有相关股票数据（关键改进）
        for code in all_codes:
            #buy_dates = [order[1] for order in self.pending_orders if order[0]==code]
            #earliest_date = min(buy_dates) if buy_dates else self.start_date
            #由于计算时要读取数据，先预缓存股票的所有历史价格数据
            self._cache_stock_data(code, self.start_date, self.backtest_end_date)
        
        # 预加载分红数据（包含待处理订单）
        self._precache_dividend_data(list(all_codes))  # 修改点：传入所有相关代码 
        
        # 执行回测循环
        valid_dates = [d for d in self.trade_dates if pd.to_datetime(d).date() <= pd.to_datetime(self.backtest_end_date).date()]
        results = []
        
        for date in valid_dates:
            try:
                #date_dt = pd.to_datetime(date).date()  # 统一为date类型
                # 强制处理分红事件（新增）
                self.process_dividends(date)

                # 处理订单（如果当日存在）
                self.process_pending_orders(date)

        
                # 无论是否有订单都计算净值（关键改进）
                daily_result = self.calculate_daily_totalvalues(date)
                results.append(daily_result)
            except Exception as e:
                log.error(f"{date}回测失败: {str(e)}")
        
        return pd.DataFrame(results)

def get_portfolio_stocks(select_path=".\input\selectlist_my.xlsx") -> pd.DataFrame:
    """
    读取特定选中股票列表，返回标准化代码与简称
    :param SELECT_PATH: 选中股票文件路径
    :return: DataFrame(代码, 名称)
    """
    # 读取上海数据[1,3](@ref)
    se_cols = {'A股代码':'code', 'buydate':'buydate',"amount":"amount"}
    se_df = pd.read_excel(select_path,
        usecols=list(se_cols.keys()),
        dtype={'A股代码': str, 'buydate' : str}
    ).rename(columns=se_cols)
    se_df = se_df[se_df['code'].notna()]  # 过滤无A股代码的记录


    # 数据清洗
    se_df.drop_duplicates(subset=['code'], keep='first', inplace=True)
    se_df.sort_values(by='buydate', inplace=True)
    
    return se_df[['code', 'buydate',"amount"]]


# ====================== 测试代码 ======================
def test_portfolio_simulator() -> pd.DataFrame:
    """测试投资组合模拟器"""
    # 初始化组合
    simulator = PortfolioSimulator(
        initial_cash=1000000,
        start_date="20230101",
        isSaveStock = False #excel中是否要存储各股票每天的价格，用作手工校验
    )
    
    # 从excel创建股票订单
    buy_df = get_portfolio_stocks(r".\input\selectlist_my.xlsx").dropna(axis=0, how='any')
    for _, row in buy_df.iterrows():
        code = row['code']
        buydate = row['buydate']
        amount = row['amount']
        amount_100 = math.floor(amount/100)*100
        simulator.buy_stock(code, buydate, amount_100)
    
    # 运行回测
    end_date = "20250616"
    results = simulator.run_backtest(end_date)
    
    # 保存结果
    results.to_csv(f".\output\portfolio_backtest_{end_date}.csv", index=False)
    print(f"回测完成，结果已保存到 portfolio_backtest_{end_date}.csv")
    
    # 打印最后5天结果
    print("\n最后5天组合表现:")
    print(results[['date', 'net_value', 'total_value', 'return']].tail())
    
    return results

if __name__ == "__main__":
    # 运行测试
    test_results = test_portfolio_simulator()
