import os
from pathlib import Path
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union, Any, Optional
import math
import time
import log4ak
import insertStockHist as ish
import insertDividendInfo as idi
import insert_major_index_valuation as imiv

base_path = Path(__file__).parent #系统绝对目录
log = log4ak.LogManager(log_level=log4ak.ERROR)# 日志配置


IS_MYSQL = True #PE数据来源，使用数据库速度快很多：数据库/Akshare  True/False

#买入策略参数
DYNAMIC_BUYMONEY = True #是否根据现有资金动态调整买入金额
EQUAL_WEIGHT_BUY = True#是否等权买入，即每支股票只买入一次
IS_BUY_K = True #是否加入指数PE分位系数
PE_PERCENTILE_YEAR = 3#PE分位回溯时长（年）
BUY_MAX_K = 2 #指数PE分位系数上限
BUY_MIN_K = 0.3 #指数PE分位系数下限

#卖出策略参数
HOLD_DAYS_CONDITION = 1305 #卖出条件持仓时间 >= 1305天
RETURN_CONDITION = 3 #卖出条件总收益率 >= 3.6倍
DRAWDOWN = -1000#卖出回撤阈值15%，设为-1000该配置无效




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
        self.re = 0.0 
        self.buy_fee = buy_fee
        self.dividend_tax = dividend_tax
        self.dividends = []  # 分红记录: [(date, 每股分红, 税后金额)]
        self.dividend_income = 0.0 # 持仓分红
        self.total_return_rate = 0.0 # 持仓收益率 
        
        
        # 获取买入价格(不复权)
        self.buy_price = buy_price
        self.max_price = 0.0 # 股票历史最高价(前复权)

        log.info(f"初始化{code}持仓: {buy_date}买入{shares}股 @ {self.buy_price:.2f}元")
    
    def _get_actual_price(self, date: str) -> float:
        """获取含手续费的买入价格"""
        try:
            #if IS_MYSQL:
            #    df = ish.get_stock_data_from_mysql(code_clean,'qfq')
            #    df = df[['日期', '收盘', '成交额']].copy()
            #    start_date_dt = pd.to_datetime(start_date, format='%Y%m%d')
            #    df['日期']= pd.to_datetime(df['日期'], format='%Y%m%d')

            #    df = df[df['日期'] > start_date_dt]

            #else:
            #    df = ak.stock_zh_a_hist(
            #        symbol=code_clean,
            #        period="daily",
            #        adjust="qfq",
            #        start_date=start_date
            #    )

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
    
    def _check_sell_conditions(self, current_price: float,current_date: str, total_return_rate: float) -> bool:
        """
        检查卖出条件（使用完整收益率）
        条件1：持仓时间 >= 600天
        条件2：总收益率 >= 3.6倍
        """
        #如果初始化后还没买入，直接返回，不做卖出判断
        if self.shares == 0:
            return False

        isSale = False

        # 计算持仓天数
        current_dt = pd.to_datetime(current_date)
        buy_dt = pd.to_datetime(self.buy_date)
        hold_days = (current_dt - buy_dt).days
        
        # 1,时间和收益条件检查
        time_condition = hold_days >= HOLD_DAYS_CONDITION
        return_condition = total_return_rate >= RETURN_CONDITION

        # 2，回撤检查
        if time_condition or return_condition:
            if current_price/self.max_price < 1- DRAWDOWN/100:
                isSale = True

        # 详细日志
        log.debug(
                    f"{self.code} 卖出检查 @ {current_date}: "
                    f"持仓{hold_days}天/总收益率{total_return_rate:.2f}x "
                    f"当前价格{current_price}，最高最高价格{self.max_price}"
                    f"结果={'卖出' if isSale else '持有'}"
                ) 

        return isSale

    def calculate_daily_positionvalues(
        self, 
        current_date: str, 
        current_price: float
    ) -> Tuple[float, float, float, float, bool]:
        """
        计算单只股票在指定日期的价值
        :param current_date: 当前日期 (YYYYMMDD)
        :param current_price: 当前不复权价格
        :return: (市值, 累计收益，累计分红，是否触发卖出)
        """
        if self.shares == 0:
            return 0,self.re, self.dividend_income, self.total_return_rate, False

        # 1. 计算基础指标
        market_value = self.shares * current_price
        cost_value = self.shares * self.buy_price
        if current_price > self.max_price: self.max_price = current_price 
        
        # 2. 计算累计分红（包含当日）
        self.dividend_income = sum(
            amount for date, _, amount in self.dividends 
            if pd.to_datetime(date) <= pd.to_datetime(current_date)
        )
        
        # 3. 计算总收益和总收益率
        self.re = (market_value - cost_value) + self.dividend_income
        self.total_return_rate = self.re / cost_value if cost_value > 0 else 0
        
        # 4. 检查卖出条件（使用完整收益率）
        should_sell = self._check_sell_conditions(current_price, current_date, self.total_return_rate)
        
        return market_value, self.re, self.dividend_income, self.total_return_rate, should_sell

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
        self.buymoney = 0.0
        self.df_hs300PEttm = imiv.get_index_pe_his('沪深300') if IS_BUY_K else None

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
        #待处理，已处理订单列表
        self.pending_orders = [] 
        self.executed_orders = []
        

        #akshare接口连续失败调用的上限以及失败次数记录
        self.MAX_TRYTIMES = 3
        self.AK_TRYTIME = 0
        #akshare接口调用失败的休眠时间
        self.AK_TRY_FAILD_SLEEPTIME = 60

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
                df=[]
                if IS_MYSQL:
                    df = ish.get_stock_data_from_mysql(code,'')
                    df = df[['日期', '收盘']].copy()
                    start_date_dt = pd.to_datetime(start_date, format='%Y%m%d')
                    df['日期']= pd.to_datetime(df['日期'], format='%Y%m%d')

                    df = df[df['日期'] >= start_date_dt]

                else:
                    df = ak.stock_zh_a_hist(
                        symbol=code,
                        period="daily",
                        start_date=start_date,
                        end_date=end_date,
                        adjust=""
                    )

                # 一次性获取股票全部历史数据
                #df = ak.stock_zh_a_hist(
                #    symbol=code,
                #    period="daily",
                #    start_date=start_date,
                #    end_date=end_date,
                #    adjust=""
                #)
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
                self.AK_TRYTIME=0
                self.dividend_cache[code] = self._get_dividend_data(code)
                log.info(f"预加载{code}分红数据: {len(self.dividend_cache[code])}条记录")

                if not IS_MYSQL: time.sleep(1) 


    def _get_trading_calendar(self, start_date: str) -> List[str]:
        """获取交易日历"""
        trade_dates = ak.tool_trade_date_hist_sina()
        trade_dates['trade_date'] = pd.to_datetime(trade_dates['trade_date'])
        return trade_dates[trade_dates['trade_date'] >= pd.to_datetime(start_date)]['trade_date'].dt.strftime('%Y%m%d').tolist()
    
    def buy_stock(self, code: str, buy_date: str, buymoney:float) -> bool:
        """买入下单处理，仅记录订单，不立即扣款"""

        #等权买入，即每支股票只买入一次的时候激活如下代码
        if EQUAL_WEIGHT_BUY and any(order[0] == code for order in self.pending_orders):
            log.error(f"{code}已有待处理订单")
            return False
            
        self.pending_orders.append((code, buy_date, buymoney))

        if (self.positions is None or 
        not isinstance(self.positions, dict) or 
        code not in self.positions or 
        self.positions[code] is None):
            ##初始化创建持仓
            position = PositionTracker(
                code, self.start_date, 0, # 初始数量设为0
                self.buy_fee, self.dividend_tax,
                0  # 初始价格设为0
            )
            self.positions[code] = position
        log.info(f"登记买入订单: {buy_date}买入{code} 购买资金{buymoney}")
        return True

    def process_pending_orders(self, current_date: str) -> None:
        """
        处理当日应执行的订单
        创建持仓PositionTracker
        创建持仓时需要根据买入日期查询股票价格
        """

        #executed_orders = []
        #强制日期格式转换

        for order in self.pending_orders:
            code, buy_date, buymoney = order
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

                #根据剩余现金和订单量计算买入资金
                pending_orders_num = len(self.pending_orders)
                executed_orders_num = len(self.executed_orders)
                waiting_orders_num = pending_orders_num - executed_orders_num

                hs300PEttm_percentile = imiv.get_pe_percentile(self.df_hs300PEttm,current_date, PE_PERCENTILE_YEAR) if IS_BUY_K else 100*(BUY_MAX_K - 1)/(BUY_MAX_K - BUY_MIN_K)
                buy_k = BUY_MAX_K - hs300PEttm_percentile * (BUY_MAX_K - BUY_MIN_K)/100

                #是否根据现金动态调整买入金额
                if DYNAMIC_BUYMONEY:                    
                    self.buymoney = self.current_cash/waiting_orders_num * buy_k if waiting_orders_num != 0.0 else 0.0
                else:
                    self.buymoney = self.initial_cash/pending_orders_num * buy_k
                log.info(f"每单可买入资金 = {self.buymoney} 系数{buy_k}")
                
                # 计算成本
                shares = math.floor(self.buymoney/(actual_price*100))*100
                cost = shares*actual_price

                
                if cost > self.current_cash:
                    log.error(f"{current_date}现金不足: 需要{cost:.2f}元, 可用{self.current_cash:.2f}元")
                    continue
                    
                # 扣减现金
                self.current_cash -= cost

                if EQUAL_WEIGHT_BUY:
                    # 创建持仓
                    position = PositionTracker(
                        code, current_date, shares,
                        self.buy_fee, self.dividend_tax,
                        actual_price  # 直接传入计算好的价格
                    )
                    self.positions[code] = position
                else:
                    self.positions[code].buy_date = current_date
                    self.positions[code].shares += shares
                    self.positions[code].buy_price = actual_price
                    
                

                
                self.executed_orders.append(order)
                log.info(f"{current_date}执行买入: {code} {shares}股计{cost}元 @ {actual_price:.2f}元")
                # 容错：自动补充缓存
                if code not in self.price_cache:
                    self._cache_stock_data(code, self.start_date, self.backtest_end_date)
                    
            except Exception as e:
                log.error(f"{current_date}执行{code}买入失败: {str(e)}")
                continue
    
    def _get_dividend_data(self, code: str) -> pd.DataFrame:
        """获取股票分红数据"""

        if IS_MYSQL:
            dividend_df = idi.get_dividend_data_mysql(code)
            if not dividend_df.empty:
                dividend_df = dividend_df[['除权除息日', '派息']]
                dividend_df['每股分红'] = dividend_df['派息'] / 10
                return dividend_df[['除权除息日', '每股分红']]
            return pd.DataFrame({
                '除权除息日': [ '20250601'],
                '每股分红': [0]
            })
        else:
            self.AK_TRYTIME += 1
            try:
                # 获取分红接数据
                dividend_df = ak.stock_history_dividend_detail(symbol=code)
            
                if not dividend_df.empty:
                    dividend_df = dividend_df[['除权除息日', '派息']]
                    dividend_df['每股分红'] = dividend_df['派息'] / 10
                    return dividend_df[['除权除息日', '每股分红']]
                return pd.DataFrame({
                    '除权除息日': [ '20250601'],
                    '每股分红': [0]
                })
            except Exception as e:
                if self.AK_TRYTIME < self.MAX_TRYTIMES:
                    log.error(f"{code}通过akshare获取分红失败{self.AK_TRYTIME} 次，休眠后重试")
                    time.sleep(self.AK_TRY_FAILD_SLEEPTIME)
                    #失败次数没到上限休眠后重新查询
                    return self._get_dividend_data(code)
                else:
                    # 备用方法：使用模拟数据
                    log.error(f"{code}通过akshare获取分红失败{self.AK_TRYTIME} 次，不在重试。使用模拟0分红数据")
                    log.error(f"错误信息: {str(e)}")
                    return pd.DataFrame({
                        '除权除息日': [ '20250601'],
                        '每股分红': [0]
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
                div_date = row['除权除息日'].strftime('%Y%m%d') if pd.notnull(row['除权除息日']) else None
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
            'sold_stocks': []  # 记录当日卖出股票
        }
        """
        # 初始化结果
        result = {
            'date': current_date,
            'cash': self.current_cash,
            'positions_value': 0,
            'total_value': self.current_cash,
            'net_value': 1,
            'return': 0,
            'sold_stocks': [] 
        }

        # 存储待删除的持仓代码
        to_remove = []
        # 统一日期格式比较[3](@ref)
        current_date_dt = pd.to_datetime(current_date)


        #1，遍历持仓并计算持仓市值
        for code, position in self.positions.items():        
            try:
                 # 获取当前价格
                current_price = self._get_current_price(code, current_date)
                # 计算持仓价值（包含卖出判断）
                (market_value, 
                 position_return, 
                 dividend_income, 
                 return_rate, 
                 should_sell) = position.calculate_daily_positionvalues(current_date, current_price)
                
                # 处理卖出信号
                if should_sell:
                    # 卖出操作：增加现金，移除持仓
                    self.current_cash += market_value
                    to_remove.append(code)
                    resean = f'{HOLD_DAYS_CONDITION}天止盈' if return_rate < RETURN_CONDITION else f'{RETURN_CONDITION}倍止盈'
                    
                    # 记录卖出信息
                    result['sold_stocks'].append({
                        'code': code,
                        'amount': market_value,
                        'return_rate': return_rate,
                        'reason': resean
                    })
                    log.info(
                        f"{current_date} 卖出 {code}，卖出原因{resean}: "
                        f"获得{market_value:.2f}元 (收益率{return_rate:.2f}x)"
                    )
                    continue  # 跳过后续持仓价值累加
                
                # 未卖出则累加持仓价值
                result['positions_value'] += market_value
                result[f'{code}_return'] = position_return                
                
                if self.isSaveStock:
                    result[f'{code}_price'] = current_price
                    result[f'{code}_dividend'] = dividend_income
                    
            except Exception as e:
                log.error(f"{code}计算失败: {str(e)}")
        
        # 移除已卖出持仓
        for code in to_remove:
            self.positions[code].shares = 0
        
        # 计算总值
        result['total_value'] = self.current_cash + result['positions_value']
        result['net_value'] = result['total_value'] / self.initial_cash
        result['return'] = result['total_value'] - self.initial_cash
        
        return result
    
    def _get_current_price(self, code: str, current_date: str) -> float:
        """安全获取当前价格"""
        try:
            if code not in self.price_cache or self.price_cache[code].empty:
                self._cache_stock_data(code, self.start_date, self.backtest_end_date)
            

            # 统一索引格式（关键改进）
            price_df = self.price_cache[code].reset_index()
            price_df['日期'] = pd.to_datetime(price_df['日期'])
            current_date_dt = pd.to_datetime(current_date)
            
            # 获取最近有效价格（优化逻辑）a
            valid_prices = price_df[price_df['日期'] <= current_date_dt]
            if not valid_prices.empty:
                return valid_prices.iloc[-1]['收盘']
            else:
                log.error(f"{code}无有效价格数据")      
                return 0.0
        except Exception as e:
            log.error(f"价格获取失败: {str(e)}")
            return 0.0


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

def get_portfolio_stocks(select_path=base_path /"..\input\selectlist_my.xlsx") -> pd.DataFrame:
    """
    读取特定选中股票列表，返回标准化代码与简称
    :param SELECT_PATH: 选中股票文件路径
    :return: DataFrame(代码, 名称)
    """
    # 读取上海数据[1,3](@ref)
    se_cols = {'A股代码':'code', 'buydate':'buydate'}
    se_df = pd.read_excel(select_path,
        usecols=list(se_cols.keys()),
        dtype={'A股代码': str, 'buydate' : str}
    ).rename(columns=se_cols)
    se_df = se_df[se_df['code'].notna()]  # 过滤无A股代码的记录


    # 数据清洗
    if EQUAL_WEIGHT_BUY:
        se_df.drop_duplicates(subset=['code'], keep='first', inplace=True) #每支股票只买一次，等权购买，如果注释也就是可以不等权
    se_df.sort_values(by='buydate', inplace=True)
    
    return se_df[['code', 'buydate']]


# ====================== 测试代码 ======================
def test_portfolio_simulator(ini_cash=5000000) -> pd.DataFrame:
    """测试投资组合模拟器"""
    """还缺少高送转后share的增加"""

    # 初始化组合
    simulator = PortfolioSimulator(
        initial_cash=ini_cash,
        start_date="20180703",
        isSaveStock = False #excel中是否要存储各股票每天的价格，用作手工校验
    )
    
    # 从excel创建股票订单
    buy_df = get_portfolio_stocks(base_path /f"..\input\buy_roe_dv_20180701.xlsx").dropna(axis=0, how='any')
    # 订单数量，用于计算每次等权购买可用的资金
    order_amount = len(buy_df)
    buymoney = ini_cash /order_amount * 2
    log.info(f"初始金额{ini_cash}，订单数{order_amount}，等权买入金额{buymoney}")

    for _, row in buy_df.iterrows():
        code = row['code']
        buydate = row['buydate']
        
        #amount = row['amount']
        #amount_100 = math.floor(amount/100)*100
        #if amount_100<100 : amount_100=100            
        simulator.buy_stock(code, buydate, 0)
    
    # 运行回测
    end_date = "20250704"
    results = simulator.run_backtest(end_date)
    
    # 保存结果
    results.to_excel(base_path /f"..\output\portfolio_backtest_{end_date}.xlsx", index=False)
    print(f"回测完成，结果已保存到 portfolio_backtest_{end_date}.xlsx")
    
    # 打印最后5天结果
    print("\n最后5天组合表现:")
    print(results[['date', 'net_value', 'total_value', 'return']].tail())
    
    return results

if __name__ == "__main__":
    # 运行测试
    IS_MYSQL = True
    test_results = test_portfolio_simulator(5000000)
