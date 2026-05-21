import os
import pymysql
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import log4ak
from getAllStock import get_select_stocks

base_path = Path(__file__).parent
log = log4ak.LogManager(log_level=log4ak.INFO)

DB_CONFIG = {
    'host': 'localhost',
    'user': 'powerbi',
    'password': 'longyu',
    'database': 'akshare',
    'port': 3306,
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

class VectorizedValuationCalculator:
    """
    基于 Pandas 向量化计算的重构版估值计算器
    解决了原版严重的 N+1 数据库查询问题，性能提升数个数量级。
    """
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.connection = None

    def connect(self):
        """建立或确保数据库连接活跃"""
        if self.connection is None or not self.connection.open:
            try:
                self.connection = pymysql.connect(**self.db_config)
                log.info(f"✅ 数据库连接成功 | MySQL版本:{self.connection.get_server_info()}")
            except Exception as e:
                log.error(f"数据库连接失败: {e}")
                raise
        else:
            self.connection.ping(reconnect=True)

    def disconnect(self):
        if self.connection and self.connection.open:
            self.connection.close()

    def get_historical_price_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """批量获取股票指定时间段的价格数据"""
        self.connect()
        sql = """
            SELECT date, close
            FROM stock_historical_data 
            WHERE stock_code = %s AND date BETWEEN %s AND %s 
            ORDER BY date ASC
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql, (stock_code, start_date, end_date))
                result = cursor.fetchall()
                if not result:
                    return pd.DataFrame()
                
                df = pd.DataFrame(result)
                df['date'] = pd.to_datetime(df['date'])
                df['close'] = df['close'].astype(float)
                df.set_index('date', inplace=True)
                return df
        except Exception as e:
            log.error(f"获取股票 {stock_code} 价格失败: {e}")
            return pd.DataFrame()

    def get_financial_reports(self, stock_code: str, min_date: str) -> pd.DataFrame:
        """
        批量获取该股票在一定日期后的所有财报数据。
        多取几年（如3年前），以便计算年初及TTM指标。
        """
        self.connect()
        sql = """
            SELECT report_date, weighted_eps, diluted_eps, 
                   adjusted_net_asset, net_asset_per_share, main_profit
            FROM stock_financial_reports 
            WHERE stock_code = %s AND report_date >= %s
            ORDER BY report_date ASC
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql, (stock_code, min_date))
                result = cursor.fetchall()
                if not result:
                    return pd.DataFrame()
                
                df = pd.DataFrame(result)
                df['report_date'] = pd.to_datetime(df['report_date'])
                for col in ['weighted_eps', 'diluted_eps', 'adjusted_net_asset', 'net_asset_per_share','main_profit']:
                    if col in df.columns:
                        df[col] = df[col].astype(float)
                df.set_index('report_date', inplace=True)
                return df
        except Exception as e:
            log.error(f"获取股票 {stock_code} 财报失败: {e}")
            return pd.DataFrame()

    def get_dividend_data(self, stock_code: str) -> pd.DataFrame:
        """批量获取所有已实施的分红记录"""
        self.connect()
        sql = """
            SELECT equity_reg_date, cash_dividend
            FROM dividend_info 
            WHERE stock_code = %s AND cash_dividend > 0 
            AND progress = '实施'
            ORDER BY equity_reg_date ASC
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql, (stock_code,))
                result = cursor.fetchall()
                if not result:
                    return pd.DataFrame()
                
                df = pd.DataFrame(result)
                df['equity_reg_date'] = pd.to_datetime(df['equity_reg_date'])
                df['cash_dividend'] = df['cash_dividend'].astype(float)
                # 每10股分红 -> 每股分红
                df['dividend_per_share'] = df['cash_dividend'] / 10.0
                return df
        except Exception as e:
            log.error(f"获取股票 {stock_code} 分红失败: {e}")
            return pd.DataFrame()

    def prepare_financial_indicators(self, fin_df: pd.DataFrame) -> pd.DataFrame:
        """
        预处理财报数据，计算每次发布财报时的 TTM EPS。
        """
        if fin_df.empty:
            return fin_df

        # 统一核心字段
        fin_df['eps'] = fin_df['weighted_eps'].fillna(fin_df['diluted_eps'])
        fin_df['navps'] = fin_df['adjusted_net_asset'].fillna(fin_df['net_asset_per_share'])

        eps_ttm_list = []
        
        for r_date, row in fin_df.iterrows():
            r_year = r_date.year
            r_month = r_date.month
            
            # 如果是年报，TTM EPS 就是当前的 EPS
            if r_month == 12:
                eps_ttm_list.append(row['eps'])
                continue
                
            # 计算滚动 TTM：最新季度累计 + 上年全年 - 上年同期累计
            # 获取上年年报 (去年12月)
            ly_ann = fin_df[(fin_df.index.year == r_year - 1) & (fin_df.index.month == 12)]
            ann_eps = ly_ann['eps'].iloc[-1] if not ly_ann.empty else np.nan
            
            # 获取上年同期季报
            ly_per = fin_df[(fin_df.index.year == r_year - 1) & (fin_df.index.month == r_month)]
            per_eps = ly_per['eps'].iloc[-1] if not ly_per.empty else np.nan
            
            if pd.notna(ann_eps) and pd.notna(per_eps) and pd.notna(row['eps']):
                eps_ttm = row['eps'] + ann_eps - per_eps
                eps_ttm_list.append(max(eps_ttm, 0.0))  # 确保非负
            else:
                eps_ttm_list.append(np.nan)

        fin_df['eps_ttm'] = eps_ttm_list
        return fin_df

    def process_stock(self, stock_code: str, stock_name: str, start_date: str, end_date: str) -> pd.DataFrame:
        """核心向量化计算引擎"""
        # 1. 拉取交易日历与价格
        price_df = self.get_historical_price_data(stock_code, start_date, end_date)
        if price_df.empty:
            log.warning(f"[{stock_code}] {stock_name} 无价格数据。")
            return pd.DataFrame()

        # 2. 拉取财报（多取3年以确保有上年/上上年数据用于TTM计算）
        start_dt = pd.to_datetime(start_date)
        fin_min_date = (start_dt - pd.DateOffset(years=3)).strftime('%Y-%m-%d')
        fin_df = self.get_financial_reports(stock_code, fin_min_date)
        
        # 预计算 TTM 指标
        fin_df = self.prepare_financial_indicators(fin_df)

        # 3. 将财报数据对齐到交易日 (Backward As-of Join)
        if not fin_df.empty:
            # 提取年报专门用于静态 PE、PB 计算
            annual_fin = fin_df[fin_df.index.month == 12][['eps', 'navps']].dropna(how='all')
            if not annual_fin.empty:
                price_df = pd.merge_asof(price_df, annual_fin, left_index=True, right_index=True, direction='backward')
                price_df.rename(columns={'eps': 'annual_eps'}, inplace=True)
            else:
                price_df['annual_eps'] = np.nan
                price_df['navps'] = np.nan

            # 提取最新财报专门用于 TTM 计算
            latest_fin = fin_df[['eps_ttm']].dropna()
            if not latest_fin.empty:
                price_df = pd.merge_asof(price_df, latest_fin, left_index=True, right_index=True, direction='backward')
            else:
                price_df['eps_ttm'] = np.nan
        else:
            price_df['annual_eps'] = np.nan
            price_df['navps'] = np.nan
            price_df['eps_ttm'] = np.nan

        # 4. 计算股息率
        div_df = self.get_dividend_data(stock_code)
        dv_ratio_s = pd.Series(0.0, index=price_df.index)
        dv_ttm_s = pd.Series(0.0, index=price_df.index)

        if not div_df.empty:
            # 根据规则，下一年1月的分红计入本财年
            def get_fiscal_year(date):
                return date.year - 1 if date.month == 1 else date.year
            
            div_df['fiscal_year'] = div_df['equity_reg_date'].apply(get_fiscal_year)
            yearly_div = div_df.groupby('fiscal_year')['dividend_per_share'].sum()
            
            # price_df 的 fiscal year 为当年减 1
            price_fy = price_df.index.year - 1
            dv_ratio_s = price_fy.map(yearly_div).fillna(0.0)

            # 滚动12个月股息率 (365天)
            div_daily = div_df.groupby('equity_reg_date')['dividend_per_share'].sum()
            idx_start = price_df.index.min() - pd.Timedelta(days=365)
            idx_end = price_df.index.max()
            full_idx = pd.date_range(idx_start, idx_end)
            
            div_daily = div_daily.reindex(full_idx, fill_value=0.0)
            rolling_div = div_daily.rolling(window=365, min_periods=1).sum()
            dv_ttm_s = rolling_div.reindex(price_df.index).fillna(0.0)

        # 5. 向量化运算所有估值指标
        price_df['pe'] = np.where((price_df['annual_eps'] > 0), price_df['close'] / price_df['annual_eps'], np.nan)
        price_df['pe_ttm'] = np.where((price_df['eps_ttm'] > 0), price_df['close'] / price_df['eps_ttm'], np.nan)
        price_df['pb'] = np.where((price_df['navps'] > 0), price_df['close'] / price_df['navps'], np.nan)
        
        price_df['dv_ratio'] = np.where(price_df['close'] > 0, (dv_ratio_s / price_df['close']) * 100, 0.0)
        price_df['dv_ttm'] = np.where(price_df['close'] > 0, (dv_ttm_s / price_df['close']) * 100, 0.0)

        # 处理 PS 及冗余字段
        # 注意: 除非有真实的 revenue，否则用 main_profit 算市销率是错误的，此处置空处理。
        price_df['ps'] = None
        price_df['ps_ttm'] = None
        price_df['total_mv'] = None
        
        # 整理输出格式
        out_df = price_df[['close', 'pe', 'pe_ttm', 'pb', 'dv_ratio', 'dv_ttm', 'ps', 'ps_ttm', 'total_mv']].copy()
        
        # 格式化，保留4位小数，NaN 转 None(为了 MySQL 兼容)
        for col in ['pe', 'pe_ttm', 'pb', 'dv_ratio', 'dv_ttm']:
            out_df[col] = out_df[col].round(4).astype(object).where(pd.notnull(out_df[col]), None)
            
        out_df.reset_index(inplace=True)
        out_df.rename(columns={'date': 'trade_date'}, inplace=True)
        out_df['stock_code'] = stock_code
        out_df['stock_name'] = stock_name
        
        return out_df

    def save_batch_valuation(self, df: pd.DataFrame):
        """批量保存 (ON DUPLICATE KEY UPDATE)"""
        if df.empty:
            return
            
        self.connect()
        sql = """
            INSERT INTO stock_pe_history (
                stock_code, stock_name, trade_date, pe, pe_ttm, pb, 
                dv_ratio, dv_ttm, ps, ps_ttm, total_mv
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                stock_name = VALUES(stock_name),
                pe = VALUES(pe),
                pe_ttm = VALUES(pe_ttm),
                pb = VALUES(pb),
                dv_ratio = VALUES(dv_ratio),
                dv_ttm = VALUES(dv_ttm),
                ps = VALUES(ps),
                ps_ttm = VALUES(ps_ttm),
                total_mv = VALUES(total_mv);
        """
        
        records = [
            (
                row.stock_code, row.stock_name, row.trade_date.strftime('%Y-%m-%d'),
                row.pe, row.pe_ttm, row.pb, row.dv_ratio, row.dv_ttm,
                row.ps, row.ps_ttm, row.total_mv
            )
            for row in df.itertuples(index=False)
        ]
        
        try:
            with self.connection.cursor() as cursor:
                cursor.executemany(sql, records)
                self.connection.commit()
                log.info(f"✅ 成功写入/更新 {df.iloc[0].stock_code} 共 {len(records)} 条估值记录")
        except Exception as e:
            self.connection.rollback()
            log.error(f"❌ 批量保存失败: {e}")

def run_valuation_job():
    start_date = '2025-12-31'
    end_date = '2026-05-20'
    
    calculator = VectorizedValuationCalculator(DB_CONFIG)
    try:
        select_df = get_select_stocks()
        if select_df is None or select_df.empty:
            log.error("未获取到自选股票列表，退出。")
            return
            
        log.info(f"开始计算 {len(select_df)} 只股票的估值 (向量化模式)")
        
        for idx, row in select_df.iterrows():
            stock_code = row['代码']
            stock_name = row['名称']
            
            try:
                res_df = calculator.process_stock(stock_code, stock_name, start_date, end_date)
                calculator.save_batch_valuation(res_df)
            except Exception as e:
                log.error(f"处理 {stock_name}({stock_code}) 时出现异常: {e}")
                
    finally:
        calculator.disconnect()
        log.info("所有任务执行完毕。")

if __name__ == "__main__":
    run_valuation_job()
