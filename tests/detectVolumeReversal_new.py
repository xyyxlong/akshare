import os
import time
from pathlib import Path
from typing import List, Tuple, Union, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import akshare as ak

# Local imports (assuming these exist in the project)
from getAllStock import get_all_stocks, get_select_stocks
import get_industry_historyPE as gi
import get_stockPE_his as gs
import insertStockHist as ish
import insert_major_index_valuation as imiv
import commTools as ct
import log4ak

log = log4ak.LogManager(log_level=log4ak.INFO)


@dataclass
class DetectConfig:
    """Configuration class to replace global variables"""
    max_consecutive_errors: int = 3
    out_time: int = 5
    all_pe_doc_num: int = 1
    keep_day: int = 3
    do_dup: float = 0.12  # 连续5个交易日放量平均增长率12%
    ground_volume_percentile: int = 5  # 地价地量阈值检测时间内5%分位
    n_years: int = 3  # 回测N_YEARS年百分位
    start_date: str = "20160701"
    is_my_select: bool = False
    is_all: bool = False
    is_mysql: bool = True
    is_buy: bool = False
    buy_with_pe_percentile: bool = False
    pe_rolling_time: int = 3
    pe_percentile: int = 5
    equal_weight_buy: bool = True
    is_buy_k: bool = True
    pe300_percentile_year: int = 3
    base_path: Path = Path(__file__).parent


class VolumeReversalDetector:
    def __init__(self, config: DetectConfig):
        self.config = config
        # Lazy loading for index PE
        self._df_hs300_pe_ttm = None

    @property
    def df_hs300_pe_ttm(self):
        if self.config.is_buy_k and self._df_hs300_pe_ttm is None:
            self._df_hs300_pe_ttm = imiv.get_index_pe_his('沪深300')
        return self._df_hs300_pe_ttm

    def get_stock_data(self, code: str) -> pd.DataFrame:
        """获取股票历史数据"""
        code_clean = code.split('.')[0]
        try:
            if self.config.is_mysql:
                df = ish.get_stock_data_from_mysql(code_clean, '')
                df = df[['日期', '收盘', '成交额']].copy()
                start_date_dt = pd.to_datetime(self.config.start_date, format='%Y%m%d')
                df['日期'] = pd.to_datetime(df['日期'], format='%Y%m%d')
                df = df[df['日期'] > start_date_dt]
            else:
                df = ak.stock_zh_a_hist(
                    symbol=code_clean,
                    period="daily",
                    adjust="qfq",
                    start_date=self.config.start_date
                )
                df = df[['日期', '收盘', '成交额']].copy()

            df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
            df = df.dropna(subset=['日期'])
            df.set_index('日期', inplace=True)
            return df
        except Exception as e:
            log.error(f"获取股票 {code} 数据失败: {e}")
            return pd.DataFrame()

    def detect_price_volume_reversal(self, stock_list: pd.DataFrame) -> Union[List[pd.DataFrame], List[Dict]]:
        result = []
        signals_list = []
        rolling_window = self.config.n_years * 243

        for row in stock_list.itertuples(index=False):
            code, name = row.代码, row.名称
            df = self.get_stock_data(code)
            if df.empty:
                continue

            hist_data = df[df['成交额'] > 0].copy().sort_index()
            
            if len(hist_data) < rolling_window:
                log.debug(f"{code} 数据不足 {rolling_window} 天，跳过")
                continue

            # Vectorized calculations for rolling rank
            # Using bottleneck if available, or pandas rolling rank
            hist_data['price_rank'] = hist_data['收盘'].rolling(rolling_window).rank(pct=True)
            hist_data['p_mask'] = hist_data['price_rank'] < (self.config.ground_volume_percentile / 100)

            hist_data['volume_rank'] = hist_data['成交额'].rolling(rolling_window).rank(pct=True)
            hist_data['v_mask'] = hist_data['volume_rank'] < (self.config.ground_volume_percentile / 100)

            hist_data['v_growth'] = hist_data['成交额'].pct_change() + 1
            # Check if rolling mean of conditions >= 1+DODUP is 1 (all true in window)
            hist_data['vg_mask'] = hist_data['v_growth'].rolling(self.config.keep_day).apply(
                lambda x: np.mean(x >= (1 + self.config.do_dup)) == 1, raw=True
            ).fillna(0).astype(bool)

            result.append(hist_data)
            log.info(f"{code}量价分位分析完毕。")

            if self.config.is_buy:
                signals = self.test_buy_signals(code, name, hist_data)
                if signals:
                    signals_list.extend(signals)

            if not self.config.is_mysql:
                time.sleep(0.3)

        return signals_list if self.config.is_buy else result

    def test_buy_signals(self, code: str, name: str, hist_data: pd.DataFrame) -> List[Dict]:
        signals_list = []
        last_valid_signal_date = None
        min_days_between_signals = 30
        excluded_early_signals = 0
        max_early_signals = 2

        if self.config.buy_with_pe_percentile:
            all_stock_pe = gs.calculate_pe_time_percentile(code, self.config.pe_rolling_time)
            hist_data = ct.merge_on_date_str_index(hist_data, all_stock_pe)
            hist_data['pe_mask'] = hist_data.get('pettm_per', 100) < self.config.pe_percentile
            hist_data['both_mask'] = hist_data['pe_mask'] & hist_data['v_mask']
        else:
            hist_data['both_mask'] = hist_data['p_mask'] & hist_data['v_mask']

        # Vectorized block ID generation
        hist_data['block_id'] = (hist_data['both_mask'] != hist_data['both_mask'].shift(1)).cumsum()

        consecutive_mask = hist_data['both_mask'].rolling(window=self.config.keep_day).min() == 1
        consecutive_dates = consecutive_mask[consecutive_mask].index

        current_block = -1
        for date in consecutive_dates:
            block_id = hist_data.loc[date, 'block_id']

            if excluded_early_signals < max_early_signals:
                excluded_early_signals += 1
                continue

            if last_valid_signal_date is not None:
                days_since = (date - last_valid_signal_date).days
                if days_since < min_days_between_signals:
                    continue

            if block_id != current_block:
                pp = hist_data.loc[date, 'pe_ttm'] if self.config.buy_with_pe_percentile else hist_data.loc[date, '收盘']
                
                try:
                    onedayPE = gs.get_stock_pe(code, date.strftime('%Y%m%d'))
                    dv_ttm = onedayPE['dv_ttm'].iloc[0] if isinstance(onedayPE, pd.DataFrame) and not onedayPE.empty else -1
                except Exception:
                    dv_ttm = -1

                hs300_pct = 0
                if self.config.is_buy_k:
                    try:
                        hs300_pct = imiv.get_pe_percentile(
                            self.df_hs300_pe_ttm, 
                            date.strftime('%Y%m%d'), 
                            imiv.PE_TTM, 
                            self.config.pe300_percentile_year
                        )
                    except Exception:
                        pass

                signals_list.append({
                    'A股代码': code,
                    'buydate': date,
                    'price/pe': pp,
                    'dv_ttm': dv_ttm,
                    '300%': '{:.2f}'.format(hs300_pct),
                    '名称': name,
                })
                last_valid_signal_date = date
                current_block = block_id
                excluded_early_signals = 0

                if self.config.equal_weight_buy:
                    break

        return signals_list

    def get_pe_after_detect(self, result: List[pd.DataFrame], stock_list: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
        resultlist = []
        pass_num = 0

        for idx, row in tqdm(enumerate(stock_list.itertuples(index=False)), total=len(stock_list)):
            code = row.代码
            if idx >= len(result) or result[idx].empty:
                continue
                
            df = result[idx].copy()
            last_row = df.iloc[-1]

            if self.config.is_all or last_row.get('p_mask', False) or last_row.get('v_mask', False) or last_row.get('vg_mask', False):
                try:
                    all_stock_pe = gs.get_stock_pe_his(code)
                    if all_stock_pe is not None and not all_stock_pe.empty:
                        df = pd.merge(df, all_stock_pe, how='left', left_index=True, right_on='日期')
                        
                        # Vectorized rolling PE percentile
                        window_size = self.config.pe_rolling_time * 243
                        if 'pe_ttm' in df.columns:
                            df['stock_pe_percentile'] = df['pe_ttm'].rolling(window_size).rank(pct=True)
                            df['stock_pe_mask'] = (df['stock_pe_percentile'] * 100) <= self.config.pe_percentile
                    
                    df.index = df.index.strftime('%Y%m%d')
                    resultlist.append((str(code), df))
                    pass_num += 1
                except Exception as e:
                    log.error(f"{code} 估值合并异常: {e}")

        log.info(f"今天检测通过数量：{pass_num}")
        return resultlist


def save_to_excel(data: Union[List[Tuple[str, pd.DataFrame]], List[Dict], pd.DataFrame], 
                 filename: Path, 
                 is_signal_list: bool = False,
                 num_chunks: int = 1):
    """Unified save to excel function"""
    if not data:
        log.error("No data to save.")
        return

    # Ensure parent directory exists
    filename.parent.mkdir(parents=True, exist_ok=True)

    try:
        if is_signal_list:
            df = pd.DataFrame(data)
            if 'buydate' in df.columns:
                df['buydate'] = pd.to_datetime(df['buydate']).dt.strftime('%Y%m%d')
            df.to_excel(filename, sheet_name="信号列表", index=False)
            log.info(f"Saved signal list to {filename}")
        else:
            total_records = len(data)
            actual_n = min(num_chunks, total_records)
            base_size = total_records // actual_n
            remainder = total_records % actual_n

            chunks = []
            start_idx = 0
            for i in range(actual_n):
                size = base_size + 1 if i < remainder else base_size
                chunks.append(data[start_idx:start_idx+size])
                start_idx += size

            for i, chunk in enumerate(chunks):
                suffix = f"_{i+1}" if actual_n > 1 else ""
                out_file = filename.with_name(f"{filename.stem}{suffix}{filename.suffix}")
                with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
                    for sheet_name, df in chunk:
                        df.to_excel(writer, sheet_name=sheet_name, index=True)
                log.info(f"Saved chunks to {out_file}")
    except PermissionError:
        log.error(f"Permission denied: {filename} is open.")


def run_pipeline(mode: str, select_file: Path):
    """Unified execution pipeline"""
    config = DetectConfig(
        n_years=5,
        start_date="20160701",
        is_mysql=True,
        is_my_select=False,
        is_all=False,
        pe_rolling_time=5,
        pe_percentile=5
    )

    if mode == 'buy':
        config.is_buy = True
        config.buy_with_pe_percentile = False

    detector = VolumeReversalDetector(config)
    test_stocks = get_select_stocks(str(select_file)) if config.is_my_select else get_select_stocks()
    
    result = detector.detect_price_volume_reversal(test_stocks)
    end_date = datetime.now().strftime("%Y%m%d")
    out_dir = config.base_path.parent / "output" / "detect"

    if mode == 'all_pe':
        write_df = detector.get_pe_after_detect(result, test_stocks)
        fname = out_dir / f"detect_allPE_{end_date}{'_my' if config.is_my_select else ''}.xlsx"
        save_to_excel(write_df, fname, num_chunks=config.all_pe_doc_num)
    elif mode == 'buy':
        fname = out_dir / f"detect_rev_BUY_{end_date}{'_my' if config.is_my_select else ''}.xlsx"
        save_to_excel(result, fname, is_signal_list=True)
    elif mode == 'last_pe':
        # Placeholder for detect_with_lastPE logic if needed
        # Requires logic from save_to_excel_filter
        pass

if __name__ == "__main__":
    base_path = Path(__file__).parent
    my_select_path = base_path.parent / "input" / "selectlist_my.xlsx"
    
    # Run desired mode
    run_pipeline('all_pe', my_select_path)
