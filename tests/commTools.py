import akshare as ak
from datetime import datetime, timedelta
import pandas as pd

# 获取全量历史交易日数据（截止到2025年）

def get_last_trade_dates() -> str:

    last_trade_dates = ""
    trade_dates = ak.tool_trade_date_hist_sina()

    # 转换为日期格式并排序
    trade_dates['trade_date'] = pd.to_datetime(trade_dates['trade_date'])
    today = datetime.now().strftime("%Y%m%d") # 当前日期
    last_trade_date = trade_dates[trade_dates['trade_date'] < today].iloc[-1]['trade_date'].strftime("%Y%m%d")

    return last_trade_date


def merge_on_date_str_index(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    高效合并日期索引(df1)与字符串日期列(df2)的DataFrame
    
    参数:
        df1: 以日期为索引的基准DataFrame (index=DatetimeIndex)
        df2: 包含'date'列的被合并DataFrame (date列格式: 'YYYYMMDD'或'YYYY-MM-DD')
    
    返回:
        合并后的DataFrame (保持df1的索引和行顺序)
    """
    # 类型检查与预处理
    if not isinstance(df1.index, pd.DatetimeIndex):
        raise TypeError("df1的索引必须是DatetimeIndex")
    
    if 'date' not in df2.columns:
        raise ValueError("df2必须包含'date'列")
    
    # 核心优化步骤
    try:
        # 1. 转换df2的date列为datetime (避免循环，向量化操作)
        df2 = df2.copy()  # 避免修改原数据
        df2['_date_'] = pd.to_datetime(df2['date'], errors='coerce')  # 静默转换失败为NaT
        
        # 2. 删除无效日期 (提升后续合并效率)
        valid_dates = df2.dropna(subset=['_date_'])
        if len(valid_dates) == 0:
            return df1  # 无有效日期时返回原df1
        
        # 3. 设置临时索引 (加速合并)
        with_date_index = valid_dates.set_index('_date_')
        
        # 4. 执行索引合并 (比merge快3-5倍)
        merged = df1.join(with_date_index, how='left')
        
        # 5. 清理临时列
        merged.drop(columns=['date'], errors='ignore', inplace=True)
        
        return merged
    
    except Exception as e:
        print(f"合并错误: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    print(f"上一个交易日：{get_last_trade_dates()}")  # 输出示例：20250611

