import akshare as ak
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from typing import List

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

def df_dflist(df, n) -> list[pd.DataFrame]:
    """
    将DataFrame分割后，将各块数据放入列表，列表元素为DataFrame。
    """
    col=df.columns
    Z_array=df.values
    ls_np=np.array_split(Z_array,n,axis=0)   
    ls_df=[pd.DataFrame(i,columns=col) for i in ls_np]
    return ls_df  

def safe_array_split(df_list: List[pd.DataFrame], n_chunks: int) -> List[List[pd.DataFrame]]:
    """
    安全分割DataFrame列表中的每个DataFrame
    
    参数:
        df_list: 包含多个DataFrame的列表
        n_chunks: 每个DataFrame要分割的块数
    
    返回:
        嵌套列表，外层是原始列表顺序，内层是分割后的DataFrame块
    """
    result = []
    for df in df_list:
        # 检查类型：如果是元组则转换为DataFrame
        if isinstance(df, tuple):
            try:
                df = pd.DataFrame(df)  # 尝试将元组转为DataFrame
            except Exception as e:
                print(f"转换元组失败: {e}")
                result.append([])  # 跳过无效数据
                continue
        # 确保处理的是DataFrame
        if not isinstance(df, pd.DataFrame):
            print(f"警告：跳过非DataFrame元素（类型：{type(df)}）")
            result.append([])
            continue
                
            # 计算实际分块数（不超过数据长度）
            actual_chunks = min(n_chunks, len(df))
            # 计算基础分块大小（至少为1）
            base_size = max(1, len(df) // actual_chunks)
            # 计算余数（需要额外分配的行数）
            remainder = len(df) % actual_chunks
            
            # 智能分块算法
            start = 0
            df_chunks = []
            for i in range(actual_chunks):
                # 计算当前分块大小（前remainder个分块多1行）
                current_size = base_size + (1 if i < remainder else 0)
                # 获取分块
                chunk = df.iloc[start:start + current_size].copy()
                # 添加到结果列表
                df_chunks.append(chunk)
                # 更新起始位置
                start += current_size
            
            result.append(df_chunks)
    return result

if __name__ == "__main__":
    print(f"上一个交易日：{get_last_trade_dates()}")  # 输出示例：20250611
    df = []

