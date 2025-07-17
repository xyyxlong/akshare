import akshare as ak
import time
import pandas as pd
import json
from typing import Optional

def safe_get_hk_stock_pe(stock_code: str, indicator: str) -> Optional[pd.DataFrame]:
    """
    安全获取港股指标数据（带错误处理和重试机制）
    
    :param stock_code: 港股代码（如'hk0700'）
    :param indicator: 指标名称（如'市盈率'）
    :return: 成功返回DataFrame，失败返回None
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 尝试获取数据
            raw_data = ak.stock_hk_indicator_eniu(stock_code, indicator)
            
            # 验证数据有效性
            if isinstance(raw_data, pd.DataFrame):
                if not raw_data.empty:
                    return raw_data
                print(f"警告：获取到空DataFrame（第{attempt+1}次尝试）")
            elif isinstance(raw_data, str):
                try:
                    # 尝试解析可能的JSON字符串
                    json_data = json.loads(raw_data)
                    return pd.DataFrame(json_data)
                except json.JSONDecodeError:
                    print(f"错误：无法解析返回的字符串数据（第{attempt+1}次尝试）")
            else:
                print(f"错误：未知返回类型 {type(raw_data)}（第{attempt+1}次尝试）")
                
        except json.JSONDecodeError as e:
            print(f"JSON解析失败（第{attempt+1}次尝试）：{str(e)}")
        except Exception as e:
            print(f"请求异常（第{attempt+1}次尝试）：{str(e)}")
        
        # 指数退避重试
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)
    
    print(f"全部{max_retries}次尝试失败，请检查：")
    print(f"1. 股票代码是否正确（当前：{stock_code}）")
    print(f"2. 指标名称是否支持（当前：{indicator}）")
    print("3. 网络连接是否正常")
    return None

# 使用示例
if __name__ == "__main__":
    # 获取腾讯控股市盈率
    df = safe_get_hk_stock_pe("hk00700", "市盈率")
    
    if df is not None:
        print("成功获取数据：")
        print(df)
    else:
        print("数据获取失败，请检查错误信息")