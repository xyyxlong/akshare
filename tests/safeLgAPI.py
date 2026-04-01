import akshare as ak
import pandas as pd
import requests
import json
from datetime import datetime
from typing import Optional
import traceback

def safe_stock_index_pe_lg(symbol: str = "沪深300") -> pd.DataFrame:
    """
    修复版的指数市盈率获取函数
    解决日期转换问题：API返回的是日期字符串，不是毫秒时间戳
    """
    try:
        # 先尝试原始函数
        df = ak.stock_index_pe_lg(symbol)
        return df
    except ValueError as e:
        if "non convertible value" in str(e) and "with the unit 'ms'" in str(e):
            print(f"检测到日期转换错误，使用修复方法获取 {symbol} 数据")
            return _stock_index_pe_lg_fixed(symbol)
        else:
            raise e

def _stock_index_pe_lg_fixed(symbol: str) -> pd.DataFrame:
    """修复日期转换问题的内部函数"""
    try:
        # 1. 准备请求参数
        symbol_map = {
            "上证50": "000016.SH",
            "沪深300": "000300.SH",
            "上证380": "000009.SH",
            "创业板50": "399673.SZ",
            "中证500": "000905.SH",
            "上证180": "000010.SH",
            "深证红利": "399324.SZ",
            "深证100": "399330.SZ",
            "中证1000": "000852.SH",
            "上证红利": "000015.SH",
            "中证100": "000903.SH",
            "中证800": "000906.SH",
        }
        
        if symbol not in symbol_map:
            raise ValueError(f"不支持的指数: {symbol}")
        
        # 2. 生成token（模拟原AKShare逻辑）
        from akshare.stock_feature.stock_a_pe_and_pb import py_mini_racer, hash_code
        js_functions = py_mini_racer.MiniRacer()
        js_functions.eval(hash_code)
        token = js_functions.call("hex", datetime.now().date().isoformat()).lower()
        
        # 3. 准备请求
        url = "https://legulegu.com/api/stockdata/index-basic-pe"
        params = {"token": token, "indexCode": symbol_map[symbol]}
        
        # 4. 获取cookie和csrf token
        from akshare.stock_feature.stock_a_pe_and_pb import get_cookie_csrf
        cookies = get_cookie_csrf(url="https://legulegu.com/stockdata/sz50-ttm-lyr")
        
        # 5. 发送请求
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Origin': 'https://legulegu.com',
            'Referer': f'https://legulegu.com/stockdata/{symbol_map[symbol].lower().replace(".", "")}-ttm-lyr',
        }
        
        response = requests.get(
            url,
            params=params,
            headers=headers,
            cookies=cookies.get('cookies', {}),
            timeout=30
        )
        
        response.raise_for_status()
        data_json = response.json()
        
        # 6. 处理数据
        temp_df = pd.DataFrame(data_json["data"])
        
        # 7. 修复日期处理 - 关键修改
        if 'date' in temp_df.columns:
            # 先尝试直接解析为日期字符串
            try:
                # 如果date是字符串格式的日期
                temp_df['date'] = pd.to_datetime(temp_df['date'])
            except Exception as e:
                print(f"日期解析异常: {e}")
                # 尝试不同的日期格式
                temp_df['date'] = pd.to_datetime(temp_df['date'], errors='coerce', format='%Y-%m-%d')
        
        # 8. 时区转换
        if temp_df['date'].dt.tz is None:
            temp_df['date'] = temp_df['date'].dt.tz_localize('UTC')
        temp_df['date'] = temp_df['date'].dt.tz_convert('Asia/Shanghai').dt.date
        
        # 9. 重命名列
        temp_df = temp_df[
            [
                "date",
                "close",
                "lyrPe",
                "addLyrPe",
                "middleLyrPe",
                "ttmPe",
                "addTtmPe",
                "middleTtmPe",
            ]
        ]
        
        temp_df.columns = [
            "日期",
            "指数",
            "等权静态市盈率",
            "静态市盈率",
            "静态市盈率中位数",
            "等权滚动市盈率",
            "滚动市盈率",
            "滚动市盈率中位数",
        ]
        
        return temp_df
        
    except Exception as e:
        print(f"修复方法获取 {symbol} 数据失败: {e}")
        traceback.print_exc()
        return pd.DataFrame()


if __name__ == "__main__":
    # 测试修复函数
    print("测试修复函数...")
    df = safe_stock_index_pe_lg('上证50')
    if not df.empty:
        print(f"上证50数据预览:\n{df.head()}")
        print(f"\n数据类型:\n{df.dtypes}")
        print(f"\n日期范围: {df['日期'].min()} 到 {df['日期'].max()}")
    else:
        print("获取数据失败")