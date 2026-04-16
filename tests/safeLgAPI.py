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
    
    功能说明：
        该函数用于安全地获取指定指数的市盈率数据。当原始AKShare函数因日期转换
        问题而失败时，会自动降级使用修复方法重新获取数据。
    
    解决的问题：
        API返回的日期是字符串格式（如"2024-01-01"），但原始函数尝试将其解析为
        毫秒时间戳，导致"non convertible value with the unit 'ms'"错误。
    
    参数：
        symbol (str): 指数代码，默认为"沪深300"。支持的指数包括：
                     - 上证50, 沪深300, 上证380, 创业板50, 中证500
                     - 上证180, 深证红利, 深证100, 中证1000
                     - 上证红利, 中证100, 中证800
    
    返回值：
        pd.DataFrame: 包含日期、指数价格、市盈率等信息的数据框
    
    异常处理：
        - 如果是日期转换错误，自动使用修复的方法重获数据
        - 如果是其他ValueError异常，直接抛出
        - 如果修复方法也失败，返回空DataFrame
    """
    try:
        # 先尝试原始的AKShare函数
        df = ak.stock_index_pe_lg(symbol)
        return df
    except ValueError as e:
        # 检测日期转换错误的特征信息
        if "non convertible value" in str(e) and "with the unit 'ms'" in str(e):
            print(f"检测到日期转换错误，使用修复方法获取 {symbol} 数据")
            # 降级使用修复方法重新获取数据
            return _stock_index_pe_lg_fixed(symbol)
        else:
            # 其他异常直接抛出
            raise e

def _stock_index_pe_lg_fixed(symbol: str) -> pd.DataFrame:
    """
    修复日期转换问题的内部函数（私有函数）
    
    工作原理：
        1. 根据指数名称查找对应的代码（如"沪深300" -> "000300.SH"）
        2. 生成API所需的token（带日期的加密参数）
        3. 构造HTTP请求头和Cookie信息
        4. 直接访问legulegu.com API获取原始JSON数据
        5. 正确处理日期字符串（不转换为时间戳）
        6. 进行时区转换（UTC -> Asia/Shanghai)
        7. 重命名列为中文名称并返回
    
    参数：
        symbol (str): 指数名称（中文），如"沪深300"
    
    返回值：
        pd.DataFrame: 处理后的指数市盈率数据，列名为中文
    
    异常处理：
        如果任何步骤失败，捕获异常并打印堆栈跟踪，最后返回空DataFrame
    """
    try:
        # ============ 步骤1: 准备请求参数 ============
        # 构建指数名称到代码的映射表
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
        
        # 验证输入的指数是否支持
        if symbol not in symbol_map:
            raise ValueError(f"不支持的指数: {symbol}")
        
        # ============ 步骤2: 生成API所需的token ============
        # token是通过JavaScript函数生成的加密参数，包含当前日期信息
        # 模拟原AKShare中的逻辑
        from akshare.stock_feature.stock_a_pe_and_pb import py_mini_racer, hash_code
        # 创建JavaScript运行环境
        js_functions = py_mini_racer.MiniRacer()
        # 加载加密hash函数
        js_functions.eval(hash_code)
        # 使用当前日期生成token
        token = js_functions.call("hex", datetime.now().date().isoformat()).lower()
        
        # ============ 步骤3: 准备HTTP请求 ============
        url = "https://legulegu.com/api/stockdata/index-basic-pe"
        params = {"token": token, "indexCode": symbol_map[symbol]}
        
        # 4. 获取cookie和csrf token
        from akshare.stock_feature.stock_a_pe_and_pb import get_cookie_csrf
        # 获取访问legulegu网站所需的Cookie
        cookies = get_cookie_csrf(url="https://legulegu.com/stockdata/sz50-ttm-lyr")
        
        # ============ 步骤5: 发送HTTP请求获取数据 ============
        # 构造请求头，模拟正常浏览器请求
        headers = {
            # 设置User-Agent伪装成Chrome浏览器
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            # Origin和Referer用于通过CORS和反爬虫检查
            'Origin': 'https://legulegu.com',
            'Referer': f'https://legulegu.com/stockdata/{symbol_map[symbol].lower().replace(".", "")}-ttm-lyr',
        }
        
        # 发送GET请求，30秒超时
        response = requests.get(
            url,
            params=params,
            headers=headers,
            cookies=cookies.get('cookies', {}),
            timeout=30
        )
        
        # 检查HTTP状态码，如果不是200会抛出异常
        response.raise_for_status()
        # 解析JSON响应
        data_json = response.json()
        
        # ============ 步骤6: 将原始JSON转换为DataFrame ============
        # 从response的data字段提取列表数据创建DataFrame
        temp_df = pd.DataFrame(data_json["data"])
        
        # ============ 步骤7: 处理日期列（关键修改点） ============
        # 这是修复原函数bug的核心：直接解析字符串格式的日期，而不是转换毫秒时间戳
        if 'date' in temp_df.columns:
            # 先尝试直接解析为日期字符串（自动识别格式）
            try:
                # 将date列从字符串解析为datetime对象
                # pd.to_datetime会自动识别常见的日期字符串格式
                temp_df['date'] = pd.to_datetime(temp_df['date'])
            except Exception as e:
                print(f"日期解析异常: {e}")
                # 如果自动解析失败，尝试指定%Y-%m-%d格式
                # errors='coerce'会将解析失败的值转为NaT
                temp_df['date'] = pd.to_datetime(temp_df['date'], errors='coerce', format='%Y-%m-%d')
        
        # ============ 步骤8: 时区处理 ============
        # API返回的时间通常是UTC时间，需要转换为中国东部时区
        # 如果datetime还没有时区信息，先添加UTC时区
        if temp_df['date'].dt.tz is None:
            temp_df['date'] = temp_df['date'].dt.tz_localize('UTC')
        # 将UTC时间转换为Asia/Shanghai时区（北京时间）
        temp_df['date'] = temp_df['date'].dt.tz_convert('Asia/Shanghai').dt.date
        
        # ============ 步骤9: 选择并重命名列 ============
        # 只保留需要的列，并将列名改为中文便于使用
        # 按需要的顺序选择列
        temp_df = temp_df[
            [
                "date",          # 交易日期
                "close",         # 指数收盘价
                "lyrPe",         # 等权静态市盈率（Last Year Earnings PE）
                "addLyrPe",      # 普通静态市盈率
                "middleLyrPe",   # 静态市盈率中位数
                "ttmPe",         # 等权滚动市盈率（Trailing Twelve Months PE）
                "addTtmPe",      # 普通滚动市盈率
                "middleTtmPe",   # 滚动市盈率中位数
            ]
        ]
        
        # 将英文列名改为中文，便于使用和理解
        temp_df.columns = [
            "日期",              # 交易日期
            "指数",              # 指数收盘价
            "等权静态市盈率",     # 使用权重平均的PE（基于过去12个月收益）
            "静态市盈率",         # 简单平均的静态PE
            "静态市盈率中位数",   # 静态PE的中位数
            "等权滚动市盈率",     # 使用权重平均的PE（基于最近12个月收益）
            "滚动市盈率",         # 简单平均的滚动PE
            "滚动市盈率中位数",   # 滚动PE的中位数
        ]
        
        # 返回处理后的DataFrame
        return temp_df
        
    except Exception as e:
        # 捕获并打印所有异常信息，便于调试
        print(f"修复方法获取 {symbol} 数据失败: {e}")
        # 打印完整的堆栈跟踪
        traceback.print_exc()
        # 返回空DataFrame而不是抛出异常，让程序能继续运行
        return pd.DataFrame()


if __name__ == "__main__":
    # 测试修复函数的主程序
    print("测试修复函数...")
    # 测试获取上证50指数的市盈率数据
    df = safe_stock_index_pe_lg('上证50')
    if not df.empty:
        # 显示前5行数据
        print(f"上证50数据预览:\n{df.head()}")
        # 显示每列的数据类型
        print(f"\n数据类型:\n{df.dtypes}")
        # 显示数据的时间范围
        print(f"\n日期范围: {df['日期'].min()} 到 {df['日期'].max()}")
    else:
        # 如果获取失败，显示错误提示
        print("获取数据失败")