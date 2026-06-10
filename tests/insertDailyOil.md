# 石油日度盘后数据抓取与清洗系统——最终详细设计文档 (V3.0)

> **版本更新说明**: V3.0 将数据源从 `yahoo-fin` 更换为 `yfinance`，后者具有更稳定的 API、更好的维护状态和更丰富的功能支持。

---

## 1. 业务背景与系统目标

本系统是一个标准的数据批处理（Batch）管道。核心任务是根据指定的日期范围，从公开金融数据接口抓取原油价格、宏观经济指标等，经过防错清洗与指标衍生后，以覆盖写入（Upsert）的方式存入 MySQL 数据库。

该数据主要服务于量化团队的策略回测，对数据的**连续性**、**准确性**（无异常尖峰）和**幂等性**（重复运行不污染数据）有极高要求。

---

## 2. 整体架构与流程设计

系统采用三阶段流水线设计：**数据采集 -> 数据清洗与特征转化 -> 数据持久化**。

```
[ 外部调用: 触发输入 start_date, end_date ]
                 │
                 ▼
     ┌───────────────────────┐
     │  1. 数据采集模块      │ ──► 过滤非交易日 -> 调用 yfinance API 分头抓取
     └───────────────────────┘
                 │ (按 trade_date 进行 Outer Join 合并)
                 ▼
     ┌───────────────────────┐
     │  2. 清洗与特征转化    │ ──► 1. 剔除负数/0值 (物理错误)
     └───────────────────────┘     ──► 2. IQR 算法消除闪崩噪声
                 │                 ──► 3. 线性插值修补缺失数据
                 │                 ──► 4. 计算均线、价差等衍生指标
                 ▼
     ┌───────────────────────┐
     │  3. 持久化模块        │ ──► 分块拼接 ON DUPLICATE KEY UPDATE
     └───────────────────────┘ ──► 批量写入 MySQL
```

---

## 3. 模块详细设计

### 3.1 数据采集模块 (Fetcher)

#### 3.1.1 yfinance vs yahoo-fin 对比

| 特性 | yahoo-fin | yfinance (推荐) |
|------|-----------|-----------------|
| 维护状态 | 更新较少 | 活跃维护，社区支持好 |
| API 稳定性 | 偶尔失效 | 更稳定，有重试机制 |
| 批量下载 | 不支持 | 支持 `yf.download()` 批量获取 |
| 数据字段 | adjclose | Adj Close (需重命名) |
| 错误处理 | 较弱 | 内置异常处理和日志 |

#### 3.1.2 接口映射关系表

开发人员需调用 `yfinance` 库，按 `trade_date`（交易日期）为基准轴进行多表外连接（Outer Join）合并。

| 数据库目标字段 | 开源库 | 具体代码接口 / Ticker | 提取字段及处理逻辑 |
|---------------|--------|----------------------|-------------------|
| `wti_close` | yfinance | `yf.download("CL=F")` | 提取 `Adj Close` (复权收盘价)，代表美油价格 |
| `brent_close` | yfinance | `yf.download("BZ=F")` | 提取 `Adj Close`，代表布油价格 |
| `usd_index` | yfinance | `yf.download("DX-Y.NYB")` | 提取 `Adj Close`，代表美元指数 |
| `vix_index` | yfinance | `yf.download("^VIX")` | 提取 `Adj Close`，代表恐慌指数 |
| `us_10y_yield` | yfinance | `yf.download("^TNX")` | 提取 `Adj Close`。**注意**：该接口返回的数值可能已为百分比格式(如4.5)，需校验，若大于10则除以10，否则直接使用 |
| `crack_spread` | yfinance | `yf.download("CRAK")` | 提取 `Adj Close`。使用美股原油ETF (CRAK) 作为裂解价差的代理指标 |
| `gpr_index` | akshare | `ak.macro_usa_gpr_index()` | 提取地缘政治风险指数。若 akshare 无此接口，暂时默认填 `NULL` |

#### 3.1.3 yfinance 核心代码示例

```python
import yfinance as yf
import pandas as pd

def fetch_single_ticker(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    使用 yfinance 获取单个 ticker 的历史数据
    
    Args:
        ticker: Yahoo Finance 股票/期货代码
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
    
    Returns:
        DataFrame 包含 trade_date 和目标字段列
    """
    # 方式一：使用 Ticker 对象
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date, auto_adjust=True)
    
    # 方式二：使用 download 函数（推荐，支持批量）
    df = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,      # 自动调整价格（复权）
        progress=False,        # 关闭进度条
        threads=True           # 启用多线程
    )
    
    if df.empty:
        return pd.DataFrame()
    
    # 提取收盘价（auto_adjust=True 时，Close 即为复权价）
    df = df[['Close']].copy()
    df.index.name = 'trade_date'
    df.reset_index(inplace=True)
    df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
    
    return df


def fetch_multiple_tickers(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    批量获取多个 ticker 数据（yfinance 优势功能）
    
    Args:
        tickers: Ticker 列表，如 ['CL=F', 'BZ=F', '^VIX']
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        合并后的 DataFrame
    """
    df = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by='ticker'  # 按 ticker 分组
    )
    
    return df
```

#### 3.1.4 采集核心逻辑

1. **节假日过滤**：在发起请求前，利用 `exchange_calendars` 库或硬编码日历，过滤掉周末及美国法定节假日，避免发起无效请求。

2. **时区与时间对齐**：统一将各类接口返回的日期转化为字符串格式 `YYYY-MM-DD`，并作为主键（基准轴）。

3. **多源合并**：由于美股、商品期货的交易日不完全一致，必须使用 **外连接（Outer Join）** 保留所有有数据的交易日，未对齐的日期产生 `NaN` 留给清洗模块处理。

4. **yfinance 特性利用**：
   - 使用 `auto_adjust=True` 参数自动获取复权价格，无需手动处理
   - 使用 `threads=True` 启用多线程加速批量下载
   - yfinance 内置请求重试机制，稳定性优于 yahoo-fin

---

### 3.2 数据清洗与特征转化模块 (Pipeline)

> ⚠️ **核心警告**：为了防止"未来函数"（即回测时使用了未发生的数据），数据处理必须严格按照时间正序（Ascending）排列后，才能进行滑动窗口计算。

#### 步骤一：基础过滤（剔除物理错误）

**逻辑**：遍历所有价格与指数列，如果发现数值 ≤ 0，直接将该数值置为 `NaN`（视为空值）。

#### 步骤二：IQR (四分位距) 算法剔除闪崩异常值

针对原油价格（`wti_close`, `brent_close`），由于接口偶尔会出现输入错误带来的异常大毛刺，使用以下算法清洗：

1. 设定一个长度为 **20天** 的滚动滑动窗口（Rolling Window）。
2. 在窗口内计算当前序列的第 25 分位数（Q₁）和第 75 分位数（Q₃）。
3. 计算分位距：`IQR = Q₃ - Q₁`。
4. 设定合规范围边界：
   - 下界 `Lower = Q₁ - 1.5 × IQR`
   - 上界 `Upper = Q₃ + 1.5 × IQR`
5. **异常处理**：若当天的价格超出了 `[Lower, Upper]` 范围，判定为异常值，将其强制重置为 `NaN`。
6. **防错机制**：若窗口内有效数据不足（如全是 NaN 导致 IQR=0），则跳过该窗口的异常检测，保留原值或置为 NaN，防止除零报错。

#### 步骤三：缺失值填充（连续性处理）

1. **线性插值（Linear Interpolation）**：针对价格和金融指标列（`wti_close`, `brent_close`, `usd_index`, `vix_index`, `us_10y_yield`），调用 Pandas 的 `interpolate(method='linear')`，根据前后有数据的日期进行线性按比例内插填充。

2. **宏观/风险指数填充**：针对 `gpr_index` 或后续扩充的低频宏观数据，采用前向填充（Forward Fill, `ffill()`），即缺失位置直接沿用最近一个历史有效交易日的值。

#### 步骤四：衍生指标计算（特征工程）

利用清洗完毕后的干净数据，计算以下数据库所需字段：

| 字段名 | 计算逻辑 |
|--------|---------|
| `wti_60dma` | 计算 `wti_close` 过去 60 个交易日的算术平均值（滚动窗口不足60天时，设置 `min_periods=1`，有几天算几天） |
| `brent_wti_spread` | `brent_close - wti_close` |
| `wti_rsi` | 标准 14 日相对强弱指标（RSI），根据 `wti_close` 过去 14 天的涨跌幅比例计算 |
| `term_structure` | **修正逻辑**：需额外获取 WTI 近月合约 (`CL=F`) 与远月合约 (如 `CLM24.NYM`) 的价格。若 近月价格 > 远月价格，赋值为 `'Backwardation'`；若 近月价格 < 远月价格，赋值为 `'Contango'`；若价差在 ±0.5 美元内，赋值为 `'Flat'`。*(注：若暂未接入远月合约接口，可暂时使用 Brent-WTI 价差作为降级替代方案，但需在代码中加注释标记)* |
| `risk_premium_est` | 暂时留空或默认填 `NULL` |

---

### 3.3 持久化模块 (Storage)

#### 3.3.1 幂等性设计 (防重复入库)

由于该任务为手动补数或重跑历史任务，同一个日期可能会被执行多次。为了防止由于 `UNIQUE KEY (trade_date)` 冲突导致任务报错中断，必须采用 **Upsert (更新或插入)** 机制。

- **SQL 语法规范**：程序员在执行 SQL 写入时，必须使用 `INSERT INTO daily_oil_metrics (...) VALUES (...) ON DUPLICATE KEY UPDATE` 结构。
- **更新行为**：当遇到表中已存在相同 `trade_date` 的记录时，系统自动用新抓取并计算出的衍生字段覆盖更新旧数据，同时触发 MySQL 自动更新 `updated_at` 字段。

#### 3.3.2 批量写入优化

**分块提交（Chunksize）**：当补数时间跨度较长（如 1 年以上）时，单次生成的 SQL 语句可能会超过 MySQL 的 `max_allowed_packet` 限制。必须将 DataFrame 按 **500~1000 条** 进行分块（`np.array_split`），循环执行 Upsert 操作。

---

## 4. 批处理任务接口设计 (API)

### 4.1 接口定义

```python
def run_oil_pipeline_job(start_date: str, end_date: str) -> dict:
    """
    原油日度数据批处理主入口
    
    Args:
        start_date: 补数/抓取的起始日期 (格式 "YYYY-MM-DD")
        end_date: 补数/抓取的结束日期 (格式 "YYYY-MM-DD")
    
    Returns:
        dict: 包含 success, records_count, elapsed_time, message 等信息
    """
    pass
```

### 4.2 内部执行时序

1. 校验输入日期格式是否合法，且 `start_date <= end_date`。
2. 执行 Fetcher 抓取该时间段内的原始数据。
3. 将数据送入 Pipeline 依次执行：异常清洗 → 线性插值 → 特征计算。
4. 将最终生成的结构化多列矩阵，分块批量提交给数据库。
5. 打印/记录执行日志（成功条数、耗时、若失败则抛出 Exception 并回滚事务）。

---

## 5. 开发建议与避坑指南

### 5.1 yfinance 特有注意事项

1. **字段名差异**：
   - yahoo-fin 使用 `adjclose`
   - yfinance 使用 `Adj Close`（空格，且首字母大写）
   - 使用 `auto_adjust=True` 时，`Close` 列即为复权价

2. **时区处理**：
   - yfinance 返回的 DatetimeIndex 带有时区信息
   - 建议使用 `df.index = df.index.tz_localize(None)` 移除时区
   - 或者直接转换为字符串格式

3. **批量下载优化**：
   ```python
   # 推荐：一次性下载多个 ticker
   tickers = ['CL=F', 'BZ=F', 'DX-Y.NYB', '^VIX', '^TNX', 'CRAK']
   df = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
   ```

### 5.2 通用注意事项

1. **缺失值断言检查**：在特征转化结束后、写入数据库前，务必加一行断言或日志检查，确保 `wti_close` 和 `brent_close` 没有包含 `NaN`。如果仍有 `NaN`，说明起始日期选得太短，导致前面的滑动窗口无法初始化，建议提示用户前推历史日期（如多传 60 天）。

2. **时序顺序强校验**：yfinance 返回的数据默认按时间正序排列，但代码逻辑的第一步仍建议显式执行一次按日期升序排列（`df.sort_values('trade_date', inplace=True)`），确保均线（MA）和 RSI 计算正确。

3. **API 限流保护**：虽然 yfinance 比 yahoo-fin 更稳定，但批量拉取历史数据时仍建议在循环中加入 `time.sleep(0.5~1)` 的随机延迟，避免触发限流。使用 `yf.download()` 批量下载时可减少延迟需求。

---

## 6. 依赖安装

```bash
# 安装 yfinance（替代 yahoo-fin）
pip install yfinance

# 其他依赖
pip install pandas numpy pymysql
```

---

## 7. 版本变更记录

| 版本 | 日期 | 变更内容 |
|------|------|---------|
| V1.0 | - | 初始版本 |
| V2.0 | - | 完善清洗与特征工程逻辑 |
| V3.0 | 2026-06-09 | **数据源从 yahoo-fin 更换为 yfinance**；新增批量下载支持；优化字段映射说明 |
