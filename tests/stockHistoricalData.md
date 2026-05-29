### 📂 模块概览：历史行情数据获取与存储模块

该模块是整个估值系统的基础数据供给层，主要负责从 **Baostock** 数据源获取原始行情数据，经过清洗和转换后，存储到 **MySQL** 数据库中，供后续的估值计算引擎使用。

| 属性 | 详情 |
| :--- | :--- |
| **模块名称** | `stockHistoricalData.py` |
| **核心职责** | 获取、清洗、存储 A股历史K线数据（日/周/月）及复权数据 |
| **外部依赖** | Baostock API, MySQL (通过 `insert2Mysql`), Pandas |
| **关键功能** | 断点续传/重试机制、自动复权处理、防封IP的随机休眠、批量入库 |

---

### 1. 架构设计与核心类

该模块采用了**分层封装架构**，将网络请求与业务逻辑分离，提高了代码的可维护性。

#### 1.1 核心类：`BaostockClient`
*   **职责**：封装 Baostock 的底层 SDK 调用，管理登录状态和连接生命周期。
*   **关键方法**：
    *   `login()` / `logout()`: 确保单例模式下的登录状态，避免频繁重复登录。
    *   `format_stock_code()`: **代码标准化**。将输入的 `600000` 转换为 Baostock 所需的 `sh.600000` 或 `sz.000001` 格式。
    *   `get_history_data()`: 核心数据拉取方法。使用 `bs.query_history_k_data_plus` 获取数据，并进行基础的字段映射和数值类型转换（如将字符串转为 float）。

#### 1.2 业务逻辑类：`StockHistoricalData`
*   **职责**：处理业务规则，包括重试逻辑、数据保存、批量任务调度。
*   **关键属性**：
    *   `MAX_TRYTIMES`: 接口调用失败上限（3次）。
    *   `AK_TRY_FAILD_SLEEPTIME`: 失败后的休眠时间（60秒），防止被服务器封禁。
*   **关键方法**：
    *   `fetch_stock_data()`: 提供对外统一的获取接口，支持参数映射（如将 `qfq` 映射为 Baostock 的 `2`）。
    *   `batch_process_stocks()`: **批量处理引擎**。结合 `tqdm` 进度条，遍历股票列表，依次拉取并保存数据。
    *   `save_to_mysql()`: 数据持久化。将清洗后的 DataFrame 写入数据库。

---

### 2. 详细设计与算法逻辑

#### 2.1 数据获取流程 (Sequence)

```plaintext
1.  调用 batch_process_stocks()
    -> 2. 遍历股票代码列表 (stock_codes)
        -> 3. 调用 fetch_stock_data()
            -> 4. 参数校验与格式化 (日期、复权类型)
                -> 5. BaostockClient.get_history_data()
                    -> 6. [网络请求] Baostock API
                    <- 7. 返回原始 List[String]
                <- 8. 转换为 DataFrame (列名映射、类型转换)
            <- 9. 返回清洗后的行情 DF
        -> 10. 调用 save_to_mysql()
            -> 11. 列名转换 (中文 -> 数据库字段名)
            -> 12. 处理缺失值 (NaN -> None)
            -> 13. 执行 SQL 批量插入
        -> 14. 随机休眠 (防封IP)
    -> 15. 下一只股票...
```

#### 2.2 关键逻辑细节

*   **复权处理**：
    *   代码明确区分了三种状态：`""`(不复权, 3), `"qfq"`(前复权, 2), `"hfq"`(后复权, 1)。
    *   **注意**：根据前序对话，估值计算通常需要**前复权**（`qfq`）数据以保证历史价格的连续性。该脚本会在主程序部分分别调用两次 `batch_process_stocks`，一次无复权（用于计算分红），一次前复权（用于计算PE/PB）。

*   **错误重试机制**：
    *   采用了 **指数退避** 的变体。如果接口调用失败，会随机休眠 `1` 到 `60` 秒之间的时间，增加请求头的随机性，模拟人类操作，避免因高频请求被服务器拒绝。

*   **数据清洗**：
    *   **振幅计算**：如果数据库返回的原始数据缺少振幅，代码会基于公式 `(最高-最低)/最低` 进行重新计算。
    *   **涨跌额**：基于 `收盘价 - 昨收盘` 计算得出。

---

### 3. 数据库设计 (Table Schema)

该模块直接对应数据库中的两张核心表（根据 `save_to_mysql` 中的逻辑判断）：

#### 3.1 表名：`stock_historical_data` (不复权)
#### 3.2 表名：`stock_historical_data_qfq` (前复权)

| 字段名 (英文) | 字段名 (中文) | 数据类型 | 描述 | 来源 |
| :--- | :--- | :--- | :--- | :--- |
| **date** | 日期 | DATE | YYYY-MM-DD | Baostock |
| **stock_code** | 股票代码 | VARCHAR | 6位代码 (000001) | 输入参数 |
| **open** | 开盘 | DECIMAL | 开盘价 | Baostock |
| **close** | 收盘 | DECIMAL | 收盘价 | Baostock |
| **high** | 最高 | DECIMAL | 最高价 | Baostock |
| **low** | 最低 | DECIMAL | 最低价 | Baostock |
| **volume** | 成交量 | BIGINT | 手 | Baostock |
| **amount** | 成交额 | DECIMAL | 元 | Baostock |
| **amplitude** | 振幅 | DECIMAL | (最高-最低)/最低 | 计算/原生 |
| **change_percent** | 涨跌幅 | DECIMAL | % | Baostock |
| **change_amount** | 涨跌额 | DECIMAL | 元 | 计算 |
| **turnover_rate** | 换手率 | DECIMAL | % | Baostock |

---

### 4. 接口与调用规范

该模块提供了灵活的调用方式，既可以作为脚本独立运行，也可以作为库导入使用。

#### 4.1 函数调用接口

*   **获取数据并保存 (批量)**
    ```python
    processor = StockHistoricalData()
    processor.batch_process_stocks(
        stock_codes=None,  # None表示获取所有自选股
        period="daily",   # daily/weekly/monthly
        adjust="qfq",     # /qfq/hfq
        start_date="20260501", # 格式 YYYYMMDD
        end_date=None       # None表示当前日期
    )
    ```

*   **仅查询数据库 (不拉取API)**
    ```python
    df = get_stock_data_from_mysql(
        stock_code="000001", 
        adjust="qfq", 
        start_date="2024-01-01", 
        end_date="2024-12-31"
    )
    ```
    *注：该函数直接返回中文列名的 DataFrame，便于 Pandas 直接分析。*

#### 4.2 命令行运行
直接运行脚本将执行默认的“全量自选股”更新任务：
```bash
python stockHistoricalData.py
```
*默认行为*：先更新不复权数据，再更新前复权数据，时间范围为 2026-05-01 至今日。

---

### 5. 部署与运维建议

1.  **运行频率**：
    *   建议配置为 **每日收盘后 (16:00)** 执行一次。
    *   原因：A股交易时间为 9:30-15:00，收盘后数据才会结算完成。当前代码中 `end_date` 默认为 `datetime.now()`，在周五（今天是周四）运行会获取到最新的交易数据。

2.  **依赖管理**：
    *   必须安装 `baostock` 库 (`pip install baostock`)。
    *   需要配置好 `insert2Mysql.py` 中的数据库连接信息。

3.  **潜在风险**：
    *   **网络波动**：Baostock 是免费接口，稳定性一般。代码中的 `MAX_TRYTIMES=3` 和随机休眠是必要的防护。
    *   **字段缺失**：代码中使用了 `get_select_stocks()` 获取股票列表，需确保 `getAllStock.py` 模块正常工作。

4.  **与估值系统的集成**：
    *   该模块是 **估值计算系统** 的前置依赖。
    *   **执行顺序**：`stockHistoricalData.py` (获取价格) -> `insertStockPEwithDBprice.py` (计算PE/PB)。
    *   只有当行情数据入库后，估值计算脚本才能基于最新的 `close` 价格和 `volume` 进行运算。