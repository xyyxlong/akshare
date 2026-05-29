### 📂 模块概览：财务指标采集与入库引擎

该模块是系统的**数据基石**，核心功能是从 AkShare 接口获取多维度的财务指标，并将其标准化后存入 MySQL 数据库。
该脚本是整个系统中的**“财务数据采集与清洗模块”**，它负责构建系统的底层宽表数据（Wide Table），为后续的估值计算（PE/PB）和选股策略提供基础财务指标。

| 属性 | 详情 |
| :--- | :--- |
| **模块名称** | `insertStockReport_new.py` |
| **核心职责** | 获取、清洗、存储 A股历史财务指标（盈利能力、成长性、偿债能力等） |
| **外部依赖** | AkShare (数据源), MySQL (存储), Pandas, ThreadPoolExecutor |
| **关键特性** | **超时熔断机制**、多线程保护、全量/自选双模式、复杂的列名映射、NaN空值处理 |

---

### 1. 系统架构与配置

该模块采用了**“生产者-消费者”**的变体架构，主循环负责调度股票列表（生产任务），`get_financial_report` 负责获取数据（消费/生产数据），`process_fin_data_batch` 负责数据转换。

#### 1.1 核心配置参数

*   **采集范围**：
    *   `STARTYEAR = "2019"`：决定了财务数据的采集起点，即只采集 2019年至今 的财报。
    *   `CHUNK_NUM = 20`：全量数据分块处理数量，用于内存控制。
*   **熔断与重试**：
    *   `MAX_CONSECUTIVE_ERRORS = 25`：最大连续错误次数，超过则终止当前批次。
    *   `OUTTIME = 5`：单次接口调用超时时间（秒），防止线程卡死。
    *   `RECONNECT_TIME = 60`：错误发生后的休眠时间（秒），用于“冷却”网络请求。
*   **数据源**：
    *   `ak.stock_financial_analysis_indicator`：AkShare 的财务指标综合接口。

#### 1.2 依赖组件流
```plaintext
Main Entry (insertStockReport)
    -> get_all_stocks / get_select_stocks (获取股票池)
    -> np.array_split (分块处理)
    -> Loop: get_financial_report (获取单只股票财报)
        -> ThreadPoolExecutor (提交AkShare任务)
        -> TimeoutError Handler (超时控制)
    -> process_fin_data_batch (清洗与映射)
        -> Column Rename (中文->数据库字段)
        -> NaN -> None (空值处理)
    -> insert_to_mysql (批量入库)
```

---

### 2. 详细设计与算法逻辑

#### 2.1 数据获取流程 (`get_financial_report`)

这是该脚本最核心的健壮性设计部分，专门针对网络接口的不稳定性进行了多重防护。

*   **多线程超时控制**：
    *   代码并没有直接调用 `ak.stock_financial_analysis_indicator`，而是将其包装在 `ThreadPoolExecutor` 中。
    *   **逻辑**：利用多线程的 `result(timeout=5)` 特性，如果接口在 5秒 内没有返回数据，主线程将不再等待，而是强制抛出 `TimeoutError`。
    *   **目的**：AkShare 的部分接口（尤其是财务数据）在数据源服务器响应慢时，会卡住进程长达几分钟，此设计防止了程序假死。

*   **重试机制**：
    *   发生超时或异常后，代码会休眠 60秒 (`RECONNECT_TIME`) 并重试，最多重试 25次 (`MAX_CONSECUTIVE_ERRORS`)。

#### 2.2 数据清洗与映射 (`process_fin_data_batch`)

该函数负责将 AkShare 的“脏数据”转换为数据库的“标准数据”。

*   **列名映射 (Column Mapping)**：
    *   **规模**：代码定义了 **90+** 个字段的映射关系。
    *   **分类**：
        *   `COLUMN_MAP`: 中文列名 -> 数据库字段名（如 `'摊薄每股收益(元)'` -> `'diluted_eps'`）。
        *   `REVERSE_COLUMN_MAP`: 数据库字段名 -> 中文列名（用于从数据库读取时还原）。
    *   **覆盖范围**：涵盖了每股指标、盈利能力(ROE/ROA)、成长能力、营运能力(周转率)、偿债能力、现金流、资产结构等六大维度。

*   **空值处理 (NaN -> None)**：
    *   **问题**：Pandas 中的 `NaN` 直接插入 MySQL 会报错或存储为字符串。
    *   **解决方案**：代码显式地将所有 `np.nan` 替换为 `None`，确保数据库存储为 `NULL`。

*   **日期格式化**：
    *   统一转换为 `YYYY-MM-DD` 格式，确保数据库 `DATE` 类型兼容性。

---

### 3. 数据库设计 (Table Schema)

该模块直接对应数据库中的核心宽表，用于存储所有财务指标。

#### 3.1 表名：`stock_financial_reports`

| 字段类别 | 包含的关键字段 (部分展示) | 描述 |
| :--- | :--- | :--- |
| **基础信息** | `stock_code`, `stock_name`, `report_date` | 股票代码、名称、报告期 |
| **每股指标** | `diluted_eps`, `weighted_eps`, `net_asset_per_share`, `operating_cash_flow_per_share` | EPS、每股净资产、每股现金流 |
| **盈利能力** | `roe`, `weighted_roe`, `roa`, `gross_profit_margin`, `net_profit_margin` | 核心回报率指标 |
| **成长能力** | `revenue_growth`, `net_profit_growth`, `total_asset_growth` | 收入与利润增长率 |
| **营运能力** | `receivables_turnover`, `inventory_turnover`, `total_asset_turnover` | 周转率与周转天数 |
| **偿债能力** | `current_ratio`, `quick_ratio`, `asset_liability_ratio`, `interest_coverage` | 流动比率、速动比率、资产负债率 |
| **现金流** | `cash_flow_to_sales`, `cash_flow_to_net_income` | 现金流质量指标 |

*注：该表结构非常宽，旨在减少后续关联查询的次数，符合数据仓库的宽表设计范式。*

---

### 4. 接口与调用规范

#### 4.1 核心函数接口

*   **`insertStockReport(path: str)`**
    *   **功能**：主入口函数，根据 `path` 参数决定是全量采集还是增量采集。
    *   **参数**：
        *   `path="all"`: 采集全市场 A股 上市公司数据。
        *   `path="select"`: 仅采集自选股列表数据。
    *   **返回值**：状态字符串。

*   **`get_stockfin_data_from_mysql(stock_code: str, start_date: str)`**
    *   **功能**：从数据库读取指定股票的财务数据，并还原为中文列名。
    *   **用途**：供其他模块（如 `selectStock.py`）调用，进行策略计算。

*   **`get_financial_report(r_code: str, start_year: str)`**
    *   **功能**：封装了带超时控制的 AkShare 接口调用。
    *   **返回值**：清洗前的 DataFrame。

#### 4.2 命令行运行
直接运行脚本将执行默认的“全量”更新任务：
```bash
python insertStockReport.py
```
*默认行为*：采集所有 A股 上市公司 2019年至今 的财务指标。

---

### 5. 运维与优化建议

1.  **运行频率**：
    *   建议配置为 **每日凌晨 (02:00)** 执行一次。
    *   原因：财务数据更新频率较低，且 AkShare 接口有频率限制。每日全量更新可以确保数据的完整性，但要注意接口的稳定性。

2.  **性能瓶颈**：
    *   **慢**：由于 `CHUNK_NUM=20` 且使用了 `time.sleep(2)`，该脚本处理全量数据（5000+只股票）将耗时极长（可能超过数小时）。
    *   **优化方案**：
        *   **增量更新**：目前的代码逻辑是全量处理。建议增加日期判断逻辑，只获取“最近一季度”或“有新财报发布”的股票，而不是每次都全量拉取 2019年至今 的数据。
        *   **并发调整**：`ThreadPoolExecutor(max_workers=1)` 限制了并发为1。虽然这是为了防止被封IP，但可以尝试在 `get_financial_report` 外层增加多线程（即同时处理多只股票），而不是在内部处理多线程（处理单只股票的超时）。目前的写法导致了“串行”处理股票，效率极低。

3.  **数据质量**：
    *   该脚本获取的数据是 AkShare 的原始数据。在后续的估值计算中（如 `insertStockPEwithDBprice.py`），通常需要基于这些基础字段（如 `total_assets`）计算衍生指标（如 `pb`）。请确保 AkShare 的字段定义与你的业务逻辑一致（例如：ROE 是加权还是摊薄）。

4.  **错误处理**：
    *   日志中如果出现 `ConsecutiveErrorException`，通常意味着 AkShare 服务端暂时不可用或网络中断。脚本会自动休眠 60秒 后尝试重连。