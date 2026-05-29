### 📂 模块概览：智能选股与财务评估引擎

该模块是系统的**业务应用层**，核心功能是遍历股票池，调用财务分析接口，根据预设的“巴菲特式”价值投资指标筛选股票，并将结果存储。
该脚本是整个系统中的**“策略选股与综合评估模块”**，它位于数据采集（`stockHistoricalData.py`）和估值计算（`insertStockPEwithDBprice.py`），财报数据获取（`insertStockReport.py`）之后，主要负责基于已有的财务和行情数据，执行具体的选股逻辑。

| 属性 | 详情 |
| :--- | :--- |
| **模块名称** | `selectStock_new.py` |
| **核心职责** | 执行多维度的财务指标筛选与PE条件判断，输出符合条件的股票列表 |
| **外部依赖** | AkShare (财务数据), `get_stockPE_his.py` (PE数据), `insertStockReport_new.py`（财报数据入库），MySQL |
| **关键特性** | 断线重连与超时保护、多线程并发控制、数据库/网络双数据源、详细的日志追踪 |

---

### 1. 系统架构与配置

该模块采用了**主从架构**，主循环负责调度，子函数负责具体的指标计算。

#### 1.1 核心配置参数

*   **运行模式**：
    *   `IS_MY`: 是否仅筛选自选股（True/False）。
    *   `IS_MYSQL`: PE数据来源优先级（True=优先查库，False=强制网络）。
    *   `CHUNK_NUM`: 分块处理数量，用于处理全市场数据时的分页。
*   **选股策略参数**（硬编码在代码中）：
    *   `ROE`: 净资产收益率 > 15%
    *   `PEMAX`: 市盈率 < 25
    *   `DEBT_RATIOS`: 资产负债率 < 70%
    *   `RECEIVABLE_DAYS`: 应收账款周转天数 < 30天
    *   `CASH2PROFIT`: 经营现金流/净利润 > 1.25

#### 1.2 依赖组件流
```plaintext
Main Loop (selectStock)
    -> get_all_stocks / get_select_stocks (获取股票池)
    -> checkRoeCashEBIT (财务指标计算)
        -> get_stockfin_data_from_mysql（insertStockReport_new.py）/ak.stock_financial_analysis_indicator (AkShare接口)
    -> check_pe_condition (估值指标计算)
        -> gsh.get_stock_pe_his (数据库查询)
        -> ak.stock_a_indicator_lg (网络回源)
    -> insert2Mysql (结果入库/缓存)
```

---

### 2. 详细设计与算法逻辑

#### 2.1 选股核心逻辑 (`checkRoeCashEBIT`)

该函数实现了深度的财务排雷与筛选，逻辑如下：

*   **数据源策略**：
    *   **优先级**：如果 `IS_MYSQL=True`，调用 `iSR.get_stockfin_data_from_mysql` 从本地数据库读取。
    *   **降级策略**：如果数据库无数据，调用 `ak.stock_a_indicator_lg`,`ak.stock_financial_analysis_indicator`  从网络获取，并自动调用 `insert_to_mysql` 将数据回填数据库
    *   **并发控制**：使用 `ThreadPoolExecutor` 配合超时（`OUTTIME`）机制，防止接口卡死。
*   **指标计算**：
    1.  **ROE (净资产收益率)**：
        *   提取过去 `PASTYEAR` (5年) 的年报数据。
        *   计算平均值 `roe_values.mean()`。
    2.  **现金流质量 (`var2`)**：
        *   计算过去5年平均“每股经营性现金流” / 平均“扣非每股收益”。
        *   *目的*：确保利润有真金白银的现金流支撑。
    3.  **净利润增长 (`var3`)**：
        *   计算“最新一年扣非净利润” / “前5年平均扣非净利润”。
        *   *目的*：筛选出业绩正在增长的公司。
    4.  **资产负债率 (`var4`)**：
        *   计算过去5年平均资产负债率，需低于阈值。
    5.  **应收账款周转天数 (`var5`)**：
        *   计算过去5年平均周转天数，需低于阈值，反映公司议价能力强。

#### 2.2 估值条件判断 (`check_pe_condition`)

该函数负责判断股票当前的估值水平是否符合买入标准。

*   **数据源策略**：
    *   **优先级**：如果 `IS_MYSQL=True`，调用 `gsh.get_stock_pe_his` 从本地数据库读取。
    *   **降级策略**：如果数据库无数据，调用 `ak.stock_a_indicator_lg` 从网络获取，并自动调用 `insert_to_mysql` 将数据回填数据库（写回缓存）。
*   **PE_TTM 计算逻辑**：
    *   获取过去 `PASTDAY` (30天) 内的所有 PE_TTM 数据。
    *   计算平均值 `valid_df['pe_ttm'].astype(float).mean()`。
    *   **空值处理**：PE的空值直接忽略（不填充），因为通常意味着亏损或极高估值，不符合“低估值”策略。
*   **股息率 (`dv_ratio`) 计算**：
    *   获取过去 `PASTYEAR` (5年) 的数据。
    *   **空值处理**：将空值填充为 0 后计算平均值。

#### 2.3 主循环控制 (`selectStock`)

*   **分块处理**：利用 `np.array_split` 将股票列表切分为 `CHUNK_NUM` 块，支持处理全市场数据。
*   **错误熔断机制**：
    *   `error_count`：连续错误计数器。
    *   如果连续错误次数超过 `MAX_CONSECUTIVE_ERRORS` (3次)，则触发 `ConsecutiveErrorException` 并终止当前批次的处理，防止因网络波动导致程序崩溃。
*   **防封策略**：每处理一只股票强制休眠 `time.sleep(2)`。

---

### 3. 数据库交互设计

该模块既是**数据消费者**也是**数据生产者**。

*   **读取**：
    *   调用 `gsh.get_stock_pe_his(stock_code)` 读取 `stock_pe_history` 表（或类似表）的历史估值数据。
*   **写入**：
    *   当网络获取 PE 数据成功后，调用 `ins.insert_to_mysql(batch_data, issp.INSERT_SQL)` 将数据持久化。
    *   **目的**：构建本地估值数据库，加速后续的选股运行速度（避免重复爬取）。

---

### 4. 接口与调用规范

#### 4.1 核心函数接口

*   **`selectStock()`**
    *   **功能**：主入口函数，执行全量选股流程。
    *   **输出**：生成 Excel 文件 (`select_result_x.xlsx`) 并返回状态字符串。

*   **`checkRoeCashEBIT(r_code, startyear)`**
    *   **输入**：股票代码，起始年份。
    *   **输出**：Tuple `(ROE_avg, CashFlow_ratio, Profit_growth, Debt_avg, Receivable_avg)`。

*   **`check_pe_condition(stock_code, stock_name)`**
    *   **输入**：股票代码，名称。
    *   **输出**：Tuple `(pe_ttm_avg, dv_ratio_avg)`。

#### 4.2 配置文件修改指南

| 配置项 | 位置 | 建议值 | 说明 |
| :--- | :--- | :--- | :--- |
| `STARTYEAR` | 全局变量 | `"2019"` | 决定了财务分析的时间跨度起点 |
| `ROE` | 全局变量 | `15` | 核心筛选指标，可根据市场热度调整 |
| `PEMAX` | 全局变量 | `25` | 通常对应 4% 的盈利收益率 |
| `IS_MYSQL` | 全局变量 | `True` | **强烈建议保持 True**，否则每次运行都会极慢 |

---

### 5. 运维与优化建议

1.  **运行时机**：
    *   建议在**盘后**运行（如 17:00）。
    *   原因：该脚本依赖当天的收盘价来计算 PE。如果在盘中运行，可能会因为数据未更新或波动导致误判。

2.  **性能瓶颈**：
    *   **慢查询**：如果 `IS_MYSQL=False`，该脚本会直接调用 AkShare 接口，速度极慢且容易被封 IP。
    *   **优化方案**：确保前序的 `stockHistoricalData.py` 和 `insertStockPEwithDBprice.py` 已经成功运行，填充了 MySQL 数据库。

3.  **结果解读**：
    *   脚本生成的 `select_result_x.xlsx` 包含详细的中间指标（ROE、现金流、负债等）。
    *   **筛选逻辑漏洞**：目前的代码逻辑是**“获取所有股票的指标，但没有根据 ROE>15 或 PE<25 进行过滤”**。
    *   **严重警告**：代码中虽然定义了 `ROE = 15` 和 `PEMAX = 26`，但在 `selectStock` 的循环中，**并没有 `if` 判断来过滤不符合条件的股票**。它只是把所有股票都计算了一遍并导出了 Excel。
    *   *修复建议*：在 `selectStock` 循环中，计算完指标后应增加逻辑：
        ```python
        if float(var1) > ROE and float(pe_ttm) < PEMAX and ...:
            df_result.loc[row_index] = {...}
        ```

4.  **错误处理**：
    *   日志中如果出现 `ConsecutiveErrorException`，通常意味着网络连接中断或 AkShare 接口暂时不可用。脚本会自动休眠 30 秒 (`RECONNECT_TIME`) 后尝试重连。