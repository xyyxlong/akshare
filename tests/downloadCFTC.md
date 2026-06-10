# 附录设计：CFTC 持仓数据自动化下载与解析设计说明（面向程序员）

## 1. 业务目标与接口定位

本模块作为 `weekly_oil_metrics` 批处理任务中 `Fetcher`（数据采集层）的一个子插件。
其任务是：在用户输入指定的日期段（`start_date` 到 `end_date`）后，自动向 **CFTC（美国商品期货交易委员会）** 官网发起请求，下载对应年份的**历史持仓大宽表 CSV 文件**，过滤提取出原油（Crude Oil）的非商业净多头持仓数据（`cftc_net_long`），并结构化返回给下游清洗管线。

---

## 2. CFTC 官方数据源下载机制分析

CFTC 官网提供按自然年（Yearly）打包的历史数据 CSV 压缩包。这意味着，即使用户只需要某一个周五的数据，程序也需要下载该周对应年份的整年压缩包进行内存过滤。

### 2.1 官方静态 URL 命名规则

CFTC 历史持仓数据（Disaggregated Futures Only）的下载链接具有极强的规律性：

* **当前年份及近年历史数据（2010年至今）网址模板**：
`https://www.cftc.gov/files/dea/history/fut_disadvg_xls_[YEAR].zip`
*例如：2026 年的下载全路径为：`https://www.cftc.gov/files/dea/history/fut_disadvg_xls_2026.zip*`

### 2.2 自动化下载流水线（程序员实现步骤）

1. **年份提取**：根据用户输入的 `start_date` 和 `end_date`，解析出涉及哪些自然年（例如输入 `2025-12-01` 至 `2026-02-01`，则涉及 `2025` 和 `2026` 两个年份）。
2. **内存流下载（避免硬盘污染）**：使用 `requests` 请求对应的 ZIP 链接，利用 Python 的 `io.BytesIO` 和 `zipfile` 库直接在内存中解压，读取里面的 `annual.txt`（该文件本质上是以逗号分隔的标准 CSV 宽表）。
3. **设置伪装请求头**：CFTC 服务器对自动化脚本有基础反爬校验，必须在 `requests.get` 中强制带上浏览器标准 `User-Agent`，否则会返回 `403 Forbidden`。

---

## 3. CSV 字段过滤与数据清洗逻辑

解压后的 CSV 包含数百个商品期货的持仓列。为了提取出我们数据库需要的字段，请指示程序员进行以下精确的过滤和数学计算：

### 3.1 核心行过滤条件 (Market Identification)

在一张包含所有商品的大表中，需要通过以下两个字段的联合查询来精准锁死**纽约商业交易所（NYMEX）的 WTI 原油连续合约**：

* 寻找列名为 `Market_and_Market_Type` 的列，过滤出其值等于：`"CRUDE OIL - NEW YORK MERCANTILE EXCHANGE"`
* 或寻找列名为 `CFTC_Market_Code` 的列，过滤出其值等于：`"067651"`

### 3.2 目标字段映射与内存计算 (Column Mapping)

成功过滤出原油所在的行序列后，进行如下字段提取和衍生计算：

| 数据库目标字段 | CFTC 原始 CSV 列名 | 程序员内存处理与计算逻辑 |
| --- | --- | --- |
| `report_date` | `Report_Date_as_YYYY-MM-DD` | 保持不变，直接提取。格式已默认为 `YYYY-MM-DD` |
| `ref_week_end` | - | **时空对齐处理**：CFTC 的报告通常在周二统计、周五发布。由于您的表结构中需要归属周五的日期，请让程序员统一将 `report_date` 顺延 3 天（或调整为当周周五），以此对齐 `ref_week_end` |
| `cftc_net_long` | `Prod_Merc_Positions_Long_All`<br>

<br>`Prod_Merc_Positions_Short_All` | **多空对冲计算**：<br>

<br>在内存中计算：`cftc_net_long` = 原始列 `Long_All` 的数值 减去 原始列 `Short_All` 的数值 |

根据输入start_date，end_date获取数据，并整合到一个CSV文件中。
下载的文件：..\input\CFTC\CFTC_YYYY-MM-DD_YYYY-MM-DD.cvs，YYYY-MM-DD为start_date到end_date的报告时间

---

## 4. 健壮性与异常防范 (Defensive Coding)

1. **多年份合并 (Concatenation)**：若用户输入的日期跨越了元旦（跨年），程序必须循环下载两年的 ZIP 包，并在内存中通过 `pandas.concat()` 将两个年份的 DataFrame 合并后再转入后续的日期切片。
2. **反爬延迟**：如果发生跨多年份的批量历史回溯下载，两次 `requests` 请求之间必须强制加入 `time.sleep(2.0)`，防止被 CFTC 官方防火墙临时封禁 IP。
3. **空值与脏数据防呆**：CFTC 原始文件在某些特殊年份的表头大小写可能不一致（如 `Report_Date_As_YYYY-MM-DD`），代码中读取 CSV 后应第一时间执行 `df.columns = df.columns.str.strip().str.upper()` 统一转换为大写，防止因表头微调导致程序崩溃。