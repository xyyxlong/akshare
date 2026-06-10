# Baker Hughes 钻井数自动化下载与解析设计说明（面向程序员）

## 1. 业务目标与接口定位

本模块作为 `weekly_oil_metrics` 批处理任务中 `Fetcher`（数据采集层）的活跃钻井数子插件。
其任务是：根据输入的日期段（`start_date` 到 `end_date`），自动访问 [Baker Hughes 官方北美钻井数据源页面](https://rigcount.bakerhughes.com/na-rig-count)，下载最新的历史及当下钻井数打包 Excel（`.xlsx`），提取出美国活跃原油钻井数（`baker_hughes_rig_count`），过滤并返回给下游清洗管线。

---

## 2. Baker Hughes 官方数据源下载机制分析

### 2.1 页面静态链接与动态变化规律

Baker Hughes 的周度钻井数据通常在**每周五美东时间下午 1:00** 准时发布。其官网的数据组织机制如下：

1. **主下载区（最新数据）**：页面顶部 `Date` 为最新周五的行中，包含一个名为 `North America Rig Count Report - New Report` 的超链接。由于该链接的内部文件 ID 可能会随每周更新而动态改变，开发人员不能在代码中写死该下载 URL。
2. **归档区（历史数据）**：页面下方的 `North America Rotary Rig Count Archive` 区域，包含历史时间段的打包主表：
* `North America Rig Count New Report (2013-Aug 2025)`
* `North America Rotary Rig Count (Jan 2000 - Mar 2024)`



### 2.2 自动化下载流水线（程序员实现步骤）

由于最新文件的 URL 每周五动态更替，程序员必须采用 **“请求网页 $\rightarrow$ 正则/XPath 提取最新 Excel 链接 $\rightarrow$ 流式下载”** 的三步走策略：

1. **第一步：解析主页，提取动态下载 URL**
* 使用 `requests.get("https://rigcount.bakerhughes.com/na-rig-count", headers=headers)` 抓取主页 HTML。
* 使用 `lxml` 的 XPath 或 `BeautifulSoup` 的 CSS 选择器，定位网页中包含文本 `North America Rig Count Report - New Report` 的 `<a>` 标签，并提取出它的 `href` 属性（即最新的 `.xlsx` 真实下载地址）。


2. **第二步：设置防封伪装请求头**
* 贝克休斯服务器对爬虫检测较为严格，请求时必须显式配置包含 `User-Agent`、`Accept-Language` 并在条件允许时携带 `Referer: https://rigcount.bakerhughes.com/` 的标准 Headers，否则会触发阻断。


3. **第三步：内存加载或暂存**
* 获取到真实下载链接后，使用 `requests.get(excel_url)` 下载该二进制文件。由于 Excel 较大，建议将其作为临时的 `temp_rig_data.xlsx` 落地或通过 `io.BytesIO` 读入 Pandas。



---

## 3. Excel 内部结构识别与字段清洗映射

下载下来的 Excel 是一个包含多个工作表（Sheets）的大型数据表。程序员需要通过特定的表头定位和数据切片技术提取目标字段。

### 3.1 目标 Sheet 定位

* **工作表名称**：加载 Excel 后，定位名为 **`Master`**（或在较新的 Excel 中名为 **`Data`** / **`Rigs by Country`**）的工作表。该工作表以时间正序（自上而下）记录了自 2013 年或更早至今的所有周度数据。

根据输入start_date，end_date获取数据，并整合到一个CSV文件中。
下载的文件：..\input\BakerHughes\BakerHughes_YYYY-MM-DD_YYYY-MM-DD.cvs，YYYY-MM-DD为start_date到end_date的报告时间

### 3.2 矩阵行过滤与字段精细映射

程序员读取该工作表转换为 DataFrame 后，必须找到以下三个标准列来对齐您的 MySQL 数据库：

| 数据库目标字段 | Excel 对应列名标识 (Header) | 程序员内存处理与转换逻辑 |
| --- | --- | --- |
| `report_date` | `Date` | **主键轴**：通常在 A 列。直接读取并转换为标准的 `YYYY-MM-DD` 字符串格式。 |
| `ref_week_end` | `Date` | 贝克休斯公布的日期本身就是**当周周五**。因此直接复制 `report_date` 的值赋给 `ref_week_end`。 |
| `baker_hughes_rig_count` | `U.S. Oil` 或 `US Oil Rig` | **目标数量值**：该列代表美国正在运转的**原油钻井数**。提取当周对应的数值，并强制转换为 `INT`（整型）。 |

> **注意（物理极限拦截）**：在提取 `U.S. Oil` 数量时，如果发现某周数据由于表格断层读到了文本或小于等于 0 的数字，直接在内存中将其置为 `None` (MySQL 中的 `NULL`)，留给上层的 Pipeline 清洗模块（线性插值）去修复。

---

## 4. 历史大跨度回溯补数策略 (Batch Backfill)

当用户输入一个跨度超过 2 年的历史日期区间（例如要求重跑 `2020-01-01` 至 `2024-01-01` 的回测基本面数据）时，单靠主页的“最新一周 Excel”将无法获取，程序需要切入**归档补数分支**：

1. **逻辑判断**：若输入日期段包含历史归档区间，程序需额外定位归档区的链接（如带有 `Archive` 关键字的链接）。
2. **历史主表解析**：归档 Excel 内部结构类似，同样存在一个全局时间序列的 Sheet（通常表头为 `Date`、`US Oil`）。
3. **分段合并 (Union)**：
* 将历史归档 Excel 提取出的 `DataFrame` 与最新周 Excel 提取出的 `DataFrame` 执行 `pd.concat()`。
* 调用 `drop_duplicates(subset=['report_date'], keep='last')` 依据报告日期进行去重，保留最新修正的数据。


4. **日期切片输出**：最后，使用 `df[(df['report_date'] >= start_date) & (df['report_date'] <= end_date)]` 切出用户指定的日期段，安全交付给下游。