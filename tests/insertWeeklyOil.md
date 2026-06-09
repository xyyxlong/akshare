# 石油周度基本面数据抓取与清洗系统——最终详细设计文档

## 1. 系统架构与依赖设计

本系统是一个面向原油基本面量化策略的**周度（Weekly）数据批处理管道**。系统通过指定的日期范围驱动，不设自动定时器。

### 1.1 核心依赖

* `requests`：用于调用 EIA 官方 RESTful API v2。
* `pandas`：核心数据结构，用于多源数据对齐、切片、特征计算。
* `numpy`：用于数学计算与 `NaN` 空值处理。
* `prophet`：用于对长期缺失的低频资金宏观数据（如 CFTC 持仓）进行时序预测填充。
* `sqlalchemy` & `pymysql`：用于与 MySQL 数据库连接，构建高并发批量 Upsert（插入或更新）语句。

### 1.2 批处理接口定义

系统对外暴露一个标准函数接口。为保证滑动窗口（如 IQR 算法）和时序预测（Prophet）拥有足够的历史基准数据，**接口内部必须包含“历史前推”逻辑**：

```python
def process_weekly_metrics(start_date: str, end_date: str):
    """
    周度盘后基本面数据批处理主入口
    :param start_date: 用户要求的入库开始日期，格式 'YYYY-MM-DD'
    :param end_date: 用户要求的入库结束日期，格式 'YYYY-MM-DD'
    
    【程序员注意 - 核心执行流程】：
    1. 内部自动将 start_date 向历史方向前推 52 周，得到 fetch_start_date。
    2. 调用 Ingestion 模块抓取 [fetch_start_date, end_date] 的完整数据。
    3. 将整个长序列送入 ETL 模块进行清洗、插值与时序填充。
    4. 将清洗后的长序列送入 Feature 模块计算周环比变化。
    5. 裁剪数据，仅保留用户原始要求的 [start_date, end_date] 范围。
    6. 调用 Storage 模块批量 Upsert 入库。
    """
    pass

```

---

## 2. 数据采集模块 (Data Ingestion)

程序员需通过多个公开渠道抓取数据。所有数据抓取后，必须统一以 **`report_date`（报告实际发布日期）** 作为主键基准轴进行 **外连接（Outer Join）** 合并，缺失日期自动留空（`NaN`）。

### 2.1 数据源与接口精细映射表

| 数据库目标字段 | 数据源分类 | 官方推荐接口 / 获取路径 | 字段定义与处理逻辑 |
| --- | --- | --- | --- |
| `report_date` | 主键轴 | 各数据源发布日期 | 统一转换为 `YYYY-MM-DD` 字符串 |
| `ref_week_end` | 时间轴 | 数据对应的截至周五 | 统一转换为 `YYYY-MM-DD` 字符串 |
| `eia_crude_inventory_chg` | EIA 能源局 | `GET /v2/petroleum/stoc/wstk/data/?api_key=X&frequency=weekly&facet[series][]=WCESTUS1` | 商业原油库存**周度变化值**。单位：万桶 |
| `eia_crude_inventory_forecast` | 财经日历 | 爬取金十数据/Investing 等 API 历史接口 | 市场在发布前对原油库存变化的**预测均值**。单位：万桶 |
| `eia_gasoline_chg` | EIA 能源局 | `GET /v2/petroleum/stoc/wstk/data/?api_key=X&frequency=weekly&facet[series][]=WGTSTUS1` | 汽油库存**周度变化值**。单位：万桶 |
| `eia_distillates_chg` | EIA 能源局 | `GET /v2/petroleum/stoc/wstk/data/?api_key=X&frequency=weekly&facet[series][]=WDISTUS1` | 馏分油库存**周度变化值**。单位：万桶 |
| `eia_cushing_inventory` | EIA 能源局 | `GET /v2/petroleum/stoc/wstk/data/?api_key=X&frequency=weekly&facet[series][]=WCSSTUS1` | 库欣地区原油库存**绝对总量**。单位：万桶 |
| `us_crude_production` | EIA 能源局 | `GET /v2/petroleum/sum/sndw/data/?api_key=X&frequency=weekly&facet[series][]=WCRFPUS2` | 美国原油日产量。单位：千桶/日 |
| `refinery_utilization` | EIA 能源局 | `GET /v2/petroleum/pnp/wiup/data/?api_key=X&frequency=weekly&facet[series][]=WPULEUS3` | 炼厂开工率。标准百分比数值（如 `91.5`） |
| `cftc_net_long` | CFTC 持仓 | 爬取 CFTC 官网每周期货持仓历史 CSV | 计算非商业持仓：`Non-Commercial Long` - `Non-Commercial Short` |
| `baker_hughes_rig_count` | 贝克休斯 | 爬取贝克休斯官网每周五公布的 Rig Count Excel | 提取美国活跃原油钻井总数（正整数） |

---

## 3. 数据清洗与预处理模块 (ETL & Cleaning)

**警告**：为杜绝引入“未来函数”，宽表合并后必须**首先按照 `report_date` 进行时间正序（从远到近）排列**，随后方可执行以下清洗滤网。

### 3.1 异常值处理 (Outlier Detection)

1. **绝对物理值过滤**：
遍历 `eia_cushing_inventory`、`us_crude_production`、`refinery_utilization`、`baker_hughes_rig_count` 这四个代表**绝对物理量**的字段。若发现数值 $\le 0$，判定为接口传输或格式错误，直接将该单元格置为 `NaN`。*(注意：库存变化量 `_chg` 字段允许为负数，不执行此过滤)*。
2. **滑动 IQR 算法过滤基本面噪点**：
针对周度变动极大的 `eia_crude_inventory_chg`（原油库存变化），使用 **52周（约1年）** 的滚动滑动窗口进行统计学过滤。
* 计算当前窗口内的第 25 分位数（$Q_1$）和第 75 分位数（$Q_3$）。
* 计算四分位距：$IQR = Q_3 - Q_1$。
* 定义合规上下界：

$$Lower = Q_1 - 2.0 \times IQR, \quad Upper = Q_3 + 2.0 \times IQR$$


* **处理**：若当周数据超出 $[Lower, Upper]$ 范围，判定为录入毛刺（如小数点错位），强制将该值重置为 `NaN`。



### 3.2 缺失值填充 (Missing Value Imputation)

1. **高频基本面数据（线性插值）**：
针对 EIA 发布的量化供需指标（库存变化、产量、开工率），若因节假日推迟报告或接口偶然缺失导致 `NaN`，使用 `df[col].interpolate(method='linear')` 进行双向线性插值，确保时序连续性。
2. **低频/长周期宏观数据（Prophet 预测填充）**：
针对 CFTC 持仓数据（`cftc_net_long`），由于其由政府机构发布且存在滞后性，若出现连续 2 周以上的缺失，必须启动 Prophet 模型。
* **实现逻辑**：将历史未缺失的 `report_date`（重命名为 `ds`）与 `cftc_net_long`（重命名为 `y`）作为训练集送入 Prophet 模型，预测并填充当前的 `NaN` 单元格。



---

## 4. 特征工程模块 (Feature Engineering)

利用清洗完毕后的干净序列，在入库前动态派生出下游最核心的资金面指标：

### 4.1 CFTC 持仓周环比变化 (`cftc_net_long_chg`)

* **算法逻辑**：计算当周净多头持仓相对于上一周的变动净值。
* **计算公式**：

$$\text{cftc\_net\_long\_chg}_t = \text{cftc\_net\_long}_t - \text{cftc\_net\_long}_{t-1}$$


* **程序员注意**：在 Pandas 中直接调用 `df['cftc_net_long_chg'] = df['cftc_net_long'].diff(1)`。由于执行此行前数据已严格按时间正序排列，计算出的差值符号可真实反映资金情绪的边际变化。

---

## 5. 数据库持久化模块 (Storage)

### 5.1 幂等 Upsert 机制设计

原版设计采用 `to_sql(if_exists='append')` 会导致补数或重跑历史区间时引发 `uk_report_date` 唯一键冲突。
**正确实现方案**：程序员必须通过 SQLAlchemy 的 `conn.execute()` 拼接 `ON DUPLICATE KEY UPDATE` 语句，实现**增量覆盖写入**。

### 5.2 核心写入 SQL 模板

```sql
INSERT INTO `weekly_oil_metrics` (
    `report_date`, `ref_week_end`, 
    `eia_crude_inventory_chg`, `eia_crude_inventory_forecast`, 
    `eia_gasoline_chg`, `eia_distillates_chg`, `eia_cushing_inventory`, 
    `us_crude_production`, `refinery_utilization`, 
    `cftc_net_long`, `cftc_net_long_chg`, `baker_hughes_rig_count`
) 
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE 
    `ref_week_end` = VALUES(`ref_week_end`),
    `eia_crude_inventory_chg` = VALUES(`eia_crude_inventory_chg`),
    `eia_crude_inventory_forecast` = VALUES(`eia_crude_inventory_forecast`),
    `eia_gasoline_chg` = VALUES(`eia_gasoline_chg`),
    `eia_distillates_chg` = VALUES(`eia_distillates_chg`),
    `eia_cushing_inventory` = VALUES(`eia_cushing_inventory`),
    `us_crude_production` = VALUES(`us_crude_production`),
    `refinery_utilization` = VALUES(`refinery_utilization`),
    `cftc_net_long` = VALUES(`cftc_net_long`),
    `cftc_net_long_chg` = VALUES(`cftc_net_long_chg`),
    `baker_hughes_rig_count` = VALUES(`baker_hughes_rig_count`);

```

---

## 6. 完整批处理工作流与生产监控建议

### 6.1 批处理 SOP 流程图解

```
[ 输入 start_date, end_date ] 
     │
     ▼ (历史自动前推 52 周)
[ 批量多源抓取宽表 Raw DataFrame ] 
     │
     ▼ (显式执行 Sort_Values 按日期升序)
[ 执行 3.1 物理过滤与局部滑动 IQR 清洗 ] 
     │
     ▼ (执行 3.2 线性插值与 Prophet 时序填补)
[ 执行 4.1 特征工程: 计算 .diff(1) 资金变化 ] 
     │
     ▼ (通过日期切片, 仅截取用户原始要求的 start_date 到 end_date 矩阵)
[ 校验 Schema 字段名及数据类型 ] 
     │
     ▼ (执行 5.2 批量 Upsert SQL 入库)
[ 打印数据溯源日志, 任务正常结束 ]

```

### 6.2 生产化与数据溯源监控建议

1. **反爬虫随机延迟**：虽然周度数据请求频次远低于日度，但在循环请求 EIA 细分序列和解析 CFTC 历史文本时，仍需在代码中嵌入 `time.sleep(random.uniform(1.0, 2.5))`。
2. **详尽的清洗日志（数据溯源）**：在清洗阶段（ETL），如果某行数据触发了 IQR 算法并被强行置为 `NaN`，或者某条记录使用了 Prophet 模型进行填充，**必须使用 Python 的 `logging` 模块将其输出到特定的日志文件中**（例如：`"INFO: 2026-03-15 的 cftc_net_long 字段触发 Prophet 预测填充"`）。这能让量化团队在回测发现异常信号时，第一时间精确追踪到底层数据的清洗逻辑。
