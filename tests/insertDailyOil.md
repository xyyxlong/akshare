石油日度盘后数据抓取与清洗系统——最终详细设计文档 (V2.0)

业务背景与系统目标
本系统是一个标准的数据批处理（Batch）管道。核心任务是根据指定的日期范围，从公开金融数据接口抓取原油价格、宏观经济指标等，经过防错清洗与指标衍生后，以覆盖写入（Upsert）的方式存入 MySQL 数据库。
该数据主要服务于量化团队的策略回测，对数据的连续性、准确性（无异常尖峰）和幂等性（重复运行不污染数据）有极高要求。

整体架构与流程设计
系统采用三阶段流水线设计：数据采集 -> 数据清洗与特征转化 -> 数据持久化。

[ 外部调用: 触发输入 start_date, end_date ]
                 │
                 ▼
     ┌───────────────────────┐
     │  1. 数据采集模块      │ ──► 过滤非交易日 -> 调用 API 分头抓取
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

模块详细设计

3.1 数据采集模块 (Fetcher)

3.1.1 接口映射关系表
开发人员需调用以下开源库及接口，按 trade_date（交易日期）为基准轴进行多表外连接（Outer Join）合并。
数据库目标字段   开源库类型   具体代码接口 / 股票代码 (Ticker)   提取字段及处理逻辑
wti_close   yahoo-fin   stock_info.get_data("CL=F")   提取 adjclose (复权收盘价)，代表美油价格

brent_close   yahoo-fin   stock_info.get_data("BZ=F")   提取 adjclose，代表布油价格

usd_index   yahoo-fin   stock_info.get_data("DX-Y.NYB")   提取 adjclose，代表美元指数

vix_index   yahoo-fin   stock_info.get_data("^VIX")   提取 adjclose，代表恐慌指数

us_10y_yield   yahoo-fin   stock_info.get_data("^TNX")   提取 adjclose。注意：该接口返回的数值可能已为百分比格式(如4.5)，需校验，若大于10则除以10，否则直接使用

crack_spread   yahoo-fin   stock_info.get_data("CRAK")   提取 adjclose。使用美股原油ETF (CRAK) 作为裂解价差的代理指标

gpr_index   akshare   ak.macro_usa_gpr_index()   提取地缘政治风险指数。若 akshare 无此接口，暂时默认填 NULL

3.1.2 采集核心逻辑
节假日过滤：在发起请求前，利用 exchange_calendars 库或硬编码日历，过滤掉周末及美国法定节假日，避免发起无效请求。
时区与时间对齐：统一将各类接口返回的日期转化为字符串格式 YYYY-MM-DD，并作为主键（基准轴）。
多源合并：由于美股、商品期货的交易日不完全一致，必须使用 外连接（Outer Join） 保留所有有数据的交易日，未对齐的日期产生 NaN 留给清洗模块处理。

3.2 数据清洗与特征转化模块 (Pipeline)

⚠️ 核心警告：为了防止“未来函数”（即回测时使用了未发生的数据），数据处理必须严格按照时间正序（Ascending）排列后，才能进行滑动窗口计算。

步骤一：基础过滤（剔除物理错误）
逻辑：遍历所有价格与指数列，如果发现数值 le 0，直接将该数值置为 NaN（视为空值）。

步骤二：IQR (四分位距) 算法剔除闪崩异常值
针对原油价格（wti_close, brent_close），由于接口偶尔会出现输入错误带来的异常大毛刺，使用以下算法清洗：
设定一个长度为 20天 的滚动滑动窗口（Rolling Window）。
在窗口内计算当前序列的第 25 分位数（Q_1）和第 75 分位数（Q_3）。
计算分位距：IQR = Q_3 - Q_1。
设定合规范围边界：下界 Lower = Q_1 - 1.5 times IQR，上界 Upper = Q_3 + 1.5 times IQR。
异常处理：若当天的价格超出了 [Lower, Upper] 范围，判定为异常值，将其强制重置为 NaN。
防错机制：若窗口内有效数据不足（如全是 NaN 导致 IQR=0），则跳过该窗口的异常检测，保留原值或置为 NaN，防止除零报错。

步骤三：缺失值填充（连续性处理）
线性插值（Linear Interpolation）：针对价格和金融指标列（wti_close, brent_close, usd_index, vix_index, us_10y_yield），调用 Pandas 的 interpolate(method='linear')，根据前后有数据的日期进行线性按比例内插填充。
宏观/风险指数填充：针对 gpr_index 或后续扩充的低频宏观数据，采用前向填充（Forward Fill, ffill()），即缺失位置直接沿用最近一个历史有效交易日的值。

步骤四：衍生指标计算（特征工程）
利用清洗完毕后的干净数据，计算以下数据库所需字段：
WTI 60日均线 (wti_60dma)：计算 wti_close 过去 60 个交易日的算术平均值（滚动窗口不足60天时，设置 min_periods=1，有几天算几天）。
Brent-WTI跨区价差 (brent_wti_spread)：text{brent_wti_spread} = text{brent_close} - text{wti_close}
WTI RSI指标 (wti_rsi)：标准 14 日相对强弱指标（RSI），根据 wti_close 过去 14 天的涨跌幅比例计算。
期货曲线结构形态 (term_structure)：
   修正逻辑：需额外获取 WTI 近月合约 (CL=F) 与远月合约 (如 CLM24.NYM) 的价格。
   若 近月价格 > 远月价格，赋值为 'Backwardation'；
   若 近月价格 < 远月价格，赋值为 'Contango'；
   若价差在 ±0.5 美元内，赋值为 'Flat'。
  (注：若暂未接入远月合约接口，可暂时使用 Brent-WTI 价差作为降级替代方案，但需在代码中加注释标记)
估算风险溢价 (risk_premium_est)：暂时留空或默认填 NULL。

3.3 持久化模块 (Storage)

3.3.1 幂等性设计 (防重复入库)
由于该任务为手动补数或重跑历史任务，同一个日期可能会被执行多次。为了防止由于 UNIQUE KEY (trade_date) 冲突导致任务报错中断，必须采用 Upsert (更新或插入) 机制。

SQL 语法规范：程序员在执行 SQL 写入时，必须使用 INSERT INTO daily_oil_metrics (...) VALUES (...) ON DUPLICATE KEY UPDATE 结构。
更新行为：当遇到表中已存在相同 trade_date 的记录时，系统自动用新抓取并计算出的衍生字段覆盖更新旧数据，同时触发 MySQL 自动更新 updated_at 字段。

3.3.2 批量写入优化
分块提交（Chunksize）：当补数时间跨度较长（如 1 年以上）时，单次生成的 SQL 语句可能会超过 MySQL 的 max_allowed_packet 限制。必须将 DataFrame 按 500~1000 条 进行分块（np.array_split），循环执行 Upsert 操作。

批处理任务接口设计 (API)

4.1 接口定义
方法名：run_oil_pipeline_job(start_date, end_date)
输入参数：
  start_date (String, 格式 "YYYY-MM-DD")：补数/抓取的起始日期。
  end_date (String, 格式 "YYYY-MM-DD")：补数/抓取的结束日期。

4.2 内部执行时序
校验输入日期格式是否合法，且 start_date <= end_date。
执行 Fetcher 抓取该时间段内的原始数据。
将数据送入 Pipeline 依次执行：异常清洗 rightarrow 线性插值 rightarrow 特征计算。
将最终生成的结构化多列矩阵，分块批量提交给数据库。
打印/记录执行日志（成功条数、耗时、若失败则抛出 Exception 并回滚事务）。

开发建议与避坑指南

缺失值断言检查：在特征转化结束后、写入数据库前，务必加一行断言或日志检查，确保 wti_close 和 brent_close 没有包含 NaN。如果仍有 NaN，说明起始日期选得太短，导致前面的滑动窗口无法初始化，建议提示用户前推历史日期（如多传 60 天）。
时序顺序强校验：调用 yahoo-fin 的数据默认可能是按时间正序或反序，代码逻辑的第一步一定要显式执行一次按日期升序排列（df.sort_values('trade_date', inplace=True)），否则均线（MA）和 RSI 会彻底算错。
API 限流保护：yahoo-fin 底层为爬虫，批量拉取历史数据时务必在循环中加入 time.sleep(1~3) 的随机延迟，避免触发反爬机制导致 IP 被封。