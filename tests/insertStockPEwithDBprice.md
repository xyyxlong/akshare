脚本名称：insertStockPEwithDBprice.py
核心目的：在无法使用 akshare 接口（如 ak.stock_a_indicator_lg）的情况下，基于本地数据库的历史行情和财务数据，自主计算股票的估值指标（PE/PB/PS/Dividend）。
主要背景：根据代码注释，从2025年8月起原接口失效，因此开发此脚本作为替代方案。计算结果会存在与旧数据的“断层”，因为财务数据的提取口径可能不同。
执行周期：设计用于计算特定时间段（如 2025-12-31 至 2026-05-20）的历史数据。

系统架构与依赖

该脚本采用单体数据库直连架构，直接读取原始数据并写入结果表。
组件   技术栈   说明
数据库   MySQL (PyMySQL)   读取源表：stock_historical_data, dividend_info, stock_financial_reports写入目标表：stock_pe_history

数据处理   Pandas, NumPy   用于时间序列处理和向量化计算

日志   log4ak   自定义日志管理器，支持 DEBUG/INFO/ERROR 级别

调度   内置 Main 函数   直接运行脚本执行，支持批量处理（Batch Size）

核心估值计算逻辑 (详细设计)

该脚本不仅计算简单的 PE，还实现了复杂的 TTM（滚动）指标计算。以下是其核心算法的详细拆解：

2.1 股息率计算 (calculate_dividend_yields)
脚本实现了两种股息率的计算，逻辑严谨，考虑了财年跨越的特殊情况。

静态股息率 (dv_ratio):
    逻辑：使用“上一财年”的分红总额。
    特殊规则：包含下一年1月发放的分红（会计入上一年度的利润分配）。
    公式: (上一财年每股分红总额 / 当前股价) * 100%
滚动股息率 (dv_ttm):
    逻辑：使用过去12个月（Last 12 Months）的实际分红。
    公式: (最近12个月每股分红总和 / 当前股价) * 100%

2.2 每股收益 (EPS) 与 市盈率 (PE)
这是脚本最复杂的部分，特别是 TTM 的计算逻辑。

静态市盈率 (PE):
    数据源：使用 get_previous_fiscal_year_report 获取上一年度12月31日的年报 EPS。
    公式: 股价 / 上年EPS
滚动市盈率 (PE_TTM):
    核心逻辑：EPS_TTM = 最新季度累计EPS + 上年全年EPS - 上年同期累计EPS
    代码逻辑 (calculate_eps_ttm):
        最新季报：获取截止计算日（如2026-05-20）最新的财报数据。
        上年年报：获取上一年度（如2025-12-31）的年报 EPS。
        上年同期：获取与最新季报相同时间段的去年数据（如最新是Q1，则取去年Q1）。
        计算：通过加减法消除重复计算部分，得出最近四个季度的真实盈利。
    容错机制：如果在1-3月且年报未出，会尝试使用“上上年”的年报进行估算。

2.3 市净率 (PB)
逻辑：使用上一年度财报中的每股净资产 (navps)。
公式: 股价 / 每股净资产。

2.4 市销率 (PS)
逻辑：基于“主营业务利润”（在代码中被用作营收的代理指标）。
TTM计算：与 PE_TTM 逻辑一致，使用 主营业务利润_TTM = 最新累计 + 上年全年 - 上年同期累计。

数据库交互设计

3.1 数据源表 (Read)
stock_historical_data: 提供 date, close (收盘价)。
dividend_info: 提供分红记录 (cash_dividend, equity_reg_date)，用于股息率计算。
stock_financial_reports: 提供核心财务指标。
   关键字段：weighted_eps (加权EPS), diluted_eps (稀释EPS), adjusted_net_asset (调整后净资产), main_profit (主营业务利润), report_date。

3.2 目标结果表 (Write)
表名：stock_pe_history
写入策略：INSERT ... ON DUPLICATE KEY UPDATE。
    如果主键（日期+代码）已存在，则更新估值字段，避免重复数据插入。
字段映射：
    pe, pe_ttm, pb, dv_ratio, dv_ttm, ps, ps_ttm, total_mv (总市值估算)。

关键业务规则与假设

在阅读代码时，需要注意以下几个关键的业务假设，这些假设直接影响计算结果的准确性：

分红单位转换：
    代码假设数据库中的 cash_dividend 是“每10股派息额”，因此在计算时统一除以 10 (df['dividend_per_share'] = df['cash_dividend'] / 10.0)。
财务数据滞后性：
    脚本在计算某一天的估值时，使用的是“截止该日期前”最新的财报数据。这意味着在财报发布空窗期，估值指标会保持不变。
市值估算：
    代码中的 estimate_market_cap 方法目前是占位符（返回 None）。这意味着 total_mv 字段在当前版本中可能无法写入有效数据，需要依赖外部实现。
日期处理：
    代码特别处理了 1月1日 至 3月31日 这个时间段。因为此时上年年报可能未发布，代码会尝试回溯查找，这可能导致年初的数据波动。
