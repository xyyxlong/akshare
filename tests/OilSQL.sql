-- ============================================================
-- 原油市场分析数据库建表脚本
-- 数据层级: 日度(Daily) / 周度(Weekly) / 月度宏观(Monthly/Macro)
-- ============================================================


-- ------------------------------------------------------------
-- 1. 日度盘后检查清单表 (Daily Oil Metrics)
--    用途: 捕捉短期情绪、宏观金融变量及盘面技术形态
-- ------------------------------------------------------------
CREATE TABLE `daily_oil_metrics` (
    `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT '自增主键',

    -- 基础信息
    `trade_date` DATE NOT NULL COMMENT '交易日期',

    -- 价格与技术面 (WTI & Brent)
    `wti_close` DECIMAL(10,4) DEFAULT NULL COMMENT 'WTI收盘价(美元/桶)',
    `brent_close` DECIMAL(10,4) DEFAULT NULL COMMENT 'Brent收盘价(美元/桶)',
    `wti_60dma` DECIMAL(10,4) DEFAULT NULL COMMENT 'WTI 60日均线',
    `wti_rsi` DECIMAL(8,4) DEFAULT NULL COMMENT 'WTI RSI指标',

    -- 宏观与金融变量
    `usd_index` DECIMAL(10,4) DEFAULT NULL COMMENT '美元指数(DXY)',
    `us_10y_yield` DECIMAL(8,4) DEFAULT NULL COMMENT '美国10年期国债收益率(%)',
    `vix_index` DECIMAL(10,4) DEFAULT NULL COMMENT 'VIX恐慌指数',

    -- 价差与期限结构
    `brent_wti_spread` DECIMAL(10,4) DEFAULT NULL COMMENT 'Brent-WTI跨区价差(美元/桶)',
    `term_structure` VARCHAR(20) DEFAULT NULL COMMENT '期货曲线结构(Backwardation/Contango/Flat)',
    `crack_spread` DECIMAL(10,4) DEFAULT NULL COMMENT '裂解价差-成品油减原油(美元/桶)',

    -- 地缘政治风险溢价
    `gpr_index` DECIMAL(10,4) DEFAULT NULL COMMENT '地缘政治风险指数(GPR)',
    `risk_premium_est` DECIMAL(10,4) DEFAULT NULL COMMENT '估算风险溢价(美元/桶)',

    -- 系统字段
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',

    -- 主键与索引
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_trade_date` (`trade_date`) COMMENT '每个交易日唯一',
    KEY `idx_trade_date` (`trade_date`) COMMENT '按日期查询优化'
) ENGINE=INNODB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='原油日度盘后检查清单(价格/技术面/宏观/价差/地缘风险)';


-- 示例插入
INSERT INTO `daily_oil_metrics`
(`trade_date`, `wti_close`, `brent_close`, `wti_60dma`, `wti_rsi`,
 `usd_index`, `us_10y_yield`, `vix_index`,
 `brent_wti_spread`, `term_structure`, `crack_spread`,
 `gpr_index`, `risk_premium_est`)
VALUES
('2026-06-06', 72.3500, 76.1200, 70.8800, 55.3200,
 104.2500, 4.2800, 18.5600,
 3.7700, 'Backwardation', 25.4300,
 112.5000, 2.5000);

-- 示例查询
SELECT * FROM `daily_oil_metrics` ORDER BY `trade_date` DESC;
SELECT * FROM `daily_oil_metrics` WHERE `term_structure` = 'Backwardation' AND `crack_spread` > 20 ORDER BY `trade_date` DESC;
SELECT `trade_date`, `wti_close`, `wti_60dma`, CASE WHEN `wti_close` > `wti_60dma` THEN '多头排列' ELSE '空头排列' END AS `trend_signal` FROM `daily_oil_metrics` ORDER BY `trade_date` DESC LIMIT 30;


-- ------------------------------------------------------------
-- 2. 周度系统性回顾表 (Weekly Oil Metrics)
--    用途: 验证供需基本面逻辑, 数据源 EIA/API/CFTC
-- ------------------------------------------------------------
CREATE TABLE `weekly_oil_metrics` (
    `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT '自增主键',

    -- 基础信息
    `report_date` DATE NOT NULL COMMENT '报告发布日期',
    `ref_week_end` DATE NOT NULL COMMENT '数据所属周的结束日期(周五)',

    -- EIA 库存与供需 (核心)
    `eia_crude_inventory_chg` DECIMAL(12,4) DEFAULT NULL COMMENT 'EIA商业原油库存变化(万桶)',
    `eia_crude_inventory_forecast` DECIMAL(12,4) DEFAULT NULL COMMENT 'EIA原油库存预期值(万桶)-用于计算预期差',
    `eia_gasoline_chg` DECIMAL(12,4) DEFAULT NULL COMMENT 'EIA汽油库存变化(万桶)',
    `eia_distillates_chg` DECIMAL(12,4) DEFAULT NULL COMMENT 'EIA馏分油库存变化(万桶)',
    `eia_cushing_inventory` DECIMAL(12,4) DEFAULT NULL COMMENT '库欣地区原油库存(万桶)',

    -- 产量与开工率
    `us_crude_production` DECIMAL(12,4) DEFAULT NULL COMMENT '美国原油产量(千桶/日)',
    `refinery_utilization` DECIMAL(8,4) DEFAULT NULL COMMENT '炼厂开工率(%)',

    -- 资金情绪与持仓 (CFTC)
    `cftc_net_long` DECIMAL(14,2) DEFAULT NULL COMMENT '非商业净多头持仓(手)',
    `cftc_net_long_chg` DECIMAL(14,2) DEFAULT NULL COMMENT '净多头持仓周环比变化(手)',

    -- 供给端高频指标
    `baker_hughes_rig_count` INT DEFAULT NULL COMMENT '贝克休斯活跃钻井数',

    -- 系统字段
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',

    -- 主键与索引
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_report_date` (`report_date`) COMMENT '每个报告日唯一',
    KEY `idx_report_date` (`report_date`) COMMENT '按报告日期查询优化',
    KEY `idx_ref_week_end` (`ref_week_end`) COMMENT '按数据所属周查询优化'
) ENGINE=INNODB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='原油周度系统性回顾(EIA库存/产量/CFTC持仓/钻井数)';


-- 示例插入
INSERT INTO `weekly_oil_metrics`
(`report_date`, `ref_week_end`,
 `eia_crude_inventory_chg`, `eia_crude_inventory_forecast`, `eia_gasoline_chg`, `eia_distillates_chg`, `eia_cushing_inventory`,
 `us_crude_production`, `refinery_utilization`,
 `cftc_net_long`, `cftc_net_long_chg`,
 `baker_hughes_rig_count`)
VALUES
('2026-06-04', '2026-05-30',
 -320.5000, -150.0000, 110.2000, -85.6000, 3250.0000,
 13200.0000, 92.5000,
 185000.00, 12500.00,
 580);

-- 示例查询
SELECT * FROM `weekly_oil_metrics` ORDER BY `report_date` DESC;
-- 计算EIA原油库存预期差
SELECT `report_date`, `eia_crude_inventory_chg`, `eia_crude_inventory_forecast`,
       (`eia_crude_inventory_chg` - `eia_crude_inventory_forecast`) AS `surprise`
FROM `weekly_oil_metrics` ORDER BY `report_date` DESC LIMIT 12;
-- CFTC净多头持仓趋势
SELECT `report_date`, `cftc_net_long`, `cftc_net_long_chg` FROM `weekly_oil_metrics` ORDER BY `report_date` DESC LIMIT 20;


-- ------------------------------------------------------------
-- 3. 月度与宏观数据表 (Monthly Macro Metrics)
--    用途: 中长期趋势判断, 评估全球宏观需求与OPEC+供给政策
-- ------------------------------------------------------------
CREATE TABLE `monthly_macro_metrics` (
    `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT '自增主键',

    -- 基础信息
    `report_month` VARCHAR(7) NOT NULL COMMENT '月份(YYYY-MM)',

    -- 宏观需求指标
    `global_pmi` DECIMAL(8,4) DEFAULT NULL COMMENT '全球制造业PMI',
    `china_crude_imports` DECIMAL(12,4) DEFAULT NULL COMMENT '中国原油进口量(万吨)',
    `us_gdp_growth` DECIMAL(8,4) DEFAULT NULL COMMENT '美国GDP增速(%)',

    -- 供给端政策
    `opec_production` DECIMAL(12,4) DEFAULT NULL COMMENT 'OPEC原油产量(千桶/日)',
    `opec_compliance_rate` DECIMAL(8,4) DEFAULT NULL COMMENT 'OPEC+减产执行率(%)',

    -- 全球库存
    `oecd_commercial_inventory` DECIMAL(12,4) DEFAULT NULL COMMENT 'OECD商业原油库存(百万桶)',
    `oecd_vs_5y_avg` DECIMAL(12,4) DEFAULT NULL COMMENT 'OECD库存较5年均值偏离量(百万桶)',

    -- 系统字段
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',

    -- 主键与索引
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_report_month` (`report_month`) COMMENT '每月唯一',
    KEY `idx_report_month` (`report_month`) COMMENT '按月份查询优化'
) ENGINE=INNODB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='原油月度宏观数据(全球PMI/中国进口/OPEC产量/OECD库存)';


-- 示例插入
INSERT INTO `monthly_macro_metrics`
(`report_month`,
 `global_pmi`, `china_crude_imports`, `us_gdp_growth`,
 `opec_production`, `opec_compliance_rate`,
 `oecd_commercial_inventory`, `oecd_vs_5y_avg`)
VALUES
('2026-05',
 50.8000, 4520.0000, 2.3000,
 27500.0000, 96.5000,
 2850.0000, -35.2000);

-- 示例查询
SELECT * FROM `monthly_macro_metrics` ORDER BY `report_month` DESC;
-- OPEC减产执行率趋势
SELECT `report_month`, `opec_production`, `opec_compliance_rate` FROM `monthly_macro_metrics` ORDER BY `report_month` DESC LIMIT 12;
-- OECD库存与5年均值偏离
SELECT `report_month`, `oecd_commercial_inventory`, `oecd_vs_5y_avg`,
       CASE WHEN `oecd_vs_5y_avg` < 0 THEN '低于5年均值' ELSE '高于5年均值' END AS `inventory_status`
FROM `monthly_macro_metrics` ORDER BY `report_month` DESC;


-- ============================================================
-- 跨表关联查询示例
-- ============================================================

-- 日度价格 + 周度库存联合分析: 查看EIA库存变化对WTI价格的影响
-- SELECT d.trade_date, d.wti_close, d.term_structure,
--        w.eia_crude_inventory_chg, w.eia_crude_inventory_forecast,
--        (w.eia_crude_inventory_chg - w.eia_crude_inventory_forecast) AS surprise
-- FROM daily_oil_metrics d
-- LEFT JOIN weekly_oil_metrics w ON d.trade_date = w.report_date
-- WHERE w.report_date IS NOT NULL
-- ORDER BY d.trade_date DESC;

-- 月度宏观 + 周度供需综合视图
-- SELECT m.report_month, m.global_pmi, m.opec_production, m.oecd_vs_5y_avg,
--        AVG(w.us_crude_production) AS avg_us_production,
--        AVG(w.refinery_utilization) AS avg_refinery_util
-- FROM monthly_macro_metrics m
-- LEFT JOIN weekly_oil_metrics w ON DATE_FORMAT(w.ref_week_end, '%Y-%m') = m.report_month
-- GROUP BY m.report_month, m.global_pmi, m.opec_production, m.oecd_vs_5y_avg
-- ORDER BY m.report_month DESC;

建表概览
1. daily_oil_metrics - 日度盘后检查清单
字段分组	关键字段
价格与技术面	wti_close, brent_close, wti_60dma, wti_rsi
宏观金融变量	usd_index, us_10y_yield, vix_index
价差与期限结构	brent_wti_spread, term_structure, crack_spread
地缘政治风险	gpr_index, risk_premium_est
- 主键: trade_date 唯一约束，每个交易日一条记录
2. weekly_oil_metrics - 周度系统性回顾
字段分组	关键字段
EIA 库存	eia_crude_inventory_chg, eia_crude_inventory_forecast(预期差字段), eia_gasoline_chg, eia_distillates_chg, eia_cushing_inventory
产量与开工	us_crude_production, refinery_utilization
CFTC 持仓	cftc_net_long, cftc_net_long_chg
高频供给	baker_hughes_rig_count
- 包含 eia_crude_inventory_forecast 预期值字段，可直接计算 Actual - Forecast 预期差
3. monthly_macro_metrics - 月度宏观数据
字段分组	关键字段
宏观需求	global_pmi, china_crude_imports, us_gdp_growth
OPEC+ 供给	opec_production, opec_compliance_rate
全球库存	oecd_commercial_inventory, oecd_vs_5y_avg
与你现有项目风格的对齐
- 统一使用 ENGINE=INNODB, CHARSET=utf8mb4, COLLATE=utf8mb4_unicode_ci
- 所有字段均带 COMMENT 注释，含单位说明
- 数值字段使用 DECIMAL 而非 FLOAT，保证精度
- 包含 created_at / updated_at 系统时间戳字段
- 提供了示例 INSERT 和常用 SELECT 查询（含跨表关联查询模板）




