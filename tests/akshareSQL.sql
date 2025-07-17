
SHOW STATUS;



-- 行业历史估值表
CREATE TABLE `industry_pe_history` (
    `trade_date` DATE NOT NULL COMMENT '交易日',
    `industry_code` VARCHAR(10) NOT NULL COMMENT '行业编码',
    `pe_weighted` FLOAT(12,4) COMMENT '加权市盈率',
    `pe_median` FLOAT(12,4) COMMENT '中位数市盈率',
    `pe_mean` FLOAT(12,4) COMMENT '算术平均市盈率',
    PRIMARY KEY (`trade_date`, `industry_code`)
) ENGINE=INNODB DEFAULT CHARSET=utf8mb4
COMMENT='行业历史PE估值表';

-- 例子
INSERT INTO `index_valuation_history` 
(`index_code`, `index_name`, `trade_date`, `index_value`, 
 `pe_equal_weight_static`, `pe_static`, `pe_static_median`,
 `pe_equal_weight_ttm`, `pe_ttm`, `pe_ttm_median`)
VALUES 
('000300', '沪深300', '2023-08-15', 3856.0200, 
 15.2300, 12.7800, 14.5600,
 14.8900, 12.3400, 14.1200);
 
SELECT * FROM `industry_pe_history` ORDER BY `trade_date` DESC
SELECT DISTINCT industry_pe_history.`industry_code` FROM industry_pe_history
SELECT COUNT(*) FROM industry_pe_history

               SELECT 
                    trade_date,
                    pe_weighted AS 'PE静-加权',
                    pe_median AS 'PE静-中位',
                    pe_mean AS 'PE静-平均'
                FROM industry_pe_history
                WHERE industry_code = 'R90'
                ORDER BY trade_date ASC


-- 指数历史估值表
CREATE TABLE `index_valuation_history` (
  `id` BIGINT(20) NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `index_code` VARCHAR(10) NOT NULL COMMENT '指数代码，如000300',
  `index_name` VARCHAR(50) NOT NULL COMMENT '指数名称，如沪深300',
  `trade_date` DATE NOT NULL COMMENT '交易日期',
  `index_value` DECIMAL(12,4) DEFAULT NULL COMMENT '指数点位',
  `pe_equal_weight_static` DECIMAL(12,4) DEFAULT NULL COMMENT '等权静态市盈率',
  `pe_static` DECIMAL(12,4) DEFAULT NULL COMMENT '静态市盈率(加权)',
  `pe_static_median` DECIMAL(12,4) DEFAULT NULL COMMENT '静态市盈率中位数',
  `pe_equal_weight_ttm` DECIMAL(12,4) DEFAULT NULL COMMENT '等权滚动市盈率(TTM)',
  `pe_ttm` DECIMAL(12,4) DEFAULT NULL COMMENT '滚动市盈率(TTM)',
  `pe_ttm_median` DECIMAL(12,4) DEFAULT NULL COMMENT '滚动市盈率中位数(TTM)',
  `update_time` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `idx_unique` (`index_code`,`trade_date`) COMMENT '防止重复数据',
  KEY `idx_date` (`trade_date`) COMMENT '按日期查询优化',
  KEY `idx_code_date` (`index_code`,`trade_date`) COMMENT '代码+日期联合查询优化'
) ENGINE=INNODB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci 
COMMENT='指数历史估值表';

-- 例子
    INSERT IGNORE INTO `index_valuation_history` 
    (`index_code`, `index_name`, `trade_date`, `index_value`, 
    `pe_equal_weight_static`, `pe_static`, `pe_static_median`,
     `pe_equal_weight_ttm`, `pe_ttm`, `pe_ttm_median`)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)

SELECT * FROM `index_valuation_history` WHERE `index_name`="沪深300" ORDER BY `trade_date` DESC
SELECT DISTINCT index_name FROM index_valuation_history
SELECT COUNT(*) FROM index_valuation_history
SELECT index_code, index_name, trade_date
FROM (
    SELECT 
        index_code, 
        index_name, 
        trade_date,
        ROW_NUMBER() OVER (PARTITION BY index_code ORDER BY trade_date ASC) AS rn
    FROM index_valuation_history
) AS subquery
WHERE rn = 1;

-- 股票历史估值表
CREATE TABLE `stock_pe_history` (
  `stock_code` VARCHAR(10) NOT NULL COMMENT '股票代码',
  `stock_name` VARCHAR(10) NOT NULL COMMENT '股票名称',
  `trade_date` DATE NOT NULL COMMENT '交易日期',
  `pe` DECIMAL(12,4) DEFAULT NULL COMMENT '静态市盈率',
  `pe_ttm` DECIMAL(12,4) DEFAULT NULL COMMENT '滚动市盈率(TTM)',
  `pb` DECIMAL(12,4) DEFAULT NULL COMMENT '市净率',
  `dv_ratio` DECIMAL(12,4) DEFAULT NULL COMMENT '股息率',
  `dv_ttm` DECIMAL(12,4) DEFAULT NULL COMMENT '滚动股息率(TTM)',
  `ps` DECIMAL(12,4) DEFAULT NULL COMMENT '市销率',
  `ps_ttm` DECIMAL(12,4) DEFAULT NULL COMMENT '滚动市销率(TTM)',
  `total_mv` DECIMAL(15,2) DEFAULT NULL COMMENT '总市值（单位：万元）',
  PRIMARY KEY (`stock_code`, `trade_date`),
  KEY `idx_total_mv` (`total_mv`) COMMENT '市值查询优化[8]'
) ENGINE=INNODB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci 
COMMENT='股票历史估值指标表';

-- 数据库预计算
CREATE MATERIALIZED VIEW pe_stats AS
SELECT stock_code, 
    MAX(pe_ttm) AS max_pe,
    MIN(pe_ttm) AS min_pe
FROM stock_pe_history
GROUP BY stock_code;


-- 检查异常值
ALTER TABLE `stock_pe_history`
ADD CONSTRAINT `chk_pe` CHECK (`pe` > 0),
ADD CONSTRAINT `chk_mv` CHECK (`total_mv` BETWEEN 0 AND 9999999999999.99);
-- ​分区建议​（年度数据量超500万时）
PARTITION BY RANGE (YEAR(trade_date)) (
    PARTITION p2010 VALUES LESS THAN (2011),
    PARTITION p2015 VALUES LESS THAN (2016),
    PARTITION p2020 VALUES LESS THAN (2021),
    PARTITION p2025 VALUES LESS THAN (2026)
);
-- 例子
INSERT INTO `stock_pe_history` 
(`stock_code`,`stock_name`,`trade_date`, `pe`, `pe_ttm`, `pb`, `dv_ratio`, `dv_ttm`, `ps`, `ps_ttm`, `total_mv`)
VALUES
("000333","美的集团",'2015-01-05',23.6896,12.2867,3.3073,2.6774,2.6774,1.0413,0.9241,12596835.71),
("000333","美的集团",'2015-01-06',25.1404,13.0392,3.5099,2.5229,2.5229,1.105,0.9807,13368328.66);

SELECT COUNT(DISTINCT stock_pe_history.`stock_code`) AS Num_pe FROM stock_pe_history
SELECT pe FROM `stock_pe_history`  ORDER BY `trade_date` 
SELECT * FROM `stock_pe_history`  WHERE stock_code="600750" ORDER BY `trade_date` DESC
SELECT * FROM `stock_pe_history`  WHERE stock_code="603198" ORDER BY `pe_ttm`
SELECT COUNT(*) FROM stock_pe_history WHERE stock_code="301090"  AND  trade_date = "20240620" 
SELECT DISTINCT stock_pe_history.`stock_code` FROM stock_pe_history WHERE stock_code IN 
("600036","000858","601919","000333","002555","002602","002460","002738")

	SELECT trade_date AS `日期`, pe, pe_ttm 
        FROM stock_pe_history 
        WHERE stock_code = "002466"
        ORDER BY trade_date DESC
`dv_ratio`
 
-- 分红信息表
CREATE TABLE dividend_info (
    id INT AUTO_INCREMENT PRIMARY KEY,
    stock_code VARCHAR(10) NOT NULL COMMENT '股票代码',
    announcement_date DATE COMMENT '公告日期',
    bonus_share FLOAT COMMENT '送股(每10股)',
    additional_shares FLOAT COMMENT '转增(每10股)',
    cash_dividend FLOAT COMMENT '派息(每10股)',
    progress VARCHAR(20) COMMENT '进度',
    ex_dividend_date DATE COMMENT '除权除息日',
    equity_reg_date DATE COMMENT '股权登记日',
    bonus_listing_date DATE COMMENT '红股上市日',
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY (stock_code, announcement_date)
) COMMENT '股票分红信息表';

SELECT * FROM dividend_info WHERE stock_code = '600348' ORDER BY `ex_dividend_date` DESC
SELECT COUNT(*) FROM dividend_info WHERE stock_code IN ('300146','600183','600596','600598','600618','601336')
SELECT COUNT(DISTINCT stock_code) FROM dividend_info

-- 配股信息表
CREATE TABLE allotment_info (
    id INT AUTO_INCREMENT PRIMARY KEY,
    stock_code VARCHAR(10) NOT NULL COMMENT '股票代码',
    announcement_date DATE COMMENT '公告日期',
    allotment_plan VARCHAR(50) COMMENT '配股方案',
    allotment_price FLOAT COMMENT '配股价格',
    base_equity FLOAT COMMENT '基准股本(万股)',
    ex_rights_date DATE COMMENT '除权日',
    equity_reg_date DATE COMMENT '股权登记日',
    payment_start DATE COMMENT '缴款起始日',
    payment_end DATE COMMENT '缴款终止日',
    allotment_listing DATE COMMENT '配股上市日',
    total_funds FLOAT COMMENT '募集资金合计(万元)',
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY (stock_code, announcement_date)
) COMMENT '股票配股信息表';   

SELECT * FROM allotment_info
SELECT COUNT(*) FROM allotment_info
SELECT COUNT(DISTINCT stock_code) FROM allotment_info
SELECT DISTINCT stock_code FROM allotment_info


-- 股票历史行情数据表
CREATE TABLE stock_historical_data (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
    DATE DATE NOT NULL COMMENT '日期',
    OPEN DECIMAL(10, 4) NOT NULL COMMENT '开盘价',
    CLOSE DECIMAL(10, 4) NOT NULL COMMENT '收盘价',
    high DECIMAL(10, 4) NOT NULL COMMENT '最高价',
    low DECIMAL(10, 4) NOT NULL COMMENT '最低价',
    volume BIGINT NOT NULL COMMENT '成交量(股)',
    amount DECIMAL(20, 4) NOT NULL COMMENT '成交额(元)',
    amplitude DECIMAL(10, 4) NOT NULL COMMENT '振幅(%)',
    change_percent DECIMAL(10, 4) NOT NULL COMMENT '涨跌幅(%)',
    change_amount DECIMAL(10, 4) NOT NULL COMMENT '涨跌额(元)',
    turnover_rate DECIMAL(10, 4) NULL COMMENT '换手率(%)',
    stock_code VARCHAR(10) NOT NULL COMMENT '股票代码',
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    UNIQUE KEY idx_stock_date (stock_code, DATE)  -- 确保同一天同一股票只有一条记录
) COMMENT '股票历史行情数据表';

SELECT * FROM stock_historical_data WHERE stock_code='603713' AND DATE< '2018-07-31' AND DATE> '2018-06-30'  ORDER BY DATE DESC
SELECT * FROM stock_historical_data WHERE stock_code='603506' ORDER BY DATE DESC
SELECT COUNT(*) FROM stock_historical_data
SELECT COUNT(DISTINCT stock_code) FROM stock_historical_data
SELECT DISTINCT stock_code FROM stock_historical_data


-- 股票历史行情数据表(qfq)
CREATE TABLE stock_historical_data_qfq (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
    DATE DATE NOT NULL COMMENT '日期',
    OPEN DECIMAL(10, 4) NOT NULL COMMENT '开盘价',
    CLOSE DECIMAL(10, 4) NOT NULL COMMENT '收盘价',
    high DECIMAL(10, 4) NOT NULL COMMENT '最高价',
    low DECIMAL(10, 4) NOT NULL COMMENT '最低价',
    volume BIGINT NOT NULL COMMENT '成交量(股)',
    amount DECIMAL(20, 4) NOT NULL COMMENT '成交额(元)',
    amplitude DECIMAL(10, 4) NOT NULL COMMENT '振幅(%)',
    change_percent DECIMAL(10, 4) NOT NULL COMMENT '涨跌幅(%)',
    change_amount DECIMAL(10, 4) NOT NULL COMMENT '涨跌额(元)',
    turnover_rate DECIMAL(10, 4) NULL COMMENT '换手率(%)',
    stock_code VARCHAR(10) NOT NULL COMMENT '股票代码',
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    UNIQUE KEY idx_stock_date (stock_code, DATE)  -- 确保同一天同一股票只有一条记录
) COMMENT '股票历史行情数据表';

SELECT * FROM stock_historical_data_qfq WHERE stock_code='603506' AND DATE>'2010-06-01'
SELECT COUNT(*) FROM stock_historical_data_qfq
SELECT COUNT(DISTINCT stock_code) FROM stock_historical_data_qfq
SELECT DISTINCT stock_code FROM stock_historical_data_qfq



DROP TABLE stock_financial_reports
-- 股票财报数据表
CREATE TABLE stock_financial_reports (
   stock_code VARCHAR(10) NOT NULL COMMENT '股票代码',
    stock_name VARCHAR(50) NOT NULL COMMENT '股票名称',
    report_date DATE NOT NULL COMMENT '财报日期',
    diluted_eps DECIMAL(10,4) COMMENT '摊薄每股收益(元)',
    weighted_eps DECIMAL(10,4) COMMENT '加权每股收益(元)',
    adjusted_eps DECIMAL(10,4) COMMENT '每股收益_调整后(元)',
    non_gaap_eps DECIMAL(10,4) COMMENT '扣除非经常性损益后的每股收益(元)',
    net_asset_per_share DECIMAL(20,4) COMMENT '每股净资产_调整前(元)',
    adjusted_net_asset DECIMAL(20,4) COMMENT '每股净资产_调整后(元)',
    operating_cash_flow_per_share DECIMAL(20,4) COMMENT '每股经营性现金流(元)',
    capital_reserve_per_share DECIMAL(20,4) COMMENT '每股资本公积金(元)',
    retained_earnings_per_share DECIMAL(20,4) COMMENT '每股未分配利润(元)',
    adjusted_net_asset_value DECIMAL(20,4) COMMENT '调整后的每股净资产(元)',
    roa DECIMAL(10,4) COMMENT '总资产利润率(%)',
    operating_profit_margin DECIMAL(10,4) COMMENT '主营业务利润率(%)',
    roa_profit_margin DECIMAL(10,4) COMMENT '总资产净利润率(%)',
    net_profit_margin DECIMAL(10,4) COMMENT '销售净利率(%)',
    capital_return_ratio DECIMAL(10,4) COMMENT '股本报酬率(%)',
    roe_return_ratio DECIMAL(10,4) COMMENT '净资产报酬率(%)',
    roe DECIMAL(10,4) COMMENT '净资产收益率(%)',
    asset_return_ratio DECIMAL(10,4) COMMENT '资产报酬率(%)',
    gross_profit_margin DECIMAL(10,4) COMMENT '销售毛利率(%)',
    cost_profit_ratio DECIMAL(10,4) COMMENT '成本费用利润率(%)',
    operating_profit_ratio DECIMAL(10,4) COMMENT '营业利润率(%)',
    main_cost_ratio DECIMAL(10,4) COMMENT '主营业务成本率(%)',
    three_expense_ratio DECIMAL(10,4) COMMENT '三项费用比重',
    non_main_ratio DECIMAL(10,4) COMMENT '非主营比重',
    main_profit_ratio DECIMAL(10,4) COMMENT '主营利润比重',
    dividend_payout_ratio DECIMAL(10,4) COMMENT '股息发放率(%)',
    investment_return_ratio DECIMAL(10,4) COMMENT '投资收益率(%)',
    weighted_roe DECIMAL(10,4) COMMENT '加权净资产收益率(%)',
    revenue_growth DECIMAL(10,4) COMMENT '主营业务收入增长率(%)',
    net_profit_growth DECIMAL(10,4) COMMENT '净利润增长率(%)',
    net_asset_growth DECIMAL(10,4) COMMENT '净资产增长率(%)',
    total_asset_growth DECIMAL(10,4) COMMENT '总资产增长率(%)',
    receivables_turnover DECIMAL(10,2) COMMENT '应收账款周转率(次)',
    receivables_days INT COMMENT '应收账款周转天数(天)',
    inventory_days INT COMMENT '存货周转天数(天)',
    inventory_turnover DECIMAL(10,2) COMMENT '存货周转率(次)',
    fixed_asset_turnover DECIMAL(10,2) COMMENT '固定资产周转率(次)',
    total_asset_turnover DECIMAL(10,2) COMMENT '总资产周转率(次)',
    total_asset_days INT COMMENT '总资产周转天数(天)',
    current_asset_turnover DECIMAL(10,2) COMMENT '流动资产周转率(次)',
    current_asset_days INT COMMENT '流动资产周转天数(天)',
    equity_turnover DECIMAL(10,2) COMMENT '股东权益周转率(次)',
    current_ratio DECIMAL(10,2) COMMENT '流动比率',
    quick_ratio DECIMAL(10,2) COMMENT '速动比率',
    cash_ratio DECIMAL(10,4) COMMENT '现金比率(%)',
    interest_coverage DECIMAL(10,2) COMMENT '利息支付倍数',
    long_term_debt_ratio DECIMAL(10,4) COMMENT '长期债务与营运资金比率(%)',
    equity_ratio DECIMAL(10,4) COMMENT '股东权益比率(%)',
    long_term_liability_ratio DECIMAL(10,4) COMMENT '长期负债比率(%)',
    equity_to_fixed_assets DECIMAL(10,4) COMMENT '股东权益与固定资产比率(%)',
    debt_to_equity DECIMAL(10,4) COMMENT '负债与所有者权益比率(%)',
    long_term_assets_ratio DECIMAL(10,4) COMMENT '长期资产与长期资金比率(%)',
    capitalization_ratio DECIMAL(10,4) COMMENT '资本化比率(%)',
    fixed_asset_net_ratio DECIMAL(10,4) COMMENT '固定资产净值率(%)',
    fixed_capitalization_ratio DECIMAL(10,4) COMMENT '资本固定化比率(%)',
    equity_multiplier DECIMAL(10,4) COMMENT '产权比率(%)',
    liquidation_value_ratio DECIMAL(10,4) COMMENT '清算价值比率(%)',
    fixed_asset_ratio DECIMAL(10,4) COMMENT '固定资产比重(%)',
    asset_liability_ratio DECIMAL(10,4) COMMENT '资产负债率(%)',
    cash_flow_to_sales DECIMAL(10,4) COMMENT '经营现金净流量对销售收入比率(%)',
    cash_flow_return_on_assets DECIMAL(10,4) COMMENT '资产的经营现金流量回报率(%)',
    cash_flow_to_net_income DECIMAL(10,4) COMMENT '经营现金净流量与净利润的比率(%)',
    cash_flow_to_debt DECIMAL(10,4) COMMENT '经营现金净流量对负债比率(%)',
    cash_flow_ratio DECIMAL(10,4) COMMENT '现金流量比率(%)',
    total_assets DECIMAL(30,4) COMMENT '总资产(元)',
    short_stock_invest DECIMAL(20,4) COMMENT '短期股票投资(元)',
    short_bond_invest DECIMAL(20,4) COMMENT '短期债券投资(元)',
    short_other_invest DECIMAL(20,4) COMMENT '短期其它经营性投资(元)',
    long_stock_invest DECIMAL(20,4) COMMENT '长期股票投资(元)',
    long_bond_invest DECIMAL(20,4) COMMENT '长期债券投资(元)',
    long_other_invest DECIMAL(20,4) COMMENT '长期其它经营性投资(元)',
    main_profit DECIMAL(20,4) COMMENT '主营业务利润(元)',
    non_gaap_net_profit DECIMAL(20,4) COMMENT '扣除非经常性损益后的净利润(元)',
    receivables_1y DECIMAL(20,4) COMMENT '1年以内应收帐款(元)',
    receivables_1_2y DECIMAL(20,4) COMMENT '1-2年以内应收帐款(元)',
    receivables_2_3y DECIMAL(20,4) COMMENT '2-3年以内应收帐款(元)',
    receivables_over_3y DECIMAL(20,4) COMMENT '3年以内应收帐款(元)',
    prepayment_1y DECIMAL(20,4) COMMENT '1年以内预付货款(元)',
    prepayment_1_2y DECIMAL(20,4) COMMENT '1-2年以内预付货款(元)',
    prepayment_2_3y DECIMAL(20,4) COMMENT '2-3年以内预付货款(元)',
    prepayment_over_3y DECIMAL(20,4) COMMENT '3年以内预付货款(元)',
    other_receivables_1y DECIMAL(20,4) COMMENT '1年以内其它应收款(元)',
    other_receivables_1_2y DECIMAL(20,4) COMMENT '1-2年以内其它应收款(元)',
    other_receivables_2_3y DECIMAL(20,4) COMMENT '2-3年以内其它应收款(元)',
    other_receivables_over_3y DECIMAL(20,4) COMMENT '3年以内其它应收款(元)',
    
    -- 系统字段
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    
    -- 索引优化
    INDEX idx_stock_code (stock_code),
    INDEX idx_report_date (report_date),
    UNIQUE INDEX uniq_stock_report (stock_code, report_date) COMMENT '股票代码+日期唯一索引'
    
) ENGINE=INNODB DEFAULT CHARSET=utf8mb4 COMMENT='上市公司财报数据表（含88字段完整版）';

SELECT * FROM stock_financial_reports WHERE stock_code = '000001' ORDER BY report_date DESC
SELECT COUNT(*) FROM stock_financial_reports WHERE stock_code = '000001' ORDER BY report_date DESC
SELECT COUNT(DISTINCT(stock_code)) FROM stock_financial_reports

