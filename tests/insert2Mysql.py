import akshare as ak
import pandas as pd
import pymysql
from pymysql import MySQLError
from datetime import datetime,timedelta
from tqdm import tqdm
import log4ak
from typing import Dict, List, Optional, Tuple

log = log4ak.LogManager(log_level=log4ak.INFO)# 日志配置

# 数据库配置（适配PyMySQL参数）
DB_CONFIG = {
    'host': 'localhost',
    'user': 'powerbi',
    'password': 'longyu',
    'database': 'akshare',
    'port': 3306,
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}



def insert_to_mysql(datapd, insertSqlStr):
    """存入本地数据主函数（批量插入优化）"""
    """需要根据具体存储表来修改代码"""
    try:

        # 建立PyMySQL连接（网页4标准连接方式）
        conn = pymysql.connect(**DB_CONFIG)
        log.debug(f"数据库相关信息：{DB_CONFIG}")
        #print("✅ 连接成功 | MySQL版本:", conn.get_server_info())
        log.info(f"✅ 连接成功 | MySQL版本:{conn.get_server_info()}")

        conn.autocommit(False)  # 禁用自动提交
        batch_size = 500  # 网页6推荐的批次大小
        log.debug(f"batch_size = {batch_size}")

        with conn.cursor() as cursor:
            # 分批次全量处理
            while len(datapd) >= batch_size:  # ✅ 循环处理所有完整分片
                        _execute_batch_insert(cursor, datapd[:batch_size],insertSqlStr)
                        conn.commit()  # 每批次提交
                        datapd = datapd[batch_size:]  # 动态更新剩余数据
       
            # 插入剩余数据（网页6剩余数据处理）
            if datapd:
                _execute_batch_insert(cursor, datapd,insertSqlStr)
                conn.commit()
            log.info("insert finished!")
            return "insert finished!"
                
    except MySQLError as e:
        if e.args[0] in (1062, 1586):  # 忽略主键冲突错误
            log.info(f"错误码({e.args[0]})，出现主键冲突，已忽略。{e}")
            pass
        else:
            raise 
    finally:
        if conn and conn.open:
            conn.close()

def _execute_batch_insert(cursor, data,insert_sql):
    """执行批量插入（需要根据具体存储表来修改代码）"""
    #insert_sql = """
    #INSERT IGNORE INTO `index_valuation_history` 
    #(`index_code`, `index_name`, `trade_date`, `index_value`, 
    #`pe_equal_weight_static`, `pe_static`, `pe_static_median`,
    # `pe_equal_weight_ttm`, `pe_ttm`, `pe_ttm_median`)
    #VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    #"""
    try:
        log.debug(f"insert data：{data}")
        log.debug(f"insert Sql：{insert_sql}")
        cursor.executemany(insert_sql, data)
    except MySQLError as e:
        if e.args[0] in (1062, 1586):  # 忽略主键冲突错误
            log.info(f"错误码({e.args[0]})，出现主键冲突，已忽略。{e}")
            pass
        else:
            raise 


def insert_batch_insert(data,insert_sql):
    try:
        # 建立PyMySQL连接（网页4标准连接方式）
        conn = pymysql.connect(**DB_CONFIG)
        log.info(f"✅ 连接成功 | MySQL版本:{conn.get_server_info()}")
        conn.autocommit(False)  # 禁用自动提交

        with conn.cursor() as cursor:
            _execute_batch_insert(cursor, data,insert_sql)
            conn.commit()
            log.info(f"✅ 成功插入 {cursor.rowcount} 条数据 ")
            return cursor.rowcount
    except MySQLError as e:
        if e.args[0] in (1062, 1586):  # 忽略主键冲突错误
            log.info(f"错误码({e.args[0]})，出现主键冲突，已忽略。{e}")
            pass
        else:
            log.error(f"Database error: {e}")
            if conn and conn.open:
                conn.rollback()
            raise
    finally:
        if conn and conn.open:
            conn.close()


def _execute_query(sql: str, params: tuple = ()) -> Optional[List[Tuple]]:
    """执行SQL查询并返回原始结果集"""
    try:
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
            # 获取列名（数据库原始列名）
            columns = [col[0] for col in cursor.description]
            # 逐行获取数据[6](@ref)
            rows = []
            while True:
                row = cursor.fetchone()
                if row is None:
                    break
                rows.append(row)
            return columns, rows
    except Exception as e:
        log.error(f"{params}数据库查询失败: {str(e)}")
        return None, None
    finally:
        if conn and conn.open:
            conn.close()

def getdata_fetchall(sql: str, params: tuple = ()) -> pd.DataFrame:
    """
    通过Mysql获取数据的接口
    """
    try:
        # 建立数据库连接
        with pymysql.connect(**DB_CONFIG) as conn:
            # 创建游标对象
            with conn.cursor() as cursor:
                log.debug(f"执行SQL查询: {sql}，参数: {params}")
                
                # 执行查询 - 使用参数化查询确保安全
                cursor.execute(sql, params)
                results = cursor.fetchall()
                
                # 转换结果为DataFrame
                df = pd.DataFrame(results)                
                
                log.info(f"从数据库获取到 {len(df)} 条历史PE数据")
                return df
            
    except pymysql.Error as dberr:
        log.error(f"数据库查询错误: {dberr}")
        return pd.DataFrame()
    except Exception as e:
        log.error(f"查询历史PE数据时发生未知异常: {e}")
        return pd.DataFrame()
    finally:
        if conn and conn.open:
            conn.close()


if __name__ == "__main__":
    df = ak.stock_index_pe_lg('中证1000')
    df["指数代码"]="000852.SH"
    df["指数名称"]="中证1000"
    print(df)
    #df = insert_to_mysql(df)
    