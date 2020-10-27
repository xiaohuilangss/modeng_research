# coding = utf-8
import pandas as pd
import pymysql
from sqlalchemy import create_engine


def get_total_table_data(conn, table_name):
    """
    function:return all stk information
    :param conn:
    :param table_name:
    :return:
    """

    if not any(table_name):
        return None

    if table_name[0].isdigit():                                     # if the first element in table_name is digital
        table_name_inner = '`' + table_name + '`'
    else:
        table_name_inner = table_name

    return pd.read_sql(con=conn, sql="select * from " + table_name_inner).drop_duplicates()


def is_table_exist(conn, database_name, table_name):
    """
    function: judge if a table exist in the database
    :param conn:
    :param database_name:
    :param table_name:
    :return:
    """
    cur = conn.cursor()
    if not any(table_name):
        return None

    return cur.execute('SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = ' + "'" + table_name + "'" +
                       ' AND TABLE_SCHEMA = ' + "'" + database_name + "'")


def is_field_exist(conn, database_name, table_name, field_name):
    cur = conn.cursor()
    if not any(table_name):
        return None

    return cur.execute('SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = ' + "'" + table_name + "'" +
                       ' AND TABLE_SCHEMA = ' + "'" + database_name + "'" + "AND COLUMN_NAME = " + "'" + field_name + "'")


def insert_row_to_database(conn_param, field, value, table_name):
    cur = conn_param.cursor()
    cur.execute('INSERT INTO ' + table_name + '(' + field + ') ' + ' VALUES(' + value + ')')


def set_primary_key(conn_param, key_field, table_name, auto_increment):
    cur = conn_param.cursor()
    if auto_increment:
        cur.execute('ALTER TABLE ' + table_name + ' MODIFY ' + key_field + ' INTEGER auto_increment')
    else:
        cur.execute('ALTER TABLE ' + table_name + ' MODIFY ' + key_field)


def add_columns(conn_param, table_name, columns_name):
    cur = conn_param.cursor()
    cur.execute('ALTER TABLE ' + table_name + ' ADD ' + columns_name + ' VARCHAR(10) NOT NULL')


def gen_db_conn(dbInfoParam, dbNameParam):

    dbTemp = {'host': dbInfoParam['host'],
               'port': dbInfoParam['port'],
               'user': dbInfoParam['user'],
               'password': dbInfoParam['password'],
               'database': dbNameParam,
               'charset': dbInfoParam['charset']}

    engine = create_engine('mysql+pymysql://%(user)s:%(password)s@%(host)s:%(port)s/%(database)s?charset=utf8&autoReconnect=true' % dbTemp)

    conn_tick = pymysql.connect(
        host=dbTemp['host'],
        port=dbTemp['port'],
        user=dbTemp['user'],
        passwd=dbTemp['password'],
        db=dbTemp['database'],
        charset=dbTemp['charset'])

    return conn_tick, engine


