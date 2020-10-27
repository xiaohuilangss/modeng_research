# encoding=utf-8
import pymysql
from sqlalchemy import create_engine

from my_config.log import MyLog
logger = MyLog('database_api').logger


class DatabaseApi:
    def __init__(self):
        pass
    
    @staticmethod
    def gen_db_conn(db_info):
        """
        建立数据库连接
        :param db_info:
        包含 ip, port, user, passwd, db_name 这些字段
        :return:
        """
        return pymysql.connect(
            host=db_info['ip'],
            port=db_info['port'],
            user=db_info['user'],
            passwd=db_info['passwd'],
            db=db_info['db_name'],
            charset='utf8')
    
    @staticmethod
    def gen_db_engine(db_info):
        """
        给定数据库信息 ip, port, user, passwd, db_name
        create_engine("mysql+pymysql://root:ypw1989@127.0.0.1:3306/temptest", max_overflow=5)
        :param db_info:
        :return:
        """

        return create_engine(
            'mysql+pymysql://%(user)s:%(passwd)s@%(ip)s:%(port)s/%(db_name)s?charset=utf8' % db_info, max_overflow=5)
    
    @staticmethod
    def gen_db_conn_eg(db_info):
        """
        同时输出conn和engine
        :return:
        """
        try:
            dbTemp = {'host': db_info['ip'],
                      'port': int(db_info['port']),
                      'user': db_info['user'],
                      'password': db_info['passwd'],
                      'database': db_info['db_name'],
                      'charset': 'utf8'}
    
            engine = create_engine(
                'mysql+pymysql://%(user)s:%(password)s@%(host)s:%(port)s/%(database)s?charset=utf8' % dbTemp)
    
            conn_tick = pymysql.connect(
                host=dbTemp['host'],
                port=dbTemp['port'],
                user=dbTemp['user'],
                passwd=dbTemp['password'],
                db=dbTemp['database'],
                charset=dbTemp['charset'])
    
            return conn_tick, engine
        except Exception as e_:
            logger.exception('生成数据库连接时出错！原因：%s' % str(e_))

