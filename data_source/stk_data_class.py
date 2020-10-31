# encoding=utf-8
import calendar

import copy
import jqdatasdk
import talib
import tushare as ts
import jqdatasdk as jq
import numpy as np
import pandas as pd
import math
import my_config.future_global_value as fgv

from data_source.Data_Sub import get_k_data_jq, add_stk_index_to_df, Index
from data_source.local_data.update_local_data import LocalData
from sdk.DataPro import relative_rank
from sdk.MyTimeOPT import get_current_date_str, get_current_datetime_str
from data_source.auth_info import jq_login
from my_config.log import MyLog
logger = MyLog('stk_data').logger
logger_eml = MyLog('stk_data_eml').logger


class StkData:
    """
    stk数据基础类，用来准备一些基本的数据
    数据预处理所用函数皆在于此
    """
    
    def __init__(self, stk_code, freq='1d'):
        
        self.freq = freq
        
        # 用此参数请确保为分钟数据
        self.freq_d = int(self.freq.replace('m', '').replace('d', ''))
        self.stk_code = stk_code
        
        self.data = pd.DataFrame()
        self.data = pd.DataFrame()
        self.week_data = pd.DataFrame()
        self.month_data = pd.DataFrame()
        
        # 通用变量，便于后续功能扩展之用！
        self.general_variable = None
        
        # 实时分钟数据更新逻辑所需变量，解决跨天后分钟间隔计算错误问题
        self.rt_minute_update_date = {True: get_current_date_str(), False: fgv.debug_date}.get(isinstance(fgv.debug_date, type(None)))
        self.rt_minute_update_minute = 0
        self.rt_minute_data = None

        # 实时天数据更新逻辑
        self.today_df = None
        # global debug_date
        self.today_df_update_date = {True: get_current_date_str(), False: fgv.debug_date}.get(isinstance(fgv.debug_date, type(None)))

        #继承类调用
        # self.update_today_df()

    def add_kama_line(self, period=30):
        self.data['kama_%s' % str(period)] = talib.KAMA(self.data['close'], period)

    def update_today_hlc(self, p_now):
        """
        根据实时价格，更新当天hl数据及close数据
        :return:
        """
        debug_date = fgv.debug_date
        date_now = {True: get_current_date_str(), False: debug_date}.get(isinstance(debug_date, type(None)))

        logger.debug('未经过hlc更新前数据：\n%s\np_now:\n%0.2f' % (self.today_df.to_string(), p_now))
        idx_tail = self.today_df.tail(1).index.values[0]

        if self.today_df.loc[idx_tail, 'date'] != date_now:
            self.today_df.loc[idx_tail+1, 'high'] = p_now
            self.today_df.loc[idx_tail+1, 'low'] = p_now
            self.today_df.loc[idx_tail + 1, 'close'] = p_now
            self.today_df.loc[idx_tail + 1, 'date'] = date_now
        else:
            if p_now > self.today_df.loc[idx_tail, 'high']:
                self.today_df.loc[idx_tail, 'high'] = p_now
            elif p_now < self.today_df.loc[idx_tail, 'low']:
                self.today_df.loc[idx_tail, 'low'] = p_now
            self.today_df.loc[idx_tail, 'close'] = p_now

        logger.debug('经过hlc更新后数据：\n%s' % self.today_df.to_string())

    def update_today_df(self):
        """
        更新当天网格
        :return:
        """
        debug_date = fgv.debug_date
        date_now = {True: get_current_datetime_str(), False: debug_date}.get(isinstance(debug_date, type(None)))

        if (self.today_df_update_date != date_now) | (isinstance(self.today_df, type(None))):
            self.today_df = self.get_today_df(debug_date=debug_date)
            self.today_df_update_date = date_now
            logger.debug('\n网格计算，检测到跨天，today_df变量更新成功！')
        else:
            logger.debug('\n网格计算，未检测到跨天，today_df变量无需更新！')

    def get_today_df(self, debug_date=None):
        df_ = get_k_data_jq(stk=self.stk_code, count=50,
                            end_date={True: get_current_datetime_str(), False: debug_date}
                            .get(isinstance(debug_date, type(None))))

        # 调整df，增加date列，增加正数index列
        df_['datetime'] = df_.index
        df_['datetime_str'] = df_.apply(lambda x: str(x['datetime']), axis=1)

        # 重置索引
        df_ = df_.reset_index(drop=True)

        return df_

    def init_today_minute_data(self, count=200, debug_datetime=None):
        """
        为了避免重复下载数据，有时需要维护最近一段时间的分钟数据
        :return:
        """
        try:
            datetime_now = {True: get_current_datetime_str(), False: fgv.debug_datetime}.get(isinstance(fgv.debug_datetime, type(None)))
            self.rt_minute_data = get_k_data_jq(stk=self.stk_code, freq=self.freq, count=count,
                                                end_date=datetime_now)
            self.rt_minute_data.loc[:, 'datetime'] = self.rt_minute_data.index
            self.rt_minute_data.loc[:, 'num'] = range(len(self.rt_minute_data))
            self.rt_minute_data = self.rt_minute_data.set_index('num')
            
            self.rt_minute_update_minute = self.get_minute_from_datetime(self.rt_minute_data.tail(1)['datetime'].values[0])
            return True
    
        except Exception as e_:
            logger_eml.exception('初始下载“rt minute数据”时出错！具体为：%s' % str(e_))
            return False
    
    @staticmethod
    def get_minute_from_datetime(datetime):
        try:
            return int(str(datetime)[11:16].replace(':', ''))
        except Exception as e_:
            logger.exception(str(e_))
    
    def update_rt_minute_global_info(self):
        debug_date = fgv.debug_date
        date_now = {True: get_current_date_str(), False: debug_date}.get(isinstance(debug_date, type(None)))
        if self.rt_minute_update_date != date_now:
            self.rt_minute_update_minute = 900
            self.rt_minute_update_date = date_now

    def update_rt_minute_data(self, rt_p):

        debug_datetime = fgv.debug_datetime

        # 获取当前分钟数
        try:

            # 检查是否跨天
            self.update_rt_minute_global_info()

            datetime_now = {True: get_current_datetime_str(), False: debug_datetime}.get(
                isinstance(debug_datetime, type(None)))

            minute_now = self.get_minute_from_datetime(datetime_now)
       
            if minute_now - self.rt_minute_update_minute >= self.freq_d:
            
                df_l = len(self.rt_minute_data)
                self.rt_minute_data.loc[df_l, 'close'] = rt_p
                self.rt_minute_data.loc[df_l, 'datetime'] = datetime_now
                logger.debug('更新对比数据：跨时段！\nrt_p：%s\n时间：%s' % (str(rt_p), datetime_now))
                logger.debug('\n【update_rt_minute_data】对比数据：%s' % self.rt_minute_data.to_string())
                
                # 记录上次更新分钟数
                self.rt_minute_update_minute = minute_now
                
            else:
                # self.rt_minute_data.loc[len(self.rt_minute_data) - 1, 'close'] = rt_p
                logger.debug('更新对比数据：同时段！\nrt_p：%s\n时间：%s' % (
                    str(rt_p), datetime_now))
                logger.debug('\n【update_rt_minute_data】对比数据：%s' % self.rt_minute_data.to_string())
    
        except Exception as e_:
            logger_eml.exception(
                '更新实时minute数据计算的对比数据时出错！原因：\n %s\n数据为：\n%s' % (str(e_), self.rt_minute_data.to_string()))

    @staticmethod
    def add_rank_to_col_smart_public(df_origin, col):
        """
        将df中的一列排名话，排名列名称为col_rank
        :param df:
        :param col:
        :return:
        """
        df = copy.deepcopy(df_origin)
        
        # 去除空值
        df = df.loc[df.apply(lambda x: not pd.isnull(x[col]), axis=1), :]
        l = len(df)
        
        # 新建索引
        df.loc[:, 'idx_origin'] = df.index
        df.loc[:, 'idx_new'] = range(len(df))
        df.set_index('idx_new')
        
        df = df.sort_values(by=col, ascending=True)
        df[col + '_rank_abs'] = list(range(l))
        df[col + '_rank'] = df.apply(lambda x: x[col + '_rank_abs'] / l, axis=1)
        
        # 恢复原有顺序及索引
        df = df.sort_index(ascending=True)
        df = df.set_index('idx_origin')
        
        df_origin = df
        
        return df_origin

    def add_rank_to_col_smart(self, col):
        """
        将df中的一列排名话
        :param df:
        :param col:
        :return:
        """
        df = copy.deepcopy(self.data)
        l = len(df)

        df = df.sort_values(by=col, ascending=True)
        df[col + '_rank_abs'] = list(range(l))
        self.data[col + '_rank'] = df.apply(lambda x: x[col + '_rank_abs'] / l, axis=1)

    def read_local_data(self, local_dir):
        self.data = LocalData.read_stk(local_dir=local_dir, stk_=self.stk_code).tail(40)
        
    def down_minute_data(self, count=400, freq=None):
        if pd.isnull(freq):
            self.data = get_k_data_jq(self.stk_code, count=count,
                                      end_date=get_current_datetime_str(), freq=self.freq)
        else:
            self.data = get_k_data_jq(self.stk_code, count=count,
                                      end_date=get_current_datetime_str(), freq=freq)
    
    def down_day_data(self, count=150, start_date=None, end_date=None):
        self.data = get_k_data_jq(
            self.stk_code,
            count=count,
            start_date=start_date,
            end_date=end_date,
            freq=self.freq)
    
    def add_week_month_data(self):
        """
        给定日线数据，计算周线/月线指标！
        :return:
        """
        
        df = self.data
        
        if len(df) < 350:
            print('函数week_MACD_stray_judge：' + self.stk_code + '数据不足！')
            return False, pd.DataFrame()
        
        # 规整
        df_floor = df.tail(math.floor(len(df) / 20) * 20 - 19)
        
        # 增加每周的星期几
        df_floor['day'] = df_floor.apply(
            lambda x: calendar.weekday(int(x['date'].split('-')[0]), int(x['date'].split('-')[1]),
                                       int(x['date'].split('-')[2])), axis=1)
        
        # 隔着5个取一个
        if df_floor.tail(1)['day'].values[0] != 4:
            df_week = pd.concat([df_floor[df_floor.day == 4], df_floor.tail(1)], axis=0)
        else:
            df_week = df_floor[df_floor.day == 4]
        
        # 隔着20个取一个（月线）
        df_month = df_floor.loc[::20, :]
        
        self.week_data = df_week
        self.month_data = df_month
    
    @staticmethod
    def normal(list_):
        """
        列表归一化
        :param list_:
        :return:
        """
        
        c = list_
        return list((c - np.min(c)) / (np.max(c) - np.min(c)))
    
    @staticmethod
    def cal_rank_sig(sig, total):
        return relative_rank(total, sig)
    
    @staticmethod
    def cal_rank(list_):
        """
        计算排名
        :return:[0, 100], 排名为0表示为这个序列中的最小值，排名为100表示为这个序列的最大值
        """
        
        return [StkData.cal_rank_sig(x, list_) for x in list_]
    
    def cal_diff_col(self, col):
        df = self.data
        df[col + '_last'] = df[col].shift(1)
        self.data[col+'_diff'] = df.apply(lambda x: x[col] - x[col + '_last'], axis=1)

    def add_index(self):
        """
        向日线数据中增加常用指标
        :return:
        """
        self.data = add_stk_index_to_df(self.data)

        # 增加其他指标
        idx = Index(self.data)

        idx.add_cci(5)
        idx.add_cci(20)

        self.data = idx.stk_df

    def add_sar_diff(self):
        self.data['sar_close_diff'] = self.data.apply(lambda x: x['SAR'] - x['close'], axis=1)

    def add_kd_diff(self):
        """
        向日线数据中增加kd的差值值
        :return:
        """
        self.data['kd_diff'] = self.data.apply(lambda x: (x['slowk'] - x['slowd']), axis=1)

    def add_boll_width(self):
        """
        向日线数据中增加布林线宽度值
        :return:
        """
        self.data['boll_width'] = self.data.apply(lambda x: x['upper'] - x['lower'], axis=1)

    def add_rank_col(self, col_name):
        """
        对日线数据的某一个字段进行排名华
        :param col_name:
        :return:
        """
        self.data[col_name + '_rank'] = self.cal_rank(self.data[col_name])
        
    def add_mean_col(self, col, m):
        """
        求某一列的mean
        :param col:
        :param m:
        :return:
        """
        self.data[col+'_m'+str(m)] = self.data[col].rolling(window=m).mean()


class StkDataRT(StkData):
    """
    包含对一只股票进行实时计算的方法
    """
    def __init__(self, stk_code):
        super().__init__(stk_code)
        # 记录上次sar的状态，在close之上为True，反之为Flase，初始化为None
        self.sar_last = None

    def check_sar_status_change(self):
        """
        检查sar指标的波动情况
        :return:
        0: sar状态没有变化
        1：向上突破
        -1：向下突破
        """
        # 下载实时数据
        self.down_minute_data(count=20, freq='1m')

        # 计算sar指数
        self.data['sar'] = talib.SAR(self.data.high, self.data.low, acceleration=0.05, maximum=0.2)

        # 获取sar最新状态
        row_tail = self.data.tail(1)
        sar_status_now = row_tail['sar'].values[0] > row_tail['close'].values[0]

        if pd.isnull(self.sar_last):
            self.sar_last = sar_status_now
            return 0

        if sar_status_now != self.sar_last:
            if sar_status_now:
                return 1
            else:
                return -1
        else:
            return 0


if __name__ == '__main__':

    jq_login()

    sd = StkData('000001')
    sd.update_today_hlc(20)

    sd = StkDataRT()

    sd.down_minute_data(count=1000, freq='5m')
    sd.add_index()
    sd.data['id'] = list(range(len(sd.data)))
    sd.data.plot('id', ['close', 'MACD'], style=['*--', '*--'], subplots=True)

    for i in range(10):
        print(str(sd.check_sar_status_change()))
