# encoding=utf-8
import talib

from data_source.stk_data_class import StkData
from pylab import *
from sdk.MyTimeOPT import minus_datetime_str, get_current_datetime_str
from sdk.pic_plot.plot_opt_sub import add_axis

import tushare as ts

import pandas as pd

"""
m20策略相关的类
"""


class StkAveData(StkData):
    def __init__(self, stk_code, freq='1d'):
        super().__init__(stk_code, freq)
        
    def add_average_line(self, m):
        self.data['m_' + str(m)] = self.data['close'].rolling(window=m).mean()

    def add_atr(self, period=14):
        self.data['atr'] = talib.ATR(self.data['high'], self.data['low'], self.data['close'], period)

    def add_average_pn(self, m):
        """
        m：mean
        pn：positive，negative
        :param m:
        :return:
        """
        self.data['m_pn_' + str(m)] = self.data.apply(lambda x: x['close'] - x['m_' + str(m)] >= 0, axis=1)

    def add_pn_pot(self, m):
        """
        找出穿越m线的点
        :return:
        """
        self.data['m_pn_last_' + str(m)] = self.data['m_pn_' + str(m)].shift(1)
        self.data['pn_pot_' + str(m)] = self.data.apply(lambda x: x['m_pn_' + str(m)] != x['m_pn_last_' + str(m)],
                                                        axis=1)

    def plot_average(self, m, m_com=None):
        """
        图示m线概况
        :param m:
        :return:
        """
        self.add_average_line(m)
        self.add_average_pn(m)
        df = self.data
    
        df = df.reset_index().reset_index()
    
        df_p = df[df.apply(lambda x: x['m_pn_' + str(m)], axis=1)]
        df_n = df[df.apply(lambda x: not x['m_pn_' + str(m)], axis=1)]
    
        fig, ax = subplots(ncols=1, nrows=1)
    
        ax.plot(df['level_0'], df['close'], 'b*--', label='close')
        ax.plot(df_p['level_0'], df_p['close'], 'r*', label='close_m' + str(m) + '_p')
        ax.plot(df_n['level_0'], df_n['close'], 'g*', label='close_m' + str(m) + '_n')
        ax.plot(df_n['level_0'], df_n['m_' + str(m)], 'y--', label='close_m%s' % str(m))

        if not isinstance(m_com, type(None)):
            ax.plot(df_n['level_0'], df_n['m_' + str(m_com)], 'gray--', label='close_m%s' % str(m_com))

        ax.legend(loc='best')
        ax = add_axis(ax, df['datetime'], 60, fontsize=5, rotation=90)
    
        plt.show()

    
class AverageStatics(StkAveData):
    def __init__(self, stk_code, freq='1d'):
        super().__init__(stk_code, freq=freq)
        self.down_day_data()

    def splice_m_seg(self, m):
        """
        运行该函数前提应该运行“add_average_pn及add_pn_pot”两个函数，
        使得数据中已有pn及pn转折点

        因为该函数单独生成另一种数据副本，所以中间操作不再self.data中进行，以便保持
        self.data的纯净。
        :param m:
        :return:
        """
    
        # 增加指数
        self.add_idx()
    
        # 增加20日线信息
        self.add_average_line(m)
        self.add_average_pn(m)
        self.add_pn_pot(m)
    
        # 分割线段
        df = self.data.dropna(axis=0)
        line_n = 0
        for idx in df.index:
            if df.loc[idx, 'pn_pot_' + str(m)]:
                line_n = line_n + 1
        
            df.loc[idx, 'm' + str(m) + '_seg_num'] = line_n
    
        # 根据分段编号进行分组，得到小段list
        seg_list = list(df.groupby(by='m' + str(m) + '_seg_num'))
    
        def seg_pro(seg):
            """
            对seg数据进行处理，总结为一个字典格式，输入的seg应该为一个df
            :param seg:
            :return:
            """
            c_list = list(seg['close'])
            return {
                'seg_length': len(seg),
                'seg_change_ratio_p': (np.max(c_list) - c_list[0]) / (c_list[0] + 0.0000000000001),
                'seg_change_ratio_n': (np.min(c_list) - c_list[0]) / (c_list[0] + 0.0000000000001),
                'bias930_rank': seg.tail(1)['bias930_rank'].values[0],
                'bias39_rank': seg.tail(1)['bias39_rank'].values[0],
                'macd_rank': seg.tail(1)['MACD_rank'].values[0],
                'seg_type': seg.tail(1)['m_pn_' + str(m)].values[0]
            }
    
        return pd.DataFrame([seg_pro(x[1]) for x in seg_list])
    
    def add_idx(self):
        # self.data = add_stk_index_to_df(self.data)
        #
        # # 增加排名
        # self.data = cal_df_col_rank(self.data, 'MACD')
        #
        # # 增加乖离度排名
        # self.data = Bias.add_bias_rank_public(self.data, span_q=3, span_s=9)
        # self.data = Bias.add_bias_rank_public(self.data, span_q=9, span_s=30)
        pass

    def plot(self, m):
        """
        图示m线概况
        :param m:
        :return:
        """
        self.add_average_line(m)
        self.add_average_pn(m)
        self.add_idx()
    
        df = self.data
    
        df = df.reset_index().reset_index()
    
        df_p = df[df.apply(lambda x: x['m_pn_' + str(m)], axis=1)]
        df_n = df[df.apply(lambda x: not x['m_pn_' + str(m)], axis=1)]
    
        fig, ax = subplots(ncols=1, nrows=2)
    
        ax[0].plot(df['level_0'], df['close'], 'b*--', label='close')
        ax[0].plot(df_p['level_0'], df_p['close'], 'r*', label='close_m' + str(m) + '_p')
        ax[0].plot(df_n['level_0'], df_n['close'], 'g*', label='close_m' + str(m) + '_n')
        ax[0].legend(loc='best')
        ax[0] = add_axis(ax[0], df['datetime'], 60, fontsize=5, rotation=90)
    
        ax[1].bar(df['level_0'], df['MACD'])
        plt.show()
        
        
class MNote(StkAveData):
    """
    均线提示类
    """
    def __init__(self, stk_code, m=20, freq='1d'):
        super().__init__(stk_code, freq)
        self.stk_code = stk_code
        self.freq = freq
        self.m = m
        self.last_result = None
    
    def last_m_pot_note(self, dt_now, dt_last, pot_type):
        """
        提示上次穿越的时间和类型
        :param pot_type:
        :param dt_now:
        :param dt_last:
        :return:
        """
        (days, minutes, secs) = minus_datetime_str(dt_now, dt_last)
        
        return '最近一次穿越M%s均线（freq=%s）的行为发生在 %d天 %d分钟 %d秒 之前，为“%s”类型' \
               %(str(self.m), self.freq, days, minutes, secs, {True: '涨破', False: '跌破'}.get(pot_type, '未知'))
        
    def get_last_m_stray(self):
        """
        找出上一次穿越m线的情况
        :return:
        """
        
        # 取出一副本，再进行drop操作
        df = self.data.dropna(axis=0)
        
        if df.empty:
            return '近期无穿越M%s均线（freq=%s）的行为' % (str(self.m), self.freq)
        
        # 取出最后一个转折点
        df_pot = df[df.apply(lambda x: x['pn_pot_' + str(self.m)], axis=1)]
        
        if df_pot.empty:
            return '近期无穿越M%s均线（freq=%s）的行为' % (str(self.m), self.freq)
        
        last_pot = df_pot.tail(1)
        self.last_result = last_pot['m_pn_' + str(self.m)].values[0]
        
        return self.last_m_pot_note(
            get_current_datetime_str(),
            last_pot['datetime'].values[0],
            last_pot['m_pn_' + str(self.m)].values[0])
    
    def cal_rt_m(self):
        """
        在非首次运行的情况下，进行实时均线状态提示
        :return:
        """
        pot_status_now = self.data.tail(1)['m_pn_' + str(self.m)].values[0]
        
    def m_rt_judge(self):
        
        # 获取均线数据
        if 'm' in self.freq:
            self.down_minute_data(count=60)
        else:
            self.down_day_data(count=60)
            
        self.add_average_pn(self.m)
        self.add_pn_pot(self.m)
        
        # 判断
        if pd.isnull(self.last_result):
            
            # 初次启动，判断上一次最近的转折
            return self.get_last_m_stray()
        
        else:
            pass


def plot(df, p_q, p_s):
    fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True)
    df.loc[:, 'num'] = range(len(df))
    if 'bs' not in df.columns:
        df.loc[:, 'bs'] = ['' for x in range(len(df))]

    df.loc[:, 'bs'] = df['bs'].fillna('')
    df_b = df.loc[df.apply(lambda x: x['bs'] == 'b', axis=1), :]
    df_s = df.loc[df.apply(lambda x: x['bs'] == 's', axis=1), :]

    ax[0].plot(df['num'], df['close'], 'k*--', label='close')

    ax[0].plot(df['num'], df['m_%s' % p_s], 'y-', label='quick', linewidth=2.5)
    ax[0].plot(df['num'], df['m_%s' % p_q], 'b-', label='slow', linewidth=2.5)

    ax[0].plot(df_b['num'], df_b['close'], 'g*', label='b_pot', markersize=10)
    ax[0].plot(df_s['num'], df_s['close'], 'r*', label='s_pot', markersize=10)

    df.loc[:, 'earn'] = df['earn'].fillna(method='ffill')
    ax[1].plot(df['num'], df['earn'], 'k*--', label='close')

    if len(ax) > 1:
        for _ax in ax:
            _ax.legend(loc='best')
    else:
        ax.legend(loc='best')

    plt.tight_layout(h_pad=0)

    plt.show()


if __name__ == '__main__':

    p_slow = 60
    p_quick = 30

    # jq_login()

    # 准备数据
    stk_code = '000001'

    sad = StkAveData(stk_code)
    # sad.down_minute_data(freq='1m', count=50000)
    sad.data = ts.get_k_data('000001', start='2000-01-01', end='2020-01-01')

    # 增加m line
    sad.add_average_line(p_quick)
    sad.add_average_line(p_slow)
    sad.add_kama_line(p_slow)
    sad.add_kama_line(p_quick)
    sad.add_atr()

    sad.add_average_pn(p_slow)
    sad.add_average_pn(p_quick)

    sad.add_pn_pot(p_quick)
    sad.add_pn_pot(p_slow)

    sad.data = sad.data.dropna()
    sad.data['num'] = range(len(sad.data))
    sad.data['idx'] = range(len(sad.data))
    opt_r = None
    earn = 0
    """
    sad.data['idx'] = range(len(sad.data))      
    sad.data.plot('idx', ['close', 'm_10', 'm_240'], subplots=False, style=['r*--', 'g--', 'y--'])  
    sad.data.loc[:, ['close', 'm_10', 'm_240', 'opt']]
    sad.data.plot('idx', ['close', 'kama'], subplots=False, style=['r*--', 'g--'])
    sad.data.plot('idx', ['close', 'atr'], subplots=True, style=['r*--', 'g--'])
    
    sad.data.plot('idx', ['close', 'kama_%s' % str(p_quick), 'kama_%s' % str(p_slow)], subplots=False, style=['r*--', 'g--', 'y--'])
    
    sad.data.loc[:, 'kama_diff'] = sad.data.apply(lambda x: x['kama_%s' % str(p_quick)] - x['kama_%s' % str(p_slow)], axis=1)
    sad.data.plot('idx', ['close', 'kama_diff'], subplots=True, style=['r*--', 'g*--'])
    
    """
    c_status = 0
    for idx in sad.data.index:

        m_q = sad.data.loc[idx, 'm_%s' % str(p_quick)]
        m_s = sad.data.loc[idx, 'm_%s' % str(p_slow)]
        atr = sad.data.loc[idx, 'atr']

        pn_q = sad.data.loc[idx, 'm_pn_%s' % str(p_quick)]
        pn_s = sad.data.loc[idx, 'm_pn_%s' % str(p_slow)]

        pn_pot_q = sad.data.loc[idx, 'pn_pot_%s' % str(p_quick)]
        pn_pot_s = sad.data.loc[idx, 'pn_pot_%s' % str(p_slow)]

        c = sad.data.loc[idx, 'close']

        # 开
        if (c_status == 0) & (c > m_s) & (c - m_q > atr):
            sad.data.loc[idx, 'bs'] = 'b'
            c_status = 1
            opt_r = c

        elif (c_status == 0) & (c < m_s) & (c - m_q < -atr):
            sad.data.loc[idx, 'bs'] = 's'
            c_status = -1
            opt_r = c

        if isinstance(opt_r, type(None)):
            continue

        # ping
        if (c_status == 1) & (c - m_q < -atr):
            sad.data.loc[idx, 'bs'] = 's'
            earn = earn + c - opt_r
            sad.data.loc[idx, 'earn'] = earn
            c_status = 0

        elif (c_status == -1) & (c - m_q > atr):
            sad.data.loc[idx, 'bs'] = 'b'
            earn = earn + opt_r - c
            sad.data.loc[idx, 'earn'] = earn
            c_status = 0


    """
    plot(sad.data, str(p_quick), str(p_slow))
    """
    end = 0
