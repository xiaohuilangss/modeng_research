# encoding=utf-8
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from data_source.Data_Sub import add_stk_index_to_df, Index
from data_source.auth_info import jq_login
from data_source.stk_data_class import StkData
from my_config.GlobalSetting import root_path
from my_config.log import MyLog
from sdk.pic_plot.plot_opt_sub import add_axis
from pylab import *
logger = MyLog('rf_class').logger

mpl.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False

"""
使用随机森林对stk进行预测
本文件存储“数据预处理”相关的类
"""


class DataProForStudy(StkData):
    """
    为随机森林模型提供“数据预处理”的类
    """
    
    def __init__(self, stk_code, count=400, freq='1d', window=20):
        super().__init__(stk_code, freq=freq)

        self.window = window
        self.count = count

        # 总结feature
        self.feature_col = [
            'kd_diff',
            'kd_diff_diff',
            'slowk',
            'slowk_diff',
            'slowd',
            'slowd_diff',

            'boll_width_self_std',
            'boll_width_self_std_diff',

            'middle_self_std',
            'middle_self_std_diff',

            'sar_close_diff_self_std',
            'sar_close_diff_self_std_diff',

            'MACD_self_std',
            'MACD_self_std_diff',

            'RSI5',
            'RSI5_diff',
            'RSI12',
            'RSI12_diff',
            'RSI30',
            'RSI30_diff',

            'MOM',
            'MOM_diff'
        ]
        
        self.label_col = 'increase_rank'

    def gen_multi_period_train_df(self, column_label, resample='T'):
        self.train_pro()
        self.add_column_label(column_label)
        self.future_adjust()
        self.data = self.data.resample(resample).bfill()
        return self.data.loc[:, [x + column_label for x in self.feature_col]]

    def set_feature_col(self):
        """
        设置标签列
        :return:
        """
        pass

    def future_adjust(self):
        """
        为了方式使用未来数据，日期进行顺延（17号使用16号数据）
        :return:
        """
        self.data.loc[:, 'idx'] = self.data.index
        self.data.loc[:, 'idx_shift'] = self.data['idx'].shift(-1)
        self.data = self.data.dropna()
        self.data = self.data.set_index('idx_shift')
    
    def add_index(self):
        """
        向日线数据中增加常用指标
        :return:
        """
        self.data = add_stk_index_to_df(self.data)
        
        # 增加其他指标
        idx = Index(self.data)
        
        # idx.add_cci(5)
        # idx.add_cci(20)
        
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
    
    def add_rank(self, col_list):
        """
        对日线数据进行排名化
        :param col_list: ['MACD', 'MOM', 'SAR', 'RSI5', 'RSI12', 'RSI30', 'boll_width', 'kd_diff', 'slowd', 'slowk']
        :return:
        """
        
        for col_name in col_list:
            self.add_rank_col(col_name)
            
            print('完成%s的rank化' % col_name)
    
    def add_diff_col(self, col_name):
        """
        获取日线数据指定列前后两天的差值
        :return:
        """
        
        # 增加前后差值
        self.data[col_name + '_last'] = self.data[col_name].shift(1)
        self.data[col_name + '_diff'] = self.data.apply(lambda x: x[col_name] - x[col_name + '_last'], axis=1)
    
    def add_ba_diff(self, col_list):
        """
        向日线数据中增加相应列的前后两天值之差
        :return:
        """
        
        for col in col_list:
            self.add_diff_col(col)
            
    def std_by_self(self, col):
        """
        根据自身进行归一化
        :return:
        """
        self.data[col + '_self_std'] = self.data.apply(lambda x: x[col] / x['close'], axis=1)
        
    def add_feature(self):
        """
        向日线数据中增加标签
        :return:
        """
        
        # 向日线数据中增加常用指标
        self.add_index()
        
        # 增加差值('kd_diff', 'sar_close_diff')
        self.add_kd_diff()
        self.add_sar_diff()
        
        # 增加布林宽度（后续可通过与当前价格相除实现归一化 'boll_width'）
        self.add_boll_width()
        
        # 自身归一化(+'_self_std')
        _ = [self.std_by_self(x) for x in ['MACD', 'sar_close_diff', 'middle', 'boll_width']]
        
        # 增加前后差值(+ '_diff')
        self.add_ba_diff([
            'kd_diff',
            'slowk',
            'slowd',
            
            'boll_width_self_std',
            'middle_self_std',
            
            'sar_close_diff_self_std',
            
            'MACD_self_std',
            
            'RSI5',
            'RSI12',
            'RSI30',
            
            'MOM'
        ])

    def add_label_median(self):
        """
        向日线数据中增加“标签”数据,
        计算未来20日收盘价相较于当前的增长率，计算中位数
        :return:
        """
        
        def ratio_median(rb):
            """
            序列除以首值后，取中位数
            :param rb:
            :return:
            """
            c = rb.values
            return np.median(c / c[0])
        
        window = self.window
        self.data['m_median_origin'] = self.data['close'].rolling(window=window).apply(ratio_median, raw=False)
        self.data['m_median'] = self.data['m_median_origin'].shift(-window)
        
        """
        self.day_data.loc[:, ['close', 'm_median_origin', 'm_median']]
        """
        
        self.add_rank_col('m_median')
        
        # 清空空值行
        self.data = self.data.dropna(axis=0)
        
        if not self.data.empty:
            self.data.loc[:, self.label_col] = self.data.apply(lambda x: math.ceil(x['m_median_rank'] / 10), axis=1)

    def add_label(self, roll_type='max'):
        """
        向日线数据中增加“标签”数据,
        :type roll_type: ['min', 'max', 'median']
        :return:
        """
        def ratio_max(rb):
            rb = rb.values
            rb = rb/rb[0]
            l = math.ceil(len(rb)/5)
            rb=list(rb)
            rb.sort(reverse=False)
            
            return np.median(rb[:l])
            
        def ratio_median(rb):
            """
            序列除以首值后，取中位数
            :param rb:
            :return:
            """
            c = rb.values
            return np.median(c / c[0])
            
        def ratio_min(rb):
            rb = rb.values
            rb = rb / rb[0]
            l = math.ceil(len(rb) / 5)
            rb = list(rb)
            rb.sort(reverse=True)
    
            return np.median(rb[:l])

        def ratio_mean(rb):
            rb = rb.values
            mean = np.mean(rb)
            return mean/rb[0]
        
        fr = {'max': ratio_max, 'min': ratio_min, 'median': ratio_median, 'mean': ratio_mean}.get(roll_type)

        self.data['change_ratio_origin'] = self.data['close'].rolling(window=self.window).apply(fr, raw=False)
        self.data['change_ratio'] = self.data['change_ratio_origin'].shift(-self.window)

        # 清空空值行
        self.data = self.data.dropna(axis=0)
        
        self.data = self.add_rank_to_col_smart_public(self.data, 'change_ratio')

        """
            self.data.loc[:, 'num'] = list(range(0, len(self.data)))
            self.data.plot('num', ['close', 'change_ratio', 'change_ratio_rank'], subplots=True)
        """

        if not self.data.empty:
            self.data[self.label_col] = self.data.apply(lambda x: math.ceil(x['change_ratio_rank']*100 / 10), axis=1)

    def add_column_label(self, label):
        for c in self.data.columns:
            self.data = self.data.rename(columns={c: c+label})
    
    def train_pro(self):
        """
        为训练进行预处理
        :return:
        """
        
        # 准备数据
        if 'd' in self.freq:
            self.down_day_data(count=self.count)
        else:
            self.down_minute_data(count=self.count, freq=self.freq)

        self.data.dropna(axis=0)

        if self.data.empty:
            return
        
        # 增加“特征”数据
        self.add_feature()
        
        # 增加“标签”数据
        self.add_label(roll_type='median')
        
        # 删除空值
        self.data = self.data.dropna(axis=0)
        """
        self.day_data.plot('datetime', ['close', 'label', 'm_median'], subplots=True, style=['*--', '*--', '*--'])
        self.day_data.plot('datetime', self.feature_rank + self.feature_diff, subplots=True)
        """

    def predict_pro(self, local_data=False):
        """
        为预测进行预处理
        :return:
        """

        # 准备数据
        if local_data:
            self.read_local_data('C:/localdata/'+self.freq+'/')
        else:
            self.down_day_data(count=self.count)

        self.data.dropna(axis=0)

        if self.data.empty:
            return

        # 增加“特征”数据
        self.add_feature()
        
        # 增加label
        self.add_label(roll_type='median')

        # 删除空值
        self.data = self.data.dropna(axis=0)


if __name__ == '__main__':
    
    jq_login()
    
    """ ------------------------------- 预测测试 -------------------------------- """
    stk = 'M2012.XDCE'
    rf_pre = RFPreprocess(save_dir=root_path + '/Server/rf/rf_model_save/', stk_list=[stk], freq='5m')
    feature = rf_pre.get_feature_data(stk, days=500)
    for idx in feature.index:
        feature.loc[idx, 'score'], feature.loc[idx, 'chance'] = rf_pre.predict_sig([feature.loc[idx, rf_pre.dpr.feature_col].values])
    
    # 画图展示
    feature.loc[:, 'date'] = feature.index
    feature.loc[:, 'date_str'] = feature.apply(lambda x: str(x['date'])[:10], axis=1)
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)

    ax[0].plot(range(0, len(feature['date'])), feature['close'], 'g*--', label='close')
    ax[1].plot(range(0, len(feature['date'])), feature['score'], 'r*--', label='score')
    ax[2].plot(range(0, len(feature['date'])), feature['chance'], 'b*--', label='chance')
    
    # 准备下标
    x_label_series = feature.apply(lambda x: x['date_str'][2:].replace('-', ''), axis=1)
    ax[0] = add_axis(ax[0], x_label_series, 20, rotation=45)
    ax[1] = add_axis(ax[1], x_label_series, 20, rotation=45)
    ax[2] = add_axis(ax[2], x_label_series, 20, rotation=45, fontsize=8)

    for ax_sig in ax:
        ax_sig.legend(loc='best')

    fig.tight_layout()                       # 调整整体空白
    plt.subplots_adjust(wspace=0, hspace=0)  # 调整子图间距
    plt.show()
    
    end = 0
