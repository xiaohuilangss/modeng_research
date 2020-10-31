# encoding=utf-8

"""
本脚本用以生成lstm训练用的数据
"""
import json
import numpy as np

from data_source.auth_info import jq_login
from sklearn.preprocessing import MinMaxScaler

from data_source.data_pro_for_study import DataProForStudy

json_max_min_info = {}


class GenLstmTrainData(DataProForStudy):
    def __init__(self, stk_code, save_dir='./', count=400, freq='1d', window=20):
        super().__init__(stk_code, count=count, freq=freq, window=window)
        
        # 各个维度的归一化信息
        self.save_dir = save_dir
        self.normal_info = {}
        
        self.ochl = ['open', 'close', 'high', 'low']
        
    def save_normal_info_json(self):
        """
        将归一化信息保存到本地
        :return:
        """

        with open(self.save_dir+'normal_info.json', 'w') as f:
            json.dump(self.normal_info, f)
    
    @staticmethod
    def cal_std_param(data_list):
        """
        计算数据序列的均值和标准差，用来对该列进行标准化
        x* = （x-mu）/delta
        :return:
        """
        mu = np.mean(data_list)
        delta = np.std(data_list)
        return delta, mu

    def train_pro_lstm(self):
        """
        为训练进行预处理
        :return:
        """
    
        # 准备数据
        self.down_day_data(count=self.count)
    
        self.data.dropna(axis=0)
    
        if self.data.empty:
            return
    
        # 增加“特征”数据
        # self.add_feature()
    
        # 增加“标签”数据
        self.add_label_max()
    
        # 删除空值
        self.data = self.data.dropna(axis=0)
        """
        self.day_data.plot('datetime', ['close', 'label', 'm_median'], subplots=True, style=['*--', '*--', '*--'])
        self.day_data.plot('datetime', self.feature_rank + self.feature_diff, subplots=True)
        """
        
    def add_ochl_normal(self):
        """
        获取ochl四列的归一化值
        :return:
        """
        delta, mu = self.cal_std_param(self.data.loc[:, self.ochl].values)
        self.normal_info['ochl'] = {"delta": delta, "mu": mu}
        
    def add_col_normal(self, col):
        """
        计算指定列的归一化信息
        :param col:
        :return:
        """
        delta, mu = self.cal_std_param(self.data.loc[:, col].values[0])
        self.normal_info[col] = {"delta": delta, "mu": mu}
        
    def normalize_col(self, col):
        """
        归一化指定列
        :param col:
        :return:
        """
        delta = self.normal_info[col]['delta']
        mu = self.normal_info[col]['mu']
        
        self.data.loc[:, col] = self.data.apply(lambda x: (x[col]-mu)/delta, axis=1)
        
    def normalize_ochl(self):
        """
        对数据进行归一化
        :return:
        """
        delta = self.normal_info['ochl']['delta']
        mu = self.normal_info['ochl']['mu']
        for col in self.ochl:
            self.data.loc[:, col] = self.data.apply(lambda x: (x[col]-mu)/delta, axis=1)

    def slice_df_to_train_data(self, length):
        """
        函数功能：

        专门为LSTM模型准备训练数据之用！

        给定原始数据df、序列长度length以及作为标签的列的名字，
        根据这些信息，将df切片，生成（feature，label）的list

        :param feature_cols:
        :param length:
        :param label_col:
        :return:
        """

        # 重置索引
        df = self.data
        df = df.reset_index()
        
        # 进行相应赋值
        feature_cols = self.feature_col
        label_col = self.label_col
    
        # 进行数据切片
        r_list = []
        for idx in df.loc[0:len(df) - length - 1, :].index:
        
            # 取出这一段的df
            df_seg = df.loc[idx:idx + length, feature_cols + [label_col]]
            
            # 保存结果
            r_list.append(
                (
                    df_seg.loc[:, feature_cols].values,
                    df_seg.loc[:, [label_col]].values
                )
            )
    
        return r_list
    
    def get_train_test_data_final(self, steps):
        """
        最终获取训练、测试数据
        :return:
        """
        
        # 获取原始数据
        self.down_minute_data(count=self.count, freq=self.freq)
        
        # 添加label轴
        self.add_label(roll_type='max')
        
        # 对数据进行归一化并记录归一化数据
        self.add_ochl_normal()
        self.normalize_ochl()
        self.save_normal_info_json()
        
        """
        self.data.loc[:, 'num'] = range(len(self.data))
        self.data.plot('num', ['close', 'increase_rank'], style=['--*', '--*'], subplots=True)
        """
        
        # 分割数据
        return self.slice_df_to_train_data(steps)


if __name__ == '__main__':
    pass

