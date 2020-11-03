# encoding=utf-8

import tensorflow as tf
import tushare as ts

from my_config.log import MyLog
from research.lstm_predict.demo1.lstm_data_process import GenLstmTrainData
from research.lstm_predict.demo1.lstm_train import NewTf

logger_eml = MyLog('lstm_predict').logger

if __name__ == '__main__':
    """---------------------------------- （三）打印测试效果 ----------------------------------------"""

    # 创建数据预处理类对象并赋值数据
    gltd = GenLstmTrainData(None)

    # 定义feature，用每日开盘价、最高价、最低价作为feature
    gltd.feature_col = ['open', 'close', 'low']

    # 用18年至今的数据作为测试数据
    df_test = ts.get_k_data('000001', start='2018-01-01')

    # 生成测试数据
    gltd.data = df_test
    gltd.add_label(roll_type='median')
    test_data = gltd.data

    # 创建TensorFlow模型，使用训练数据进行学习
    nt = NewTf()
    nt.set_config(input_shape=(41, 3))
    nt.contract_tf_model()

    gltd.test_predict_effect(test_df=test_data, lstm_model=nt.tf_model, length=40)