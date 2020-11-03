# encoding=utf-8
import tensorflow as tf
import tushare as ts
import numpy as np
import pandas as pd
import os

from pylab import plt

from my_config.log import MyLog
from research.lstm_predict.demo1.lstm_data_process import GenLstmTrainData

logger_eml = MyLog('lstm_predict').logger

"""

"""


class NewTf:
    def __init__(self):
        self.tf_model = None

        self.x_train = None
        self.y_train = None

        self.x_test = None
        self.y_test = None

        self.input_shape = None

        self.model_name = None
        self.save_dir = None
        self.model_save_url = None

    def add_data(self, x_train, y_train):
        """
        输入数据范例：x_train 1975*60      y_train 1975

        :param y_test:
        :param x_test:
        :param x_train:
        :param y_train:
        :return:
        """
        self.x_train = x_train
        self.y_train = y_train

        # self.x_test = x_test
        # self.y_test = y_test

    def contract_tf_model(self):
        """
        构造tf
        :return:
        """

        regressor = tf.keras.Sequential()

        # Adding the first LSTM layer and some Dropout regularisation
        regressor.add(tf.keras.layers.LSTM(units=50, return_sequences=True,
                                           input_shape=self.input_shape))
        regressor.add(tf.keras.layers.Dropout(0.2))

        # Adding a second LSTM layer and some Dropout regularisation
        regressor.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
        regressor.add(tf.keras.layers.Dropout(0.2))

        # Adding a third LSTM layer and some Dropout regularisation
        regressor.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
        regressor.add(tf.keras.layers.Dropout(0.2))

        # Adding a fourth LSTM layer and some Dropout regularisation
        regressor.add(tf.keras.layers.LSTM(units=50))
        regressor.add(tf.keras.layers.Dropout(0.2))

        # Adding the output layer
        regressor.add(tf.keras.layers.Dense(units=1))

        # Compiling the RNN
        regressor.compile(optimizer='adam', loss='mean_squared_error')
        self.tf_model = regressor

        # 尝试加载模型
        self._load_model()

    def tf_fit(self, epochs=100, batch_size=32):
        """
        训练模型
        :return:
        """
        self.tf_model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size)

        # 尝试加载模型
        self._save_model()

    def tf_test(self):
        predict = list(self.tf_model.predict(self.x_test).reshape(-1))

        df_predict = pd.DataFrame({'实际': [x[-1][0] for x in self.y_test], '预测': predict})
        df_predict.loc[:, 'num'] = list(range(0, len(df_predict)))

        df_predict.plot('num', ['实际', '预测'])
        plt.show()
        plt.close()

    def predict(self):
        pass

    def set_config(self, input_shape, output_length=1, save_dir='./lstm_model_save/', model_name='lstm'):
        """
        一定要在add_data()函数之后
        :param output_length:
        :param input_shape: 注意，其输入为tuple，（3， 12），中间空格有和没有很重要，因为模型名字中使用了这个tuple的字符串形式，对空格敏感！
        :param save_dir:
        :param model_name:
        :return:
        """
        self.input_shape = input_shape
        self.model_name = model_name + '_input_%s_output_%s' % (
            str(input_shape), str(output_length))
        self.save_dir = save_dir
        self.model_save_url = self.save_dir + self.model_name

    def _save_model(self):
        try:
            # 模型保存路径如果不存在，创建
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

            self.tf_model.save_weights(self.save_dir + self.model_name)
            logger_eml.debug('成功将模型保存到路径【%s】！' % str(self.model_save_url))
            return True

        except Exception as e_:
            logger_eml.exception('保存lstm模型时出错！原因：\n%s' % str(e_))
            return False

    def _load_model(self):
        if os.path.exists(self.model_save_url + '.data-00000-of-00001'):
            self.tf_model.load_weights(self.model_save_url)
            logger_eml.debug('从【%s】加载已有模型成功！' % str(self.model_save_url))
            return True

        else:
            logger_eml.debug('模型保存路径为None，模型加载失败！')
            return False


if __name__ == '__main__':

    """---------------------------------- （一）生成训练和测试数据 ----------------------------------------"""
    # 以民生银行01年~18年，18年的数据作为训练数据
    df_train = ts.get_k_data('000001', start='2001-01-01', end='2018-0101')

    # 创建数据预处理类对象并赋值数据
    gltd = GenLstmTrainData(None)
    gltd.data = df_train

    # 定义feature，用每日开盘价、最高价、最低价作为feature
    gltd.feature_col = ['open', 'close', 'low']

    # 以未来20日收盘价的中位数作为label
    gltd.add_label(roll_type='median')

    # 分割数据，用于预测，用之前四十天的数据预测未来20日的走势
    train_data = gltd.slice_df_to_train_data(40)

    """---------------------------------- （二）定义并训练LSTM模型 ----------------------------------------"""
    # 创建TensorFlow模型，使用训练数据进行学习
    nt = NewTf()

    # 设置训练数据和测试数据
    nt.add_data(
        x_train=np.array([x[0] for x in train_data]),
        y_train=np.array([x[1] for x in train_data])
    )
    nt.set_config(input_shape=(41, 3))

    # 定义TensorFlow 的 LSTM模型并训练
    nt.contract_tf_model()
    nt.tf_fit(epochs=100)

    end = 0
