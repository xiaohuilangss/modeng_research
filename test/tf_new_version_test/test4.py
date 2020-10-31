# encoding=utf-8
import tensorflow as tf
import tushare as ts
import numpy as np
import pandas as pd

from research.lstm_test.lstm_data_process import GenLstmTrainData

"""

"""


class NewTf:
    def __init__(self):
        self.tf_model = None

        self.x_train = None
        self.y_train = None

        self.x_test = None
        self.y_test = None

    def add_data(self, x_train, y_train, x_test, y_test):
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

        self.x_test = x_test
        self.y_test = y_test

    def contract_tf_model(self):
        """
        构造tf
        :return:
        """

        regressor = tf.keras.Sequential()

        # Adding the first LSTM layer and some Dropout regularisation
        regressor.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(self.x_train.shape[1], self.x_train.shape[2])))
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

    def tf_fit(self, epochs=100, batch_size=32):
        """
        训练模型
        :return:
        """
        self.tf_model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size)

    def tf_test(self):
        predict = list(self.tf_model.predict(self.x_test).reshape(-1))

        df_predict = pd.DataFrame({'实际': [x[-1][0] for x in self.y_test], '预测': predict})
        df_predict.loc[:, 'num'] = list(range(0, len(df_predict)))

        df_predict.plot('num', ['实际', '预测'])

    def predict(self):
        pass


if __name__ == '__main__':

    # 以民生银行01年~18年，18年的数据作为训练数据， 用18年至今的数据作为测试数据
    df_train = ts.get_k_data('000001', start='2001-01-01', end='2018-0101')
    df_test = ts.get_k_data('000001', start='2018-01-01')

    """---------------------------------- （一）生成训练和测试数据 ----------------------------------------"""
    # 创建数据预处理类对象并赋值数据
    gltd = GenLstmTrainData(None)
    gltd.data = df_train

    # 定义feature，用每日开盘价、最高价、最低价作为feature
    gltd.feature_col = ['open', 'close', 'low']

    # 以未来20日收盘价的中位数作为label
    gltd.add_label_median()

    # 分割数据，用于预测，用之前四十天的数据预测未来20日的走势
    train_data = gltd.slice_df_to_train_data(40)

    # 生成测试数据
    gltd.data = df_test
    gltd.add_label_median()
    test_data = gltd.slice_df_to_train_data(40)

    """---------------------------------- （二）定义并训练LSTM模型 ----------------------------------------"""
    # 创建TensorFlow模型，使用训练数据进行学习
    nt = NewTf()

    # 设置训练数据和测试数据
    nt.add_data(
        x_train=np.array([x[0] for x in train_data]),
        y_train=np.array([x[1] for x in train_data]),
        x_test=np.array([x[0] for x in test_data]),
        y_test=np.array([x[1] for x in test_data])
    )

    # 定义TensorFlow 的 LSTM模型并训练
    nt.contract_tf_model()
    nt.tf_fit(epochs=5)

    # 打印预测效果
    nt.tf_test()

    end = 0