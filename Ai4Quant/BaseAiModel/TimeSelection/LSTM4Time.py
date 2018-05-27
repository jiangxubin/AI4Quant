from keras.layers import LSTM, Dense, Activation, Input, Dropout
from keras.models import Model
import tushare as ts
import logging
import pandas as pd
import numpy as np
from multiprocessing import Queue, Process, Pool
from Ai4Quant import util
from sklearn.preprocessing import StandardScaler

batch_size = 16
lstm_time_step = 10
step_vector_size = 5
dropout_ratio = 0.5


class LSTM4Regression:
    def __init__(self, universe: list, start_date: str, end_date: str):
        """
        :param universe: 初始化股票池
        """
        self.universe = universe
        self.start_date = start_date
        self.end_date = end_date
        self.all_df = []
        self.model = None

    def get_feature_label(self)->tuple:
        """
        Get X for feature and y for label
        :return: DataFrame of raw data
        """
        raw_data = util.get_universe_data(self.universe, start_date=self.start_date, end_date=self.end_date)# get raw data
        X, y = util.feature_label_split(raw_data)
        return X, y

    def __process(self)->pd.DataFrame:
        """
        Preproess raw data to standard factor
        :return: Nature factor
        """
        raw_data = self.__get_raw_data()
        scaler = StandardScaler()
        feature = scaler.fit_transform(raw_data)
        return feature

    def __build_model(self):
        stock_feature = Input(shape=(lstm_time_step, step_vector_size))
        X = LSTM(lstm_time_step)(stock_feature)
        X = Dropout(rate=dropout_ratio)(X)
        # X = LSTM(lstm_time_step)(X)
        X = Dense(2)(X)
        y = Activation('sigmoid')(X)

        model = Model(inputs=[stock_feature], outputs=[y])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def fit(self, X:np.array, y:np.array):
        """
        Train the LSTM model defined bove
        :param X: DataFrame of Feature
        :param y: DataFrame of label
        :return: None
        """
        self.model = self.__build_model()
        self.model.fit(X, y, batch_size=batch_size, epochs=40)

    def eva


if __name__ == "__main__":
    spe_universe = util.get_universe()
    strategy = LSTM4Regression(list(spe_universe.code), start_date='2018-01-03', end_date='2018-05-26')
    X, y = strategy.get_feature_label()
    strategy.fit(X, y)