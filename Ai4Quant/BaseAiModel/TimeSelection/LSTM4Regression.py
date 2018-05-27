from keras.layers import LSTM, Dense,Activation, Input, Dropout
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
step_vector_size = 8
dropout_ratio = 0.8


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

    def __get_raw_data(self)->pd.DataFrame:
        """
        Get raw data of universe
        :return: DataFrame of raw data
        """
        raw_data = util.get_universe_data(self.universe, start_date=self.start_date, end_date=self.end_date)
        return raw_data

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
        X = Input(shape=(lstm_time_step, step_vector_size), batch_shape=(64, ))
        X = LSTM(lstm_time_step, return_state=False, return_sequence=True)(X)
        X = Dropout(rate=dropout_ratio)(X)
        X = LSTM(return_state=True, return_sequences=True)(X)
        X = Dense(1)(X)
        y = Activation('sigmoid')(X)

        Model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return Model

    def fit(self, X:pd.DataFrame, y:pd.DataFrame):
        """
        Train the LSTM model defined bove
        :param X: DataFrame of Feature
        :param y: DataFrame of label
        :return: None
        """
        self.model = self.__build_model()
        self.model.fit(X, y, batch_size=batch_size, epochs=10)


if __name__ == "__main__":
    spe_universe = LSTM4Regression.build_universe()
    # logging.info(print(spe_universe.code))
    raw_data = util.get_universe_data(spe_universe, '2018-01-04', '2018-05-24')
    X, y = util.feature_label_split(raw_data)