from keras.layers import LSTM, Dense, Softmax
import tushare as ts
import logging
import pandas as pd
from multiprocessing import Queue, Process, Pool
from Ai4Quant import util


class LSTM4Regression:
    def __init__(self, universe: list, start_date: str, end_date: str):
        """
        :param universe: 初始化股票池
        """
        self.universe = universe
        self.start_date = start_date
        self.end_date = end_date
        self.all_df = []

    @staticmethod
    def build_universe():
        today_universe = ts.get_hs300s()
        top_universe = today_universe[today_universe['weight'] > today_universe['weight'].quantile(0.8)]
        return top_universe

    def __build_model(self):
        return None

    def fit(self, start_date, end_date):
        model = self.__buid_model()
        return None


if __name__ == "__main__":
    spe_universe = LSTM4Regression.build_universe()
    logging.info(print(spe_universe.code))
    res_df = util.get_universe_data(list(spe_universe.code), '2018-01-04', '2018-05-24')
    logging.info(print(res_df.shape))
    logging.info("Hello")