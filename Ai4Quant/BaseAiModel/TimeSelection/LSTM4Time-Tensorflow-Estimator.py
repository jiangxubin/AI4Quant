# from keras.layers import LSTM, Dense, Activation, Input, Dropout
# from keras.models import Model
import pandas as pd
import numpy as np
import util
from sklearn.preprocessing import StandardScaler
import sys
import tensorflow as tf


# parser = argparse.ArgumentParser()
# parser.add_argument('batch_size', type=int, help="Number of examples of each Bacth", default=32)
# parser.add_argument('lstm_time_step', type=int, help="Length of time steps of Sequence model", default=10)
# parser.add_argument("step_vector_size", type=int, help="Number of features of each example", default=5)
# parser.add_argument("dropout_ratio", type=float, help="Ratio to random dropuout neurons", default=0.5)
# parser.add_argument("epochs", type=int, help="Num of how much model run through all models", default=50)
# args = parser.parse_args()

batch_size = 16
# batch_size = args.batch_size
lstm_time_step = 10
# lstm_time_step = args.lstm_time_step
step_vector_size = 5
# step_vector_size = args.step_vector_size
dropout_ratio = 0.5
# dropout_ratio = args.dropout_ratio
epochs = 50
# epochs = args.epochs


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

    def get_train_data_fn(self):
        """
        Get X for feature and y for label
        :return: DataFrame of raw data
        """
        # Create Dataset
        raw_data = util.get_universe_data(self.universe, start_date=self.start_date, end_date=self.end_date)# get raw data
        X, y = util.feature_label_split_tf(raw_data)
        stock_sequence_dataset = tf.data.Dataset.from_tensor_slices((X, y))
        #Create  iterator
        iter = stock_sequence_dataset.make_one_shot_iterator()
        feature_ele, label_ele = iter.get_next()
        return feature_ele, label_ele

    def get_test_data_fn(self):
        """
        Build data pipline for test data
        :return: a dict of features and a tensor of labels
        """

    def build_Estimator(self):
        """
        Build estimator to build the LSTM model
        :return:
        """

if __name__ == "__main__":
    spe_universe = util.get_universe()
    strategy = LSTM4Regression(list(spe_universe.code), start_date='2018-01-03', end_date='2018-05-26')
    a = strategy.build_dataset()
