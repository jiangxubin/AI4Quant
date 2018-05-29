from keras.layers import LSTM, Dense, Activation, Input, Dropout
from keras.models import Model
import pandas as pd
import numpy as np
import util
from sklearn.preprocessing import StandardScaler
import sys

# parser = argparse.ArgumentParser()
# parser.add_argument('batch_size', type=int, help="Number of examples of each Bacth", default=32)
# parser.add_argument('lstm_units', type=int, help="Length of time steps of Sequence model", default=10)
# parser.add_argument("step_vector_size", type=int, help="Number of features of each example", default=5)
# parser.add_argument("dropout_ratio", type=float, help="Ratio to random dropuout neurons", default=0.5)
# parser.add_argument("epochs", type=int, help="Num of how much model run through all models", default=50)
# args = parser.parse_args()

batch_size = 16
# batch_size = args.batch_size
lstm_units = 5
lstm_time_steps = 10
# lstm_units = args.lstm_units
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
        """
        Build the LSTM model for traning with keras
        :return: model
        """
        stock_feature = Input(shape=(lstm_time_steps, step_vector_size))
        X = LSTM(lstm_units, return_sequences=True)(stock_feature)
        # 如果要堆叠LSTM, return_sequences必须设置为True，否则只有时间刻度最后的那个输出，不足以传递给下一层LSTM层
        X = Dropout(rate=dropout_ratio)(X)
        X = LSTM(20, return_sequences=False)(X)
        X = Dropout(rate=dropout_ratio)(X)
        X = Dense(2)(X)
        y = Activation('sigmoid')(X)

        model = Model(inputs=[stock_feature], outputs=[y])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def fit(self, X: np.array, y: np.array):
        """
        Train the LSTM model defined bove
        :param X: DataFrame of Feature
        :param y: DataFrame of label
        :return: None
        """
        self.model = self.__build_model()
        self.model.fit(X, y, batch_size=batch_size, epochs=100)


if __name__ == "__main__":
    # print(sys.path[0])
    spe_universe = util.get_universe()
    strategy = LSTM4Regression(list(spe_universe.code), start_date='2018-01-03', end_date='2018-05-26')
    X, y = strategy.get_feature_label()
    strategy.fit(X, y)