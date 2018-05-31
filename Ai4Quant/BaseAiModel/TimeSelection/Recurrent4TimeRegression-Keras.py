from keras.layers import LSTM, SimpleRNN, GRU, Dense, Activation, Input, Dropout
from keras.models import Model
import pandas as pd
import numpy as np
import util
from sklearn.preprocessing import StandardScaler
import sys

# parser = argparse.ArgumentParser()
# parser.add_argument('batch_size', type=int, help="Number of examples of each Bacth", default=32)
# parser.add_argument('hidden_units', type=int, help="Length of time steps of Sequence model", default=10)
# parser.add_argument("step_vector_size", type=int, help="Number of features of each example", default=5)
# parser.add_argument("dropout_ratio", type=float, help="Ratio to random dropuout neurons", default=0.5)
# parser.add_argument("epochs", type=int, help="Num of how much model run through all models", default=50)
# args = parser.parse_args()

batch_size = 16
# batch_size = args.batch_size
hidden_units_1 = 5
hidden_units_2 = 5
time_step = 10
# hidden_units = args.hidden_units
step_vector_size = 5
# step_vector_size = args.step_vector_size
dropout_ratio = 0.8
# dropout_ratio = args.dropout_ratio
epochs = 300
# epochs = args.epochs
class_num = 2

class Recurrent4Time:
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
        raw_data = util.StockRawData.get_universe_data(self.universe, start_date=self.start_date, end_date=self.end_date)# get raw data
        X, y = util.FatureEngineering.feature_label_split(raw_data)
        return X, y

    def __build_lstm_model(self):
        """
        Build the LSTM model for traning with keras
        :return: model
        """
        stock_feature = Input(shape=(time_step, step_vector_size))
        X = LSTM(hidden_units_1, return_sequences=True)(stock_feature)
        # 如果要堆叠LSTM, return_sequences必须设置为True，否则只有时间刻度最后的那个输出，不足以传递给下一层LSTM层
        X = Dropout(rate=dropout_ratio)(X)
        X = LSTM(hidden_units_2, return_sequences=False)(X)
        X = Dropout(rate=dropout_ratio)(X)
        X = Dense(class_num)(X)
        y = Activation('sigmoid')(X)

        model = Model(inputs=[stock_feature], outputs=[y])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def __build_gru_model(self):
        """
        Build the GRU model for traning with keras
        :return: model
        """
        stock_feature = Input(shape=(time_step, step_vector_size))
        X = GRU(hidden_units_1, return_sequences=True)(stock_feature)
        # 如果要堆叠GRU, return_sequences必须设置为True，否则只有时间刻度最后的那个输出，不足以传递给下一层LSTM层
        X = Dropout(rate=dropout_ratio)(X)
        X = GRU(hidden_units_2, return_sequences=False)(X)
        X = Dropout(rate=dropout_ratio)(X)
        X = Dense(class_num)(X)
        y = Activation('sigmoid')(X)

        model = Model(inputs=[stock_feature], outputs=[y])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def __build_rnn_model(self):
        """
        Build the SimpleRNN model for traning with keras
        :return: model
        """
        stock_feature = Input(shape=(time_step, step_vector_size))
        X = SimpleRNN(hidden_units_1, return_sequences=True)(stock_feature)
        # 如果要堆叠GRU, return_sequences必须设置为True，否则只有时间刻度最后的那个输出，不足以传递给下一层LSTM层
        X = Dropout(rate=dropout_ratio)(X)
        X = SimpleRNN(hidden_units_2, return_sequences=False)(X)
        X = Dropout(rate=dropout_ratio)(X)
        X = Dense(class_num)(X)
        y = Activation('sigmoid')(X)

        model = Model(inputs=[stock_feature], outputs=[y])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def fit(self, X: np.array, y: np.array, cell='lstm'):
        """
        Train the LSTM model defined bove
        :param X: DataFrame of Feature
        :param y: DataFrame of labelLSTM4Time-Keras.py
        :return: None
        """
        if cell == 'lstm':
            self.model = self.__build_lstm_model()
            self.model.fit(X, y, batch_size=batch_size, epochs=100)
        elif cell == 'rnn':
            self.model = self.__build_rnn_model()
            self.model.fit(X, y, batch_size=batch_size, epochs=100)
        elif cell == 'gru':
            self.model = self.__build_gru_model()
            self.model.fit(X, y, batch_size=batch_size, epochs=100)


if __name__ == "__main__":
    spe_universe = util.StockRawData.get_universe()
    strategy = Recurrent4Time(list(spe_universe.code), start_date='2018-01-03', end_date='2018-05-26')
    X, y = strategy.get_feature_label()
    strategy.fit(X, y, cell='gru')