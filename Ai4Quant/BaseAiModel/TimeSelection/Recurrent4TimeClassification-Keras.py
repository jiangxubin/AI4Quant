from keras.layers import LSTM, SimpleRNN, GRU, Dense, Activation, Input, Dropout
from keras.models import Model
import numpy as np
from utils import RawData, FeatureEngineering
from sklearn.model_selection import train_test_split

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
step_vector_size = 6
# step_vector_size = args.step_vector_size
dropout_ratio = 0.8
# dropout_ratio = args.dropout_ratio
epochs = 50
# epochs = args.epochs
class_num = 5


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
        raw_data = RawData.RawData.get_raw_data(path=r'G:\AI4Quant\HugeData\Stock\ele_info.csv', columns_path=r'G:\AI4Quant\HugeData\Stock\columns.csv')# get raw data
        X, y = FeatureEngineering.FatureEngineering.rolling_sampling_classification(raw_data, window=time_step)
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
        y = Activation('softmax')(X)

        self.model = Model(inputs=[stock_feature], outputs=[y])
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        # return self.model

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
        y = Activation('softmax')(X)

        self.model = Model(inputs=[stock_feature], outputs=[y])
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        # return model

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
        y = Activation('softmax')(X)

        self.model = Model(inputs=[stock_feature], outputs=[y])
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        # return model

    def fit(self, X: np.array, y: np.array, cell='lstm'):
        """
        Train the LSTM model defined bove
        :param X: DataFrame of Feature
        :param y: DataFrame of labelLSTM4Time-Keras.py
        :return: None
        """
        if cell == 'lstm':
            self.__build_lstm_model()
            self.model.fit(X, y, batch_size=batch_size, epochs=epochs)
        elif cell == 'rnn':
            self.__build_rnn_model()
            self.model.fit(X, y, batch_size=batch_size, epochs=epochs)
        elif cell == 'gru':
            self.__build_gru_model()
            self.model.fit(X, y, batch_size=batch_size, epochs=epochs)

    def evaluate(self, X:np.array, y:np.array):
        """
        Evaluate the model on test dataset
        :param X: X_test
        :param y: y_test
        :return:
        """
        eve_result = self.model.evaluate(X, y, batch_size=batch_size)
        result = {"Evaluation Loss": eve_result[0], "Evaluation Accuracy": eve_result[1]}
        return result


if __name__ == "__main__":
    spe_universe = DataIO.StockRawData.get_universe()
    strategy = Recurrent4Time(list(spe_universe.code), start_date='2018-01-03', end_date='2018-05-26')
    X, y = strategy.get_feature_label()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    strategy.fit(X_train, y_train, cell='lstm')
    result = strategy.evaluate(X_test, y_test)