from keras.layers import LSTM, SimpleRNN, GRU, Dense, Activation, Input, Dropout
from keras.models import Model
import numpy as np
from utils import RawData, FeatureEngineering, Metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import plot_model
from utils.Technical_Index import CalculateFeatures

# parser = argparse.ArgumentParser()
# parser.add_argument('batch_size', type=int, help="Number of examples of each Bacth", default=32)
# parser.add_argument('hidden_units', type=int, help="Length of time steps of Sequence model", default=10)
# parser.add_argument("feature_size", type=int, help="Number of features of each example", default=5)
# parser.add_argument("dropout_ratio", type=float, help="Ratio to random dropuout neurons", default=0.5)
# parser.add_argument("epochs", type=int, help="Num of how much model run through all models", default=50)
# args = parser.parse_args()

batch_size = 16
# batch_size = args.batch_size
hidden_units_1 = 128
hidden_units_2 = 128
step_size = 30
# hidden_units = args.hidden_units
# feature_size = 7
feature_size = 17
# feature_size = args.feature_size
dropout_ratio = 0.8
# dropout_ratio = args.dropout_ratio
epochs = 20
# epochs = args.epochs
output_size = 1


class Recurrent4Time:
    def __init__(self):
        self.model = None

    def get_feature_label(self)->tuple:
        """
        Get X for feature and y for label when everyday has a single feature
        :return: DataFrame of raw data
        """
        raw_data = RawData.RawData.get_raw_data(r'E:\DX\HugeData\Index\test.csv', r'E:\DX\HugeData\Index\nature_columns.csv')
        X, y, scaler = FeatureEngineering.FatureEngineering.rooling_single_object_regression(raw_data, window=feature_size, step_size=step_size)
        return X, y

    @staticmethod
    def get_feature_label_multi_features()->tuple:
        """
        Get X for feature and y for label when everydayy has multi features
        :return: DataFrame of raw data
        """
        raw_data = RawData.RawData.get_raw_data(r'E:\DX\HugeData\Index\test.csv', r'E:\DX\HugeData\Index\nature_columns.csv')
        tech_indexed_data = CalculateFeatures.get_all_technical_index(raw_data)
        X, y, scalers, origin_y = FeatureEngineering.FatureEngineering.multi_features_regression(tech_indexed_data, step_size=step_size)
        return X, y, scalers, origin_y

    def __build_lstm_model(self):
        """
        Build the LSTM model for traning with keras when everyday has a single feature
        :return: model
        """
        stock_feature = Input(shape=(step_size, feature_size))
        X = LSTM(hidden_units_1, return_sequences=True)(stock_feature)
        # 如果要堆叠LSTM, return_sequences必须设置为True，否则只有时间刻度最后的那个输出，不足以传递给下一层LSTM层
        X = Dropout(rate=dropout_ratio)(X)
        X = LSTM(hidden_units_2, return_sequences=False)(X)
        X = Dropout(rate=dropout_ratio)(X)
        X = Dense(feature_size)(X)
        y = Activation('linear')(X)

        self.model = Model(inputs=[stock_feature], outputs=[y])
        self.model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape', 'cosine'])

    def __build_lstm_model_multi_features(self):
        """
        Build the LSTM model for traning with keras when everydayy has multi features
        :return: model
        """
        stock_feature = Input(shape=(step_size, feature_size))
        X = LSTM(hidden_units_1, return_sequences=True)(stock_feature)
        # 如果要堆叠LSTM, return_sequences必须设置为True，否则只有时间刻度最后的那个输出，不足以传递给下一层LSTM层
        X = Dropout(rate=dropout_ratio)(X)
        X = LSTM(hidden_units_2, return_sequences=False)(X)
        X = Dropout(rate=dropout_ratio)(X)
        X = Dense(output_size)(X)
        y = Activation('linear')(X)

        self.model = Model(inputs=[stock_feature], outputs=[y])
        self.model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape', 'cosine'])

    def __build_gru_model(self):
        """
        Build the GRU model for traning with keras
        :return: model
        """
        stock_feature = Input(shape=(step_size, feature_size))
        X = GRU(hidden_units_1, return_sequences=True)(stock_feature)
        # 如果要堆叠LSTM, return_sequences必须设置为True，否则只有时间刻度最后的那个输出，不足以传递给下一层LSTM层
        X = Dropout(rate=dropout_ratio)(X)
        X = GRU(hidden_units_2, return_sequences=False)(X)
        X = Dropout(rate=dropout_ratio)(X)
        X = Dense(feature_size)(X)
        y = Activation('linear')(X)

        self.model = Model(inputs=[stock_feature], outputs=[y])
        self.model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape', 'cosine'])

    def __build_rnn_model(self):
        """
        Build the SimpleRNN model for traning with keras
        :return: model
        """
        stock_feature = Input(shape=(step_size, feature_size))
        X = SimpleRNN(hidden_units_1, return_sequences=True)(stock_feature)
        # 如果要堆叠LSTM, return_sequences必须设置为True，否则只有时间刻度最后的那个输出，不足以传递给下一层LSTM层
        X = Dropout(rate=dropout_ratio)(X)
        X = SimpleRNN(hidden_units_2, return_sequences=False)(X)
        X = Dropout(rate=dropout_ratio)(X)
        X = Dense(feature_size)(X)
        y = Activation('linear')(X)

        self.model = Model(inputs=[stock_feature], outputs=[y])
        self.model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape', 'cosine'])

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

    def fit_multi_features(self, X: np.array, y: np.array, cell='lstm'):
        """
        Train the LSTM model defined bove
        :param X: DataFrame of Feature
        :param y: DataFrame of labelLSTM4Time-Keras.py
        :return: None
        """
        if cell == 'lstm':
            self.__build_lstm_model_multi_features()
            self.model.fit(X, y, batch_size=batch_size, epochs=epochs)
            self.model.save("Daul-LSTM-Regression.h5")
            self.model.save("Daul-LSTM-Regression-Addtion-Features.h5")
            # plot_model(self.model, to_file='Dual-LSTM-Regression.png', show_shapes=True)
        # elif cell == 'rnn':
        #     self.__build_rnn_model_multi_features()
        #     self.model.fit(X, y, batch_size=batch_size, epochs=epochs)
        # elif cell == 'gru':
        #     self.__build_gru_model_multi_features()
        #     self.model.fit(X, y, batch_size=batch_size, epochs=epochs)

    def evaluate(self, X_val:np.array, y_val:np.array, batch_size=32):
        """
        Evaluate the model on test dataset
        :param X: X_test
        :param y: y_test
        :return:
        """
        eve_result = self.model.evaluate(X_val, y_val, batch_size=batch_size)
        return eve_result

    @staticmethod
    def plot_contract(model):
        """
        Plot real close and predicted close in one figure
        :param model: Fitted model
        :return:
        """
        raw_data = RawData.RawData.get_raw_data(r'E:\DX\HugeData\Index\test.csv')
        X, y, scaler = FeatureEngineering.FatureEngineering.rooling_single_object_regression(raw_data,
        # X, y, scaler = FeatureEngineering.FatureEngineering.rooling_single_object_regression(raw_data,
                                                                                             window=feature_size,
                                                                                             step_size=step_size)
        reversed_y = list(reversed(scaler.inverse_transform(y).flatten()))
        y_pred = model.predict(X, batch_size=batch_size)
        reversed_y_pred = list(reversed(scaler.inverse_transform(y_pred).flatten()))
        plt.plot(reversed_y, label='Real Index of SH000001')
        plt.plot(reversed_y_pred, label='Predicted index of SH000001')
        plt.title("sh000001 Index of Real and Predicted by LSTM model")
        plt.legend(loc='best')
        plt.show()
        return reversed_y, reversed_y_pred

    @staticmethod
    def train_val_test_split(X: np.array, y: np.array, train_size=0.7, validation_size=0.2):
        """
        Split whole dataset into train/validation/test dataset without mixing future data into training dataset
        :param X: Features array
        :param y: Labels array
        :return:
        """
        X_train = X[:int(X.shape[0] * train_size)]
        y_train = y[:int(y.shape[0] * train_size)]
        X_val = X[int(X.shape[0] * train_size):int(X.shape[0] * (train_size+validation_size))]
        y_val = y[int(y.shape[0] * train_size):int(y.shape[0] * (train_size+validation_size))]
        X_test = X[int(X.shape[0] * (train_size+validation_size)):]
        y_test = y[int(y.shape[0] * (train_size+validation_size)):]
        return X_train, X_val, X_test, y_train, y_val, y_test

    @staticmethod
    def plot_contract_multi_features(model, X, y, scalers, origin_y):
        """
        Plot real close and predicted close in one figure
        :param model:fitted model
        :return:
        """
        X_test = X[int(X.shape[0]*0.9):]
        y_test = y[int(X.shape[0]*0.9):]
        scalers = scalers[int(X.shape[0]*0.9):]
        pred_y = model.predict(X_test, batch_size=batch_size)
        reversed_y_pred = [scaler.inverse_transform(np.array([y]*feature_size).reshape(1, -1))[0, 1] for scaler, y in zip(scalers, pred_y)]
        reversed_y_real = [scaler.inverse_transform(np.array([y]*feature_size).reshape(1, -1))[0, 1] for scaler, y in zip(scalers, y_test)]
        plt.plot(origin_y[int(X.shape[0]*0.9):], label='Real Index of SH000001 Modelled by Basic features')
        plt.plot(reversed_y_real, label='Reversed Real Index of SH000001 ')
        plt.plot(reversed_y_pred, label='Predicted index of SH000001')
        # plt.title("sh000001 Index of Real and Predicted by LSTM model")
        plt.title("sh000001 Index of Real and Predicted by LSTM model with additional features")
        plt.legend(loc='best')
        plt.show()
        return reversed_y_pred, reversed_y_real, origin_y[int(X.shape[0]*0.9):]


if __name__ == "__main__":
    strategy = Recurrent4Time()
    X, y, scalers, origin_y = Recurrent4Time.get_feature_label_multi_features()
    all_datasets = Recurrent4Time.train_val_test_split(X, y, train_size=0.7, validation_size=0.2)
    X_train, X_val = all_datasets[0], all_datasets[1]
    y_train, y_val = all_datasets[3], all_datasets[4]
    strategy.fit_multi_features(X_train, y_train, cell='lstm')
    evaluation_results = strategy.evaluate(X_val, y_val)
    # eve_result = strategy.evaluate(X_test, y_test)
    # y, y_pred = strategy.plot_contract()
    reversed_y_pred, reversed_y_real, origin_y = strategy.plot_contract_multi_features(strategy.model, X, y, scalers, origin_y)
    test_real_y = origin_y
    test_pred_y = reversed_y_pred
    ur, dr = Metrics.positive_comparision(test_real_y, test_pred_y)
