from keras.layers import LSTM, SimpleRNN, GRU, Dense, Activation, Input, Dropout
from keras.models import Model
import numpy as np
from utils import Metrics
from utils.FeatureEngineering import FeatureTarget4DL, Auxiliary
from utils.RawData import RawData
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import plot_model
from utils.Technical_Index import CalculateFeatures
from Ai4Quant.BaseAiModel.TimeSelection import BaseStrategy


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
feature_size = 16
# feature_size = args.feature_size
dropout_ratio = 0.5
# dropout_ratio = args.dropout_ratio
epochs = 50
# epochs = args.epochs
output_size = 5


class Recurrent4Time(BaseStrategy.BaseStrategy):
    def get_feature_target(self, index_name=r'sh000001', predict_day=2)->tuple:
        """
        Get X for feature and y for label when everydayy has multi features
        :return: DataFrame of raw data
        """
        raw_data = RawData.get_raw_data(index_name, r'E:\DX\HugeData\Index\test.csv', r'E:\DX\HugeData\Index\nature_columns.csv')
        tech_indexed_data = CalculateFeatures.get_all_technical_index(raw_data)
        X, y, scalers = FeatureTarget4DL.feature_target4lstm_classification(tech_indexed_data, step_size=step_size,predict_day=predict_day, categories=output_size)
        return X, y, scalers

    def __build_model(self):
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
        y = Activation('softmax')(X)

        self.model = Model(inputs=[stock_feature], outputs=[y])
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    def fit(self, X: np.array, y: np.array, X_val:np.array, y_val:np.array, cell='lstm'):
        """
        Train the LSTM model defined bove
        :param X: DataFrame of Feature
        :param y: DataFrame of labelLSTM4Time-Keras.py
        :return: None
        """
        if cell == 'lstm':
            self.__build_model()
            hist = self.model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))
            return hist

    def evaluation(self, X_val:np.array, y_val:np.array, batch_size=32):
        """
        Evaluate the model on test dataset
        :param X: X_test
        :param y: y_test
        :return:
        """
        eve_result = self.model.evaluate(X_val, y_val, batch_size=batch_size)
        return eve_result


if __name__ == "__main__":
    strategy = Recurrent4Time()
    X, y, scalers = strategy.get_feature_target(r'sh000002')
    X_all, y_all = Auxiliary.train_val_test_split(X, y, train_size=0.7, validation_size=0.2)
    X_train, X_val, X_test = X_all[0], X_all[1], X_all[2]
    y_train, y_val, y_test = y_all[0], y_all[1], y_all[2]
    history = strategy.fit(X_train, y_train, cell='lstm', X_val=X_val, y_val=y_val)
    evaluation_results = strategy.evaluation(X_val, y_val)
    predicted = strategy.model.predict(X_test, batch_size=batch_size)


