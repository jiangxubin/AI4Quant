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
dropout_ratio = 0.8
# dropout_ratio = args.dropout_ratio
epochs = 2
# epochs = args.epochs
output_size = 1


class Recurrent4Time(BaseStrategy.BaseStrategy):
    def get_feature_label(self, index_name=r'sh000001', predict_day=2)->tuple:
        """
        Get X for feature and y for label when everydayy has multi features
        predict_day: predict close price of t+N day
        :return: DataFrame of raw data
        """
        raw_data = RawData.get_raw_data(index_name, r'E:\DX\HugeData\Index\test.csv', r'E:\DX\HugeData\Index\nature_columns.csv')
        tech_indexed_data = CalculateFeatures.get_all_technical_index(raw_data)
        # X, y, scalers, origin_y = FatureEngineering.multi_features_regression(tech_indexed_data, step_size=step_size)
        X, y, scalers, origin_y = FeatureTarget4DL.feature_target4lstm_regression(tech_indexed_data, step_size=step_size,
                                                                               predict_day=predict_day)
        return X, y, scalers, origin_y

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
            self.__build_model()
            self.model.fit(X, y, batch_size=batch_size, epochs=epochs)
            # self.model.save("Daul-LSTM-Regression.h5")
            # self.model.save("ModelOutput/Daul-LSTM-Regression-Addtion-Features.h5")
            # plot_model(self.model, to_file='Dual-LSTM-Regression.png', show_shapes=True)
        # elif cell == 'rnn':
        #     self.__build_rnn_model_multi_features()
        #     self.model.fit(X, y, batch_size=batch_size, epochs=epochs)
        # elif cell == 'gru':
        #     self.__build_gru_model_multi_features()
        #     self.model.fit(X, y, batch_size=batch_size, epochs=epochs)

    def evaluation(self, X_val:np.array, y_val:np.array, batch_size=32):
        """
        Evaluate the model on test dataset
        :param X: X_test
        :param y: y_test
        :return:
        """
        eve_result = self.model.evaluate(X_val, y_val, batch_size=batch_size)
        return eve_result

    def plot_contract(self, model, X, y, scalers):
        """
        Plot real close and predicted close in one figure
        :param model:fitted model
        :return:
        """
        def judge(k):
            if k >= 0:
                return 1
            else:
                return -1
        pred_y = model.predict(X, batch_size=batch_size)
        pred_y_reversed = np.array(
            [scaler.inverse_transform(np.array([y] * feature_size).reshape(1, -1))[0, 1] for scaler, y in
             zip(scalers, pred_y)])
        pred_diff = pred_y_reversed - np.roll(pred_y_reversed, shift=1)
        pred_up_down = list(map(judge, pred_diff))
        y_reversed = np.array(
            [scaler.inverse_transform(np.array([y] * feature_size).reshape(1, -1))[0, 1] for scaler, y in
             zip(scalers, y)])
        real_diff = y_reversed - np.roll(y_reversed, shift=1)
        real_up_down = list(map(judge, real_diff))
        y_all, y_pred_all = Auxiliary.train_val_test_split(y_reversed, pred_y_reversed, train_size=0.7, validation_size=0.2)
        y_train_pred, y_test_pred = y_pred_all[0], y_pred_all[2]
        y_all_pred = pred_y_reversed
        y_train_real, y_test_real = y_all[0], y_all[2]

        y_all_real = y_reversed
        plt.subplot(311)
        plt.plot(y_test_real, label='Test Real Index of SH000001 ')
        plt.plot(y_test_pred, label='Train Real Index of SH000001 ')
        plt.title("Real and Predicted test part")
        plt.subplot(312)
        plt.plot(y_train_real, label='Train Real Index of SH000001 ')
        plt.plot(y_train_pred, label='Train Predicted index of SH000001')
        plt.title(" Real and Predicted train part ")
        plt.subplot(313)
        plt.plot(y_all_real, label='All Real index of SH000001')
        plt.plot(y_all_pred, label='All Predicted index of SH000001')
        plt.title("Index of Real and Predicted")
        plt.legend(loc='best')
        plt.show()
        return pred_up_down, real_up_down


if __name__ == "__main__":
    strategy = Recurrent4Time()
    X, y, scalers, origin_y = strategy.get_feature_label(predict_day=5)
    X_all, y_all = Auxiliary.train_val_test_split(X, y, train_size=0.5, validation_size=0)
    X_train, X_val = X_all[0], X_all[1]
    y_train, y_val = y_all[0], y_all[1]
    strategy.fit(X_train, y_train, cell='lstm')
    evaluation_results = strategy.evaluation(X_val, y_val)
    predicted_updown, real_updown = strategy.plot_contract(strategy.model, X, y, scalers)
    predicted_all, real_all = Auxiliary.train_val_test_split(predicted_updown, real_updown)
    total_score = Metrics.all_classification_score(real_updown, predicted_updown)
    train_score = Metrics.all_classification_score(real_all[0], predicted_all[0])
    test_score = Metrics.all_classification_score(real_all[2], predicted_all[2])

