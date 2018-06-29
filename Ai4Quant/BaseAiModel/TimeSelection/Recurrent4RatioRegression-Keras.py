from keras.layers import LSTM, SimpleRNN, GRU, Dense, Activation, Input, Dropout
from keras.models import Model
import numpy as np
from utils.FeatureEngineering import FeatureTarget4DL, Auxiliary
from utils.RawData import RawData
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score
import matplotlib.pyplot as plt
from utils.Technical_Index import CalculateFeatures
from Ai4Quant.BaseAiModel.TimeSelection import BaseStrategy
from utils.float_to_categorical import multi_categorical_value

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
epochs = 20
# epochs = args.epochs
output_size = 1


class Recurrent4Time(BaseStrategy.BaseStrategy):
    def get_feature_target(self, index_name=r'sh000001', predict_day=2)->tuple:
        """
        Get X for feature and y for label when everydayy has multi features
        predict_day: predict close price of t+N day
        :return: DataFrame of raw data
        """
        raw_data = RawData.get_raw_data(index_name, r'E:\DX\HugeData\Index\test.csv', r'E:\DX\HugeData\Index\nature_columns.csv', ratio=True)
        tech_indexed_data = CalculateFeatures.get_all_technical_index(raw_data)
        # X, y, scalers= FeatureTarget4DL.feature_target4lstm_ratio_regression(tech_indexed_data, step_size=step_size,
        X, y, X_scalers, y_scaler = FeatureTarget4DL.feature_target4lstm_ratio_regression(tech_indexed_data, step_size=step_size,
                                                                               predict_day=predict_day)
        # return X, y, scalers
        return X, y

    def __build_model(self):
        """
        Build the LSTM model for traning with keras when everydayy has multi features
        :return: model
        """
        index_feature = Input(shape=(step_size, feature_size))
        X = LSTM(hidden_units_1, return_sequences=True)(index_feature)
        # 如果要堆叠LSTM, return_sequences必须设置为True，否则只有时间刻度最后的那个输出，不足以传递给下一层LSTM层
        X = Dropout(rate=dropout_ratio)(X)
        X = LSTM(hidden_units_2, return_sequences=False)(X)
        X = Dropout(rate=dropout_ratio)(X)
        X = Dense(output_size)(X)
        y = Activation('linear')(X)

        self.model = Model(inputs=[index_feature], outputs=[y])
        self.model.compile(loss='mse', optimizer='rmsprop', metrics=['mse', 'mae', 'mape', 'cosine'])

    def fit(self, X: np.array, y: np.array, cell='lstm'):
        """
        Train the LSTM model defined bove
        :param X: DataFrame of Feature
        :param y: DataFrame of labelLSTM4Time-Keras.py
        :param cell: type of neuron cell,default lstm
        :return: None
        """
        if cell == 'lstm':
            self.__build_model()
            hist = self.model.fit(X, y, batch_size=batch_size, epochs=epochs, shuffle=True)
            # self.model.save("Daul-LSTM-Regression.h5")
            # self.model.save("ModelOutput/Daul-LSTM-Regression-Addtion-Features.h5")
            # plot_model(self.model, to_file='Dual-LSTM-Regression.png', show_shapes=True)
            return hist
        # elif cell == 'rnn':
        #     self.__build_rnn_model_multi_features()
        #     self.model.fit(X, y, batch_size=batch_size, epochs=epochs)
        # elif cell == 'gru':
        #     self.__build_gru_model_multi_features()
        #     self.model.fit(X, y, batch_size=batch_size, epochs=epochs)

    def evaluation(self, X_val:np.array, y_val:np.array, batch_size=32):
        """
        Evaluate the model on validation dataset
        :param X: X_test
        :param y: y_test
        :return:
        """
        return self.model.evaluate(X_val, y_val, batch_size=batch_size)

    def predict(self, X: np.array)->np.array:
        """
        Return predicted results of trained model
        :return: predicted results
        """
        return self.model.predict(X, batch_size=batch_size, verbose=2)


if __name__ == "__main__":
    strategy = Recurrent4Time()
    # X, y, scalers = strategy.get_feature_target(predict_day=1)
    X, y  = strategy.get_feature_target(predict_day=1, index_name='sz399001')
    X_all, y_all = Auxiliary.train_val_test_split(X, y, train_size=0.7, validation_size=0.1)
    X_train, X_val, X_test = X_all[0], X_all[1], X_all[2]
    y_train, y_val, y_test = y_all[0], y_all[1], y_all[2]
    history = strategy.fit(X_train, y_train, cell='lstm')
    evaluation_results = strategy.evaluation(X_val, y_val)
    predicted_results = strategy.predict(X_test)
    predicted_category = multi_categorical_value(predicted_results)
    real_category = multi_categorical_value(y_test)
    acc = accuracy_score(real_category, predicted_category)
    f1 = f1_score(real_category, predicted_category, average='micro')
    f1_1 = f1_score(real_category, predicted_category, average='macro')
    prc = precision_score(real_category, predicted_category, average='micro')
    prc_1 = precision_score(real_category, predicted_category, average='macro')


