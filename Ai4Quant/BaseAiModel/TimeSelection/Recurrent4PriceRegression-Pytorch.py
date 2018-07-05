import numpy as np
from utils import Metrics
from utils.FeatureEngineering import FeatureTarget4DL, Auxiliary
from utils.RawData import RawData
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score
import matplotlib.pyplot as plt
import pandas as pd
from utils.TechnicalIndex import CalculateFeatures
from Ai4Quant.BaseAiModel.TimeSelection import BaseStrategy
from torch import optim, nn, tensor
from torch.nn import functional as F
import torch

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


class Recurrent4Time(BaseStrategy.BaseStrategy):
    def get_feature_target(self, predict_day=2)->tuple:
        """
        Get X for feature and y for label when everydayy has multi features
        predict_day: predict close price of t+N day
        :return: DataFrame of raw data
        """
        raw_data = RawData.get_raw_data(r'E:\DX\HugeData\Index\test.csv', r'E:\DX\HugeData\Index\nature_columns.csv')
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
        self.model = Model()

    def fit(self, X: np.array, y: np.array):
        """
        Train the LSTM model defined bove
        :param X: DataFrame of Feature
        :param y: DataFrame of labelLSTM4Time-Keras.py
        :return: None
        """
        # Define loss function and optimizer
        loss = nn.MSELoss()
        optimizer = optim.RMSprop(params=self.model.parameters(), lr=0.1)
        # Train the model
        for epoch in range(epochs):
            permutation = torch.randperm(X.shape[0])
            for i in range(0, X.shape[0], batch_size):
                optimizer.zero_grad()
                indices = permutation[i:i+batch_size]
                batch_X, batch_y = X[indices, :, :], y[indices]
                predicted = self.model(batch_X)


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


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bn1 = nn.BatchNorm1d()
        self.lstm1 = nn.LSTM(input_size=feature_size, hidden_size=hidden_units_1, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_ratio)
        self.lstm2 = nn.LSTM(input_size=hidden_units_1, hidden_size=hidden_units_2, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_ratio)
        self.fc = nn.Linear(in_features=hidden_units_2, out_features=output_size)
        self.hidden1 = self.init_hidden(1, batch_size, hidden_units_1)
        self.hidden2 = self.init_hidden(1, batch_size, hidden_units_2)

    def init_hidden(self, num_layers, batch_size, num_hidden_units):
        # Set initial hidden and cell states for lstm layer
        h0 = torch.rand(1, batch_size, hidden_units_1, requires_grad=True)
        c0 = torch.rand(1, batch_size, hidden_units_1, requires_grad=True)
        return h0, c0

    def forward(self, feature_matrix):
        x = self.bn1(feature_matrix)
        output_1, self.hidden1 = self.lstm1(x, self.hidden1)
        output_1 = self.dropout1(output_1)
        output_2, self.hidden2 = self.lstm2(output_1, self.hidden2)
        output_2 = self.dropout2(output_2)
        predicted = self.fc(output_2[-1, :, :])
        return predicted


if __name__ == "__main__":
    strategy = Recurrent4Time()
    X, y, scalers, origin_y = strategy.get_feature_target(predict_day=5)
    X_all, y_all = Auxiliary.train_val_test_split(X, y, train_size=0.5, validation_size=0)
    X_train, X_val = X_all[0], X_all[1]
    y_train, y_val = y_all[0], y_all[1]
    strategy.fit(X_train, y_train, cell='lstm')
    # evaluation_results = strategy.evaluation(X_val, y_val)
    # predicted_updown, real_updown = strategy.plot_contract(strategy.model, X, y, scalers)
    # predicted_all, real_all = Auxiliary.train_val_test_split(predicted_updown, real_updown)
    # total_score = Metrics.all_classification_score(real_updown, predicted_updown)
    # train_score = Metrics.all_classification_score(real_all[0], predicted_all[0])
    # test_score = Metrics.all_classification_score(real_all[2], predicted_all[2])

