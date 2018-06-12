from keras.models import Model
import numpy as np
from utils import RawData, FeatureEngineering
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import plot_model
from utils.Technical_Index import CalculateFeatures


class BaseStrategy:
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

    def __build_model(self):
        """
        Build the ML/DL model for index prediction or selelction
        :return: model
        """
        return self.model

    def fit(self, X: np.array, y: np.array):
        """
        Train the ML/DL model defined bove
        :param X: np.array of Feature
        :param y: np.array of labels
        :return: None
        """

    def evaluation(self, X_valuation: np.array, y_evaluation: np.array):
        """
        Evaluating the strategy model fitted above to tune hyperameters
        :param X_valuation: np.array of features for evaluation
        :param y_evaluation: np.array of labels for
        :return: Evaluation results
        """

    def predict(self, today_features: np.array)->float:
        """
        According to passed data of realtime to predict tomorow's index price
        :param today_features: np.array of features of shape(30, 17)
        :return: Predicted index price
        """