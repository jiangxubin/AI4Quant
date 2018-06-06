from sklearn.svm import SVC
import pandas as pd
import numpy as np
from utils.FeatureEngineering import FatureEngineering
from utils.RawData import RawData
from utils.DataIO import DataIO
from utils.feature_computation import CalculateFeatures


class SVM4Classification:
    def __init__(self):
        self.model = None

    def get_feature_label(self):
        """
        Prepare X(features matrix) and y(labels) for model training, validation, and test
        :return: X, y
        """
        raw_data = RawData.get_raw_data(r'E:\DX\HugeData\Index\test.csv', r'E:\DX\HugeData\Index\columns.csv')
        tech_indexed_data = CalculateFeatures.get_all_technical_index(raw_data)
        X, y = FatureEngineering.multi_features_classification(tech_indexed_data)
        return X, y

    def __build_model(self):
        return None

    def fit(self, X, y):
        return None

    def plot_contrast(self):
        return None




