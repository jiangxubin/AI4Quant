from sklearn.svm import SVC
import pandas as pd
import numpy as np
from utils.FeatureEngineering import FatureEngineering
from utils.RawData import RawData
from utils.DataIO import DataIO
from utils.Technical_Index import CalculateFeatures
from sklearn.model_selection import train_test_split


class SVM4Classification:
    def __init__(self):
        self.model = None

    def get_feature_label(self):
        """
        Prepare X(features matrix) and y(labels) for model training, validation, and test
        :return: X, y
        """
        raw_data = RawData.get_raw_data(r'E:\DX\HugeData\Index\test.csv', r'E:\DX\HugeData\Index\nature_columns.csv')
        tech_indexed_data = CalculateFeatures.get_all_technical_index(raw_data)
        X, y, _ = FatureEngineering.multi_features_classification(tech_indexed_data)
        return X, y

    def __build_model(self):
        """
        Build SVM model
        :return:
        """
        clf = SVC(C=0.8, gamma=0.1, kernel='rbf')
        self.model = clf
        print(type(self.model))

    def fit(self, X, y):
        """
        Train model
        :param X: features matrix X
        :param y: labels matrix y
        :return:
        """
        self.__build_model()
        self.model.fit(X, y)

    def predict(self, example)->int:
        """
        Predict the distance between example and seperating hyperplane which is equal to label
        :param example: np array of shape(n, 1)
        :return: laebl
        """
        dist = self.model.predict(example)
        if dist > 0:
            return 1
        else:
            return -1

    def plot_contrast(self):
        return None


if __name__ == "__main__":
    strategy = SVM4Classification()
    X, y = strategy.get_feature_label()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    strategy.fit(X_train, y_train)


