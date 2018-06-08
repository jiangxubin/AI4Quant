from sklearn.svm import SVC
import pandas as pd
import numpy as np
from utils.FeatureEngineering import FatureEngineering
from utils.RawData import RawData
from utils.DataIO import DataIO
from utils.Technical_Index import CalculateFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score, classification_report
import matplotlib.pyplot as plt


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

    def __build_model(self, C, gamma, kernel):
        """
        Build SVM model
        :return:
        """
        clf = SVC(C=0.8, gamma=0.1, kernel='rbf')
        self.model = clf
        print(type(self.model))

    def fit(self, X:np.array, y:np.array, C:float, gamma:float, kernel:str):
        """
        Train model
        :param X: features matrix X
        :param y: labels matrix y
        :return:
        """
        self.__build_model(C, gamma, kernel)
        self.model.fit(X, y)

    def predict(self, X: np.array, y:np.array):
        """
        Predict the distance between example and seperating hyperplane which is equal to label
        :param example: np array of shape(n, 1)
        :return: laebl
        """
        dist = self.model.predict(X)
        fpr, tpr, thred = roc_curve(y, dist, pos_label=1)
        f1 = f1_score(y, dist, pos_label=1)
        auc = roc_auc_score(y, dist)
        plt.plot(fpr, tpr)
        plt.plot()
        plt.title("ROC Curve")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.show()
        return auc, f1

    def tune_hyperparam(self, X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array):
        """
        Tune the model to find the most suitable param
        :return: Best tuned param
        """
        param_grid = [{'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear']},
            {'C': [0.1, 0.3, 1, 3, 10], 'gamma': [0.01, 0.03, 0.1, 0.3, 1, 3], 'kernel': ['rbf', 'sigmoid']}
        ]
        # scores = ['roc_auc', 'f1', 'precision', 'accuracy']
        # for score in scores:
        clf = GridSearchCV(SVC(), param_grid, cv=5, scoring=['roc_auc', 'f1', 'precision', 'accuracy'], refit=False, return_train_score=True)
        clf.fit(X_train, y_train)
        # print("Best parameters set found on development set:")
        # print(clf.best_params_)
        # print("Grid scores on development set:")
        # print("Best socre:", clf.best_score_)
        cv_df = pd.DataFrame(clf.cv_results_)
        # means = clf.cv_results_['mean_test_score']
        # stds = clf.cv_results_['std_test_score']
        # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        #     print("%0.3f (+/-%0.03f) for %r"
        #           % (mean, std * 2, params))
        # y_true, y_pred = y_test, clf.best_estimator_.predict(X_test)
        # print(classification_report(y_true, y_pred))
        return cv_df


if __name__ == "__main__":
    strategy = SVM4Classification()
    X, y = strategy.get_feature_label()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # strategy.fit(X_train, y_train, C=0.9, gamma=0.1, kernel='rbf')
    # auc, f1 = strategy.predict(X_test, y_test)
    cv_results_df = strategy.tune_hyperparam(X_train, y_train, X_test, y_test)
    chosen_model = cv_results_df.loc['params']



