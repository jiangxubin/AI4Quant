from sklearn.svm import SVC
import pandas as pd
import numpy as np
from utils.FeatureEngineering import FatureEngineering
from utils.RawData import RawData
from utils.Technical_Index import CalculateFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from Ai4Quant.BaseAiModel.TimeSelection import BaseStrategy
from utils import Metrics


feature_size = 17

class SVM4Classification(BaseStrategy.BaseStrategy):
    def get_feature_label(self):
        """
        Prepare X(features matrix) and y(labels) for model training, validation, and test
        :return: X, y
        """
        raw_data = RawData.get_raw_data(r'E:\DX\HugeData\Index\test.csv', r'E:\DX\HugeData\Index\nature_columns.csv')
        tech_indexed_data = CalculateFeatures.get_all_technical_index(raw_data)
        X, y, scaler = FatureEngineering.svm_multi_features_classification(tech_indexed_data)
        return X, y, scaler

    def __build_model(self, C: float, gamma, kernel: str):
        """
        Build SVM model
        :return:
        """
        if gamma is None:
            clf = SVC(C=0.8, kernel='linear')
            self.model = clf
        else:
            clf = SVC(C=C, gamma=gamma, kernel=kernel)
            self.model = clf

    def fit(self, X: np.array, y: np.array, C: float, gamma: float, kernel: str):
        '''
        Train model
        :param X: features matrix X
        :param y: labels matrix y
        :param C: Penalize the softened variable
        :param gamma: Coefficient of the innner multiplication of two Vectors
        :param kernel:
        :return:
        '''
        self.__build_model(C, gamma, kernel)
        self.model.fit(X, y)

    def predict(self, X: np.array):
        """
        Predict the distance between example and seperating hyperplane which is equal to label
        :param example: np array of shape(n, 1)
        :return: laebl
        """
        pred = self.model.predict(X)
        return pred

    def tune_hyperparams(self, X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array):
        """
        Tune the model to find the most suitable param
        :return: Best tuned param
        """
        # param_grid = [{'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear']},
        # param_grid = [
        #     {'C': [0.1, 0.3, 1, 3, 10], 'gamma': [0.01, 0.03, 0.1, 0.3, 1, 3], 'kernel': ['rbf', 'sigmoid']}
        # ]
        param_grid = [
            {'C': [0.1, 0.3, 1, 3, 10], 'gamma': [0.1, 0.3, 1, 3, 10], 'kernel': ['rbf', 'sigmoid']}
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
        cv_df.to_csv(r'E:\DX\Ai4Quant\ModelOutput\Tuning SVM hyperparameters.csv')
        # means = clf.cv_results_['mean_test_score']
        # stds = clf.cv_results_['std_test_score']
        # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        #     print("%0.3f (+/-%0.03f) for %r"
        #           % (mean, std * 2, params))
        # y_true, y_pred = y_test, clf.best_estimator_.predict(X_test)
        # print(classification_report(y_true, y_pred))
        return cv_df

    def tune_hyperameter_single_metrics(self, X_train:np.array, y_train:np.array, X_test:np.array,y_test:np.array):
        """
        Choose hyperameter based on single metrcis
        :param X_train: 
        :param y_train: 
        :param X_test: 
        :param y_test: 
        :return: Chosen hyperameter for SVM model 
        """""
        param_grid = [{'C': [0.1, 0.3, 1, 3, 10], 'gamma': [0.01, 0.03, 0.1, 0.3, 1, 3], 'kernel': ['rbf', 'sigmoid']}]
        # param_grid = [{'C': [0.1, 0.3, 1, 3, 10],  'kernel': ['linear']}]
        scores = ['roc_auc', 'f1', 'precision', 'accuracy']
        # for score in scores:
            # clf = GridSearchCV(SVC(), param_grid, cv=5, scoring=score, return_train_score=True)
        clf = GridSearchCV(SVC(), param_grid, cv=5, scoring=scores, return_train_score=True, refit=False)
        clf.fit(X_train, y_train)
            # print("Best parameters set found on development set:")
            # print(clf.best_params_)
            # print("Grid scores on development set:")
            # print("Best socre:", clf.best_score_)
        cv_df = pd.DataFrame(clf.cv_results_)
        cv_df.to_csv(r'E:\DX\Ai4Quant\ModelOutput\Tuning results of SVM hyperparameters.csv')
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
    X, y, scaler = strategy.get_feature_label()
    X_all, y_all = FatureEngineering.train_val_test_split(X, y, train_size=0.7, validation_size=0.2)
    cv_results_df = strategy.tune_hyperparams(X_all[0], y_all[0], X_all[2], y_all[2])
    top_params = []
    for top in cv_results_df.filter(regex=r'rank', axis=1).columns:
        params = cv_results_df[cv_results_df[top] == 1]['params'].values[0]
        # print(params)
        top_params.append(params)
    predict_all = {}
    for params in top_params:
        strategy.fit(X_all[0], y_all[0], C=params['C'], gamma=params['gamma'], kernel=params['kernel'])
        y_pred = strategy.predict(X_all[2])
        predict_all[str(params)] = Metrics.all_classification_score(y_all[2], y_pred)





