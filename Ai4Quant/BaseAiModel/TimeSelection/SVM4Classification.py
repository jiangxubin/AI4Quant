from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import pandas as pd
import numpy as np
from utils.FeatureEngineering import FeatureTarget4ML, Auxiliary
from utils.RawData import RawData
from utils.Technical_Index import CalculateFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from Ai4Quant.BaseAiModel.TimeSelection import BaseStrategy
from utils import Metrics
from sklearn.preprocessing import LabelBinarizer, label_binarize
from sklearn.metrics import accuracy_score, precision_score, f1_score

feature_size = 17


class SVM4Classification(BaseStrategy.BaseStrategy):
    def get_feature_target(self, index_name=r'sh000001', stepsize=10, predict_day=2, categories=5):
        """
        Prepare X(features matrix) and y(labels) for model training, validation, and test
        :return: X, y
        """
        raw_data = RawData.get_raw_data(index_name, ratio=True)
        tech_indexed_data = CalculateFeatures.get_all_technical_index(raw_data)
        X, y, scaler = FeatureTarget4ML.feature_target4svm_classification(tech_indexed_data, step_size=stepsize, predict_day=predict_day, categories=categories)
        # lb = LabelBinarizer()
        # y = lb.fit_transform(y)
        return X, y, scaler

    def __build_model(self, C: float, gamma, kernel: str):
        """
        Build SVM model
        :return:
        """
        if gamma is None:
            # clf = OneVsRestClassifier(SVC(C=0.8, kernel='linear'))
            clf = SVC(C=0.8, kernel='linear', decision_function_shape='ovo')
            self.model = clf
        else:
            # clf = OneVsRestClassifier(SVC(C=C, gamma=gamma, kernel=kernel))
            clf = SVC(C=C, gamma=gamma, kernel=kernel, decision_function_shape='ovo ')
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

    def predict(self, X:np.array):
        """
        Predict the distance between example and seperating hyperplane which is equal to label
        :param X: feature array of shape(n, 1)
        :return: label
        """
        pred = self.model.predict(X)
        return pred

    def tune_hyperparams(self, X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array, laebls='multiple'):
        """
        Tune the model to find the most suitable param
        :return: Best tuned param
        """
        # param_grid = [{'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear']},
        # param_grid = [
        #     {'C': [0.1, 0.3, 1, 3, 10], 'gamma': [0.01, 0.03, 0.1, 0.3, 1, 3], 'kernel': ['rbf', 'sigmoid']}
        # ]
        param_grid = [
            {'C': [0.1, 0.3, 1, 3, 10], 'gamma': [0.1, 0.3, 1, 3, 10], 'kernel': ['rbf', 'sigmoid', 'poly']}
        ]
        if laebls == 'multiple':
            clf = SVC(decision_function_shape='ovo')
            # https://stackoverflow.com/questions/26210471/scikit-learn-gridsearch-giving-valueerror-multiclass-format-is-not-supported
            clf = GridSearchCV(clf, param_grid=param_grid, cv=5, scoring=['accuracy'], refit=False,
                               return_train_score=True)
        else:
            clf = SVC(decision_function_shape='ovo')
            clf = GridSearchCV(clf, param_grid, cv=5, scoring=['accuracy'], refit=False,
                               return_train_score=True)
        clf.fit(X_train, y_train)
        cv_df = pd.DataFrame(clf.cv_results_)
        cv_df.to_csv(r'E:\DX\Ai4Quant\ModelOutput\Tuning SVM hyperparameters.csv')
        return cv_df

    def tune_hyperameter_single_metrics(self, X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array):
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
        clf = GridSearchCV(SVC(decision_function_shape='ovo'), param_grid, cv=5, scoring=scores, return_train_score=True, refit=False)
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
# https://datascience.stackexchange.com/questions/28742/how-to-structure-data-and-model-for-multiclass-classification-in-svm


if __name__ == "__main__":
    strategy = SVM4Classification()
    X, y, scaler = strategy.get_feature_target(index_name=r'sh000002', stepsize=10, predict_day=2, categories=5)
    X_all, y_all = Auxiliary.train_val_test_split(X, y, train_size=0.7, validation_size=0.2)
    X_train, X_val, X_test = X_all[0], X_all[1], X_all[2]
    y_train, y_val, y_test = y_all[0], y_all[1], y_all[2]
    strategy.fit(X_train, y_train, C=10, gamma=1, kernel='linear')
    y_pred = strategy.model.decision_function(X_test)
    y_pred_1 = strategy.model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_1)
    prc = precision_score(y_test, y_pred_1, average='micro')
    prc_1 = precision_score(y_test, y_pred_1, average=None)
    f1 = f1_score(y_test, y_pred_1,  average=None)
    f1_1 = f1_score(y_test, y_pred_1,  average='micro')
    cv_results_df = strategy.tune_hyperparams(X_train, y_train, X_test, y_test)
    # top_params = []
    # for top in cv_results_df.filter(regex=r'rank', axis=1).columns:
    #     params = cv_results_df[cv_results_df[top] == 1]['params'].values[0]
    #     # print(params)
    #     top_params.append(params)
    # predict_all = {}
    # for params in top_params:
    #     strategy.fit(X_all[0], y_all[0], C=params['C'], gamma=params['gamma'], kernel=params['kernel'])
    #     y_pred = strategy.predict(X_all[2])
    #     predict_all[str(params)] = Metrics.all_classification_score(y_all[2], y_pred)
