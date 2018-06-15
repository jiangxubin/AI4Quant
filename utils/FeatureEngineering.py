from utils import DataIO, RawData
import math
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from utils import Technical_Index
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


class FatureEngineering:
    @staticmethod
    def rooling_single_object_regression(raw_data, window=5, step_size=6):
        close = list(raw_data.iloc[:, 1])
        close = np.array(close).reshape((-1, 1))
        scaler = MinMaxScaler().fit(close)
        close = list(scaler.transform(close))
        seq = [close[i*window:(i+1)*window] for i in range(len(close)//window)]
        X = np.array([seq[j:j+step_size] for j in range(len(seq)-step_size)]).squeeze()
        y = np.array([seq[k+step_size]for k in range(len(seq)-step_size)]).squeeze()
        # https://stackoverflow.com/questions/22053050/difference-between-numpy-array-shape-r-1-and-r
        return X, y, scaler

    @staticmethod
    def multi_features_regressionN(raw_data, step_size=30, predict_day=2):
        """
        对多FEATURES模型进行特征/target分离
        :param raw_data: 原始数据
        :param step_size: lstm步长
        :return:X, y
        """
        raw_data = raw_data.dropna(axis=0, how='any')
        raw_data = raw_data.sort_index(ascending=False)
        features = raw_data.values
        length = raw_data.shape[0]
        # seq = [features[i:i+step_size+predict_day, :] for i in range(length - step_size - predict_day)]
        data_set = [np.concatenate((features[i:i+step_size, :], features[i+step_size+predict_day-1, :].reshape(1, -1))) for i in range(length - step_size - predict_day)]
        origin_y = []
        temp = []
        scalers = []
        for item in data_set:
            try:
                origin_y.append(item[-1, 1])
                scaler = MinMaxScaler().fit(item)
                scaled_item = scaler.transform(item)
                temp.append(scaled_item)
                scalers.append(scaler)
            except ValueError:
                continue
        X = np.array([FatureEngineering.cut_extreme(item[:step_size, :]) for item in temp])
        y = np.array([item[-1, 1]for item in temp])
        return X, y, scalers, origin_y

    @staticmethod
    def svm_multi_features_classification(raw_data, step_size=1, labels='multiple')->tuple:
        """
        Split features and labels for SVM classification model
        :param raw_data: Features of DataFrame
        :return: X, y
        """
        raw_data = raw_data.dropna(axis=0, how='any')
        close = raw_data.iloc[:, 1].sort_index(ascending=False)
        diff = close - close.shift(1)
        change = close.pct_change()
        big_up_quantile = change.quantile(0.8)
        up_quantile = change.quantile(0.6)
        down_quantile = change.quantile(0.4)
        big_down_quantile = change.quantile(0.2)

        def binary_categorical(x)->int:
            if x > 0:
                return 1
            else:
                return -1

        def multi_categorical(x)->int:
            if x > big_up_quantile:
                return 2
            elif up_quantile < x <= big_up_quantile:
                return 1
            elif down_quantile < x <= up_quantile:
                return 0
            elif big_down_quantile < x <= down_quantile:
                return -1
            elif x < big_down_quantile:
                return -2
        if labels == 'binary':
            diff = list(map(binary_categorical, diff))
        elif labels == 'multiple':
            diff = list(map(multi_categorical, change))
        features = raw_data.values
        features = FatureEngineering.cut_extreme(features)
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        # reduced_features = FatureEngineering.dimension_reduction(normalized_features, remaining=10)
        length = raw_data.shape[0]
        X = np.array([features[i, :] for i in range(length-1)])
        y = np.array([diff[j+step_size] for j in range(length-1)])
        # scaled_X = []
        # scalers = []
        # for item in X:
        #     # item = FatureEngineering.cut_extreme(item)
        #     scaler = StandardScaler().fit(item)
        #     item = scaler.transform(item)
        #     scalers.append(scaler)
        #     # item = FatureEngineering.dimension_reduction(item, remaining=10)
        #     scaled_X.append(item)
        return X, y, scaler



    @staticmethod
    def test_pct_diff(raw_data, step_size=1):
        raw_data = raw_data.dropna(axis=0, how='any')
        close = raw_data.iloc[:, 1].sort_index(ascending=False)
        diff = close - close.shift(1)
        cha = close.pct_change()
        return diff, cha

    @staticmethod
    def lstm_multi_features_classification(raw_data, step_size=30):
        """
        对多FEATURES模型进行特征/target分离
        :param raw_data: 原始数据
        :param step_size: lstm步长
        :return:X, y
        """
        raw_data = raw_data.dropna(axis=0, how='any')
        close = raw_data.iloc[:, 1].sort_index(ascending=False)
        diff = close - close.shift(1)

        def two_categorical(x)->int:
            if x > 0:
                return 1
            else:
                return 0
        labels = list(map(two_categorical, diff))
        features = raw_data.values
        length = raw_data.shape[0]
        X = [features[i:i+step_size] for i in range(length - step_size - 1)]
        y = np.array([labels[i+step_size] for i in range(length - step_size - 1)])
        temp = []
        for item in X:
            try:
                scaler = MinMaxScaler().fit(item)
                scaled_item = scaler.transform(item)
                temp.append(scaled_item)
            except ValueError:
                continue
        X = np.array([FatureEngineering.cut_extreme(item) for item in temp])
        return X, y

    @staticmethod
    def dimension_reduction(X: np.array, remaining=10)->np.array:
        """
        Lower the dimension of features matrix by reduce the features which share some relativeness
        :param X: Features matrix
        :return: Matrix reduced array of features
        """
        pca = PCA(n_components=remaining)
        X = pca.fit_transform(X)
        return X

    @staticmethod
    def cut_extreme(features: np.array)->np.array:
        """
        Remove extreme value to the extent of mean+_3*std
        :param features: Array of features
        :return: processed features
        """
        features_mean = np.mean(features, axis=1)
        features_std = np.std(features, axis=1)
        features_pos_limit = features_mean + 3*features_std
        features_neg_limit = features_mean - 3*features_std
        for i in range(features.shape[1]):
            features[:, i] = np.clip(features[:, i], features_neg_limit[i], features_pos_limit[i])
        return features

    @staticmethod
    def feature_label_split(raw_data: list)->tuple:
        """
        Split feature and label from raw data, for the use of Keras model
        :return:X , encoded_y
        """
        X = np.array([item.values[0:10, :] for item in raw_data if item.shape[0] >= 20])
        y_all = np.array([item.loc[item.index[11], (slice(None), 'close')] for item in raw_data if item.shape[0] >= 20] )
        y = (y_all > np.mean(y_all))*1
        encoded_y = to_categorical(y, num_classes=2)
        return X, encoded_y

    @staticmethod
    def feature_label_split_tf(raw_data: list)->tuple:
        """
        Split feature and label from raw data, for the use of tensorflow model
        :return:
        """
        # raw_data = self.__get_raw_data()
        # print(raw_data[0].shape)
        X = np.array([item.values[0:10, :] for item in raw_data if item.shape[0] >= 20])
        X_T = X.transpose((1, 0, 2))
        y_all = np.array([item.loc[item.index[11], (slice(None), 'close')] for item in raw_data if item.shape[0] >= 20] )
        y = (y_all > np.mean(y_all))*1
        encoded_y = to_categorical(y, num_classes=2)
        return X_T, encoded_y

    @staticmethod
    def train_val_test_split(*sequences, train_size=0.7, validation_size=0.2):
        """
        Split whole dataset into train/validation/test dataset without mixing future data into training dataset
        :param X: Features array
        :param y: Labels array
        :return:
        """
        output_datasets = []
        for item in sequences:
            length = len(item)
            train = item[:int(length * train_size)]
            val = item[int(length * train_size):int(length * (train_size+validation_size))]
            test = item[int(length * (train_size+validation_size)):]
            output_datasets.append((train, val, test))
        return output_datasets


if __name__ == "__main__":
    raw_data = RawData.RawData.get_raw_data()
    # X, y, scaler = FatureEngineering.rooling_single_object_regression(raw_data, 5, 6)
    # X, y, scalers = FatureEngineering.multi_features__regression(raw_data, step_size=30)
    technical_indexed_data = Technical_Index.CalculateFeatures.get_all_technical_index(raw_data)
    # X, y, scalers, origin_y = FatureEngineering.multi_features_regressionN(technical_indexed_data, step_size=30, predict_day=1)
    # X, y = FatureEngineering.lstm_multi_features_classification(technical_indexed_data, step_size=30)
    # diff, cha = FatureEngineering.test_pct_diff(technical_indexed_data)
    X, y, scaler = FatureEngineering.svm_multi_features_classification(technical_indexed_data, step_size=1)