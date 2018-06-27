from utils import DataIO, RawData
import math
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from utils import Technical_Index
from sklearn.decomposition import PCA
from utils.float_to_categorical import multi_categorical_pct, multi_categorical_value


class FeatureTarget4DL:
    @staticmethod
    def feature_target4lstm_regression(raw_data, step_size=30, predict_day=2):
        """
        对多FEATURES模型进行特征/target分离
        :param raw_data: 原始数据
        :param step_size: lstm步长
        :param predict_day: 预测滞后天数
        :return:X, y
        """
        raw_data = raw_data.dropna(axis=0, how='any')
        # raw_data = raw_data.sort_index(ascending=False)
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
        X = np.array([Auxiliary.cut_extreme(item[:step_size, :]) for item in temp])
        y = np.array([item[-1, 1]for item in temp])
        return X, y, scalers, origin_y

    @staticmethod
    def feature_target4lstm_classification(raw_data, step_size=30, predict_day=2, categories=5):
        """
        对多FEATURES模型进行特征/target分离
        :param raw_data: 原始数据
        :param step_size: lstm步长
        :param predict_day: 预测滞后天数
        :return:X, y,scalers
        """
        raw_data = raw_data.dropna(axis=0, how='any')
        # raw_data = raw_data.sort_index(ascending=True)
        close = raw_data.iloc[:, 1]
        diff = close.pct_change(periods=1)
        length = raw_data.shape[0]
        raw_data = raw_data.values
        features = [raw_data[i:i+step_size, :] for i in range(length-step_size-predict_day)]
        labels = [diff[i+step_size+predict_day] for i in range(length-step_size-predict_day)]
        scaled_features = []
        scalers = []
        for item in features:
            scaler = MinMaxScaler().fit(item)
            scaled_item = scaler.transform(item)
            scaled_features.append(scaled_item)
            scalers.append(scaler)
        X = np.array(scaled_features)
        # labels_oh = multi_categorical_pct(labels, categories)
        labels_oh = multi_categorical_value(X, labels, categories)
        y = labels_oh
        return X, y, scalers


class FeatureTarget4ML:
    @staticmethod
    def feature_target4svm_classification(raw_data, predict_day=2):
        """
        Split features and labels for SVM classification model
        :param raw_data: Features of DataFrame
        :return: X, y
        """
        return X, y, scaler


class Auxiliary:
    @staticmethod
    def test_pct_diff(raw_data, step_size=1):
        raw_data = raw_data.dropna(axis=0, how='any')
        close = raw_data.iloc[:, 1].sort_index(ascending=False)
        diff = close - close.shift(1)
        cha = close.pct_change()
        return diff, cha

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
    raw_data = RawData.RawData.get_raw_data(index_name=r'sh000002')
    technical_indexed_data = Technical_Index.CalculateFeatures.get_all_technical_index(raw_data)
    # X, y, scalers, origin_y = FeatureTarget4DL.feature_target4lstm_regression(technical_indexed_data, step_size=30, predict_day=1)
    # X, y, scalers = FeatureTarget4DL.feature_target4lstm_classification(technical_indexed_data, step_size=30, predict_day=3)
    # diff, cha = Auxiliary.test_pct_diff(technical_indexed_data)
    X, y, scaler = FeatureTarget4ML.feature_target4svm_classification(technical_indexed_data, predict_day=2)