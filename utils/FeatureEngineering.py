from utils import DataIO, RawData
import math
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler,StandardScaler


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
    def multi_features__regression(raw_data, step_size=30):
        """
        对多FEATURES模型进行特征/target分离
        :param raw_data: 原始数据
        :param step_size: lstm步长
        :return:X, y
        """
        features = raw_data.values
        length = raw_data.shape[0]
        seq = [features[i:i+step_size+1, :] for i in range(length - step_size - 1)]
        # np.random.shuffle(seq) Not necessary here, because train_test_split can randomly select the train and test dataset
        temp = []
        scalers = []
        for item in seq:
            try:
                scaler = MinMaxScaler().fit(item)
                scaled_item = scaler.transform(item)
                temp.append(scaled_item)
                scalers.append(scaler)
            except ValueError:
                continue
        X = [FatureEngineering.cut_extreme(item[:step_size, :]) for item in temp]
        X = np.array(X)
        y = [ob[step_size, 1]for ob in temp]
        y = np.array(y)
        return X, y, scalers

    @staticmethod
    def multi_features_classification(raw_data, step_size=):
        """
        Split features and labels for SVM classification model
        :param raw_data: Features of DataFrame
        :return: X, y
        """
        features = raw_data.values
        length = raw_data.shape[0]
        seq = [features[i:i+step_size+1, :] for i in range(length - step_size - 1)]
        # np.random.shuffle(seq) Not necessary here, because train_test_split can randomly select the train and test dataset
        temp = []
        scalers = []
        for item in seq:
            try:
                scaler = MinMaxScaler().fit(item)
                scaled_item = scaler.transform(item)
                temp.append(scaled_item)
                scalers.append(scaler)
            except ValueError:
                continue
        X = [FatureEngineering.cut_extreme(item[:step_size, :]) for item in temp]
        X = np.array(X)
        y = [ob[step_size, 1]for ob in temp]
        y = np.array(y)
        return X, y

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


if __name__ == "__main__":
    raw_data = RawData.RawData.get_raw_data()
    # X, y, scaler = FatureEngineering.rooling_single_object_regression(raw_data, 5, 6)
    X, y = FatureEngineering.multi_features__regression(raw_data, step_size=30)