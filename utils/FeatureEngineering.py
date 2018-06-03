from utils import DataIO, RawData
import math
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler,StandardScaler


class FatureEngineering:
    @staticmethod
    def rooling_single_object_regression(raw_data, window=5, step_size=6):
        close = list(raw_data.iloc[:, 2])
        close = np.array(close).reshape((-1, 1))
        scaler = MinMaxScaler().fit(close)
        close = list(scaler.transform(close))
        seq = [close[i*window:(i+1)*window] for i in range(len(close)//window)]
        X = np.array([seq[j:j+step_size] for j in range(len(seq)-step_size)]).squeeze()
        y = np.array([seq[k+step_size]for k in range(len(seq)-step_size)]).squeeze()
        # https://stackoverflow.com/questions/22053050/difference-between-numpy-array-shape-r-1-and-r
        return X, y, scaler

    @staticmethod
    def rooling_single_object_classification(raw_data, window=5, step_size=6):
        gp = raw_data.groupby('')
        return X, y, scaler

    @staticmethod
    def rolling_sampling_classification(raw_data, window=10):
        """
        Rolling through the whole dataframe of a single stock with a fixed window of single stock to sample data for training and test
        :return: X:np.array of shape(sample, time-step, feature), y:np.array of shape (sample, label)
        """
        all_feature = []
        all_output = []
        for stock_data in raw_data:
            if stock_data.shape[0] == 93:
                for i in range(0, stock_data.shape[0] - window - 1):
                    feature = stock_data.values[i:i + window, :]
                    ret = stock_data.values[i + window, -1]
                    category = 0
                    if math.fabs(ret) < 0.02:
                        category = 2
                    elif math.fabs(ret) < 0.06:
                        if ret > 0:
                            category = 3
                        else:
                            category = 1
                    else:
                        if ret > 0:
                            category = 4
                        else:
                            category = 0
                    all_feature.append(feature)
                    all_output.append(category)
            else:
                continue
        all_feature = np.array(all_feature)
        all_output = np.array(all_output)
        encoded_output = to_categorical(all_output, 5)
        return all_feature, encoded_output

    @staticmethod
    def rolling_sampling_regression(raw_data, window=10):
        """
        Rolling through the whole dataframe of a single stock with a fixed window of single stock to sample data for training and test
        :return: X:np.array of shape(sample, time-step, feature), y:np.array of shape (sample, target)
        """
        all_feature = []
        all_output = []
        for stock_data in raw_data:
            if stock_data.shape[0] == 93:
                for i in range(0, stock_data.shape[0] - window - 1):
                    feature = stock_data.values[i:i + window, :]
                    close = stock_data.values[i + window, -2]
                    all_feature.append(feature)
                    all_output.append(close)
            else:
                continue
        all_feature = np.array(all_feature)
        all_output = np.array(all_output)
        return all_feature, all_output

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
    X, y, scaler = FatureEngineering.rooling_single_object_regression(raw_data, 5, 6)
