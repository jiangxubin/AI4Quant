import numpy as np
import math
from keras.utils import to_categorical


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

def multi_features_regression(raw_data, step_size=30):
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
    seq = [features[i:i+step_size+1, :] for i in range(length - step_size - 1)]
    # np.random.shuffle(seq) Not necessary here, because train_test_split can randomly select the train and test dataset
    origin_y = []
    temp = []
    scalers = []
    for item in seq:
        try:
            origin_y.append(item[step_size, 1])
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
    return X, y, scalers, origin_y


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
