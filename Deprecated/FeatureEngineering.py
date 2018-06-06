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