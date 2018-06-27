'''
Convert a float to categorical
'''
# RF:https://docs.scipy.org/doc/numpy/reference/generated/numpy.eye.html
import numpy as np
import pandas as pd
from utils.RawData import RawData
from utils.Technical_Index import CalculateFeatures

def binary_categorical(x) -> int:
    if x > 0:
        return 1
    else:
        return 0


def multi_categorical_pct(x: np.array, category=5) ->np.array:
    '''
    Convert given array of percentage change to categorical matrix
    :param x: array of percentage change
    :param category: number of categories
    :return: array of one-hot categorical
    '''
    quantiles = [np.percentile(x, int(100/category*i))for i in range(category+1)]
    categories = []
    for item in x:
        if item == quantiles[-1]:
            categories.append(4)
            continue
        for seq in range(1, category+1):
            if quantiles[seq] > item >= quantiles[seq-1]:
                categories.append(seq-1)
    target_oh = np.eye(category)[categories, :]
    return target_oh


def multi_categorical_value(features:np.array, x: np.array, category=5) ->np.array:
    '''
    Convert given array of percentage change to categorical matrix
    :param x: array of percentage change
    :param category: number of categories
    :return: array of one-hot categorical
    '''
    categories = []
    for item in x:
        if item <= -0.01:
            categories.append(0)
        elif -0.01 < item <= -0.003:
            categories.append(1)
        elif -0.003 <= item <= 0.003:
            categories.append(2)
        elif 0.003 < item <= 0.01:
            categories.append(3)
        elif 0.01 < item:
            categories.append(4)
    target_oh = np.eye(category)[categories, :]
    return target_oh


if __name__ == '__main__':
    raw_data = RawData.get_raw_data(index_name=r'sh000002')
    tech_data = CalculateFeatures.get_all_technical_index(raw_data)
    # cat = multi_categorical_pct(np.random.randn(30), 5)
    cat = multi_categorical_value(np.random.randn(30), 5)