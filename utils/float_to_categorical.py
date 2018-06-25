'''
Convert a float to categorical
'''
# RF:https://docs.scipy.org/doc/numpy/reference/generated/numpy.eye.html
import numpy as np
import pandas as pd


def binary_categorical(x) -> int:
    if x > 0:
        return 1
    else:
        return 0


def multi_categorical(x: np.array, category=5) ->np.array:
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


if __name__ == '__main__':
    cat = multi_categorical(np.random.randn(30), 5)