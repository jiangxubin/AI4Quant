import pandas as pd
import numpy as np


def positive_comparision(y, y_pred)->tuple:
    """
    Count the ratio of predicted up-rising/down with real uprising/down
    :param y: real index
    :param y_pred: predicted index
    :return: ratio
    """
    y = pd.Series(y)
    y_pred = pd.Series(y_pred)
    delta_real = y - y.shift(1)
    delta_pred = y_pred - y_pred.shift(1)
    up_index = delta_real[delta_real > 0].index
    down_index = delta_real[delta_real < 0].index
    up_corresp = delta_pred[up_index]
    down_corresp = delta_pred[down_index]
    pred_up = up_corresp[up_corresp > 0]
    pred_down = down_corresp[down_corresp < 0]
    up_ratio = len(pred_up) / len(up_index)
    down_ratio = len(pred_down) / len(down_index)
    return up_ratio, down_ratio


def negative_comparision(y, y_pred)->tuple:
    """
    Count the ratio of real up-rising/down with predicted uprising/down
    :param y: real index
    :param y_pred: predicted index
    :return: ratio
    """
    y = pd.Series(y)
    y_pred = pd.Series(y_pred)
    delta_pred = y_pred - y_pred.shift(1)
    up_index = delta_pred[delta_pred > 0].index
    down_index = delta_pred[delta_pred < 0].index
    delta_real = y - y.shift(1)
    up_contract = delta_real[up_index]
    down_contract = delta_real[down_index]
    real_up = up_contract[up_contract > 0]
    real_down = down_contract[down_contract < 0]
    up_ratio = len(real_up) / len(up_index)
    down_ratio = len(real_down) / len(down_index)
    return up_ratio, down_ratio


if __name__ == "__main__":
    y = np.random.rand(20)
    y_pred = np.random.rand(20)
    ur, dr = positive_comparision(y, y_pred)