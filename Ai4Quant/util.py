import talib
import numpy as np
import tushare as ts 
import pandas as pd
import os
import re
import chardet
from multiprocessing import Pool
import logging
logging.basicConfig(filename='logger.log', level=logging.INFO)


def get_stock_data(stock: str, start_date: str, end_date: str)->pd.DataFrame:
    """
    获取指定股票指定时间段的历史交易数据
    :param stock: 股票代码
    :param start_date: 开始日期
    :param end_date: 结束日期
    :return: 历史数据DataFrame
    """
    logging.info(print("Downloading data of {}".format(stock)))
    df = ts.get_k_data(stock, start=start_date, end=end_date)
    logging.info(print("Shape of stock df is {}".format(df.shape)))
    stock_code = df.code.iloc[0]
    df = df.drop(columns=['code'])
    df.columns = pd.MultiIndex.from_product([[stock_code], df.columns])
    return df


def get_universe_data(universe: list, start_date: str, end_date: str)->pd.DataFrame:
    """
    多进程获取所有股票历史数据
    :param universe: 股票池列表
    :param start_date: 开始日期
    :param end_date: 结束日期
    :return: 历史数据DataFrame
    """
    pool = Pool(4)
    params = zip(universe, [start_date]*len(universe), [end_date]*len(universe))
    result = pool.starmap(get_stock_data, params)
    result_df = pd.concat(result, axis=0)
    return result_df


def detect_encode_style(file_path):
    with open(file_path, 'rb') as f:
        lines = f.readlines()
        res = chardet.detect(lines[0])
        encoding = res['encoding']
    return encoding


def get_descendant_file_path(parent_path):
    """
    Load descendant file of certain directory
    """
    csv_relative_path = []
    for root, dirs, files in os.walk(parent_path):
        for file in files:
            words = file.split(r'.')
            if words[-1] == 'csv':
                file_path = os.path.join(parent_path, file)
                csv_relative_path.append(file_path)
    return csv_relative_path


def load_data(parent_path):
    csv_file_path = get_descendant_file_path(parent_path)
    df_all = []
    for path in csv_file_path:
        df = pd.read_csv(path, sep=r'\t', encoding='UTF-16', skipinitialspace=True, engine='python')
        df_all.append(df)
    res_df = pd.concat(df_all, axis=0)
    pd.to_pickle(res_df, path='stock_daily_price.pkl')
    return res_df


if __name__ == "__main__":
    res_df = get_stock_data('600000', '2018-01-04', '2018-05-24')
    print(None)