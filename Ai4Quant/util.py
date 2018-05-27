import numpy as np
import tushare as ts 
import pandas as pd
import os
import re
import collections
import chardet
from multiprocessing import Pool
import logging
logging.basicConfig(filename='logger.log', level=logging.INFO)


def get_stock_data(stock: str, start_date: str, end_date: str):
    """
    获取指定股票指定时间段的历史交易数据
    :param stock: 股票代码
    :param start_date: 开始日期
    :param end_date: 结束日期
    :return: 历史数据DataFrame
    """
    print(stock)
    # logging.info(print("Shape of stock df is {}".format(df.shape)))
    if os.path.exists(r'E:\DX\Ai4Quant\Data\{}.csv'.format(stock)):
        print("File already exists")
        return None
    else:
        try:
            df = ts.get_k_data(stock, start=start_date, end=end_date)
            stock_code = df.code.iloc[0]
            df.index = df.iloc[1:, 0]
            df = df.drop(labels=['code', 'date'], axis=1)
            df.columns = pd.MultiIndex.from_product([[stock_code], df.columns])
            df.to_csv(r'E:\DX\Ai4Quant\Data\{}.csv'.format(stock))
            return df
        except AttributeError:
            print(stock)
            return None


def get_universe()->pd.DataFrame:
    """
    Get universe of today hs300s
    :return: list
    """
    today_universe = ts.get_hs300s()
    top_universe = today_universe[today_universe['weight'] > today_universe['weight'].quantile(0.8)]
    return top_universe


def get_universe_data(universe: list, start_date: str, end_date: str)->list:
    """
    多进程获取所有股票历史数据
    :param universe: 股票池列表
    :param start_date: 开始日期
    :param end_date: 结束日期
    :return: 历史数据DataFrame
    """
    try:
        pool = Pool(6)
        params = zip(universe, [start_date]*len(universe), [end_date]*len(universe))
        result = pool.starmap(get_stock_data, params)
        result = list(filter(None, result))
        result_df = pd.concat(result, axis=1)
        # result_df.index = result_df.iloc[:, 0]
        # result_df = result_df.drop(labels='date', axis=1, level=1)
        return result
    except ValueError:
        print("All stock data has been downloaded, load local data")
        local_root_path = r'E:\DX\Ai4Quant\Data'
        result = list()
        for root, dirs, files in os.walk(local_root_path):
            for file in files:
                file_path = os.path.join(local_root_path, file)
                df = pd.read_csv(file_path)
                result.append(df)
        return result


def feature_label_split(raw_data: list)->tuple:
    """
    Split feature and label dataframe,
    :return:
    """
    # raw_data = self.__get_raw_data()
    X = np.array([item.iloc[0:10, :].values for item in raw_data])
    y = np.array([item.iloc[11, (slice(None), 'close')] for item in raw_data])
    return X, y


def preprocess_raw_data(raw_data:pd.DataFrame)->pd.DataFrame:
    """
    Preprocess raw data to factor
    :param raw_data: DataFrame of exchange data of stock from choosen universe
    :return: processed factor
    """


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
    universe = get_universe()
    universe_code = list(universe.code)
    print(universe_code)
    res = get_universe_data(universe_code, start_date='2018-01-03', end_date='2018-05-26')
    # re = get_stock_data('600000', start_date='2018-01-03', end_date='2018-05-26')
    print(None)