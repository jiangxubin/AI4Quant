"""
Common auxiliary functions for data mining, feature engineering, file IO .etc
"""
import tushare as ts
import pandas as pd
import os
from multiprocessing import Pool
from utils import DataIO
import logging
logging.basicConfig(filename='logger.log', level=logging.INFO)

project_path = r'E:\\DX'


class RawData:
    @staticmethod
    def get_raw_data(path=r'E:\DX\HugeData\Index\test.csv', columns_path=r'E:\DX\HugeData\Index\columns.csv', index='sh000001'):
        """
        Load index data and produce Input data
        :param path:Path of csv file
        :return:list of input data
        """
        encoding = DataIO.DataIO.detect_encode_style(path)
        # print(encoding)
        df = pd.read_csv(path, sep=r',', encoding=encoding)
        col = pd.read_csv(columns_path, sep=r',', index_col=[0], header=[0])
        df.columns = pd.MultiIndex.from_arrays((col['Feature'], col['Comment']))
        sh01 = df.iloc[:6593, 1:-1]
        return sh01

    @staticmethod
    def get_stock_data(stock: str, start_date: str, end_date: str):
        """
        获取指定股票指定时间段的历史交易数据
        :param stock: 股票代码
        :param start_date: 开始日期
        :param end_date: 结束日期
        :return: 历史数据DataFrame
        """
        if os.path.exists(os.path.join(project_path, r'Data\{}.csv'.format(stock))):
            print("{} File already exists".format(stock))
            return None
        else:
            try:
                print("Downloading {}".format(stock))
                df = ts.get_k_data(stock, start=start_date, end=end_date)
                ret = df['close'].pct_change()
                df['ret'] = ret
                # print(df)
                df = df.drop(1, axis=0)
                stock_code = df.code.values[0]
                df.index = df.iloc[:, 0]
                del df.index.name
                df = df.drop(labels=['code', 'date'], axis=1)
                df.columns = pd.MultiIndex.from_product([[stock_code], df.columns], names=['code', 'data'])
                df.to_csv(os.path.join(project_path, r'Data\{}.csv'.format(stock)))
                return df
            except AttributeError:
                print("{} is down during this period of time".format(stock))
                return None
            except KeyError:
                print("{}is strange at its 0 index".format(stock))
                return None

    @staticmethod
    def get_universe()->pd.DataFrame:
        """
        Get universe of today hs300s
        :return: list
        """
        today_universe = ts.get_hs300s()
        # top_universe = today_universe[today_universe['weight'] > today_universe['weight'].quantile(0.6)]
        return today_universe

    @staticmethod
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
            result = pool.starmap(StockRawData.get_stock_data, params)
            result = list(filter(None, result))
            result_df = pd.concat(result, axis=1)
            return result
        except ValueError:
            print("All stock data has been downloaded, load local data")
            local_root_path = os.path.join(project_path, r'Data')
            result = list()
            for root, dirs, files in os.walk(local_root_path):
                for file in files:
                    file_path = os.path.join(local_root_path, file)
                    # print(file_path)
                    df = pd.read_csv(file_path, index_col=0, header=[0, 1])
                    result.append(df)
            return result


if __name__ == "__main__":
    sh01 = RawData.get_raw_data()