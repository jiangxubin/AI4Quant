"""
Common auxiliary functions for data mining, feature engineering, file IO .etc
"""
import pandas as pd
import os
import chardet
import logging
logging.basicConfig(filename='logger.log', level=logging.INFO)


# project_path = os.path.abspath('DataIO.py').split(os.sep)[:-1]
# project_path = os.sep.join(project_path)
project_path = r'E:\\DX'


class DataIO:
    @staticmethod
    def detect_encode_style(file_path):
        '''
        Detect the file code style
        :param file_path:
        :return:
        '''
        with open(file_path, 'rb') as f:
            lines = f.readlines()
            res = chardet.detect(lines[0])
            encoding = res['encoding']
        return encoding

    @staticmethod
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

    @staticmethod
    def load_data(parent_path):
        csv_file_path = DataIO.get_descendant_file_path(parent_path)
        df_all = []
        for path in csv_file_path:
            df = pd.read_csv(path, sep=r'\t', encoding='UTF-16', skipinitialspace=True, engine='python')
            df_all.append(df)
        res_df = pd.concat(df_all, axis=0)
        pd.to_pickle(res_df, path='stock_daily_price.pkl')
        return res_df

    @staticmethod
    def strip_df(df_object: pd.DataFrame)->pd.DataFrame:
        """
        Remove blank str like \t \n from dataframe
        :param df_object:
        :return: Non-blank dataframe
        """
        df_str = df_object.select_dtypes(include=['object'])
        df_object[df_str.columns] = df_str.applymap(lambda x: x.strip(r' '))
        return df_object

    @staticmethod
    def strip_df_1(df_object: pd.DataFrame)->pd.DataFrame:
        """
        Remove blank str like \t \n from dataframe
        :param df_object:
        :return: Non-blank dataframe
        """
        df_object.applymap(lambda x: x.strip(r'\t\n') if x is str else x)
        return df_object
# 1. https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.select_dtypes.html#pandas.DataFrame.select_dtypes
# 2. https://stackoverflow.com/questions/40950310/strip-trim-all-strings-of-a-dataframe
# 3. https://stackoverflow.com/questions/19798153/difference-between-map-applymap-and-apply-methods-in-pandas

if __name__ == "__main__":
    df_col = pd.read_csv(r'E:\DX\HugeData\Index\nature_columns.csv', encoding=r'GB2312')
    df_o = DataIO.strip_df(df_col)
    df_o_1 = DataIO.strip_df_1(df_col)

