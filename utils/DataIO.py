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


if __name__ == "__main__":
    print(None)
