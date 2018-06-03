import pandas as pd
import os


def raw_to_pandas(path)->pd.DataFrame:
    """
    Extract columns which will be addded to Huge data from raw csv file
    :param path: Path of raw data
    :return: Dataframe of columns
    """
    df = pd.read_csv(path, sep=r'|', header=[-1])
    df = df.drop([0, 3, 5], axis=1)
    df.columns = ['Feature', 'Dtype', 'Comment']
    parent_path = os.path.abspath(path).split(os.sep)[:-1]
    parent_path = os.sep.join(parent_path)
    df.to_csv(os.path.join(parent_path, 'nature_columns.csv'))
    return df


if __name__ == '__main__':
    # test_df = raw_to_pandas(r'G:\AI4Quant\HugeData\Index\columns.csv')
    test_df = raw_to_pandas(r'G:\AI4Quant\HugeData\Stock\columns.csv')