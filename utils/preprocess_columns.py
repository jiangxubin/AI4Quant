import pandas as pd


def columns(path=r''):
    df = pd.read_csv(path, header=0, sep='|', index_col=None)
    df = df.applymap(lambda x: x.replace(' ', '') if type(x) == str else x).drop(index=0)
    cols = [column.replace(' ', '') for column in df.columns]
    df.columns = cols
    df = df.dropna(axis=1, how='all')
    df = df.drop(columns=['Label'], axis=1)
    return df


if __name__ == '__main__':
    col = columns(r'G:\DX\HugeData\Index\columns.csv')
    col.to_csv(r'G:\DX\HugeData\Index\nature_columns.csv')