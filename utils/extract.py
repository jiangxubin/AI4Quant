import zipfile
import rarfile
import pandas as pd
rarfile.UNRAR_TOOL = r"C:\Program Files\winrar\UnRAR.exe"
import patoolib
from pyunpack import Archive
import os
import re
# https://rarfile.readthedocs.io/en/latest/api.html
# https://tom2fanxing.github.io/2018/05/15/rarfile-error/
# https://blog.csdn.net/big_talent/article/details/52367184

def extract_zip(zip_path=r'G:\AI4Quant\HugeData\Stock'):
    zip_files = os.listdir(zip_path)
    all_zip = []
    for file in zip_files:
        print(file)
        file_type = file.split(r'.')[-1]
        if file_type == 'zip':
            all_zip.append(file)

    for file in all_zip:
        with zipfile.ZipFile(os.path.join(zip_path, file), 'r') as f:
            f.extractall(r'G:\AI4Quant\HugeData\Stock\extracted_factors')


def extract_rar(rar_path=r'G:\Novels\zipped', to_path=r'G:\Novels\extracted'):
    files_path = os.listdir(rar_path)
    print(files_path)
    for file_name in files_path:
        file_path = os.path.join(rar_path, file_name)
        print(file_path)
        # Archive(file_path).extractall(os.path.join(to_path, file_name.split(r'.')[-2]), auto_create_dir=True)
        # patoolib.extract_archive(file_path, outdir=os.path.join(to_path, file_name.split(r'.')[-2]))
        with rarfile.RarFile(file_path) as rf:
            rf.extractall(os.path.join(to_path, file_name.split(r'.')[-2]))


def file_tree(path=r'G:\Novels\extracted'):
    index = 1
    all_book = []
    for a, b, c in os.walk(path):
        print("第{}层".format(index))
        index += 1
        print(a)
        for path in b:
            print(dir)
        if len(c) == 0:
            continue
        else:
            for file in c:
                if file.split(r'.')[-1]== 'pdf':
                    all_book.append(file.split(r'.')[-2])
    df = pd.DataFrame({'BOOK': all_book})
    return df


if __name__ == '__main__':
    # extract_rar(rar_path=r'G:\Novels\zipped', to_path=r'G:\Novels\extracted')
    book_df = file_tree(path=r'G:\Novels\extracted')