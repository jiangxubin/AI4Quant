import talib
import pandas as pd
import numpy as np
from utils.RawData import RawData


class CalculateFeatures:
    @staticmethod
    def get_all_technical_index(raw_data: pd.DataFrame)->pd.DataFrame:
        """
        Calculate MACD, KDJ.etc and attach those index to the raw_data
        :param raw_data: sh000001 history index
        :return: DataFrame of Technical index attached to the raw_dat
        """
        close = raw_data.iloc[:, 1]
        dif, dea, macd = talib.MACD(close, fastperiod=6, slowperiod=12, signalperiod=9)
        rsi = talib.RSI(close, timeperiod=14)
        ema_5 = talib.EMA(close, timeperiod=5)
        ema_10 = talib.EMA(close, timeperiod=10)
        roc = talib.ROC(close, timeperiod=10)
        upband, midband, lowband = talib.BBANDS(close, timeperiod=5, nbdevup=3, nbdevdn=3)
        tech_index = pd.DataFrame({'DIF': dif, 'DEA': dea, "MACD": macd, "RSI": rsi, "EMA_5": ema_5,"EMA_10":ema_10, "ROC":roc, "UPBAND":upband,  "MIDBAND":midband, "LOWBAND":lowband})
        tech_indexed_data = pd.concat([raw_data, tech_index], axis=1)
        return tech_indexed_data


if __name__ == "__main__":
    raw_data = RawData.get_raw_data('sh000002', r'E:\DX\HugeData\Index\test.csv', r'E:\DX\HugeData\Index\nature_columns.csv')
    t_i = CalculateFeatures.get_all_technical_index(raw_data)