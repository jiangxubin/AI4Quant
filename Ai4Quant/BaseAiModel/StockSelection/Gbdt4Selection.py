from Ai4Quant.BaseAiModel.StockSelection import BaseStrategy
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor


class Gbdt4Selection(BaseStrategy.BaseStrategy):

    def get_features_target(self, *args, **kwargs):
        """
        Override the get_eatures_target method
        :param args:
        :param kwargs:
        :return:
        """

    def __build_model(self, *args, **kwargs):
        """
        Build the Gbdt model
        :param args:
        :param kwargs:
        :return:
        """
        clf =
