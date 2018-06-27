"""
Base strategy for time selection which must be implemented detailed by its subclass
"""


class BaseStrategy:
    def __init__(self):
        self.model = None

    def get_feature_label(self, *args, **kwargs)->tuple:
        """
        Get X for feature and y for label when everyday has a single feature
        :return: DataFrame of raw data
        """

    def __build_model(self, *args, **kwargs):
        """
        Build the ML/DL model for index prediction or selelction
        :return: model
        """

    def fit(self, *args, **kwargs):
        """
        Train the ML/DL model defined bove
        :param X: np.array of Feature
        :param y: np.array of labels
        :return: None
        """

    def evaluation(self, *args, **kwargs):
        """
        Evaluating the strategy model fitted above to tune hyperameters
        """

    def plot_contract(self, *args, **kwargs):
        """
        Plot contract figurre of predicted index with respect to real index
        :param args:
        :param kwargs:
        :return:
        """

    def predict(self, *args, **kwargs)->float:
        """
        According to passed data of realtime to predict tomorow's index price
        """