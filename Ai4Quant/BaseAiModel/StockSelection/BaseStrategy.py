"""
Base strategy used for selecting stocks out of certain stock universe
"""


class BaseStrategy:
    def __init__(self):
        self.model = None

    def get_features_target(self, *args, **kwargs):
        """
        Get features matrix and target matrix
        :return: X, y
        """
        return X, y

    def __build_model(self, *args, **kwargs):
        '''
        Build specified strategy based on different ML model
        :return:
        '''

    def fit(self, *args, **kwargs):
        '''
        Fit model the train features and train targets
        :return:
        '''

    def evaluation(self, *args, **kwargs):
        '''
        Evaluate the model with the validation features and validation target
        :return: Evaluation results
        '''

    def predict(self, *args, **kwargs):
        '''
        Provided for trading strategy to predict data
        :param args:
        :param kwargs:
        :return:
        '''
