import xgboost as xgb 
import talib
import numpy as np 


class xgb4Selection:
    def __init__(self):
        self.num_boost_round = 100
        self.model = None
        self.universe = None
        self.factor_pool = None
        self.ret_label = None
        self.params = {
            'booster': 'gbtree',
            'objective': 'multi:softmax',  # 多分类的问题
            'num_class': 2,               # 类别数，与 multisoftmax 并用
            'gamma': 0.1,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
            'max_depth': 12,               # 构建树的深度，越大越容易过拟合
            'lambda': 2,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
            'subsample': 0.7,              # 随机采样训练样本
            'colsample_bytree': 0.7,       # 生成树时进行的列采样
            'min_child_weight': 3,
            'silent': 1,                   # 设置成1则没有运行信息输出，最好是设置为0.
            'eta': 0.007,                  # 如同学习率
            'seed': 1000,
            'nthread': 4,                  # cpu 线程数
        }

    def __get_data(self, universe:list, factor_pool:list, ret_label:str, start_date, end_date)->tuple:
        """
        Get specified data from DataBase given the stock universe, factor pool and label of ret
        """
        X = None
        y = None
        return (X, y)

    def __build_model(self, X: np.array, y: np.array):
        """
        Build xgboost model by invoking feature matrix:factors, label matrix:y
        """
        dtrain = xgb.DMatrix(X, label=y)
        watch_list = [(dtrain,'train'), (dval, 'val')]
        bst = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=self.num_boost_round, evals=watch_list)
        bst.save_model(r"E:\DX\Ai4Quant\Model\xgb4ai.model")
        return bst

    def __predict(self, predict_start_date, predict_end_date)->dict:
        """
        Predict and return results given trained model and stock universe
        """
        X, y = self.__get_data(self.universe, self.factor_pool, self.ret_label, predict_start_date, predict_end_date)
        res_dict = self.model.predict(X)
        return res_dict
        
    def fit(self,train_universe:list, factor_pool:list, ret_label:str,  start_date='2014-01-03', end_date='2018-05-24'):
        """
        Fit xgboost model given these params
        """
        self.universe = train_universe
        self.factor_pool = factor_pool
        self.ret_label = ret_label
        X, y = self.__get_data(train_universe, self.factor_pool, self.ret_label, start_date, end_date)
        self.model = self.__build_model(X, y)

    def predict_select(self, predict_start_date, predict_end_date,percentile=98)->dict:
        """
        Method aims to choose the stock of top uprising possibility out of provided stock universe which
        is calculated by predict method
        """
        res_dict = self.__predict(predict_start_date, predict_end_date)
        res_dict = sorted(res_dict.items(), key=lambda x:x[1], reverse=True)
        top_percentile_stock_prob = list(np.percentile(res_dict.values(), percentile, interpolation='higher'))
        top_percentile_stock_code = [res_dict.keys[res_dict.values().index(prob)] for prob in top_percentile_stock_prob]
        return {k: v for k, v in zip(top_percentile_stock_code, top_percentile_stock_prob)}

    def back_test(self, turnover_frequency):
        """
        Specify a certain time period and turnover_frequency,then calculate the benchmark ret of 
        original matket index and the ret of selected universe of the same period
        """

    def __plot_contrast_fig(self):
        return None


if __name__ == "__main__":
    xgb_strategy = xgb4Selection()
    xgb_strategy.fit([], [], ret_label='Month', start_date='2014-01-03', end_date='2018-05-24')
    
