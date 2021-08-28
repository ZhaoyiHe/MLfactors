import abc

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV,TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from data_handler import DataHandler


class Factor:
    def generate_factor(self):
        pass


class MLFactor(Factor, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def train_model(self):
        pass

    @abc.abstractmethod
    def validate_model(self):
        pass

    @abc.abstractmethod
    def test_model(self):
        pass

    @abc.abstractmethod
    def generate_factor(self):
        pass


class RandomForestFactor(MLFactor):
    def __init__(self, history_data):
        self.X_y = history_data

        self._split_dataset()

        # self.test_model()
        self.train_model()
        self.test_model()
        self.generate_factor()

    def _split_dataset(self, split_ratio=0.85):
        dates = self.X_y.reset_index().date.unique()
        date_index=int(np.ceil(split_ratio*len(dates)))
        split_date = dates[date_index]

        self.X_train = self.X_y[dates[0]:split_date].iloc[:,:-1].reset_index().set_index(['date','order_book_id'])
        self.y_train = self.X_y[dates[0]:split_date].iloc[:,-1]

        self.X_test = self.X_y[split_date:dates[-1]].iloc[:, :-1].reset_index().set_index(['date','order_book_id'])
        self.y_test = self.X_y[split_date:dates[-1]].iloc[:, -1]

        self.all_X = self.X_y.reset_index().set_index(['date','order_book_id'])[dates[0]:dates[-2]].iloc[:, :-1]
        self.all_y = self.X_y.reset_index().set_index(['date', 'order_book_id'])[dates[0]:dates[-2]].iloc[:, -1]
        self.last_X = self.X_y.reset_index().set_index(['date','order_book_id'])[dates[-1]:dates[-1]].iloc[:, :-1]

    def train_model(self):
        print("Training_model... ")
        param_grid = {
            'max_depth': [8,9,10],
            'n_estimators': [8,10,11,13,15],

            'min_samples_split': [2,3]
        }
        tscv = TimeSeriesSplit(n_splits=4)
        rf_cv = GridSearchCV(estimator=RandomForestClassifier(random_state=1002,max_features="sqrt"), param_grid=param_grid,
                             scoring='roc_auc', cv=tscv,n_jobs=-1)
        self.trained_model = rf_cv.fit(self.X_train, self.y_train)

    def validate_model(self):
        pass

    def test_model(self):
        print("Testing model...")
        test_est = self.trained_model.predict(self.X_test)
        print("Accuracy:")
        print(metrics.classification_report(self.y_test, test_est))
        print("AUC:")
        fpr, tpr, th = metrics.roc_curve(self.y_test, test_est)
        print("AUC = %.4f" % metrics.auc(fpr, tpr))

        print(self.trained_model.best_params_)

    def generate_factor(self):
        classifier = RandomForestClassifier(random_state=1002,max_features='sqrt',max_depth=self.trained_model.best_params_['max_depth'],
                                             n_estimators=self.trained_model.best_params_['n_estimators'],
                                             min_samples_split=self.trained_model.best_params_['min_samples_split'])
        self.final_model = classifier.fit(self.all_X,self.all_y)
        self.rf_factor = pd.DataFrame(data=self.final_model.predict_proba(self.last_X))[1]


if __name__ == "__main__":
    # Input data sets
    print('Processing data...')
    stock_data = pd.read_csv("stock_data.csv")
    factor_data = pd.read_csv("factor_data.csv")
    data = DataHandler(stockdata=stock_data, factordata=factor_data)
    data_X = data.factor_data
    data_y = data.next_labels
    data_X_y = data.X_y.set_index("date")  # index setted

    all_dates = data.all_dates
    stock_returns = data.stock_returns

    # For each trading day, train model and predict the probability to get factor data
    burn_in_period = 30
    trading_dates = all_dates[burn_in_period:]
    factor_dict = dict()
    # model_dict = {}
    for date in trading_dates:
        print('Training model for {}'.format(date))
        trade_day_index = np.where(all_dates == date)[0]
        history_start = all_dates[trade_day_index - burn_in_period][0]
        history_end = all_dates[trade_day_index-1][0]
        history_data = data_X_y[history_start:history_end]

        # data_for_decision = data_X_y[date:date]
        # print(history_data)
        RF_Trainer = RandomForestFactor(history_data = history_data)
        factor = RF_Trainer.rf_factor
        # data_for_decision = data_X[date]
        factor_dict[date]=factor
    factor = pd.DataFrame.from_dict(data=factor_dict,orient="index")
    factor.columns = data.stocks
    factor.to_csv('random_forest_factor.csv')
