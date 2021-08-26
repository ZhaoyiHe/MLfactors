import numpy as np
import pandas as pd


def standardize(series):
    """
    Standardize data
    :param series:
    :return:
    """
    mean = series.mean()
    std = series.std()
    return ((series - mean) / std)


def filter_extreme_values(series, n=3):
    median = series.quantile(0.5)
    md = ((series - median).abs()).quantile(0.5)
    max_limit = median + n * md
    min_limit = median - n * md
    return (np.clip(series, min_limit, max_limit))


class DataHandler(object):
    def __init__(self, stockdata, factordata):
        self.factor_data = factordata
        self.stock_data = stockdata
        self._aggregate()
        self._split_X_y()

    def _aggregate(self):
        """
        Calculate returns of each period and combine the X and y dataset.
        """
        self.stock_data.date = pd.to_datetime(self.stock_data.date)
        self.factor_data.date = pd.to_datetime(self.factor_data.date)

        self.stock_returns = self.stock_data.reset_index(drop=True).set_index(["date", "order_book_id"]).groupby(
            level=1).close.pct_change().reset_index(name="returns")  # no useful index
        self.next_stock_returns = self.stock_data.reset_index(drop=True).set_index(["date", "order_book_id"]).groupby(
            level=1).close.apply(lambda x: x.shift(-1) / x - 1).reset_index(name="returns")
        self.next_labels = self.next_stock_returns.set_index(["date", "order_book_id"]).groupby(
            level=0).returns.apply(lambda y: y.apply(lambda x: 1 if x >= y.quantile(0.8) else 0)).reset_index(name="label")
        # self.next_labels = self.next_stock_returns.set_index(["date", "order_book_id"]).groupby(
        #    level=0).returns.apply(lambda y: y.apply(lambda x:  -1 if x <= y.quantile(0.1) else (1 if x >= y.quantile(0.9) else 0))).reset_index(name="label")
        # index settled down, standardize and filt extreme values
        self.factor_data = self.factor_data.reset_index(drop=True).set_index(["date", "order_book_id"])
        self.factor_data = self.factor_data.apply(
            lambda y: y.groupby(level=0).apply(lambda x: standardize(filter_extreme_values(x))), axis=0)
        self.factor_data = self.factor_data.apply(
            lambda y: y.groupby(level=0).apply(lambda x: x.fillna(value=x.quantile(0.5))), axis=0).fillna(0)
        # index settled down
        self.X_y = pd.merge(self.factor_data, self.next_labels, how="inner", on=["order_book_id", "date"])
        self.all_dates = self.X_y.date.unique()
        self.stocks = self.X_y.order_book_id.unique().tolist()

    def _split_X_y(self):
        self.factor_data = self.X_y.iloc[:, :-1]
        self.next_labels = self.X_y.iloc[:, -1]
