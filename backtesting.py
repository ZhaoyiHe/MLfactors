import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from data_handler import DataHandler

# Import and preprocess the data
print('Processing data...')
stock_data = pd.read_csv("stock_data.csv")
raw_factor_data = pd.read_csv("factor_data.csv")
data = DataHandler(stockdata=stock_data, factordata=raw_factor_data)
all_dates = data.all_dates
next_stock_returns = data.next_stock_returns
rf_factor_data = pd.read_csv("random_forest_factor.csv")
rf_factor_data.date = pd.to_datetime(rf_factor_data.date)


class QuantilePortfolioConstruction(object):
    """
    Build portfolio from the top quantile.
    """
    def __init__(self, factor_data, next_stock_return,group_num):
        self.factor_data = factor_data
        self.next_stock_return = next_stock_return
        self.group_num = group_num
        self.groups = self.group_by_quantile()
        self.portfolios = self.construct_portfolio()
        self.net_value = self.calculate_net_value()

    def group_by_quantile(self):
        group_labels = list(range(1, self.group_num + 1))
        self.factor_data = self.factor_data.reset_index().set_index(["date", "order_book_id"])
        groups = ((self.factor_data.iloc[:,-1]).groupby(level=0, group_keys=False).apply(pd.qcut, q=self.group_num, labels=group_labels))
        return groups

    def construct_portfolio(self, selected_group_number=1):
        print("Constructing portfolios...")
        selected_group = self.groups[self.groups == selected_group_number]
        portfolios = selected_group.groupby(level = "date").apply(
            lambda x:  pd.Series(1 / len(x),index=x.index.get_level_values("order_book_id"))
        )

        # portfolios = OrderedDict()
        # for date, x in selected_group.groupby(level="date"):
        #     print("Generating portfolio for "+format(date)+"...")
        #     portfolios[date] = pd.Series(1 / len(x), index=x.index.get_level_values("order_book_id"))
        # portfolios = pd.DataFrame.from_dict(portfolios,orient="index").reset_index()
        return portfolios

    def calculate_net_value(self):
        print("Calculating net value ...")

        weights_return = pd.merge(self.portfolios.reset_index(), self.next_stock_return.reset_index(), how="right",
                          on=["order_book_id", "date"]).fillna(0)
        weights_return = weights_return.drop(columns="index").set_index(["date", "order_book_id"])
        total_return = weights_return.groupby(level=0).apply(lambda x: sum(np.multiply(x.iloc[:,-2], x.iloc[:,-1])) )
        cum_return = (1+total_return.shift(1)).cumprod()
        return cum_return


if __name__ == "__main__":
    # Build portfolio and calculate the netvalue, save the net value curve
    bact = QuantilePortfolioConstruction(rf_factor_data, next_stock_returns,10)
    bact.net_value.plot(figsize= (12,6))
    bact.net_value.to_csv('net_value.jpg')
    plt.savefig("net_value.jpg")
