import pandas as pd
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from data_handler import DataHandler

# Input data sets
print('Processing data...')
stock_data = pd.read_csv("stock_data.csv")
raw_factor_data = pd.read_csv("factor_data.csv")
data = DataHandler(stockdata=stock_data, factordata=raw_factor_data)
all_dates = data.all_dates
next_stock_returns = data.next_stock_returns
rf_factor_data = pd.read_csv("random_forest_factor.csv")
rf_factor_data = rf_factor_data.rename(columns={"Unnamed: 0": "date"}).set_index('date').stack().reset_index().rename(
    columns={"level_1": "order_book_id", 0: "factor_value"})
rf_factor_data.date = pd.to_datetime(rf_factor_data.date)


class Evaluation(object):
    """
    Calculate metrics of a factor.
    """

    def __init__(self, next_stock_return, factor_data):
        self.next_stock_return = next_stock_return
        self.factor_data = factor_data
        self._match_data()
        self.calculate_IC()
        self.OLS_reg()
        self.combine_metrics()

    def _match_data(self):
        print("Matching data...")

        self.factor_vs_return = pd.merge(self.factor_data, self.next_stock_return, how="inner",
                                         on=["order_book_id", "date"]).dropna(axis=0).set_index(
            ["date", "order_book_id"])
        # self.next_stock_return = self.factor_vs_return.iloc[:, -1]
        # self.factor_data = self.factor_vs_return.iloc[:, :-1]

    def calculate_IC(self):
        print("Calculating IC...")
        IC_df = pd.DataFrame(
            data=self.factor_vs_return.groupby(level="date").apply(
                lambda x: stats.spearmanr(x.iloc[:, -1], x.iloc[:, -2])[0])
            , index=self.factor_vs_return.index.get_level_values("date").unique())
        self.IC_mean = IC_df.mean()
        self.IR = IC_df.mean() / IC_df.std()

    def OLS_reg(self):
        print("Calculating factor return...")
        factor_return_df = pd.DataFrame(
            data=self.factor_vs_return.groupby(level="date").apply(
                lambda x: sm.OLS(x.iloc[:, -1], x.iloc[:, -2]).fit().params[0])
            , index=self.factor_vs_return.index.get_level_values("date").unique())
        self.factor_return = (1 + factor_return_df).cumprod().iloc[-1] - 1

    def combine_metrics(self):
        metrics_info = pd.DataFrame([self.IC_mean, self.IR, self.factor_return]).T
        self.metrics_info = metrics_info.reset_index(drop=True).rename(columns={0: "IC_mean",
                                                                                1: "IR", 2: "Factor Return"})


if __name__ == "__main__":
    eva = Evaluation(next_stock_returns, rf_factor_data)
    print(eva.metrics_info)
