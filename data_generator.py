import pandas as pd
import rqdatac as rq
from rqdatac import *
rq.init() # initialize rqdata
if __name__ == "__main__":
    # Get the last trading day of each month
    all_days = get_trading_dates('2020-03-31', '2021-06-30', market='cn')
    # dateRange = []
    #tempYear = None
    #dictYears = pd.DatetimeIndex(all_days).groupby(pd.DatetimeIndex(all_days).year)
    #for yr in dictYears.keys():
    #    tempYear = pd.DatetimeIndex(dictYears[yr]).groupby(pd.DatetimeIndex(dictYears[yr]).month)
    #    for m in tempYear.keys():
    #        dateRange.append(max(tempYear[m])) # Select the last available date
    # Get stock data on the selected trading day
    dateRange = all_days
    stocks = index_components("000906.XSHG")[:100]
    data = get_price(stocks,'2020-03-31', '2021-06-30',expect_df=True).reset_index()
    data['date'] = pd.to_datetime(data['date'])
    stock_data = data.set_index('date').loc[dateRange]
    stock_data.to_csv('stock_data.csv')

    factors = pd.read_csv('factor document.csv')['字段']
    factor_data = get_factor(stocks, factors.tolist(), '2020-03-31', '2021-06-30').reset_index()
    factor_data['date'] = pd.to_datetime(factor_data['date'])
    factor_data = factor_data.set_index('date')
    factor_data = factor_data.loc[dateRange]
    factor_data.to_csv('factor_data.csv')