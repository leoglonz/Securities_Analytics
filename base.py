"""
Contains the base-level analytical functions and other data prep algorithms used frequently throughout this project.

IN PROGRESS
Last Revised: 15 Mar 2023
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import Dense, LSTM

from pandas_datareader import data as pdr
import datetime as dt

def datImport(ticker, start, end=dt.datetime.today(), verbose=False):
    # importing daily equity or market data.
    data = pdr.get_data_yahoo(ticker, start, end).rename(columns= {'Adj Close': 'AdjClose'})

    if verbose:
        print(data.shape[0], "days loaded with attributes: \n", data.keys())

        fig, ax = plt.subplots(1,1, dpi=300, figsize=(16,8),
            constrained_layout=False)

        ax.plot(data.index, data.AdjClose)

        ax.set_title('Adjusted Closing Prices for %s (USD), %s-%s' 
                    %(ticker, start.year, end.year))
        ax.set_xlabel('Date', fontsize=18)
        ax.set_ylabel('Adjusted Closing Price (USD)', fontsize=18)

        # Set major and minor date tick locators
        maj_loc = mdates.MonthLocator(bymonth=np.arange(1,12,6))
        ax.xaxis.set_major_locator(maj_loc)
        min_loc = mdates.MonthLocator()
        ax.xaxis.set_minor_locator(min_loc)

        # Set major date tick formatter
        zfmts = ['', '%b\n%Y', '%b', '%b-%d', '%H:%M', '%H:%M']
        maj_fmt = mdates.ConciseDateFormatter(maj_loc, zero_formats=zfmts, 
                                            show_offset=False)
        ax.xaxis.set_major_formatter(maj_fmt)

        ax.figure.autofmt_xdate(rotation=0, ha='center')
        ax.set_xlim(data.index.min(), data.index.max());

    return data


def series_to_supervised(data, n_in=5, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
    data: Sequence of observations as a list or NumPy array.
    n_in: Number of lag observations as input (X).
    n_out: Number of observations as output (y).
    dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
    Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    if i == 0:
        names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    else:
        names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
        
    return agg


def beta(df, market=None):
    # Calculating betas using prices from every business day)
    # If the market values are not passed,
    # I'll assume they are located in a column
    # named 'Market'.  If not, this will fail.
    if market is None:
        market = df['MarketClose']
        df = df.drop('MarketClose', axis=1)
    X = market.values.reshape(-1, 1)
    X = np.concatenate([np.ones_like(X), X], axis=1)
    b = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(df.values)
    
    return float(b[1])


def roll(df, window=252):
    # Takes 'w'-sized slices from dataframe, incrementing 1 entry at a time.
    for i in range(df.shape[0] - w + 1):
        yield pd.DataFrame(df.values[i:i+window, :], df.index[i:i+window],
                           df.columns)
        

def addBeta(stock, market, window=252, verbose=False):
    # Calc beta across given df of closing prices.
    # Window default is stet to calculate yearly beta.
    betas = np.array([])
    data = pd.concat([stock.AdjClose, market.MarketClose], axis=1)

    for  i, sdf in enumerate(roll(data.pct_change().dropna(), window )):
        betas = np.append(betas, beta(sdf))

    full_data = data.drop(index=data.index[:window], axis=0, inplace=False)
    full_data['Beta'] = betas.tolist()

    return full_data


