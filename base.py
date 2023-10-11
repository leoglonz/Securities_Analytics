"""
Contains the base-level analytical functions and other data prep algorithms used frequently throughout this project.

IN PROGRESS
Last Revised: 04 Oct 2023
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

from pandas_datareader import data as pdr
import yfinance as yfin
import datetime as dt
import warnings

yfin.pdr_override()
pd.options.mode.chained_assignment = None  # default='warn'


def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str("UserWarning: ") + str(msg) + '\n'

warnings.formatwarning = custom_formatwarning


def roll(df, window=252):
    """
    Takes 'w'-sized slices from dataframe, incrementing 1 entry at a time.
    """
    for i in range(df.shape[0] - window + 1):
        yield pd.DataFrame(df.values[i:i+window, :], df.index[i:i+window],
                           df.columns)


def datImport(ticker, start=dt.datetime(2023,1,1), end=dt.datetime.today(), 
              verbose=False):
    """
    For importing window of equity or market data identified by 'ticker' 
    from Yahoo Finance. 
    Verbose will enable a basic plot readout for the imported security.
    Arguments:
    ticker: string, the ticker identifier for the security/market.
    start: dt.datetime, date of first observation for ticker.
    end: dt.datetime, date of last observation for ticker.
    verbose: bool, True enables basic plot of Adj. closing prices for ticker.
    Returns:
    Pandas DataFrame of all data for ticker from Yahoo in the designated window.
    """
    data = pdr.get_data_yahoo(ticker, start, end).rename(columns= {'Adj Close': 'AdjClose'})

    if verbose:
        print(data.shape[0], "days loaded with attributes: \n", data.keys())

        fig, ax = plt.subplots(1,1, dpi=300, figsize=(16,8),
            constrained_layout=False)

        ax.plot(data.index, data.AdjClose)

        ax.set_title("Adjusted Closing Prices for %s (USD), %s-%s" 
                    %(ticker, start.year, end.year))
        ax.set_xlabel('Date', fontsize=18)
        ax.set_ylabel("Adjusted Closing Price (USD)", fontsize=18)

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
        agg: DataFrame, series framed for supervised learning.
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


def data_split(data, lag=60, days=1, train_ratio=0.70,
               validation=False, backtest=False):
    """
    Prepping stock data for neural net; scaling down 
    values and making train-test split.
    data: DataFrame, all stock data.
    lag: int, number of days used for prediction.
    days: int, number of days to predict.
    train_ratio: float, percentage of data for training.
    validation: bool, split data into train/valid/test when True.
    backtest: bool, only performs x-y split when True.
    Returns
        X_train: array, independent training features.
        y_train: array, objective training feature.
        X_test: array, independent test features.
        y_test: array, objective test feature.
        X_valid: array, independent validation features.
        y_valid: array, objective validation feature.
        X: array, independent features.
        y: array, target feature.
    """
    # Selecting 'AdjClose' prices as input and target feature for time series.
    data_adj = data.filter(['AdjClose']).values

    # Scaling data. Ensures quicker convergence to solution.
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data_adj)

    # Splitting input features and target object, X and y.
    supervised_data = series_to_supervised(scaled_data, n_in=lag, n_out=days)
    X = supervised_data.loc[:, supervised_data.columns != 'var1(t)'] 
    y = supervised_data['var1(t)'] # Isolating target object.

    # Selecting converted data for train-test split.
    len_training = int(np.ceil(len(scaled_data) * train_ratio))

    X_train = X.iloc[0:len_training].to_numpy()
    y_train = y.iloc[0:len_training].to_numpy()
    # X_train, y_train = np.array(X_train), np.array(y_train)

    # Making validation/test split.
    if validation:
        len_valid = int((len(X) - len_training)/2)
        len_valid += len_training-60

        # We subtract lag days since they are needed to actually  
        X_valid = X.iloc[len_training-60:len_valid].to_numpy()
        y_valid = data_adj[len_training:len_valid]

        X_test = X.iloc[len_valid-60:].to_numpy()
        y_test = data_adj[len_valid:]

    else:
        X_test = X.iloc[len_training-60:].to_numpy()
        y_test = data_adj[len_training:]

    # Reshaping to obtain 3D reps (currently 2d) to pass into LSTM.
    # LSTM expects d1 # of samples, d2 # of timesteps, and d3 # of features.
    X_train = np.reshape(X_train, (X_train.shape[0],
                                   X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0],
                                 X_test.shape[1], 1))

    if len(X_test) != len(y_test):
        raise Warning('X, y length mismatch.')
    
    if validation:
        X_valid = np.reshape(X_valid, (X_valid.shape[0],
                                       X_valid.shape[1], 1))
        return X_train, y_train, X_valid, y_valid, X_test, y_test, scaler
    
    elif backtest:
        return X, y, scaler

    elif not backtest and not validation:
        return X_train, y_train, X_test, y_test, scaler
    
    else:
        ValueError(
            "Cannot simultaneously perform 'backtest' and 'validation'."
            )
        exit()


def backtest_split(data, lag=60, days=1, train_ratio=0.70,
                   n_splits=5, method=TimeSeriesSplit, verbose=False):
    """
    Splitting data for backtesting using either rolling or
    expanding window.
    data: DataFrame, all stock data.
    lag: int, number of days used for prediction.
    days: int, number of days to predict.
    train_ratio: float, percentage of data for training.
    validation: bool, split data into train/valid/test when True.
    backtest: bool, only performs x-y split when True.
    Returns
        X_train: array, independent training features.
        y_train: array, objective training feature.
        X_test: array, independent test features.
        y_test: array, objective test feature.
        X_valid: array, independent validation features.
        y_valid: array, objective validation feature.
        X: array, independent features.
        y: array, target feature.
    """
    series_split = method(n_splits=n_splits)

    # Separating input and target features in data.
    X, y, scaler = data_split(data, lag=lag, days=days, 
                              train_ratio=train_ratio, backtest=True)

    X = np.asarray(X) # series_split requires np arrays.
    y = np.asarray(y)

    if verbose:
        for fold, (train_index, test_index) in enumerate(series_split.split(X, y)):
            print("Fold: {}".format(fold))
            print("TRAIN indices:", train_index, "\n", "TEST indices:", test_index)
            print("\n")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
    
    return X, y, series_split, scaler


def plot_cv_indices(cv, X, y, ax, n_splits, linewidth=10):
    """
    Plotting folds for walk forward-validation.
    """
    # Generate the training/testing visualizations for each CV split.
    for i, (train, test) in enumerate(cv.split(X=X, y=y, groups=None)):
        # Fill in indices with the training/test groups.
        indices = np.array([np.nan]*len(X))
        indices[train] = 1
        indices[test] = 0

        ax.scatter(
            range(len(indices)),
            [i+1] * len(indices),
            c=indices, marker='_',
            cmap='coolwarm',
            linewidth=linewidth,
            vmin=-.2, vmax=1.2
            )

    # Formatting
    ax.set(
        yticks=np.arange(1,n_splits+1),
        xlabel="Historical Days Included",
        ylabel="CV Iteration",
        ylim=[n_splits+1.2, -.1]
        )
    ax.set_title("Walk-forward Splits for %i CV Iterations"
                 %n_splits, fontsize=15)
    ax.legend([Patch(color='tomato'), Patch(color='royalblue')],
          ['Train', 'Test'])
    
    return ax


###########################
#### Class Definitions ####
###########################

class SlidingSeriesSplit():
    """
    parameters
    ----------
    n_test_folds: int
        number of folds to be used as testing at each iteration.
        by default, 1.
    """
    def __init__(self, n_splits):
        self.n_splits = n_splits
    
    def get_n_splits(self, X, y, groups):
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        if self.n_splits > n_samples:
            raise ValueError(
                "Cannot have number of folds =%i greater"
                " than the number of samples: %i." %(self.n_splits, n_samples))

        margin = 0
        for i in range(self.n_splits):
            # Sets the overlap of each block.
            start = i * int(k_fold_size/3)
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]
