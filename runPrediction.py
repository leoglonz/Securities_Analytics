"""
Designed to run a prediction routine for a desired stock from the command 
line. This should only be used after `lstm_optimization.py` has been run, 
or an optimized model (tag 'model_') is already present in the root 
directory. If a model is not already present, run `lstm_optimize.py` first
(see instructions in `README.md`).


IN PROGRESS
Last Revised: 14 Sep 2023
"""

from base import *
from analysis import *

import yfinance as yfin
import datetime as dt
from dateutil.relativedelta import relativedelta
import pickle

yfin.pdr_override()
pd.options.mode.chained_assignment = None  # Default = 'warn'


TICKER = 'AAPL'
START = dt.datetime(2012, 1, 1)
END = dt.datetime(2022, 12, 1)

LAG = 60
DAYS = 1
TRAIN_RATIO = 0.70

UNITS = 128
CLASSES = 25
BATCHSIZE = 1 #128
EPOCHS = 1 #10

BACKTEST = False
N_SPLITS = 5
S_TYPE = SlidingSeriesSplit # TimeSeriesSplit


ticker = TICKER
start = START
end = END
backtest = False

lag = LAG
days = DAYS
train_ratio = TRAIN_RATIO


data = datImport(ticker, start, end)

if backtest:
    # Using backtest validation (sliding or expanding window).
    # Sliding window usually gives better results.

    split = data_split(data, lag, days, train_ratio, backtest=True)
    X, y = split[0], split[1]


# Actually, we might want to open up the backtest validation here because this is tuned specifically to optimization. Perhaps also consider using the backtesting found in "stock_prediction.ipynb"
    agg_rmse = backtest_validation(
        data,
        make_model=create_model,
        n_splits=N_SPLITS,
        batch_size=BATCHSIZE,
        epochs=EPOCHS,
        method=SlidingSeriesSplit,
        verbose=True
    )

print("Average RMSE on %i folds: %0.4f" %(N_SPLITS, agg_rmse))






# request this number from the user
ZOOM = 2 # Number of years to get higher res for.

# Plotting results.
data = stock_short.filter(['Date', 'AdjClose'])

train = data[:len(X_train)]
valid = data[len(X_train):] # actual data that model predicted.
valid['Predictions'] = y_preds


fig, ax = plt.subplots(1,1, dpi=400, figsize=(16,8),
    constrained_layout=False)

ax.plot(train.Date, train.AdjClose, label='Train')
ax.plot(valid.Date, valid.AdjClose, label='Validation')
ax.plot(valid.Date, valid.Predictions, label='Prediction')

ax.set_title('Adjusted Closing Price Preds for %s (USD), %s-%s' 
             %(TICKER, START.year, END.year))
ax.set_xlabel('Date', fontsize=18)
ax.set_ylabel('Adjusted Closing Price (USD)', fontsize=18)
ax.legend(fontsize='18')

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
# ax.set_xlim(data.Date.min(), data.Date.max());

plt.savefig('Prediction.png')


# Zoom-in Plot.
mid = END - relativedelta(years=ZOOM)
train_z = train[train.Date >= pd.Timestamp(mid)]
valid_z = valid[valid.Date >= pd.Timestamp(mid)]

fig, ax = plt.subplots(1,1, dpi=400, figsize=(16,8),
    constrained_layout=False) 

ax.plot(train_z.Date, train_z.AdjClose)
ax.plot(valid_z.Date, valid_z.AdjClose, label='Validation (Actual)')
ax.plot(valid_z.Date, valid_z.Predictions, label='Predictions')

ax.set_title('Adjusted Closing Price Preds for %s (USD), %s-%s' 
             %(TICKER, mid.year, END.year))
ax.set_xlabel('Date', fontsize=18)
ax.set_ylabel('Adjusted Closing Price (USD)', fontsize=18)
ax.legend(fontsize=18)

# Set major and minor date tick locators
maj_loc = mdates.MonthLocator(bymonth=np.arange(1,12,2))
ax.xaxis.set_major_locator(maj_loc)
min_loc = mdates.MonthLocator()
ax.xaxis.set_minor_locator(min_loc)

# Set major date tick formatter
zfmts = ['', '%b\n%Y', '%b', '%b-%d', '%H:%M', '%H:%M']
maj_fmt = mdates.ConciseDateFormatter(maj_loc, zero_formats=zfmts, 
                                      show_offset=False)
ax.xaxis.set_major_formatter(maj_fmt)

ax.figure.autofmt_xdate(rotation=0, ha='center')
# ax.set_xlim(data.Date.min(), data.Date.max());

plt.savefig('Prediction_Zoom.png')
plt.show()





# TODO

# 0) request info from user
# 1) import data 
# 2) run the cleaning
# 3) import the optimized model
# 4) train the model
# 5) run the predictions, spit out the graphs, then save them to root directory.

