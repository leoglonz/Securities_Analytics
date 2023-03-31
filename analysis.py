"""
Will hold generalized stock analysis functions akin to those 'Stock_Analysis.ipynb'.

IN PROGRESS
Last Revised: 30 Mar 2023
"""

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization

from base import *


def beta(df, market=None):
    """
    Calculating betas using prices from every business day.
    If the market values are not passed, I'll assume they 
    are located in a column named 'Market'.  If not, 
    this will fail.
    """
    if market is None:
        market = df['MarketClose']
        df = df.drop('MarketClose', axis=1)
    X = market.values.reshape(-1, 1)
    X = np.concatenate([np.ones_like(X), X], axis=1)
    b = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(df.values)
    
    return float(b[1])


def addBeta(stock, market, window=252, verbose=False):
    """
    Calc beta across given df of closing prices.
    Window default is stet to calculate yearly beta.
    """
    betas = np.array([])
    data = pd.concat([stock.AdjClose, market.MarketClose], axis=1)

    for  i, sdf in enumerate(roll(data.pct_change().dropna(), window )):
        betas = np.append(betas, beta(sdf))

    full_data = data.drop(index=data.index[:window], axis=0, inplace=False)
    full_data['Beta'] = betas.tolist()

    return full_data


def backtest_validation(data, make_model, trial=None, lag=60, days=1,
                        train_ratio=0.7, n_splits=5, batch_size=1,
                        epochs=1, method=TimeSeriesSplit, verbose=False):
    """
    Performing rolling or expanding window backtest
    validation for time series data that cannot be
    subjected to the typical procedure of cross validation.
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
    # Fetching feature split and backtest method.
    prams = backtest_split(data, lag=lag, days=days,
                           train_ratio=train_ratio, n_splits=n_splits,
                           method=method, verbose=False)
    
    X, y = prams[0], prams[1]
    series_split, scaler = prams[2], prams[3]
    
    agg_rmse = np.zeros(n_splits) # Initializing; collects all model RMSEs.

    for i, (train_index, test_index) in enumerate(series_split.split(X, y)):
        if verbose:
            print("[Start fold %i/%i]" %(i+1, n_splits))

        # Collect train and test data for the fold.
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Build new model for the fold.
        model = None
        in_shape = X_train.shape[1]
        
        if trial==None:
            model = make_model(in_shape)
        else:
            # We pass a trial object for Optuna hyperpram optimization.
            model = make_model(trial, in_shape)

        model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            shuffle=True,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
        )

        # Calculate the fold's RMSE.
        y_preds = model.predict(X_test)
        y_preds = scaler.inverse_transform(y_preds)
        rmse = np.sqrt(np.sum((y_test-y_preds)**2) / len(y_test))

        if verbose:
            print("Loss for fold %i: %.3e" %(i+1, rmse))

        agg_rmse[i] = rmse

    return agg_rmse


def create_model(trial, in_shape):
    '''
    A modification on a vanilla model function, where in
    this case we pass a trial object that Optuna uses both in
    its optimization routine, and for passing values for the
    hyperparameters in the case of model fitting.
    in_shape: int, gives the col shape (# of features) that the
        first LSTM node should expect to receive.
    '''
    units = trial.suggest_int('units', 64, 150, step=2)
    dropout = trial.suggest_float('dropout', 0, 1)
    classes = trial.suggest_int('classes', 13, 50, step=1)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    # activation = trial.suggest_categorical('dense_activation', [None, 'tanh', 'sigmoid'])
    # recurrent_dropout = trial.suggest_float('recurrent_droupout', 0, 1)

    model = Sequential()
    model.add(
        LSTM(
            units=units,
            activation='tanh',
            recurrent_activation='sigmoid',
            unroll=False,
            use_bias=True,
            dropout=dropout,
            # recurrent_dropout=recurrent_dropout,
            return_sequences=True,
            input_shape=(in_shape, 1)
        )
    )
    model.add(
        LSTM(
            units=int(units/2),
            activation='tanh',
            recurrent_activation='sigmoid',
            unroll=False,
            use_bias=True,
            dropout=dropout,
            # recurrent_dropout=recurrent_dropout,
            return_sequences=False,
        )
    )
    model.add(
        Dense(
            classes,
            activation=None,
            use_bias=True
        )
    )
    model.add(
        Dense(
            1,
            activation= None, # activation,
            use_bias=True
        )
    )

    # Only use 'accuracy' metric for classification.
    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        ),
        metrics=['mean_squared_error'] # ['mean_absolute_percentage_error']
    )
    
    return model
