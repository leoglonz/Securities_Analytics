"""
Optuna implementation that optimizes an LSTM neural network lag series 
stock data using Keras.

We optimize LSTM units, LSTM layer dropout, dense layer classes, learning rate,
batch size, and epochs.

IN PROGRESS
Last Revised: 19 Mar 2023
"""

import sys
import os

import optuna

from tensorflow import keras
from keras.backend import clear_session
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error


from base import *
import datetime as dt


TICKER = 'AMZN'
START = dt.datetime(2012, 1, 1)
END = dt.datetime.today()

LAG = 60
DAYS = 1
TRAIN_RATIO = 0.70

# CLASSES = 25
# BATCHSIZE = 128 #128
# EPOCHS = 1 #10

STUDY = 'OptDebug'
N_TRIALS = 10
BACKTEST = True
S_TYPE = TimeSeriesSplit


def create_model(trial, in_shape):
    units = trial.suggest_int('units', 64, 128, step=2)
    dropout = trial.suggest_float('dropout', 0, 1)
    classes = trial.suggest_int('classes', 13, 30, step=1)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
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
            dropout=dropout ,
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
            activation=None, # try 'relu'
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

    if trial.should_prune():
            raise optuna.TrialPruned()
    
    return model


def objective(trial):
    keras.backend.clear_session()

    # Importing and prepping data:
    data = datImport(TICKER, start = START, end=END, verbose=False)

    batchsize = trial.suggest_int('batchsize', 1, 128, step=1)
    epochs = trial.suggest_int('epochs', 1, 10, step=1)

    if BACKTEST:
        # Using backtest validation (sliding or expanding window)
        # as the target optimization metric for Optuna.
        split = data_split(data, LAG, DAYS, TRAIN_RATIO, backtest=True)
        X, y = split[0], split[1]

        # n_splits = trial.suggest_int('n_splits', 5, 10, step=1)
        # s_type = trial.suggest_categproca;('s_type', ['TimeSeriesSplit', 
        #                                               'SlidingSeriesSplit'])

        model = BaseWrapper(
            model=create_model(trial, split),
            shuffle=True,
            epochs=epochs,
            batch_size=batchsize,
            verbose=True
            )
        series_split = S_TYPE(n_splits=5)

        cv_scores = -1 * cross_val_score(
            model,
            X, y,
            cv=series_split,
            scoring='neg_mean_squared_error',
            n_jobs=1
            )
        
        return cv_scores.mean()

    else:
        # Using standard RMSE as target optimization metric.
        split = data_split(data, LAG, DAYS, TRAIN_RATIO)
        X_train, y_train = split[0], split[1]
        X_test, y_test = split[2], split[3]
        scaler = split[4]

        in_shape = split[0].shape[1]
        
        model = create_model(trial, in_shape)
        model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            shuffle=True,
            batch_size=batchsize,
            epochs=epochs,
            verbose=True,
        )

        # Evaluate model accuracy on the validation set w/ RMSE.
        y_preds = model.predict(X_test)
        y_preds = scaler.inverse_transform(y_preds)
        rmse = mean_squared_error(y_test, y_preds, squared=False)
        
        return rmse


if __name__ == "__main__":
    study_name = STUDY
    n_trials = N_TRIALS

    if len(sys.argv) > 1:
        n_trials = int(sys.argv[1])

    if len(sys.argv) > 2:
        study_name = str(sys.argv[2])

    # Suppress TensorFlow debug output.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.warn(
        "Layer LSTM will only use cuDNN high-efficiency kernals "
        "when training with layer params 'activation==tanh' "
        "'recurrent_activation==sigmoid', 'unroll=False', "
        "'use_bias=True', and 'recurrent_dropout=0.0'."
    )

    print("---------------------- \n"
          "Beginning %i trials under '%s'. \n" 
          "----------------------" 
          %(n_trials, study_name))

    study = optuna.create_study(direction="minimize", study_name=study_name)
    # Use n_jobs=-1 for full parallelization.
    study.optimize(objective, n_trials=n_trials, n_jobs=1, timeout=600)

    print("Number of finished trials: %i" %len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("    RMSE Value: %.3e" %trial.value)

    print("    Params: ")
    for key, value in trial.params.items():
        print("    %s: %.3e" %(key, value))
