"""
Optuna implementation that optimizes an LSTM neural network lag series 
stock data using Keras.

We optimize LSTM units, LSTM layer dropout, dense layer classes, learning rate,
batch size, and epochs.

IN PROGRESS
Last Revised: 21 Mar 2023
"""
import sys
import os

import optuna
from optuna.samplers import TPESampler, QMCSampler

from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from keras.backend import clear_session
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

from base import *
from analysis import *
import datetime as dt

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)



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

STUDY = 'OptDebug'
N_JOBS = 2

N_TRIALS = 20
BACKTEST = False
N_SPLITS = 5
S_TYPE = SlidingSeriesSplit # TimeSeriesSplit


def create_model(trial, in_shape):
    units = trial.suggest_int('units', 64, 150, step=2)
    dropout = trial.suggest_float('dropout', 0, 1)
    classes = trial.suggest_int('classes', 13, 50, step=1)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    # activation = trial.suggest_categorical('dense_activation', [None, 'tanh', 'sigmoid'])
    # recurrent_dropout = trial.suggest_float('recurrent_droupout', 0, 1)

    print(['UNITS etc.:', units, dropout, classes, learning_rate])

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
            activation= None, #activation,
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


def objective(trial):
    clear_session()

    # Importing and prepping data:
    data = datImport(TICKER, start = START, end=END, verbose=False)

    batchsize = trial.suggest_int('batchsize', 1, 128, step=1)
    epochs = trial.suggest_int('epochs', 1, 10, step=1)

    print(['batchsize etc.:', batchsize, epochs])

    if BACKTEST:
        # Using backtest validation (sliding or expanding window)
        # as the target optimization metric for Optuna.
        split = data_split(data, LAG, DAYS, TRAIN_RATIO, backtest=True)
        X, y = split[0], split[1]

        # n_splits = trial.suggest_int('n_splits', 5, 10, step=1)
        # s_type = trial.suggest_categproca;('s_type', ['TimeSeriesSplit', 
        #                                               'SlidingSeriesSplit'])

        agg_rmse = backtest_validation(
            data,
            make_model=create_model,
            trial=trial,
            n_splits=N_SPLITS,
            batch_size=batchsize,
            epochs=epochs,
            method=TimeSeriesSplit,
            verbose=True
        )

        return agg_rmse.mean()
        
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
        
        # Choosing the loss metric we want to minimize.
        # if LOSS_METRIC == 'mean_absolute_percentage_error':
        #     loss = np.sum(np.abs((y_test-y_preds) / y_test)) / len(y_test)

        # if LOSS_METRIC == 'mean_squared_error':
        #     loss = np.sqrt(np.sum((y_test-y_preds)**2) / len(y_test))

        loss = mean_squared_error(y_test, y_preds, squared=False)
        
        # # Save the current model iteration.
        # path = '/Users/leoglonz/Desktop/stock_analysis/opt_cache/'
    
        # with open(path + "{}_{}.pickle".format(study.study_name,
        #                                      trial.number), "wb") as fout:
        #     pickle.dump(model, fout)

        return loss


if __name__ == "__main__":
    study_name = STUDY
    n_trials = N_TRIALS

    if len(sys.argv) == 2:
        n_trials = int(sys.argv[1])

    if len(sys.argv) == 3:
        study_name = str(sys.argv[1])
        n_trials = int(sys.argv[2])


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

    # Note: pruning is not currently implemented.
    study = optuna.create_study(
        direction='minimize',
        # sampler=TPESampler(),
        # pruner=optuna.pruners.MedianPruner(
        #     n_startup_trials=5,
        #     n_warmup_steps=30,
        #     interval_steps=10 
        #     ),
        study_name=study_name
        )
    # Use n_jobs=-1 for full parallelization.
    study.optimize(objective, n_trials=n_trials, n_jobs=N_JOBS, timeout=None)

    print("Number of finished trials: %i" %len(study.trials))
    print("Best trial:")

    trial = study.best_trial
    print("    RMSE Value: %.3e" %trial.value)
    print("    Params: ")
    for key, value in trial.params.items():
        print("    %s: %.3s" %(key, str(value)))


    # Saving Study:
    dir = 'opt_cache/'
    parent_path = '/Users/leoglonz/Desktop/stock_analysis/'
    path = os.path.join(parent_path, dir)

    # Creating file dir if none exists.
    if not os.path.isdir(path):
        print("Creating directory '%s'" %dir)
        os.makdir(path)

    # Setting a cross-validation tag for the file.
    if BACKTEST:
        bt = '_wCV'
    else:
        bt = ''

    with open(path + "{}{}.pickle".format(study.study_name, bt),
              'wb') as fout:
        pickle.dump(study, fout)
    print("Best params saved to: ", path + "{}{}.pickle".format(study.study_name, bt))
    