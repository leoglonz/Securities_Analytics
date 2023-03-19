"""
Optuna implementation that optimizes an LSTM neural network lag series 
stock data using Keras.

We optimize LSTM units, LSTM layer dropout, dense layer classes, and learning rate.

IN PROGRESS
Last Revised: 15 Mar 2023
"""

import sys
import os

import optuna

from keras.backend import clear_session
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
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


def create_model(trial, split):
    X_train, y_train = split[0], split[1]
    X_test, y_test = split[2], split[3]

    model = Sequential()

    units = trial.suggest_int('units', 64, 128, step=2)
    model.add(
        LSTM(
            units=units,
            activation='tanh',
            recurrent_activation='sigmoid',
            unroll=False,
            use_bias=True,
            dropout=trial.suggest_float('dropout', 0, 1),
            # recurrent_dropout=trial.suggest_float('recurrent_droupout', 0, 1),
            return_sequences=True,
            input_shape=(X_train.shape[1], 1)
        )
    )
    model.add(
        LSTM(
            units=int(units/2),
            activation='tanh',
            recurrent_activation='sigmoid',
            unroll=False,
            use_bias=True,
            dropout=trial.suggest_float('droupout', 0, 1),
            # recurrent_dropout=trial.suggest_float('recurrent_droupout', 0, 1),
            return_sequences=False,
        )
    )
    classes = trial.suggest_int('classes', 13, 30, step=1)
    model.add(
        Dense(
            classes,
            activation=None,
            # use_bias=True
        )
    )
    model.add(
        Dense(
            1,
            # activation='relu',
            # use_bias=True
        )
    )

    # We compile our model with a sampled learning rate.
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    # Only use 'accuracy' metric for classification.
    model.compile(
        loss="mean_squared_error",
        optimizer=Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        ),
        metrics=['mean_squared_error']#['mean_absolute_percentage_error']
    )

    if trial.should_prune():
            raise optuna.TrialPruned()
    
    return model


def objective(trial):
    # Importing and prepping data:
    data = datImport(TICKER, start = START, end=END, verbose=False)
    split = data_split(data, LAG, DAYS, TRAIN_RATIO)
    X_train, y_train = split[0], split[1]
    X_test, y_test = split[2], split[3]
    scaler = split[4]

    batchsize = trial.suggest_int('batchsize', 1, 128, step=1)
    epochs = trial.suggest_int('epochs', 1, 10, step=1)
    model = create_model(trial, split)
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        shuffle=True,
        batch_size=batchsize,
        epochs=epochs,
        verbose=True,
    )

    # Evaluate the model accuracy on the validation set.
    y_preds = model.predict(X_test)
    y_preds = scaler.inverse_transform(y_preds)

    # # RMSE.
    # rmse = np.sqrt(np.mean(predictions - y_test)**2)

    rmse = mean_squared_error(y_test, y_preds, squared=False)
    
    return rmse


if __name__ == "__main__":
    study_name = STUDY #str(sys.argv[1])
    n_trials = N_TRIALS # int(sys.argv[1])

    # Suppress TensorFlow debug output.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.warn(
        "Layer LSTM will only use cuDNN high-efficiency kernals "
        "when training with layer params 'activation==tanh' "
        "'recurrent_activation==sigmoid', 'unroll=False', "
        "'use_bias=True', and 'recurrent_dropout=0.0'."
    )

    study = optuna.create_study(direction="minimize", study_name=study_name)
    # Use n_jobs=-1 for full parallelization.
    study.optimize(objective, n_trials=n_trials, n_jobs=2, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
