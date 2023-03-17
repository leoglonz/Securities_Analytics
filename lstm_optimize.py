"""
Optuna implementation that optimizes an LSTM neural network lag series 
stock data using Keras.

We optimize LSTM units, LSTM layer dropout, and learning rate.

IN PROGRESS
Last Revised: 15 Mar 2023
"""

import warnings
import sys
import os

import optuna
import base.py 

from keras.backend import clear_session
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam


BATCHSIZE = 1 #128
CLASSES = 10
EPOCHS = 1 #10


def create_model(trial):
    model = Sequential()

    units=trial.suggest_int('unit', 64, 128, step=2)
    model.add(
        LSTM(
            units=units,
            activation='tanh',
            recurrent_activation='sigmoid',
            unroll=False,
            use_bias=True,
            dropout=trial.suggest_float('droupout', 0, 1),
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
    model.add(
        Dense(
            CLASSES,
            activation='relu',
            use_bias=True
        )
    )
    model.add(
        Dense(
            1,
            activation='relu',
            use_bias=True
        )
    )

    # We compile our model with a sampled learning rate.
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    model.compile(
        loss="mean_squared_error",
        optimizer=Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        ),
        metrics=["accuracy"]
    )

    if trial.should_prune():
            raise optuna.TrialPruned()
    
    return model


def objective(trial):
    model = create_model(trial)
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        shuffle=True,
        batch_size=BATCHSIZE,
        epochs=EPOCHS,
        verbose=True,
    )

    # Evaluate the model accuracy on the validation set.
    score = model.evaluate(X_train, y_train, verbose=True)
    
    return score[1]


if __name__ == "__main__":
    study_name = str(sys.argv[1])
    n_trials = int(sys.argv[1])

    warnings.warn(
        "Layer LSTM will only use cuDNN high-efficiency kernals "
        "when training with layer params 'activation==tanh' "
        "'recurrent_activation==sigmoid', 'unroll=False', "
        "'use_bias=True', and 'recurrent_dropout=0.0'."
    )
    study = optuna.create_study(direction="maximize", study_name=study_name)
    # Use n_jobs=-1 for full parallelization.
    study.optimize(objective, n_trials=n_trials, n_jobs=1, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
