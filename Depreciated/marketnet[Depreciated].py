import math
import pandas_datareader as pdr
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import datetime



start = '2010-07-29'
end = datetime.datetime.today()

def ml_base(ticker, start, end):
    '''
    -> Basic ML algorithm to compute next day ticker closing price from previous 60 days of closing-price data.
    '''
    #--------------------------------#
    
    # scraping stock.
    df = pdr.DataReader('ticker', data_source='yahoo', start=start, end=end)

    close = df.filter(['Close']).values
    len_training = math.ceil(len(close) * 0.8)

    # scaling data.
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_close = scaler.fit_transform(close)

    # scaled training data.
    train_close = scaled_close[0:len_training,:]

    x_train = [] # features
    y_train = [] # desired prediction

    for i in range(60, len(train_close)):
        x_train.append(train_close[i-60:i,0])
        y_train.append(train_close[i,0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)

    # reshaping to obtain 3D reps for x_train (which is currently 2d) to pass into LSTM.
    # LSTM expectt d1 # of samples, d2 # of timesteps, and d3 # of features.

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


    # using 2 LSTM layers + Dense 25 neuron + Dense 1 neuron.
    model = Sequential()

    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))





model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=10, batch_size=32, shuffle=True)








    # compiling.
    model.compile(optimizer='adam', loss='mean_squared_error')


    # testing.
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    test_data = scaled_close[len_training - 60:, :]

    x_test = []
    y_test = close[len_training:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
        
    x_test = np.array(x_test)

    # reshaping 2d x_test into 3d.
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


















import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM



# importing data.
company = 'FB'

start = dt.datetime(2012,1,1)
end = dt.datetime(2020,1,1)

data = web.DataReader(company, 'yahoo', start, end)

# data prep.
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1)) #or could use adjusted close.

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x,0])
    
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# now building the model.
model = Sequential()

# layers.
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=10, batch_size=32)


### testing model accuracy on test data ###

# loading in test data
test_start = dt.datetime(2020,1,1)
test_end = dt.datetime(2021,2,2)

test_data = web.DataReader(company, 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)


# making predictions on test data.

x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# ploting test predictions.
plt.plot(actual_prices, color='b', label='Actual Price')
plt.plot(predicted_prices, color='r', label='Predicted Price')

plt.title(f'Predicted {company} Share Price')
plt.xlabel('Time (Days)')
plt.ylabel('Share Price ($USD$)')

plt.legend()
plt.show()


# next-day predictions

real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs+1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1],1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f'prediction: {prediction}')
