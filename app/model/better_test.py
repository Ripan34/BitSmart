import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta

df = pd.read_csv("model/BTC-USD.csv")
seq_length = 30

def preprocess_data(df):
    df = pd.DataFrame(df)
    df = df[['Date', 'High', 'Low', 'Close', 'Volume']]
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.fillna(method='ffill', inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def create_model(seq_length, features):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(seq_length, features)))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=features))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs, batch_size):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    model.save("bitcoin_price_prediction_model_better.h5")

def predict_with_date(model, scaler, df, input_date, seq_length):
    if df.index.name != 'Date':
        df.set_index('Date', inplace=True)
    idx = df.index.get_loc(input_date)
    last_data = df.iloc[idx-seq_length:idx][['High', 'Low', 'Close', 'Volume']]
    last_data_scaled = scaler.transform(last_data)
    last_data_reshaped = last_data_scaled.reshape(1, seq_length, last_data.shape[1])
    next_data_point = model.predict(last_data_reshaped)
    next_data_point = scaler.inverse_transform(next_data_point)
    predicted_prices = []
    for i in range(7):
        predicted_prices.append(next_data_point)
        last_data = np.concatenate((last_data[1:], next_data_point), axis=0)
        last_data_scaled = scaler.transform(last_data)
        last_data_reshaped = last_data_scaled.reshape(1, seq_length, last_data.shape[1])
        next_data_point = model.predict(last_data_reshaped)
        next_data_point = scaler.inverse_transform(next_data_point)
    return predicted_prices, idx

def get_index(input_date):
    try:
        idx = df.index.get_loc(input_date)
        return idx
    except KeyError:
        return -1

def predict_with_date_wrapper(inp_date):
    model = load_model("model/bitcoin_price_prediction_model_better.h5")
    predicted_prices, _ = predict_with_date(model, scaler, df, inp_date, seq_length)
    return predicted_prices

def get_actual_pred(inp_date):
    target_date = datetime.strptime("2024-05-12", "%Y-%m-%d")
    date = datetime.strptime(inp_date, "%Y-%m-%d")
    if date > target_date:
        return []
    idx = get_index(inp_date)
    if idx == -1:
        return []
    idx += 1
    rows_list = []
    for i in range(7):
        rw = df.iloc[i+idx]
        rw = rw[['High', 'Low', 'Close', 'Volume']].values
        rows_list.append(rw)
    return np.array(rows_list).tolist()

scaled_data, scaler = preprocess_data(df)

def initialize():
    X, y = create_sequences(scaled_data, seq_length)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = create_model(seq_length, X_train.shape[2])
    train_model(model, X_train, y_train, epochs=30, batch_size=10)

def evaluate_model(model, scaler, X_test, y_test):
    predicted_prices = model.predict(X_test)
    
    predicted_prices = predicted_prices.reshape(predicted_prices.shape[0], predicted_prices.shape[1])
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])

    predicted_prices = scaler.inverse_transform(predicted_prices)
    y_test = scaler.inverse_transform(y_test)

    mae = mean_absolute_error(y_test, predicted_prices)
    mse = mean_squared_error(y_test, predicted_prices)

    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")

    plt.figure(figsize=(14, 7))
    plt.plot(y_test[:, 2], color='blue', label='Actual Close Price')
    plt.plot(predicted_prices[:, 2], color='red', label='Predicted Close Price')
    plt.title('Bitcoin Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def get_strategy(predictions, inp_date):
    idx = get_index(inp_date)
    if idx == -1:
        return '', ''

    row = df.iloc[idx]
    open_price = row["Open"]

    max_profit = 0
    sell_date = ''
    buy_date = ''
    date_format = '%Y-%m-%d'
    date = datetime.strptime(inp_date, date_format)

    for sell_ind in range(len(predictions)):
        sell_price = predictions[sell_ind][2]

        profit = sell_price - open_price
        if profit > max_profit:
            max_profit = profit
            sell_date = date + timedelta(days=sell_ind)
        for buy_ind in range(sell_ind + 1, len(predictions)):
            buy_price = predictions[buy_ind][2]

            profit = (sell_price - open_price) + (open_price - buy_price)
            if profit > max_profit:
                max_profit = profit
                sell_date = date + timedelta(days=sell_ind)
                buy_date = date + timedelta(days=buy_ind)

    return sell_date,  buy_date


# X, y = create_sequences(scaled_data, seq_length)


# train_size = int(len(X) * 0.8)
# X_train, X_test = X[:train_size], X[train_size:]
# y_train, y_test = y[:train_size], y[train_size:]


# model = load_model("bitcoin_price_prediction_model_better.h5")
# evaluate_model(model, scaler, X_test, y_test)

#initialize()

