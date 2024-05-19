import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def preprocess_data(df):
    df = pd.DataFrame(df)
    df = df[['Date', 'High', 'Low', 'Close']]
    df['Date'] = pd.to_datetime(df['Date'])

    df.set_index('Date', inplace=True)
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
    model.add(LSTM(units=50, return_sequences=True, input_shape=(seq_length, features)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=features))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs, batch_size):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    model.save("bitcoin_price_prediction_model.h5")


def predict_with_date(model, scaler, df, input_date, seq_length):
    #df['Date'] = pd.to_datetime(df['Date'])
    # Set the 'Date' column as the index
    if df.index.name != 'Date':
        df.set_index('Date', inplace=True)

    
    idx = df.index.get_loc(input_date)
    
    last_data = df.iloc[idx-seq_length:idx][['High', 'Low', 'Close']]
    
    
    last_data_scaled = scaler.transform(last_data)
    
    
    last_data_reshaped = last_data_scaled.reshape(1, seq_length, 3)  # Assuming 3 features
    
    
    next_data_point = model.predict(last_data_reshaped)
    
    
    next_data_point = scaler.inverse_transform(next_data_point)
    
    
    predicted_prices = []
    
    # Predict prices for the next 7 days
    for i in range(7):
       
        predicted_prices.append(next_data_point)
        
        
        last_data = np.concatenate((last_data[1:], next_data_point), axis=0)
        last_data_scaled = scaler.transform(last_data)
        last_data_reshaped = last_data_scaled.reshape(1, seq_length, 3)
        
        
        next_data_point = model.predict(last_data_reshaped)
        
        next_data_point = scaler.inverse_transform(next_data_point)
    
    return predicted_prices, idx

def get_index(input_date):
    idx = df.index.get_loc(input_date)
    return idx


df = pd.read_csv("BTC.csv")
seq_length = 30

scaled_data, scaler = preprocess_data(df)

X, y = create_sequences(scaled_data, seq_length)


train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


model = load_model("bitcoin_price_prediction_model.h5")

# predicted_prices = model.predict(X_test)
# predicted_prices = scaler.inverse_transform(predicted_prices)
# y_test = scaler.inverse_transform(y_test)

input_date = "2023-11-10"
# to_plot = []
#predicted_prices, idx = predict_with_date(model, scaler, df, input_date, seq_length)
# for ele in predicted_prices:
#     to_plot.append(ele[0])
#     break

# actual_to_plot = df.iloc[get_index("2023-11-11")]
# actual_to_plot = actual_to_plot[['High', 'Low', 'Close']].values

# to_plot = np.array(to_plot).flatten()
# print(actual_to_plot)
# print("---------------------")
# print(to_plot)
# plt.plot(['High', 'Low', 'Close'], actual_to_plot, label='Actual', marker='o')
# plt.plot(['High', 'Low', 'Close'], to_plot, label='Predicted', marker='x')

# # Add labels and legend
# plt.ylim(0, 50000)
# plt.xlabel('Index')
# plt.ylabel('Price')
# plt.title('Comparison of Actual and Predicted Prices')
# plt.legend()

# # Show plot
# plt.show()
def predict_with_date_wrapper(inp_date):
    predicted_prices, _ = predict_with_date(model, scaler, df, inp_date, seq_length)
    return predicted_prices

def get_actual_pred(inp_date):
    idx = get_index(inp_date) + 1
    # Number of iterations
    num_iterations = 7

    # Empty list to store rows
    rows_list = []

    # Iterate over the DataFrame from the start_index for num_iterations times
    for i in range(7):
        # Append the row to the list
        rw = df.iloc[i+idx]
        rw = rw[['High', 'Low', 'Close']].values
        rows_list.append(rw)
    return np.array(rows_list).tolist()

def evaluate_model(model, scaler, X_test, y_test, seq_length):
    # Make predictions
    predicted_prices = model.predict(X_test)
    
    # Reshape predicted prices and actual prices for inverse transformation
    predicted_prices = predicted_prices.reshape(predicted_prices.shape[0], predicted_prices.shape[1])
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])

    # Inverse transform the predictions and actual prices
    predicted_prices = scaler.inverse_transform(predicted_prices)
    y_test = scaler.inverse_transform(y_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, predicted_prices)
    mse = mean_squared_error(y_test, predicted_prices)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predicted_prices)

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared (R2 ): {r2}")

    # Plotting actual vs predicted prices
    plt.figure(figsize=(14, 7))
    plt.plot(y_test[:, 2], color='blue', label='Actual Close Price')  # Plotting Close Price
    plt.plot(predicted_prices[:, 2], color='red', label='Predicted Close Price')  # Plotting Close Price
    plt.title('Bitcoin Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Example usage
evaluate_model(model, scaler, X_test, y_test, seq_length)