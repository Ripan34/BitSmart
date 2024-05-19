import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from .better_test import predict_with_date_wrapper, get_actual_pred, get_strategy


def predict_bitcoin_prices(input_date):    
    predicted_prices = predict_with_date_wrapper(input_date)
    res = np.array(predicted_prices).tolist()
    flat_list = [item for sublist in res for item in sublist]
    sell_date, buy_date = get_strategy(flat_list, input_date)
    return flat_list, sell_date, buy_date

def compute_metrics(res):
    high = max(row[0] for row in res)
    low = min(row[1] for row in res)
    close_avg = sum(row[2] for row in res) / len(res)
    return high, low, close_avg

def get_actual_pred_service(inp_date):
    return get_actual_pred(inp_date)