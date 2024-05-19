from flask import Flask, jsonify
from model.predict_service import predict_bitcoin_prices, compute_metrics, get_actual_pred_service
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/predict/<string:inp_date>")
def predict(inp_date):
    res, sell_date, buy_date = predict_bitcoin_prices(inp_date)
    high, low, close_avg = compute_metrics(res)
    res_obj = {}
    res_obj["high"] = round(high, 2)
    res_obj["low"] = round(low, 2)
    res_obj["close_avg"] = round(close_avg, 2)
    res_obj["predictions"] = res
    res_obj["sell_date"] = sell_date
    res_obj["buy_date"] = buy_date
    actuals = get_actual_pred_service(inp_date)
    if len(actuals) != 0:
        high_actual, low_actual, close_avg_actual = compute_metrics(actuals)
        res_obj["high_actual"] = high_actual
        res_obj["low_actual"] = low_actual
        res_obj["close_avg_actual"] = close_avg_actual
    return jsonify(res_obj)