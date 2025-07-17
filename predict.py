import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import talib as ta
import json


def transform_data(stock_path,stock_one_hot_cols):
    stock_data = pd.read_csv(stock_path)
    stock_data = stock_data.rename(columns={'LTP':'Close'})

    # TA-Lib indicators
    stock_data['SMA_20'] = ta.SMA(stock_data['Close'], timeperiod=20)
    stock_data['EMA_50'] = ta.EMA(stock_data['Close'], timeperiod=50)
    stock_data['RSI_14'] = ta.RSI(stock_data['Close'], timeperiod=14)
    
    macd, macds, macdh = ta.MACD(stock_data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    stock_data['MACD_12_26_9'] = macd
    stock_data['MACDs_12_26_9'] = macds
    stock_data['MACDh_12_26_9'] = macdh
    
    upper, middle, lower = ta.BBANDS(stock_data['Close'], timeperiod=20)
    stock_data['BBU_20_2.0'] = upper
    stock_data['BBM_20_2.0'] = middle
    stock_data['BBL_20_2.0'] = lower
    stock_data['BBB_20_2.0'] = upper - lower
    stock_data['BBP_20_2.0'] = (stock_data['Close'] - lower) / (upper - lower)
    
    stock_data['Stock'] = 'ADBL'
    df_one_hot = pd.get_dummies(stock_data['Stock'], prefix='Stock')
    stock_data = pd.concat([stock_data, df_one_hot], axis=1)
    
    
    feature_cols = ['Close', 'High', 'Low', 'Open'] + stock_one_hot_cols + [
    'SMA_20', 'EMA_50', 'RSI_14',
    'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9',
    'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0']
    for col in stock_one_hot_cols:
        if col not in stock_data.columns:
            stock_data[col] = 0

    stock_data = stock_data.dropna()
    latest_data = stock_data[feature_cols].tail(60)

    return latest_data


def inference(latest_data, model):
    scaler = MinMaxScaler()
    input_features = scaler.fit_transform(latest_data)
    model_input = np.expand_dims(input_features, axis=0)
    # breakpoint()
    predictions = model.predict(model_input)
    predicted_class = np.argmax(predictions)
    if predicted_class == 2:
        print(f"class: {predicted_class}->Bullish")
    elif predicted_class == 0:
        print(f"class: {predicted_class}->Bearish")
    else:
        print(f"class: {predicted_class}->Neutral")


if __name__ == "__main__":
    stock_path = "/Users/jibanchaudhary/Documents/Projects/Trading_bot/chart_data/ADBL.csv"
    with open('stock_one_hot_cols.json','r') as f:
        stock_one_hot_cols = json.load(f)  
    latest_data = transform_data(stock_path,stock_one_hot_cols)
    model = load_model('final_multi_stock_lstm_model.keras')

    inference(latest_data, model)
