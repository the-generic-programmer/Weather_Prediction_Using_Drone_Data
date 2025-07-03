import pandas as pd
import numpy as np
import os
import joblib
import logging
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler

def load_wind_scaler():
    scaler_path = os.path.join("models", "wind_scaler.joblib")
    return joblib.load(scaler_path)

def load_lstm_model(model_name):
    return load_model(os.path.join("models", model_name))

def prepare_lstm_input(df, feature_cols, lookback=12):
    X = []
    for i in range(len(df) - lookback):
        X.append(df[feature_cols].iloc[i:i+lookback].values)
    return np.array(X)

def predict_wind_lstm(input_csv, speed_model_path, dir_model_path, lookback=12):
    df = pd.read_csv(input_csv)
    scaler = load_wind_scaler()
    feature_cols = [col for col in df.columns if col not in ["timestamp", "forecast_wind_speed", "forecast_wind_direction"]]
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(df[feature_cols])
    X = prepare_lstm_input(df_scaled, feature_cols, lookback)
    speed_model = load_lstm_model(speed_model_path)
    dir_model = load_lstm_model(dir_model_path)
    speed_preds = speed_model.predict(X)
    dir_preds = dir_model.predict(X)
    return speed_preds, dir_preds

if __name__ == "__main__":
    # Automatically use the latest available forecast file
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    forecast_files = [f for f in os.listdir(data_dir) if f.startswith("openmeteo_forecast_") and f.endswith("_train.csv")]
    if forecast_files:
        forecast_files.sort(reverse=True)
        input_csv = os.path.join(data_dir, forecast_files[0])
        print(f"Using input file: {input_csv}")
    else:
        raise FileNotFoundError("No forecast training CSV found in data directory.")
    speed_model_path = os.path.join("..", "models", "wind_speed_lstm.h5")
    dir_model_path = os.path.join("..", "models", "wind_dir_lstm.h5")
    speed_preds, dir_preds = predict_wind_lstm(input_csv, speed_model_path, dir_model_path)
    print("Predicted wind speeds:", speed_preds.flatten())
    print("Predicted wind directions:", dir_preds.flatten())
