import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_wind_scaler():
    return joblib.load(os.path.join("models", "wind_scaler.joblib"))

def load_lstm_model(model_name):
    return load_model(os.path.join("models", model_name))

def prepare_lstm_input(df, feature_cols, lookback=12):
    X = []
    for i in range(len(df) - lookback):
        X.append(df[feature_cols].iloc[i:i+lookback].values)
    return np.array(X)

def predict_wind_lstm(input_csv, speed_model_path, dir_model_path, lookback=12):
    df = pd.read_csv(input_csv)
    
    # Add wind_dir_sin and wind_dir_cos in the same order as train_lstm.py
    if 'forecast_wind_direction' in df.columns:
        df['wind_dir_sin'] = np.sin(np.radians(df['forecast_wind_direction']))
        df['wind_dir_cos'] = np.cos(np.radians(df['forecast_wind_direction']))
    else:
        raise ValueError("CSV missing 'forecast_wind_direction' column")
    
    scaler = load_wind_scaler()
    # Explicitly define feature order to match training
    feature_cols = [
        'temperature_2m', 'relative_humidity_2m', 'pressure_msl',
        'cloudcover', 'precipitation', 'weathercode',
        'wind_dir_sin', 'wind_dir_cos'
    ]
    
    # Verify all expected features are present
    missing_features = [f for f in feature_cols if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in input data: {missing_features}")
    
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(df[feature_cols])
    X = prepare_lstm_input(df_scaled, feature_cols, lookback)
    speed_model = load_lstm_model(speed_model_path)
    dir_model = load_lstm_model(dir_model_path)
    speed_preds = speed_model.predict(X)
    dir_preds = dir_model.predict(X)
    return speed_preds, dir_preds

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    forecast_files = [f for f in os.listdir(data_dir) if f.startswith("openmeteo_forecast_") and f.endswith("_train.csv")]
    if not forecast_files:
        raise FileNotFoundError("No forecast training CSV found in data directory")
    
    forecast_files.sort(reverse=True)
    input_csv = os.path.join(data_dir, forecast_files[0])
    print(f"Using input file: {input_csv}")
    
    speed_model_path = os.path.join("models", "wind_speed_lstm.h5")
    dir_model_path = os.path.join("models", "wind_dir_lstm.h5")
    speed_preds, dir_preds = predict_wind_lstm(input_csv, speed_model_path, dir_model_path)
    print("Predicted wind speeds:", speed_preds.flatten())
    print("Predicted wind directions:", dir_preds.flatten())