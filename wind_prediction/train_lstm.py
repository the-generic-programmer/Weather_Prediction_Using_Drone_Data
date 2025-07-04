import sys
import os
import warnings

# Suppress TensorFlow oneDNN and CUDA warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml_weather.fetch_meteostat_hourly import fetch_meteostat_hourly

import logging
import joblib
import numpy as np
import pandas as pd
import argparse
import random
import socket
import time
import json
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def build_lstm_model(input_shape):
    """Builds and compiles a Bidirectional LSTM model."""
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LayerNormalization())
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def prepare_lstm_data(df, feature_cols, target_col, lookback=24):
    """Prepares data for LSTM input with lookback window."""
    X, y = [], []
    for i in range(len(df) - lookback):
        X.append(df[feature_cols].iloc[i:i+lookback].values)
        y.append(df[target_col].iloc[i+lookback])
    return np.array(X), np.array(y)

def preprocess_data(df, target_col):
    """Cleans and scales dataframe, adds wind direction features if needed."""
    df = df.dropna(axis=1, how='all')
    nunique = df.nunique()
    df = df[[col for col in df.columns if nunique[col] > 1 or col in ["forecast_wind_speed", "forecast_wind_direction"]]]

    if "wind_dir_sin" not in df.columns and "forecast_wind_direction" in df.columns:
        df["wind_dir_sin"] = np.sin(np.deg2rad(df["forecast_wind_direction"]))
    if "wind_dir_cos" not in df.columns and "forecast_wind_direction" in df.columns:
        df["wind_dir_cos"] = np.cos(np.deg2rad(df["forecast_wind_direction"]))

    feature_cols = [col for col in df.columns if col not in ["timestamp", "forecast_wind_speed", "forecast_wind_direction"]]
    feature_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]

    if not feature_cols:
        raise ValueError("No numeric feature columns found.")

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Ensure the models directory exists before saving the scaler
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, "wind_scaler.joblib"))

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    return df, feature_cols

def train_and_save_model(X, y, input_shape, model_path, epochs=100, batch_size=32):
    """Trains LSTM model and saves the best model to disk."""
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
    model = build_lstm_model(input_shape)
    logging.info(f"Model summary for {model_path}:")
    model.summary(print_fn=logging.info)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
        ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss')
    ]

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=2
    )
    return model, history

def get_latest_drone_gps_from_tcp(host='127.0.0.1', port=9000, timeout=30):
    """Connect to MAVSDK logger TCP and get the latest GPS coordinates."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)
                s.connect((host, port))
                data = s.recv(4096)
                if not data:
                    continue
                for line in data.decode(errors='ignore').split('\n'):
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line)
                        lat = msg.get('latitude')
                        lon = msg.get('longitude')
                        if lat is not None and lon is not None:
                            return float(lat), float(lon)
                    except Exception:
                        continue
        except Exception:
            time.sleep(2)
    raise RuntimeError("Could not get GPS from MAVSDK logger TCP.")

def main():
    parser = argparse.ArgumentParser(description="Train LSTM models for wind speed and direction prediction.")
    parser.add_argument('--lookback', type=int, default=24, help='Lookback window for LSTM')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--meteostat_csv', type=str, default=None, help='Path to Meteostat CSV for training (optional)')
    parser.add_argument('--use_meteostat_from_drone', action='store_true', help='Fetch Meteostat data using latest drone GPS from MAVSDK logger TCP')
    parser.add_argument('--years', type=int, default=5, help='Number of years of Meteostat data to fetch if using drone GPS')
    parser.add_argument('--tcp_port', type=int, default=9000, help='TCP port for MAVSDK logger')
    args = parser.parse_args()

    if args.use_meteostat_from_drone:
        print("Connecting to MAVSDK logger TCP to get latest drone GPS...")
        lat, lon = get_latest_drone_gps_from_tcp(port=args.tcp_port)
        print(f"Got drone GPS: lat={lat}, lon={lon}")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * args.years)
        output_csv = f"data/meteostat_{lat}_{lon}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        fetch_meteostat_hourly(lat, lon, start_date, end_date, output_csv)
        data_path = output_csv
        print(f"Using Meteostat data: {data_path}")
    elif args.meteostat_csv:
        data_path = args.meteostat_csv
        print(f"Using Meteostat data: {data_path}")
    else:
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
        os.makedirs(data_dir, exist_ok=True)
        print("Select data source for training:")
        print("1. Open-Meteo")
        print("2. NOAA GFS")
        print("3. Meteostat (manual location)")
        choice = input("Enter 1, 2, or 3: ").strip()
        if choice == "2":
            gfs_files = [f for f in os.listdir(data_dir) if f.startswith("gfs_wind_") and f.endswith(".csv")]
            gfs_files.sort(reverse=True)
            if not gfs_files:
                sys.exit("No GFS data found.")
            data_path = os.path.join(data_dir, gfs_files[0])
        elif choice == "3":
            lat = float(input("Enter latitude: ").strip())
            lon = float(input("Enter longitude: ").strip())
            years = int(input("How many years of data? (default 5): ").strip() or 5)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * years)
            output_csv = os.path.join(data_dir, f"meteostat_{lat}_{lon}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv")
            success = fetch_meteostat_hourly(lat, lon, start_date, end_date, output_csv)
            if not success or not os.path.exists(output_csv):
                sys.exit(f"No Meteostat data available for {lat},{lon}. Training aborted.")
            data_path = output_csv
            print(f"Using Meteostat data: {data_path}")
        else:
            forecast_files = [f for f in os.listdir(data_dir) if f.startswith("openmeteo_forecast_") and f.endswith("_train.csv")]
            forecast_files.sort(reverse=True)
            data_path = os.path.join(data_dir, forecast_files[0]) if forecast_files else os.path.join(data_dir, "historical_weather.csv")
            if not os.path.exists(data_path):
                sys.exit(f"No forecast data found at {data_path}.")

    logging.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    # Wind Speed Model
    try:
        df_speed, feature_cols_speed = preprocess_data(df.copy(), "forecast_wind_speed")
        X_s, y_s = prepare_lstm_data(df_speed, feature_cols_speed, "forecast_wind_speed", args.lookback)
        model_speed_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "wind_speed_lstm.h5"))
        model_speed, hist_speed = train_and_save_model(X_s, y_s, X_s.shape[1:], model_speed_path, args.epochs, args.batch_size)
        logging.info(f"Wind speed LSTM model trained and saved to {model_speed_path}")
    except Exception as e:
        logging.error(f"Error training wind speed model: {e}")

    # Wind Direction Model
    try:
        df_dir, feature_cols_dir = preprocess_data(df.copy(), "forecast_wind_direction")
        X_d, y_d = prepare_lstm_data(df_dir, feature_cols_dir, "forecast_wind_direction", args.lookback)
        model_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "wind_dir_lstm.h5"))
        model_dir, hist_dir = train_and_save_model(X_d, y_d, X_d.shape[1:], model_dir_path, args.epochs, args.batch_size)
        logging.info(f"Wind direction LSTM model trained and saved to {model_dir_path}")
    except Exception as e:
        logging.error(f"Error training wind direction model: {e}")

    logging.info("LSTM models training complete.")

if __name__ == "__main__":
    main()
