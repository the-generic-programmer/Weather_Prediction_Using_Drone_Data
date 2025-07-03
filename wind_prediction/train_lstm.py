import pandas as pd
import numpy as np
import os
import joblib
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_lstm_data(df, feature_cols, target_col, lookback=12):
    X, y = [], []
    for i in range(len(df) - lookback):
        X.append(df[feature_cols].iloc[i:i+lookback].values)
        y.append(df[target_col].iloc[i+lookback])
    return np.array(X), np.array(y)

if __name__ == "__main__":
    import sys
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(data_dir, exist_ok=True)
    print("Select data source for training:")
    print("1. Open-Meteo")
    print("2. NOAA GFS")
    choice = input("Enter 1 or 2: ").strip()
    if choice == "2":
        # Use latest GFS file
        gfs_files = [f for f in os.listdir(data_dir) if f.startswith("gfs_wind_") and f.endswith(".csv")]
        if gfs_files:
            gfs_files.sort(reverse=True)
            data_path = os.path.join(data_dir, gfs_files[0])
            print(f"Using GFS data: {data_path}")
        else:
            print("No GFS data found. Please run fetch_gfs_wind.py first.")
            sys.exit(1)
    else:
        forecast_files = [f for f in os.listdir(data_dir) if f.startswith("openmeteo_forecast_") and f.endswith("_train.csv")]
        if forecast_files:
            forecast_files.sort(reverse=True)
            data_path = os.path.join(data_dir, forecast_files[0])
            print(f"Using Open-Meteo data: {data_path}")
        else:
            data_path = os.path.join(data_dir, "historical_weather.csv")
            print(f"Using fallback data: {data_path}")
    df = pd.read_csv(data_path)
    # Remove columns with all NaN or only one unique value (not useful for ML)
    df = df.dropna(axis=1, how='all')
    nunique = df.nunique()
    df = df[[col for col in df.columns if nunique[col] > 1 or col in ["forecast_wind_speed", "forecast_wind_direction"]]]
    # Add sine/cosine encoding for wind direction
    if "wind_dir_sin" not in df.columns and "forecast_wind_direction" in df.columns:
        df["wind_dir_sin"] = np.sin(np.deg2rad(df["forecast_wind_direction"]))
    if "wind_dir_cos" not in df.columns and "forecast_wind_direction" in df.columns:
        df["wind_dir_cos"] = np.cos(np.deg2rad(df["forecast_wind_direction"]))
    # If no feature columns, try to fetch new data with more features
    feature_cols = [col for col in df.columns if col not in ["timestamp", "forecast_wind_speed", "forecast_wind_direction"]]
    if not feature_cols:
        print("No feature columns found. Attempting to fetch new data with more features...")
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
        from fetch_wind_data import fetch_openmeteo_wind_data
        lat, lon = 51.5074, -0.1278
        fetch_openmeteo_wind_data(lat, lon, hours_ahead=48, save_to_file=True)
        print("New data fetched. Please rerun the script.")
        exit(1)
    # Drop any non-numeric columns from feature_cols
    feature_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
    if not feature_cols:
        raise ValueError(f"No numeric feature columns found for scaling. Columns in data: {list(df.columns)}. Check your input data.")
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    joblib.dump(scaler, os.path.join(os.path.dirname(__file__), "..", "models", "wind_scaler.joblib"))
    lookback = 24
    # Wind speed
    X_s, y_s = prepare_lstm_data(df, feature_cols, "forecast_wind_speed", lookback)
    # Wind direction
    X_d, y_d = prepare_lstm_data(df, feature_cols, "forecast_wind_direction", lookback)
    X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(X_s, y_s, test_size=0.2, random_state=42)
    X_train_d, X_val_d, y_train_d, y_val_d = train_test_split(X_d, y_d, test_size=0.2, random_state=42)
    # Build and train models
    model_speed = build_lstm_model(X_train_s.shape[1:])
    model_dir = build_lstm_model(X_train_d.shape[1:])
    from tensorflow.keras.callbacks import EarlyStopping
    es = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    model_speed.fit(X_train_s, y_train_s, epochs=60, batch_size=32, validation_data=(X_val_s, y_val_s), callbacks=[es])
    model_dir.fit(X_train_d, y_train_d, epochs=60, batch_size=32, validation_data=(X_val_d, y_val_d), callbacks=[es])
    model_speed.save(os.path.join(os.path.dirname(__file__), "..", "models", "wind_speed_lstm.h5"))
    model_dir.save(os.path.join(os.path.dirname(__file__), "..", "models", "wind_dir_lstm.h5"))
    print("LSTM models trained and saved.")
