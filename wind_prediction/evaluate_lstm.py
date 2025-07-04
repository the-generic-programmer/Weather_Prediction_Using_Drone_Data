import os
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import requests
from datetime import datetime, timedelta
from meteostat import Hourly, Point

def fetch_meteostat_eval_data(lat, lon, hours_ahead=24):
    now = datetime.utcnow()
    end_time = now + timedelta(hours=hours_ahead)
    location = Point(lat, lon)
    data = Hourly(location, now, end_time)
    df = data.fetch().reset_index()
    if df.empty:
        raise ValueError("No Meteostat data available for the given location/time.")
    # Standardize column names to match expected format
    df = df.rename(columns={
        'time': 'timestamp',
        'wspd': 'forecast_wind_speed',
        'wdir': 'forecast_wind_direction',
        'temp': 'temperature_2m',
        'rhum': 'relative_humidity_2m',
        'pres': 'pressure_msl',
        'coco': 'weathercode',
        'prcp': 'precipitation',
        'cldc': 'cloudcover'
    })
    # Add sine/cosine encoding for wind direction
    df["wind_dir_sin"] = np.sin(np.deg2rad(df["forecast_wind_direction"]))
    df["wind_dir_cos"] = np.cos(np.deg2rad(df["forecast_wind_direction"]))
    return df

def prepare_lstm_input(df, feature_cols, lookback=12):
    X = []
    for i in range(len(df) - lookback):
        X.append(df[feature_cols].iloc[i:i+lookback].values)
    return np.array(X)

def evaluate_lstm_models(eval_df, scaler, speed_model, dir_model, lookback=12):
    # Add sine/cosine encoding for wind direction if not present
    if "wind_dir_sin" not in eval_df.columns and "forecast_wind_direction" in eval_df.columns:
        eval_df["wind_dir_sin"] = np.sin(np.deg2rad(eval_df["forecast_wind_direction"]))
    if "wind_dir_cos" not in eval_df.columns and "forecast_wind_direction" in eval_df.columns:
        eval_df["wind_dir_cos"] = np.cos(np.deg2rad(eval_df["forecast_wind_direction"]))
    feature_cols = [col for col in eval_df.columns if col not in ["timestamp", "forecast_wind_speed", "forecast_wind_direction"]]
    # Drop columns with all NaN or only one unique value
    eval_df = eval_df.dropna(axis=1, how='all')
    nunique = eval_df.nunique()
    feature_cols = [col for col in feature_cols if col in eval_df.columns and nunique[col] > 1 and pd.api.types.is_numeric_dtype(eval_df[col])]
    if not feature_cols:
        raise ValueError(f"No valid feature columns for evaluation. Columns: {list(eval_df.columns)}")
    eval_df_scaled = eval_df.copy()
    # Ensure scaler is fitted for all feature_cols
    missing_cols = [col for col in feature_cols if col not in scaler.feature_names_in_]
    if missing_cols:
        raise ValueError(f"Scaler is missing columns: {missing_cols}. Please retrain the model with consistent features.")
    eval_df_scaled[feature_cols] = scaler.transform(eval_df[feature_cols])
    X = prepare_lstm_input(eval_df_scaled, feature_cols, lookback)
    y_true_speed = eval_df["forecast_wind_speed"].iloc[lookback:].values
    y_true_dir = eval_df["forecast_wind_direction"].iloc[lookback:].values
    y_pred_speed = speed_model.predict(X).flatten()
    y_pred_dir = dir_model.predict(X).flatten()
    # For wind direction, compute angular error
    wind_dir_err = np.abs(((y_pred_dir - y_true_dir + 180) % 360) - 180)
    rmse_speed = np.sqrt(mean_squared_error(y_true_speed, y_pred_speed))
    r2_speed = r2_score(y_true_speed, y_pred_speed)
    rmse_dir = np.sqrt(np.mean(wind_dir_err ** 2))
    r2_dir = r2_score(y_true_dir, y_pred_dir)
    print(f"Wind Speed - RMSE: {rmse_speed:.3f}, R2: {r2_speed:.3f}")
    print(f"Wind Direction - RMSE (deg): {rmse_dir:.3f}, R2: {r2_dir:.3f}")
    return {
        'rmse_speed': rmse_speed, 'r2_speed': r2_speed,
        'rmse_dir': rmse_dir, 'r2_dir': r2_dir,
        'y_true_speed': y_true_speed, 'y_pred_speed': y_pred_speed,
        'y_true_dir': y_true_dir, 'y_pred_dir': y_pred_dir,
        'wind_dir_err': wind_dir_err
    }

if __name__ == "__main__":
    # Example: London
    lat, lon = 51.5074, -0.1278
    eval_df = fetch_meteostat_eval_data(lat, lon, hours_ahead=48)
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    scaler_path = os.path.join(model_dir, "wind_scaler.joblib")
    speed_model_path = os.path.join(model_dir, "wind_speed_lstm.h5")
    dir_model_path = os.path.join(model_dir, "wind_dir_lstm.h5")
    if not (os.path.exists(scaler_path) and os.path.exists(speed_model_path) and os.path.exists(dir_model_path)):
        raise FileNotFoundError(f"Required model/scaler files not found in {model_dir}. Please train the models first.")
    scaler = joblib.load(scaler_path)
    speed_model = load_model(speed_model_path)
    dir_model = load_model(dir_model_path)
    evaluate_lstm_models(eval_df, scaler, speed_model, dir_model, lookback=12)
