import os
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import requests
from datetime import datetime, timedelta

def fetch_openmeteo_eval_data(lat, lon, hours_ahead=24):
    now = datetime.utcnow()
    end_time = now + timedelta(hours=hours_ahead)
    start_date = now.strftime("%Y-%m-%d")
    end_date = end_time.strftime("%Y-%m-%d")
    url = (
        f"https://api.open-meteo.com/v1/ecmwf?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=windspeed_10m,winddirection_10m,temperature_2m,relative_humidity_2m,pressure_msl,cloudcover,precipitation,weathercode"
        f"&start_date={start_date}&end_date={end_date}&timezone=UTC"
    )
    resp = requests.get(url, timeout=10)
    data = resp.json()
    if "hourly" not in data:
        raise ValueError("Unexpected forecast response format.")
    time_series = data["hourly"]["time"]
    speeds = data["hourly"]["windspeed_10m"]
    directions = data["hourly"]["winddirection_10m"]
    temp = data["hourly"].get("temperature_2m", [None]*len(time_series))
    rh = data["hourly"].get("relative_humidity_2m", [None]*len(time_series))
    pressure = data["hourly"].get("pressure_msl", [None]*len(time_series))
    cloud = data["hourly"].get("cloudcover", [None]*len(time_series))
    precip = data["hourly"].get("precipitation", [None]*len(time_series))
    wcode = data["hourly"].get("weathercode", [None]*len(time_series))
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(time_series),
        "forecast_wind_speed": speeds,
        "forecast_wind_direction": directions,
        "temperature_2m": temp,
        "relative_humidity_2m": rh,
        "pressure_msl": pressure,
        "cloudcover": cloud,
        "precipitation": precip,
        "weathercode": wcode
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
    eval_df = fetch_openmeteo_eval_data(lat, lon, hours_ahead=48)
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
