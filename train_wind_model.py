import pandas as pd
import numpy as np
import os
import joblib
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def evaluate_model(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    logging.info(f"{label} RMSE: {rmse:.3f}")
    logging.info(f"{label} R2: {r2:.3f}")
    return rmse, r2

def train_and_save_model(df: pd.DataFrame):
    try:
        df = df.drop(columns=[col for col in ["timestamp", "lat", "lon"] if col in df.columns])
        for col in ["forecast_wind_speed", "forecast_wind_direction"]:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        X = df.drop(columns=["forecast_wind_speed", "forecast_wind_direction"])
        y_speed = df["forecast_wind_speed"]
        y_dir = df["forecast_wind_direction"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train_s, y_test_s = train_test_split(X_scaled, y_speed, test_size=0.2, random_state=42)
        _, _, y_train_d, y_test_d = train_test_split(X_scaled, y_dir, test_size=0.2, random_state=42)
        model_speed = RandomForestRegressor(n_estimators=100, random_state=42)
        model_dir = RandomForestRegressor(n_estimators=100, random_state=42)
        model_speed.fit(X_train, y_train_s)
        model_dir.fit(X_train, y_train_d)
        y_pred_s = model_speed.predict(X_test)
        y_pred_d = model_dir.predict(X_test)
        evaluate_model(y_test_s, y_pred_s, label="Wind Speed")
        evaluate_model(y_test_d, y_pred_d, label="Wind Direction")
        joblib.dump(model_speed, os.path.join(MODEL_DIR, "wind_speed_model.joblib"))
        joblib.dump(model_dir, os.path.join(MODEL_DIR, "wind_dir_model.joblib"))
        joblib.dump(scaler, os.path.join(MODEL_DIR, "wind_scaler.joblib"))
        logging.info("Models and scaler saved.")
    except Exception as e:
        logging.error(f"Training failed: {e}")

if __name__ == "__main__":
    data_path = "data/wind_training_data.csv"
    if not os.path.exists(data_path):
        logging.error(f"{data_path} not found.")
        exit(1)
    df = pd.read_csv(data_path)
    train_and_save_model(df)