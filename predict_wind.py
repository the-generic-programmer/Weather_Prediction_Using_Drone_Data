import joblib
import socket
import pandas as pd
import time
from datetime import datetime
import logging
from feature_engineer import prepare_features
from wind_estimator import estimate_wind_from_drift
from fetch_forecast import fetch_ecmwf_forecast

MODEL_DIR = "models"
OUTPUT_CSV = "wind_forecast_output.csv"
TCP_PORT = 9500

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def load_models():
    speed_model = joblib.load(f"{MODEL_DIR}/wind_speed_model.joblib")
    dir_model = joblib.load(f"{MODEL_DIR}/wind_dir_model.joblib")
    scaler = joblib.load(f"{MODEL_DIR}/wind_scaler.joblib")
    return speed_model, dir_model, scaler

def predict_forecast(drone_data: dict, lat: float, lon: float, hours: int = 12):
    wind_est = estimate_wind_from_drift(
        north_m_s=drone_data["north_m_s"],
        east_m_s=drone_data["east_m_s"],
        heading_deg=drone_data.get("heading_deg")
    )

    forecast_df = fetch_ecmwf_forecast(lat, lon, hours_ahead=hours)
    if forecast_df.empty:
        raise RuntimeError("Forecast data not available")

    features_df = prepare_features(wind_est, forecast_df, lat, lon)
    speed_model, dir_model, scaler = load_models()
    X = features_df.drop(columns=["timestamp", "lat", "lon"])
    X_scaled = scaler.transform(X)

    features_df["predicted_wind_speed"] = speed_model.predict(X_scaled)
    features_df["predicted_wind_direction"] = dir_model.predict(X_scaled)

    features_df[[
        "timestamp", "lat", "lon",
        "predicted_wind_speed", "predicted_wind_direction"
    ]].to_csv(OUTPUT_CSV, index=False)

    return features_df

def start_tcp_broadcast(forecast_df, host='0.0.0.0', port=TCP_PORT, interval=30):
    logging.info(f"Broadcasting wind forecast on TCP {host}:{port} ...")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(1)
    conn, addr = server.accept()
    logging.info(f"Client connected: {addr}")

    try:
        for _, row in forecast_df.iterrows():
            msg = {
                "timestamp": row["timestamp"].isoformat() if hasattr(row["timestamp"], 'isoformat') else str(row["timestamp"]),
                "wind_speed": round(row["predicted_wind_speed"], 2),
                "wind_direction": round(row["predicted_wind_direction"], 1),
                "lat": row["lat"],
                "lon": row["lon"]
            }
            conn.sendall((str(msg) + "\n").encode('utf-8'))
            time.sleep(interval)
    except BrokenPipeError:
        logging.warning("Client disconnected")
    finally:
        conn.close()
        server.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Wind Forecasting and TCP Broadcast")
    parser.add_argument('--north', type=float, required=True)
    parser.add_argument('--east', type=float, required=True)
    parser.add_argument('--lat', type=float, default=13.02)
    parser.add_argument('--lon', type=float, default=77.60)
    parser.add_argument('--heading', type=float, default=None)
    args = parser.parse_args()

    drone_input = {
        "north_m_s": args.north,
        "east_m_s": args.east,
        "heading_deg": args.heading
    }

    df = predict_forecast(drone_input, args.lat, args.lon)
    logging.info("Wind forecast saved to CSV.")
    start_tcp_broadcast(df)