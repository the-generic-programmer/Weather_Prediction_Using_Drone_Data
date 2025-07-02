import numpy as np
import pandas as pd
from datetime import datetime
from math import sin, cos, radians
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def compute_solar_features(timestamp: pd.Timestamp):
    hour = timestamp.hour + timestamp.minute / 60.0
    solar_angle = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, 1)
    return {
        "hour": hour,
        "solar_angle": solar_angle
    }

def compute_directional_features(wind_direction_deg):
    angle_rad = radians(wind_direction_deg)
    return {
        "wind_dir_sin": sin(angle_rad),
        "wind_dir_cos": cos(angle_rad)
    }

def prepare_features(drone_estimate, forecast_df, lat, lon):
    try:
        features = []
        for _, row in forecast_df.iterrows():
            ts = row["timestamp"]
            solar_feats = compute_solar_features(ts)
            wind_dir_feats = compute_directional_features(row["wind_direction"])
            feature_row = {
                "timestamp": ts,
                "lat": lat,
                "lon": lon,
                "forecast_wind_speed": row["wind_speed"],
                "forecast_wind_direction": row["wind_direction"],
                "drone_wind_speed": drone_estimate["wind_speed"],
                "drone_wind_direction": drone_estimate["wind_direction"],
                "ground_speed": drone_estimate["ground_speed"],
                "track_angle": drone_estimate["track_angle"],
                "drift_angle": drone_estimate.get("drift_angle", 0),
                **solar_feats,
                **wind_dir_feats
            }
            features.append(feature_row)
        return pd.DataFrame(features)
    except Exception as e:
        logging.error(f"Feature engineering error: {e}")
        return pd.DataFrame()