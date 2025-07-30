#!/usr/bin/env python3

# --- Safe helpers ---
import numpy as np
from typing import Any, Optional
def _is_nullish(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, str) and val.strip().lower() in ("", "nan", "none", "null", "na"):
        return True
    try:
        if np.isnan(val):
            return True
    except Exception:
        pass
    return False

def safe(val: Any) -> Optional[Any]:
    return None if _is_nullish(val) else val

def safe_float(val: Any, default: float = 0.0) -> float:
    v = safe(val)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default

# -----------------------------------------------------------------------------
# Standard libs
# -----------------------------------------------------------------------------
import os
import sys
import csv
import json
import time
import math
import socket
import logging
import sqlite3
import warnings
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Tuple

# -----------------------------------------------------------------------------
# Third-party
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import requests

# Optional libs (graceful degradation)
try:
    import joblib
except ImportError:  # we'll error when we actually try to load model
    joblib = None

try:
    from astral import LocationInfo
    from astral.sun import sun
except ImportError:
    LocationInfo = None
    sun = None

try:
    from timezonefinder import TimezoneFinder
except ImportError:
    TimezoneFinder = None

try:
    from geopy.geocoders import Nominatim
except ImportError:
    Nominatim = None

import pytz  # usually available; if not, fallbacks below

# -----------------------------------------------------------------------------
# Warning suppression
# -----------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Trying to unpickle estimator")
warnings.filterwarnings("ignore", message="incompatible dtype")  # pandas FutureWarning pattern
try:
    from urllib3.exceptions import InsecureRequestWarning
    warnings.simplefilter('ignore', InsecureRequestWarning)
except Exception:
    pass

# -----------------------------------------------------------------------------
# Import WindResult for type hints (optional only)
# -----------------------------------------------------------------------------
try:
    from wind_calculator import WindResult  # type: ignore
except Exception:
    WindResult = None

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_DIR = "models"
MODEL_FILE = "precip_model.joblib"
SCALER_FILE = "scaler.joblib"
WIND_SPEED_MODEL_FILE = "wind_speed_model.joblib"
WIND_DIRECTION_MODEL_FILE = "wind_direction_model.joblib"

EXPECTED_COORDS = (13.0, 77.625)  # user home (Bengaluru)
DRONE_COORDS = (27.6889981, 86.726715)  # fallback (Lukla)

TCP_HOST_DEFAULT = "127.0.0.1"
TCP_PORT_DEFAULT = 9000
TCP_RETRY_DELAY = 5
PREDICTION_INTERVAL = 60  # s

SQLITE_PATH = "weather_predict.db"

# Wind shared file(s) written by wind_calculator.py
WIND_SHARED_PATH = "wind_latest.json"         # single current
WIND_HISTORY_PATH = "wind_measurements.json"  # optional rolling log

WIND_MAX_AGE_S = 5.0          # how fresh must wind be to use it
_WIND_WARN_INTERVAL_S = 10.0  # throttle missing-wind warnings
_LAST_WIND_WARN_T = 0.0       # updated in get_wind_data()

# Weather sanity / model blending
MODEL_VALID_TEMP_RANGE = (-40.0, 60.0)
MODEL_VALID_PRESS_RANGE = (870.0, 1050.0)
MODEL_VALID_RH_RANGE = (0.0, 100.0)
MODEL_API_BLEND_THRESH_C = 8.0
MODEL_MIN_WEIGHT = 0.05
MODEL_MAX_WEIGHT = 1.0

# Wind prediction confidence
WIND_SPEED_CONF_RANGE = 2.0  # ± m/s for wind speed confidence
WIND_DIR_CONF_RANGE = 20.0   # ± degrees for wind direction confidence

CREATE_PREDICTIONS_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    prediction_time_readable TEXT,
    predicted_temperature REAL,
    temperature_api REAL,
    humidity REAL,
    wind_speed REAL,
    wind_direction REAL,
    confidence_range REAL,
    humidity_timestamp TEXT,
    sunrise_drone TEXT,
    sunset_drone TEXT,
    sunrise_user TEXT,
    sunset_user TEXT,
    info TEXT,
    humidity_source TEXT,
    drone_location TEXT,
    user_location TEXT,
    rain_chance_2h REAL,
    cloud_cover REAL,
    wind_confidence REAL,
    wind_samples INTEGER,
    wind_std_dev_speed REAL,
    wind_std_dev_direction REAL,
    wind_source TEXT,
    wind_speed_2h REAL,
    wind_direction_2h REAL,
    wind_speed_confidence REAL,
    wind_direction_confidence REAL
);
"""

DB_COLUMNS = {
    "timestamp": "TEXT",
    "prediction_time_readable": "TEXT",
    "predicted_temperature": "REAL",
    "temperature_api": "REAL",
    "humidity": "REAL",
    "wind_speed": "REAL",
    "wind_direction": "REAL",
    "confidence_range": "REAL",
    "humidity_timestamp": "TEXT",
    "sunrise_drone": "TEXT",
    "sunset_drone": "TEXT",
    "sunrise_user": "TEXT",
    "sunset_user": "TEXT",
    "info": "TEXT",
    "humidity_source": "TEXT",
    "drone_location": "TEXT",
    "user_location": "TEXT",
    "rain_chance_2h": "REAL",
    "cloud_cover": "REAL",
    "wind_confidence": "REAL",
    "wind_samples": "INTEGER",
    "wind_std_dev_speed": "REAL",
    "wind_std_dev_direction": "REAL",
    "wind_source": "TEXT",
    "wind_speed_2h": "REAL",
    "wind_direction_2h": "REAL",
    "wind_speed_confidence": "REAL",
    "wind_direction_confidence": "REAL",
}

def initialize_database(reset: bool = False) -> None:
    """Create DB / table if absent; add any missing cols; safe to call repeatedly."""
    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.cursor()
    try:
        if reset:
            cur.execute("DROP TABLE IF EXISTS predictions")
        cur.execute(CREATE_PREDICTIONS_SQL)
        cur.execute("PRAGMA table_info(predictions)")
        existing = {row[1] for row in cur.fetchall()}
        for col, ctype in DB_COLUMNS.items():
            if col not in existing:
                cur.execute(f"ALTER TABLE predictions ADD COLUMN {col} {ctype}")
        conn.commit()
    finally:
        cur.close()
        conn.close()

def log_prediction_to_db(result: Dict[str, Any]) -> bool:
    """Persist a prediction row; gracefully handles missing keys. Returns True on success."""
    success = False
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        cur = conn.cursor()
        ts = result.get("timestamp", datetime.now(timezone.utc).isoformat())
        vals = (
            ts,
            str(result.get('prediction_time_readable', '')),
            safe_float(result.get('predicted_temperature')),
            safe_float(result.get('temperature_api (°C)')),
            safe_float(result.get('humidity (% RH)')),
            safe_float(result.get('wind_speed_knots')),
            safe_float(result.get('wind_direction_degrees')),
            safe_float(result.get('confidence_range (±°C)')),
            str(result.get('humidity_timestamp', '')),
            str(result.get('sunrise_at_drone_location', '')),
            str(result.get('sunset_at_drone_location', '')),
            str(result.get('sunrise_at_user_location', '')),
            str(result.get('sunset_at_user_location', '')),
            str(result.get('info', '')),
            str(result.get('humidity_source', '')),
            str(result.get('drone_location', '')),
            str(result.get('user_location', '')),
            safe_float(result.get('rain_chance_2h (%)')),
            safe_float(result.get('cloud_cover (%)')),
            safe_float(result.get('wind_confidence', 0.0)),
            int(safe_float(result.get('wind_samples', 0))),
            safe_float(result.get('wind_std_dev_speed', 0.0)),
            safe_float(result.get('wind_std_dev_direction', 0.0)),
            str(result.get('wind_source', 'external')),
            safe_float(result.get('wind_speed_2h'), None),
            safe_float(result.get('wind_direction_2h'), None),
            safe_float(result.get('wind_speed_confidence'), None),
            safe_float(result.get('wind_direction_confidence'), None),
        )
        cur.execute(
            """
            INSERT INTO predictions (
                timestamp, prediction_time_readable, predicted_temperature, temperature_api, humidity,
                wind_speed, wind_direction, confidence_range,
                humidity_timestamp, sunrise_drone, sunset_drone,
                sunrise_user, sunset_user, info,
                humidity_source, drone_location, user_location,
                rain_chance_2h, cloud_cover, wind_confidence, wind_samples,
                wind_std_dev_speed, wind_std_dev_direction, wind_source,
                wind_speed_2h, wind_direction_2h, wind_speed_confidence, wind_direction_confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            vals,
        )
        conn.commit()
        success = True
    except Exception as e:
        logging.error(f"SQLite Database Error: {e}")
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass
    return success

# -----------------------------------------------------------------------------
# External weather helpers
# -----------------------------------------------------------------------------
def _om_get_json(url: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
    try:
        resp = requests.get(url, timeout=timeout, verify=False)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logging.error(f"HTTP error fetching weather: {e}")
        return None

def get_weather_data(lat: float, lon: float) -> Tuple[float, float, float, str, str]:
    """
    Returns (temp_api, rh, cloud_cover, timestamp, source_url).
    Safe fallbacks used on error.
def _is_nullish(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, str) and val.strip().lower() in ("", "nan", "none", "null", "na"):
        return True
    try:
        if np.isnan(val):
            return True
    except Exception:
        pass
    return False

def safe(val: Any) -> Optional[Any]:
    return None if _is_nullish(val) else val

def safe_float(val: Any, default: float = 0.0) -> float:
    v = safe(val)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default

    """
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}¤t=temperature_2m,relative_humidity_2m,cloudcover"
    )
    data = _om_get_json(url)
    if data:
        cur = data.get("current", {})
        return (
            cur.get("temperature_2m", 15.0),
            cur.get("relative_humidity_2m", 80.0),
            cur.get("cloudcover", 50.0),
            cur.get("time", datetime.now(timezone.utc).isoformat()),
            url,
        )
    return 15.0, 80.0, 50.0, datetime.now(timezone.utc).isoformat(), url

def get_rain_chance_in_2_hours(lat: float, lon: float) -> float:
    now = datetime.now(timezone.utc)
    future = now + timedelta(hours=2)
    date_str = now.strftime("%Y-%m-%d")
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&hourly=precipitation_probability&start_date={date_str}&end_date={date_str}&timezone=UTC"
    )
    data = _om_get_json(url)
    if data:
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        probs = hourly.get("precipitation_probability", [])
        target = future.strftime("%Y-%m-%dT%H:00")
        for t, p in zip(times, probs):
            if t >= target:
                return float(p if p is not None else 50)
    return 50.0

def get_forecast_2h(lat: float, lon: float) -> Dict[str, float]:
    now = datetime.now(timezone.utc)
    future = now + timedelta(hours=2)
    date_str = now.strftime("%Y-%m-%d")
    hour_str = future.strftime("%H:00")
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,cloudcover,precipitation_probability"
        f"&start_date={date_str}&end_date={date_str}&timezone=UTC"
    )
    data = _om_get_json(url)
    if data:
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        clouds = hourly.get("cloudcover", [])
        rains = hourly.get("precipitation_probability", [])
        target = future.strftime("%Y-%m-%dT") + hour_str
        for i, t in enumerate(times):
            if t >= target:
                return {
                    "forecast_temp_2h": float(temps[i]) if i < len(temps) else None,
                    "forecast_cloud_cover_2h": float(clouds[i]) if i < len(clouds) else None,
                    "forecast_rain_chance_2h": float(rains[i]) if i < len(rains) else None,
                }
        if temps and clouds and rains:
            return {
                "forecast_temp_2h": float(temps[-1]),
                "forecast_cloud_cover_2h": float(clouds[-1]),
                "forecast_rain_chance_2h": float(rains[-1]),
            }
    return {
        "forecast_temp_2h": None,
        "forecast_cloud_cover_2h": None,
        "forecast_rain_chance_2h": None,
    }

def format_humidity_time(iso_str: str) -> str:
    try:
        iso_str = iso_str.replace('Z', '+00:00')
        dt = datetime.fromisoformat(iso_str)
        return f"{dt.strftime('%I:%M %p UTC')} on {dt.strftime('%Y-%m-%d')}"
    except Exception:
        return "Unknown time"

# -----------------------------------------------------------------------------
# Timezone / location utils
# -----------------------------------------------------------------------------
if TimezoneFinder:
    tf = TimezoneFinder()
else:
    tf = None

if Nominatim:
    geolocator = Nominatim(user_agent="weather_predictor")
else:
    geolocator = None

def get_timezone(lat: float, lon: float) -> str:
    if tf is None:
        return "Asia/Kolkata" if (lat, lon) == EXPECTED_COORDS else "Asia/Kathmandu"
    try:
        tz = tf.timezone_at(lat=lat, lng=lon)
        if tz:
            return tz
    except Exception:
        pass
    return "Asia/Kolkata" if (lat, lon) == EXPECTED_COORDS else "Asia/Kathmandu"

def get_location_name(lat: float, lon: float) -> str:
    if geolocator is None:
        return "Unknown, India"
    try:
        loc = geolocator.reverse((lat, lon), language='en', timeout=10)
        if not loc:
            return f"Lat: {lat:.3f}, Lon: {lon:.3f}"
        addr = loc.raw.get('address', {})
        city = addr.get('village', addr.get('town', addr.get('city', 'Unknown')))
        country = addr.get('country', 'Unknown')
        return f"{city}, {country}"
    except Exception:
        return f"Lat: {lat:.3f}, Lon: {lon:.3f}"

def get_sunrise_sunset(lat: float, lon: float, tz_str: str) -> Dict[str, str]:
    if LocationInfo is None or sun is None:
        return {"sunrise_readable": "Unknown", "sunset_readable": "Unknown"}
    try:
        city = LocationInfo(latitude=lat, longitude=lon)
        tz = pytz.timezone(tz_str)
        s = sun(city.observer, date=datetime.now(timezone.utc).date(), tzinfo=tz)
        return {
            "sunrise_readable": s['sunrise'].strftime("%I:%M %p %Z"),
            "sunset_readable": s['sunset'].strftime("%I:%M %p %Z"),
        }
    except Exception:
        return {
            "sunrise_readable": "Unknown",
            "sunset_readable": "Unknown"
        }

# -----------------------------------------------------------------------------
# Wind fetch (from wind_calculator.py's wind_latest.json)
# -----------------------------------------------------------------------------
def _parse_wind_payload(obj: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    """
    Return (speed_knots, direction_deg, age_s) if parse ok else None.
    Accepts several key variants for compatibility.
    """
    if not isinstance(obj, dict):
        return None
    ts_raw = obj.get("timestamp") or obj.get("time") or obj.get("ts")
    age_s = None
    if ts_raw:
        try:
            ts = datetime.fromisoformat(str(ts_raw).replace('Z', '+00:00'))
            age_s = (datetime.now(timezone.utc) - ts).total_seconds()
        except Exception:
            age_s = None
    if "wind_speed_knots" in obj:
        spd_kt = safe_float(obj["wind_speed_knots"], 0.0)
    elif "speed_knots" in obj:
        spd_kt = safe_float(obj["speed_knots"], 0.0)
    elif "wind_speed" in obj:  # assume m/s
        spd_kt = safe_float(obj["wind_speed"], 0.0) * 1.94384
    elif "speed_ms" in obj:
        spd_kt = safe_float(obj["speed_ms"], 0.0) * 1.94384
    elif "wind_speed_m_s" in obj:
        spd_kt = safe_float(obj["wind_speed_m_s"], 0.0) * 1.94384
    else:
        spd_kt = 0.0
    if "wind_direction_degrees" in obj:
        dir_deg = safe_float(obj["wind_direction_degrees"], 0.0)
    elif "direction_degrees" in obj:
        dir_deg = safe_float(obj["direction_degrees"], 0.0)
    elif "wind_direction" in obj:
        dir_deg = safe_float(obj["wind_direction"], 0.0)
    else:
        dir_deg = 0.0
    return (spd_kt, dir_deg, age_s if age_s is not None else 0.0)

def _get_latest_wind() -> Optional[Tuple[float, float]]:
    """
    Read most recent wind from WIND_SHARED_PATH.
    Return (speed_knots, direction_deg) if fresh else None.
    """
    if not os.path.exists(WIND_SHARED_PATH):
        return None
    try:
        with open(WIND_SHARED_PATH, "r") as f:
            data = json.load(f)
    except Exception as e:
        logging.debug(f"wind load failed: {e}")
        return None
    obj = data[-1] if isinstance(data, list) and data else data
    parsed = _parse_wind_payload(obj)
    if not parsed:
        return None
    spd_kt, dir_deg, age_s = parsed
    if age_s is not None and age_s > WIND_MAX_AGE_S:
        return None
    return (spd_kt, dir_deg)

# -----------------------------------------------------------------------------
# WeatherPredictor (model wrapper; no internal wind calc)
# -----------------------------------------------------------------------------
class WeatherPredictor:
    def __init__(self, model_dir: str = MODEL_DIR):
        if joblib is None:
            raise ImportError("joblib missing; pip install joblib")
        precip_model_path = os.path.join(model_dir, MODEL_FILE)
        scaler_path = os.path.join(model_dir, SCALER_FILE)
        wind_speed_model_path = os.path.join(model_dir, WIND_SPEED_MODEL_FILE)
        wind_direction_model_path = os.path.join(model_dir, WIND_DIRECTION_MODEL_FILE)
        for path, name in [
            (precip_model_path, "Precipitation model"),
            (scaler_path, "Scaler"),
            (wind_speed_model_path, "Wind speed model"),
            (wind_direction_model_path, "Wind direction model")
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} not found: {path}")
        self.precip_model = joblib.load(precip_model_path)
        self.wind_speed_model = joblib.load(wind_speed_model_path)
        self.wind_direction_model = joblib.load(wind_direction_model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = list(getattr(self.scaler, 'feature_names_in_', [])) or [
            'hour', 'month', 'relative_humidity_2m', 'pressure_msl',
            'wind_speed_10m', 'wind_gusts_10m', 'wind_direction_10m', 'cloudcover'
        ]
        logging.info("WeatherPredictor initialized (wind external).")

    def get_wind_data(self) -> Dict[str, Any]:
        global _LAST_WIND_WARN_T
        latest = _get_latest_wind()
        if latest is None:
            now_t = time.time()
            if now_t - _LAST_WIND_WARN_T > _WIND_WARN_INTERVAL_S:
                logging.warning("wind_calculator.py has not provided recent wind. Using 0 kt. Is it running?")
                _LAST_WIND_WARN_T = now_t
            return {
                "wind_speed_ms": 0.0,
                "wind_speed_knots": 0.0,
                "wind_direction_degrees": 0.0,
                "wind_confidence": None,
                "wind_samples": None,
                "wind_std_dev_speed": None,
                "wind_std_dev_direction": None,
                "wind_source": "none",
            }
        spd_kt, dir_deg = latest
        return {
            "wind_speed_ms": spd_kt * 0.514444,
            "wind_speed_knots": spd_kt,
            "wind_direction_degrees": dir_deg,
            "wind_confidence": None,
            "wind_samples": None,
            "wind_std_dev_speed": None,
            "wind_std_dev_direction": None,
            "wind_source": "external",
        }

    def prepare_features(self, drone_data: Dict[str, Any], rh: float, cloud_cover: float,
                         wind_speed_ms: float, wind_direction: float, api_temp: float) -> pd.DataFrame:
        now = datetime.now(timezone.utc)
        altitude = safe_float(drone_data.get('altitude_from_sealevel'), 0.0)
        base_pressure = safe_float(drone_data.get('pressure_hpa'), 1013.0)
        if altitude > 10:
            adjusted_pressure = base_pressure * (1 - 0.0065 * altitude / 288.15) ** 5.257
        else:
            adjusted_pressure = base_pressure
        wind_gusts = wind_speed_ms * 1.3
        hour_temp_factor = 1.0 + 0.1 * math.sin(2 * math.pi * (now.hour - 6) / 24)
        humidity_temp_factor = 1.0 - (rh - 50) * 0.005
        features = {
            'hour': now.hour,
            'month': now.month,
            'relative_humidity_2m': rh,
            'pressure_msl': adjusted_pressure,
            'wind_speed_10m': wind_speed_ms,
            'wind_gusts_10m': wind_gusts,
            'wind_direction_10m': wind_direction,
            'cloudcover': cloud_cover,
            'hour_temp_factor': hour_temp_factor,
            'humidity_temp_factor': humidity_temp_factor,
            'api_temperature_reference': api_temp,
        }
        # fill any model-required features that we didn't compute
        for name in self.feature_names:
            if name not in features:
                if name == 'altitude':
                    features[name] = altitude
                elif name in ('temperature_2m', 'temp'):
                    features[name] = api_temp
                elif name in ('precipitation', 'rain'):
                    features[name] = 0.0
                else:
                    features[name] = 0.0
        df = pd.DataFrame([features])
        # ensure float dtype to avoid FutureWarning in noise injection
        df = df.astype(float)
        return df

    def predict_raw(self, features: pd.DataFrame) -> Dict[str, float]:
        try:
            if hasattr(self.scaler, 'feature_names_in_'):
                features = features.reindex(columns=self.scaler.feature_names_in_, fill_value=0.0)
            X_scaled = self.scaler.transform(features)
            precip_pred = float(self.precip_model.predict(X_scaled)[0])
            wind_speed_pred = float(self.wind_speed_model.predict(X_scaled)[0])
            wind_dir_pred = float(self.wind_direction_model.predict(X_scaled)[0])
            return {
                'precipitation': precip_pred,
                'wind_speed_2h': wind_speed_pred,
                'wind_direction_2h': wind_dir_pred
            }
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            return {
                'precipitation': float('nan'),
                'wind_speed_2h': float('nan'),
                'wind_direction_2h': float('nan')
            }

# -----------------------------------------------------------------------------
# Prediction fusion / gating
# -----------------------------------------------------------------------------
def _feature_out_of_range(temp_api: float, rh: float, p_msl: float) -> bool:
    if not (MODEL_VALID_TEMP_RANGE[0] <= temp_api <= MODEL_VALID_TEMP_RANGE[1]):
        return True
    if not (MODEL_VALID_RH_RANGE[0] <= rh <= MODEL_VALID_RH_RANGE[1]):
        return True
    if not (MODEL_VALID_PRESS_RANGE[0] <= p_msl <= MODEL_VALID_PRESS_RANGE[1]):
        return True
    return False

def fuse_prediction(model_pred: float, api_temp: float, rh: float, p_msl: float) -> Tuple[float, float]:
    weight = 1.0
    if math.isnan(model_pred):
        weight = 0.0
    if _feature_out_of_range(api_temp, rh, p_msl):
        weight *= 0.5
    if not math.isnan(model_pred):
        diff = abs(model_pred - api_temp)
        if diff > MODEL_API_BLEND_THRESH_C:
            weight *= max(0.0, 1.0 - ((diff - MODEL_API_BLEND_THRESH_C) / 20.0))
        weight = max(MODEL_MIN_WEIGHT, min(MODEL_MAX_WEIGHT, weight))
    fused = weight * model_pred + (1 - weight) * api_temp if not math.isnan(model_pred) else api_temp
    conf = max(0.1, round((1 - weight) * 5.0, 2))
    return fused, conf

# -----------------------------------------------------------------------------
# Global predictor singleton
# -----------------------------------------------------------------------------
_GLOBAL_PREDICTOR: Optional[WeatherPredictor] = None

def get_predictor() -> WeatherPredictor:
    global _GLOBAL_PREDICTOR
    if _GLOBAL_PREDICTOR is None:
        _GLOBAL_PREDICTOR = WeatherPredictor()
    return _GLOBAL_PREDICTOR

# -----------------------------------------------------------------------------
# Core processing of a telemetry record → prediction dict
# -----------------------------------------------------------------------------
def process_live_prediction(
    data: Dict[str, Any],
    predictor: Optional[WeatherPredictor] = None,
) -> Dict[str, Any]:
    if predictor is None:
        predictor = get_predictor()

    # Check required fields
    missing = [f for f in ('latitude', 'longitude') if safe(data.get(f)) is None]
    if missing:
        logging.warning(f"Missing required fields: {missing}")
        return {"error": f"Missing required data fields: {missing}", "input": data}

    lat = safe_float(data.get("latitude", DRONE_COORDS[0]))
    lon = safe_float(data.get("longitude", DRONE_COORDS[1]))
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        logging.warning(f"Invalid coords {lat},{lon}; using fallback")
        lat, lon = DRONE_COORDS

    # External weather
    temp_api, rh, cloud_cover_percent, rh_time, rh_url = get_weather_data(lat, lon)
    rain_chance_2h = get_rain_chance_in_2_hours(lat, lon)

    # Pressure at drone altitude
    p_station = safe_float(data.get('pressure_hpa'), 1013.0)
    alt = safe_float(data.get('altitude_from_sealevel'), 0.0)
    p_msl = p_station * (1 - 0.0065 * alt / 288.15) ** -5.257 if alt > 10 else p_station

    # Wind data
    wd = predictor.get_wind_data()
    wind_speed_ms = wd["wind_speed_ms"]
    wind_speed_knots = wd["wind_speed_knots"]
    wind_direction_degrees = wd["wind_direction_degrees"]

    # Features → model
    features = predictor.prepare_features(
        data, rh=rh, cloud_cover=cloud_cover_percent,
        wind_speed_ms=wind_speed_ms, wind_direction=wind_direction_degrees,
        api_temp=temp_api
    )
    model_preds = predictor.predict_raw(features)
    precip_pred = model_preds['precipitation']
    wind_speed_2h = model_preds['wind_speed_2h']
    wind_direction_2h = model_preds['wind_direction_2h']
    # Note: No temperature prediction in model_preds, using API temp as fallback
    fused_temp, ci = fuse_prediction(temp_api, temp_api, rh, p_msl)  # No model temp prediction

    # Monte Carlo-ish CI augmentation for precipitation and wind
    try:
        feats_np = features.to_numpy(dtype=float)
        precip_preds = []
        speed_preds = []
        dir_preds = []
        for _ in range(10):
            noise = np.random.normal(0, 0.02, size=feats_np.shape[1])
            noisy = feats_np[0] * (1 + noise)
            noisy_df = pd.DataFrame([noisy], columns=features.columns)
            preds = predictor.predict_raw(noisy_df)
            if not math.isnan(preds['precipitation']):
                precip_preds.append(preds['precipitation'])
            if not math.isnan(preds['wind_speed_2h']):
                speed_preds.append(preds['wind_speed_2h'])
            if not math.isnan(preds['wind_direction_2h']):
                dir_preds.append(preds['wind_direction_2h'])
        ci_precip = WIND_SPEED_CONF_RANGE
        ci_dir = WIND_DIR_CONF_RANGE
        if len(precip_preds) >= 2:
            sd_precip = float(np.std(precip_preds))
            ci = max(ci, round(1.96 * sd_precip, 2))
        if len(speed_preds) >= 2:
            sd_speed = float(np.std(speed_preds))
            ci_precip = max(WIND_SPEED_CONF_RANGE, round(1.96 * sd_speed, 2))
        if len(dir_preds) >= 2:
            sd_dir = float(np.std(dir_preds))
            ci_dir = max(WIND_DIR_CONF_RANGE, round(1.96 * sd_dir, 2))
    except Exception as e:
        logging.debug(f"Noise CI calc failed: {e}")
        ci_precip = WIND_SPEED_CONF_RANGE
        ci_dir = WIND_DIR_CONF_RANGE

    # Location & astro
    humidity_time_fmt = format_humidity_time(rh_time)
    drone_tz = get_timezone(lat, lon)
    user_tz = get_timezone(*EXPECTED_COORDS)
    drone_loc = get_location_name(lat, lon)
    user_loc = get_location_name(*EXPECTED_COORDS)
    drone_sun = get_sunrise_sunset(lat, lon, drone_tz)
    user_sun = get_sunrise_sunset(*EXPECTED_COORDS, user_tz)
    ts_in = safe(data.get("timestamp"))
    ts_out = ts_in if isinstance(ts_in, str) and ts_in else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    prediction_time_readable = format_humidity_time(ts_out)

    # Forecast
    forecast = get_forecast_2h(lat, lon)

    # Round wind values for display
    wind_speed_2h_rounded = round(wind_speed_2h, 2) if not math.isnan(wind_speed_2h) else None
    wind_dir_2h_rounded = round(wind_direction_2h, 1) if not math.isnan(wind_direction_2h) else None

    # Friendly summary
    print(
        f"🌬️ Wind {wind_speed_knots:.2f} kt @ {wind_direction_degrees:.1f}° | "
        f"Now {fused_temp:.1f}°C | +2h Wind {wind_speed_2h_rounded or 0.0:.1f} m/s @ {wind_dir_2h_rounded or 0.0:.1f}°"
    )

    # Full JSON output
    output = {
        "timestamp": ts_out,
        "prediction_time_readable": prediction_time_readable,
        "rain_chance_2h (%)": round(float(precip_pred), 2) if not math.isnan(precip_pred) else float(rain_chance_2h),
        "cloud_cover (%)": float(cloud_cover_percent),
        "temperature_api (°C)": round(temp_api, 1),
        "predicted_temperature": round(fused_temp, 2),
        "confidence_range (±°C)": ci,
        "humidity (% RH)": round(rh, 0),
        "humidity_timestamp": humidity_time_fmt,
        "sunrise_at_drone_location": drone_sun['sunrise_readable'],
        "sunset_at_drone_location": drone_sun['sunset_readable'],
        "sunrise_at_user_location": user_sun['sunrise_readable'],
        "sunset_at_user_location": user_sun['sunset_readable'],
        "info": f"Precip raw={precip_pred:.2f}% | API={temp_api:.2f}°C | fused={fused_temp:.2f}°C | 95% CI ±{ci}°C",
        "humidity_source": rh_url,
        "drone_location": drone_loc,
        "user_location": user_loc,
        "wind_speed_knots": wind_speed_knots,
        "wind_direction_degrees": wind_direction_degrees,
        "wind_speed_2h": wind_speed_2h_rounded,
        "wind_direction_2h": wind_dir_2h_rounded,
        "wind_speed_confidence": ci_precip,
        "wind_direction_confidence": ci_dir,
        "forecast_temp_2h": forecast["forecast_temp_2h"],
        "forecast_cloud_cover_2h": forecast["forecast_cloud_cover_2h"],
        "forecast_rain_chance_2h": forecast["forecast_rain_chance_2h"],
        "input": data,
    }
    return output

# -----------------------------------------------------------------------------
# TCP client loop (live streaming)
# -----------------------------------------------------------------------------
def run_tcp_client(host: str = TCP_HOST_DEFAULT, port: int = TCP_PORT_DEFAULT,
                   retry_delay: int = TCP_RETRY_DELAY, period: int = PREDICTION_INTERVAL):
    predictor = get_predictor()
    logging.warning(f"Connecting to TCP server {host}:{port}")
    last_pred_time = 0.0
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(10)
                s.connect((host, port))
                logging.info("Connected to TCP server.")
                buffer = ''
                while True:
                    try:
                        chunk = s.recv(4096)
                        if not chunk:
                            logging.warning("Connection closed by server. Reconnecting...")
                            break
                        buffer += chunk.decode('utf-8', errors='ignore')
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                drone_data = json.loads(line)
                            except json.JSONDecodeError as e:
                                logging.error(f"JSON decode error: {e} line[:100]={line[:100]}")
                                continue
                            now = time.time()
                            if period <= 0 or (now - last_pred_time) >= period:
                                result = process_live_prediction(drone_data, predictor=predictor)
                                if "error" not in result:
                                    print(json.dumps(result, indent=2, ensure_ascii=False))
                                    if log_prediction_to_db(result):
                                        logging.debug("DB row inserted.")
                                    last_pred_time = now
                                else:
                                    logging.error(result["error"])
                    except socket.timeout:
                        continue
        except ConnectionRefusedError as e:
            logging.error(f"Connection failed: {e}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
        except Exception as e:
            logging.error(f"Unexpected error: {e}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)

# -----------------------------------------------------------------------------
# CSV one-shot mode
# -----------------------------------------------------------------------------
def predict_from_csv(csv_path: str) -> None:
    predictor = get_predictor()
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found: {csv_path}")
        return
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"CSV read error: {e}")
        return
    if df.empty:
        logging.error("CSV file is empty")
        return
    # iterate backward for last valid row
    for idx in range(len(df) - 1, -1, -1):
        row = df.iloc[idx]
        data = {}
        for col in df.columns:
            v = row[col]
            if pd.notna(v):
                data[col] = v
        if not data:
            continue
        data.setdefault('timestamp', datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"))
        data.setdefault('latitude', DRONE_COORDS[0])
        data.setdefault('longitude', DRONE_COORDS[1])
        result = process_live_prediction(data, predictor=predictor)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if "error" not in result:
            log_prediction_to_db(result)
        break
    else:
        logging.error("No valid rows found in CSV file")

# -----------------------------------------------------------------------------
# CLI entry
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Weather prediction TCP client or CSV mode.")
    parser.add_argument('--tcp', action='store_true', help='Run as TCP client')
    parser.add_argument('--csv', type=str, help='Path to CSV for prediction')
    parser.add_argument('--reset-db', action='store_true', help='Drop & recreate predictions table (DATA LOSS)')
    parser.add_argument('--host', type=str, default=TCP_HOST_DEFAULT, help='TCP host')
    parser.add_argument('--port', type=int, default=TCP_PORT_DEFAULT, help='TCP port')
    parser.add_argument('--period', type=int, default=PREDICTION_INTERVAL, help='Seconds between predictions (0=every row)')
    parser.add_argument('--wind-path', type=str, default=WIND_SHARED_PATH,
                        help='Path to JSON file written by wind_calculator.py')
    parser.add_argument('--wind-max-age', type=float, default=WIND_MAX_AGE_S,
                        help='Max age (s) of wind data before falling back to 0 kt')
    args = parser.parse_args()

    # apply CLI overrides
    if args.wind_path != WIND_SHARED_PATH:
        WIND_SHARED_PATH = args.wind_path  # type: ignore[assignment]
    if args.wind_max_age != WIND_MAX_AGE_S:
        WIND_MAX_AGE_S = args.wind_max_age  # type: ignore[assignment]

    # --- DB status print ---
    pre_exists = os.path.exists(SQLITE_PATH)
    initialize_database(reset=args.reset_db)
    if args.reset_db:
        logging.info(f"SQLite DB reset & recreated: {SQLITE_PATH}")
        print(f"[DB] reset & recreated: {SQLITE_PATH}")
    else:
        if pre_exists:
            logging.info(f"SQLite DB exists: {SQLITE_PATH}")
            print(f"[DB] using existing: {SQLITE_PATH}")
        else:
            logging.info(f"SQLite DB created: {SQLITE_PATH}")
            print(f"[DB] created new: {SQLITE_PATH}")

    try:
        if args.tcp:
            run_tcp_client(host=args.host, port=args.port, period=args.period)
        elif args.csv:
            predict_from_csv(args.csv)
        else:
            logging.error("Please use --tcp or --csv.")
            sys.exit(1)
    except KeyboardInterrupt:
        logging.info("Script interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)