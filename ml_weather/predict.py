#!/usr/bin/env python3
"""
Enhanced Weather Prediction Pipeline (Wind Fetched from wind_calculator.py)
===========================================================================

- Reads *latest processed* wind (speed + direction) written by wind_calculator.py.
- Predicts near-surface ambient temperature using ML model + API blending.
- Works with live telemetry streamed from `MAVSDK_logger.py` over TCP (JSON lines).
- Falls back gracefully (0 kt) if no recent wind is available; warns user.
- Logs predictions to SQLite DB; supports one‑shot CSV mode.

IMPORTANT: wind_calculator.py must regularly write a JSON file (see WIND_SHARED_PATH)
with at least: {"timestamp": "...", "wind_speed_knots": <float>, "wind_direction_degrees": <float>}
"""

import os
import sys
import csv
import json
import time
import socket
import logging
import sqlite3
import math
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Optional/3rd-party imports with graceful degradation
# ---------------------------------------------------------------------------
try:
    import joblib
except ImportError:
    joblib = None
    logging.warning("joblib not found. Install in venv: pip install joblib")

try:
    from astral import LocationInfo
    from astral.sun import sun
except ImportError:
    LocationInfo = None
    sun = None
    logging.warning("astral not installed; sunrise/sunset disabled")

try:
    from timezonefinder import TimezoneFinder
except ImportError:
    TimezoneFinder = None
    logging.warning("timezonefinder not installed; falling back to defaults")

try:
    from geopy.geocoders import Nominatim
except ImportError:
    Nominatim = None
    logging.warning("geopy not installed; location names will be lat/lon")

import pytz

# ---------------------------------------------------------------------------
# Wind Calculator types (for type hints only; we do NOT recompute wind here)
# ---------------------------------------------------------------------------
# We import WindResult dataclass for convenience; we don't instantiate EnhancedWindCalculator.
try:
    from wind_calculator import WindResult  # type: ignore
except Exception:
    WindResult = None  # fallback; not required to run

# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------
MODEL_DIR = "models"
EXPECTED_COORDS = (13.0, 77.625)  # User: Bengaluru
DRONE_COORDS = (27.6889981, 86.726715)  # Lukla fallback
TCP_HOST_DEFAULT = "127.0.0.1"
TCP_PORT_DEFAULT = 9000
TCP_RETRY_DELAY = 5
PREDICTION_INTERVAL = 60  # seconds; CLI overridable
SQLITE_PATH = "weather_predict.db"

# Shared wind file produced by wind_calculator.py (update path if needed)
WIND_SHARED_PATH = "wind_latest.json"        # single most recent measurement
WIND_HISTORY_PATH = "wind_measurements.json" # optional rolling log (ignored here)

# How fresh must wind be?
WIND_MAX_AGE_S = 5.0  # seconds
_WIND_WARN_INTERVAL_S = 10.0  # throttle warnings about missing wind
_LAST_WIND_WARN_T = 0.0

# Sanity / blending thresholds
MODEL_VALID_TEMP_RANGE = (-40.0, 60.0)
MODEL_VALID_PRESS_RANGE = (870.0, 1050.0)
MODEL_VALID_RH_RANGE = (0.0, 100.0)
MODEL_API_BLEND_THRESH_C = 8.0
MODEL_MIN_WEIGHT = 0.05
MODEL_MAX_WEIGHT = 1.0

# -----------------------------------------------------------------------------
# Safe helpers
# -----------------------------------------------------------------------------
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
# DB schema management
# -----------------------------------------------------------------------------
CREATE_PREDICTIONS_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
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
    wind_source TEXT
);
"""

DB_COLUMNS = {
    "timestamp": "TEXT", "predicted_temperature": "REAL", "temperature_api": "REAL",
    "humidity": "REAL", "wind_speed": "REAL", "wind_direction": "REAL",
    "confidence_range": "REAL", "humidity_timestamp": "TEXT", "sunrise_drone": "TEXT",
    "sunset_drone": "TEXT", "sunrise_user": "TEXT", "sunset_user": "TEXT",
    "info": "TEXT", "humidity_source": "TEXT", "drone_location": "TEXT",
    "user_location": "TEXT", "rain_chance_2h": "REAL", "cloud_cover": "REAL",
    "wind_confidence": "REAL", "wind_samples": "INTEGER",
    "wind_std_dev_speed": "REAL", "wind_std_dev_direction": "REAL", "wind_source": "TEXT",
}

def initialize_database(reset: bool = False) -> None:
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

def log_prediction_to_db(result: Dict[str, Any]) -> None:
    """Persist prediction row; missing wind meta fields are filled with 0/None."""
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        cur = conn.cursor()
        ts = result.get("timestamp", datetime.now(timezone.utc).isoformat())
        vals = (
            ts,
            safe_float(result.get('predicted_temperature')),
            safe_float(result.get('temperature_api (°C)')),
            safe_float(result.get('humidity (% RH)')),
            safe_float(result.get('wind_speed_knots')),        # stored as 'wind_speed' in DB
            safe_float(result.get('wind_direction_degrees')),  # stored as 'wind_direction' in DB
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
            safe_float(result.get('wind_samples', 0)),
            safe_float(result.get('wind_std_dev_speed', 0.0)),
            safe_float(result.get('wind_std_dev_direction', 0.0)),
            str(result.get('wind_source', 'external')),
        )
        cur.execute(
            """
            INSERT INTO predictions (
                timestamp, predicted_temperature, temperature_api, humidity,
                wind_speed, wind_direction, confidence_range,
                humidity_timestamp, sunrise_drone, sunset_drone,
                sunrise_user, sunset_user, info,
                humidity_source, drone_location, user_location,
                rain_chance_2h, cloud_cover, wind_confidence, wind_samples,
                wind_std_dev_speed, wind_std_dev_direction, wind_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            vals,
        )
        conn.commit()
    except Exception as e:
        logging.error(f"SQLite Database Error: {e}")
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass

# -----------------------------------------------------------------------------
# External weather helpers
# -----------------------------------------------------------------------------
def get_weather_data(lat: float, lon: float) -> Tuple[float, float, float, str, str]:
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,cloudcover"
    )
    try:
        resp = requests.get(url, timeout=10, verify=False)
        resp.raise_for_status()
        data = resp.json()
        cur = data.get("current", {})
        return (
            cur.get("temperature_2m", 15.0),
            cur.get("relative_humidity_2m", 80.0),
            cur.get("cloudcover", 50.0),
            cur.get("time", datetime.now(timezone.utc).isoformat()),
            url,
        )
    except Exception as e:
        logging.error(f"HTTP error fetching weather: {e}")
        return 15.0, 80.0, 50.0, datetime.now(timezone.utc).isoformat(), "API_ERROR"

def get_rain_chance_in_2_hours(lat: float, lon: float) -> float:
    now = datetime.now(timezone.utc)
    future = now + timedelta(hours=2)
    date_str = now.strftime("%Y-%m-%d")
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&hourly=precipitation_probability&start_date={date_str}&end_date={date_str}&timezone=UTC"
    )
    try:
        resp = requests.get(url, timeout=10, verify=False)
        resp.raise_for_status()
        data = resp.json().get("hourly", {})
        times = data.get("time", [])
        probs = data.get("precipitation_probability", [])
        target = future.strftime("%Y-%m-%dT%H:00")
        for t, p in zip(times, probs):
            if t >= target:
                return float(p if p is not None else 50)
        return 50.0
    except Exception as e:
        logging.error(f"HTTP error fetching rain chance: {e}")
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
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("hourly", {})
        times = data.get("time", [])
        temps = data.get("temperature_2m", [])
        clouds = data.get("cloudcover", [])
        rains = data.get("precipitation_probability", [])
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
    except Exception as e:
        logging.error(f"Forecast API error: {e}")
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
        return f"Lat: {lat:.3f}, Lon: {lon:.3f}"
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
        return {"sunrise_readable": "Unknown", "sunset_readable": "Unknown"}

# -----------------------------------------------------------------------------
# Wind fetch (from wind_calculator.py shared file)
# -----------------------------------------------------------------------------
def _parse_wind_payload(obj: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    """
    Returns (speed_knots, direction_deg, age_s) or None.
    Accepts several key variants.
    """
    if not isinstance(obj, dict):
        return None
    # Timestamp
    ts_raw = obj.get("timestamp") or obj.get("time") or obj.get("ts")
    age_s = None
    if ts_raw:
        try:
            ts = datetime.fromisoformat(str(ts_raw).replace('Z', '+00:00'))
            age_s = (datetime.now(timezone.utc) - ts).total_seconds()
        except Exception:
            age_s = None
    # Speed
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
    # Direction
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
    Read the most recent processed wind from WIND_SHARED_PATH.
    Returns (speed_knots, direction_deg) or None if stale/unavailable.
    """
    if not os.path.exists(WIND_SHARED_PATH):
        return None
    try:
        with open(WIND_SHARED_PATH, "r") as f:
            data = json.load(f)
    except Exception as e:
        logging.debug(f"wind load failed: {e}")
        return None

    if isinstance(data, list) and data:
        obj = data[-1]
    else:
        obj = data

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
            raise ImportError("joblib missing; install in venv: pip install joblib")
        model_path = os.path.join(model_dir, 'weather_model.joblib')
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = list(getattr(self.scaler, 'feature_names_in_', [])) or [
            'hour', 'month', 'relative_humidity_2m', 'pressure_msl',
            'wind_speed_10m', 'wind_gusts_10m', 'wind_direction_10m', 'cloudcover'
        ]
        logging.info("WeatherPredictor initialized (wind external).")

    # We keep an interface so caller can ask for wind; we just read the shared file.
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
        altitude = safe_float(drone_data.get('altitude_from_sealevel', 0))
        base_pressure = safe_float(drone_data.get('pressure_hpa', 1013))
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
        return pd.DataFrame([features])

    def predict_raw(self, features: pd.DataFrame) -> float:
        try:
            if hasattr(self.scaler, 'feature_names_in_'):
                features = features.reindex(columns=self.scaler.feature_names_in_, fill_value=0.0)
            X_scaled = self.scaler.transform(features)
            return float(self.model.predict(X_scaled)[0])
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            return float('nan')

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

    # Validate required fields
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

    # Wind data (external)
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
    model_pred = predictor.predict_raw(features)
    fused_temp, ci = fuse_prediction(model_pred, temp_api, rh, p_msl)

    # Monte Carlo-ish confidence refinement
    try:
        preds = []
        for _ in range(10):
            noise = np.random.normal(0, 0.02, size=features.shape[1])  # 2% noise per feature
            noisy_df = features.copy()
            noisy_df.iloc[0] = noisy_df.iloc[0] * (1 + noise)
            preds.append(predictor.predict_raw(noisy_df))
        preds = [p for p in preds if not math.isnan(p)]
        if len(preds) >= 2:
            sd = float(np.std(preds))
            ci_mc = 1.96 * sd
            ci = max(ci, round(ci_mc, 2))
    except Exception as e:
        logging.debug(f"Noise CI calc failed: {e}")

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

    # Forecast details
    forecast = get_forecast_2h(lat, lon)

    # Friendly summary line (your requested format)
    print(
        f"🌬️ Wind {wind_speed_knots:.2f} kt @ {wind_direction_degrees:.1f}° | "
        f"Now {fused_temp:.1f}°C | +2h {forecast['forecast_temp_2h']}°C"
    )

    # Full JSON output
    output = {
        "timestamp": ts_out,
        "rain_chance_2h (%)": float(rain_chance_2h),
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
        "info": f"Model raw={model_pred:.2f}°C | API={temp_api:.2f}°C | fused={fused_temp:.2f}°C | 95% CI ±{ci}°C",
        "humidity_source": rh_url,
        "drone_location": drone_loc,
        "user_location": user_loc,
        "wind_speed_knots": wind_speed_knots,
        "wind_direction_degrees": wind_direction_degrees,
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
                logging.warning("Connected to TCP server.")
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
                                    log_prediction_to_db(result)
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
# CSV one‑shot mode
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

    # Allow CLI override of wind file / age
    if args.wind_path != WIND_SHARED_PATH:
        globals()['WIND_SHARED_PATH'] = args.wind_path
    if args.wind_max_age != WIND_MAX_AGE_S:
        globals()['WIND_MAX_AGE_S'] = args.wind_max_age

    try:
        initialize_database(reset=args.reset_db)
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
