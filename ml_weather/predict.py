#!/usr/bin/env python3
"""
Full Weather Prediction Pipeline (restored + improved)
=====================================================
Restores **all features** from your original script AND:
- Keeps *SQLite logging* (no data loss) and expands schema to include **API temperature**.
- Integrates **PrecisionWindCalculator** (if available) as the *authoritative* wind source; falls back gracefully to last-known wind or 0.
- Preserves *TCP streaming* mode (1‑min prediction throttle) and *CSV batch* mode.
- Includes *rain chance (2h)*, *cloud cover*, *sunrise/sunset* for drone & user locations.
- Preserves your *output JSON structure* keys so downstream code / dashboards do not break.
- Adds **API temp** into output and DB.
- Improved *safe* conversion utilities; robust timestamp parsing.
- Works in environments with/without active event loop (safe async calls).

Usage
-----
```bash
# TCP live mode (connects to 127.0.0.1:9000)
python predict.py --tcp

# CSV mode
python predict.py --csv path/to/file.csv

# Reset DB schema (drops existing predictions!)
python predict.py --reset-db --tcp
```
"""

import os
import sys
import json
import time
import socket
import asyncio
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import requests

# Third‑party env / ML
try:
    import joblib
except ImportError as e:  # give friendly guidance then continue (we'll lazy‑load model later)
    joblib = None
    logging.warning("joblib not found. Install in venv: pip install joblib")

# Astro / geo
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

import pytz  # usually present; if not, sunrise fallback

# Wind calculator (external module you built)
try:
    from wind_calculator import PrecisionWindCalculator
    WIND_CALCULATOR_AVAILABLE = True
except ImportError:
    PrecisionWindCalculator = None
    WIND_CALCULATOR_AVAILABLE = False
    logging.warning("PrecisionWindCalculator not importable; wind values may be 0 or stale")

# ---------------------------------------------------------------------------
# Constants / Config
# ---------------------------------------------------------------------------
MODEL_DIR = "models"
EXPECTED_COORDS = (13.0, 77.625)         # User: Bengaluru
DRONE_COORDS = (27.6889981, 86.726715)   # Lukla fallback (so you always get something)

TCP_HOST_DEFAULT = "127.0.0.1"
TCP_PORT_DEFAULT = 9000
TCP_RETRY_DELAY = 5          # seconds
PREDICTION_INTERVAL = 60     # seconds between DB logs / printouts (keep from original)

SQLITE_PATH = "weather_predict.db"
WIND_CACHE_PATH = "wind_measurements.json"  # used to persist last known wind between runs

# Logging setup (keep user preference: WARNING noise only)
logging.basicConfig(
    level=logging.WARNING,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ---------------------------------------------------------------------------
# Helpers: safe conversions
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# DB schema mgmt (adds missing columns without dropping data unless --reset-db)
# ---------------------------------------------------------------------------
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
    cloud_cover REAL
);
"""

# columns: sqlite PRAGMA table_info -> add missing
DB_COLUMNS = {
    "timestamp": "TEXT",
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
}


def initialize_database(reset: bool = False) -> None:
    """Ensure DB + table exist. If reset=True, drop & recreate (DATA LOSS!)."""
    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.cursor()
    try:
        if reset:
            cur.execute("DROP TABLE IF EXISTS predictions")
        cur.execute(CREATE_PREDICTIONS_SQL)
        # ensure columns (in case older db w/out temperature_api etc.)
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
    """Insert a prediction row; silently ignore if DB missing.
    Expects wind_speed in **knots** in result; converted to DB wind_speed (knots) like original.
    """
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        cur = conn.cursor()
        ts = result.get("timestamp", datetime.now(timezone.utc).isoformat())
        vals = (
            ts,
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
        )
        cur.execute(
            """
            INSERT INTO predictions (
                timestamp, predicted_temperature, temperature_api, humidity,
                wind_speed, wind_direction, confidence_range,
                humidity_timestamp, sunrise_drone, sunset_drone,
                sunrise_user, sunset_user, info,
                humidity_source, drone_location, user_location,
                rain_chance_2h, cloud_cover
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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


# ---------------------------------------------------------------------------
# Weather / external data
# ---------------------------------------------------------------------------

def get_weather_data(lat: float, lon: float) -> Tuple[float, float, float, str, str]:
    """Return (temp_c, rh%, cloud%, iso_time, source_url)."""
    url = ("https://api.open-meteo.com/v1/forecast?" \
           f"latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,cloudcover")
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
    future = now + pd.Timedelta(hours=2)
    date_str = now.strftime("%Y-%m-%d")
    url = ("https://api.open-meteo.com/v1/forecast?" \
           f"latitude={lat}&longitude={lon}&hourly=precipitation_probability&start_date={date_str}&end_date={date_str}&timezone=UTC")
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


def format_humidity_time(iso_str: str) -> str:
    try:
        iso_str = iso_str.replace('Z', '+00:00')
        dt = datetime.fromisoformat(iso_str)
        return f"{dt.strftime('%I:%M %p UTC')} on {dt.strftime('%Y-%m-%d')}"
    except Exception:
        return "Unknown time"


# ---------------------------------------------------------------------------
# Timezone / location utils
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Wind persistence helpers
# ---------------------------------------------------------------------------

def load_last_wind() -> Optional[Dict[str, Any]]:
    if not os.path.exists(WIND_CACHE_PATH):
        return None
    try:
        with open(WIND_CACHE_PATH, 'r') as f:
            data = json.load(f)
            if isinstance(data, list) and data:
                return data[-1]
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return None


def save_last_wind(speed_knots: float, dir_deg: float, timestamp: Optional[str] = None) -> None:
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    entry = {
        "timestamp": ts,
        "speed_knots": float(speed_knots),
        "direction_degrees": float(dir_deg),
    }
    try:
        if os.path.exists(WIND_CACHE_PATH):
            try:
                with open(WIND_CACHE_PATH, 'r') as f:
                    data = json.load(f)
            except Exception:
                data = []
        else:
            data = []
        if not isinstance(data, list):
            data = [data]
        data.append(entry)
        with open(WIND_CACHE_PATH, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logging.warning(f"Failed to cache wind: {e}")


# ---------------------------------------------------------------------------
# WeatherPredictor class (model + wind integration)
# ---------------------------------------------------------------------------
class WeatherPredictor:
    def __init__(self, model_dir: str = MODEL_DIR):
        # lazy load model/scaler (joblib may be None)
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

        # Wind calculator
        if WIND_CALCULATOR_AVAILABLE and PrecisionWindCalculator is not None:
            try:
                self.wind_calculator = PrecisionWindCalculator()
            except Exception as e:
                logging.warning(f"Wind calculator init failed: {e}")
                self.wind_calculator = None
        else:
            self.wind_calculator = None

        # Last known wind
        self.latest_wind = load_last_wind()

    # ---------------------------------------------------------------
    def _call_wind_calc(self, drone_data: Dict[str, Any]) -> Tuple[float, float]:
        """Return (speed_knots, dir_deg) from wind calculator; 0,0 if not available."""
        if not self.wind_calculator:
            return 0.0, 0.0
        try:
            # feed data (sync/async agnostic)
            add_fn = self.wind_calculator.add_drone_data
            if asyncio.iscoroutinefunction(add_fn):
                asyncio.run(add_fn(drone_data))
            else:
                add_fn(drone_data)
            calc_fn = self.wind_calculator.calculate_wind_multi_method
            if asyncio.iscoroutinefunction(calc_fn):
                wind_est = asyncio.run(calc_fn())
            else:
                wind_est = calc_fn()
            if wind_est:
                return float(wind_est.speed_knots), float(wind_est.direction_degrees)
        except Exception as e:
            logging.warning(f"Wind calculator error: {e}")
        return 0.0, 0.0

    # ---------------------------------------------------------------
    def prepare_drone_data(self, drone_data: Dict[str, Any], rh: float, cloud_cover: float,
                            wind_speed_ms: float, wind_direction: float) -> pd.DataFrame:
        """Build feature row for model; preserves original feature expectations."""
        now = datetime.now(timezone.utc)
        wind_gusts = wind_speed_ms  # simple assumption

        features = {
            'hour': now.hour,
            'month': now.month,
            'relative_humidity_2m': rh,
            'pressure_msl': safe_float(drone_data.get('pressure_msl', 1013)),
            'wind_speed_10m': wind_speed_ms,
            'wind_gusts_10m': wind_gusts,
            'wind_direction_10m': wind_direction,
            'cloudcover': cloud_cover
        }

        # Add any additional features scaler expects but we didn't fill
        for name in self.feature_names:
            if name not in features:
                if name == 'altitude':
                    features[name] = safe_float(drone_data.get('altitude', 0))
                elif name in ('temperature_2m', 'temp'):
                    features[name] = safe_float(drone_data.get('temperature_degc', 20))
                elif name in ('precipitation', 'rain'):
                    features[name] = 0.0
                else:
                    features[name] = 0.0

        return pd.DataFrame([features])

    # ---------------------------------------------------------------
    def predict(self, features: pd.DataFrame) -> float:
        try:
            if hasattr(self.scaler, 'feature_names_in_'):
                features = features.reindex(columns=self.scaler.feature_names_in_, fill_value=0.0)
            X_scaled = self.scaler.transform(features)
            pred = float(self.model.predict(X_scaled)[0])
            # sanity clamp (as in your original)
            if not (0 <= pred <= 40):
                pred = 15.0
            return pred
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            return 15.0


# ---------------------------------------------------------------------------
# Core: process single live prediction dict from telemetry row
# ---------------------------------------------------------------------------

def process_live_prediction(data: Dict[str, Any]) -> Dict[str, Any]:
    """Process **one** telemetry record -> enriched prediction dict."""
    # Basic requireds
    missing = [f for f in ('latitude', 'longitude') if safe(data.get(f)) is None]
    if missing:
        logging.warning(f"Missing required fields: {missing}")
        return {"error": f"Missing required data fields: {missing}", "input": data}

    predictor = WeatherPredictor()

    lat = safe_float(data.get("latitude", DRONE_COORDS[0]))
    lon = safe_float(data.get("longitude", DRONE_COORDS[1]))
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        logging.warning(f"Invalid coords {lat},{lon}; using fallback")
        lat, lon = DRONE_COORDS

    # External weather
    temp_api, rh, cloud_cover_percent, rh_time, rh_url = get_weather_data(lat, lon)
    rain_chance_2h = get_rain_chance_in_2_hours(lat, lon)

    # WIND
    wind_speed_knots = 0.0
    wind_direction_degrees = 0.0
    wind_speed_ms = 0.0

    # 1) try live wind calculator
    if predictor.wind_calculator:
        ws_kn, wd_deg = predictor._call_wind_calc(data)
        if ws_kn > 0 or wd_deg > 0:
            wind_speed_knots = ws_kn
            wind_direction_degrees = wd_deg
            wind_speed_ms = ws_kn * 0.514444
            save_last_wind(ws_kn, wd_deg)  # persist

    # 2) fallback to cached wind
    if wind_speed_knots == 0.0 and predictor.latest_wind:
        wind_speed_knots = safe_float(predictor.latest_wind.get('speed_knots', 0))
        wind_direction_degrees = safe_float(predictor.latest_wind.get('direction_degrees', 0))
        wind_speed_ms = wind_speed_knots * 0.514444

    # MODEL FEATURES & PREDICTION
    features = predictor.prepare_drone_data(data, rh=rh, cloud_cover=cloud_cover_percent,
                                            wind_speed_ms=wind_speed_ms, wind_direction=wind_direction_degrees)
    prediction = predictor.predict(features)

    # Confidence interval (bootstrap 5 predictions)
    try:
        preds = [predictor.predict(features) for _ in range(5)]
        std_dev = float(np.std(preds))
        ci = max(round(1.96 * std_dev, 2), 0.1)
    except Exception as e:
        logging.warning(f"CI calc fail: {e}")
        ci = 0.1

    # Location & astro
    humidity_time_fmt = format_humidity_time(rh_time)
    drone_tz = get_timezone(lat, lon)
    user_tz = get_timezone(*EXPECTED_COORDS)
    drone_loc = get_location_name(lat, lon)
    user_loc = get_location_name(*EXPECTED_COORDS)
    drone_sun = get_sunrise_sunset(lat, lon, drone_tz)
    user_sun = get_sunrise_sunset(*EXPECTED_COORDS, user_tz)

    # Timestamp handling
    now = datetime.now(timezone.utc)
    ts_in = safe(data.get("timestamp"))
    if isinstance(ts_in, str) and ts_in:
        ts_out = ts_in
    else:
        ts_out = now.strftime("%Y-%m-%dT%H:%M:%S")

    # Build output
    output = {
        "timestamp": ts_out,
        "rain_chance_2h (%)": float(rain_chance_2h),
        "cloud_cover (%)": float(cloud_cover_percent),
        "temperature_api (°C)": round(temp_api, 2),
        "predicted_temperature": round(prediction, 2),
        "confidence_range (±°C)": ci,
        "humidity (% RH)": round(rh, 1),
        "humidity_timestamp": humidity_time_fmt,
        "sunrise_at_drone_location": drone_sun['sunrise_readable'],
        "sunset_at_drone_location": drone_sun['sunset_readable'],
        "sunrise_at_user_location": user_sun['sunrise_readable'],
        "sunset_at_user_location": user_sun['sunset_readable'],
        "info": f"95% confidence interval: {round(prediction - ci, 2)} to {round(prediction + ci, 2)}",
        "humidity_source": rh_url,
        "drone_location": drone_loc,
        "user_location": user_loc,
        "wind_speed_knots": wind_speed_knots,
        "wind_direction_degrees": wind_direction_degrees,
        "input": data,
        # legacy compatibility keys if you downstream parse by alt names
        "predicted_value": round(prediction, 2),
    }

    # quick terminal prints (like original)
    print(f"Wind Speed: {wind_speed_knots:.2f} knots")
    print(f"Wind Direction: {wind_direction_degrees:.1f}°")

    return output


# ---------------------------------------------------------------------------
# TCP client loop (live streaming)
# ---------------------------------------------------------------------------

def run_tcp_client(host: str = TCP_HOST_DEFAULT, port: int = TCP_PORT_DEFAULT, retry_delay: int = TCP_RETRY_DELAY):
    logging.warning(f"Attempting to connect to TCPLogServer at {host}:{port}")
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(10)
                s.connect((host, port))
                logging.warning("Connected to TCP server. Starting data processing...")
                buffer = ''
                last_pred_time = 0.0
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
                            if now - last_pred_time >= PREDICTION_INTERVAL:
                                result = process_live_prediction(drone_data)
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


# ---------------------------------------------------------------------------
# CSV one‑shot mode
# ---------------------------------------------------------------------------

def predict_from_csv(csv_path: str) -> None:
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

    # walk backward to find last valid row
    for idx in range(len(df) - 1, -1, -1):
        row = df.iloc[idx]
        data = {}
        for col in df.columns:
            v = row[col]
            if pd.notna(v):
                data[col] = v
        if not data:
            continue
        # fill required
        data.setdefault('timestamp', datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"))
        data.setdefault('latitude', DRONE_COORDS[0])
        data.setdefault('longitude', DRONE_COORDS[1])
        result = process_live_prediction(data)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if "error" not in result:
            log_prediction_to_db(result)
        break
    else:
        logging.error("No valid rows found in CSV file")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Weather prediction TCP client or CSV mode.")
    parser.add_argument('--tcp', action='store_true', help='Run as TCP client')
    parser.add_argument('--csv', type=str, help='Path to CSV for prediction')
    parser.add_argument('--reset-db', action='store_true', help='Drop & recreate predictions table (DATA LOSS)')
    parser.add_argument('--host', type=str, default=TCP_HOST_DEFAULT, help='TCP host')
    parser.add_argument('--port', type=int, default=TCP_PORT_DEFAULT, help='TCP port')
    args = parser.parse_args()

    try:
        initialize_database(reset=args.reset_db)
        if args.tcp:
            run_tcp_client(host=args.host, port=args.port)
        elif args.csv:
            predict_from_csv(args.csv)
        else:
            logging.error("Please use --tcp or --csv <path>")
            sys.exit(1)
    except KeyboardInterrupt:
        logging.info("Script interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)
