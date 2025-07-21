#!/usr/bin/env python3
"""
Enhanced Weather Prediction Pipeline with Wind Calculator Integration
===================================================================
- Integrates EnhancedWindCalculator for accurate, filtered wind measurements across samples.
- Predicts near-surface ambient temperature using ML model *plus* sanity blending with API reference.
- Uses live telemetry streamed from `MAVSDK_logger.py` over TCP (JSON lines).
- Includes fallback external weather (Open-Meteo) + astro info + DB logging.
- Preserves all original features you wanted (TCP, CSV, DB, sunrise/sunset, humidity, rain chance, etc.).

Key Behavior
------------
1. **Wind**: Every telemetry row is fed into an `EnhancedWindCalculator` ring buffer (50 samples). Filtered wind is output when enough data & min interval elapsed. Confidence and sample count reported.
2. **Temperature Prediction**: Features built from telemetry + external weather. Raw ML output is blended toward API temperature if out-of-family (prevents nonsense 0°C in Mumbai heat). Blend strength proportional to disagreement.
3. **Database Logging**: Fused predicted temp, API temp, wind, humidity, etc. Extra wind confidence + sample count logged.
4. **Prediction Rate**: Default once per 60s in TCP mode (change with `--period <sec>`). CSV mode processes last valid row once.

Usage
-----
```bash
# Live telemetry (default host 127.0.0.1 port 9000)
python predict.py --tcp

# Live telemetry, faster updates (every 10s)
python predict.py --tcp --period 10

# One-shot from CSV
python predict.py --csv path/to/file.csv

# Reset DB schema (drops table) then run TCP
python predict.py --reset-db --tcp
```
"""

import os
import sys
import csv
import json
import time
import socket
import asyncio
import logging
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Tuple
from collections import deque
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd  # You said pandas is now working; leaving in.
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

# =============================================================================
# Integrated Wind Calculator (inline, simplified from prior wind_calculator.py)
# =============================================================================
@dataclass
class WindResult:
    timestamp: str
    wind_speed: float         # m/s
    wind_speed_knots: float   # kt
    wind_direction: float     # deg FROM (met convention)
    ground_speed: float       # m/s (horizontal)
    airspeed: float           # TAS m/s
    confidence: float         # %
    label: str                # high/medium/low
    sample_count: int         # raw samples in window
    std_dev_speed: float      # m/s
    std_dev_direction: float  # deg (currently unused detailed)
    outliers_rejected: int    # count
    temperature_c: Optional[float] = None
    pressure_hpa: Optional[float] = None


class EnhancedWindCalculator:
    """Maintain rolling telemetry samples → filtered wind estimate.

    Input dict keys expected (as produced by MAVSDK_logger.py):
      north_m_s, east_m_s, airspeed_m_s, yaw_deg, temperature_degc, pressure_hpa
    """
    def __init__(self, maxlen: int = 50):
        self.raw = deque(maxlen=maxlen)
        self.last_valid_dir = None
        self.last_valid_speed = None
        self.reject_counts = {"validate": 0}
        self._last_calc_t = 0.0
        self.KT_PER_MS = 1.94384
        self.MAX_WIND_SPEED = 100.0
        self.OUTLIER_THRESHOLD = 1.5
        self.MIN_CALC_INTERVAL = 1.0  # seconds between filtered outputs

    # ----- helpers -----
    def _f(self, x: Any, default: Optional[float] = None) -> Optional[float]:
        if x is None:
            return default
        try:
            v = float(x)
            return v if math.isfinite(v) else default
        except (TypeError, ValueError):
            return default

    def _norm_heading(self, yaw: Optional[float]) -> Optional[float]:
        if yaw is None:
            return None
        try:
            y = float(yaw)
            if not math.isfinite(y):
                return None
            return y % 360.0
        except (TypeError, ValueError):
            return None

    def ias_to_tas(self, ias: float, temp_c: Optional[float], press_hpa: Optional[float]) -> float:
        """Convert IAS to TAS using density scaling; fall back to IAS."""
        R_AIR = 287.05
        RHO0 = 1.225
        if temp_c is None or press_hpa is None:
            return ias
        try:
            if not (-60.0 <= temp_c <= 60.0) or not (800.0 <= press_hpa <= 1100.0):
                return ias
            t_k = temp_c + 273.15
            p_pa = press_hpa * 100.0
            rho = p_pa / (R_AIR * t_k)
            if rho <= 0:
                return ias
            ratio = RHO0 / rho
            return ias * math.sqrt(ratio) if 0.5 <= ratio <= 2.0 else ias
        except Exception:
            return ias

    def calc_wind_dir_from(self, wn: float, we: float) -> float:
        # meteorological FROM direction
        return (math.degrees(math.atan2(-we, -wn)) + 360.0) % 360.0

    def circ_mean(self, degs):
        if not degs:
            return 0.0
        rad = np.radians(degs)
        x = np.cos(rad).sum()
        y = np.sin(rad).sum()
        return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

    def remove_outliers_iqr(self, data, threshold=None):
        if threshold is None:
            threshold = self.OUTLIER_THRESHOLD
        if len(data) < 4:
            return list(data), 0
        arr = np.array(data)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lb = q1 - threshold * iqr
        ub = q3 + threshold * iqr
        mask = (arr >= lb) & (arr <= ub)
        return arr[mask].tolist(), int((~mask).sum())

    def _validate(self, vn, ve, ias, heading):
        if any(x is None for x in [vn, ve, ias, heading]):
            self.reject_counts["validate"] += 1
            return False
        if abs(vn) > 200 or abs(ve) > 200 or ias > 200 or ias < 0:
            self.reject_counts["validate"] += 1
            return False
        return True

    # ----- main ingest -----
    def process(self, data: Dict[str, Any]) -> Optional[WindResult]:
        vn = self._f(data.get("north_m_s"))
        ve = self._f(data.get("east_m_s"))
        ias = self._f(data.get("airspeed_m_s"))
        temp = self._f(data.get("temperature_degc"))
        press = self._f(data.get("pressure_hpa"))
        heading = self._norm_heading(data.get("yaw_deg"))

        if not self._validate(vn, ve, ias, heading):
            return None

        tas = self.ias_to_tas(ias, temp, press)
        gs = math.hypot(vn, ve)
        hr = math.radians(heading)
        va_n = tas * math.cos(hr)
        va_e = tas * math.sin(hr)
        wn = vn - va_n
        we = ve - va_e
        w_speed = math.hypot(wn, we)
        w_dir = self.calc_wind_dir_from(wn, we)
        if w_speed > self.MAX_WIND_SPEED:
            return None

        self.raw.append({
            "wn": wn, "we": we, "w_speed": w_speed, "w_dir": w_dir,
            "gs": gs, "tas": tas, "t": time.time(), "temp": temp, "press": press,
        })

        now = time.time()
        if now - self._last_calc_t < self.MIN_CALC_INTERVAL:
            return None
        self._last_calc_t = now
        return self._filtered_result(data.get("timestamp"))

    def _filtered_result(self, ts: Optional[str]) -> WindResult:
        arr = list(self.raw)
        speeds = [s["w_speed"] for s in arr]
        dirs = [s["w_dir"] for s in arr]
        gses = [s["gs"] for s in arr]
        tases = [s["tas"] for s in arr]

        clean_speeds, out_speed = self.remove_outliers_iqr(speeds)
        clean_gs, out_gs = self.remove_outliers_iqr(gses)

        w_speed_f = float(np.median(clean_speeds)) if clean_speeds else (speeds[-1] if speeds else 0.0)
        gs_f = float(np.median(clean_gs)) if clean_gs else (gses[-1] if gses else 0.0)
        tas_f = float(np.median(tases)) if tases else 0.0
        w_dir_f = self.circ_mean(dirs)
        w_speed_std = float(np.std(clean_speeds)) if len(clean_speeds) > 1 else 0.0

        # Low-wind stabilization
        if w_speed_f < 1.5:
            if self.last_valid_dir is not None:
                w_dir_f = 0.05 * w_dir_f + 0.95 * self.last_valid_dir
            if self.last_valid_speed is not None:
                w_speed_f = 0.7 * w_speed_f + 0.3 * self.last_valid_speed
        else:
            self.last_valid_dir = w_dir_f
            self.last_valid_speed = w_speed_f

        outliers_total = out_speed + out_gs
        conf = self._confidence(len(arr), w_speed_std, outliers_total, w_speed_f, tas_f)
        lbl = "high" if conf >= 85 else ("medium" if conf >= 65 else "low")

        return WindResult(
            timestamp=ts or time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            wind_speed=w_speed_f,
            wind_speed_knots=w_speed_f * self.KT_PER_MS,
            wind_direction=w_dir_f,
            ground_speed=gs_f,
            airspeed=tas_f,
            confidence=conf,
            label=lbl,
            sample_count=len(arr),
            std_dev_speed=w_speed_std,
            std_dev_direction=0.0,
            outliers_rejected=outliers_total,
            temperature_c=None,
            pressure_hpa=None,
        )

    def _confidence(self, n: int, s_std: float, outliers: int, w_speed: float, tas: float) -> float:
        base = 98.0 if n >= 40 else (90.0 if n >= 25 else (80.0 if n >= 15 else 65.0))
        speed_pen = min(s_std * 15.0, 25.0)
        out_pen = min(outliers * 5.0, 25.0)
        low_w_pen = (2.0 - w_speed) * 10.0 if w_speed < 2.0 else 0.0
        low_tas_pen = (10.0 - tas) * 2.0 if tas < 10.0 else 0.0
        conf = base - speed_pen - out_pen - low_w_pen - low_tas_pen
        return max(0.0, min(100.0, conf))


# ---------------------------------------------------------------------------
# Config (these override earlier duplicates; kept for backward readability)
# ---------------------------------------------------------------------------
MODEL_DIR = "models"
EXPECTED_COORDS = (13.0, 77.625)         # User: Bengaluru
DRONE_COORDS = (27.6889981, 86.726715)   # Lukla fallback
TCP_HOST_DEFAULT = "127.0.0.1"
TCP_PORT_DEFAULT = 9000
TCP_RETRY_DELAY = 5
PREDICTION_INTERVAL = 60  # seconds; CLI overridable
SQLITE_PATH = "weather_predict.db"
WIND_CACHE_PATH = "wind_measurements.json"

# Sanity / blending thresholds
MODEL_VALID_TEMP_RANGE = (-40.0, 60.0)
MODEL_VALID_PRESS_RANGE = (870.0, 1050.0)
MODEL_VALID_RH_RANGE = (0.0, 100.0)
MODEL_API_BLEND_THRESH_C = 8.0
MODEL_MIN_WEIGHT = 0.05
MODEL_MAX_WEIGHT = 1.0

# ---------------------------------------------------------------------------
# Safe helpers
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
# DB schema mgmt
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
    cloud_cover REAL,
    wind_confidence REAL,
    wind_samples INTEGER
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
            safe_float(result.get('wind_confidence')),
            safe_float(result.get('wind_samples')),
        )
        cur.execute(
            """
            INSERT INTO predictions (
                timestamp, predicted_temperature, temperature_api, humidity,
                wind_speed, wind_direction, confidence_range,
                humidity_timestamp, sunrise_drone, sunset_drone,
                sunrise_user, sunset_user, info,
                humidity_source, drone_location, user_location,
                rain_chance_2h, cloud_cover, wind_confidence, wind_samples
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    future = now + timedelta(hours=2)
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


def get_forecast_2h(lat: float, lon: float) -> Dict[str, float]:
    """Fetch temperature, cloud cover, and rain chance for 2 hours ahead using Open-Meteo API."""
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
        # fallback: last available
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
# Wind cache helpers
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
    entry = {"timestamp": ts, "speed_knots": float(speed_knots), "direction_degrees": float(dir_deg)}
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
# WeatherPredictor (model + feature prep + blending)
# ---------------------------------------------------------------------------
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
        self.wind_calculator = EnhancedWindCalculator()
        self.latest_wind_cache = load_last_wind()

    # ------------------------ feature prep ------------------------
    def prepare_features(self, drone_data: Dict[str, Any], rh: float, cloud_cover: float,
                         wind_speed_ms: float, wind_direction: float, api_temp: float) -> pd.DataFrame:
        now = datetime.now(timezone.utc)
        # Source fields
        altitude = safe_float(drone_data.get('altitude_from_sealevel', 0))
        base_pressure = safe_float(drone_data.get('pressure_hpa', 1013))
        # crude barometric altitude reduction (avoid nonsense model inputs)
        if altitude > 10:  # m
            # simple scale vs ISA; not rigorous but better than raw high-alt pressure
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
            # extras (may be dropped by scaler)
            'hour_temp_factor': hour_temp_factor,
            'humidity_temp_factor': humidity_temp_factor,
            'api_temperature_reference': api_temp,
        }
        # Ensure all scaler cols present
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

    # ------------------------ raw model predict ------------------------
    def predict_raw(self, features: pd.DataFrame) -> float:
        try:
            if hasattr(self.scaler, 'feature_names_in_'):
                features = features.reindex(columns=self.scaler.feature_names_in_, fill_value=0.0)
            X_scaled = self.scaler.transform(features)
            return float(self.model.predict(X_scaled)[0])
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            return float('nan')


# ---------------------------------------------------------------------------
# Prediction fusion / gating
# ---------------------------------------------------------------------------

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
    conf = max(0.1, round((1 - weight) * 5.0, 2))  # degC 95% CI-ish
    return fused, conf


# ---------------------------------------------------------------------------
# Global predictor singleton
# ---------------------------------------------------------------------------
_GLOBAL_PREDICTOR: Optional[WeatherPredictor] = None

def get_predictor() -> WeatherPredictor:
    global _GLOBAL_PREDICTOR
    if _GLOBAL_PREDICTOR is None:
        _GLOBAL_PREDICTOR = WeatherPredictor()
    return _GLOBAL_PREDICTOR


# ---------------------------------------------------------------------------
# Core processing of a telemetry record → prediction dict
# ---------------------------------------------------------------------------

def process_live_prediction(data: Dict[str, Any], predictor: Optional[WeatherPredictor] = None) -> Dict[str, Any]:
    if predictor is None:
        predictor = get_predictor()

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

    # Pressure at drone altitude (if present)
    p_station = safe_float(data.get('pressure_hpa'), 1013.0)
    alt = safe_float(data.get('altitude_from_sealevel'), 0.0)
    # crude sea-level reduction for gating
    if alt > 10:
        p_msl = p_station * (1 - 0.0065 * alt / 288.15) ** -5.257  # invert above; approx
    else:
        p_msl = p_station

    # Wind ingest
    wind_result = predictor.wind_calculator.process(data)
    if wind_result:
        wind_speed_ms = wind_result.wind_speed
        wind_speed_knots = wind_result.wind_speed_knots
        wind_direction_degrees = wind_result.wind_direction
        wind_confidence = wind_result.confidence
        wind_samples = wind_result.sample_count
        save_last_wind(wind_speed_knots, wind_direction_degrees, wind_result.timestamp)
        print(f"🌬️ Enhanced Wind: {wind_speed_knots:.2f} kt from {wind_direction_degrees:.1f}° (conf {wind_confidence:.1f}%, n={wind_samples})")
    else:
        # fallback to cached wind if present and none computed yet
        cached = predictor.latest_wind_cache or load_last_wind()
        if cached:
            wind_speed_knots = safe_float(cached.get('speed_knots'), 0.0)
            wind_direction_degrees = safe_float(cached.get('direction_degrees'), 0.0)
            wind_speed_ms = wind_speed_knots * 0.514444
            wind_confidence = 0.0
            wind_samples = 0
            print("ℹ️ Using cached last-known wind.")
        else:
            wind_speed_knots = 0.0
            wind_direction_degrees = 0.0
            wind_speed_ms = 0.0
            wind_confidence = 0.0
            wind_samples = 0
            # only print occasionally? We'll always print minimal.
            print("⚠️ No wind data available yet.")

    # Features → model
    features = predictor.prepare_features(data, rh=rh, cloud_cover=cloud_cover_percent,
                                          wind_speed_ms=wind_speed_ms, wind_direction=wind_direction_degrees,
                                          api_temp=temp_api)
    model_pred = predictor.predict_raw(features)
    fused_temp, ci = fuse_prediction(model_pred, temp_api, rh, p_msl)

    # Montecarlo-ish confidence refinement (adds to CI)
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
            ci = max(ci, round(ci_mc, 2))  # widen if MC larger
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

    # Timestamp
    ts_in = safe(data.get("timestamp"))
    ts_out = ts_in if isinstance(ts_in, str) and ts_in else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    # Get forecast for 2 hours ahead
    forecast = get_forecast_2h(lat, lon)

    output = {
        "timestamp": ts_out,
        "rain_chance_2h (%)": float(rain_chance_2h),
        "cloud_cover (%)": float(cloud_cover_percent),
        "temperature_api (°C)": round(temp_api, 2),
        "predicted_temperature": round(fused_temp, 2),
        "confidence_range (±°C)": ci,
        "humidity (% RH)": round(rh, 1),
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
        "wind_confidence": wind_confidence,
        "wind_samples": wind_samples,
        "forecast_temp_2h": forecast["forecast_temp_2h"],
        "forecast_cloud_cover_2h": forecast["forecast_cloud_cover_2h"],
        "forecast_rain_chance_2h": forecast["forecast_rain_chance_2h"],
        "input": data,
    }

    # Quick terminal echo (compact)
    print(
        f"Wind: {wind_speed_knots:.2f} kt from {wind_direction_degrees:.1f}° | "
        f"Temp {fused_temp:.1f}°C (API {temp_api:.1f}°C) | CI ±{ci}°C | "
        f"Forecast 2h: Temp {forecast['forecast_temp_2h']}°C, Cloud {forecast['forecast_cloud_cover_2h']}%, Rain {forecast['forecast_rain_chance_2h']}%"
    )

    return output


# ---------------------------------------------------------------------------
# TCP client loop (live streaming)
# ---------------------------------------------------------------------------

def run_tcp_client(host: str = TCP_HOST_DEFAULT, port: int = TCP_PORT_DEFAULT, retry_delay: int = TCP_RETRY_DELAY, period: int = PREDICTION_INTERVAL):
    predictor = get_predictor()
    logging.warning(f"Attempting to connect to TCPLogServer at {host}:{port}")
    last_pred_time = 0.0
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(10)
                s.connect((host, port))
                logging.warning("Connected to TCP server. Starting data processing...")
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


# ---------------------------------------------------------------------------
# CSV one‑shot mode
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Weather prediction TCP client or CSV mode.")
    parser.add_argument('--tcp', action='store_true', help='Run as TCP client')
    parser.add_argument('--csv', type=str, help='Path to CSV for prediction')
    parser.add_argument('--reset-db', action='store_true', help='Drop & recreate predictions table (DATA LOSS)')
    parser.add_argument('--host', type=str, default=TCP_HOST_DEFAULT, help='TCP host')
    parser.add_argument('--port', type=int, default=TCP_PORT_DEFAULT, help='TCP port')
    parser.add_argument('--period', type=int, default=PREDICTION_INTERVAL, help='Seconds between predictions (0=every row)')
    args = parser.parse_args()

    try:
        initialize_database(reset=args.reset_db)
        if args.tcp:
            run_tcp_client(host=args.host, port=args.port, period=args.period)
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
