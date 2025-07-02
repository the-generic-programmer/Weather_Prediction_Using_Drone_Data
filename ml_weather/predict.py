#!/usr/bin/env python3

import os
import json
import joblib
import socket
import pandas as pd
import numpy as np
import requests
import sys
import time
import logging
from datetime import datetime
from astral import LocationInfo
from astral.sun import sun
from timezonefinder import TimezoneFinder
from geopy.geocoders import Nominatim
import pytz
import mysql.connector
from mysql.connector import Error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def initialize_database():
    """Initialize the weather_predict database at script start."""
    try:
        temp_config = {
            'host': 'localhost',
            'user': 'root',
            'password': 'Root@123'
        }
        conn = mysql.connector.connect(**temp_config)
        cursor = conn.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS weather_predict")
        conn.commit()
        cursor.close()
        conn.close()
        logging.info("Database weather_predict created or exists")

        # Create predictions table
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='Root@123',
            database='weather_predict'
        )
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME,
                predicted_temperature FLOAT,
                humidity FLOAT,
                wind_speed FLOAT,
                wind_direction FLOAT,
                confidence_range FLOAT,
                humidity_timestamp TEXT,
                sunrise_drone TEXT,
                sunset_drone TEXT,
                sunrise_user TEXT,
                sunset_user TEXT,
                info TEXT,
                humidity_source TEXT,
                location TEXT,
                warning TEXT,
                rain_chance_2h FLOAT,
                cloud_cover FLOAT
            )
        ''')
        conn.commit()
        cursor.close()
        conn.close()
        logging.info("Table predictions created or exists")
    except Error as e:
        logging.error(f"Failed to initialize database weather_predict: {e}")
        raise

# Define database logging function
def log_prediction_to_db(result):
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='Root@123',
            database='weather_predict'
        )
        cursor = conn.cursor()

        ts = result.get("timestamp")
        if ts:
            try:
                ts = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                ts = datetime.utcnow()
        else:
            ts = datetime.utcnow()

        wind_speed = np.sqrt(
            (safe(result['input'].get('north_m_s', 0)) or 0) ** 2 +
            (safe(result['input'].get('east_m_s', 0)) or 0) ** 2
        )
        wind_direction = (
            np.degrees(np.arctan2(
                safe(result['input'].get('east_m_s', 0)) or 0,
                safe(result['input'].get('north_m_s', 0)) or 0
            )) % 360
            if safe(result['input'].get('north_m_s')) is not None else 0
        )

        values = (
            ts,
            safe(result.get('predicted_temperature')),
            safe(result.get('humidity (% RH)')),
            wind_speed,
            wind_direction,
            safe(result.get('confidence_range (±°C)')),
            safe(result.get('humidity_timestamp')),
            safe(result.get('sunrise_at_drone_location')),
            safe(result.get('sunset_at_drone_location')),
            safe(result.get('sunrise_at_user_location')),
            safe(result.get('sunset_at_user_location')),
            safe(result.get('info')),
            safe(result.get('humidity_source')),
            safe(result.get('location')),
            safe(result.get('warning')),
            safe(result.get('rain_chance_2h (%)')),
            safe(result.get('cloud_cover (%)'))
        )

        cursor.execute('''
            INSERT INTO predictions (
                timestamp, predicted_temperature, humidity,
                wind_speed, wind_direction, confidence_range,
                humidity_timestamp, sunrise_drone, sunset_drone,
                sunrise_user, sunset_user, info,
                humidity_source, location, warning,
                rain_chance_2h, cloud_cover
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', values)

        conn.commit()
        cursor.close()
        conn.close()
        logging.info("Prediction logged to database")
    except Error as e:
        logging.error(f"Database Error: {e}")

MODEL_DIR = "models"
EXPECTED_COORDS = (13.0, 77.625)
tf = TimezoneFinder()
geolocator = Nominatim(user_agent="weather_predictor")

def safe(val):
    return None if pd.isnull(val) or str(val).lower() in ("nan", "none", "") else val

def get_timezone(lat, lon):
    try:
        return tf.timezone_at(lat=lat, lng=lon) or "UTC"
    except Exception as e:
        logging.error(f"Timezone lookup failed: {e}")
        return "UTC"

def get_location_name(lat, lon):
    try:
        location = geolocator.reverse((lat, lon), language='en')
        return location.address if location else "Unknown location"
    except Exception as e:
        logging.error(f"Location lookup failed: {e}")
        return "Unknown location"

def get_live_humidity(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=relative_humidity_2m,cloudcover"
        response = requests.get(url, timeout=10)
        response.encoding = 'utf-8'
        data = response.json()
        if "current" in data:
            rh = data["current"].get("relative_humidity_2m", 50)
            cloud_cover = data["current"].get("cloudcover", 0)
            rh_time = data["current"].get("time", datetime.utcnow().isoformat())
            logging.info(f"API response: humidity={rh}%, cloud_cover={cloud_cover}%")
            return rh, cloud_cover, rh_time, url
        else:
            logging.error(f"Invalid Open-Meteo response: {data}")
            return 50, 0, datetime.utcnow().isoformat(), url
    except Exception as e:
        logging.error(f"Failed to fetch humidity and cloud cover: {e}")
        return 50, 0, datetime.utcnow().isoformat(), url

def get_rain_chance_in_2_hours(lat, lon):
    try:
        now = datetime.utcnow()
        future_time = now + pd.Timedelta(hours=2)
        date_str = now.strftime("%Y-%m-%d")
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&hourly=precipitation_probability"
            f"&start_date={date_str}&end_date={date_str}"
            f"&timezone=UTC"
        )
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        times = data.get("hourly", {}).get("time", [])
        chances = data.get("hourly", {}).get("precipitation_probability", [])
        if not isinstance(times, list) or not isinstance(chances, list) or len(times) != len(chances):
            logging.error(f"Malformed rain chance data: times={times}, chances={chances}")
            return 0
        for i, t in enumerate(times):
            if t >= future_time.strftime("%Y-%m-%dT%H:00"):
                logging.info(f"Rain chance for {t}: {chances[i]}%")
                return chances[i]
        logging.warning("No matching future time found for rain chance")
        return 0
    except requests.RequestException as e:
        logging.error(f"HTTP error fetching rain chance: {e}")
        return 0
    except Exception as e:
        logging.error(f"Failed to fetch rain chance: {e}")
        return 0

def format_humidity_time(iso):
    try:
        dt = datetime.fromisoformat(iso.replace('Z', '+00:00'))
        if not isinstance(dt, datetime):
            raise ValueError("Parsed humidity time is not a datetime object")
        return f"{dt.strftime('%I:%M %p UTC')} on {dt.strftime('%Y-%m-%d')}"
    except Exception as e:
        logging.error(f"Error formatting humidity time: {e} (input: {iso})")
        return "Unknown time"

def get_sunrise_sunset(lat, lon, timezone_str):
    try:
        city = LocationInfo(latitude=lat, longitude=lon)
        observer_date = datetime.utcnow().date()
        s = sun(city.observer, date=observer_date, tzinfo=pytz.timezone(timezone_str))
        return {
            "sunrise_readable": s['sunrise'].strftime("%I:%M %p %Z"),
            "sunset_readable": s['sunset'].strftime("%I:%M %p %Z")
        }
    except Exception as e:
        logging.error(f"Sunrise/sunset calculation failed: {e}")
        return {"sunrise_readable": "Unknown", "sunset_readable": "Unknown"}

class WeatherPredictor:
    def __init__(self, model_dir: str = "models"):
        try:
            self.model = joblib.load(os.path.join(model_dir, 'weather_model.joblib'))
            self.scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
            logging.info("Weather model and scaler loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load model or scaler: {e}")
            raise

    def prepare_drone_data(self, drone_data: dict) -> pd.DataFrame:
        current_time = datetime.utcnow()
        wind_speed = np.sqrt(
            (safe(drone_data.get('north_m_s', 0)) or 0) ** 2 +
            (safe(drone_data.get('east_m_s', 0)) or 0) ** 2
        )
        wind_direction = (
            np.degrees(np.arctan2(
                safe(drone_data.get('east_m_s', 0)) or 0,
                safe(drone_data.get('north_m_s', 0)) or 0
            )) % 360 if safe(drone_data.get('north_m_s')) is not None else 0
        )

        temperature = safe(drone_data.get('temperature_degc', None))
        if temperature is None:
            logging.warning("temperature_degc missing, model will predict temperature")

        features = {
            'hour': current_time.hour,
            'month': current_time.month,
            'relative_humidity_2m': safe(drone_data.get('relative_humidity_2m', 50)) or 50,
            'pressure_msl': safe(drone_data.get('pressure_msl', 1013)) or 1013,
            'wind_speed_10m': wind_speed,
            'wind_direction_10m': wind_direction,
            'cloudcover': safe(drone_data.get('cloud_cover', 0)) or 0
        }
        return pd.DataFrame([features])

    def predict(self, features: pd.DataFrame) -> float:
        try:
            X_scaled = self.scaler.transform(features)
            return self.model.predict(X_scaled)[0]
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            raise

def process_live_prediction(data: dict):
    try:
        required_fields = ['latitude', 'longitude', 'north_m_s', 'east_m_s']
        if not all(safe(data.get(field)) is not None for field in required_fields):
            logging.warning(f"Missing required fields in data: {data}")
            return {"error": "Missing required data fields", "input": data}

        predictor = WeatherPredictor()
        features = predictor.prepare_drone_data(data)
        pred = predictor.predict(features) if safe(data.get('temperature_degc')) is None else float(safe(data.get('temperature_degc')))
        
        lat = float(safe(data.get("latitude", EXPECTED_COORDS[0])) or EXPECTED_COORDS[0])
        lon = float(safe(data.get("longitude", EXPECTED_COORDS[1])) or EXPECTED_COORDS[1])

        rh, cloud_cover_percent, rh_time, rh_url = get_live_humidity(lat, lon)
        if rh is None or rh_time is None:
            rh, cloud_cover_percent, rh_time, rh_url = 50, 0, datetime.utcnow().isoformat(), "N/A"
            logging.warning("Using default humidity and cloud cover due to API failure")

        rain_chance_2h = get_rain_chance_in_2_hours(lat, lon)
        humidity_time_fmt = format_humidity_time(rh_time)

        drone_tz = get_timezone(lat, lon)
        user_tz = get_timezone(*EXPECTED_COORDS)
        drone_loc = get_location_name(lat, lon)

        drone_sun = get_sunrise_sunset(lat, lon, drone_tz)
        user_sun = get_sunrise_sunset(*EXPECTED_COORDS, user_tz)

        ci = 0.1
        if safe(data.get('temperature_degc')) is None:
            preds = [predictor.predict(features) for _ in range(50)]
            std_dev = np.std(preds)
            ci = round(1.96 * std_dev, 2)
            ci = ci if ci > 0 else 0.1

        now = datetime.utcnow()
        timestamp = safe(data.get("timestamp")) or now.strftime("%Y-%m-%dT%H:%M:%S")

        output = {
            "timestamp": timestamp,
            "rain_chance_2h (%)": float(rain_chance_2h),
            "cloud_cover (%)": float(cloud_cover_percent),
            "predicted_temperature": round(pred, 2),
            "confidence_range (±°C)": ci,
            "humidity (% RH)": round(rh, 1),
            "humidity_timestamp": humidity_time_fmt,
            "sunrise_at_drone_location": drone_sun['sunrise_readable'],
            "sunset_at_drone_location": drone_sun['sunset_readable'],
            "sunrise_at_user_location": user_sun['sunrise_readable'],
            "sunset_at_user_location": user_sun['sunset_readable'],
            "info": f"95% chance the actual temperature lies between {round(pred - ci, 2)}°C and {round(pred + ci, 2)}°C",
            "humidity_source": rh_url,
            "location": drone_loc,
            "input": data
        }

        if abs(lat - EXPECTED_COORDS[0]) > 2 or abs(lon - EXPECTED_COORDS[1]) > 2:
            output["warning"] = f"Warning: Drone location is far from expected coordinates ({EXPECTED_COORDS})"

        return output

    except Exception as e:
        logging.error(f"Prediction Error: {e}")
        return {"error": f"Prediction Error: {e}", "input": data}

def run_tcp_client(host='127.0.0.1', port=9000, max_retries=10, retry_delay=15):
    logging.info(f"Attempting to connect to TCPLogServer at {host}:{port}")
    retry_count = 0
    
    # Initial delay to allow MAVSDK_logger.py to start
    logging.info(f"Waiting {retry_delay} seconds before first connection attempt...")
    time.sleep(retry_delay)
    
    while retry_count < max_retries:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(10)
                s.connect((host, port))
                logging.info("Connected to TCP server. Waiting for data...")
                buffer = ''
                last_prediction_time = 0
                while True:
                    data = s.recv(4096)
                    if not data:
                        logging.warning("Connection closed by server.")
                        break
                    buffer += data.decode('utf-8', errors='ignore')
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        try:
                            drone_data = json.loads(line)
                            now = time.time()
                            if now - last_prediction_time >= 60:
                                result = process_live_prediction(drone_data)
                                if "error" not in result:
                                    print(json.dumps(result, indent=2, ensure_ascii=False))
                                    log_prediction_to_db(result)
                                    last_prediction_time = now
                                else:
                                    logging.error(result["error"])
                        except Exception as e:
                            logging.error(f"Error processing data: {e}")
                return
        except ConnectionRefusedError as e:
            retry_count += 1
            logging.error(f"Connection attempt {retry_count}/{max_retries} failed: {e}")
            if retry_count < max_retries:
                logging.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logging.error("Max retries reached. Could not connect to server.")
                sys.exit(1)
        except Exception as e:
            retry_count += 1
            logging.error(f"Connection attempt {retry_count}/{max_retries} failed: {e}")
            if retry_count < max_retries:
                logging.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logging.error("Max retries reached. Could not connect to server.")
                sys.exit(1)

def predict_from_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        for idx in range(len(df)-1, -1, -1):
            row = df.iloc[idx]
            if not pd.isnull(row.get('north_m_s')) or not pd.isnull(row.get('relative_humidity_2m')):
                data = {
                    'timestamp': safe(row.get('timestamp', '')) or datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
                    'relative_humidity_2m': safe(row.get('relative_humidity_2m', 0)) or 0,
                    'pressure_msl': safe(row.get('pressure_msl', 0)) or 0,
                    'north_m_s': safe(row.get('north_m_s', 0)) or 0,
                    'east_m_s': safe(row.get('east_m_s', 0)) or 0,
                    'latitude': safe(row.get('latitude', EXPECTED_COORDS[0])) or EXPECTED_COORDS[0],
                    'longitude': safe(row.get('longitude', EXPECTED_COORDS[1])) or EXPECTED_COORDS[1],
                    'temperature_degc': safe(row.get('temperature_degc', None)),
                    'cloud_cover': safe(row.get('cloud_cover', 0)) or 0
                }
                result = process_live_prediction(data)
                print(json.dumps(result, indent=2, ensure_ascii=False))
                if "error" not in result:
                    log_prediction_to_db(result)
                break
    except Exception as e:
        logging.error(f"Error processing CSV: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Weather prediction TCP client or CSV mode.")
    parser.add_argument('--tcp', action='store_true', help='Run as TCP client')
    parser.add_argument('--csv', type=str, help='Path to CSV for prediction')
    args = parser.parse_args()

    try:
        initialize_database()  # Ensure database is created at start
        if args.tcp:
            run_tcp_client()
        elif args.csv:
            predict_from_csv(args.csv)
        else:
            logging.error("Please use --tcp or --csv <path>")
            sys.exit(1)
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)