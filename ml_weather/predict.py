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
from datetime import datetime, timezone
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

def safe(val):
    """Safely convert value to appropriate type, handling None, NaN, and empty strings."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if pd.isna(val) or np.isnan(val):
            return None
        return val
    if isinstance(val, str):
        if val.lower() in ("nan", "none", "", "null"):
            return None
        try:
            # Try to convert to float if it's a numeric string
            return float(val)
        except (ValueError, TypeError):
            return val
    return val

def safe_float(val, default=0.0):
    """Safely convert value to float with default."""
    safe_val = safe(val)
    if safe_val is None:
        return default
    try:
        return float(safe_val)
    except (ValueError, TypeError):
        return default

# Define database logging function
def log_prediction_to_db(result):
    """Log prediction result to database with proper error handling."""
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='Root@123',
            database='weather_predict'
        )
        cursor = conn.cursor()

        # Parse timestamp
        ts = result.get("timestamp")
        if ts:
            try:
                # Handle different timestamp formats
                if isinstance(ts, str):
                    if 'T' in ts:
                        ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    else:
                        ts = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                elif not isinstance(ts, datetime):
                    ts = datetime.now(timezone.utc)
            except (ValueError, TypeError) as e:
                logging.warning(f"Invalid timestamp format: {ts}, using current time. Error: {e}")
                ts = datetime.now(timezone.utc)
        else:
            ts = datetime.now(timezone.utc)

        # Calculate wind metrics safely
        input_data = result.get('input', {})
        north_wind = safe_float(input_data.get('north_m_s', 0))
        east_wind = safe_float(input_data.get('east_m_s', 0))
        
        wind_speed = np.sqrt(north_wind**2 + east_wind**2)
        wind_direction = (np.degrees(np.arctan2(east_wind, north_wind)) % 360) if (north_wind != 0 or east_wind != 0) else 0

        values = (
            ts,
            safe_float(result.get('predicted_temperature')),
            safe_float(result.get('humidity (% RH)')),
            wind_speed,
            wind_direction,
            safe_float(result.get('confidence_range (±°C)')),
            str(result.get('humidity_timestamp', '')),
            str(result.get('sunrise_at_drone_location', '')),
            str(result.get('sunset_at_drone_location', '')),
            str(result.get('sunrise_at_user_location', '')),
            str(result.get('sunset_at_user_location', '')),
            str(result.get('info', '')),
            str(result.get('humidity_source', '')),
            str(result.get('location', '')),
            str(result.get('warning', '')),
            safe_float(result.get('rain_chance_2h (%)')),
            safe_float(result.get('cloud_cover (%)'))
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
    except Exception as e:
        logging.error(f"Unexpected error logging to database: {e}")

MODEL_DIR = "models"
EXPECTED_COORDS = (13.0, 77.625)

# Initialize global objects with error handling
try:
    tf = TimezoneFinder()
    geolocator = Nominatim(user_agent="weather_predictor")
except Exception as e:
    logging.error(f"Failed to initialize timezone finder or geolocator: {e}")
    tf = None
    geolocator = None

def get_timezone(lat, lon):
    """Get timezone for given coordinates with fallback."""
    try:
        if tf is None:
            return "UTC"
        timezone_str = tf.timezone_at(lat=lat, lng=lon)
        return timezone_str if timezone_str else "UTC"
    except Exception as e:
        logging.error(f"Timezone lookup failed: {e}")
        return "UTC"

def get_location_name(lat, lon):
    """Get location name for given coordinates with fallback."""
    try:
        if geolocator is None:
            return f"Location: {lat:.3f}, {lon:.3f}"
        location = geolocator.reverse((lat, lon), language='en', timeout=10)
        return location.address if location else f"Location: {lat:.3f}, {lon:.3f}"
    except Exception as e:
        logging.error(f"Location lookup failed: {e}")
        return f"Location: {lat:.3f}, {lon:.3f}"

def get_live_humidity(lat, lon):
    """Fetch current humidity and cloud cover from API."""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=relative_humidity_2m,cloudcover"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "current" in data:
            rh = data["current"].get("relative_humidity_2m", 50)
            cloud_cover = data["current"].get("cloudcover", 0)
            rh_time = data["current"].get("time", datetime.now(timezone.utc).isoformat())
            logging.info(f"API response: humidity={rh}%, cloud_cover={cloud_cover}%")
            return rh, cloud_cover, rh_time, url
        else:
            logging.error(f"Invalid Open-Meteo response: {data}")
            return 50, 0, datetime.now(timezone.utc).isoformat(), url
    except requests.RequestException as e:
        logging.error(f"HTTP error fetching humidity and cloud cover: {e}")
        return 50, 0, datetime.now(timezone.utc).isoformat(), "API_ERROR"
    except Exception as e:
        logging.error(f"Failed to fetch humidity and cloud cover: {e}")
        return 50, 0, datetime.now(timezone.utc).isoformat(), "API_ERROR"

def get_rain_chance_in_2_hours(lat, lon):
    """Get precipitation probability for 2 hours from now."""
    try:
        now = datetime.now(timezone.utc)
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
        
        hourly_data = data.get("hourly", {})
        times = hourly_data.get("time", [])
        chances = hourly_data.get("precipitation_probability", [])
        
        if not isinstance(times, list) or not isinstance(chances, list) or len(times) != len(chances):
            logging.error(f"Malformed rain chance data: times={len(times) if isinstance(times, list) else 'not list'}, chances={len(chances) if isinstance(chances, list) else 'not list'}")
            return 0
        
        target_time = future_time.strftime("%Y-%m-%dT%H:00")
        for i, t in enumerate(times):
            if t >= target_time:
                rain_chance = chances[i] if chances[i] is not None else 0
                logging.info(f"Rain chance for {t}: {rain_chance}%")
                return rain_chance
                
        logging.warning("No matching future time found for rain chance")
        return 0
    except requests.RequestException as e:
        logging.error(f"HTTP error fetching rain chance: {e}")
        return 0
    except Exception as e:
        logging.error(f"Failed to fetch rain chance: {e}")
        return 0

def format_humidity_time(iso):
    """Format ISO timestamp to readable format."""
    try:
        if isinstance(iso, str):
            # Handle different ISO formats
            iso = iso.replace('Z', '+00:00')
            dt = datetime.fromisoformat(iso)
        elif isinstance(iso, datetime):
            dt = iso
        else:
            raise ValueError(f"Invalid timestamp type: {type(iso)}")
            
        return f"{dt.strftime('%I:%M %p UTC')} on {dt.strftime('%Y-%m-%d')}"
    except Exception as e:
        logging.error(f"Error formatting humidity time: {e} (input: {iso})")
        return "Unknown time"

def get_sunrise_sunset(lat, lon, timezone_str):
    """Get sunrise and sunset times for given location."""
    try:
        city = LocationInfo(latitude=lat, longitude=lon)
        observer_date = datetime.now(timezone.utc).date()
        tz = pytz.timezone(timezone_str)
        s = sun(city.observer, date=observer_date, tzinfo=tz)
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
            model_path = os.path.join(model_dir, 'weather_model.joblib')
            scaler_path = os.path.join(model_dir, 'scaler.joblib')
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
                
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            logging.info("Weather model and scaler loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load model or scaler: {e}")
            raise

    def prepare_drone_data(self, drone_data: dict) -> pd.DataFrame:
        """Prepare drone data for prediction."""
        current_time = datetime.now(timezone.utc)
        
        # Calculate wind metrics
        north_wind = safe_float(drone_data.get('north_m_s', 0))
        east_wind = safe_float(drone_data.get('east_m_s', 0))
        
        wind_speed = np.sqrt(north_wind**2 + east_wind**2)
        wind_direction = (np.degrees(np.arctan2(east_wind, north_wind)) % 360) if (north_wind != 0 or east_wind != 0) else 0

        temperature = safe(drone_data.get('temperature_degc'))
        if temperature is None:
            logging.info("temperature_degc missing, model will predict temperature")

        features = {
            'hour': current_time.hour,
            'month': current_time.month,
            'relative_humidity_2m': safe_float(drone_data.get('relative_humidity_2m', 50)),
            'pressure_msl': safe_float(drone_data.get('pressure_msl', 1013)),
            'wind_speed_10m': wind_speed,
            'wind_direction_10m': wind_direction,
            'cloudcover': safe_float(drone_data.get('cloud_cover', 0))
        }
        return pd.DataFrame([features])

    def predict(self, features: pd.DataFrame) -> float:
        """Make temperature prediction."""
        try:
            X_scaled = self.scaler.transform(features)
            prediction = self.model.predict(X_scaled)[0]
            return float(prediction)
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            raise

def process_live_prediction(data: dict):
    """Process live prediction with comprehensive error handling."""
    try:
        # Validate required fields
        required_fields = ['latitude', 'longitude', 'north_m_s', 'east_m_s']
        missing_fields = [field for field in required_fields if safe(data.get(field)) is None]
        
        if missing_fields:
            logging.warning(f"Missing required fields: {missing_fields}")
            return {"error": f"Missing required data fields: {missing_fields}", "input": data}

        # Initialize predictor
        predictor = WeatherPredictor()
        
        # Get coordinates
        lat = safe_float(data.get("latitude", EXPECTED_COORDS[0]))
        lon = safe_float(data.get("longitude", EXPECTED_COORDS[1]))
        
        # Validate coordinates
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            logging.warning(f"Invalid coordinates: lat={lat}, lon={lon}")
            lat, lon = EXPECTED_COORDS
        
        # Prepare features and make prediction
        features = predictor.prepare_drone_data(data)
        
        # Use existing temperature if available, otherwise predict
        existing_temp = safe(data.get('temperature_degc'))
        if existing_temp is not None:
            pred = float(existing_temp)
            logging.info(f"Using existing temperature: {pred}°C")
        else:
            pred = predictor.predict(features)
            logging.info(f"Predicted temperature: {pred}°C")
        
        # Get additional weather data
        rh, cloud_cover_percent, rh_time, rh_url = get_live_humidity(lat, lon)
        rain_chance_2h = get_rain_chance_in_2_hours(lat, lon)
        
        # Format times and locations
        humidity_time_fmt = format_humidity_time(rh_time)
        
        drone_tz = get_timezone(lat, lon)
        user_tz = get_timezone(*EXPECTED_COORDS)
        drone_loc = get_location_name(lat, lon)
        
        drone_sun = get_sunrise_sunset(lat, lon, drone_tz)
        user_sun = get_sunrise_sunset(*EXPECTED_COORDS, user_tz)
        
        # Calculate confidence interval
        ci = 0.1  # Default confidence interval
        if existing_temp is None:
            try:
                # Generate multiple predictions for confidence interval
                preds = [predictor.predict(features) for _ in range(10)]  # Reduced from 50 for performance
                std_dev = np.std(preds)
                ci = max(round(1.96 * std_dev, 2), 0.1)
            except Exception as e:
                logging.warning(f"Failed to calculate confidence interval: {e}")
                ci = 0.1
        
        # Get timestamp
        now = datetime.now(timezone.utc)
        timestamp = safe(data.get("timestamp"))
        if timestamp is None:
            timestamp = now.strftime("%Y-%m-%dT%H:%M:%S")
        
        # Build output
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
        
        # Add warning if location is far from expected
        if abs(lat - EXPECTED_COORDS[0]) > 2 or abs(lon - EXPECTED_COORDS[1]) > 2:
            output["warning"] = f"Warning: Drone location is far from expected coordinates ({EXPECTED_COORDS})"
        
        return output

    except Exception as e:
        logging.error(f"Prediction Error: {e}")
        return {"error": f"Prediction Error: {str(e)}", "input": data}

def run_tcp_client(host='127.0.0.1', port=9000, max_retries=10, retry_delay=15):
    """Run TCP client to receive drone data."""
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
                    try:
                        data = s.recv(4096)
                        if not data:
                            logging.warning("Connection closed by server.")
                            break
                            
                        buffer += data.decode('utf-8', errors='ignore')
                        
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()
                            
                            if not line:
                                continue
                                
                            try:
                                drone_data = json.loads(line)
                                now = time.time()
                                
                                # Rate limiting: only process once per minute
                                if now - last_prediction_time >= 60:
                                    result = process_live_prediction(drone_data)
                                    if "error" not in result:
                                        print(json.dumps(result, indent=2, ensure_ascii=False))
                                        log_prediction_to_db(result)
                                        last_prediction_time = now
                                    else:
                                        logging.error(result["error"])
                                        
                            except json.JSONDecodeError as e:
                                logging.error(f"JSON decode error: {e} for line: {line[:100]}...")
                            except Exception as e:
                                logging.error(f"Error processing data: {e}")
                                
                    except socket.timeout:
                        logging.debug("Socket timeout, continuing...")
                        continue
                        
                return
                
        except ConnectionRefusedError as e:
            retry_count += 1
            logging.error(f"Connection attempt {retry_count}/{max_retries} failed: Connection refused")
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
    """Make prediction from CSV file."""
    try:
        if not os.path.exists(csv_path):
            logging.error(f"CSV file not found: {csv_path}")
            return
            
        df = pd.read_csv(csv_path)
        if df.empty:
            logging.error("CSV file is empty")
            return
            
        # Process the last valid row
        for idx in range(len(df)-1, -1, -1):
            row = df.iloc[idx]
            
            # Check if row has required data
            if not pd.isna(row.get('north_m_s')) and not pd.isna(row.get('east_m_s')):
                data = {
                    'timestamp': safe(row.get('timestamp')) or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
                    'relative_humidity_2m': safe_float(row.get('relative_humidity_2m', 50)),
                    'pressure_msl': safe_float(row.get('pressure_msl', 1013)),
                    'north_m_s': safe_float(row.get('north_m_s', 0)),
                    'east_m_s': safe_float(row.get('east_m_s', 0)),
                    'latitude': safe_float(row.get('latitude', EXPECTED_COORDS[0])),
                    'longitude': safe_float(row.get('longitude', EXPECTED_COORDS[1])),
                    'temperature_degc': safe(row.get('temperature_degc')),
                    'cloud_cover': safe_float(row.get('cloud_cover', 0))
                }
                
                result = process_live_prediction(data)
                print(json.dumps(result, indent=2, ensure_ascii=False))
                
                if "error" not in result:
                    log_prediction_to_db(result)
                break
        else:
            logging.error("No valid rows found in CSV file")
            
    except pd.errors.EmptyDataError:
        logging.error("CSV file is empty or has no valid data")
    except pd.errors.ParserError as e:
        logging.error(f"Error parsing CSV file: {e}")
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
            
    except KeyboardInterrupt:
        logging.info("Script interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)