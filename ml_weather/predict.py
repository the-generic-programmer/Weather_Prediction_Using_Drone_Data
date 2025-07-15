#!/usr/bin/env python3

import os
import json
import joblib
import socket
import pandas as pd
import numpy as np
import requests
import sys
import logging
import sqlite3
import time
import asyncio
from datetime import datetime, timezone
from astral import LocationInfo
from astral.sun import sun
from timezonefinder import TimezoneFinder
from geopy.geocoders import Nominatim
import pytz

# Try to import wind calculator - make it optional
try:
    from wind_calculator import PrecisionWindCalculator
    WIND_CALCULATOR_AVAILABLE = True
except ImportError:
    WIND_CALCULATOR_AVAILABLE = False
    logging.warning("Wind calculator module not available - wind calculations will be limited")

# Configure logging to show only WARNING and above
logging.basicConfig(
    level=logging.WARNING,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def initialize_database():
    """Initialize the weather_predict SQLite database with updated schema."""
    try:
        conn = sqlite3.connect('weather_predict.db')
        cursor = conn.cursor()
        
        # Drop existing table to ensure correct schema
        cursor.execute('DROP TABLE IF EXISTS predictions')
        
        cursor.execute('''
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                predicted_temperature REAL,
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
            )
        ''')
        conn.commit()
        logging.info("SQLite database weather_predict and table predictions recreated with updated schema")
    except sqlite3.Error as e:
        logging.error(f"Failed to initialize SQLite database: {e}")
        raise
    finally:
        if conn:
            cursor.close()
            conn.close()

def safe(val):
    """Safely convert value to appropriate type."""
    if val is None or (isinstance(val, (int, float)) and pd.isna(val)):
        return None
    if isinstance(val, str) and val.lower() in ("nan", "none", "", "null"):
        return None
    try:
        return float(val) if isinstance(val, (int, float, str)) else val
    except (ValueError, TypeError):
        return val

def safe_float(val, default=0.0):
    """Safely convert value to float with default."""
    safe_val = safe(val)
    return float(safe_val) if safe_val is not None else default

def log_prediction_to_db(result):
    """Log prediction result to SQLite database."""
    try:
        conn = sqlite3.connect('weather_predict.db')
        cursor = conn.cursor()

        ts = result.get("timestamp")
        if ts:
            try:
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                elif not isinstance(ts, datetime):
                    ts = datetime.now(timezone.utc)
            except (ValueError, TypeError):
                ts = datetime.now(timezone.utc)
        else:
            ts = datetime.now(timezone.utc)
        ts_str = ts.isoformat()

        values = (
            ts_str,
            safe_float(result.get('predicted_temperature')),
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
            safe_float(result.get('cloud_cover (%)'))
        )

        cursor.execute('''
            INSERT INTO predictions (
                timestamp, predicted_temperature, humidity,
                wind_speed, wind_direction, confidence_range,
                humidity_timestamp, sunrise_drone, sunset_drone,
                sunrise_user, sunset_user, info,
                humidity_source, drone_location, user_location,
                rain_chance_2h, cloud_cover
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', values)

        conn.commit()
        logging.info("Prediction logged to SQLite database")
    except sqlite3.Error as e:
        logging.error(f"SQLite Database Error: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()

MODEL_DIR = "models"
EXPECTED_COORDS = (13.0, 77.625)  # User location: Bengaluru, India
DRONE_COORDS = (27.6889981, 86.726715)  # Lukla coordinates as fallback

try:
    tf = TimezoneFinder()
    geolocator = Nominatim(user_agent="weather_predictor")
except Exception as e:
    logging.error(f"Failed to initialize timezone finder or geolocator: {e}")
    tf = None
    geolocator = None

def get_timezone(lat, lon):
    """Get timezone for given coordinates."""
    try:
        if tf is None:
            return "Asia/Kolkata" if (lat, lon) == EXPECTED_COORDS else "Asia/Kathmandu"
        timezone_str = tf.timezone_at(lat=lat, lng=lon)
        return timezone_str if timezone_str else ("Asia/Kolkata" if (lat, lon) == EXPECTED_COORDS else "Asia/Kathmandu")
    except Exception as e:
        logging.error(f"Timezone lookup failed: {e}")
        return "Asia/Kolkata" if (lat, lon) == EXPECTED_COORDS else "Asia/Kathmandu"

def get_location_name(lat, lon):
    """Get simplified location name for given coordinates."""
    try:
        if geolocator is None:
            return f"Lat: {lat:.3f}, Lon: {lon:.3f}"
        location = geolocator.reverse((lat, lon), language='en', timeout=10)
        if not location:
            return f"Lat: {lat:.3f}, Lon: {lon:.3f}"
        address = location.raw.get('address', {})
        city = address.get('village', address.get('town', address.get('city', 'Unknown')))
        country = address.get('country', 'Unknown')
        return f"{city}, {country}"
    except Exception as e:
        logging.error(f"Location lookup failed: {e}")
        return f"Lat: {lat:.3f}, Lon: {lon:.3f}"

def get_weather_data(lat, lon):
    """Fetch current weather data from API, including temperature, humidity, and cloud cover."""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,cloudcover"
        response = requests.get(url, timeout=10, verify=False)  # Disable SSL verification
        response.raise_for_status()
        data = response.json()
        
        if "current" in data:
            temp = data["current"].get("temperature_2m", 15)
            rh = data["current"].get("relative_humidity_2m", 80)  # Default for Lukla
            cloud_cover = data["current"].get("cloudcover", 50)  # Default for Lukla
            rh_time = data["current"].get("time", datetime.now(timezone.utc).isoformat())
            return temp, rh, cloud_cover, rh_time, url
        else:
            logging.error(f"Invalid Open-Meteo response: {data}")
            return 15, 80, 50, datetime.now(timezone.utc).isoformat(), url
    except requests.RequestException as e:
        logging.error(f"HTTP error fetching weather data: {e}")
        return 15, 80, 50, datetime.now(timezone.utc).isoformat(), "API_ERROR"
    except Exception as e:
        logging.error(f"Failed to fetch weather data: {e}")
        return 15, 80, 50, datetime.now(timezone.utc).isoformat(), "API_ERROR"

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
        response = requests.get(url, timeout=10, verify=False)  # Disable SSL verification
        response.raise_for_status()
        data = response.json()
        
        hourly_data = data.get("hourly", {})
        times = hourly_data.get("time", [])
        chances = hourly_data.get("precipitation_probability", [])
        
        if not isinstance(times, list) or not isinstance(chances, list) or len(times) != len(chances):
            logging.error(f"Malformed rain chance data: times={len(times) if isinstance(times, list) else 'not list'}, chances={len(chances) if isinstance(chances, list) else 'not list'}")
            return 50  # Default for Lukla
        
        target_time = future_time.strftime("%Y-%m-%dT%H:00")
        for i, t in enumerate(times):
            if t >= target_time:
                rain_chance = chances[i] if chances[i] is not None else 50
                return rain_chance
                
        logging.warning("No matching future time found for rain chance")
        return 50
    except requests.RequestException as e:
        logging.error(f"HTTP error fetching rain chance: {e}")
        return 50
    except Exception as e:
        logging.error(f"Failed to fetch rain chance: {e}")
        return 50

def format_humidity_time(iso):
    """Format ISO timestamp to readable format."""
    try:
        if isinstance(iso, str):
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
            
            self.feature_names = list(self.scaler.feature_names_in_) if hasattr(self.scaler, 'feature_names_in_') else [
                'hour', 'month', 'relative_humidity_2m', 'pressure_msl',
                'wind_speed_10m', 'wind_gusts_10m', 'wind_direction_10m', 'cloudcover'
            ]
            
            # Initialize wind calculator if available
            if WIND_CALCULATOR_AVAILABLE:
                try:
                    self.wind_calculator = PrecisionWindCalculator()
                except Exception as wind_err:
                    logging.warning(f"Wind calculator failed to load: {wind_err}")
                    self.wind_calculator = None
            else:
                self.wind_calculator = None
                
            # Load latest wind measurements
            self.latest_wind = None
            try:
                with open("wind_measurements.json", 'r') as f:
                    data = json.load(f)
                    if data:
                        self.latest_wind = data[-1]
            except (FileNotFoundError, json.JSONDecodeError):
                logging.warning("No wind measurements file found")
                
        except Exception as e:
            logging.error(f"Failed to load model or scaler: {e}")
            raise

    def prepare_drone_data(self, drone_data: dict, rh: float = 80, cloud_cover: float = 50, wind_speed: float = 0, wind_direction: float = 0) -> pd.DataFrame:
        """Prepare drone data for prediction."""
        current_time = datetime.now(timezone.utc)
        
        wind_gusts = wind_speed
        
        # Use latest wind measurements if available
        if self.latest_wind:
            wind_speed = safe_float(self.latest_wind.get('speed_knots', 0)) * 0.514444  # knots to m/s
            wind_direction = safe_float(self.latest_wind.get('direction_degrees', 0))
            wind_gusts = max(wind_speed, wind_gusts)
        
        # Use wind calculator if available
        if self.wind_calculator and WIND_CALCULATOR_AVAILABLE:
            try:
                # Use synchronous call or handle async properly
                if hasattr(self.wind_calculator, 'add_drone_data'):
                    # Check if it's async
                    if asyncio.iscoroutinefunction(self.wind_calculator.add_drone_data):
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # Create a new task if loop is running
                            task = loop.create_task(self.wind_calculator.add_drone_data(drone_data))
                        else:
                            loop.run_until_complete(self.wind_calculator.add_drone_data(drone_data))
                    else:
                        self.wind_calculator.add_drone_data(drone_data)
                
                if hasattr(self.wind_calculator, 'calculate_wind_multi_method'):
                    if asyncio.iscoroutinefunction(self.wind_calculator.calculate_wind_multi_method):
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # For running loop, we'll skip this and use fallback
                            pass
                        else:
                            wind_est = loop.run_until_complete(self.wind_calculator.calculate_wind_multi_method())
                            if wind_est:
                                wind_speed = wind_est.speed_knots * 0.514444
                                wind_direction = wind_est.direction_degrees
                                wind_gusts = max(wind_speed, wind_gusts)
                    else:
                        wind_est = self.wind_calculator.calculate_wind_multi_method()
                        if wind_est:
                            wind_speed = wind_est.speed_knots * 0.514444
                            wind_direction = wind_est.direction_degrees
                            wind_gusts = max(wind_speed, wind_gusts)
            except Exception as e:
                logging.warning(f"Wind calculator error: {e}")
                pass  # Fall back to saved wind data
        
        features = {
            'hour': current_time.hour,
            'month': current_time.month,
            'relative_humidity_2m': rh,
            'pressure_msl': safe_float(drone_data.get('pressure_msl', 1013)),
            'wind_speed_10m': wind_speed,
            'wind_gusts_10m': wind_gusts,
            'wind_direction_10m': wind_direction,
            'cloudcover': cloud_cover
        }
        
        # Handle additional features that might be required
        for feature_name in self.feature_names:
            if feature_name not in features:
                if feature_name == 'altitude':
                    features[feature_name] = safe_float(drone_data.get('altitude', 0))
                elif feature_name in ['temperature_2m', 'temp']:
                    features[feature_name] = safe_float(drone_data.get('temperature_degc', 20))
                elif feature_name in ['precipitation', 'rain']:
                    features[feature_name] = 0.0
                else:
                    features[feature_name] = 0.0
        
        return pd.DataFrame([features])

    def predict(self, features: pd.DataFrame) -> float:
        """Make prediction."""
        try:
            if hasattr(self.scaler, 'feature_names_in_'):
                features = features.reindex(columns=self.scaler.feature_names_in_, fill_value=0.0)
            
            X_scaled = self.scaler.transform(features)
            prediction = self.model.predict(X_scaled)[0]
            if not (0 <= prediction <= 40):
                prediction = 15.0  # Default for Lukla
            return float(prediction)
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            raise

def process_live_prediction(data: dict):
    """Process live prediction."""
    try:
        required_fields = ['latitude', 'longitude']
        missing_fields = [field for field in required_fields if safe(data.get(field)) is None]
        
        if missing_fields:
            logging.warning(f"Missing required fields: {missing_fields}")
            return {"error": f"Missing required data fields: {missing_fields}", "input": data}

        predictor = WeatherPredictor()
        
        lat = safe_float(data.get("latitude", DRONE_COORDS[0]))
        lon = safe_float(data.get("longitude", DRONE_COORDS[1]))
        
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            logging.warning(f"Invalid coordinates: lat={lat}, lon={lon}")
            lat, lon = DRONE_COORDS
        
        # Get weather data
        temp, rh, cloud_cover_percent, rh_time, rh_url = get_weather_data(lat, lon)
        rain_chance_2h = get_rain_chance_in_2_hours(lat, lon)
        
        # Initialize wind values
        wind_speed_knots = 0.0
        wind_direction_degrees = 0.0
        wind_speed_ms = 0.0
        
        # Use latest wind measurements if available
        if predictor.latest_wind:
            wind_speed_knots = safe_float(predictor.latest_wind.get('speed_knots', 0))
            wind_direction_degrees = safe_float(predictor.latest_wind.get('direction_degrees', 0))
            wind_speed_ms = wind_speed_knots * 0.514444
        
        # Use wind calculator if available
        if predictor.wind_calculator and WIND_CALCULATOR_AVAILABLE:
            try:
                # Handle async properly
                if hasattr(predictor.wind_calculator, 'add_drone_data'):
                    if asyncio.iscoroutinefunction(predictor.wind_calculator.add_drone_data):
                        try:
                            loop = asyncio.get_event_loop()
                            if not loop.is_running():
                                loop.run_until_complete(predictor.wind_calculator.add_drone_data(data))
                        except RuntimeError:
                            # No event loop, create one
                            asyncio.run(predictor.wind_calculator.add_drone_data(data))
                    else:
                        predictor.wind_calculator.add_drone_data(data)
                
                if hasattr(predictor.wind_calculator, 'calculate_wind_multi_method'):
                    if asyncio.iscoroutinefunction(predictor.wind_calculator.calculate_wind_multi_method):
                        try:
                            loop = asyncio.get_event_loop()
                            if not loop.is_running():
                                wind_est = loop.run_until_complete(predictor.wind_calculator.calculate_wind_multi_method())
                            else:
                                wind_est = None  # Skip if loop is running
                        except RuntimeError:
                            wind_est = asyncio.run(predictor.wind_calculator.calculate_wind_multi_method())
                    else:
                        wind_est = predictor.wind_calculator.calculate_wind_multi_method()
                    
                    if wind_est:
                        wind_speed_knots = wind_est.speed_knots
                        wind_direction_degrees = wind_est.direction_degrees
                        wind_speed_ms = wind_est.speed_knots * 0.514444
            except Exception as e:
                logging.warning(f"Wind calculator error: {e}")
                pass  # Fall back to latest_wind or defaults
        
        # Prepare features and make prediction
        features = predictor.prepare_drone_data(data, rh=rh, cloud_cover=cloud_cover_percent, wind_speed=wind_speed_ms, wind_direction=wind_direction_degrees)
        prediction = predictor.predict(features)
        
        # Validate prediction
        if not (0 <= prediction <= 40):
            logging.warning(f"Model prediction {prediction}°C unrealistic, using API temperature {temp}°C")
            prediction = temp
        
        # Format humidity time
        humidity_time_fmt = format_humidity_time(rh_time)
        
        # Get location information
        drone_tz = get_timezone(lat, lon)
        user_tz = get_timezone(*EXPECTED_COORDS)
        drone_loc = get_location_name(lat, lon)
        user_loc = get_location_name(*EXPECTED_COORDS)
        
        # Get sunrise/sunset times
        drone_sun = get_sunrise_sunset(lat, lon, drone_tz)
        user_sun = get_sunrise_sunset(*EXPECTED_COORDS, user_tz)
        
        # Calculate confidence interval
        try:
            preds = [predictor.predict(features) for _ in range(5)]
            std_dev = np.std(preds)
            ci = max(round(1.96 * std_dev, 2), 0.1)
        except Exception as e:
            logging.warning(f"Failed to calculate confidence interval: {e}")
            ci = 0.1
        
        # Handle timestamp
        now = datetime.now(timezone.utc)
        timestamp = safe(data.get("timestamp"))
        if timestamp is None:
            timestamp = now.strftime("%Y-%m-%dT%H:%M:%S")
        
        # Create output dictionary
        output = {
            "timestamp": timestamp,
            "rain_chance_2h (%)": float(rain_chance_2h),
            "cloud_cover (%)": float(cloud_cover_percent),
            "predicted_value": round(prediction, 2),
            "confidence_range": ci,
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
            "predicted_temperature": round(prediction, 2),
            "confidence_range (±°C)": ci
        }
        
        # Print wind speed and direction in terminal output
        print(f"Wind Speed: {wind_speed_knots:.2f} knots")
        print(f"Wind Direction: {wind_direction_degrees:.1f}°")
        
        return output

    except Exception as e:
        logging.error(f"Prediction Error: {e}")
        return {"error": f"Prediction Error: {str(e)}", "input": data}

def run_tcp_client(host='127.0.0.1', port=9000, retry_delay=5):
    """Run TCP client to receive drone data continuously."""
    logging.warning(f"Attempting to connect to TCPLogServer at {host}:{port}")
    
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(10)
                s.connect((host, port))
                logging.warning("Connected to TCP server. Starting data processing...")
                
                buffer = ''
                last_prediction_time = 0
                
                while True:
                    try:
                        data = s.recv(4096)
                        if not data:
                            logging.warning("Connection closed by server. Reconnecting...")
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
                                
                                if now - last_prediction_time >= 60:  # Predict every 60 seconds
                                    result = process_live_prediction(drone_data)
                                    if "error" not in result:
                                        # Print all output including wind speed/direction
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
                        continue
                        
        except ConnectionRefusedError as e:
            logging.error(f"Connection failed: {e}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        except Exception as e:
            logging.error(f"Unexpected error: {e}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

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
            
            data = {}
            for col in df.columns:
                val = row.get(col)
                if pd.notna(val):
                    data[col] = val
            
            if data:
                # Add default values if missing
                if 'timestamp' not in data:
                    data['timestamp'] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
                
                if 'latitude' not in data:
                    data['latitude'] = DRONE_COORDS[0]
                if 'longitude' not in data:
                    data['longitude'] = DRONE_COORDS[1]
                
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
        initialize_database()
        
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