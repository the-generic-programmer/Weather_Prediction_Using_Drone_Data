#!/usr/bin/env python3
"""
Full pipeline script for drone weather and wind ML system.
Runs: fetch historical data, train weather model, train wind model, predict weather, predict wind, and test MySQL connection.
"""
import subprocess
import sys
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_WEATHER_DIR = os.path.join(BASE_DIR, 'ml_weather')
DATA_DIR = os.path.join(BASE_DIR, 'data')

steps = [
    ("Fetch historical weather data", [sys.executable, os.path.join(ML_WEATHER_DIR, 'fetch_historical_data.py')]),
    ("Train weather model", [sys.executable, os.path.join(ML_WEATHER_DIR, 'train_model.py')]),
    ("Train wind model", [sys.executable, os.path.join(BASE_DIR, 'train_wind_model.py')]),
    ("Test MySQL connection", [sys.executable, os.path.join(BASE_DIR, 'python3 test_mysql.py')]),
    # Add more steps as needed, e.g. prediction, TCP, etc.
]

def run_step(name, cmd):
    logging.info(f"=== {name} ===")
    try:
        result = subprocess.run(cmd, check=True)
        logging.info(f"{name} completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"{name} failed: {e}")
        sys.exit(1)

def main():
    for name, cmd in steps:
        run_step(name, cmd)
    logging.info("Pipeline completed. You can now run prediction or logging scripts as needed.")

if __name__ == "__main__":
    main()
