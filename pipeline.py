#!/usr/bin/env python3
import subprocess
import sys
import os
import time
import logging
import json
import requests

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

PYTHON = sys.executable

# Helper to run a script and wait for it to finish
def run_script(script, args=None, cwd=None, new_terminal=False, background=False):
    cmd = [PYTHON, script]
    if args:
        cmd += args
    if new_terminal:
        # Use gnome-terminal or x-terminal-emulator for Linux
        terminal_cmd = [
            'gnome-terminal', '--', PYTHON, script
        ]
        if args:
            terminal_cmd += args
        logging.info(f"Launching in new terminal: {' '.join(terminal_cmd)}")
        subprocess.Popen(terminal_cmd, cwd=cwd)
        return None
    if background:
        logging.info(f"Running in background: {' '.join(cmd)}")
        subprocess.Popen(cmd, cwd=cwd)
        return None
    logging.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        logging.error(f"Script failed: {script}")
        sys.exit(result.returncode)

def get_current_country():
    try:
        # Use Open-Meteo IP geolocation API for simplicity
        resp = requests.get('https://ip-api.io/json')
        if resp.status_code == 200:
            data = resp.json()
            return data.get('country_name', None)
    except Exception as e:
        logging.warning(f"Could not determine current country: {e}")
    return None

def get_model_country(metadata_path):
    if not os.path.exists(metadata_path):
        return None
    try:
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
            return meta.get('country', None)
    except Exception as e:
        logging.warning(f"Could not read model metadata: {e}")
        return None

def save_model_country(metadata_path, country):
    try:
        with open(metadata_path, 'w') as f:
            json.dump({'country': country}, f)
    except Exception as e:
        logging.warning(f"Could not save model metadata: {e}")

if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    model_meta_path = os.path.join(root, 'models', 'model_metadata.json')
    current_country = get_current_country()
    model_country = get_model_country(model_meta_path)
    retrain_needed = False
    if current_country is None:
        logging.warning("Could not determine current country. Will retrain model as precaution.")
        retrain_needed = True
    elif model_country is None:
        logging.info("No model metadata found. Will train model for current country.")
        retrain_needed = True
    elif model_country != current_country:
        logging.info(f"Model was trained for {model_country}, but current country is {current_country}. Will retrain.")
        retrain_needed = True
    else:
        logging.info(f"Model is already trained for {model_country}. Skipping retraining.")

    # 1. Fetch historical weather data
    fetch_hist = os.path.join(root, 'ml_weather', 'fetch_historical_data.py')
    run_script(fetch_hist)
    # 2. Fetch wind data (Open-Meteo forecast)
    fetch_wind = os.path.join(root, 'wind_prediction', 'fetch_wind_data.py')
    run_script(fetch_wind)
    # 3. Fetch GFS wind data (optional, comment out if not needed)
    fetch_gfs = os.path.join(root, 'wind_prediction', 'fetch_gfs_wind.py')
    if os.path.exists(fetch_gfs):
        run_script(fetch_gfs)
    # 4. Train classical ML model (only if retrain needed)
    train_model = os.path.join(root, 'ml_weather', 'train_model.py')
    if retrain_needed:
        run_script(train_model)
        save_model_country(model_meta_path, current_country or "Unknown")
    else:
        logging.info(f"Skipping training. Model is for {model_country}.")
    # 5. Train LSTM model (opens in new terminal for interactive input, only if retrain needed)
    train_lstm = os.path.join(root, 'wind_prediction', 'train_lstm.py')
    if retrain_needed:
        run_script(train_lstm, new_terminal=True)
    else:
        logging.info(f"Skipping LSTM training. Model is for {model_country}.")
    # 6. Predict using LSTM (optional, can be run after training)
    predict_lstm = os.path.join(root, 'wind_prediction', 'predict_lstm.py')
    if os.path.exists(predict_lstm):
        run_script(predict_lstm)
    # 7. Merge predictions with logs (optional)
    merge_logs = os.path.join(root, 'wind_prediction', 'merge_predictions_with_logs.py')
    if os.path.exists(merge_logs):
        run_script(merge_logs)
    # 8. Start MAVSDK logger and TCP client in new terminals (if needed)
    mavsdk_logger = os.path.join(root, 'MAVSDK_logger.py')
    tcp_client = os.path.join(root, 'tcp_client.py')
    if os.path.exists(mavsdk_logger):
        run_script(mavsdk_logger, new_terminal=True, background=True)
    if os.path.exists(tcp_client):
        run_script(tcp_client, new_terminal=True, background=True)
    logging.info("Pipeline complete. All main scripts have been executed.")
