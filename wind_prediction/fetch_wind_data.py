import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import logging

def fetch_openmeteo_wind_data(lat, lon, hours_ahead=48, save_to_file=True):
    """Fetch wind data from Open-Meteo and save as training-ready CSV with extra features."""
    now = datetime.utcnow()
    end_time = now + timedelta(hours=hours_ahead)
    start_date = now.strftime("%Y-%m-%d")
    end_date = end_time.strftime("%Y-%m-%d")
    url = (
        f"https://api.open-meteo.com/v1/ecmwf?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=windspeed_10m,winddirection_10m,temperature_2m,relative_humidity_2m,pressure_msl,cloudcover,precipitation,weathercode"
        f"&start_date={start_date}&end_date={end_date}&timezone=UTC"
    )
    logging.info(f"Fetching ECMWF forecast from: {url}")
    resp = requests.get(url, timeout=10)
    data = resp.json()
    if "hourly" not in data:
        raise ValueError("Unexpected forecast response format.")
    time_series = data["hourly"]["time"]
    speeds = data["hourly"]["windspeed_10m"]
    directions = data["hourly"]["winddirection_10m"]
    temp = data["hourly"].get("temperature_2m", [None]*len(time_series))
    rh = data["hourly"].get("relative_humidity_2m", [None]*len(time_series))
    pressure = data["hourly"].get("pressure_msl", [None]*len(time_series))
    cloud = data["hourly"].get("cloudcover", [None]*len(time_series))
    precip = data["hourly"].get("precipitation", [None]*len(time_series))
    wcode = data["hourly"].get("weathercode", [None]*len(time_series))
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(time_series),
        "forecast_wind_speed": speeds,
        "forecast_wind_direction": directions,
        "temperature_2m": temp,
        "relative_humidity_2m": rh,
        "pressure_msl": pressure,
        "cloudcover": cloud,
        "precipitation": precip,
        "weathercode": wcode
    })
    if save_to_file:
        os.makedirs("data", exist_ok=True)
        filename = f"data/openmeteo_forecast_{lat}_{lon}_{now.strftime('%Y%m%dT%H%M%S')}_train.csv"
        df.to_csv(filename, index=False)
        logging.info(f"Forecast data saved to {filename}")
    return df

if __name__ == "__main__":
    # Example: London
    lat, lon = 51.5074, -0.1278
    fetch_openmeteo_wind_data(lat, lon)
