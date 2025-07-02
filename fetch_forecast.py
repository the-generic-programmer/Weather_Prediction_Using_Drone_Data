import requests
import pandas as pd
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)

def fetch_ecmwf_forecast(lat, lon, hours_ahead=12):
    try:
        now = datetime.utcnow()
        end_time = now + timedelta(hours=hours_ahead)
        start_date = now.strftime("%Y-%m-%d")
        end_date = end_time.strftime("%Y-%m-%d")

        url = (
            f"https://api.open-meteo.com/v1/ecmwf?"
            f"latitude={lat}&longitude={lon}"
            f"&hourly=windspeed_10m,winddirection_10m"
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

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(time_series),
            "wind_speed": speeds,
            "wind_direction": directions
        }).set_index("timestamp")

        df_30min = df.resample("30T").interpolate(method='linear').reset_index()
        df_30min = df_30min[df_30min["timestamp"] <= end_time]
        return df_30min

    except Exception as e:
        logging.error(f"Failed to fetch or process ECMWF forecast: {e}")
        return pd.DataFrame()