#!/usr/bin/env python3

import requests
import pandas as pd
from datetime import datetime, timedelta
import os

def fetch_historical_weather(latitude: float, longitude: float, 
                           start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical weather data from Open-Meteo API
    
    Args:
        latitude (float): Location latitude
        longitude (float): Location longitude
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
    
    Returns:
        pd.DataFrame: DataFrame containing hourly weather data
    """
    
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "pressure_msl",
            "wind_speed_10m",
            "wind_direction_10m",
            "precipitation",
            "cloud_cover"
        ]
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Defensive: check for 'hourly' in data
        if "hourly" not in data or not isinstance(data["hourly"], dict):
            print(f"Malformed API response: {data}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data["hourly"])
        
        # Defensive: check for required columns
        required_cols = ["time", "temperature_2m", "relative_humidity_2m", "pressure_msl", "wind_speed_10m", "wind_direction_10m", "cloud_cover"]
        for col in required_cols:
            if col not in df.columns:
                print(f"Missing column in API data: {col}")
                return None
        
        # Convert time to datetime
        df["time"] = pd.to_datetime(df["time"])
        
        return df
        
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return None

def main():
    # Example coordinates (you can change these)
    LATITUDE = 51.5074  # London coordinates
    LONGITUDE = -0.1278
    
    # Get data for last 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)
    
    # Format dates
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    print(f"Fetching weather data from {start_str} to {end_str}")
    
    # Fetch data
    df = fetch_historical_weather(LATITUDE, LONGITUDE, start_str, end_str)
    
    if df is not None:
        # Drop rows with NaN in required columns
        df = df.dropna(subset=["temperature_2m", "relative_humidity_2m", "pressure_msl", "wind_speed_10m", "wind_direction_10m", "cloud_cover"])
        
        # Create output directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Save to CSV
        output_file = "data/historical_weather.csv"
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
        print(f"Total records: {len(df)}")
    else:
        print("Failed to fetch or process weather data.")

if __name__ == "__main__":
    main()
