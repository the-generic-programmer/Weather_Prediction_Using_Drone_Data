#!/usr/bin/env python3

import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import argparse

# Supported variables for Open-Meteo historical API (July 2025)
SUPPORTED_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "pressure_msl",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "precipitation",
    "precipitation_hours",
    "weathercode",
    "cloudcover",
    "cloudcover_low",
    "cloudcover_mid",
    "cloudcover_high",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "evapotranspiration",
    "et0_fao_evapotranspiration",
    "vapor_pressure_deficit",
    "cape",
    "snowfall",
    "snow_depth",
    "freezinglevel_height",
    "visibility",
    "soil_temperature_0cm",
    "soil_moisture_0_1cm"
]

def fetch_historical_weather(latitude: float, longitude: float, 
                           start_date: str, end_date: str, 
                           variables=None) -> pd.DataFrame:
    """
    Fetch historical weather data from Open-Meteo API
    
    Args:
        latitude (float): Location latitude
        longitude (float): Location longitude
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        variables (list): List of variable names to fetch (Open-Meteo API names)
    
    Returns:
        pd.DataFrame: DataFrame containing hourly weather data
    """
    if variables is None:
        variables = [
            "temperature_2m",
            "relative_humidity_2m",
            "pressure_msl",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "precipitation",
            "cloudcover"
        ]
        # Wind features (wind_speed_10m, wind_direction_10m, wind_gusts_10m) are always included for wind prediction tasks
    # Validate variables
    valid_vars = [v for v in variables if v in SUPPORTED_VARIABLES]
    invalid_vars = [v for v in variables if v not in SUPPORTED_VARIABLES]
    if invalid_vars:
        print(f"Warning: The following variables are not supported and will be ignored: {invalid_vars}")
    if not valid_vars:
        print("Error: No valid variables specified. Exiting.")
        return None
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": valid_vars
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
        required_cols = ["time"] + valid_vars
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
    parser = argparse.ArgumentParser(description="Fetch historical weather data from Open-Meteo API.")
    parser.add_argument('--latitude', type=float, default=51.5074, help='Latitude (default: 51.5074)')
    parser.add_argument('--longitude', type=float, default=-0.1278, help='Longitude (default: -0.1278)')
    parser.add_argument('--years', type=int, default=5, help='Number of years of data to fetch (default: 5)')
    parser.add_argument('--variables', nargs='+', default=None, help='Variables to fetch (default: all core variables)')
    args = parser.parse_args()

    LATITUDE = args.latitude
    LONGITUDE = args.longitude
    years = args.years
    variables = args.variables

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*years)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    print(f"Fetching weather data from {start_str} to {end_str}")
    if variables:
        print(f"Requested variables: {variables}")
    else:
        print("Variables: default (core supported variables)")

    df = fetch_historical_weather(LATITUDE, LONGITUDE, start_str, end_str, variables)

    if df is not None:
        # Drop rows with NaN in required columns
        drop_cols = [v for v in (variables if variables else [
            "temperature_2m", "relative_humidity_2m", "pressure_msl", "wind_speed_10m", "wind_direction_10m", "cloudcover", "precipitation"
        ]) if v in df.columns]
        df = df.dropna(subset=drop_cols)
        os.makedirs("data", exist_ok=True)
        output_file = "data/historical_weather.csv"
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
        print(f"Total records: {len(df)}")
    else:
        print("Failed to fetch or process weather data.")

if __name__ == "__main__":
    main()
