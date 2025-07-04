import pandas as pd
import warnings
from meteostat import Point, Hourly
from datetime import datetime, timedelta
import os
import argparse

def fetch_meteostat_hourly(lat, lon, start_date, end_date, output_csv):
    try:
        location = Point(lat, lon)
        data = Hourly(location, start=start_date, end=end_date)
        df = data.fetch()
        if df.empty:
            warnings.warn(f"No Meteostat data returned for {lat},{lon} from {start_date} to {end_date}")
            if os.path.exists(output_csv):
                os.remove(output_csv)
            print(f"No data fetched for {lat},{lon}. File not created.")
            return False
        else:
            df.reset_index(inplace=True)
            df.to_csv(output_csv, index=False)
            print(f"Saved Meteostat data to {output_csv}")
            print(df.head())
            return True
    except Exception as e:
        warnings.warn(f"Error fetching Meteostat data: {e}")
        if os.path.exists(output_csv):
            os.remove(output_csv)
        print(f"Error fetching data, file not created.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Meteostat hourly data for a location and date range.")
    parser.add_argument('--lat', type=float, default=-35.2809, help='Latitude (default: Canberra)')
    parser.add_argument('--lon', type=float, default=149.1300, help='Longitude (default: Canberra)')
    parser.add_argument('--years', type=int, default=5, help='Number of years of data to fetch (default: 5)')
    args = parser.parse_args()

    lat, lon = args.lat, args.lon
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * args.years)
    output_csv = f"data/meteostat_{lat}_{lon}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    os.makedirs("data", exist_ok=True)
    fetch_meteostat_hourly(lat, lon, start_date, end_date, output_csv)
