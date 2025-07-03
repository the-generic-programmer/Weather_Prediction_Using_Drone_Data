import requests
import cfgrib
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def download_gfs_grib(run_time, forecast_hour, save_path):
    """Download GFS GRIB2 file for a given run time and forecast hour."""
    base_url = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"
    date_str = run_time[:8]
    hour_str = run_time[9:11]
    fhr = f"{forecast_hour:03d}"
    url = f"{base_url}/gfs.{date_str}/{hour_str}/atmos/gfs.t{hour_str}z.pgrb2.0p25.f{fhr}"
    print(f"Downloading {url}")
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Saved to {save_path}")
    else:
        raise Exception(f"Failed to download: {url}")

def extract_wind_from_grib(grib_path, lat, lon):
    ds = cfgrib.open_dataset(grib_path, filter_by_keys={'typeOfLevel': 'isobaricInhPa'})
    u = ds['u'].sel(longitude=lon, latitude=lat, method='nearest')
    v = ds['v'].sel(longitude=lon, latitude=lat, method='nearest')
    wind_speed = np.sqrt(u**2 + v**2)
    wind_dir = (np.arctan2(u, v) * 180 / np.pi) % 360
    df = pd.DataFrame({
        'pressure_hPa': u['isobaricInhPa'].values,
        'wind_speed': wind_speed.values,
        'wind_direction': wind_dir.values
    })
    return df

def fetch_gfs_wind_timeseries(lat, lon, start_time, hours=24, run_hour='00'):
    """Fetch wind timeseries for a location from GFS for a given start time and duration."""
    os.makedirs('data', exist_ok=True)
    dfs = []
    run_time = start_time.strftime('%Y%m%d_') + run_hour
    for fh in range(0, hours+1, 3):
        grib_file = f"data/gfs_{run_time}_f{fh:03d}.grib2"
        try:
            download_gfs_grib(run_time, fh, grib_file)
            df = extract_wind_from_grib(grib_file, lat, lon)
            df['forecast_hour'] = fh
            df['timestamp'] = start_time + timedelta(hours=fh)
            dfs.append(df)
        except Exception as e:
            print(f"Failed for hour {fh}: {e}")
    if dfs:
        result = pd.concat(dfs, ignore_index=True)
        result.to_csv(f"data/gfs_wind_{lat}_{lon}_{start_time.strftime('%Y%m%d')}.csv", index=False)
        print(f"Saved timeseries to data/gfs_wind_{lat}_{lon}_{start_time.strftime('%Y%m%d')}.csv")
        return result
    else:
        print("No data fetched.")
        return None

if __name__ == "__main__":
    # Example: London, 2025-07-02 00Z
    lat, lon = 51.5074, -0.1278
    start_time = datetime(2025, 7, 2, 0)
    fetch_gfs_wind_timeseries(lat, lon, start_time, hours=24, run_hour='00')
