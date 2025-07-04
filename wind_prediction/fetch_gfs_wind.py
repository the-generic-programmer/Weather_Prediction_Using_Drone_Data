import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import requests
import xarray as xr
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from tqdm import tqdm

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
        total = int(r.headers.get('content-length', 0))
        with open(save_path, 'wb') as f, tqdm(
            desc=f"Downloading {os.path.basename(save_path)}",
            total=total, unit='B', unit_scale=True, unit_divisor=1024
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        print(f"Saved to {save_path}")
    else:
        raise Exception(f"Failed to download: {url} (status {r.status_code})")

def extract_wind_from_grib(grib_path, lat, lon, level=850):
    """Extract wind speed and direction from GRIB2 file at given lat/lon and pressure level."""
    try:
        ds = xr.open_dataset(
            grib_path,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {
                    "typeOfLevel": "isobaricInhPa",
                    "level": level
                }
            }
        )
    except Exception as e:
        raise RuntimeError(f"Failed to open GRIB file: {e}")

    # Defensive: check for u and v wind components
    u_var = None
    v_var = None
    for cand in ['u', 'u10', 'u_component_of_wind_isobaric']:  # try common names
        if cand in ds.variables:
            u_var = cand
            break
    for cand in ['v', 'v10', 'v_component_of_wind_isobaric']:
        if cand in ds.variables:
            v_var = cand
            break
    if not u_var or not v_var:
        raise ValueError(f"Wind components 'u' and 'v' not found in dataset. Available: {list(ds.variables)}")

    u = ds[u_var].sel(longitude=lon, latitude=lat, method='nearest')
    v = ds[v_var].sel(longitude=lon, latitude=lat, method='nearest')
    wind_speed = np.sqrt(u**2 + v**2)
    wind_dir = (np.arctan2(u, v) * 180 / np.pi) % 360
    df = pd.DataFrame({
        'wind_speed': wind_speed.values.flatten(),
        'wind_direction': wind_dir.values.flatten(),
        'pressure_level_hPa': level
    })
    return df

def fetch_gfs_wind_timeseries(lat, lon, start_time, hours=24, run_hour='00', level=850):
    """Fetch wind timeseries for a location from GFS for a given start time and duration."""
    os.makedirs('data', exist_ok=True)
    dfs = []
    run_time = start_time.strftime('%Y%m%d_') + run_hour
    for fh in range(0, hours + 1, 3):
        grib_file = f"data/gfs_{run_time}_f{fh:03d}.grib2"
        try:
            download_gfs_grib(run_time, fh, grib_file)
            df = extract_wind_from_grib(grib_file, lat, lon, level=level)
            df['forecast_hour'] = fh
            df['timestamp'] = start_time + timedelta(hours=fh)
            dfs.append(df)
        except Exception as e:
            print(f"Failed for hour {fh}: {e}")
    if dfs:
        result = pd.concat(dfs, ignore_index=True)
        # Remove outliers for LSTM training (optional, can be tuned)
        result = result[(result['wind_speed'] < 100) & (result['wind_direction'] <= 360)]
        csv_path = f"data/gfs_wind_{lat}_{lon}_{start_time.strftime('%Y%m%d')}_level{level}.csv"
        result.to_csv(csv_path, index=False)
        print(f"Saved timeseries to {csv_path}")
        return result
    else:
        print("No data fetched.")
        return None

if __name__ == "__main__":
    # Example: London, 2025-07-02 00Z at 850 hPa
    lat, lon = 51.5074, -0.1278
    start_time = datetime(2025, 7, 2, 0)
    fetch_gfs_wind_timeseries(lat, lon, start_time, hours=24, run_hour='00', level=850)
