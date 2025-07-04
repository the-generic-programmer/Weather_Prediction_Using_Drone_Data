import cdsapi
import xarray as xr
import pandas as pd
import os

def fetch_era5_hourly(year, month, day, area, variables, output_nc, output_csv):
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': variables,
            'year': str(year),
            'month': f"{month:02d}",
            'day': f"{day:02d}",
            'time': [f"{h:02d}:00" for h in range(24)],
            'area': area,  # North, West, South, East
        },
        output_nc
    )
    print(f"Downloaded {output_nc}")
    # Convert NetCDF to CSV
    ds = xr.open_dataset(output_nc)
    df = ds.to_dataframe().reset_index()
    df.to_csv(output_csv, index=False)
    print(f"Saved CSV to {output_csv}")

if __name__ == "__main__":
    # Example: London region, July 2, 2025
    year = 2025
    month = 7
    day = 2
    # [North, West, South, East] (lat/lon)
    area = [52, -1, 51, 0]  # London region
    variables = [
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
        '2m_temperature',
        'surface_pressure',
        'total_precipitation',
        'cloud_cover'
    ]
    output_nc = "data/era5_london_20250702.nc"
    output_csv = "data/era5_london_20250702.csv"
    os.makedirs("data", exist_ok=True)
    fetch_era5_hourly(year, month, day, area, variables, output_nc, output_csv)
