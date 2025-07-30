import csv
import json
import math

CSV_FILE = 'xplane_data.csv'
MODEL_FILE = 'ml_weather/wind_latest.json'

def load_xplane_data():
    with open(CSV_FILE, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def load_model_data():
    with open(MODEL_FILE, 'r') as f:
        return json.load(f)

def angular_error(a, b):
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)

def main():
    xplane_data = load_xplane_data()
    model_data = load_model_data()

    model_time = model_data.get('timestamp')
    best_row = None
    min_diff = float('inf')
    for row in xplane_data:
        row_time = row['timestamp']
        diff = abs((row_time > model_time) - (row_time < model_time))
        if diff < min_diff:
            min_diff = diff
            best_row = row

    if not best_row:
        print("No matching X-Plane data found.")
        return

    xplane_wind_speed = float(best_row['wind_speed_kt'])
    xplane_wind_dir = float(best_row['wind_direction_degt'])
    model_wind_speed = float(model_data['wind_speed_knots'])
    model_wind_dir = float(model_data['wind_direction_degrees'])

    speed_error = abs(model_wind_speed - xplane_wind_speed)
    dir_error = angular_error(model_wind_dir, xplane_wind_dir)

    print("=== Weather/Wind Model Evaluation ===")
    print(f"X-Plane Wind Speed: {xplane_wind_speed:.2f} kt")
    print(f"Model Wind Speed:  {model_wind_speed:.2f} kt")
    print(f"Speed Error:       {speed_error:.2f} kt")
    print(f"X-Plane Wind Dir:  {xplane_wind_dir:.1f}°")
    print(f"Model Wind Dir:    {model_wind_dir:.1f}°")
    print(f"Direction Error:   {dir_error:.1f}°")
    print(f"Model Confidence:  {model_data.get('confidence', 'N/A')}%")
    print(f"Flight State:      {model_data.get('flight_state', 'N/A')}")
    print(f"Method Weights:    {model_data.get('method_weights', {})}")

if __name__ == "__main__":
    main()
