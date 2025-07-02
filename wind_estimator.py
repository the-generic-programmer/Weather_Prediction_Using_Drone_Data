import numpy as np
import math
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def estimate_wind_from_drift(north_m_s, east_m_s, heading_deg=None):
    try:
        if north_m_s is None or east_m_s is None:
            raise ValueError("north_m_s and east_m_s must not be None")
        ground_speed = math.sqrt(north_m_s**2 + east_m_s**2)
        track_rad = math.atan2(east_m_s, north_m_s)
        track_deg = (math.degrees(track_rad) + 360) % 360
        drift_angle = None
        if heading_deg is not None:
            drift_angle = (track_deg - heading_deg + 360) % 360
            if drift_angle > 180:
                drift_angle -= 360
        wind_direction = (track_deg + 180) % 360
        wind_speed = ground_speed
        return {
            'wind_speed': round(wind_speed, 2),
            'wind_direction': round(wind_direction, 2),
            'ground_speed': round(ground_speed, 2),
            'track_angle': round(track_deg, 2),
            'drift_angle': round(drift_angle, 2) if drift_angle is not None else None
        }
    except Exception as e:
        logging.error(f"Wind estimation failed: {e}")
        return {"error": f"Wind estimation failed: {e}"}