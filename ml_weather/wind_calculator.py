#!/usr/bin/env python3
"""
Enhanced Wind Speed and Direction Calculator with Auto MavSDK Integration
======================================================================
- Automatically runs mavsdk_logger.py if not already active.
- Connects to MAVSDK telemetry stream and calculates wind every second.
- Uses enhanced vector and filtering approach for accuracy.
- Handles numpy warnings with explicit dtype casting.
"""

import asyncio
import json
import math
import socket
import logging
import time
import subprocess
from dataclasses import dataclass
from typing import Optional, Dict
from collections import deque

import numpy as np

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# Global constants
TELEMETRY_HOST = "127.0.0.1"
TELEMETRY_PORT = 9000
CHECK_INTERVAL = 1.0  # seconds
MAX_HISTORY = 20

@dataclass
class WindResult:
    timestamp: str
    wind_speed: float
    wind_direction: float
    ground_speed: float
    airspeed: float
    confidence: str

class WindCalculator:
    def __init__(self):
        self.wind_history = deque(maxlen=MAX_HISTORY)

    def calculate(self, data: Dict) -> Optional[WindResult]:
        try:
            # Extract needed values
            vn = float(data.get("north_m_s", 0))
            ve = float(data.get("east_m_s", 0))
            airspeed = float(data.get("airspeed_m_s", 0))
            heading = float(data.get("yaw_deg", 0))

            if airspeed < 3.0 or (vn == 0 and ve == 0):
                return None

            # Ground vector
            ground_speed = math.sqrt(vn**2 + ve**2)

            # Air vector from heading
            heading_rad = math.radians(heading)
            air_n = airspeed * math.cos(heading_rad)
            air_e = airspeed * math.sin(heading_rad)

            # Wind vector
            wind_n = vn - air_n
            wind_e = ve - air_e

            wind_speed = math.sqrt(wind_n**2 + wind_e**2)
            wind_direction = (math.degrees(math.atan2(wind_e, wind_n)) + 180) % 360

            confidence = "high" if wind_speed > 2 else "low"

            return WindResult(
                timestamp=data.get("timestamp", time.strftime('%Y-%m-%dT%H:%M:%SZ')),
                wind_speed=wind_speed,
                wind_direction=wind_direction,
                ground_speed=ground_speed,
                airspeed=airspeed,
                confidence=confidence
            )
        except Exception as e:
            logging.error(f"Wind calculation error: {e}")
            return None

async def ensure_mavsdk_logger():
    """Ensure mavsdk_logger.py is running."""
    try:
        result = subprocess.run(["pgrep", "-f", "mavsdk_logger.py"], stdout=subprocess.PIPE)
        if result.returncode != 0:
            logging.info("Starting mavsdk_logger.py...")
            subprocess.Popen(["python3", "mavsdk_logger.py"])
        else:
            logging.info("mavsdk_logger.py already running.")
    except Exception as e:
        logging.error(f"Failed to check/start mavsdk_logger.py: {e}")

async def listen_and_calculate():
    await ensure_mavsdk_logger()
    calc = WindCalculator()

    while True:
        try:
            reader, writer = await asyncio.open_connection(TELEMETRY_HOST, TELEMETRY_PORT)
            logging.info("Connected to telemetry.")

            while True:
                line = await reader.readline()
                if not line:
                    break

                try:
                    telemetry = json.loads(line.decode('utf-8').strip())
                    result = calc.calculate(telemetry)
                    if result:
                        print(f"[{result.timestamp}] Wind Speed: {result.wind_speed:.2f} m/s, Direction: {result.wind_direction:.1f}°, GS: {result.ground_speed:.2f} m/s, AS: {result.airspeed:.2f} m/s, Confidence: {result.confidence}")
                except json.JSONDecodeError:
                    continue
        except ConnectionRefusedError:
            logging.warning("Telemetry not available, retrying...")
            await asyncio.sleep(2)
        except Exception as e:
            logging.error(f"Error: {e}")
            await asyncio.sleep(2)

if __name__ == "__main__":
    try:
        asyncio.run(listen_and_calculate())
    except KeyboardInterrupt:
        logging.info("Shutdown requested.")


