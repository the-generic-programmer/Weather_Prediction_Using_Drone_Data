#!/usr/bin/env python3
"""
MAVSDK Logger (updated)
-----------------------
Adds support for filling the following CSV/database columns when data are available from the autopilot:
    temperature_degc, pressure_hpa,
    angular_velocity_forward_rad_s, angular_velocity_right_rad_s, angular_velocity_down_rad_s,
    linear_acceleration_forward_m_s2, linear_acceleration_right_m_s2, linear_acceleration_down_m_s2,
    magnetic_field_forward_gauss, magnetic_field_right_gauss, magnetic_field_down_gauss,
    airspeed_m_s.

Key changes vs. your previous version:
1. **Use `telemetry.imu()`** rather than `raw_imu()` because the high‑level IMU struct exposes angular velocity, acceleration (FRD), magnetic field, and temperature (degC). Pressure is *not* in IMU; see #2.  
2. **Add `telemetry.scaled_pressure()` subscription** and map `absolute_pressure_hpa` → `pressure_hpa`. If differential pressure is available you *could* compute airspeed; we store both for completeness (not all vehicles publish).  
3. **Use `telemetry.fixedwing_metrics()`** to obtain `airspeed_m_s` (works even on some non‑fixed‑wing vehicles; if not published you fall back to derived value from differential pressure).  
4. Removed the broken direct MAVLink stream code path that caused: `Error in MAVLink message processing: 'System' object has no attribute 'mavlink'`. MAVSDK‑Python doesn’t expose the C++ MavlinkPassthrough plugin; attempting to access raw MAVLink from the `System` object raises errors.  
5. Centralized *data fusion* so that whenever position arrives we emit a combined row that includes the latest values from all other topics (IMU, pressure, battery, velocity, airspeed). This maximizes field fill‑rate without excessive row spam.  
6. Lightweight *rate limiting* for CSV/SQL/TCP output using a min period (`LOG_PUSH_PERIOD_S`) so you don’t log thousands of rows/second when any high‑rate topic updates. Position updates trigger pushes but are coalesced.

Implementation notes:
- Some autopilots publish NaN for unavailable fields; we normalize NaN to empty string (CSV) or None (dict) so DB inserts are clean.
- Magnetic field units in MAVSDK IMU are **Gauss**; we pass through directly.
- `pressure_hpa` recorded from the latest ScaledPressure message; if multiple baros present you get whichever autopilot marks primary.
- If both `fixedwing_metrics().airspeed_m_s` and `scaled_pressure().differential_pressure_hpa` are present we trust the former; the latter can be converted to airspeed = sqrt(2 * dp_pa / rho) but requires air density (rho) from baro temp/pressure; stub included.

See in‑code TODOs where you may extend the derived airspeed calculation.
"""

import logging
import asyncio
import csv
import json
import socket
import datetime
from pathlib import Path
from threading import Thread, Lock
import signal
import sys
from typing import Optional, Dict, Any, List
import sqlite3
from mavsdk import System
import os
import math
from datetime import UTC
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "mavsdk_logger.log")

# Configure logging (INFO level for minimal terminal output)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

# Ensure database directory exists
DB_DIR = Path(__file__).parent / "database"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "weather_mavsdk_logger.db"
logging.debug(f"DB_PATH: {DB_PATH}")
LOG_DIR = Path("mavsdk_logs")
LOG_DIR.mkdir(exist_ok=True)
CSV_FILE = LOG_DIR / f"drone_log_{datetime.datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.csv"

# TCP config
TCP_PORT = 9000
PORT_RANGE = 50  # Try ports 9000–9049

# Logging cadence control
LOG_PUSH_PERIOD_S = 0.2  # minimum seconds between combined log rows (5 Hz max)
CSV_FLUSH_PERIOD_S = 5.0

shutdown_flag = False

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _clean(value):
    """Normalize NaN/inf to None; pass through normal scalars."""
    try:
        if value is None:
            return None
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    except Exception:
        return None


def find_free_port(start_port: int, max_attempts: int = PORT_RANGE) -> Optional[int]:
    """Find an available port with retries."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("0.0.0.0", port))
                logging.info(f"Using TCP port {port} for predict.py connection")
                return port
        except OSError:
            continue
    logging.error(f"No available ports in range {start_port} to {start_port + max_attempts - 1}")
    return None


def prompt_for_port() -> Optional[int]:
    """Prompt user for a custom port."""
    logging.error("No available ports. Please free a port or configure a different one.")
    return None

# -----------------------------------------------------------------------------
# SQLite logger
# -----------------------------------------------------------------------------
class WeatherSQLiteLogger:
    def __init__(self):
        self.db_path = DB_PATH
        self.table_name = 'weather_logs'
        self.fields = [
            "timestamp", "latitude", "longitude", "altitude_from_sealevel", "relative_altitude",
            "voltage", "remaining_percent", "north_m_s", "east_m_s", "down_m_s",
            "temperature_degc", "pressure_hpa", "roll_deg", "pitch_deg", "yaw_deg",
            "angular_velocity_forward_rad_s", "angular_velocity_right_rad_s", "angular_velocity_down_rad_s",
            "linear_acceleration_forward_m_s2", "linear_acceleration_right_m_s2", "linear_acceleration_down_m_s2",
            "magnetic_field_forward_gauss", "magnetic_field_right_gauss", "magnetic_field_down_gauss",
            "system_time", "unix_time", "speed_m_s", "direction_deg", "airspeed_m_s"
        ]
        self.conn = None
        self.cursor = None
        self.lock = Lock()  # thread safety
        self._setup()

    def _setup(self) -> None:
        try:
            if not self.db_path.parent.exists():
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
            os.chmod(self.db_path.parent, 0o755)
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.cursor = self.conn.cursor()
            columns = ', '.join([f'"{f}" TEXT' for f in self.fields])
            self.cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS "{self.table_name}" (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    {columns}
                )
            ''')
            self.conn.commit()
            logging.info("SQLite table created or exists")
        except Exception as e:
            logging.error(f"SQLite setup failed: {e}")
            raise

    def log(self, row: Dict[str, Any]) -> None:
        if not self.cursor:
            return
        try:
            with self.lock:
                values = [str(row.get(field, '')) for field in self.fields]
                placeholders = ', '.join(['?'] * len(self.fields))
                columns = ', '.join([f'"{f}"' for f in self.fields])
                sql = f"INSERT INTO {self.table_name} ({columns}) VALUES ({placeholders})"
                self.cursor.execute(sql, values)
                self.conn.commit()
                logging.debug(f"Data logged to SQLite: {row}")
        except Exception as e:
            logging.error(f"SQLite insert failed: {e}")

    def close(self) -> None:
        try:
            with self.lock:
                if self.cursor:
                    self.cursor.close()
                if self.conn:
                    self.conn.close()
                logging.info("SQLite connection closed")
        except Exception:
            pass

# -----------------------------------------------------------------------------
# TCP broadcast server
# -----------------------------------------------------------------------------
class TCPServer:
    def __init__(self, port: int = TCP_PORT):
        self.clients: List[socket.socket] = []
        self.lock = Lock()
        self.running = False
        self.port = port
        self.server_socket: Optional[socket.socket] = None
        self.accept_thread: Optional[Thread] = None

    def start(self) -> None:
        try:
            found_port = find_free_port(self.port)
            if not found_port:
                found_port = prompt_for_port()
                if not found_port:
                    raise OSError("No available ports. Please free port or configure a different one.")
            self.port = found_port
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(("0.0.0.0", self.port))
            self.server_socket.listen(5)
            self.running = True
            self.accept_thread = Thread(target=self._accept_clients, daemon=True)
            self.accept_thread.start()
            logging.info(f"TCP server started on port {self.port}")
        except Exception as e:
            logging.error(f"Error starting TCP server: {e}")
            raise

    def _accept_clients(self) -> None:
        while self.running:
            try:
                if self.server_socket:
                    self.server_socket.settimeout(1.0)
                    client_sock, addr = self.server_socket.accept()
                    logging.info(f"Client connected from {addr}")
                    with self.lock:
                        self.clients.append(client_sock)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logging.error(f"Error accepting client: {e}")
                break

    def send(self, data: Dict[str, Any]) -> None:
        if not self.clients:
            return
        try:
            msg = json.dumps(data, default=str) + '\n'
            with self.lock:
                disconnected_clients = []
                for client in self.clients:
                    try:
                        client.sendall(msg.encode())
                    except Exception:
                        disconnected_clients.append(client)
                for client in disconnected_clients:
                    self.clients.remove(client)
                    try:
                        client.close()
                    except:  # noqa: E722
                        pass
            logging.debug(f"Sent TCP data: {data}")
        except Exception as e:
            logging.error(f"Error sending data: {e}")

    def stop(self) -> None:
        logging.info("Stopping TCP server...")
        self.running = False
        with self.lock:
            for client in self.clients[:]:
                try:
                    client.close()
                except:  # noqa: E722
                    pass
            self.clients.clear()
        if self.server_socket:
            try:
                self.server_socket.close()
            except:  # noqa: E722
                pass
        if self.accept_thread and self.accept_thread.is_alive():
            self.accept_thread.join(timeout=2.0)
        logging.info("TCP server stopped")

# -----------------------------------------------------------------------------
# CSV logger (buffered)
# -----------------------------------------------------------------------------
class CSVLogger:
    FIXED_FIELDS = [
        "timestamp", "source", "sequence",
        "latitude", "longitude", "altitude_from_sealevel", "relative_altitude",
        "voltage", "remaining_percent", "north_m_s", "east_m_s", "down_m_s",
        "temperature_degc", "pressure_hpa", "roll_deg", "pitch_deg", "yaw_deg",
        "angular_velocity_forward_rad_s", "angular_velocity_right_rad_s", "angular_velocity_down_rad_s",
        "linear_acceleration_forward_m_s2", "linear_acceleration_right_m_s2", "linear_acceleration_down_m_s2",
        "magnetic_field_forward_gauss", "magnetic_field_right_gauss", "magnetic_field_down_gauss",
        "system_time", "unix_time", "speed_m_s", "direction_deg", "airspeed_m_s"
    ]

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.lock = Lock()
        self.rows: List[Dict[str, Any]] = []
        self.seq = 0
        self.headers_written = False

    def log(self, source: str, data_dict: Dict[str, Any]) -> None:
        try:
            self.seq += 1
            row = {
                "timestamp": datetime.datetime.now(datetime.UTC).isoformat(timespec='milliseconds') + 'Z',
                "source": source,
                "sequence": self.seq
            }
            for key, value in data_dict.items():
                if value is not None:
                    row[key] = value
            with self.lock:
                self.rows.append(row)
            logging.debug(f"Logged row to CSV buffer: {row}")
        except Exception as e:
            logging.error(f"Error logging data: {e}")

    def flush(self) -> None:
        try:
            with self.lock:
                if not self.rows:
                    return
                rows_to_write = self.rows[:]
                self.rows.clear()

            file_exists = self.file_path.exists() and self.headers_written
            with open(self.file_path, "a", newline="", encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.FIXED_FIELDS, extrasaction='ignore')
                if not file_exists:
                    writer.writeheader()
                    self.headers_written = True
                for row in rows_to_write:
                    complete_row = {field: row.get(field, '') for field in self.FIXED_FIELDS}
                    writer.writerow(complete_row)
            logging.info(f"Flushed {len(rows_to_write)} rows to {self.file_path}")
        except Exception as e:
            logging.error(f"Error flushing CSV data: {e}")

# -----------------------------------------------------------------------------
# Signal handling
# -----------------------------------------------------------------------------

def signal_handler(signum, frame):  # noqa: D401
    global shutdown_flag
    logging.info("Shutdown signal received...")
    shutdown_flag = True


def initialize_databases() -> None:
    try:
        weather_sql_logger = WeatherSQLiteLogger()
        weather_sql_logger.close()
        logging.info("Databases initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize databases: {e}")
        raise

# -----------------------------------------------------------------------------
# Drone connection
# -----------------------------------------------------------------------------
async def connect_drone(drone: System) -> bool:
    """Wait indefinitely for drone connection."""
    try:
        logging.info("Attempting to connect to drone...")
        await drone.connect(system_address="udp://:14550")
        logging.info("Drone connected successfully")

        # Wait for core to be ready
        async for state in drone.core.connection_state():
            if state.is_connected:
                logging.info("Drone core is ready")
                return True
            if shutdown_flag:
                break
    except Exception as e:
        logging.error(f"Error connecting to drone: {e}")
    return False

# -----------------------------------------------------------------------------
# Main run loop
# -----------------------------------------------------------------------------
async def run() -> None:
    global shutdown_flag

    import argparse
    parser = argparse.ArgumentParser(description="MAVSDK Logger")
    parser.add_argument('--log-to', type=str, default=None, help='Comma separated logging options: sql,csv,tcp')
    args, _ = parser.parse_known_args()

    user_input = None
    if args.log_to:
        user_input = args.log_to.strip().lower()
        logging.info(f"Logging options provided via --log-to: {user_input}")
    else:
        logging.info("Select logging options (comma separated, e.g. sql,csv,tcp):")
        logging.info("Options: sql, csv, tcp")
        try:
            user_input = input("Log to (default: csv,tcp): ").strip().lower()
        except EOFError:
            user_input = "csv,tcp"
            logging.info("No input detected, defaulting to: csv,tcp")
    if not user_input or user_input == "default":
        user_input = "csv,tcp"

    log_to_sql = "sql" in user_input.split(',')
    log_to_csv = "csv" in user_input.split(',')
    log_to_tcp = "tcp" in user_input.split(',')
    logging.info(f"Logging options: sql={log_to_sql}, csv={log_to_csv}, tcp={log_to_tcp}")

    # Initialize TCP server
    tcp_server = None
    if log_to_tcp:
        try:
            tcp_server = TCPServer()
            tcp_server.start()
        except Exception as e:
            logging.error(f"Failed to start TCP server: {e}")
            return

    # Initialize loggers
    logger = None
    weather_sql_logger = None
    if log_to_csv:
        logger = CSVLogger(CSV_FILE)
    if log_to_sql:
        weather_sql_logger = WeatherSQLiteLogger()

    # Wait for drone connection
    drone = System()
    if not await connect_drone(drone):
        logging.error("Failed to connect to drone. Script will wait until connection is established.")
        while not shutdown_flag:
            if await connect_drone(drone):
                break
            await asyncio.sleep(5)  # Check every 5 seconds
        if shutdown_flag:
            logging.info("Shutdown signal received while waiting for drone connection.")
            return

    logging.info("Using real drone data")

    # Shared latest datapoints (all sanitized via _clean())
    shared_data = {
        'position': {},
        'attitude': {},
        'velocity': {},
        'battery': {},
        'imu': {},
        'pressure': {},
        'airspeed': {},
    }
    data_lock = Lock()
    last_push_time = 0.0

    # ------------------------------------------------------------------
    # Centralized logging
    # ------------------------------------------------------------------
    def _compose_row() -> Dict[str, Any]:
        combined: Dict[str, Any] = {}
        for cat in shared_data.values():
            combined.update(cat)
        return combined

    def log_all(source: str) -> None:
        nonlocal last_push_time
        now = asyncio.get_event_loop().time()
        if now - last_push_time < LOG_PUSH_PERIOD_S:
            return  # rate limit
        last_push_time = now

        row = _compose_row()
        row["system_time"] = datetime.datetime.now().isoformat(timespec='milliseconds') + 'Z'
        row["unix_time"] = str(int(datetime.datetime.now(UTC).timestamp()))

        # Ensure mandatory numeric fields are strings safe for DB/CSV
        # (CSVLogger handles blanks; SQLiteLogger stores as text strings)
        if log_to_csv and logger:
            logger.log(source, row)
        if log_to_sql and weather_sql_logger:
            weather_sql_logger.log(row)
        if log_to_tcp and tcp_server:
            # Only stream if we have a valid position
            if row.get('latitude') is not None and row.get('longitude') is not None:
                tcp_server.send(row)

    # ------------------------------------------------------------------
    # Telemetry stream coroutines
    # ------------------------------------------------------------------
    async def stream_position():
        async for pos in drone.telemetry.position():
            if shutdown_flag:
                break
            data = {
                "latitude": _clean(getattr(pos, 'latitude_deg', None)),
                "longitude": _clean(getattr(pos, 'longitude_deg', None)),
                "altitude_from_sealevel": _clean(getattr(pos, 'absolute_altitude_m', None)),
                "relative_altitude": _clean(getattr(pos, 'relative_altitude_m', None)),
            }
            with data_lock:
                shared_data['position'] = data
            log_all("position")

    async def stream_attitude():
        async for att in drone.telemetry.attitude_euler():
            if shutdown_flag:
                break
            data = {
                "roll_deg": _clean(getattr(att, 'roll_deg', None)),
                "pitch_deg": _clean(getattr(att, 'pitch_deg', None)),
                "yaw_deg": _clean(getattr(att, 'yaw_deg', None)),
            }
            with data_lock:
                shared_data['attitude'] = data

    async def stream_velocity():
        async for vel in drone.telemetry.velocity_ned():
            if shutdown_flag:
                break
            north_ms = _clean(getattr(vel, 'north_m_s', None))
            east_ms = _clean(getattr(vel, 'east_m_s', None))
            down_ms = _clean(getattr(vel, 'down_m_s', None))
            speed_ms = None
            direction_deg = None
            if north_ms is not None and east_ms is not None:
                speed_ms = math.sqrt(north_ms**2 + east_ms**2)
                direction_deg = (math.degrees(math.atan2(east_ms, north_ms)) % 360)
            data = {
                "north_m_s": north_ms,
                "east_m_s": east_ms,
                "down_m_s": down_ms,
                "speed_m_s": _clean(speed_ms),
                "direction_deg": _clean(direction_deg)
            }
            with data_lock:
                shared_data['velocity'] = data

    async def stream_battery():
        async for bat in drone.telemetry.battery():
            if shutdown_flag:
                break
            data = {
                "voltage": _clean(getattr(bat, 'voltage_v', None)),
                "remaining_percent": _clean(getattr(bat, 'remaining_percent', None)),
            }
            with data_lock:
                shared_data['battery'] = data

    async def stream_imu():
        """High‑level IMU: accel/gyro/mag + temp."""
        async for imu in drone.telemetry.imu():
            if shutdown_flag:
                break
            # NOTE: field names in Python binding mirror C++ names (acceleration_frd, etc.)
            acc = getattr(imu, 'acceleration_frd', None)
            ang = getattr(imu, 'angular_velocity_frd', None)
            mag = getattr(imu, 'magnetic_field_frd', None)
            data = {
                "linear_acceleration_forward_m_s2": _clean(getattr(acc, 'forward_m_s2', None)) if acc else None,
                "linear_acceleration_right_m_s2": _clean(getattr(acc, 'right_m_s2', None)) if acc else None,
                "linear_acceleration_down_m_s2": _clean(getattr(acc, 'down_m_s2', None)) if acc else None,
                "angular_velocity_forward_rad_s": _clean(getattr(ang, 'forward_rad_s', None)) if ang else None,
                "angular_velocity_right_rad_s": _clean(getattr(ang, 'right_rad_s', None)) if ang else None,
                "angular_velocity_down_rad_s": _clean(getattr(ang, 'down_rad_s', None)) if ang else None,
                "magnetic_field_forward_gauss": _clean(getattr(mag, 'forward_gauss', None)) if mag else None,
                "magnetic_field_right_gauss": _clean(getattr(mag, 'right_gauss', None)) if mag else None,
                "magnetic_field_down_gauss": _clean(getattr(mag, 'down_gauss', None)) if mag else None,
                "temperature_degc": _clean(getattr(imu, 'temperature_degc', None)),
            }
            with data_lock:
                shared_data['imu'] = data

    async def stream_scaled_pressure():
        """Baro & differential pressure (for airspeed calc)."""
        async for sp in drone.telemetry.scaled_pressure():
            if shutdown_flag:
                break
            abs_hpa = _clean(getattr(sp, 'absolute_pressure_hpa', None))
            diff_hpa = _clean(getattr(sp, 'differential_pressure_hpa', None))
            temp_deg = _clean(getattr(sp, 'temperature_deg', None))
            data = {
                "pressure_hpa": abs_hpa,
            }
            # Use temp_deg if IMU temp missing (don't overwrite if present)
            if temp_deg is not None and shared_data['imu'].get('temperature_degc') is None:
                shared_data['imu']['temperature_degc'] = temp_deg

            # Optionally compute fallback airspeed from differential pressure if fixedwing_metrics not present
            if diff_hpa is not None:
                # placeholder compute; will be overridden by fixedwing_metrics if available
                rho = None  # if you have local air density, fill here
                if rho and diff_hpa is not None:
                    dp_pa = diff_hpa * 100.0
                    data['_airspeed_dp_fallback_m_s'] = math.sqrt(max(0.0, 2.0 * dp_pa / rho))
            with data_lock:
                shared_data['pressure'] = data

    async def stream_airspeed():
        """Get airspeed from fixedwing_metrics() if available."""
        try:
            async for fwm in drone.telemetry.fixedwing_metrics():
                if shutdown_flag:
                    break
                data = {
                    "airspeed_m_s": _clean(getattr(fwm, 'airspeed_m_s', None))
                }
                with data_lock:
                    shared_data['airspeed'] = data
        except Exception as e:  # If vehicle doesn't publish fixedwing metrics
            logging.warning(f"fixedwing_metrics stream unavailable: {e}")

    # ------------------------------------------------------------------
    # Periodic CSV flushing
    # ------------------------------------------------------------------
    async def periodic_flush() -> None:
        if not log_to_csv or not logger:
            return
        try:
            while not shutdown_flag:
                logger.flush()
                await asyncio.sleep(CSV_FLUSH_PERIOD_S)
        except Exception as e:
            logging.error(f"Error in periodic_flush: {e}")

    # ------------------------------------------------------------------
    # Monitor connection
    # ------------------------------------------------------------------
    async def monitor_connection() -> None:
        try:
            async for state in drone.core.connection_state():
                if not state.is_connected:
                    logging.warning("Drone disconnected!")
                    break
                if shutdown_flag:
                    break
        except Exception as e:
            logging.error(f"Error monitoring connection: {e}")

    logging.info("Logging started. Waiting for drone connection. Press Ctrl+C to stop.")

    # Create and run tasks
    tasks = [
        asyncio.create_task(stream_position()),
        asyncio.create_task(stream_attitude()),
        asyncio.create_task(stream_velocity()),
        asyncio.create_task(stream_battery()),
        asyncio.create_task(stream_imu()),
        asyncio.create_task(stream_scaled_pressure()),
        asyncio.create_task(stream_airspeed()),
        asyncio.create_task(periodic_flush()),
        asyncio.create_task(monitor_connection()),
    ]

    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        logging.info("Tasks cancelled")
    except Exception as e:
        logging.error(f"Error in main tasks: {e}")
    finally:
        logging.info("Cleaning up...")
        shutdown_flag = True
        for task in tasks:
            if not task.done():
                task.cancel()
        # Final flush and cleanup
        if logger:
            logger.flush()
            logging.info(f"Final data saved to {CSV_FILE}")
        if weather_sql_logger:
            weather_sql_logger.close()
        if tcp_server:
            tcp_server.stop()

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        logging.info("Starting MAVSDK logger...")
        initialize_databases()
        asyncio.run(run())
    except KeyboardInterrupt:
        logging.info("Logging stopped by user.")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)
