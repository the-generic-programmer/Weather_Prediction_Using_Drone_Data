#!/usr/bin/env python3

import warnings
import logging
import asyncio
import csv
import json
import socket
import datetime
from pathlib import Path
from threading import Thread, Lock
import traceback
import signal
import sys
from typing import Optional, Dict, Any
import sqlite3
from mavsdk import System

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('mavsdk_logger.log'),
        logging.StreamHandler()
    ]
)

DB_PATH = Path(__file__).parent / "weather_mavsdk_logger.db"

# Makes a directory called mavsdk_logs if it doesn't exist
LOG_DIR = Path("mavsdk_logs")
LOG_DIR.mkdir(exist_ok=True)
CSV_FILE = LOG_DIR / f"drone_log_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"

# TCP config
TCP_PORT = 9000
MAVSDK_STREAM_RATE = 20.0  # Hz
TCP_STREAM_RATE = 5.0  # Hz
shutdown_flag = False

def find_free_port(start_port: int, max_attempts: int = 5) -> Optional[int]:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("0.0.0.0", port))
                logging.info(f"Port {port} is available")
                return port
        except OSError as e:
            logging.warning(f"Port {port} is in use: {e}")
    logging.error(f"No available ports in range {start_port} to {start_port + max_attempts - 1}")
    return None

def prompt_for_port() -> Optional[int]:
    """Prompt user for a custom port if default range is unavailable."""
    print(f"No available ports in range {TCP_PORT} to {TCP_PORT + 4}.")
    print("Please enter a custom port number (1024-65535) or press Enter to exit:")
    try:
        user_input = input("Custom port: ").strip()
        if not user_input:
            return None
        port = int(user_input)
        if 1024 <= port <= 65535:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(("0.0.0.0", port))
                    logging.info(f"Custom port {port} is available")
                    return port
            except OSError:
                logging.error(f"Custom port {port} is also in use")
                return None
        else:
            logging.error("Port must be between 1024 and 65535")
            return None
    except ValueError:
        logging.error("Invalid port number entered")
        return None

class WeatherSQLiteLogger:
    def __init__(self):
        self.db_path = DB_PATH
        self.table_name = 'weather_logs'
        self.fields = [
            "timestamp", "latitude", "longitude", "altitude_from_sealevel", "relative_altitude",
            "voltage", "remaining_percent", "north_m_s", "east_m_s", "down_m_s",
            "temperature_degc", "roll_deg", "pitch_deg", "yaw_deg"
        ]
        self.conn = None
        self.cursor = None
        self._setup()

    def _setup(self):
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            columns = ', '.join([f'"{f}" TEXT' for f in self.fields])
            self.cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS "{self.table_name}" (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    {columns}
                )
            ''')
            self.conn.commit()
            logging.info(f"Table {self.table_name} created or exists (SQLite)")
        except Exception as e:
            logging.error(f"SQLite setup failed: {e}")
            self.conn = None
            self.cursor = None
            raise

    def log(self, row):
        if not self.cursor:
            return
        try:
            values = [str(row.get(field, '')) for field in self.fields]
            placeholders = ', '.join(['?'] * len(self.fields))
            columns = ', '.join([f'"{f}"' for f in self.fields])
            sql = f"INSERT INTO {self.table_name} ({columns}) VALUES ({placeholders})"
            self.cursor.execute(sql, values)
            self.conn.commit()
            logging.info("Data logged to weather_logs (SQLite)")
        except Exception as e:
            logging.error(f"SQLite insert failed: {e}")

    def close(self):
        try:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
            logging.info("WeatherSQLiteLogger connection closed")
        except:
            pass

def set_tcp_stream_rate(rate_hz: float):
    global TCP_STREAM_RATE
    if rate_hz > 0:
        TCP_STREAM_RATE = rate_hz
        logging.info(f"TCP stream rate set to {TCP_STREAM_RATE} Hz")
    else:
        logging.warning("TCP stream rate must be positive.")

default_shutdown_flag = False
shutdown_flag = default_shutdown_flag

def remove_none(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}

class TCPServer:
    def __init__(self, port: int = TCP_PORT):
        self.clients = []
        self.lock = Lock()
        self.running = False
        self.port = port
        self.server_socket: Optional[socket.socket] = None
        self.accept_thread: Optional[Thread] = None

    def start(self):
        try:
            # Find a free port
            self.port = find_free_port(self.port)
            if not self.port:
                self.port = prompt_for_port()
                if not self.port:
                    raise OSError(
                        f"No available ports. Please free port 9000 (e.g., 'sudo kill -9 12345' for PID 12345) "
                        "or configure a different port in config.json. Check processes with: sudo netstat -tulnp | grep 9000"
                    )
            
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(("0.0.0.0", self.port))
            self.server_socket.listen(5)
            self.running = True
            self.accept_thread = Thread(target=self._accept_clients, daemon=True)
            self.accept_thread.start()
            logging.info(f"TCP Server started on port {self.port}")
        except Exception as e:
            logging.error(f"Error starting TCP server: {e}")
            raise

    def _accept_clients(self):
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

    def send(self, data: Dict[str, Any]):
        if not self.clients:
            return
        try:
            msg = json.dumps(data) + '\n'
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
                    except:
                        pass
        except Exception as e:
            logging.error(f"Error sending data: {e}")

    def stop(self):
        logging.info("Stopping TCP server...")
        self.running = False
        with self.lock:
            for client in self.clients[:]:
                try:
                    client.close()
                except:
                    pass
            self.clients.clear()
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        if self.accept_thread and self.accept_thread.is_alive():
            self.accept_thread.join(timeout=2.0)
        logging.info("TCP server stopped")

class CSVLogger:
    FIXED_FIELDS = [
        "timestamp", "source", "sequence",
        "latitude", "longitude", "altitude_from_sealevel", "relative_altitude",
        "voltage", "remaining_percent",
        "angular_velocity_forward_rad_s", "angular_velocity_right_rad_s", "angular_velocity_down_rad_s",
        "linear_acceleration_forward_m_s2", "linear_acceleration_right_m_s2", "linear_acceleration_down_m_s2",
        "magnetic_field_forward_gauss", "magnetic_field_right_gauss", "magnetic_field_down_gauss", "temperature_degc",
        "roll_deg", "pitch_deg", "yaw_deg", "timestamp_us",
        "north_m_s", "east_m_s", "down_m_s",
        "system_time", "unix_time"
    ]
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.lock = Lock()
        self.rows = []
        self.seq = 0
        self.headers_written = False

    def log(self, source: str, data_dict: Dict[str, Any]):
        try:
            self.seq += 1
            row = {
                "timestamp": datetime.datetime.utcnow().isoformat(timespec='milliseconds') + 'Z',
                "source": source,
                "sequence": self.seq
            }
            for key, value in data_dict.items():
                if value is not None:
                    row[key] = value
            with self.lock:
                self.rows.append(row)
        except Exception as e:
            logging.error(f"Error logging data: {e}")

    def flush(self):
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
                logging.info(f"CSV data flushed to {self.file_path}")
        except Exception as e:
            logging.error(f"Error flushing CSV data: {e}")

def signal_handler(signum, frame):
    global shutdown_flag
    logging.info("Shutdown signal received...")
    shutdown_flag = True

def initialize_databases():
    try:
        weather_sql_logger = WeatherSQLiteLogger()
        weather_sql_logger.close()
    except Exception as e:
        logging.error(f"Failed to initialize databases: {e}")
        raise

async def run():
    global shutdown_flag
    print("Select logging options (comma separated, e.g. sql,csv,tcp):")
    print("Options: sql, csv, tcp")
    user_input = input("Log to (default: csv,tcp): ").strip().lower()
    if not user_input or user_input == "default":
        user_input = "csv,tcp"
    log_to_sql = "sql" in user_input.split(',')
    log_to_csv = "csv" in user_input.split(',')
    log_to_tcp = "tcp" in user_input.split(',')
    logging.info(f"Logging options: sql={log_to_sql}, csv={log_to_csv}, tcp={log_to_tcp}")

    drone = System()
    tcp_server = None
    logger = None
    weather_sql_logger = None
    latest_imu = {}
    latest_attitude = {}
    latest_topic_data = {
        'imu': {},
        'attitude': {},
        'position': {},
        'battery': {},
        'velocity': {},
        'system_time': {},
        'wind': {}
    }
    try:
        if log_to_tcp:
            logging.info("Initializing TCP server...")
            tcp_server = TCPServer(port=TCP_PORT)
            try:
                tcp_server.start()
            except OSError as e:
                logging.error(f"Failed to start TCP server: {e}")
                logging.error(
                    f"Port {tcp_server.port} is in use. To free it, run: sudo kill -9 12345\n"
                    "Check processes with: sudo netstat -tulnp | grep 9000\n"
                    "Alternatively, edit config.json to use a different port."
                )
                raise

        logging.info("Waiting for drone to connect...")
        connection_timeout = 30
        start_time = asyncio.get_event_loop().time()
        async for state in drone.core.connection_state():
            if state.is_connected:
                logging.info("Drone connected!")
                break
            if asyncio.get_event_loop().time() - start_time > connection_timeout:
                raise TimeoutError("Drone connection timeout")
            if shutdown_flag:
                return

        if log_to_csv:
            logger = CSVLogger(CSV_FILE)
            logging.info(f"CSV logger initialized for {CSV_FILE}")
        if log_to_sql:
            weather_sql_logger = WeatherSQLiteLogger()

        def log_all(source, data):
            row = {**data, 'source': source}
            if log_to_csv and logger:
                logger.log(source, data)
            if log_to_sql and weather_sql_logger:
                weather_sql_logger.log(row)

        async def stream_imu():
            try:
                async for imu in drone.telemetry.raw_imu():
                    if shutdown_flag:
                        break
                    data = {}
                    if hasattr(imu, 'angular_velocity_frd'):
                        ang_vel = imu.angular_velocity_frd
                        data.update({
                            "angular_velocity_forward_rad_s": getattr(ang_vel, 'forward_rad_s', None),
                            "angular_velocity_right_rad_s": getattr(ang_vel, 'right_rad_s', None),
                            "angular_velocity_down_rad_s": getattr(ang_vel, 'down_rad_s', None)
                        })
                    if hasattr(imu, 'linear_acceleration_frd'):
                        lin_acc = imu.linear_acceleration_frd
                        data.update({
                            "linear_acceleration_forward_m_s2": getattr(lin_acc, 'forward_m_s2', None),
                            "linear_acceleration_right_m_s2": getattr(lin_acc, 'right_m_s2', None),
                            "linear_acceleration_down_m_s2": getattr(lin_acc, 'down_m_s2', None)
                        })
                    if hasattr(imu, 'magnetic_field_frd'):
                        mag_field = imu.magnetic_field_frd
                        data.update({
                            "magnetic_field_forward_gauss": getattr(mag_field, 'forward_gauss', None),
                            "magnetic_field_right_gauss": getattr(mag_field, 'right_gauss', None),
                            "magnetic_field_down_gauss": getattr(mag_field, 'down_gauss', None)
                        })
                    if hasattr(imu, 'temperature_degc'):
                        data["temperature_degc"] = getattr(imu, 'temperature_degc', None)
                    latest_imu.clear()
                    latest_imu.update(data)
                    log_all("scaled_imu", data)
                    latest_topic_data['imu'] = data.copy()
                    await asyncio.sleep(1 / MAVSDK_STREAM_RATE)
            except Exception as e:
                logging.error(f"Error in stream_imu: {e}")

        async def stream_attitude():
            try:
                async for att in drone.telemetry.attitude_euler():
                    if shutdown_flag:
                        break
                    data = {
                        "roll_deg": getattr(att, 'roll_deg', None),
                        "pitch_deg": getattr(att, 'pitch_deg', None),
                        "yaw_deg": getattr(att, 'yaw_deg', None),
                        "timestamp_us": getattr(att, 'timestamp_us', None)
                    }
                    latest_attitude.clear()
                    latest_attitude.update(data)
                    log_all("attitude", data)
                    latest_topic_data['attitude'] = data.copy()
                    await asyncio.sleep(1 / MAVSDK_STREAM_RATE)
            except Exception as e:
                logging.error(f"Error in stream_attitude: {e}")

        async def stream_position():
            try:
                async for pos in drone.telemetry.position():
                    if shutdown_flag:
                        break
                    data = {
                        "latitude": getattr(pos, 'latitude_deg', None),
                        "longitude": getattr(pos, 'longitude_deg', None),
                        "altitude_from_sealevel": getattr(pos, 'absolute_altitude_m', None),
                        "relative_altitude": getattr(pos, 'relative_altitude_m', None)
                    }
                    data.update(latest_imu)
                    data.update(latest_attitude)
                    data = {k: v for k, v in data.items() if v is not None}
                    log_all("global_position_int", data)
                    if log_to_sql and weather_sql_logger:
                        try:
                            minimal_row = {
                                "timestamp": datetime.datetime.utcnow().isoformat(),
                                "latitude": data.get("latitude"),
                                "longitude": data.get("longitude"),
                                "altitude_from_sealevel": data.get("altitude_from_sealevel"),
                                "relative_altitude": data.get("relative_altitude"),
                                "voltage": latest_topic_data['battery'].get("voltage"),
                                "remaining_percent": latest_topic_data['battery'].get("remaining_percent"),
                                "north_m_s": latest_topic_data['velocity'].get("north_m_s"),
                                "east_m_s": latest_topic_data['velocity'].get("east_m_s"),
                                "down_m_s": latest_topic_data['velocity'].get("down_m_s"),
                                "temperature_degc": data.get("temperature_degc"),
                                "roll_deg": data.get("roll_deg"),
                                "pitch_deg": data.get("pitch_deg"),
                                "yaw_deg": data.get("yaw_deg")
                            }
                            weather_sql_logger.log(minimal_row)
                        except Exception as e:
                            logging.warning(f"SQLite weather log skipped: {e}")
                    latest_topic_data['position'] = data.copy()
                    await asyncio.sleep(1 / MAVSDK_STREAM_RATE)
            except Exception as e:
                logging.error(f"Error in stream_position: {e}")

        async def stream_battery():
            try:
                async for bat in drone.telemetry.battery():
                    if shutdown_flag:
                        break
                    data = {
                        "voltage": getattr(bat, 'voltage_v', None),
                        "remaining_percent": getattr(bat, 'remaining_percent', None)
                    }
                    data.update(latest_imu)
                    data.update(latest_attitude)
                    data = {k: v for k, v in data.items() if v is not None}
                    log_all("battery_status", data)
                    latest_topic_data['battery'] = data.copy()
                    await asyncio.sleep(1 / MAVSDK_STREAM_RATE)
            except Exception as e:
                logging.error(f"Error in stream_battery: {e}")

        async def stream_wind():
            try:
                async for wind in drone.telemetry.wind():
                    if shutdown_flag:
                        break
                    data = {
                        "speed_m_s": getattr(wind, 'speed_m_s', None),
                        "direction_deg": getattr(wind, 'direction_deg', None)
                    }
                    data.update(latest_imu)
                    data.update(latest_attitude)
                    data = {k: v for k, v in data.items() if v is not None}
                    log_all("wind", data)
                    latest_topic_data['wind'] = data.copy()
                    await asyncio.sleep(1 / MAVSDK_STREAM_RATE)
            except Exception as e:
                logging.error(f"Error in stream_wind: {e}")

        async def stream_velocity():
            try:
                async for vel in drone.telemetry.velocity_ned():
                    if shutdown_flag:
                        break
                    data = {
                        "north_m_s": getattr(vel, 'north_m_s', None),
                        "east_m_s": getattr(vel, 'east_m_s', None),
                        "down_m_s": getattr(vel, 'down_m_s', None)
                    }
                    data.update(latest_imu)
                    data.update(latest_attitude)
                    data = {k: v for k, v in data.items() if v is not None}
                    log_all("velocity", data)
                    latest_topic_data['velocity'] = data.copy()
                    await asyncio.sleep(1 / MAVSDK_STREAM_RATE)
            except Exception as e:
                logging.error(f"Error in stream_velocity: {e}")

        async def stream_system_time():
            try:
                while not shutdown_flag:
                    data = {
                        "system_time": datetime.datetime.utcnow().isoformat(timespec='milliseconds') + 'Z',
                        "unix_time": datetime.datetime.utcnow().timestamp()
                    }
                    data.update(latest_imu)
                    data.update(latest_attitude)
                    data = {k: v for k, v in data.items() if v is not None}
                    log_all("system_time", data)
                    latest_topic_data['system_time'] = data.copy()
                    await asyncio.sleep(1 / MAVSDK_STREAM_RATE)
            except Exception as e:
                logging.error(f"Error in stream_system_time: {e}")

        async def tcp_publisher():
            if not log_to_tcp or not tcp_server:
                return
            try:
                from copy import deepcopy
                while not shutdown_flag:
                    combined = {}
                    for topic_data in latest_topic_data.values():
                        combined.update(topic_data)
                    if combined:
                        row = {field: combined.get(field, '') for field in CSVLogger.FIXED_FIELDS}
                        tcp_server.send(row)
                    await asyncio.sleep(1 / TCP_STREAM_RATE)
            except Exception as e:
                logging.error(f"Error in tcp_publisher: {e}")

        async def periodic_flush():
            if not log_to_csv or not logger:
                return
            try:
                while not shutdown_flag:
                    logger.flush()
                    await asyncio.sleep(5)
            except Exception as e:
                logging.error(f"Error in periodic_flush: {e}")

        async def monitor_connection():
            try:
                async for state in drone.core.connection_state():
                    if not state.is_connected:
                        logging.info("Drone disconnected!")
                        break
                    if shutdown_flag:
                        break
            except Exception as e:
                logging.error(f"Error monitoring connection: {e}")

        logging.info("Logging started. Press Ctrl+C to stop.")
        tasks = [
            asyncio.create_task(stream_position()),
            asyncio.create_task(stream_battery()),
            asyncio.create_task(stream_imu()),
            asyncio.create_task(stream_velocity()),
            asyncio.create_task(stream_attitude()),
            asyncio.create_task(stream_system_time()),
            asyncio.create_task(periodic_flush()),
            asyncio.create_task(monitor_connection()),
            asyncio.create_task(stream_wind()),
        ]
        if log_to_tcp and tcp_server:
            tasks.append(asyncio.create_task(tcp_publisher()))
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            logging.info("Tasks cancelled")
    except Exception as e:
        logging.error(f"Error in main loop: {e}")
        logging.error(traceback.format_exc())
    finally:
        logging.info("Cleaning up...")
        shutdown_flag = True
        for task in asyncio.all_tasks():
            if not task.done():
                task.cancel()
        if logger:
            logger.flush()
            logging.info(f"Data saved to {CSV_FILE}")
        if weather_sql_logger:
            weather_sql_logger.close()
        if tcp_server:
            tcp_server.stop()

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
        logging.error(traceback.format_exc())
    finally:
        logging.info("Logger shutdown complete.")
