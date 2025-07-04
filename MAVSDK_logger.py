#!/usr/bin/env python3

import logging
import asyncio
import csv
import json
import socket
import datetime
import random
from pathlib import Path
from threading import Thread, Lock
import signal
import sys
from typing import Optional, Dict, Any, List
import sqlite3
from mavsdk import System
import os

# Configure logging (INFO level for minimal terminal output)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('mavsdk_logger.log'),
        logging.StreamHandler()
    ]
)

# Ensure database directory exists
DB_DIR = Path(__file__).parent / "database"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "weather_mavsdk_logger.db"
logging.debug(f"DB_PATH: {DB_PATH}")
LOG_DIR = Path("mavsdk_logs")
LOG_DIR.mkdir(exist_ok=True)
CSV_FILE = LOG_DIR / f"drone_log_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"

# TCP config
TCP_PORT = 9000
PORT_RANGE = 50  # Try ports 9000–9049
MAVSDK_STREAM_RATE = 20.0  # Hz
TCP_STREAM_RATE = 5.0  # Hz
DRONE_CONNECTION_TIMEOUT = 5.0  # Reduced from 15.0
DRONE_CONNECTION_RETRIES = 3  # Reduced from 5
shutdown_flag = False
USE_SIMULATED_DATA = False

def find_free_port(start_port: int, max_attempts: int = PORT_RANGE) -> Optional[int]:
    """Find an available port with retries."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("0.0.0.0", port))
                logging.info(f"Using TCP port {port} for predict.py connection")
                return port
        except OSError as e:
            logging.debug(f"Port {port} is in use: {e}")
    logging.error(f"No available ports in range {start_port} to {start_port + max_attempts - 1}")
    return None

def prompt_for_port() -> Optional[int]:
    """Prompt user for a custom port."""
    print(f"No available ports in range {TCP_PORT} to {TCP_PORT + PORT_RANGE - 1}.")
    print("Enter a custom port (1024–65535) or press Enter to exit:")
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
                    logging.info(f"Using custom TCP port {port} for predict.py connection")
                    return port
            except OSError:
                logging.error(f"Custom port {port} is in use")
                return None
        else:
            logging.error("Port must be between 1024 and 65535")
            return None
    except ValueError:
        logging.error("Invalid port number")
        return None

class WeatherSQLiteLogger:
    def __init__(self):
        self.db_path = DB_PATH
        self.table_name = 'weather_logs'
        self.fields = [
            "timestamp", "latitude", "longitude", "altitude_from_sealevel", "relative_altitude",
            "voltage", "remaining_percent", "north_m_s", "east_m_s", "down_m_s",
            "temperature_degc", "roll_deg", "pitch_deg", "yaw_deg",
            "angular_velocity_forward_rad_s", "angular_velocity_right_rad_s", "angular_velocity_down_rad_s",
            "linear_acceleration_forward_m_s2", "linear_acceleration_right_m_s2", "linear_acceleration_down_m_s2",
            "magnetic_field_forward_gauss", "magnetic_field_right_gauss", "magnetic_field_down_gauss",
            "system_time", "unix_time", "speed_m_s", "direction_deg"
        ]
        self.conn = None
        self.cursor = None
        self.lock = Lock()  # Added thread safety
        self._setup()

    def _setup(self):
        try:
            # Ensure directory is writable
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

    def log(self, row):
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

    def close(self):
        try:
            with self.lock:
                if self.cursor:
                    self.cursor.close()
                if self.conn:
                    self.conn.close()
                logging.info("SQLite connection closed")
        except Exception:
            pass

class TCPServer:
    def __init__(self, port: int = TCP_PORT):
        self.clients: List[socket.socket] = []
        self.lock = Lock()
        self.running = False
        self.port = port
        self.server_socket: Optional[socket.socket] = None
        self.accept_thread: Optional[Thread] = None

    def start(self):
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
            msg = json.dumps(data, default=str) + '\n'  # Added default=str for datetime serialization
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
            logging.debug(f"Sent TCP data: {data}")
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
        "voltage", "remaining_percent", "north_m_s", "east_m_s", "down_m_s",
        "temperature_degc", "roll_deg", "pitch_deg", "yaw_deg",
        "angular_velocity_forward_rad_s", "angular_velocity_right_rad_s", "angular_velocity_down_rad_s",
        "linear_acceleration_forward_m_s2", "linear_acceleration_right_m_s2", "linear_acceleration_down_m_s2",
        "magnetic_field_forward_gauss", "magnetic_field_right_gauss", "magnetic_field_down_gauss",
        "system_time", "unix_time", "speed_m_s", "direction_deg"
    ]
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.lock = Lock()
        self.rows: List[Dict[str, Any]] = []
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
            logging.debug(f"Logged row to CSV buffer: {row}")
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
            logging.info(f"Flushed {len(rows_to_write)} rows to {self.file_path}")
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
        logging.info("Databases initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize databases: {e}")
        raise

async def connect_drone_with_retries(drone: System, retries: int = DRONE_CONNECTION_RETRIES, timeout: float = DRONE_CONNECTION_TIMEOUT):
    for attempt in range(1, retries + 1):
        try:
            logging.debug(f"Drone connection attempt {attempt}/{retries}")
            await asyncio.wait_for(drone.connect(system_address="udp://:14540"), timeout=timeout)
            logging.info("Drone connected successfully")
            
            # Wait for core to be ready
            async for state in drone.core.connection_state():
                if state.is_connected:
                    logging.info("Drone core is ready")
                    return True
                break
            
        except asyncio.TimeoutError:
            logging.warning(f"Drone connection attempt {attempt}/{retries} timed out after {timeout} seconds")
        except Exception as e:
            logging.error(f"Drone connection attempt {attempt}/{retries} failed: {e}")
        
        if attempt < retries:
            logging.info(f"Retrying in 1 second...")  # Reduced from 5 seconds
            await asyncio.sleep(1)
    
    logging.error("Failed to connect to drone after all retries")
    return False

class SimulatedDataGenerator:
    def __init__(self):
        self.base_lat = 13.0
        self.base_lon = 77.625
        self.base_alt = 300.0
        self.time_offset = 0
        
    async def generate_data(self):
        """Generate more realistic simulated data"""
        while not shutdown_flag:
            self.time_offset += 1/MAVSDK_STREAM_RATE
            
            # Add some realistic variations
            lat_variation = 0.001 * random.uniform(-1, 1)
            lon_variation = 0.001 * random.uniform(-1, 1)
            alt_variation = 10 * random.uniform(-1, 1)
            
            data = {
                "timestamp": datetime.datetime.utcnow().isoformat(timespec='milliseconds') + 'Z',
                "latitude": self.base_lat + lat_variation,
                "longitude": self.base_lon + lon_variation,
                "altitude_from_sealevel": self.base_alt + alt_variation,
                "relative_altitude": 50 + alt_variation,
                "north_m_s": 2 * random.uniform(-1, 1),
                "east_m_s": 2 * random.uniform(-1, 1),
                "down_m_s": 0.5 * random.uniform(-1, 1),
                "temperature_degc": 25 + 5 * random.uniform(-1, 1),
                "roll_deg": 5 * random.uniform(-1, 1),
                "pitch_deg": 5 * random.uniform(-1, 1),
                "yaw_deg": (self.time_offset * 10) % 360,  # Slow rotation
                "voltage": 11.5 + 0.5 * random.uniform(-1, 1),
                "remaining_percent": max(0, 100 - self.time_offset * 0.1),
                "angular_velocity_forward_rad_s": 0.1 * random.uniform(-1, 1),
                "angular_velocity_right_rad_s": 0.1 * random.uniform(-1, 1),
                "angular_velocity_down_rad_s": 0.1 * random.uniform(-1, 1),
                "linear_acceleration_forward_m_s2": 0.5 * random.uniform(-1, 1),
                "linear_acceleration_right_m_s2": 0.5 * random.uniform(-1, 1),
                "linear_acceleration_down_m_s2": 9.8 + 0.5 * random.uniform(-1, 1),
                "magnetic_field_forward_gauss": 0.05 * random.uniform(-1, 1),
                "magnetic_field_right_gauss": 0.05 * random.uniform(-1, 1),
                "magnetic_field_down_gauss": 0.05 * random.uniform(-1, 1),
                "speed_m_s": 3 + 2 * random.uniform(-1, 1),
                "direction_deg": (self.time_offset * 5) % 360,
                "system_time": datetime.datetime.utcnow().isoformat(timespec='milliseconds') + 'Z',
                "unix_time": datetime.datetime.utcnow().timestamp()
            }
            yield data
            await asyncio.sleep(1 / MAVSDK_STREAM_RATE)

async def run():
    global shutdown_flag, USE_SIMULATED_DATA
    
    print("Select logging options (comma separated, e.g. sql,csv,tcp):")
    print("Options: sql, csv, tcp")
    user_input = input("Log to (default: csv,tcp): ").strip().lower()
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

    # Attempt drone connection
    drone = System()
    if await connect_drone_with_retries(drone):
        USE_SIMULATED_DATA = False
        logging.info("Using real drone data")
    else:
        USE_SIMULATED_DATA = True
        logging.info("Using simulated data due to drone connection failure")

    # Shared data storage
    shared_data = {
        'latest_imu': {},
        'latest_attitude': {},
        'latest_position': {},
        'latest_battery': {},
        'latest_velocity': {},
        'latest_wind': {}
    }
    data_lock = Lock()

    def log_all(source: str, data: Dict[str, Any]):
        """Centralized logging function"""
        try:
            row = {**data, 'source': source}
            
            if log_to_csv and logger:
                logger.log(source, row)
            
            if log_to_sql and weather_sql_logger:
                weather_sql_logger.log(row)
            
            if log_to_tcp and tcp_server:
                required_fields = ['latitude', 'longitude']
                if all(field in row and row[field] is not None for field in required_fields):
                    tcp_server.send(row)
                else:
                    logging.debug(f"Skipped TCP send, missing required fields: {row}")
                    
        except Exception as e:
            logging.error(f"Error in log_all: {e}")

    # Initialize simulated data generator
    sim_gen = SimulatedDataGenerator()

    async def stream_telemetry():
        """Combined telemetry streaming function"""
        try:
            if USE_SIMULATED_DATA:
                async for data in sim_gen.generate_data():
                    if shutdown_flag:
                        break
                    
                    # Update shared data
                    with data_lock:
                        shared_data['latest_imu'] = {
                            k: v for k, v in data.items() 
                            if k.startswith(('angular_velocity', 'linear_acceleration', 'magnetic_field', 'temperature'))
                        }
                        shared_data['latest_attitude'] = {
                            k: v for k, v in data.items() 
                            if k.endswith('_deg') and k.startswith(('roll', 'pitch', 'yaw'))
                        }
                        shared_data['latest_position'] = {
                            k: v for k, v in data.items() 
                            if k in ['latitude', 'longitude', 'altitude_from_sealevel', 'relative_altitude']
                        }
                        shared_data['latest_velocity'] = {
                            k: v for k, v in data.items() 
                            if k.endswith('_m_s')
                        }
                        shared_data['latest_battery'] = {
                            k: v for k, v in data.items() 
                            if k in ['voltage', 'remaining_percent']
                        }
                        shared_data['latest_wind'] = {
                            k: v for k, v in data.items() 
                            if k in ['speed_m_s', 'direction_deg']
                        }
                    
                    # Log complete data
                    log_all("simulated_telemetry", data)
                    
            else:
                # Real drone data streaming
                tasks = []
                
                async def stream_position():
                    async for pos in drone.telemetry.position():
                        if shutdown_flag:
                            break
                        data = {
                            "latitude": getattr(pos, 'latitude_deg', None),
                            "longitude": getattr(pos, 'longitude_deg', None),
                            "altitude_from_sealevel": getattr(pos, 'absolute_altitude_m', None),
                            "relative_altitude": getattr(pos, 'relative_altitude_m', None),
                        }
                        with data_lock:
                            shared_data['latest_position'] = data
                            combined_data = {}
                            for category in shared_data.values():
                                combined_data.update(category)
                        log_all("position", combined_data)
                
                async def stream_attitude():
                    async for att in drone.telemetry.attitude_euler():
                        if shutdown_flag:
                            break
                        data = {
                            "roll_deg": getattr(att, 'roll_deg', None),
                            "pitch_deg": getattr(att, 'pitch_deg', None),
                            "yaw_deg": getattr(att, 'yaw_deg', None),
                        }
                        with data_lock:
                            shared_data['latest_attitude'] = data
                
                async def stream_velocity():
                    async for vel in drone.telemetry.velocity_ned():
                        if shutdown_flag:
                            break
                        data = {
                            "north_m_s": getattr(vel, 'north_m_s', None),
                            "east_m_s": getattr(vel, 'east_m_s', None),
                            "down_m_s": getattr(vel, 'down_m_s', None),
                        }
                        with data_lock:
                            shared_data['latest_velocity'] = data
                
                async def stream_battery():
                    async for bat in drone.telemetry.battery():
                        if shutdown_flag:
                            break
                        data = {
                            "voltage": getattr(bat, 'voltage_v', None),
                            "remaining_percent": getattr(bat, 'remaining_percent', None),
                        }
                        with data_lock:
                            shared_data['latest_battery'] = data
                
                async def stream_imu():
                    async for imu in drone.telemetry.raw_imu():
                        if shutdown_flag:
                            break
                        data = {
                            "angular_velocity_forward_rad_s": getattr(imu.angular_velocity_frd, 'forward_rad_s', None),
                            "angular_velocity_right_rad_s": getattr(imu.angular_velocity_frd, 'right_rad_s', None),
                            "angular_velocity_down_rad_s": getattr(imu.angular_velocity_frd, 'down_rad_s', None),
                            "linear_acceleration_forward_m_s2": getattr(imu.linear_acceleration_frd, 'forward_m_s2', None),
                            "linear_acceleration_right_m_s2": getattr(imu.linear_acceleration_frd, 'right_m_s2', None),
                            "linear_acceleration_down_m_s2": getattr(imu.linear_acceleration_frd, 'down_m_s2', None),
                            "magnetic_field_forward_gauss": getattr(imu.magnetic_field_frd, 'forward_gauss', None),
                            "magnetic_field_right_gauss": getattr(imu.magnetic_field_frd, 'right_gauss', None),
                            "magnetic_field_down_gauss": getattr(imu.magnetic_field_frd, 'down_gauss', None),
                            "temperature_degc": getattr(imu, 'temperature_degc', None),
                        }
                        with data_lock:
                            shared_data['latest_imu'] = data
                
                tasks = [
                    asyncio.create_task(stream_position()),
                    asyncio.create_task(stream_attitude()),
                    asyncio.create_task(stream_velocity()),
                    asyncio.create_task(stream_battery()),
                    asyncio.create_task(stream_imu()),
                ]
                
                await asyncio.gather(*tasks, return_exceptions=True)
                
        except Exception as e:
            logging.error(f"Error in stream_telemetry: {e}")

    async def periodic_flush():
        """Periodic CSV flushing"""
        if not log_to_csv or not logger:
            return
        try:
            while not shutdown_flag:
                logger.flush()
                await asyncio.sleep(5)
        except Exception as e:
            logging.error(f"Error in periodic_flush: {e}")

    async def monitor_connection():
        """Monitor drone connection"""
        if USE_SIMULATED_DATA:
            return
        try:
            async for state in drone.core.connection_state():
                if not state.is_connected:
                    logging.warning("Drone disconnected!")
                    break
                if shutdown_flag:
                    break
        except Exception as e:
            logging.error(f"Error monitoring connection: {e}")

    logging.info("Logging started. Ready to run predict.py. Press Ctrl+C to stop.")
    
    # Create and run tasks
    tasks = [
        asyncio.create_task(stream_telemetry()),
        asyncio.create_task(periodic_flush()),
        asyncio.create_task(monitor_connection())
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
        
        # Cancel all tasks
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