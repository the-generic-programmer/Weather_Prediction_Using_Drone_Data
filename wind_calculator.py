#!/usr/bin/env python3

import json
import socket
import logging
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import statistics
import math
from datetime import datetime

# Configure logging with more detailed output
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more info
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

@dataclass
class DroneState:
    timestamp: float
    latitude: float
    longitude: float
    altitude: float
    relative_altitude: float
    north_velocity: float
    east_velocity: float
    down_velocity: float
    roll: float
    pitch: float
    yaw: float
    angular_vel_x: float
    angular_vel_y: float
    angular_vel_z: float
    linear_acc_x: float
    linear_acc_y: float
    linear_acc_z: float
    temperature: float
    ground_speed: float
    ground_track: float

@dataclass
class WindData:
    speed: float
    direction: float
    altitude: float
    timestamp: float
    confidence: float
    method: str
    sample_count: int
    std_dev_speed: float
    std_dev_direction: float
    air_density: float

class AerodynamicWindCalculator:
    def __init__(self, buffer_size: int = 200):
        self.buffer_size = buffer_size
        self.data_buffer = deque(maxlen=buffer_size)
        self.wind_estimates = deque(maxlen=100)
        self.lock = threading.Lock()
        self.gravity = 9.81
        self.air_density_sea_level = 1.225
        self.drone_mass = 2.0
        self.drag_coefficient = 0.3
        self.frontal_area = 0.15
        self.rotor_thrust_coeff = 0.05
        self.min_ground_speed = 0.2
        self.max_attitude_angle = 90.0
        self.min_samples_for_calc = 3
        self.max_acceleration = 100.0
        self.max_reasonable_wind = 30.0
        self.wind_change_rate_limit = 3.0
        self.wind_north = 0.0
        self.wind_east = 0.0
        self.wind_variance = [1.0, 1.0]
        self.process_noise = 0.02
        self.measurement_noise = 0.2
        self.last_update_time = 0.0
        self.data_received_count = 0
        self.data_accepted_count = 0
        logging.info("🛠️ Enhanced EKF Wind Calculator initialized with aerodynamic fusion")

    def add_drone_data(self, data: Dict[str, Any]) -> bool:
        self.data_received_count += 1
        
        # Log the first few data packets to see the structure
        if self.data_received_count <= 5:
            logging.info(f"📦 Sample data packet #{self.data_received_count}: {data}")
        
        try:
            # More flexible field mapping - try multiple possible field names
            field_mappings = {
                'latitude': ['latitude', 'lat', 'position_lat', 'lat_deg'],
                'longitude': ['longitude', 'lon', 'lng', 'position_lon', 'lon_deg'],
                'altitude_from_sealevel': ['altitude_from_sealevel', 'altitude', 'alt', 'absolute_altitude', 'altitude_msl'],
                'relative_altitude': ['relative_altitude', 'altitude_rel', 'alt_rel', 'height'],
                'north_m_s': ['north_m_s', 'velocity_north', 'vel_north', 'vn', 'velocity_n'],
                'east_m_s': ['east_m_s', 'velocity_east', 'vel_east', 've', 'velocity_e'],
                'down_m_s': ['down_m_s', 'velocity_down', 'vel_down', 'vd', 'velocity_d'],
                'roll_deg': ['roll_deg', 'roll', 'attitude_roll', 'roll_angle'],
                'pitch_deg': ['pitch_deg', 'pitch', 'attitude_pitch', 'pitch_angle'],
                'yaw_deg': ['yaw_deg', 'yaw', 'attitude_yaw', 'yaw_angle', 'heading'],
                'unix_time': ['unix_time', 'timestamp', 'time', 'time_unix', 'unix_timestamp']
            }
            
            extracted_data = {}
            missing_fields = []
            
            for required_field, possible_names in field_mappings.items():
                found = False
                for name in possible_names:
                    if name in data and data[name] is not None:
                        try:
                            extracted_data[required_field] = float(data[name])
                            found = True
                            break
                        except (ValueError, TypeError):
                            continue
                
                if not found:
                    missing_fields.append(required_field)
            
            if missing_fields:
                if self.data_received_count <= 5:
                    logging.warning(f"❌ Missing fields: {missing_fields}")
                    logging.info(f"📋 Available fields: {list(data.keys())}")
                return False
            
            # Log successful field extraction for first few packets
            if self.data_received_count <= 5:
                logging.info(f"✅ Successfully extracted: {extracted_data}")
            
            ground_speed = math.hypot(extracted_data['north_m_s'], extracted_data['east_m_s'])
            ground_track = math.degrees(math.atan2(extracted_data['east_m_s'], extracted_data['north_m_s'])) % 360

            drone_state = DroneState(
                timestamp=extracted_data['unix_time'],
                latitude=extracted_data['latitude'],
                longitude=extracted_data['longitude'],
                altitude=extracted_data['altitude_from_sealevel'],
                relative_altitude=extracted_data['relative_altitude'],
                north_velocity=extracted_data['north_m_s'],
                east_velocity=extracted_data['east_m_s'],
                down_velocity=extracted_data['down_m_s'],
                roll=extracted_data['roll_deg'],
                pitch=extracted_data['pitch_deg'],
                yaw=extracted_data['yaw_deg'],
                angular_vel_x=float(data.get('angular_velocity_forward_rad_s', 0.0)),
                angular_vel_y=float(data.get('angular_velocity_right_rad_s', 0.0)),
                angular_vel_z=float(data.get('angular_velocity_down_rad_s', 0.0)),
                linear_acc_x=float(data.get('linear_acceleration_forward_m_s2', 0.0)),
                linear_acc_y=float(data.get('linear_acceleration_right_m_s2', 0.0)),
                linear_acc_z=float(data.get('linear_acceleration_down_m_s2', 0.0)),
                temperature=float(data.get('temperature_degc', 20.0)),
                ground_speed=ground_speed,
                ground_track=ground_track
            )

            if not self._is_valid_sample(drone_state):
                if self.data_received_count <= 5:
                    logging.warning(f"❌ Sample validation failed for: {drone_state}")
                return False

            with self.lock:
                self.data_buffer.append(drone_state)
            
            self.data_accepted_count += 1
            if self.data_accepted_count % 10 == 0:
                logging.info(f"📊 Accepted {self.data_accepted_count}/{self.data_received_count} samples, buffer size: {len(self.data_buffer)}")
            
            return True

        except Exception as e:
            if self.data_received_count <= 5:
                logging.error(f"❌ Data parsing error: {e}, data: {data}")
            return False

    def _is_valid_sample(self, state: DroneState) -> bool:
        try:
            # Check for finite values
            if not all(map(math.isfinite, [
                state.roll, state.pitch, state.yaw,
                state.north_velocity, state.east_velocity,
                state.altitude, state.temperature
            ])):
                return False
            
            # Check for reasonable ranges
            if abs(state.roll) > 90 or abs(state.pitch) > 90:
                return False
            
            if state.ground_speed > 50:  # 50 m/s is very fast for a drone
                return False
            
            return True
        except Exception:
            return False

    def calculate_air_density(self, altitude: float, temperature: float) -> float:
        try:
            altitude = max(0, min(10000, altitude))
            temp_k = max(223.15, min(323.15, temperature + 273.15))
            pressure_ratio = (1 - 0.0065 * altitude / 288.15) ** 5.255
            return max(0.1, min(1.5, self.air_density_sea_level * pressure_ratio * (288.15 / temp_k)))
        except Exception:
            return self.air_density_sea_level

    def estimate_airspeed_ekf(self, state: DroneState) -> Tuple[float, float]:
        try:
            roll_rad, pitch_rad = map(math.radians, [state.roll, state.pitch])
            tilt_mag = math.sqrt(roll_rad**2 + pitch_rad**2)
            yaw_rad = math.radians(state.yaw)

            if tilt_mag < math.radians(0.5):
                return state.north_velocity, state.east_velocity

            air_density = self.calculate_air_density(state.altitude, state.temperature)
            thrust = self.drone_mass * (self.gravity + state.linear_acc_z) * self.rotor_thrust_coeff
            airspeed = math.sqrt(max(0, (2 * thrust * math.tan(tilt_mag)) / 
                                  (air_density * self.drag_coefficient * self.frontal_area)))
            airspeed = min(airspeed, 25.0)
            air_north = airspeed * math.cos(yaw_rad + math.atan2(roll_rad, pitch_rad))
            air_east = airspeed * math.sin(yaw_rad + math.atan2(roll_rad, pitch_rad))
            return air_north, air_east
        except Exception:
            return state.north_velocity, state.east_velocity

    def calculate_wind_ekf(self) -> Optional[WindData]:
        try:
            with self.lock:
                buffer_size = len(self.data_buffer)
                
            if buffer_size < self.min_samples_for_calc:
                logging.debug(f"⏳ Buffer size {buffer_size} < min_samples_for_calc {self.min_samples_for_calc}")
                return None
                
            with self.lock:
                recent_data = list(self.data_buffer)[-self.min_samples_for_calc:]

            logging.debug(f"🔄 Calculating wind with {len(recent_data)} samples")

            dt = time.time() - self.last_update_time
            if dt > 0:
                self.wind_variance[0] += self.process_noise * dt
                self.wind_variance[1] += self.process_noise * dt

            wind_north_meas, wind_east_meas = 0.0, 0.0
            valid_measurements = 0

            for state in recent_data:
                ground_n, ground_e = state.north_velocity, state.east_velocity
                air_n, air_e = self.estimate_airspeed_ekf(state)
                wind_n = ground_n - air_n
                wind_e = ground_e - air_e

                if math.isfinite(wind_n) and math.isfinite(wind_e) and math.hypot(wind_n, wind_e) <= self.max_reasonable_wind:
                    wind_north_meas += wind_n
                    wind_east_meas += wind_e
                    valid_measurements += 1

            if valid_measurements == 0:
                logging.debug("❌ No valid wind measurements in this batch")
                return None

            wind_north_meas /= valid_measurements
            wind_east_meas /= valid_measurements

            kalman_gain_n = self.wind_variance[0] / (self.wind_variance[0] + self.measurement_noise**2)
            kalman_gain_e = self.wind_variance[1] / (self.wind_variance[1] + self.measurement_noise**2)
            self.wind_north += kalman_gain_n * (wind_north_meas - self.wind_north)
            self.wind_east += kalman_gain_e * (wind_east_meas - self.wind_east)
            self.wind_variance[0] *= (1 - kalman_gain_n)
            self.wind_variance[1] *= (1 - kalman_gain_e)

            wind_speed = math.hypot(self.wind_north, self.wind_east)
            wind_dir = (math.degrees(math.atan2(-self.wind_east, -self.wind_north)) + 360) % 360

            # Calculate statistics
            wind_speeds = []
            wind_dirs = []
            for state in recent_data:
                ground_n, ground_e = state.north_velocity, state.east_velocity
                air_n, air_e = self.estimate_airspeed_ekf(state)
                wind_n = ground_n - air_n
                wind_e = ground_e - air_e
                if math.isfinite(wind_n) and math.isfinite(wind_e):
                    wind_speeds.append(math.hypot(wind_n, wind_e))
                    wind_dirs.append((math.degrees(math.atan2(-wind_e, -wind_n)) + 360) % 360)

            std_speed = statistics.stdev(wind_speeds) if len(wind_speeds) > 1 else 0.0
            std_dir = self._circular_std_dev(wind_dirs)

            confidence = self._calculate_confidence(wind_speeds, wind_dirs, std_speed, std_dir)

            wind = WindData(
                speed=wind_speed,
                direction=wind_dir,
                altitude=statistics.mean([s.altitude for s in recent_data]),
                timestamp=time.time(),
                confidence=confidence,
                method="EKF-Aerodynamic",
                sample_count=len(recent_data),
                std_dev_speed=std_speed,
                std_dev_direction=std_dir,
                air_density=self.calculate_air_density(statistics.mean([s.altitude for s in recent_data]), 
                                                    statistics.mean([s.temperature for s in recent_data]))
            )
            self.wind_estimates.append(wind)
            self.last_update_time = time.time()
            logging.info(f"✅ Wind calculated: {wind_speed:.2f} m/s at {wind_dir:.1f}° (confidence: {confidence:.2f})")
            return wind
        except Exception as e:
            logging.error(f"❌ Wind calculation error: {e}")
            return None

    def _circular_std_dev(self, angles: List[float]) -> float:
        if len(angles) < 2:
            return 0.0
        try:
            x = sum(math.cos(math.radians(a)) for a in angles if math.isfinite(a))
            y = sum(math.sin(math.radians(a)) for a in angles if math.isfinite(a))
            R = math.sqrt(x**2 + y**2) / len(angles)
            return math.degrees(math.sqrt(-2 * math.log(R))) if 0 < R < 1 else 0.0
        except Exception:
            return 0.0

    def _calculate_confidence(self, speeds: List[float], dirs: List[float],
                           std_speed: float, std_dir: float) -> float:
        try:
            if not speeds or not dirs:
                return 0.0
            mean_speed = statistics.mean(speeds)
            speed_consistency = 1.0 / (1.0 + std_speed / (mean_speed + 0.01))
            dir_consistency = 1.0 / (1.0 + std_dir / 180.0)
            sample_factor = min(1.0, len(speeds) / self.min_samples_for_calc)
            return max(0.0, min(1.0, 0.4 * speed_consistency + 0.4 * dir_consistency + 0.2 * sample_factor))
        except Exception:
            return 0.0

    def get_wind_statistics(self) -> Dict[str, Any]:
        with self.lock:
            buf_len = len(self.data_buffer)
            est_count = len(self.wind_estimates)
            stats = {
                'samples_in_buffer': buf_len,
                'wind_estimates_made': est_count,
                'data_received': self.data_received_count,
                'data_accepted': self.data_accepted_count,
                'wind_north': self.wind_north,
                'wind_east': self.wind_east,
                'variance_north': self.wind_variance[0],
                'variance_east': self.wind_variance[1]
            }
            if est_count:
                last_10 = list(self.wind_estimates)[-10:]
                stats.update({
                    'mean_speed': statistics.mean([w.speed for w in last_10]),
                    'mean_direction': sum([w.direction for w in last_10]) / len(last_10),
                    'mean_confidence': statistics.mean([w.confidence for w in last_10]),
                })
            return stats

class TCPWindClient:
    def __init__(self, host: str = "localhost", port: int = 9000):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        self.wind_calculator = AerodynamicWindCalculator()
        self.lines_received = 0
        self.last_stats_time = time.time()

    def connect(self) -> bool:
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.connect((self.host, self.port))
            self.running = True
            logging.info(f"✅ Connected to MAVSDK TCP stream at {self.host}:{self.port}")
            return True
        except Exception as e:
            logging.error(f"❌ Connection failed: {e}")
            return False

    def disconnect(self):
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        logging.info("🔌 Disconnected from MAVSDK")

    def listen_and_calculate(self):
        buffer = ""
        last_calc = 0
        calc_interval = 0.5

        while self.running:
            try:
                data = self.socket.recv(4096).decode('utf-8')
                if not data:
                    logging.warning("⚠️ No data received, stream may be closed")
                    break

                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if not line:
                        continue
                    
                    self.lines_received += 1
                    
                    # Log first few lines to see raw data
                    if self.lines_received <= 3:
                        logging.info(f"📥 Raw line #{self.lines_received}: {line[:200]}...")
                    
                    try:
                        json_data = json.loads(line)
                        self.wind_calculator.add_drone_data(json_data)
                    except json.JSONDecodeError as e:
                        if self.lines_received <= 3:
                            logging.warning(f"❌ JSON decode error on line #{self.lines_received}: {e}")
                        continue
                    except Exception as e:
                        logging.error(f"❌ Error processing data line: {e}")
                        continue

                # Show statistics periodically
                if time.time() - self.last_stats_time >= 5.0:
                    stats = self.wind_calculator.get_wind_statistics()
                    logging.info(f"📊 Stats: {stats}")
                    self.last_stats_time = time.time()

                if time.time() - last_calc >= calc_interval:
                    self.display_wind_result()
                    last_calc = time.time()

            except socket.timeout:
                continue
            except Exception as e:
                logging.error(f"⚠️ Socket error: {e}")
                break

    def display_wind_result(self):
        wind = self.wind_calculator.calculate_wind_ekf()
        if wind:
            beaufort = self._beaufort_scale(wind.speed)
            compass = self._direction_to_compass(wind.direction)
            print(f"\n🌬️  Wind Report — {datetime.now().strftime('%H:%M:%S')}")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"📈 Speed     : {wind.speed:.2f} m/s  ({wind.speed * 3.6:.2f} km/h)")
            print(f"🧭 Direction : {wind.direction:.1f}° ({compass})")
            print(f"🌡️  Altitude  : {wind.altitude:.1f} m")
            print(f"🔒 Confidence: {wind.confidence:.2f} | Samples: {wind.sample_count}")
            print(f"🌪️  Category  : {beaufort}")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        else:
            stats = self.wind_calculator.get_wind_statistics()
            print(f"⏳ Gathering data... Buffer: {stats['samples_in_buffer']}, "
                  f"Received: {stats['data_received']}, Accepted: {stats['data_accepted']}")

    def _beaufort_scale(self, speed: float) -> str:
        bft = [(0.3, "0 (Calm)"), (1.6, "1 (Light Air)"), (3.4, "2 (Light Breeze)"),
               (5.5, "3 (Gentle Breeze)"), (8.0, "4 (Moderate Breeze)"), (10.8, "5 (Fresh Breeze)"),
               (13.9, "6 (Strong Breeze)"), (17.2, "7 (High Wind)"), (20.8, "8 (Gale)"),
               (24.5, "9 (Strong Gale)"), (28.5, "10 (Storm)"), (32.6, "11 (Violent Storm)")]
        for threshold, label in bft:
            if speed < threshold:
                return label
        return "12 (Hurricane)"

    def _direction_to_compass(self, angle: float) -> str:
        directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                      "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        idx = int((angle + 11.25) / 22.5) % 16
        return directions[idx]

def main():
    print("🌬️  Advanced EKF Wind Estimator - DIAGNOSTIC MODE")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🔧 Method     : EKF with Aerodynamic + IMU Fusion")
    print("📡 Source     : Live MAVSDK TCP stream (port 9000)")
    print("🧠 Features   : Real-time EKF, Air Density Correction, High Accuracy")
    print("📊 Output     : Speed, Direction, Confidence, Compass & Beaufort")
    print("🛑 Stop with  : Ctrl+C")
    print("🔍 Debug Mode : Enhanced logging and field detection")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    client = TCPWindClient()
    if client.connect():
        try:
            client.listen_and_calculate()
        except KeyboardInterrupt:
            logging.info("👋 Interrupted by user (Ctrl+C)")
        finally:
            client.disconnect()
            print("✅ Disconnected cleanly.")
    else:
        print("❌ Failed to connect to MAVSDK logger.")
        print("💡 Make sure the logger is running and broadcasting on TCP port 9000.")

if __name__ == "__main__":
    main()