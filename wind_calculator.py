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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
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

        # Physical constants and drone parameters
        self.gravity = 9.81
        self.air_density_sea_level = 1.225
        self.drone_mass = 2.0  # kg (adjust based on SITL model)
        self.drag_coefficient = 0.3  # Increased for better aerodynamic fit
        self.frontal_area = 0.15  # m² (adjust based on SITL model)
        self.rotor_thrust_coeff = 0.05  # Empirical thrust coefficient

        # Limits and thresholds
        self.min_ground_speed = 0.2  # Lowered for quicker detection
        self.max_attitude_angle = 60.0  # Increased for SITL flexibility
        self.min_samples_for_calc = 5  # Reduced to 5 for faster output
        self.max_acceleration = 30.0  # Increased for SITL dynamics
        self.max_reasonable_wind = 30.0
        self.wind_change_rate_limit = 3.0

        # EKF state
        self.wind_north = 0.0
        self.wind_east = 0.0
        self.wind_variance = [1.0, 1.0]  # [north, east] variance
        self.process_noise = 0.02  # Reduced for faster convergence
        self.measurement_noise = 0.2  # Adjusted for SITL noise
        self.last_update_time = 0.0

        logging.info("🛠️ Enhanced EKF Wind Calculator initialized with aerodynamic fusion")

    def add_drone_data(self, data: Dict[str, Any]) -> bool:
        try:
            # Mandatory fields for valid wind estimation
            fields = ['latitude', 'longitude', 'altitude_from_sealevel', 'relative_altitude',
                      'north_m_s', 'east_m_s', 'down_m_s', 'roll_deg', 'pitch_deg', 'yaw_deg', 'unix_time']
            for field in fields:
                if field not in data or data[field] is None:
                    return False
                float(data[field])  # Validate type

            ground_speed = math.hypot(float(data['north_m_s']), float(data['east_m_s']))
            ground_track = math.degrees(math.atan2(float(data['east_m_s']), float(data['north_m_s']))) % 360

            drone_state = DroneState(
                timestamp=float(data['unix_time']),
                latitude=float(data['latitude']),
                longitude=float(data['longitude']),
                altitude=float(data['altitude_from_sealevel']),
                relative_altitude=float(data['relative_altitude']),
                north_velocity=float(data['north_m_s']),
                east_velocity=float(data['east_m_s']),
                down_velocity=float(data['down_m_s']),
                roll=float(data['roll_deg']),
                pitch=float(data['pitch_deg']),
                yaw=float(data['yaw_deg']),
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
                return False

            with self.lock:
                self.data_buffer.append(drone_state)
            return True

        except (ValueError, TypeError, KeyError):
            return False

    def _is_valid_sample(self, state: DroneState) -> bool:
        try:
            if (abs(state.roll) > self.max_attitude_angle or
                abs(state.pitch) > self.max_attitude_angle or
                math.sqrt(state.linear_acc_x**2 + state.linear_acc_y**2 + state.linear_acc_z**2) > self.max_acceleration or
                state.ground_speed > 100.0 or  # Increased for SITL range
                not all(map(math.isfinite, [
                    state.roll, state.pitch, state.yaw,
                    state.north_velocity, state.east_velocity,
                    state.altitude, state.temperature
                ]))):
                logging.debug(f"Sample rejected: {state}")
                return False
            return True
        except Exception:
            return False

    def calculate_air_density(self, altitude: float, temperature: float) -> float:
        try:
            altitude = max(0, min(10000, altitude))
            temp_k = max(223.15, min(323.15, temperature + 273.15))
            pressure_ratio = (1 - 0.0065 * altitude / 288.15) ** 5.255
            density = self.air_density_sea_level * pressure_ratio * (288.15 / temp_k)
            return max(0.1, min(1.5, density))
        except Exception:
            return self.air_density_sea_level

    def estimate_airspeed_ekf(self, state: DroneState) -> Tuple[float, float]:
        try:
            # Use IMU and attitude for airspeed estimation
            roll_rad, pitch_rad = map(math.radians, [state.roll, state.pitch])
            tilt_mag = math.sqrt(roll_rad**2 + pitch_rad**2)
            yaw_rad = math.radians(state.yaw)

            if tilt_mag < math.radians(0.5):  # Minimal tilt, use ground velocity
                air_north, air_east = state.north_velocity, state.east_velocity
            else:
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
                if len(self.data_buffer) < self.min_samples_for_calc:
                    logging.debug(f"Buffer size {len(self.data_buffer)} < min_samples_for_calc {self.min_samples_for_calc}")
                    return None
                recent_data = list(self.data_buffer)[-self.min_samples_for_calc:]

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
                return None

            wind_north_meas /= valid_measurements
            wind_east_meas /= valid_measurements

            # EKF update
            kalman_gain_n = self.wind_variance[0] / (self.wind_variance[0] + self.measurement_noise**2)
            kalman_gain_e = self.wind_variance[1] / (self.wind_variance[1] + self.measurement_noise**2)
            self.wind_north += kalman_gain_n * (wind_north_meas - self.wind_north)
            self.wind_east += kalman_gain_e * (wind_east_meas - self.wind_east)
            self.wind_variance[0] *= (1 - kalman_gain_n)
            self.wind_variance[1] *= (1 - kalman_gain_e)

            wind_speed = math.hypot(self.wind_north, self.wind_east)
            wind_dir = (math.degrees(math.atan2(-self.wind_east, -self.wind_north)) + 360) % 360

            speeds = [math.hypot(ground_n - air_n, ground_e - air_e) for state in recent_data 
                     if math.isfinite(ground_n - air_n) and math.isfinite(ground_e - air_e)]
            dirs = [(math.degrees(math.atan2(-(ground_e - air_e), -(ground_n - air_n))) + 360) % 360 
                   for state in recent_data if math.isfinite(ground_n - air_n) and math.isfinite(ground_e - air_e)]
            std_speed = statistics.stdev(speeds) if len(speeds) > 1 else 0.0
            std_dir = self._circular_std_dev(dirs)

            confidence = self._calculate_confidence(speeds, dirs, std_speed, std_dir)

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
            return wind
        except Exception as e:
            logging.error(f"Wind calculation failed: {e}")
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
                'wind_north': self.wind_north,
                'wind_east': self.wind_east,
                'variance_north': self.wind_variance[0],
                'variance_east': self.wind_variance[1]
            }
            if est_count:
                last_10 = list(self.wind_estimates)[-10:]
                stats.update({
                    'mean_speed': statistics.mean([w.speed for w in last_10]),
                    'mean_direction': sum([w.direction for w in last_10]) / len(last_10),  # Simplified for speed
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
        calc_interval = 0.5  # Reduced to 0.5s for faster updates

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
                    try:
                        json_data = json.loads(line)
                        logging.debug(f"Received JSON data: {json_data}")  # Debug log
                        if not self.wind_calculator.add_drone_data(json_data):
                            logging.debug(f"Data rejected: {json_data}")  # Debug rejected data
                    except json.JSONDecodeError as e:
                        logging.debug(f"JSON error: {e} in line: {line[:60]}")
                        continue
                    except Exception as e:
                        logging.error(f"Error processing data line: {e}")
                        continue

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
            print("⏳ Gathering data... Waiting for stable wind estimate...")

    def _beaufort_scale(self, speed: float) -> str:
        bft = [
            (0.3, "0 (Calm)"), (1.6, "1 (Light Air)"), (3.4, "2 (Light Breeze)"),
            (5.5, "3 (Gentle Breeze)"), (8.0, "4 (Moderate Breeze)"), (10.8, "5 (Fresh Breeze)"),
            (13.9, "6 (Strong Breeze)"), (17.2, "7 (High Wind)"), (20.8, "8 (Gale)"),
            (24.5, "9 (Strong Gale)"), (28.5, "10 (Storm)"), (32.6, "11 (Violent Storm)"),
        ]
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
    print("🌬️  Advanced EKF Wind Estimator")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🔧 Method     : EKF with Aerodynamic + IMU Fusion")
    print("📡 Source     : Live MAVSDK TCP stream (port 9000)")
    print("🧠 Features   : Real-time EKF, Air Density Correction, High Accuracy")
    print("📊 Output     : Speed, Direction, Confidence, Compass & Beaufort")
    print("🛑 Stop with  : Ctrl+C")
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