#!/usr/bin/env python3

import asyncio
import json
import logging
import math
import statistics
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple
from datetime import datetime
import time

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
    north_velocity: float
    east_velocity: float
    down_velocity: float
    roll: float
    pitch: float
    yaw: float
    ground_speed: float
    ground_track: float
    temperature: float = 20.0
    acceleration_x: float = 0.0  # Optional IMU data
    acceleration_y: float = 0.0  # Optional IMU data

@dataclass
class WindData:
    speed_knots: float
    direction_degrees: float
    altitude: float
    timestamp: float
    confidence: float
    method: str
    sample_count: int
    u_component: float  # East wind component
    v_component: float  # North wind component

class PrecisionWindCalculator:
    def __init__(self, buffer_duration: float = 10.0):  # Changed to duration-based
        self.buffer_duration = buffer_duration  # Seconds of data to retain
        self.data_buffer = deque(maxlen=200)  # Initial max length, adjusted dynamically
        self.wind_estimates = deque(maxlen=50)
        self.lock = asyncio.Lock()
        
        self.min_samples_for_calc = 6
        self.min_ground_speed = 0.5
        self.max_reasonable_wind_knots = 100
        
        self.wind_u = 0.0
        self.wind_v = 0.0
        self.wind_u_variance = 4.0
        self.wind_v_variance = 4.0
        self.process_noise = 0.1
        self.measurement_noise = 2.0
        
        self.data_received_count = 0
        self.data_accepted_count = 0
        self.calculations_made = 0
        self.last_motion_state = None
        
        logging.info("🌬️ Precision Wind Calculator initialized")

    async def add_drone_data(self, data: dict) -> bool:
        self.data_received_count += 1
        
        try:
            field_mappings = {
                'timestamp': ['timestamp', 'unix_time', 'time', 'system_time'],
                'latitude': ['latitude', 'lat', 'lat_deg'],
                'longitude': ['longitude', 'lon', 'lng', 'lon_deg'],
                'altitude': ['altitude_from_sealevel', 'altitude', 'alt', 'absolute_altitude_m', 'relative_altitude'],
                'north_velocity': ['north_m_s', 'velocity_north', 'vel_north', 'vn'],
                'east_velocity': ['east_m_s', 'velocity_east', 'vel_east', 've'],
                'down_velocity': ['down_m_s', 'velocity_down', 'vel_down', 'vd'],
                'roll': ['roll_deg', 'roll', 'attitude_roll_deg'],
                'pitch': ['pitch_deg', 'pitch', 'attitude_pitch_deg'],
                'yaw': ['yaw_deg', 'yaw', 'heading', 'attitude_yaw_deg'],
                'temperature': ['temperature_degc', 'temperature', 'temp'],
                'acceleration_x': ['acceleration_x', 'accel_x'],
                'acceleration_y': ['acceleration_y', 'accel_y']
            }
            
            extracted_data = {}
            
            for field, possible_names in field_mappings.items():
                value = None
                for name in possible_names:
                    if name in data and data[name] is not None and str(data[name]).strip() != '':
                        try:
                            value = float(data[name])
                            break
                        except (ValueError, TypeError):
                            continue
                
                if value is not None:
                    extracted_data[field] = value
                elif field == 'timestamp':
                    extracted_data[field] = time.time()
                elif field == 'temperature':
                    extracted_data[field] = 20.0
                elif field in ['down_velocity', 'acceleration_x', 'acceleration_y']:
                    extracted_data[field] = 0.0
                else:
                    return False
            
            ground_speed = math.hypot(extracted_data['north_velocity'], extracted_data['east_velocity'])
            ground_track = math.degrees(math.atan2(extracted_data['east_velocity'], extracted_data['north_velocity'])) % 360
            
            drone_state = DroneState(
                timestamp=extracted_data['timestamp'],
                latitude=extracted_data['latitude'],
                longitude=extracted_data['longitude'],
                altitude=extracted_data['altitude'],
                north_velocity=extracted_data['north_velocity'],
                east_velocity=extracted_data['east_velocity'],
                down_velocity=extracted_data['down_velocity'],
                roll=extracted_data['roll'],
                pitch=extracted_data['pitch'],
                yaw=extracted_data['yaw'],
                ground_speed=ground_speed,
                ground_track=ground_track,
                temperature=extracted_data['temperature'],
                acceleration_x=extracted_data['acceleration_x'],
                acceleration_y=extracted_data['acceleration_y']
            )
            
            if not self._is_valid_sample(drone_state):
                return False
            
            async with self.lock:
                # Dynamic buffer adjustment based on duration
                while (len(self.data_buffer) > 0 and 
                       drone_state.timestamp - self.data_buffer[0].timestamp > self.buffer_duration):
                    self.data_buffer.popleft()
                self.data_buffer.append(drone_state)
            
            self.data_accepted_count += 1
            return True
            
        except Exception as e:
            if self.data_received_count <= 3:
                logging.debug(f"Data parsing error: {e}")
            return False

    def _is_valid_sample(self, state: DroneState) -> bool:
        try:
            if not all(math.isfinite(x) for x in [
                state.latitude, state.longitude, state.altitude,
                state.north_velocity, state.east_velocity, state.down_velocity,
                state.roll, state.pitch, state.yaw, state.ground_speed
            ]):
                return False
            
            if abs(state.roll) > 90 or abs(state.pitch) > 90:
                return False
            
            if state.ground_speed > 50:
                return False
            
            if abs(state.latitude) > 90 or abs(state.longitude) > 180:
                return False
            
            return True
        except Exception:
            return False

    def classify_motion(self, samples: List[DroneState]) -> str:
        """Classify motion based on ground speed and yaw standard deviation"""
        if len(samples) < 3:
            return "unknown"
        
        speeds = [s.ground_speed for s in samples]
        yaws = [s.yaw for s in samples]
        
        avg_speed = np.mean(speeds)
        yaw_std = np.std(yaws) if len(yaws) > 1 else 0
        
        if avg_speed < 1.0 and yaw_std < 10.0:
            return "hover"
        elif avg_speed < 1.0 and yaw_std > 30.0:
            return "turning"
        elif avg_speed >= 1.0 and yaw_std < 15.0:
            return "forward"
        else:
            return "complex"

    async def calculate_wind_multi_method(self) -> Optional[WindData]:
        try:
            async with self.lock:
                if len(self.data_buffer) < self.min_samples_for_calc:
                    return None
                
                recent_data = list(self.data_buffer)
            
            motion_state = self.classify_motion(recent_data)
            if self.last_motion_state and motion_state != self.last_motion_state:
                self._reset_kalman_on_motion_change()
                self.last_motion_state = motion_state
            elif not self.last_motion_state:
                self.last_motion_state = motion_state
            
            # Log raw method results
            method_results = {}
            wind_velocity = self._calculate_velocity_triangle_method(recent_data)
            if wind_velocity:
                method_results['velocity'] = wind_velocity
            wind_drift = self._calculate_drift_method(recent_data)
            if wind_drift:
                method_results['drift'] = wind_drift
            wind_accel = self._calculate_acceleration_method(recent_data)
            if wind_accel:
                method_results['accel'] = wind_accel
            wind_position = self._calculate_position_drift_method(recent_data)
            if wind_position:
                method_results['position'] = wind_position
            logging.debug(f"Raw method results: {method_results}")
            
            wind_estimates = [v for v in method_results.values() if v]
            weights = self._get_adaptive_weights(motion_state, len(wind_estimates))
            
            if not wind_estimates:
                return None
            
            final_wind = self._weighted_average_estimates(wind_estimates, weights)
            final_wind = self._apply_kalman_filter(final_wind)
            wind_data = self._create_wind_data(final_wind, recent_data)
            
            async with self.lock:
                self.wind_estimates.append(wind_data)
            
            self.calculations_made += 1
            return wind_data
        except Exception as e:
            logging.error(f"Wind calculation error: {e}")
            return None

    def _get_adaptive_weights(self, motion_state: str, num_methods: int) -> List[float]:
        """Set weights based on motion type"""
        base_weights = {'velocity': 0.4, 'drift': 0.3, 'accel': 0.2, 'position': 0.1}
        if num_methods == 0:
            return []
        
        if motion_state == "hover":
            return [base_weights.get(k, 0.0) for k in ['position', 'velocity', 'drift', 'accel']]
        elif motion_state == "turning":
            return [base_weights.get(k, 0.0) for k in ['drift', 'velocity', 'accel', 'position']]
        elif motion_state == "forward":
            return [base_weights.get(k, 0.0) for k in ['velocity', 'accel', 'drift', 'position']]
        else:  # complex
            return [base_weights.get(k, 0.0) for k in ['velocity', 'drift', 'accel', 'position']]
        return [w / num_methods for w in [0.4, 0.3, 0.2, 0.1] if w > 0][:num_methods]

    def _calculate_velocity_triangle_method(self, samples: List[DroneState]) -> Optional[Tuple[float, float]]:
        if len(samples) < 4:
            return None
        
        wind_estimates = []
        
        for i in range(1, len(samples)):
            curr = samples[i]
            prev = samples[i-1]
            
            if curr.ground_speed < self.min_ground_speed:
                continue
            
            intended_north = curr.ground_speed * math.cos(math.radians(curr.yaw))
            intended_east = curr.ground_speed * math.sin(math.radians(curr.yaw))
            
            actual_north = curr.north_velocity
            actual_east = curr.east_velocity
            
            wind_north = actual_north - intended_north
            wind_east = actual_east - intended_east
            
            wind_speed = math.hypot(wind_east, wind_north)
            if wind_speed < 25:
                wind_estimates.append((wind_east, wind_north))
        
        if len(wind_estimates) < 3:
            return None
        
        wind_estimates = self._remove_outliers(wind_estimates)
        if not wind_estimates:
            return None
        
        wind_u = np.mean([est[0] for est in wind_estimates])
        wind_v = np.mean([est[1] for est in wind_estimates])
        
        return (wind_u, wind_v)

    def _calculate_drift_method(self, samples: List[DroneState]) -> Optional[Tuple[float, float]]:
        if len(samples) < 4:
            return None
        
        wind_estimates = []
        
        for sample in samples:
            if sample.ground_speed < self.min_ground_speed:
                continue
            
            drift_angle = (sample.ground_track - sample.yaw + 360) % 360
            if drift_angle > 180:
                drift_angle -= 360
            
            if abs(drift_angle) < 2:
                continue
            
            drift_rad = math.radians(drift_angle)
            wind_speed = sample.ground_speed * math.sin(drift_rad)
            
            wind_dir_rad = math.radians(sample.yaw) + (math.pi/2 if drift_angle > 0 else -math.pi/2)
            
            wind_u = wind_speed * math.cos(wind_dir_rad)
            wind_v = wind_speed * math.sin(wind_dir_rad)
            
            if abs(wind_speed) < 20:
                wind_estimates.append((wind_u, wind_v))
        
        if len(wind_estimates) < 2:
            return None
        
        wind_estimates = self._remove_outliers(wind_estimates)
        if not wind_estimates:
            return None
        
        wind_u = np.mean([est[0] for est in wind_estimates])
        wind_v = np.mean([est[1] for est in wind_estimates])
        
        return (wind_u, wind_v)

    def _calculate_acceleration_method(self, samples: List[DroneState]) -> Optional[Tuple[float, float]]:
        if len(samples) < 6:
            return None
        
        wind_estimates = []
        
        for i in range(2, len(samples)-2):
            curr = samples[i]
            prev = samples[i-1]
            next_sample = samples[i+1]
            
            dt1 = curr.timestamp - prev.timestamp
            dt2 = next_sample.timestamp - curr.timestamp
            
            if dt1 <= 0 or dt2 <= 0 or dt1 > 2 or dt2 > 2:
                continue
            
            dvn_dt = (next_sample.north_velocity - prev.north_velocity) / (dt1 + dt2)
            dve_dt = (next_sample.east_velocity - prev.east_velocity) / (dt1 + dt2)
            
            # Optional IMU enhancement
            accel_factor = math.hypot(curr.acceleration_x, curr.acceleration_y) if curr.acceleration_x or curr.acceleration_y else 1.0
            wind_u = dve_dt * 2.0 * accel_factor
            wind_v = dvn_dt * 2.0 * accel_factor
            
            if math.hypot(wind_u, wind_v) < 15:
                wind_estimates.append((wind_u, wind_v))
        
        if len(wind_estimates) < 2:
            return None
        
        wind_estimates = self._remove_outliers(wind_estimates)
        if not wind_estimates:
            return None
        
        wind_u = np.mean([est[0] for est in wind_estimates])
        wind_v = np.mean([est[1] for est in wind_estimates])
        
        return (wind_u, wind_v)

    def _calculate_position_drift_method(self, samples: List[DroneState]) -> Optional[Tuple[float, float]]:
        if len(samples) < 10:
            return None
        
        stationary_periods = []
        
        for i in range(5, len(samples)-5):
            window = samples[i-5:i+5]
            speeds = [s.ground_speed for s in window]
            
            if max(speeds) < 1.0:
                stationary_periods.extend(window)
        
        if len(stationary_periods) < 5:
            return None
        
        wind_estimates = []
        
        for i in range(1, len(stationary_periods)):
            curr = stationary_periods[i]
            prev = stationary_periods[i-1]
            
            dt = curr.timestamp - prev.timestamp
            if dt <= 0 or dt > 5:
                continue
            
            dlat_m = (curr.latitude - prev.latitude) * 111320
            dlon_m = (curr.longitude - prev.longitude) * 111320 * math.cos(math.radians(curr.latitude))
            
            wind_v = dlat_m / dt
            wind_u = dlon_m / dt
            
            if math.hypot(wind_u, wind_v) < 10:
                wind_estimates.append((wind_u, wind_v))
        
        if len(wind_estimates) < 2:
            return None
        
        wind_estimates = self._remove_outliers(wind_estimates)
        if not wind_estimates:
            return None
        
        wind_u = np.mean([est[0] for est in wind_estimates])
        wind_v = np.mean([est[1] for est in wind_estimates])
        
        return (wind_u, wind_v)

    def _remove_outliers(self, estimates: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(estimates) < 4:
            return estimates
        
        u_values = [est[0] for est in estimates]
        v_values = [est[1] for est in estimates]
        
        def filter_outliers(values):
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return [i for i, v in enumerate(values) if lower_bound <= v <= upper_bound]
        
        u_indices = set(filter_outliers(u_values))
        v_indices = set(filter_outliers(v_values))
        valid_indices = u_indices.intersection(v_indices)
        
        filtered_estimates = [estimates[i] for i in valid_indices]
        
        # Vector magnitude check
        if filtered_estimates:
            magnitudes = [math.hypot(u, v) for u, v in filtered_estimates]
            median_mag = np.median(magnitudes)
            std_mag = np.std(magnitudes) if len(magnitudes) > 1 else 0
            return [(u, v) for u, v in filtered_estimates 
                    if abs(math.hypot(u, v) - median_mag) <= 1.5 * std_mag]
        return filtered_estimates

    def _weighted_average_estimates(self, estimates: List[Tuple[float, float]], weights: List[float]) -> Tuple[float, float]:
        if not estimates:
            return (0.0, 0.0)
        
        total_weight = sum(weights)
        if total_weight == 0:
            return (0.0, 0.0)
        
        wind_u = sum(est[0] * weight for est, weight in zip(estimates, weights)) / total_weight
        wind_v = sum(est[1] * weight for est, weight in zip(estimates, weights)) / total_weight
        
        # Use median angle for direction
        angles = [math.degrees(math.atan2(-u, -v)) % 360 for u, v in estimates]
        median_angle = np.median(angles) if angles else 0
        wind_dir_rad = math.radians(median_angle)
        wind_speed = math.hypot(wind_u, wind_v)
        wind_u = wind_speed * math.cos(wind_dir_rad)
        wind_v = wind_speed * math.sin(wind_dir_rad)
        
        return (wind_u, wind_v)

    def _apply_kalman_filter(self, measurement: Tuple[float, float]) -> Tuple[float, float]:
        wind_u_meas, wind_v_meas = measurement
        
        wind_u_pred = self.wind_u
        wind_v_pred = self.wind_v
        
        self.wind_u_variance += self.process_noise
        self.wind_v_variance += self.process_noise
        
        ku = self.wind_u_variance / (self.wind_u_variance + self.measurement_noise)
        kv = self.wind_v_variance / (self.wind_v_variance + self.measurement_noise)
        
        self.wind_u = wind_u_pred + ku * (wind_u_meas - wind_u_pred)
        self.wind_v = wind_v_pred + kv * (wind_v_meas - wind_v_pred)
        
        self.wind_u_variance = (1 - ku) * self.wind_u_variance
        self.wind_v_variance = (1 - kv) * self.wind_v_variance
        
        return (self.wind_u, self.wind_v)

    def _create_wind_data(self, wind_components: Tuple[float, float], samples: List[DroneState]) -> WindData:
        wind_u, wind_v = wind_components
        
        wind_speed_ms = math.hypot(wind_u, wind_v)
        wind_speed_knots = wind_speed_ms * 1.94384
        
        wind_direction = (math.degrees(math.atan2(-wind_u, -wind_v)) + 360) % 360
        
        altitudes = [s.altitude for s in samples if s.altitude is not None]
        avg_altitude = statistics.mean(altitudes) if altitudes else 0
        
        confidence = self._calculate_confidence(samples, wind_speed_knots)
        
        return WindData(
            speed_knots=wind_speed_knots,
            direction_degrees=wind_direction,
            altitude=avg_altitude,
            timestamp=time.time(),
            confidence=confidence,
            method="Multi-Method Precision",
            sample_count=len(samples),
            u_component=wind_u,
            v_component=wind_v
        )

    def _calculate_confidence(self, samples: List[DroneState], wind_speed: float) -> float:
        if len(samples) < 3:
            return 0.2
        
        sample_factor = min(1.0, len(samples) / 30.0)
        speeds = [s.ground_speed for s in samples]
        speed_std = np.std(speeds) if len(speeds) > 1 else 0
        speed_factor = min(1.0, speed_std / 3.0)
        
        headings = [s.yaw for s in samples]
        heading_std = np.std(headings) if len(headings) > 1 else 0
        heading_factor = min(1.0, heading_std / 30.0)
        
        wind_factor = 1.0 if 3 <= wind_speed <= 25 else 0.7
        
        confidence = 0.3 * sample_factor + 0.2 * speed_factor + 0.2 * heading_factor + 0.3 * wind_factor
        
        return max(0.1, min(1.0, confidence))

    def _reset_kalman_on_motion_change(self):
        """Reset Kalman filter state on drastic motion change"""
        self.wind_u = 0.0
        self.wind_v = 0.0
        self.wind_u_variance = 4.0
        self.wind_v_variance = 4.0
        logging.info("Kalman filter reset due to motion change")

    async def get_statistics(self) -> dict:
        """Get calculator statistics asynchronously"""
        async with self.lock:
            return {
                'samples_in_buffer': len(self.data_buffer),
                'data_received': self.data_received_count,
                'data_accepted': self.data_accepted_count,
                'calculations_made': self.calculations_made,
                'wind_estimates_stored': len(self.wind_estimates),
                'current_wind_speed_knots': math.hypot(self.wind_u, self.wind_v) * 1.94384,
                'current_wind_direction': (math.degrees(math.atan2(-self.wind_u, -self.wind_v)) + 360) % 360,
                'acceptance_rate': self.data_accepted_count / max(1, self.data_received_count),
                'buffer_utilization': len(self.data_buffer) / 200  # Fixed maxlen
            }

class TCPWindMonitor:
    def __init__(self, host: str = "127.0.0.1", port: int = 9000):
        self.host = host
        self.port = port
        self.wind_calculator = PrecisionWindCalculator(buffer_duration=10.0)
        self.running = False
        self.last_stats_time = time.time()
        self.wind_output_file = "wind_measurements.json"
        self.reader = None
        self.writer = None

    async def connect(self):
        """Establish TCP connection with retry logic"""
        while True:
            try:
                self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
                logging.info(f"🔗 Connected to TCP server at {self.host}:{self.port}")
                break
            except Exception as e:
                logging.error(f"Failed to connect to TCP server: {e}. Retrying in 5 seconds...")
                await asyncio.sleep(5)

    async def receive_data(self) -> Optional[dict]:
        """Receive and parse JSON data from TCP"""
        try:
            if not self.reader:
                await self.connect()
            if not self.reader:
                raise ConnectionError("Failed to establish connection")
            line = await asyncio.wait_for(self.reader.readline(), timeout=2.0)
            if not line:
                raise ConnectionError("Connection closed by server")
            data = json.loads(line.decode().strip())
            logging.debug(f"Received data: {data}")
            return data
        except (asyncio.TimeoutError, json.JSONDecodeError, ConnectionError) as e:
            logging.error(f"Error receiving data: {e}")
            self.reader = self.writer = None
            await asyncio.sleep(1)
            return None

    async def save_wind_measurement(self, wind: WindData):
        """Save wind measurement to JSON file asynchronously"""
        try:
            measurement = {
                'timestamp': wind.timestamp,
                'datetime': datetime.fromtimestamp(wind.timestamp).isoformat(),
                'speed_knots': wind.speed_knots,
                'direction_degrees': wind.direction_degrees,
                'altitude_m': wind.altitude,
                'confidence': wind.confidence,
                'method': wind.method,
                'sample_count': wind.sample_count,
                'u_component_ms': wind.u_component,
                'v_component_ms': wind.v_component
            }
            
            async with asyncio.Lock():
                try:
                    async with asyncio.open('wind_measurements.json', 'r') as f:
                        data = json.loads(await f.read()) if await f.read() else []
                except (FileNotFoundError, json.JSONDecodeError):
                    data = []
                
                data.append(measurement)
                if len(data) > 1000:
                    data = data[-1000:]
                
                async with asyncio.open(self.wind_output_file, 'w') as f:
                    await f.write(json.dumps(data, indent=2))
        except Exception as e:
            logging.error(f"Error saving wind measurement: {e}")

    async def start_monitoring(self):
        """Start monitoring live TCP data"""
        self.running = True
        logging.info(f"🔍 Starting precision wind monitoring on {self.host}:{self.port}")
        
        while self.running:
            try:
                data = await self.receive_data()
                if data and 'north_m_s' in data and 'east_m_s' in data:
                    await self.wind_calculator.add_drone_data(data)
                
                wind = await self.wind_calculator.calculate_wind_multi_method()
                if wind:
                    self.display_wind_result(wind)
                    await self.save_wind_measurement(wind)
                
                if time.time() - self.last_stats_time >= 10.0:
                    stats = await self.wind_calculator.get_statistics()
                    logging.info(f"📊 Stats: {stats}")
                    self.last_stats_time = time.time()
                
                await asyncio.sleep(0.1)  # 10 Hz update rate
            except KeyboardInterrupt:
                break
            except Exception as e:
                logging.error(f"Monitoring error: {e}")
                await asyncio.sleep(1)

    def display_wind_result(self, wind: WindData):
        """Display wind calculation results"""
        beaufort = self._beaufort_scale(wind.speed_knots)
        compass = self._direction_to_compass(wind.direction_degrees)
        
        print(f"\n🌬️  PRECISION WIND MEASUREMENT — {datetime.now().strftime('%H:%M:%S')}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"💨 Speed      : {wind.speed_knots:.2f} knots ({wind.speed_knots * 1.852:.1f} km/h)")
        print(f"🧭 Direction  : {wind.direction_degrees:.1f}° ({compass})")
        print(f"🏔️ Altitude   : {wind.altitude:.1f} m")
        print(f"📊 Components : E={wind.u_component:.2f} m/s, N={wind.v_component:.2f} m/s")
        print(f"🌪️ Category   : {beaufort}")
        print(f"⚙️ Method     : {wind.method}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    def _beaufort_scale(self, speed_knots: float) -> str:
        beaufort_scale = [
            (1, "0 (Calm)"), (3, "1 (Light Air)"), (6, "2 (Light Breeze)"),
            (10, "3 (Gentle Breeze)"), (16, "4 (Moderate Breeze)"), (21, "5 (Fresh Breeze)"),
            (27, "6 (Strong Breeze)"), (33, "7 (Near Gale)"), (40, "8 (Gale)"),
            (47, "9 (Strong Gale)"), (55, "10 (Storm)"), (63, "11 (Violent Storm)")
        ]
        
        for threshold, description in beaufort_scale:
            if speed_knots < threshold:
                return description
        return "12 (Hurricane)"

    def _direction_to_compass(self, angle: float) -> str:
        directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                     "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        idx = int((angle + 11.25) / 22.5) % 16
        return directions[idx]

    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False

async def main():
    monitor = TCPWindMonitor(host="127.0.0.1", port=9000)
    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        monitor.stop_monitoring()
        logging.info("🛑 Monitoring stopped by user")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())