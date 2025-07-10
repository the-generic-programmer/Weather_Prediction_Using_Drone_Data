#!/usr/bin/env python3

import csv
import os
import time
import logging
import math
import statistics
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import threading
import glob
import json

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
    airspeed: Optional[float] = None
    temperature: float = 20.0

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
    def __init__(self, buffer_size: int = 200):
        self.buffer_size = buffer_size
        self.data_buffer = deque(maxlen=buffer_size)
        self.wind_estimates = deque(maxlen=50)
        self.lock = threading.Lock()
        
        # Optimized parameters for better detection
        self.min_samples_for_calc = 6
        self.min_ground_speed = 0.5  # Reduced from 2.0 for hovering scenarios
        self.max_reasonable_wind_knots = 100
        
        # Kalman filter parameters
        self.wind_u = 0.0
        self.wind_v = 0.0
        self.wind_u_variance = 4.0
        self.wind_v_variance = 4.0
        self.process_noise = 0.1
        self.measurement_noise = 2.0
        
        # Statistics
        self.data_received_count = 0
        self.data_accepted_count = 0
        self.calculations_made = 0
        
        logging.info("🌬️ Precision Wind Calculator initialized")

    def add_drone_data(self, data: Dict[str, Any]) -> bool:
        """Add drone data with improved field mapping"""
        self.data_received_count += 1
        
        try:
            # Enhanced field mapping
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
                'temperature': ['temperature_degc', 'temperature', 'temp']
            }
            
            extracted_data = {}
            
            # Extract fields with better error handling
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
                elif field in ['down_velocity']:
                    extracted_data[field] = 0.0  # Optional field
                else:
                    return False
            
            # Calculate derived values
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
                temperature=extracted_data['temperature']
            )
            
            if not self._is_valid_sample(drone_state):
                return False
            
            with self.lock:
                self.data_buffer.append(drone_state)
            
            self.data_accepted_count += 1
            return True
            
        except Exception as e:
            if self.data_received_count <= 3:
                logging.debug(f"Data parsing error: {e}")
            return False

    def _is_valid_sample(self, state: DroneState) -> bool:
        """Validate drone state with relaxed constraints"""
        try:
            # Check for finite values
            if not all(math.isfinite(x) for x in [
                state.latitude, state.longitude, state.altitude,
                state.north_velocity, state.east_velocity, state.down_velocity,
                state.roll, state.pitch, state.yaw, state.ground_speed
            ]):
                return False
            
            # Relaxed range checks
            if abs(state.roll) > 90 or abs(state.pitch) > 90:
                return False
            
            if state.ground_speed > 50:  # 50 m/s max
                return False
            
            if abs(state.latitude) > 90 or abs(state.longitude) > 180:
                return False
            
            return True
            
        except Exception:
            return False

    def calculate_wind_multi_method(self) -> Optional[WindData]:
        """Calculate wind using multiple methods for maximum accuracy"""
        try:
            with self.lock:
                if len(self.data_buffer) < self.min_samples_for_calc:
                    return None
                
                recent_data = list(self.data_buffer)[-100:]
            
            # Method 1: Velocity Triangle Method (Most Accurate)
            wind_velocity = self._calculate_velocity_triangle_method(recent_data)
            
            # Method 2: Drift Analysis Method
            wind_drift = self._calculate_drift_method(recent_data)
            
            # Method 3: Acceleration-Based Method
            wind_accel = self._calculate_acceleration_method(recent_data)
            
            # Method 4: Position Drift Method
            wind_position = self._calculate_position_drift_method(recent_data)
            
            # Combine methods with weights
            wind_estimates = []
            weights = []
            
            if wind_velocity:
                wind_estimates.append(wind_velocity)
                weights.append(0.4)  # Highest weight for velocity method
            
            if wind_drift:
                wind_estimates.append(wind_drift)
                weights.append(0.3)
            
            if wind_accel:
                wind_estimates.append(wind_accel)
                weights.append(0.2)
            
            if wind_position:
                wind_estimates.append(wind_position)
                weights.append(0.1)
            
            if not wind_estimates:
                return None
            
            # Weighted average of estimates
            final_wind = self._weighted_average_estimates(wind_estimates, weights)
            
            # Apply Kalman filtering for smoothing
            final_wind = self._apply_kalman_filter(final_wind)
            
            # Create wind data object
            wind_data = self._create_wind_data(final_wind, recent_data)
            
            with self.lock:
                self.wind_estimates.append(wind_data)
            
            self.calculations_made += 1
            return wind_data
            
        except Exception as e:
            logging.error(f"Wind calculation error: {e}")
            return None

    def _calculate_velocity_triangle_method(self, samples: List[DroneState]) -> Optional[Tuple[float, float]]:
        """Most accurate method: True airspeed vs ground speed vector triangle"""
        if len(samples) < 4:
            return None
        
        wind_estimates = []
        
        for i in range(1, len(samples)):
            curr = samples[i]
            prev = samples[i-1]
            
            # Skip if no movement
            if curr.ground_speed < self.min_ground_speed:
                continue
            
            # Calculate intended velocity vector (from heading)
            intended_north = curr.ground_speed * math.cos(math.radians(curr.yaw))
            intended_east = curr.ground_speed * math.sin(math.radians(curr.yaw))
            
            # Actual velocity vector
            actual_north = curr.north_velocity
            actual_east = curr.east_velocity
            
            # Wind vector = actual - intended
            wind_north = actual_north - intended_north
            wind_east = actual_east - intended_east
            
            # Filter reasonable values
            wind_speed = math.hypot(wind_east, wind_north)
            if wind_speed < 25:  # 25 m/s reasonable limit
                wind_estimates.append((wind_east, wind_north))
        
        if len(wind_estimates) < 3:
            return None
        
        # Remove outliers and average
        wind_estimates = self._remove_outliers(wind_estimates)
        if not wind_estimates:
            return None
        
        wind_u = np.mean([est[0] for est in wind_estimates])
        wind_v = np.mean([est[1] for est in wind_estimates])
        
        return (wind_u, wind_v)

    def _calculate_drift_method(self, samples: List[DroneState]) -> Optional[Tuple[float, float]]:
        """Classic drift method using ground track vs heading"""
        if len(samples) < 4:
            return None
        
        wind_estimates = []
        
        for sample in samples:
            if sample.ground_speed < self.min_ground_speed:
                continue
            
            # Calculate drift angle
            drift_angle = (sample.ground_track - sample.yaw + 360) % 360
            if drift_angle > 180:
                drift_angle -= 360
            
            # Skip if drift is too small
            if abs(drift_angle) < 2:
                continue
            
            # Calculate wind using drift geometry
            drift_rad = math.radians(drift_angle)
            wind_speed = sample.ground_speed * math.sin(drift_rad)
            
            # Wind direction perpendicular to intended heading
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
        """Use acceleration data to detect wind effects"""
        if len(samples) < 6:
            return None
        
        wind_estimates = []
        
        for i in range(2, len(samples)-2):
            curr = samples[i]
            prev = samples[i-1]
            next_sample = samples[i+1]
            
            # Calculate acceleration from velocity changes
            dt1 = curr.timestamp - prev.timestamp
            dt2 = next_sample.timestamp - curr.timestamp
            
            if dt1 <= 0 or dt2 <= 0 or dt1 > 2 or dt2 > 2:
                continue
            
            # Velocity changes
            dvn_dt = (next_sample.north_velocity - prev.north_velocity) / (dt1 + dt2)
            dve_dt = (next_sample.east_velocity - prev.east_velocity) / (dt1 + dt2)
            
            # Expected acceleration for straight flight should be minimal
            # Large accelerations indicate wind effects
            if abs(dvn_dt) > 0.5 or abs(dve_dt) > 0.5:
                # Estimate wind from unexpected acceleration
                wind_u = dve_dt * 2.0  # Empirical scaling
                wind_v = dvn_dt * 2.0
                
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
        """Analyze position drift over time"""
        if len(samples) < 10:
            return None
        
        # Look for periods of intended stationary flight
        stationary_periods = []
        
        for i in range(5, len(samples)-5):
            window = samples[i-5:i+5]
            speeds = [s.ground_speed for s in window]
            
            if max(speeds) < 1.0:  # Hovering period
                stationary_periods.extend(window)
        
        if len(stationary_periods) < 5:
            return None
        
        # Calculate drift rate during stationary periods
        wind_estimates = []
        
        for i in range(1, len(stationary_periods)):
            curr = stationary_periods[i]
            prev = stationary_periods[i-1]
            
            dt = curr.timestamp - prev.timestamp
            if dt <= 0 or dt > 5:
                continue
            
            # Position change rate during hover
            # Convert lat/lon to meters (approximate)
            dlat_m = (curr.latitude - prev.latitude) * 111320
            dlon_m = (curr.longitude - prev.longitude) * 111320 * math.cos(math.radians(curr.latitude))
            
            # Wind velocity from position drift
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
        """Remove outliers using IQR method"""
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
        
        return [estimates[i] for i in valid_indices]

    def _weighted_average_estimates(self, estimates: List[Tuple[float, float]], weights: List[float]) -> Tuple[float, float]:
        """Calculate weighted average of wind estimates"""
        if not estimates:
            return (0.0, 0.0)
        
        total_weight = sum(weights)
        if total_weight == 0:
            return (0.0, 0.0)
        
        wind_u = sum(est[0] * weight for est, weight in zip(estimates, weights)) / total_weight
        wind_v = sum(est[1] * weight for est, weight in zip(estimates, weights)) / total_weight
        
        return (wind_u, wind_v)

    def _apply_kalman_filter(self, measurement: Tuple[float, float]) -> Tuple[float, float]:
        """Apply Kalman filter for smooth wind estimates"""
        wind_u_meas, wind_v_meas = measurement
        
        # Prediction step (assuming constant wind)
        wind_u_pred = self.wind_u
        wind_v_pred = self.wind_v
        
        # Update variances
        self.wind_u_variance += self.process_noise
        self.wind_v_variance += self.process_noise
        
        # Kalman gain
        ku = self.wind_u_variance / (self.wind_u_variance + self.measurement_noise)
        kv = self.wind_v_variance / (self.wind_v_variance + self.measurement_noise)
        
        # Update estimates
        self.wind_u = wind_u_pred + ku * (wind_u_meas - wind_u_pred)
        self.wind_v = wind_v_pred + kv * (wind_v_meas - wind_v_pred)
        
        # Update variances
        self.wind_u_variance = (1 - ku) * self.wind_u_variance
        self.wind_v_variance = (1 - kv) * self.wind_v_variance
        
        return (self.wind_u, self.wind_v)

    def _create_wind_data(self, wind_components: Tuple[float, float], samples: List[DroneState]) -> WindData:
        """Create WindData object from wind components"""
        wind_u, wind_v = wind_components
        
        # Calculate wind speed and direction
        wind_speed_ms = math.hypot(wind_u, wind_v)
        wind_speed_knots = wind_speed_ms * 1.94384
        
        # Wind direction (meteorological convention)
        wind_direction = (math.degrees(math.atan2(-wind_u, -wind_v)) + 360) % 360
        
        # Statistics
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
        """Calculate confidence based on multiple factors"""
        if len(samples) < 3:
            return 0.2
        
        # Sample count factor
        sample_factor = min(1.0, len(samples) / 30.0)
        
        # Speed variability factor
        speeds = [s.ground_speed for s in samples]
        speed_std = np.std(speeds) if len(speeds) > 1 else 0
        speed_factor = min(1.0, speed_std / 3.0)
        
        # Heading variability factor
        headings = [s.yaw for s in samples]
        heading_std = np.std(headings) if len(headings) > 1 else 0
        heading_factor = min(1.0, heading_std / 30.0)
        
        # Wind strength factor (moderate winds are more reliable)
        wind_factor = 1.0 if 3 <= wind_speed <= 25 else 0.7
        
        # Combined confidence
        confidence = 0.3 * sample_factor + 0.2 * speed_factor + 0.2 * heading_factor + 0.3 * wind_factor
        
        return max(0.1, min(1.0, confidence))

    def get_statistics(self) -> Dict[str, Any]:
        """Get calculator statistics"""
        with self.lock:
            return {
                'samples_in_buffer': len(self.data_buffer),
                'data_received': self.data_received_count,
                'data_accepted': self.data_accepted_count,
                'calculations_made': self.calculations_made,
                'wind_estimates_stored': len(self.wind_estimates),
                'current_wind_speed_knots': math.hypot(self.wind_u, self.wind_v) * 1.94384,
                'current_wind_direction': (math.degrees(math.atan2(-self.wind_u, -self.wind_v)) + 360) % 360,
                'acceptance_rate': self.data_accepted_count / max(1, self.data_received_count),
                'buffer_utilization': len(self.data_buffer) / self.buffer_size
            }

class CSVWindMonitor:
    def __init__(self, csv_directory: str = "mavsdk_logs"):
        self.csv_directory = csv_directory
        self.wind_calculator = PrecisionWindCalculator()
        self.last_file_size = {}
        self.running = False
        self.last_stats_time = time.time()
        self.wind_output_file = "wind_measurements.json"
        
    def find_latest_csv(self) -> Optional[str]:
        """Find the most recent CSV file"""
        try:
            pattern = os.path.join(self.csv_directory, "drone_log_*.csv")
            csv_files = glob.glob(pattern)
            
            if not csv_files:
                return None
            
            latest_file = max(csv_files, key=os.path.getctime)
            return latest_file
            
        except Exception as e:
            logging.error(f"Error finding CSV files: {e}")
            return None
    
    def read_new_data(self, csv_file: str) -> List[Dict[str, Any]]:
        """Read new data from CSV file efficiently"""
        try:
            current_size = os.path.getsize(csv_file)
            last_size = self.last_file_size.get(csv_file, 0)
            
            if current_size <= last_size:
                return []
            
            new_rows = []
            with open(csv_file, 'r', newline='', encoding='utf-8') as f:
                if last_size > 0:
                    f.seek(last_size)
                    f.readline()  # Skip partial line
                
                csv_reader = csv.DictReader(f)
                for row in csv_reader:
                    new_rows.append(row)
            
            self.last_file_size[csv_file] = current_size
            return new_rows
            
        except Exception as e:
            logging.error(f"Error reading CSV file: {e}")
            return []
    
    def save_wind_measurement(self, wind: WindData):
        """Save wind measurement to JSON file"""
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
            
            try:
                with open(self.wind_output_file, 'r') as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                data = []
            
            data.append(measurement)
            
            # Keep only last 1000 measurements
            if len(data) > 1000:
                data = data[-1000:]
            
            with open(self.wind_output_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error saving wind measurement: {e}")
    
    def start_monitoring(self):
        """Start monitoring CSV files for new data"""
        self.running = True
        logging.info(f"🔍 Starting precision wind monitoring in: {self.csv_directory}")
        
        while self.running:
            try:
                latest_csv = self.find_latest_csv()
                
                if latest_csv:
                    new_rows = self.read_new_data(latest_csv)
                    
                    if new_rows:
                        for row in new_rows:
                            self.wind_calculator.add_drone_data(row)
                    
                    # Calculate wind using multi-method approach
                    wind = self.wind_calculator.calculate_wind_multi_method()
                    if wind:
                        self.display_wind_result(wind)
                        self.save_wind_measurement(wind)
                    
                    # Show statistics periodically
                    if time.time() - self.last_stats_time >= 10.0:
                        stats = self.wind_calculator.get_statistics()
                        logging.info(f"📊 Stats: {stats}")
                        self.last_stats_time = time.time()
                
                else:
                    logging.warning(f"⚠️ No CSV files found in {self.csv_directory}")
                
                time.sleep(0.2)  # High frequency monitoring
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logging.error(f"Monitoring error: {e}")
                time.sleep(1)
    
    def display_wind_result(self, wind: WindData):
        """Display wind calculation results"""
        beaufort = self._beaufort_scale(wind.speed_knots)
        compass = self._direction_to_compass(wind.direction_degrees)
        
        print(f"\n🌬️  PRECISION WIND MEASUREMENT — {datetime.now().strftime('%H:%M:%S')}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"💨 Speed      : {wind.speed_knots:.2f} knots ({wind.speed_knots * 1.852:.1f} km/h)")
        print(f"🧭 Direction  : {wind.direction_degrees:.1f}° ({compass})")
        print(f"🏔️ Altitude from sealevel  : {wind.altitude:.1f} m")
#        print(f"🎯 Confidence : {wind.confidence:.3f} | Samples: {wind.sample_count}")
 #       print(f"📊 Components : E={wind.u_component:.2f} m/s, N={wind.v_component:.2f} m/s")
        print(f"🌪️ Category   : {beaufort}")
        print(f"⚙️ Method     : {wind.method}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    def _beaufort_scale(self, speed_knots: float) -> str:
        """Convert wind speed to Beaufort scale"""
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
        """Convert angle to compass direction"""
        directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                     "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        idx = int((angle + 11.25) / 22.5) % 16
        return directions[idx]
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False

if __name__ == "__main__":
    try:
        monitor = CSVWindMonitor(csv_directory="mavsdk_logs")
        monitor.start_monitoring()
    except KeyboardInterrupt:
        monitor.stop_monitoring()
        logging.info("🛑 Monitoring stopped by user")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
