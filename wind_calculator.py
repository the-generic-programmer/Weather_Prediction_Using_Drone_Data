#!/usr/bin/env python3

import asyncio
import json
import logging
import math
import statistics
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple, Callable
from datetime import datetime
import time

# MAVSDK imports
from mavsdk import System
from mavsdk.telemetry import Position, VelocityNed, EulerAngle, Imu

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
    acceleration_x: float = 0.0
    acceleration_y: float = 0.0

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
    def __init__(self, buffer_duration: float = 8.0):
        self.buffer_duration = buffer_duration
        self.data_buffer = deque(maxlen=150)
        self.wind_estimates = deque(maxlen=50)
        self.lock = asyncio.Lock()
        
        # Configuration parameters
        self.min_samples_for_calc = 5
        self.min_ground_speed = 0.3
        self.max_reasonable_wind_knots = 100
        
        # Kalman filter state
        self.wind_u = 0.0
        self.wind_v = 0.0
        self.wind_u_variance = 4.0
        self.wind_v_variance = 4.0
        self.process_noise = 0.05
        self.measurement_noise = 1.5
        
        # Statistics
        self.data_received_count = 0
        self.data_accepted_count = 0
        self.calculations_made = 0
        self.last_motion_state = None
        
        logging.info("🌬️ Precision Wind Calculator initialized")

    async def add_drone_data(self, drone_state: DroneState) -> bool:
        """Add drone state data to the buffer"""
        self.data_received_count += 1
        
        if not self._is_valid_sample(drone_state):
            return False
        
        async with self.lock:
            # Remove old data based on timestamp
            current_time = drone_state.timestamp
            while (len(self.data_buffer) > 0 and 
                   current_time - self.data_buffer[0].timestamp > self.buffer_duration):
                self.data_buffer.popleft()
            
            self.data_buffer.append(drone_state)
        
        self.data_accepted_count += 1
        return True

    def _is_valid_sample(self, state: DroneState) -> bool:
        """Validate drone state sample"""
        try:
            # Check for finite values
            if not all(math.isfinite(x) for x in [
                state.latitude, state.longitude, state.altitude,
                state.north_velocity, state.east_velocity, state.down_velocity,
                state.roll, state.pitch, state.yaw, state.ground_speed
            ]):
                return False
            
            # Check reasonable ranges
            if abs(state.roll) > 90 or abs(state.pitch) > 90:
                return False
            
            if state.ground_speed > 50:  # 50 m/s reasonable max
                return False
            
            if abs(state.latitude) > 90 or abs(state.longitude) > 180:
                return False
            
            return True
        except Exception:
            return False

    def _classify_motion(self, samples: List[DroneState]) -> str:
        """Classify drone motion type"""
        if len(samples) < 3:
            return "unknown"
        
        speeds = [s.ground_speed for s in samples]
        yaws = [s.yaw for s in samples]
        
        avg_speed = np.mean(speeds)
        yaw_std = np.std(yaws) if len(yaws) > 1 else 0
        
        if avg_speed < 0.8 and yaw_std < 8.0:
            return "hover"
        elif avg_speed < 0.8 and yaw_std > 25.0:
            return "turning"
        elif avg_speed >= 0.8 and yaw_std < 12.0:
            return "forward"
        else:
            return "complex"

    async def calculate_wind_multi_method(self) -> Optional[WindData]:
        """Calculate wind using multiple methods with adaptive weighting"""
        try:
            async with self.lock:
                if len(self.data_buffer) < self.min_samples_for_calc:
                    return None
                
                recent_data = list(self.data_buffer)
            
            # Classify motion and adapt filter if needed
            motion_state = self._classify_motion(recent_data)
            if self.last_motion_state and motion_state != self.last_motion_state:
                self._reset_kalman_on_motion_change()
            self.last_motion_state = motion_state
            
            # Calculate wind using different methods
            method_results = {}
            
            # Velocity triangle method (most reliable for moving drone)
            wind_velocity = self._calculate_velocity_triangle_method(recent_data)
            if wind_velocity:
                method_results['velocity'] = wind_velocity
            
            # Drift method (good for steady flight)
            wind_drift = self._calculate_drift_method(recent_data)
            if wind_drift:
                method_results['drift'] = wind_drift
            
            # Acceleration method (sensitive to wind changes)
            wind_accel = self._calculate_acceleration_method(recent_data)
            if wind_accel:
                method_results['accel'] = wind_accel
            
            # Position drift method (good for hovering)
            wind_position = self._calculate_position_drift_method(recent_data)
            if wind_position:
                method_results['position'] = wind_position
            
            if not method_results:
                return None
            
            # Apply adaptive weighting based on motion state
            wind_estimates = list(method_results.values())
            weights = self._get_adaptive_weights(motion_state, method_results.keys())
            
            # Combine estimates
            final_wind = self._weighted_average_estimates(wind_estimates, weights)
            
            # Apply Kalman filter for smoothing
            final_wind = self._apply_kalman_filter(final_wind)
            
            # Create wind data object
            wind_data = self._create_wind_data(final_wind, recent_data, method_results.keys())
            
            async with self.lock:
                self.wind_estimates.append(wind_data)
            
            self.calculations_made += 1
            return wind_data
            
        except Exception as e:
            logging.error(f"Wind calculation error: {e}")
            return None

    def _get_adaptive_weights(self, motion_state: str, methods: List[str]) -> List[float]:
        """Get adaptive weights based on motion state and available methods"""
        method_weights = {
            'hover': {'velocity': 0.3, 'drift': 0.2, 'accel': 0.2, 'position': 0.3},
            'turning': {'velocity': 0.4, 'drift': 0.3, 'accel': 0.2, 'position': 0.1},
            'forward': {'velocity': 0.5, 'drift': 0.3, 'accel': 0.2, 'position': 0.0},
            'complex': {'velocity': 0.4, 'drift': 0.3, 'accel': 0.2, 'position': 0.1}
        }
        
        weights = []
        total_weight = 0
        
        for method in methods:
            weight = method_weights.get(motion_state, {}).get(method, 0.25)
            weights.append(weight)
            total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(methods)] * len(methods)
        
        return weights

    def _calculate_velocity_triangle_method(self, samples: List[DroneState]) -> Optional[Tuple[float, float]]:
        """Calculate wind using velocity triangle method"""
        if len(samples) < 3:
            return None
        
        wind_estimates = []
        
        for sample in samples:
            if sample.ground_speed < self.min_ground_speed:
                continue
            
            # Calculate intended velocity (based on heading)
            intended_north = sample.ground_speed * math.cos(math.radians(sample.yaw))
            intended_east = sample.ground_speed * math.sin(math.radians(sample.yaw))
            
            # Wind is the difference between actual and intended velocity
            wind_north = sample.north_velocity - intended_north
            wind_east = sample.east_velocity - intended_east
            
            # Filter unreasonable estimates
            wind_speed = math.hypot(wind_east, wind_north)
            if wind_speed < 25:  # 25 m/s max reasonable wind
                wind_estimates.append((wind_east, wind_north))
        
        if len(wind_estimates) < 2:
            return None
        
        # Remove outliers and average
        wind_estimates = self._remove_outliers(wind_estimates)
        if not wind_estimates:
            return None
        
        wind_u = np.mean([est[0] for est in wind_estimates])
        wind_v = np.mean([est[1] for est in wind_estimates])
        
        return (wind_u, wind_v)

    def _calculate_drift_method(self, samples: List[DroneState]) -> Optional[Tuple[float, float]]:
        """Calculate wind using drift angle method"""
        if len(samples) < 3:
            return None
        
        wind_estimates = []
        
        for sample in samples:
            if sample.ground_speed < self.min_ground_speed:
                continue
            
            # Calculate drift angle
            drift_angle = (sample.ground_track - sample.yaw + 360) % 360
            if drift_angle > 180:
                drift_angle -= 360
            
            # Skip if drift is too small (noise)
            if abs(drift_angle) < 2:
                continue
            
            # Calculate wind speed from drift
            drift_rad = math.radians(drift_angle)
            wind_speed = sample.ground_speed * math.sin(drift_rad)
            
            # Calculate wind direction
            wind_dir_rad = math.radians(sample.yaw) + (math.pi/2 if drift_angle > 0 else -math.pi/2)
            
            wind_u = wind_speed * math.cos(wind_dir_rad)
            wind_v = wind_speed * math.sin(wind_dir_rad)
            
            if abs(wind_speed) < 20:  # Reasonable wind speed
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
        """Calculate wind using acceleration method"""
        if len(samples) < 5:
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
            
            # Calculate acceleration
            dvn_dt = (next_sample.north_velocity - prev.north_velocity) / (dt1 + dt2)
            dve_dt = (next_sample.east_velocity - prev.east_velocity) / (dt1 + dt2)
            
            # Wind estimate from acceleration (simplified)
            wind_u = dve_dt * 1.5  # Scaling factor
            wind_v = dvn_dt * 1.5
            
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
        """Calculate wind using position drift during hover"""
        if len(samples) < 8:
            return None
        
        # Find stationary periods
        stationary_samples = []
        for i in range(3, len(samples)-3):
            window = samples[i-3:i+3]
            speeds = [s.ground_speed for s in window]
            
            if max(speeds) < 0.8:  # Hovering
                stationary_samples.append(samples[i])
        
        if len(stationary_samples) < 5:
            return None
        
        wind_estimates = []
        
        for i in range(1, len(stationary_samples)):
            curr = stationary_samples[i]
            prev = stationary_samples[i-1]
            
            dt = curr.timestamp - prev.timestamp
            if dt <= 0 or dt > 5:
                continue
            
            # Calculate position drift
            dlat_m = (curr.latitude - prev.latitude) * 111320  # Convert to meters
            dlon_m = (curr.longitude - prev.longitude) * 111320 * math.cos(math.radians(curr.latitude))
            
            # Wind components
            wind_v = dlat_m / dt  # North component
            wind_u = dlon_m / dt  # East component
            
            if math.hypot(wind_u, wind_v) < 8:  # Reasonable drift
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
        """Combine multiple wind estimates using weighted average"""
        if not estimates:
            return (0.0, 0.0)
        
        total_weight = sum(weights)
        if total_weight == 0:
            return (0.0, 0.0)
        
        wind_u = sum(est[0] * weight for est, weight in zip(estimates, weights)) / total_weight
        wind_v = sum(est[1] * weight for est, weight in zip(estimates, weights)) / total_weight
        
        return (wind_u, wind_v)

    def _apply_kalman_filter(self, measurement: Tuple[float, float]) -> Tuple[float, float]:
        """Apply Kalman filter for smoothing"""
        wind_u_meas, wind_v_meas = measurement
        
        # Predict step
        wind_u_pred = self.wind_u
        wind_v_pred = self.wind_v
        
        # Update variance
        self.wind_u_variance += self.process_noise
        self.wind_v_variance += self.process_noise
        
        # Kalman gain
        ku = self.wind_u_variance / (self.wind_u_variance + self.measurement_noise)
        kv = self.wind_v_variance / (self.wind_v_variance + self.measurement_noise)
        
        # Update step
        self.wind_u = wind_u_pred + ku * (wind_u_meas - wind_u_pred)
        self.wind_v = wind_v_pred + kv * (wind_v_meas - wind_v_pred)
        
        # Update variance
        self.wind_u_variance = (1 - ku) * self.wind_u_variance
        self.wind_v_variance = (1 - kv) * self.wind_v_variance
        
        return (self.wind_u, self.wind_v)

    def _create_wind_data(self, wind_components: Tuple[float, float], samples: List[DroneState], methods: List[str]) -> WindData:
        """Create WindData object from wind components"""
        wind_u, wind_v = wind_components
        
        # Convert to speed and direction
        wind_speed_ms = math.hypot(wind_u, wind_v)
        wind_speed_knots = wind_speed_ms * 1.94384  # Convert m/s to knots
        
        # Wind direction (where wind is coming from)
        wind_direction = (math.degrees(math.atan2(-wind_u, -wind_v)) + 360) % 360
        
        # Average altitude
        altitudes = [s.altitude for s in samples if s.altitude is not None]
        avg_altitude = statistics.mean(altitudes) if altitudes else 0
        
        # Calculate confidence
        confidence = self._calculate_confidence(samples, wind_speed_knots)
        
        return WindData(
            speed_knots=wind_speed_knots,
            direction_degrees=wind_direction,
            altitude=avg_altitude,
            timestamp=time.time(),
            confidence=confidence,
            method=f"Multi-Method ({'+'.join(methods)})",
            sample_count=len(samples),
            u_component=wind_u,
            v_component=wind_v
        )

    def _calculate_confidence(self, samples: List[DroneState], wind_speed: float) -> float:
        """Calculate confidence level for wind estimate"""
        if len(samples) < 3:
            return 0.2
        
        # Sample size factor
        sample_factor = min(1.0, len(samples) / 25.0)
        
        # Speed consistency factor
        speeds = [s.ground_speed for s in samples]
        speed_std = np.std(speeds) if len(speeds) > 1 else 0
        speed_factor = min(1.0, max(0.1, 1.0 - speed_std / 5.0))
        
        # Heading consistency factor
        headings = [s.yaw for s in samples]
        heading_std = np.std(headings) if len(headings) > 1 else 0
        heading_factor = min(1.0, max(0.1, 1.0 - heading_std / 45.0))
        
        # Wind speed reasonableness factor
        wind_factor = 1.0 if 2 <= wind_speed <= 30 else 0.6
        
        confidence = 0.3 * sample_factor + 0.3 * speed_factor + 0.2 * heading_factor + 0.2 * wind_factor
        
        return max(0.1, min(1.0, confidence))

    def _reset_kalman_on_motion_change(self):
        """Reset Kalman filter on motion change"""
        self.wind_u_variance = 4.0
        self.wind_v_variance = 4.0
        logging.debug("Kalman filter reset due to motion change")

    async def get_statistics(self) -> dict:
        """Get calculator statistics"""
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
                'buffer_utilization': len(self.data_buffer) / 150
            }

class MAVSDKWindMonitor:
    def __init__(self, connection_string: str = "udp://:14540"):
        self.connection_string = connection_string
        self.system = System()
        self.wind_calculator = PrecisionWindCalculator(buffer_duration=8.0)
        self.running = False
        self.last_stats_time = time.time()
        self.wind_callbacks = []
        
        # Data storage
        self.current_position = None
        self.current_velocity = None
        self.current_attitude = None
        self.current_imu = None

    def add_wind_callback(self, callback: Callable[[WindData], None]):
        """Add callback function to receive wind updates"""
        self.wind_callbacks.append(callback)

    async def connect(self):
        """Connect to drone via MAVSDK"""
        try:
            logging.info(f"🔗 Connecting to drone at {self.connection_string}")
            await self.system.connect(system_address=self.connection_string)
            
            # Wait for connection
            async for state in self.system.core.connection_state():
                if state.is_connected:
                    logging.info("✅ Connected to drone")
                    break
            
            # Wait for position estimate
            async for health in self.system.telemetry.health():
                if health.is_global_position_ok:
                    logging.info("🛰️ Global position estimate OK")
                    break
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Connection failed: {e}")
            return False

    async def start_telemetry_streams(self):
        """Start telemetry data streams"""
        try:
            # Start position stream
            asyncio.create_task(self._position_stream())
            
            # Start velocity stream
            asyncio.create_task(self._velocity_stream())
            
            # Start attitude stream
            asyncio.create_task(self._attitude_stream())
            
            # Start IMU stream (if available)
            asyncio.create_task(self._imu_stream())
            
            logging.info("📡 Telemetry streams started")
            
        except Exception as e:
            logging.error(f"❌ Failed to start telemetry streams: {e}")

    async def _position_stream(self):
        """Handle position telemetry stream"""
        try:
            async for position in self.system.telemetry.position():
                self.current_position = position
                await self._update_wind_calculation()
        except Exception as e:
            logging.error(f"Position stream error: {e}")

    async def _velocity_stream(self):
        """Handle velocity telemetry stream"""
        try:
            async for velocity in self.system.telemetry.velocity_ned():
                self.current_velocity = velocity
                await self._update_wind_calculation()
        except Exception as e:
            logging.error(f"Velocity stream error: {e}")

    async def _attitude_stream(self):
        """Handle attitude telemetry stream"""
        try:
            async for attitude in self.system.telemetry.attitude_euler():
                self.current_attitude = attitude
                await self._update_wind_calculation()
        except Exception as e:
            logging.error(f"Attitude stream error: {e}")

    async def _imu_stream(self):
        """Handle IMU telemetry stream"""
        try:
            async for imu in self.system.telemetry.imu():
                self.current_imu = imu
        except Exception as e:
            logging.debug(f"IMU stream error (optional): {e}")

    async def _update_wind_calculation(self):
        """Update wind calculation when new data arrives"""
        if not all([self.current_position, self.current_velocity, self.current_attitude]):
            return
        
        try:
            # Calculate ground speed and track
            ground_speed = math.hypot(self.current_velocity.north_m_s, self.current_velocity.east_m_s)
            ground_track = math.degrees(math.atan2(self.current_velocity.east_m_s, self.current_velocity.north_m_s)) % 360
            
            # Get IMU data if available
            accel_x = self.current_imu.acceleration_frd.forward_m_s2 if self.current_imu else 0.0
            accel_y = self.current_imu.acceleration_frd.right_m_s2 if self.current_imu else 0.0
            
            # Create drone state
            drone_state = DroneState(
                timestamp=time.time(),
                latitude=self.current_position.latitude_deg,
                longitude=self.current_position.longitude_deg,
                altitude=self.current_position.absolute_altitude_m,
                north_velocity=self.current_velocity.north_m_s,
                east_velocity=self.current_velocity.east_m_s,
                down_velocity=self.current_velocity.down_m_s,
                roll=self.current_attitude.roll_deg,
                pitch=self.current_attitude.pitch_deg,
                yaw=self.current_attitude.yaw_deg,
                ground_speed=ground_speed,
                ground_track=ground_track,
                acceleration_x=accel_x,
                acceleration_y=accel_y
            )
            
            # Add data to wind calculator
            await self.wind_calculator.add_drone_data(drone_state)
            
            # Calculate wind
            wind_data = await self.wind_calculator.calculate_wind_multi_method()
            
            if wind_data:
                # Call all registered callbacks
                for callback in self.wind_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(wind_data)
                        else:
                            callback(wind_data)
                    except Exception as e:
                        logging.error(f"Wind callback error: {e}")
                
                # Display wind data
                self._display_wind_data(wind_data)
            
        except Exception as e:
            logging.error(f"Wind calculation update error: {e}")

    def _display_wind_data(self, wind: WindData):
        """Display wind data in a formatted way"""
        beaufort = self._beaufort_scale(wind.speed_knots)
        compass = self._direction_to_compass(wind.direction_degrees)
        
        # Only print every second to avoid spam
        if time.time() - getattr(self, '_last_display_time', 0) >= 1.0:
            print(f"\n🌬️  LIVE WIND DATA — {datetime.now().strftime('%H:%M:%S')}")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"💨 Speed      : {wind.speed_knots:.1f} knots ({wind.speed_knots * 1.852:.1f} km/h)")
            print(f"🧭 Direction  : {wind.direction_degrees:.1f}° ({compass})")
            print(f"🏔️ Altitude   : {wind.altitude:.1f} m")
            print(f"📊 Confidence : {wind.confidence:.1%}")
            print(f"🌪️ Category   : {beaufort}")
            print(f"⚙️ Method     : {wind.method}")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            self._last_display_time = time.time()

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

    async def start_monitoring(self):
        """Start the wind monitoring system"""
        self.running = True
        
        # Connect to drone
        if not await self.connect():
            return False
        
        # Start telemetry streams
        await self.start_telemetry_streams()
        
        # Start statistics reporting
        asyncio.create_task(self._statistics_reporter())
        
        logging.info("🌬️ Wind monitoring system started")
        
        try:
            # Keep the system running
            while self.running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logging.info("🛑 Monitoring stopped by user")
        except Exception as e:
            logging.error(f"❌ Monitoring error: {e}")
        finally:
            self.running = False
        
        return True

    async def _statistics_reporter(self):
        """Periodically report statistics"""
        while self.running:
            try:
                await asyncio.sleep(30)  # Report every 30 seconds
                
                if time.time() - self.last_stats_time >= 30:
                    stats = await self.wind_calculator.get_statistics()
                    logging.info(f"📊 Wind Calculator Stats: {stats}")
                    self.last_stats_time = time.time()
                    
            except Exception as e:
                logging.error(f"Statistics reporter error: {e}")

    def stop_monitoring(self):
        """Stop the wind monitoring system"""
        self.running = False
        logging.info("🛑 Wind monitoring stopped")

# Example usage and callback functions
class WindDataHandler:
    def __init__(self):
        self.wind_history = deque(maxlen=100)
        self.wind_file = "live_wind_data.json"
    
    async def on_wind_update(self, wind_data: WindData):
        """Handle new wind data"""
        self.wind_history.append(wind_data)
        
        # Save to file
        await self._save_wind_data(wind_data)
        
        # Custom processing
        await self._process_wind_data(wind_data)
    
    async def _save_wind_data(self, wind_data: WindData):
        """Save wind data to JSON file"""
        try:
            wind_record = {
                'timestamp': wind_data.timestamp,
                'datetime': datetime.fromtimestamp(wind_data.timestamp).isoformat(),
                'speed_knots': wind_data.speed_knots,
                'direction_degrees': wind_data.direction_degrees,
                'altitude_m': wind_data.altitude,
                'confidence': wind_data.confidence,
                'method': wind_data.method,
                'sample_count': wind_data.sample_count,
                'u_component_ms': wind_data.u_component,
                'v_component_ms': wind_data.v_component
            }
            
            # Read existing data
            try:
                with open(self.wind_file, 'r') as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                data = []
            
            # Add new record
            data.append(wind_record)
            
            # Keep only last 1000 records
            if len(data) > 1000:
                data = data[-1000:]
            
            # Save back to file
            with open(self.wind_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error saving wind data: {e}")
    
    async def _process_wind_data(self, wind_data: WindData):
        """Process wind data for custom applications"""
        # Example: Check for strong wind conditions
        if wind_data.speed_knots > 25:
            logging.warning(f"⚠️  Strong wind detected: {wind_data.speed_knots:.1f} knots")
        
        # Example: Check for wind direction changes
        if len(self.wind_history) >= 2:
            prev_wind = self.wind_history[-2]
            dir_change = abs(wind_data.direction_degrees - prev_wind.direction_degrees)
            if dir_change > 180:
                dir_change = 360 - dir_change
            
            if dir_change > 30:  # Significant direction change
                logging.info(f"🔄 Wind direction change: {prev_wind.direction_degrees:.1f}° → {wind_data.direction_degrees:.1f}°")
    
    def get_wind_statistics(self) -> dict:
        """Get wind statistics from history"""
        if not self.wind_history:
            return {}
        
        speeds = [w.speed_knots for w in self.wind_history]
        directions = [w.direction_degrees for w in self.wind_history]
        
        return {
            'count': len(self.wind_history),
            'avg_speed_knots': statistics.mean(speeds),
            'max_speed_knots': max(speeds),
            'min_speed_knots': min(speeds),
            'speed_std': statistics.stdev(speeds) if len(speeds) > 1 else 0,
            'avg_direction': statistics.mean(directions),
            'latest_update': datetime.fromtimestamp(self.wind_history[-1].timestamp).isoformat()
        }

# Simple wind display callback
def simple_wind_display(wind_data: WindData):
    """Simple callback to display wind data"""
    print(f"Wind: {wind_data.speed_knots:.1f} knots @ {wind_data.direction_degrees:.1f}° (Confidence: {wind_data.confidence:.1%})")

# Main execution function
async def main():
    """Main execution function"""
    # Create wind monitor
    monitor = MAVSDKWindMonitor(connection_string="udp://:14540")
    
    # Create wind data handler
    wind_handler = WindDataHandler()
    
    # Add callbacks
    monitor.add_wind_callback(wind_handler.on_wind_update)
    monitor.add_wind_callback(simple_wind_display)
    
    # Start monitoring
    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        monitor.stop_monitoring()
        logging.info("🛑 Program terminated by user")
    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}")
    finally:
        # Print final statistics
        if wind_handler.wind_history:
            stats = wind_handler.get_wind_statistics()
            logging.info(f"📈 Final Wind Statistics: {stats}")

# Alternative connection strings for different setups
CONNECTION_EXAMPLES = {
    'simulator': 'udp://:14540',           # PX4 SITL simulator
    'serial': 'serial:///dev/ttyUSB0',     # Serial connection
    'tcp': 'tcp://192.168.1.100:5760',    # TCP connection
    'udp_custom': 'udp://192.168.1.100:14550'  # Custom UDP
}

if __name__ == "__main__":
    # You can change the connection string here based on your setup
    # monitor = MAVSDKWindMonitor(CONNECTION_EXAMPLES['simulator'])
    
    asyncio.run(main())