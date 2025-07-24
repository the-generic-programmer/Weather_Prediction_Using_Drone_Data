#!/usr/bin/env python3
"""
Super-Enhanced Precision Wind Measurement (v2.1 - Merged Multi-Method Integration)
====================================================================================
### Key Features:
- Multi-method wind calculation: velocity triangle, drift angle, acceleration, hover drift
- Dynamic method weighting based on flight conditions (hovering, cruising, turning)
- Kalman filter (wind speed) using filterpy, exponential weighted mean (wind direction)
- Enhanced flight state detection with optional acceleration checks
- Removed debug rejected counts line as requested
**Run tip:** If telemetry produces no lines, confirm mavsdk_logger.py is running and streaming JSON to 127.0.0.1:9000.
"""
import asyncio
import json
import math
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import numpy as np
from filterpy.kalman import KalmanFilter

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
TELEMETRY_HOST = "127.0.0.1"
TELEMETRY_PORT = 9000
MAX_HISTORY = 50  # number of recent raw samples used for stats
ACCEL_HISTORY = 10  # how many samples used for accel-based wind
CHECK_INTERVAL = 1.0  # seconds between reconnect attempts
MIN_AIRSPEED_RELIABLE = 3.0  # airspeed for high-confidence wind calculation (m/s)
MIN_AIRSPEED_ABSOLUTE = 0.5  # absolute minimum airspeed to attempt calculation (m/s)
MIN_GROUND_SPEED_RELIABLE = 2.0  # ground speed for high-confidence wind calculation (m/s)
MIN_GROUND_SPEED_ABSOLUTE = 0.2  # absolute minimum ground speed (m/s)
MIN_COMBINED_MOTION = 1.0  # minimum of (airspeed + ground_speed) for calculation
MAX_WIND_SPEED = 100.0  # sanity cap
OUTLIER_THRESHOLD = 1.5  # IQR multiplier
PRINT_EVERY = 1.0  # seconds; throttle pretty prints
DEBUG_REJECTS = False  # Disabled to remove rejected counts debug line
SHOW_TEMP_PRESS = True  # include 🌡️ & ⏲️ lines if present
DEBUG_FLIGHT_STATE = True  # show detailed flight state debugging
KALMAN_R = 0.5  # Measurement noise covariance
KALMAN_Q = 0.01  # Process noise covariance
DIR_SMOOTH_ALPHA = 0.1  # Alpha for exponential weighted mean of direction
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
KT_PER_MS = 1.94384

# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------
@dataclass
class WindResult:
    timestamp: str
    wind_speed: float  # m/s
    wind_speed_knots: float  # kt
    wind_direction: float  # deg FROM
    ground_speed: float  # m/s (horizontal)
    airspeed: float  # TAS m/s
    confidence: float  # %
    label: str  # high/medium/low
    sample_count: int  # raw samples in window
    std_dev_speed: float  # m/s
    std_dev_direction: float  # deg
    outliers_rejected: int  # count
    flight_state: str  # flying/ground/insufficient_speed
    temperature_c: Optional[float] = None
    pressure_hpa: Optional[float] = None
    method_weights: Optional[Dict[str, float]] = None  # Weights for each method

# -----------------------------------------------------------------------------
# Safe parsing helpers
# -----------------------------------------------------------------------------
def _f(x: Optional[object], default: Optional[float] = None) -> Optional[float]:
    """Parse to float; return default on failure; coerce NaN/inf to default."""
    if x is None:
        return default
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return v

def _norm_heading(yaw: Optional[float]) -> Optional[float]:
    """Normalize yaw to 0..360 deg. Accepts -inf..inf; returns None if missing."""
    if yaw is None:
        return None
    try:
        y = float(yaw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(y):
        return None
    y = y % 360.0
    if y < 0:
        y += 360.0
    return y

# -----------------------------------------------------------------------------
# Physics utils
# -----------------------------------------------------------------------------
def ias_to_tas(ias: float, temp_c: Optional[float], press_hpa: Optional[float]) -> float:
    """Convert IAS->TAS; validate ranges; fall back to IAS on bad data."""
    R_AIR = 287.05
    RHO0 = 1.225
    if temp_c is None or press_hpa is None:
        return ias
    try:
        if not (-60.0 <= temp_c <= 60.0) or not (800.0 <= press_hpa <= 1100.0):
            return ias
        t_k = temp_c + 273.15
        p_pa = press_hpa * 100.0
        rho = p_pa / (R_AIR * t_k)
        if rho <= 0:
            return ias
        ratio = RHO0 / rho
        if not (0.5 <= ratio <= 2.0):
            return ias
        return ias * math.sqrt(ratio)
    except Exception:
        return ias

def calc_wind_dir_from(wn: float, we: float) -> float:
    return (math.degrees(math.atan2(-we, -wn)) + 360.0) % 360.0

def circ_mean(degs: List[float]) -> float:
    if not degs:
        return 0.0
    rad = np.radians(degs)
    x = np.cos(rad).sum()
    y = np.sin(rad).sum()
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

def circ_std(degs: List[float], mean_deg: Optional[float] = None) -> float:
    if not degs:
        return 0.0
    if mean_deg is None:
        mean_deg = circ_mean(degs)
    rad = np.radians(degs)
    mean_rad = math.radians(mean_deg)
    cos_sum = np.cos(rad - mean_rad).sum()
    R = cos_sum / len(degs)
    R = max(min(R, 1.0), -1.0)
    circ_var = 1.0 - R
    if circ_var <= 0:
        return 0.0
    try:
        return math.degrees(math.sqrt(-2.0 * math.log(1.0 - circ_var)))
    except ValueError:
        return 0.0

def remove_outliers_iqr(data: List[float], threshold: float = OUTLIER_THRESHOLD) -> Tuple[List[float], int]:
    """Return (clean_list, outlier_count)."""
    if len(data) < 4:
        return list(data), 0
    arr = np.array(data)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lb = q1 - threshold * iqr
    ub = q3 + threshold * iqr
    mask = (arr >= lb) & (arr <= ub)
    return arr[mask].tolist(), int((~mask).sum())

# -----------------------------------------------------------------------------
# Enhanced calculator with multi-method integration
# -----------------------------------------------------------------------------
class EnhancedWindCalculator:
    def __init__(self):
        self.raw = deque(maxlen=MAX_HISTORY)
        self.vn_hist = deque(maxlen=ACCEL_HISTORY)
        self.ve_hist = deque(maxlen=ACCEL_HISTORY)
        self.time_hist = deque(maxlen=ACCEL_HISTORY)
        self.track_hist = deque(maxlen=ACCEL_HISTORY)
        self.heading_hist = deque(maxlen=ACCEL_HISTORY)
        self.last_valid_dir = None
        self.last_valid_speed = None
        self.dir_smooth = None
        self.reject_counts = {"validate": 0, "not_flying": 0, "low_speed": 0}
        self._last_print_t = 0.0
        self._last_status_print = 0.0
        # Initialize Kalman filter for wind speed
        self.wind_speed_kf = KalmanFilter(dim_x=1, dim_z=1)
        self.wind_speed_kf.x = np.array([[0]])
        self.wind_speed_kf.F = np.array([[1]])
        self.wind_speed_kf.H = np.array([[1]])
        self.wind_speed_kf.P *= 1000
        self.wind_speed_kf.R *= KALMAN_R
        self.wind_speed_kf.Q *= KALMAN_Q

    def _validate_basic(self, vn: Optional[float], ve: Optional[float], ias: Optional[float], heading: Optional[float], acc_x: Optional[float] = None, acc_y: Optional[float] = None) -> bool:
        """Basic data validation - check if we have all required fields."""
        if vn is None or ve is None or ias is None or heading is None:
            self.reject_counts["validate"] += 1
            return False
        if abs(vn) > 200 or abs(ve) > 200 or ias > 200 or ias < 0:
            self.reject_counts["validate"] += 1
            return False
        if acc_x is not None and acc_y is not None:
            if abs(acc_x) > 50 or abs(acc_y) > 50:  # Reasonable acceleration limits
                self.reject_counts["validate"] += 1
                return False
        return True

    def _check_flight_state(self, gs: float, tas: float, acc_x: Optional[float], acc_y: Optional[float]) -> str:
        """Determine flight state and wind calculation viability."""
        if tas < MIN_AIRSPEED_ABSOLUTE and gs < MIN_GROUND_SPEED_ABSOLUTE:
            return "stationary"
        combined_motion = tas + gs
        if combined_motion < MIN_COMBINED_MOTION:
            return "minimal_motion"
        acc_mag = math.hypot(acc_x or 0.0, acc_y or 0.0) if acc_x is not None and acc_y is not None else 0.0
        if tas >= MIN_AIRSPEED_RELIABLE and gs >= MIN_GROUND_SPEED_RELIABLE and acc_mag < 2.0:
            return "flying_high_confidence"
        elif tas >= MIN_AIRSPEED_ABSOLUTE or gs >= MIN_GROUND_SPEED_ABSOLUTE:
            return "flying_low_confidence"
        else:
            return "insufficient_motion"

    def _print_status_message(self, flight_state: str, gs: float, tas: float, vn: float, ve: float, ias: float):
        """Print status based on flight state."""
        now = time.time()
        if now - self._last_status_print < 2.0:
            return
        self._last_status_print = now
        current_time = time.strftime("%H:%M:%S")
        combined_motion = tas + gs
        print(f"📍 AIRCRAFT STATUS — {current_time}")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        if DEBUG_FLIGHT_STATE:
            print(f"🔍 DEBUG: Raw values - VN: {vn:.2f} m/s, VE: {ve:.2f} m/s, IAS: {ias:.2f} m/s")
            print(f"🔍 Combined Motion: {combined_motion:.2f} m/s (TAS + GS)")
        if flight_state == "stationary":
            print(f"⏹️ Aircraft appears stationary (TAS: {tas:.2f} < {MIN_AIRSPEED_ABSOLUTE}, GS: {gs:.2f} < {MIN_GROUND_SPEED_ABSOLUTE})")
            print(" No wind calculation possible - aircraft not moving through air or over ground")
        elif flight_state == "minimal_motion":
            print(f"⚠️ Minimal motion detected (Combined: {combined_motion:.2f} < {MIN_COMBINED_MOTION} m/s)")
            print(" Motion too low for reliable wind calculation")
        elif flight_state == "insufficient_motion":
            print(f"⚠️ Insufficient motion for wind calculation")
            print(f" TAS: {tas:.2f} m/s, GS: {gs:.2f} m/s - both below reliable thresholds")
        print(f"🚁 Ground Speed: {gs:.2f} m/s (calculated from VN: {vn:.2f}, VE: {ve:.2f})")
        print(f"✈️ True Airspeed: {tas:.2f} m/s (from IAS: {ias:.2f} m/s)")
        print(f"📏 Thresholds: Reliable TAS: {MIN_AIRSPEED_RELIABLE}, GS: {MIN_GROUND_SPEED_RELIABLE}")
        print(f" Absolute minimums: TAS: {MIN_AIRSPEED_ABSOLUTE}, GS: {MIN_GROUND_SPEED_ABSOLUTE}")
        print("🌬️ Wind calculation: PAUSED")
        print("")

    def _velocity_triangle_method(self, vn: float, ve: float, tas: float, heading: float) -> Tuple[float, float]:
        """Calculate wind using velocity triangle (ground velocity - air velocity)."""
        hr = math.radians(heading or 0.0)
        va_n = tas * math.cos(hr)
        va_e = tas * math.sin(hr)
        wn = vn - va_n
        we = ve - va_e
        w_speed = math.hypot(wn, we)
        w_dir = calc_wind_dir_from(wn, we)
        return w_speed, w_dir

    def _drift_angle_method(self, vn: float, ve: float, tas: float, heading: float, track: Optional[float]) -> Optional[Tuple[float, float]]:
        """Calculate wind using drift angle (difference between heading and track)."""
        if track is None:
            return None
        track_rad = math.radians(_norm_heading(track) or 0.0)
        heading_rad = math.radians(heading)
        gs = math.hypot(vn, ve)
        if tas < MIN_AIRSPEED_ABSOLUTE or gs < MIN_GROUND_SPEED_ABSOLUTE:
            return None
        drift_angle = track_rad - heading_rad
        if abs(drift_angle) > math.radians(45):  # Limit to reasonable drift angles
            return None
        wind_cross = gs * math.sin(drift_angle)
        wind_along = gs * math.cos(drift_angle) - tas
        wn = wind_along * math.cos(heading_rad) - wind_cross * math.sin(heading_rad)
        we = wind_along * math.sin(heading_rad) + wind_cross * math.cos(heading_rad)
        w_speed = math.hypot(wn, we)
        w_dir = calc_wind_dir_from(wn, we)
        return w_speed, w_dir

    def _accel_wind(self, vn_hist: deque, ve_hist: deque, time_hist: deque) -> Optional[Tuple[float, float]]:
        """Estimate wind using acceleration data derived from velocity history."""
        if len(vn_hist) < 3:
            return None
        try:
            times = np.array(time_hist)
            vns = np.array(vn_hist)
            ves = np.array(ve_hist)
            dt = times - times[0]
            vn_slope = np.polyfit(dt, vns, 1)[0]
            ve_slope = np.polyfit(dt, ves, 1)[0]
            acc_mag = math.hypot(vn_slope, ve_slope)
            wind_speed = acc_mag * 2.0  # Scale factor for wind effect
            wind_dir = calc_wind_dir_from(vn_slope, ve_slope)
            return wind_speed, wind_dir
        except Exception:
            return None

    def _hover_drift_method(self, vn: float, ve: float, tas: float, gs: float) -> Optional[Tuple[float, float]]:
        """Estimate wind during hover (TAS near zero, GS indicates wind)."""
        if tas > MIN_AIRSPEED_ABSOLUTE or gs < MIN_GROUND_SPEED_ABSOLUTE:
            return None
        wind_speed = math.hypot(vn, ve)
        wind_dir = calc_wind_dir_from(vn, ve)
        return wind_speed, wind_dir

    def _assign_method_weights(self, flight_state: str, tas: float, gs: float, acc_x: Optional[float], acc_y: Optional[float]) -> Dict[str, float]:
        """Assign weights to each method based on flight conditions."""
        weights = {"velocity_triangle": 0.4, "drift_angle": 0.3, "accel": 0.2, "hover_drift": 0.1}
        acc_mag = math.hypot(acc_x or 0.0, acc_y or 0.0) if acc_x is not None and acc_y is not None else 0.0
        if flight_state == "flying_high_confidence":
            # Cruising: prioritize velocity triangle and drift angle
            weights = {"velocity_triangle": 0.5, "drift_angle": 0.3, "accel": 0.15, "hover_drift": 0.05}
        elif flight_state == "flying_low_confidence":
            if tas < MIN_AIRSPEED_ABSOLUTE and gs >= MIN_GROUND_SPEED_ABSOLUTE:
                # Hovering: prioritize hover drift
                weights = {"velocity_triangle": 0.2, "drift_angle": 0.1, "accel": 0.1, "hover_drift": 0.6}
            else:
                # Low-speed flight: balance methods
                weights = {"velocity_triangle": 0.3, "drift_angle": 0.3, "accel": 0.2, "hover_drift": 0.2}
        if acc_mag > 2.0:
            # High acceleration (turning): increase acceleration method weight
            weights["accel"] += 0.2
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}
        return weights

    def _smooth_wind(self, wind_speed: float, wind_dir: float) -> Tuple[float, float]:
        """Smooth wind speed with Kalman filter and direction with exponential weighted mean."""
        self.wind_speed_kf.predict()
        self.wind_speed_kf.update(np.array([[wind_speed]]))
        smooth_speed = float(self.wind_speed_kf.x[0])
        if self.dir_smooth is None:
            self.dir_smooth = wind_dir
        else:
            diff = ((wind_dir - self.dir_smooth + 180) % 360) - 180
            self.dir_smooth = (self.dir_smooth + DIR_SMOOTH_ALPHA * diff) % 360
        return smooth_speed, self.dir_smooth

    def process(self, data: Dict) -> Optional[WindResult]:
        # Parse telemetry
        vn = _f(data.get("north_m_s"))
        ve = _f(data.get("east_m_s"))
        ias = _f(data.get("airspeed_m_s"))
        temp = _f(data.get("temperature_degc"))
        press = _f(data.get("pressure_hpa"))
        heading = _norm_heading(data.get("yaw_deg"))
        track = _norm_heading(data.get("ground_track_deg") or data.get("course_over_ground_deg") or data.get("desired_heading_deg"))
        acc_x = _f(data.get("acc_x_m_s2"))
        acc_y = _f(data.get("acc_y_m_s2"))
        timestamp = data.get("timestamp")
        ts_now = time.time()

        # Store histories for acceleration and drift angle
        self.vn_hist.append(vn)
        self.ve_hist.append(ve)
        self.time_hist.append(ts_now)
        self.heading_hist.append(heading or 0.0)
        self.track_hist.append(track or heading or 0.0)

        # Basic validation
        if not self._validate_basic(vn, ve, ias, heading, acc_x, acc_y):
            return None

        # Calculate basic metrics
        tas = ias_to_tas(ias, temp, press)
        gs = math.hypot(vn, ve)

        # Check flight state
        flight_state = self._check_flight_state(gs, tas, acc_x, acc_y)
        if flight_state in ["stationary", "minimal_motion", "insufficient_motion"]:
            if flight_state == "stationary":
                self.reject_counts["not_flying"] += 1
            elif flight_state in ["minimal_motion", "insufficient_motion"]:
                self.reject_counts["low_speed"] += 1
            self._print_status_message(flight_state, gs, tas, vn, ve, ias)
            return None

        # Calculate wind using multiple methods
        velocity_triangle = self._velocity_triangle_method(vn, ve, tas, heading)
        drift_angle = self._drift_angle_method(vn, ve, tas, heading, track) if gs > 0.5 and tas > 0.5 and heading is not None and track is not None else None
        accel_method = self._accel_wind(self.vn_hist, self.ve_hist, self.time_hist) if len(self.vn_hist) >= 3 else None
        hover_drift = self._hover_drift_method(vn, ve, tas, gs) if tas < MIN_AIRSPEED_ABSOLUTE and gs > MIN_GROUND_SPEED_ABSOLUTE else None

        # Combine estimates with dynamic weights
        estimates = []
        names = []
        if velocity_triangle:
            estimates.append(velocity_triangle)
            names.append("velocity_triangle")
        if drift_angle:
            estimates.append(drift_angle)
            names.append("drift_angle")
        if accel_method:
            estimates.append(accel_method)
            names.append("accel")
        if hover_drift:
            estimates.append(hover_drift)
            names.append("hover_drift")
        if not estimates:
            return None

        weights_dict = self._assign_method_weights(flight_state, tas, gs, acc_x, acc_y)
        weights = np.array([weights_dict.get(name, 0.0) for name in names])
        w_speeds = np.array([est[0] for est in estimates])
        w_dirs = np.array([est[1] for est in estimates])
        wind_speed = float(np.average(w_speeds, weights=weights)) if sum(weights) > 0 else w_speeds[-1]
        wind_dir = circ_mean(w_dirs.tolist())
        used_weights = {name: weights_dict.get(name, 0.0) for name in names}

        # Apply smoothing
        wind_speed, wind_dir = self._smooth_wind(wind_speed, wind_dir)
        if wind_speed > MAX_WIND_SPEED:
            return None

        # Append raw sample
        self.raw.append({
            "w_speed": wind_speed,
            "w_dir": wind_dir,
            "gs": gs,
            "tas": tas,
            "t": ts_now,
            "temp": temp,
            "press": press,
        })

        # Throttle prints
        now = time.time()
        if now - self._last_print_t < PRINT_EVERY:
            return None
        self._last_print_t = now

        return self._filtered_result(timestamp, flight_state, used_weights)

    def _filtered_result(self, ts: Optional[str], flight_state: str, method_weights: Dict[str, float]) -> WindResult:
        arr = list(self.raw)
        speeds = [s["w_speed"] for s in arr]
        dirs = [s["w_dir"] for s in arr]
        gses = [s["gs"] for s in arr]
        tases = [s["tas"] for s in arr]
        temps = [s["temp"] for s in arr if s["temp"] is not None]
        presss = [s["press"] for s in arr if s["press"] is not None]

        # Outlier removal
        clean_speeds, out_speed = remove_outliers_iqr(speeds)
        clean_gs, out_gs = remove_outliers_iqr(gses)
        clean_tas, out_tas = remove_outliers_iqr(tases)

        # Calculate filtered values
        w_speed_f = float(np.median(clean_speeds)) if clean_speeds else (speeds[-1] if speeds else 0.0)
        gs_f = float(np.median(clean_gs)) if clean_gs else (gses[-1] if gses else 0.0)
        tas_f = float(np.median(clean_tas)) if clean_tas else (tases[-1] if tases else 0.0)
        w_dir_f = circ_mean(dirs)
        w_dir_std = circ_std(dirs, w_dir_f)
        w_speed_std = float(np.std(clean_speeds)) if len(clean_speeds) > 1 else 0.0

        # Low-wind stabilization
        if w_speed_f < 1.5:
            if self.last_valid_dir is not None:
                alpha = 0.05
                w_dir_f = alpha * w_dir_f + (1 - alpha) * self.last_valid_dir
            if self.last_valid_speed is not None:
                w_speed_f = 0.7 * w_speed_f + 0.3 * self.last_valid_speed
        else:
            self.last_valid_dir = w_dir_f
            self.last_valid_speed = w_speed_f

        outliers_total = out_speed + out_gs + out_tas
        conf = self._confidence(len(arr), w_speed_std, w_dir_std, outliers_total, w_speed_f, tas_f, gs_f, flight_state)
        if conf >= 85:
            lbl = "high"
        elif conf >= 65:
            lbl = "medium"
        else:
            lbl = "low"

        temp_f = float(np.median(temps)) if temps else None
        press_f = float(np.median(presss)) if presss else None

        return WindResult(
            timestamp=ts or time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            wind_speed=w_speed_f,
            wind_speed_knots=w_speed_f * KT_PER_MS,
            wind_direction=w_dir_f,
            ground_speed=gs_f,
            airspeed=tas_f,
            confidence=conf,
            label=lbl,
            sample_count=len(arr),
            std_dev_speed=w_speed_std,
            std_dev_direction=w_dir_std,
            outliers_rejected=outliers_total,
            flight_state=flight_state,
            temperature_c=temp_f,
            pressure_hpa=press_f,
            method_weights=method_weights
        )

    def _confidence(self, n: int, s_std: float, d_std: float, outliers: int, w_speed: float, tas: float, gs: float, flight_state: str) -> float:
        """Calculate confidence based on sample quality and flight conditions."""
        if n >= 40:
            base = 98.0
        elif n >= 25:
            base = 90.0
        elif n >= 15:
            base = 80.0
        elif n >= 8:
            base = 65.0
        else:
            base = 40.0
        base_adjustment = 0.0 if flight_state == "flying_high_confidence" else -15.0
        speed_pen = min(s_std * 15.0, 25.0)
        dir_pen = min(d_std * 0.8, 20.0)
        out_pen = min(outliers * 5.0, 25.0)
        low_w_pen = (2.0 - w_speed) * 10.0 if w_speed < 2.0 else 0.0
        low_motion_pen = max(0, (2.0 - (tas + gs)) * 8.0)
        conf = base + base_adjustment - speed_pen - dir_pen - out_pen - low_w_pen - low_motion_pen
        return max(0.0, min(100.0, conf))

# -----------------------------------------------------------------------------
# Pretty print with method weights
# -----------------------------------------------------------------------------
def pretty_print(res: WindResult, reject_counts: Optional[Dict[str, int]] = None):
    try:
        WIND_SHARED_PATH = "wind_latest.json"
        with open(WIND_SHARED_PATH, "w") as wf:
            json.dump({
                "timestamp": res.timestamp,
                "wind_speed_knots": round(res.wind_speed_knots, 2),
                "wind_direction_degrees": round(res.wind_direction, 2),
                "flight_state": res.flight_state,
                "confidence": round(res.confidence, 1),
                "method_weights": res.method_weights
            }, wf)
    except Exception as e:
        logging.warning(f"Failed to write wind_latest.json: {e}")

    current_time = time.strftime("%H:%M:%S")
    speed_acc = ("EXCELLENT" if res.std_dev_speed < 0.05 else
                 "VERY GOOD" if res.std_dev_speed < 0.10 else
                 "GOOD" if res.std_dev_speed < 0.20 else
                 "FAIR")
    dir_acc = ("EXCELLENT" if res.std_dev_direction < 1.0 else
               "VERY GOOD" if res.std_dev_direction < 2.0 else
               "GOOD" if res.std_dev_direction < 4.0 else
               "FAIR")
    print(f"🌬️ PRECISION WIND MEASUREMENT — {current_time}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"✅ Flight State: {res.flight_state.upper()}")
    print(f"💨 Wind Speed : {res.wind_speed_knots:.2f} knots ({res.wind_speed*3.6:.1f} km/h) ±{res.std_dev_speed:.3f} m/s [{speed_acc}]")
    print(f"🧭 Direction : {res.wind_direction:.1f}° ±{res.std_dev_direction:.1f}° [{dir_acc}]")
    print(f"🚀 GS : {res.ground_speed:.2f} m/s")
    print(f"✈️ TAS : {res.airspeed:.2f} m/s")
    if SHOW_TEMP_PRESS:
        if res.temperature_c is not None:
            print(f"🌡️ Temp : {res.temperature_c:.1f} °C")
        if res.pressure_hpa is not None:
            print(f"⏲️ Pressure : {res.pressure_hpa:.1f} hPa")
    print(f"📊 Confidence : {res.confidence:.1f}% ({res.label}, n={res.sample_count}, outliers={res.outliers_rejected})")
    print(f"📈 Accuracy : Speed ±{res.std_dev_speed*3.6:.2f} km/h | Dir ±{res.std_dev_direction:.1f}°")
    if res.method_weights:
        print(f"⚖️ Method Weights: {', '.join(f'{k}: {v:.2f}' for k, v in res.method_weights.items())}")
    print("")
    print(f"[{res.timestamp}] Wind: {res.wind_speed:5.2f} m/s ({res.wind_speed_knots:5.1f} kt) "
          f"from {res.wind_direction:06.1f}° | GS {res.ground_speed:5.2f} m/s | TAS {res.airspeed:5.2f} m/s | "
          f"Conf {res.confidence:05.1f}% ({res.label}, n={res.sample_count})")
    print("")

# -----------------------------------------------------------------------------
# Telemetry loop
# -----------------------------------------------------------------------------
async def listen_and_calculate():
    calc = EnhancedWindCalculator()
    measurement_count = 0
    start_time = time.time()
    while True:
        try:
            reader, _ = await asyncio.open_connection(TELEMETRY_HOST, TELEMETRY_PORT)
            logging.info("Connected to enhanced telemetry system.")
            while True:
                line = await reader.readline()
                if not line:
                    logging.warning("Telemetry stream closed.")
                    break
                try:
                    telemetry = json.loads(line.decode('utf-8').strip())
                except json.JSONDecodeError:
                    continue
                res = calc.process(telemetry)
                if res is None:
                    continue
                measurement_count += 1
                pretty_print(res, calc.reject_counts if DEBUG_REJECTS else None)
                if measurement_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = measurement_count / elapsed if elapsed > 0 else 0.0
                    logging.info(f"Performance: {measurement_count} outputs in {elapsed:.1f}s ({rate:.1f} Hz)")
        except ConnectionRefusedError:
            logging.warning("Enhanced telemetry not available, retrying...")
            await asyncio.sleep(2)
        except Exception as e:
            logging.error(f"Enhanced system error: {e}")
            await asyncio.sleep(2)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        print("🚀 Starting Super-Enhanced Precision Wind Measurement System v2.1")
        print("🔧 Key Features: Multi-method wind calculation with Kalman filter and dynamic weighting")
        print(f"📋 Reliable: TAS ≥ {MIN_AIRSPEED_RELIABLE} m/s, GS ≥ {MIN_GROUND_SPEED_RELIABLE} m/s")
        print(f"📋 Minimum: TAS ≥ {MIN_AIRSPEED_ABSOLUTE} m/s OR GS ≥ {MIN_GROUND_SPEED_ABSOLUTE} m/s")
        print(f"📋 Combined motion: ≥ {MIN_COMBINED_MOTION} m/s (TAS + GS)")
        print("🎯 Methods: Velocity triangle, drift angle, acceleration, hover drift")
        print("🐞 Debug mode enabled - will show detailed flight state and method weights")
        asyncio.run(listen_and_calculate())
    except KeyboardInterrupt:
        logging.info("System shutdown requested.")