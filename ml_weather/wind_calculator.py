#!/usr/bin/env python3
"""
Enhanced Precision Wind Measurement with Superior Accuracy (v1.1 hotfix)
=========================================================================

### Additional improvements
- Full debug line in the original style:
  `[2025-07-21T04:40:44Z] Wind:  7.35 m/s ( 14.3 kt) from 0353.0° | GS 32.35 m/s | TAS 38.75 m/s | Conf 100.0% (high, n=20)`

**Run tip:** If telemetry produces no lines, confirm `mavsdk_logger.py` is running and streaming JSON to `127.0.0.1:9000`.
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

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
TELEMETRY_HOST = "127.0.0.1"
TELEMETRY_PORT = 9000
MAX_HISTORY = 50              # number of recent raw samples used for stats
CHECK_INTERVAL = 1.0          # seconds between reconnect attempts
MIN_AIRSPEED = 2.0            # below this -> low confidence but still process
MAX_WIND_SPEED = 100.0        # sanity cap
OUTLIER_THRESHOLD = 1.5       # IQR multiplier
PRINT_EVERY = 1.0             # seconds; throttle pretty prints
DEBUG_REJECTS = True          # show counts of dropped samples
SHOW_TEMP_PRESS = True        # include 🌡️ & ⏲️ lines if present

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

KT_PER_MS = 1.94384

# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------
@dataclass
class WindResult:
    timestamp: str
    wind_speed: float         # m/s
    wind_speed_knots: float   # kt
    wind_direction: float     # deg FROM
    ground_speed: float       # m/s (horizontal)
    airspeed: float           # TAS m/s
    confidence: float         # %
    label: str                # high/medium/low
    sample_count: int         # raw samples in window
    std_dev_speed: float      # m/s
    std_dev_direction: float  # deg
    outliers_rejected: int    # count
    temperature_c: Optional[float] = None
    pressure_hpa: Optional[float] = None

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
    # normalize
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
# Enhanced calculator
# -----------------------------------------------------------------------------
class EnhancedWindCalculator:
    def __init__(self):
        self.raw = deque(maxlen=MAX_HISTORY)
        self.last_valid_dir = None
        self.last_valid_speed = None
        self.reject_counts = {"validate":0}
        self._last_print_t = 0.0

    def _validate(self, vn: Optional[float], ve: Optional[float], ias: Optional[float], heading: Optional[float]) -> bool:
        if vn is None or ve is None or ias is None or heading is None:
            self.reject_counts["validate"] += 1
            return False
        # only sanity: extreme values -> reject
        if abs(vn) > 200 or abs(ve) > 200 or ias > 200 or ias < 0:
            self.reject_counts["validate"] += 1
            return False
        return True

    def process(self, data: Dict) -> Optional[WindResult]:
        # Parse
        vn = _f(data.get("north_m_s"))
        ve = _f(data.get("east_m_s"))
        ias = _f(data.get("airspeed_m_s"))
        temp = _f(data.get("temperature_degc"))
        press = _f(data.get("pressure_hpa"))
        heading = _norm_heading(data.get("yaw_deg"))

        if not self._validate(vn, ve, ias, heading):
            return None

        tas = ias_to_tas(ias, temp, press)
        gs = math.hypot(vn, ve)

        # Air vector components from heading
        hr = math.radians(heading)
        va_n = tas * math.cos(hr)
        va_e = tas * math.sin(hr)

        # Wind components
        wn = vn - va_n
        we = ve - va_e
        w_speed = math.hypot(wn, we)
        w_dir = calc_wind_dir_from(wn, we)

        # Sanity check
        if w_speed > MAX_WIND_SPEED:
            return None

        # Append raw sample
        self.raw.append({
            "wn": wn,
            "we": we,
            "w_speed": w_speed,
            "w_dir": w_dir,
            "gs": gs,
            "tas": tas,
            "t": time.time(),
            "temp": temp,
            "press": press,
        })

        # Throttle prints
        now = time.time()
        if now - self._last_print_t < PRINT_EVERY:
            return None
        self._last_print_t = now

        return self._filtered_result(data.get("timestamp"))

    # ------------------------------------------------------------------
    def _filtered_result(self, ts: Optional[str]) -> WindResult:
        arr = list(self.raw)
        speeds = [s["w_speed"] for s in arr]
        dirs = [s["w_dir"] for s in arr]
        gses = [s["gs"] for s in arr]
        tases = [s["tas"] for s in arr]
        temps = [s["temp"] for s in arr if s["temp"] is not None]
        presss = [s["press"] for s in arr if s["press"] is not None]

        # Outlier removal on *speeds* only (direction handled circularly)
        clean_speeds, out_speed = remove_outliers_iqr(speeds)
        clean_gs, out_gs = remove_outliers_iqr(gses)
        clean_tas, out_tas = remove_outliers_iqr(tases)

        # For direction, don't drop; use circular mean/robustness
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

        # Confidence
        conf = self._confidence(len(arr), w_speed_std, w_dir_std, outliers_total, w_speed_f, tas_f)
        if conf >= 85: lbl = "high"
        elif conf >= 65: lbl = "medium"
        else: lbl = "low"

        # Temperature / pressure summary (median)
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
            temperature_c=temp_f,
            pressure_hpa=press_f,
        )

    # ------------------------------------------------------------------
    def _confidence(self, n: int, s_std: float, d_std: float, outliers: int, w_speed: float, tas: float) -> float:
        if n >= 40: base = 98.0
        elif n >= 25: base = 90.0
        elif n >= 15: base = 80.0
        elif n >= 8: base = 65.0
        else: base = 40.0
        speed_pen = min(s_std * 15.0, 25.0)
        dir_pen = min(d_std * 0.8, 20.0)
        out_pen = min(outliers * 5.0, 25.0)
        low_w_pen = (2.0 - w_speed) * 10.0 if w_speed < 2.0 else 0.0
        low_tas_pen = (10.0 - tas) * 2.0 if tas < 10.0 else 0.0
        conf = base - speed_pen - dir_pen - out_pen - low_w_pen - low_tas_pen
        return max(0.0, min(100.0, conf))

# -----------------------------------------------------------------------------
# Pretty print
# -----------------------------------------------------------------------------

def pretty_print(res: WindResult, reject_counts: Optional[Dict[str,int]] = None):
    current_time = time.strftime("%H:%M:%S")

    speed_acc = ("EXCELLENT" if res.std_dev_speed < 0.05 else
                 "VERY GOOD" if res.std_dev_speed < 0.10 else
                 "GOOD" if res.std_dev_speed < 0.20 else
                 "FAIR")
    dir_acc = ("EXCELLENT" if res.std_dev_direction < 1.0 else
               "VERY GOOD" if res.std_dev_direction < 2.0 else
               "GOOD" if res.std_dev_direction < 4.0 else
               "FAIR")

    print(f"🌬️  PRECISION WIND MEASUREMENT — {current_time}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"💨 Wind Speed : {res.wind_speed_knots:.2f} knots ({res.wind_speed*3.6:.1f} km/h) ±{res.std_dev_speed:.3f} m/s [{speed_acc}]")
    print(f"🧭 Direction  : {res.wind_direction:.1f}° ±{res.std_dev_direction:.1f}° [{dir_acc}]")
    print(f"🚀 GS         : {res.ground_speed:.2f} m/s")
    print(f"✈️ TAS        : {res.airspeed:.2f} m/s")
    if SHOW_TEMP_PRESS:
        if res.temperature_c is not None:
            print(f"🌡️ Temp       : {res.temperature_c:.1f} °C")
        if res.pressure_hpa is not None:
            print(f"⏲️ Pressure   : {res.pressure_hpa:.1f} hPa")
    print(f"📊 Confidence : {res.confidence:.1f}% ({res.label}, n={res.sample_count}, outliers={res.outliers_rejected})")
    print(f"📈 Accuracy   : Speed ±{res.std_dev_speed*3.6:.2f} km/h | Dir ±{res.std_dev_direction:.1f}°")
    print("")
    # Original detailed debug line (compat format)
    print(f"[{res.timestamp}] Wind:  {res.wind_speed:5.2f} m/s ({res.wind_speed_knots:5.1f} kt) "
          f"from {res.wind_direction:06.1f}° | GS {res.ground_speed:5.2f} m/s | TAS {res.airspeed:5.2f} m/s | "
          f"Conf {res.confidence:05.1f}% ({res.label}, n={res.sample_count}, outliers={res.outliers_rejected})")
    if DEBUG_REJECTS and reject_counts:
        print(f"[debug] rejected so far: {reject_counts}")
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

                # perf stats
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
        print("🚀 Starting Enhanced Precision Wind Measurement System")
        print("📈 Improvements: Advanced filtering, outlier detection, dynamic confidence")
        print("🎯 If you see no output: check telemetry stream, or enable DEBUG_REJECTS.")
        asyncio.run(listen_and_calculate())
    except KeyboardInterrupt:
        logging.info("Enhanced system shutdown requested.")

