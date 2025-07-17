import streamlit as st
import socket
import json
import threading
import time
import os
import psutil

TCP_HOST = '127.0.0.1'
TCP_PORT = 5761

st.set_page_config(page_title="Live Wind & Weather Dashboard", layout="wide")
st.title("Live Wind & Weather Dashboard")

PROCESS_SCRIPTS = ["MAVSDK_logger.py", "wind_calculator.py", "tcp_client.py"]

status_placeholder = st.empty()
wind_placeholder = st.empty()
pred_placeholder = st.empty()

wind_data = {}
process_status = {name: False for name in PROCESS_SCRIPTS}

# --- Process Monitoring ---
def check_processes():
    while True:
        for name in PROCESS_SCRIPTS:
            process_status[name] = any(
                any(name in (str(arg) if arg else "") for arg in (p.info.get('cmdline') or []))
                for p in psutil.process_iter(['cmdline'])
            )
        time.sleep(2)

threading.Thread(target=check_processes, daemon=True).start()

# --- TCP Wind Data Listener ---
def listen_wind_data():
    global wind_data
    while True:
        try:
            with socket.create_connection((TCP_HOST, TCP_PORT), timeout=5) as sock:
                buffer = b''
                while True:
                    data = sock.recv(4096)
                    if not data:
                        break
                    buffer += data
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        try:
                            wind_data = json.loads(line.decode('utf-8'))
                        except Exception:
                            continue
        except Exception:
            time.sleep(2)

threading.Thread(target=listen_wind_data, daemon=True).start()

# --- Main Dashboard Loop ---
while True:
    # Process Status
    with status_placeholder.container():
        st.subheader("Process Status")
        cols = st.columns(len(PROCESS_SCRIPTS))
        for i, name in enumerate(PROCESS_SCRIPTS):
            status = process_status[name]
            cols[i].metric(label=name, value="Running" if status else "Stopped", delta=None)
    # Wind Data
    with wind_placeholder.container():
        st.subheader("Live Wind Estimate")
        if wind_data:
            st.metric("Wind Speed (m/s)", f"{wind_data.get('wind_speed', 0.0):.2f}")
            st.metric("Wind Speed (knots)", f"{wind_data.get('wind_speed', 0.0) * 1.944:.2f}")
            st.metric("Direction (°)", f"{wind_data.get('wind_direction', 0.0):.2f}")
            st.metric("Altitude (m)", f"{wind_data.get('altitude', 0.0):.2f}")
            st.metric("Confidence", f"{wind_data.get('confidence', 0.0):.2f}")
            st.write(f"Timestamp: {wind_data.get('datetime', wind_data.get('timestamp', ''))}")
        else:
            st.warning("No wind data received yet. Check that all pipeline components are running and connected.")
    # Weather/Wind Predictions (placeholder)
    with pred_placeholder.container():
        st.subheader("Weather & Wind Predictions")
        st.info("Prediction features coming soon!")
    time.sleep(2)
