#!/bin/bash
# Pipeline: Start MAVSDK logger and wind calculator in sequence

set -e

# Activate virtual environment if present
env_bin=".venv/bin/activate"
if [ -f "$env_bin" ]; then
    source "$env_bin"
fi

# Start MAVSDK logger in the background
echo "[PIPELINE] Starting MAVSDK_logger.py..."
python MAVSDK_logger.py &
LOGGER_PID=$!

# Wait for TCP server to be ready (port 9000)
echo "[PIPELINE] Waiting for TCP server on port 9000..."
for i in {1..20}; do
    if nc -z 127.0.0.1 9000; then
        echo "[PIPELINE] TCP server is up."
        break
    fi
    sleep 1
done

# Start wind calculator (waits for TCP server itself, but we check above for speed)
echo "[PIPELINE] Starting wind_calculator.py..."
python ml_weather/wind_calculator.py --host 127.0.0.1 --port 9000

# On exit, kill MAVSDK_logger.py
kill $LOGGER_PID
