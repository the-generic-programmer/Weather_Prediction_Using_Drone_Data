import subprocess
import time
import logging
import sys
import os

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    ("MAVSDK Logger", os.path.join(SCRIPT_DIR, "MAVSDK_logger.py")),
    ("Wind Calculator", os.path.join(SCRIPT_DIR, "wind_calculator.py")),
    ("TCP Client", os.path.join(SCRIPT_DIR, "tcp_client.py")),
    ("Dashboard", [sys.executable, "-m", "streamlit", "run", os.path.join(SCRIPT_DIR, "dashboard.py")]),
]

PROCESSES = []

try:
    for name, script in SCRIPTS:
        logging.info(f"Starting {name}...")
        if isinstance(script, list):
            proc = subprocess.Popen(script, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            proc = subprocess.Popen([sys.executable, script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        PROCESSES.append((name, proc))
        time.sleep(2)  # Stagger startup
    logging.info("All components started. Monitoring processes...")
    while True:
        for name, proc in PROCESSES:
            retcode = proc.poll()
            if retcode is not None:
                logging.error(f"{name} exited with code {retcode}.")
                sys.exit(1)
        time.sleep(5)
except KeyboardInterrupt:
    logging.info("Pipeline interrupted by user. Shutting down...")
    for name, proc in PROCESSES:
        proc.terminate()
    logging.info("All processes terminated.")
except Exception as e:
    logging.error(f"Pipeline error: {e}")
    for name, proc in PROCESSES:
        proc.terminate()
    sys.exit(1)
