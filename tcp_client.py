import socket
import json
import logging
from typing import Any
import os

TCP_HOST = '127.0.0.1'  # Change if connecting to a remote server
TCP_PORT = 5761         # Must match the logger's TCP_PORT

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "tcp_client.log")

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)


def main() -> None:
    """Connect to the wind data TCP server and print received wind data."""
    logging.info(f"Connecting to TCP server at {TCP_HOST}:{TCP_PORT}...")
    try:
        with socket.create_connection((TCP_HOST, TCP_PORT)) as sock:
            logging.info("Connected. Waiting for wind data...")
            buffer = b''
            while True:
                data = sock.recv(4096)
                if not data:
                    logging.warning("Server closed connection.")
                    break
                buffer += data
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    try:
                        msg: Any = json.loads(line.decode('utf-8'))
                        print(f"Timestamp : {msg.get('datetime', msg.get('timestamp'))}")
                        print(f"Wind Speed: {msg.get('wind_speed', 0.0):.2f} m/s ({msg.get('wind_speed', 0.0) * 1.944:.2f} knots)")
                        print(f"Direction : {msg.get('wind_direction', 0.0):.2f}°")
                        print(f"Altitude  : {msg.get('altitude', 0.0):.2f} m")
                        print(f"Confidence: {msg.get('confidence', 0.0):.2f}")
                        print("-------------------------------------------")
                    except json.JSONDecodeError:
                        logging.error(f"Received malformed JSON: {line}")
    except ConnectionRefusedError:
        logging.error(f"Could not connect to {TCP_HOST}:{TCP_PORT}. Is the server running?")
    except KeyboardInterrupt:
        logging.info("Client stopped by user.")
    except Exception as e:
        logging.error(f"Error: {e}")


if __name__ == "__main__":
    main()
