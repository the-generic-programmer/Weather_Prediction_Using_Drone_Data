import socket
import sys
import json
import logging

TCP_HOST = '127.0.0.1'  # Change if connecting to a remote server
TCP_PORT = 9000         # Must match the logger's TCP_PORT

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def main():
    logging.info(f"Connecting to TCP server at {TCP_HOST}:{TCP_PORT}...")
    try:
        with socket.create_connection((TCP_HOST, TCP_PORT)) as sock:
            logging.info("Connected! Waiting for telemetry data...")
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
                        msg = json.loads(line.decode('utf-8'))
                        print(json.dumps(msg, indent=2))
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
