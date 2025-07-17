#!/usr/bin/env python3

import logging
import math
import json
import socket
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List
from datetime import datetime
import os

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "wind_calculator.log")

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

@dataclass
class DroneData:
    timestamp: float
    ground_speed: float
    airspeed: float
    heading: float
    track: float
    position: Tuple[float, float]
    altitude: float

@dataclass
class WindData:
    speed: float
    direction: float
    altitude: float
    timestamp: float
    confidence: float

class WindEstimator:
    def __init__(self, buffer_size: int = 10, min_ground_speed: float = 2.0):
        self.data_buffer: List[DroneData] = []
        self.buffer_size = buffer_size
        self.min_ground_speed = min_ground_speed

    def add_data(self, data: DroneData) -> None:
        """Add new drone data to the buffer."""
        self.data_buffer.append(data)
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer.pop(0)

    def estimate_wind(self) -> Optional[WindData]:
        """Estimate wind using the drift method."""
        if len(self.data_buffer) < self.buffer_size:
            return None
        recent_data = self.data_buffer[-1]
        if recent_data.ground_speed < self.min_ground_speed:
            return None
        drift_angles = []
        for data in self.data_buffer:
            drift = (data.track - data.heading + 360) % 360
            if drift > 180:
                drift -= 360
            drift_angles.append(drift)
        avg_drift = sum(drift_angles) / len(drift_angles)
        if recent_data.airspeed > 0:
            wind_speed = math.sqrt(
                recent_data.ground_speed**2 + 
                recent_data.airspeed**2 - 
                2 * recent_data.ground_speed * recent_data.airspeed * 
                math.cos(math.radians(avg_drift))
            )
        else:
            wind_speed = recent_data.ground_speed * math.sin(math.radians(abs(avg_drift)))
        wind_direction = (recent_data.track + 180 - avg_drift) % 360
        drift_std = math.sqrt(sum((x - avg_drift)**2 for x in drift_angles) / len(drift_angles))
        confidence = max(0, min(1, 1 - (drift_std / 45)))
        return WindData(
            speed=wind_speed,
            direction=wind_direction,
            altitude=recent_data.altitude,
            timestamp=recent_data.timestamp,
            confidence=confidence
        )

class TCPServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.running = False

    def start(self) -> None:
        """Start the TCP server."""
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        self.running = True
        logging.info(f"TCP Server listening on {self.host}:{self.port}")

    def stop(self) -> None:
        """Stop the TCP server."""
        self.running = False
        self.socket.close()

class WindCalculatorService:
    def __init__(self, input_host: str = "localhost", input_port: int = 5760,
                 output_host: str = "localhost", output_port: int = 5761):
        self.wind_estimator = WindEstimator()
        self.input_server = TCPServer(input_host, input_port)
        self.output_server = TCPServer(output_host, output_port)
        self.clients: List[socket.socket] = []
        self.running = False
        self.input_connections_started = False

    def start(self) -> None:
        """Start the wind calculator service."""
        self.running = True
        self.input_server.start()
        self.output_server.start()
        threading.Thread(target=self._handle_input_connections, daemon=True).start()
        threading.Thread(target=self._handle_output_connections, daemon=True).start()
        logging.info("WindCalculatorService: Waiting for input connections on port %d...", self.input_server.port)

    def _handle_input_connections(self) -> None:
        self.input_connections_started = True
        while self.running:
            try:
                client_socket, address = self.input_server.socket.accept()
                logging.info(f"New input connection from {address}")
                threading.Thread(target=self._handle_input_client, args=(client_socket,), daemon=True).start()
            except Exception as e:
                if self.running:
                    logging.error(f"Input connection error: {e}")
        logging.info("Input connection handler stopped.")

    def _handle_output_connections(self) -> None:
        while self.running:
            try:
                client_socket, address = self.output_server.socket.accept()
                logging.info(f"New output connection from {address}")
                self.clients.append(client_socket)
            except Exception as e:
                if self.running:
                    logging.error(f"Output connection error: {e}")

    def _handle_input_client(self, client_socket: socket.socket) -> None:
        buffer = ""
        while self.running:
            try:
                data = client_socket.recv(1024).decode('utf-8')
                if not data:
                    break
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    self._process_drone_data(line)
            except Exception as e:
                logging.error(f"Error processing input data: {e}")
                break
        client_socket.close()

    def _process_drone_data(self, data_str: str) -> None:
        try:
            data = json.loads(data_str)
            drone_data = DroneData(
                timestamp=time.time(),
                ground_speed=data.get('ground_speed', 0.0),
                airspeed=data.get('airspeed', 0.0),
                heading=data.get('heading', 0.0),
                track=data.get('track', 0.0),
                position=(data.get('latitude', 0.0), data.get('longitude', 0.0)),
                altitude=data.get('altitude', 0.0)
            )
            self.wind_estimator.add_data(drone_data)
            wind_data = self.wind_estimator.estimate_wind()
            if wind_data:
                # Log wind calculation details
                logging.info(
                    f"Wind Calculation | Timestamp: {wind_data.timestamp:.2f}, "
                    f"Speed: {wind_data.speed:.2f} m/s, Direction: {wind_data.direction:.2f}°, "
                    f"Altitude: {wind_data.altitude:.2f} m, Confidence: {wind_data.confidence:.2f}"
                )
                self._broadcast_wind_data(wind_data)
                self._display_wind_data(wind_data)
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON data received: {e}")
        except Exception as e:
            logging.error(f"Error processing drone data: {e}")

    def _broadcast_wind_data(self, wind_data: WindData) -> None:
        message = json.dumps({
            'timestamp': wind_data.timestamp,
            'datetime': datetime.fromtimestamp(wind_data.timestamp).isoformat(),
            'wind_speed': wind_data.speed,
            'wind_direction': wind_data.direction,
            'altitude': wind_data.altitude,
            'confidence': wind_data.confidence
        }) + '\n'
        disconnected_clients = []
        for client in self.clients:
            try:
                client.send(message.encode('utf-8'))
            except Exception:
                disconnected_clients.append(client)
        for client in disconnected_clients:
            self.clients.remove(client)
            client.close()

    def _display_wind_data(self, wind_data: WindData) -> None:
        print(f"WIND ESTIMATION — {datetime.now().strftime('%H:%M:%S')}")
        print(f"Speed     : {wind_data.speed:.1f} m/s ({wind_data.speed * 1.944:.1f} knots)")
        print(f"Direction : {wind_data.direction:.1f}°")
        print(f"Altitude  : {wind_data.altitude:.1f} m")
        print(f"Confidence: {wind_data.confidence:.2f}")
        print("-------------------------------------------")

    def stop(self) -> None:
        self.running = False
        self.input_server.stop()
        self.output_server.stop()
        for client in self.clients:
            client.close()
        self.clients.clear()

def main() -> None:
    service = WindCalculatorService(
        input_host="localhost",
        input_port=5760,
        output_host="localhost",
        output_port=5761
    )
    try:
        service.start()
        logging.info("Wind Calculator Service started and running.")
        while True:
            try:
                time.sleep(1)
            except Exception as e:
                logging.error(f"Main loop error: {e}")
                continue
    except KeyboardInterrupt:
        logging.info("Shutting down Wind Calculator Service...")
        service.stop()
    except Exception as e:
        logging.error(f"Service error: {e}")
        service.stop()
    finally:
        logging.info("Wind Calculator Service stopped.")

if __name__ == "__main__":
    main()
