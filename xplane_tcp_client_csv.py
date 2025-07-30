import socket
import csv
import struct
import time

TCP_HOST = '127.0.0.1'
TCP_PORT = 9100
CSV_FILE = 'xplane_data.csv'

DATAREFS = [
    "latitude",
    "longitude",
    "elevation",
    "true_airspeed",
    "wind_speed_kt",
    "wind_direction_degt"
]

def parse_xplane_packet(data):
    values = []
    for i in range(len(DATAREFS)):
        offset = 5 + 4 + i * 4
        val = struct.unpack('<f', data[offset:offset+4])[0]
        values.append(val)
    return values

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((TCP_HOST, TCP_PORT))
    print(f"Connected to TCP transmitter at {TCP_HOST}:{TCP_PORT}")

    with open(CSV_FILE, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp'] + DATAREFS)
        while True:
            data = sock.recv(4096)
            if not data:
                break
            vals = parse_xplane_packet(data)
            writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S')] + vals)

    sock.close()

if __name__ == "__main__":
    main()
