import socket
import struct
import time

# X-Plane UDP settings
XPLANE_IP = '127.0.0.1'
XPLANE_UDP_PORT = 49000

# TCP settings
TCP_HOST = '0.0.0.0'
TCP_PORT = 9100

# DataRefs to request (example: you can add more as needed)
DATAREFS = [
    "sim/flightmodel/position/latitude",
    "sim/flightmodel/position/longitude",
    "sim/flightmodel/position/elevation",
    "sim/flightmodel/position/true_airspeed",
    "sim/weather/wind_speed_kt",
    "sim/weather/wind_direction_degt"
]

def build_dataref_request(dataref):
    header = b'DREF0'
    dr_bytes = dataref.encode('utf-8') + b'\x00' * (400 - len(dataref))
    freq = struct.pack('<i', 1)
    return header + freq + dr_bytes

def main():
    tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    tcp_sock.bind((TCP_HOST, TCP_PORT))
    tcp_sock.listen(1)
    print(f"Waiting for TCP client on {TCP_HOST}:{TCP_PORT}...")
    conn, addr = tcp_sock.accept()
    print(f"TCP client connected: {addr}")

    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.settimeout(0.5)

    for dr in DATAREFS:
        udp_sock.sendto(build_dataref_request(dr), (XPLANE_IP, XPLANE_UDP_PORT))

    while True:
        try:
            data, _ = udp_sock.recvfrom(4096)
            conn.sendall(data)
        except socket.timeout:
            continue
        except Exception as e:
            print(f"Error: {e}")
            break

    conn.close()
    udp_sock.close()

if __name__ == "__main__":
    main()
