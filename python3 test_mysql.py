import mysql.connector
from mysql.connector import Error
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

try:
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='Root@123',
        database='weather_predict'
    )
    logging.info("Connection successful")
    conn.close()
except Error as e:
    logging.error(f"Connection failed: {e}")
