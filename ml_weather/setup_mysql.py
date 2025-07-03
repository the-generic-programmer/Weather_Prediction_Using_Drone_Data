
#!/usr/bin/env python3

import subprocess
import json
import logging
import mysql.connector
from mysql.connector import Error
from pathlib import Path
import sys
import os
import getpass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('setup_mysql.log'),
        logging.StreamHandler()
    ]
)

CONFIG_FILE = Path("config.json")
MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Root@123',  # Default password if none exists
    'database': 'weather_predict'  # Will be updated per database
}

def run_command(command, error_message, input_text=None):
    """Run a shell command and handle errors."""
    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True,
            input=input_text
        )
        logging.info(f"Command succeeded: {command}")
        return result
    except subprocess.CalledProcessError as e:
        logging.error(f"{error_message}: {e.stderr}")
        raise

def check_mysql_installed():
    """Check if MySQL is installed."""
    try:
        result = subprocess.run("mysql --version", shell=True, capture_output=True, text=True)
        logging.info("MySQL is already installed")
        return True
    except subprocess.CalledProcessError:
        logging.info("MySQL not found")
        return False

def install_mysql():
    """Install MySQL server and client on Ubuntu/Debian."""
    logging.info("Installing MySQL...")
    try:
        run_command("sudo apt-get update", "Failed to update package list")
        run_command(
            "sudo DEBIAN_FRONTEND=noninteractive apt-get install -y mysql-server mysql-client",
            "Failed to install MySQL"
        )
        run_command("sudo systemctl start mysql", "Failed to start MySQL service")
        run_command("sudo systemctl enable mysql", "Failed to enable MySQL service")
        logging.info("MySQL installed and started")
    except Exception as e:
        logging.error(f"Failed to install MySQL: {e}")
        raise

def test_mysql_root_access(password=None):
    """Test if root access is possible; return password if successful."""
    try:
        conn = mysql.connector.connect(
            host=MYSQL_CONFIG['host'],
            user=MYSQL_CONFIG['user'],
            password=password or ''
        )
        conn.close()
        logging.info("MySQL root access successful")
        return password or ''
    except Error as e:
        logging.debug(f"MySQL root access failed: {e}")
        return None

def configure_mysql():
    """Configure MySQL root password."""
    try:
        # Try accessing with no password
        existing_password = test_mysql_root_access()
        if existing_password is not None:
            logging.info("MySQL root user has no password or empty password")
            # Set default password
            run_command(
                f"sudo mysql -e \"ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '{MYSQL_CONFIG['password']}'; FLUSH PRIVILEGES;\"",
                "Failed to set MySQL root password"
            )
            logging.info(f"Set MySQL root password to {MYSQL_CONFIG['password']}")
            return MYSQL_CONFIG['password']
        else:
            # Prompt user for existing password
            password = getpass.getpass("Enter existing MySQL root password: ")
            if test_mysql_root_access(password):
                logging.info("Provided root password is correct")
                return password
            else:
                logging.error("Incorrect root password provided")
                raise ValueError("Invalid MySQL root password")
    except Exception as e:
        logging.error(f"Failed to configure MySQL root user: {e}")
        raise

def create_databases(password):
    """Create databases and tables for MAVSDK_logger and predict."""
    try:
        temp_config = {**MYSQL_CONFIG, 'password': password}
        temp_config.pop('database', None)
        conn = mysql.connector.connect(**temp_config)
        cursor = conn.cursor()

        # Create weather_mavsdk_logger database
        cursor.execute("CREATE DATABASE IF NOT EXISTS weather_mavsdk_logger")
        logging.info("Database weather_mavsdk_logger created or exists")

        # Create weather_predict database
        cursor.execute("CREATE DATABASE IF NOT EXISTS weather_predict")
        logging.info("Database weather_predict created or exists")

        conn.commit()
        cursor.close()
        conn.close()

        # Create weather_logs table
        conn = mysql.connector.connect(**{**temp_config, 'database': 'weather_mavsdk_logger'})
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS weather_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp TEXT,
                latitude TEXT,
                longitude TEXT,
                altitude_from_sealevel TEXT,
                relative_altitude TEXT,
                voltage TEXT,
                remaining_percent TEXT,
                north_m_s TEXT,
                east_m_s TEXT,
                down_m_s TEXT,
                temperature_degc TEXT,
                roll_deg TEXT,
                pitch_deg TEXT,
                yaw_deg TEXT
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        logging.info("Table weather_logs created or exists")

        # Create flight table (example)
        from datetime import datetime
        table_name = f"flight_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        fields = [
            "timestamp", "source", "sequence", "latitude", "longitude", "altitude_from_sealevel",
            "relative_altitude", "voltage", "remaining_percent",
            "angular_velocity_forward_rad_s", "angular_velocity_right_rad_s", "angular_velocity_down_rad_s",
            "linear_acceleration_forward_m_s2", "linear_acceleration_right_m_s2", "linear_acceleration_down_m_s2",
            "magnetic_field_forward_gauss", "magnetic_field_right_gauss", "magnetic_field_down_gauss",
            "temperature_degc", "roll_deg", "pitch_deg", "yaw_deg", "timestamp_us",
            "north_m_s", "east_m_s", "down_m_s", "system_time", "unix_time"
        ]
        columns = ', '.join([f'`{field}` TEXT' for field in fields])
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS `{table_name}` (
                id INT AUTO_INCREMENT PRIMARY KEY,
                {columns}
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        logging.info(f"Table {table_name} created or exists")
        conn.commit()
        cursor.close()
        conn.close()

        # Create predictions table
        conn = mysql.connector.connect(**{**temp_config, 'database': 'weather_predict'})
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME,
                temperature_2m FLOAT,
                relative_humidity_2m FLOAT,
                windspeed_10m FLOAT,
                winddirection_10m FLOAT,
                confidence_range FLOAT,
                precipitation FLOAT,
                weathercode INT,
                rain_chance_2h FLOAT,
                cloudcover FLOAT,
                humidity_timestamp TEXT,
                sunrise_drone TEXT,
                sunset_drone TEXT,
                sunrise_user TEXT,
                sunset_user TEXT,
                info TEXT,
                humidity_source TEXT,
                location TEXT,
                warning TEXT
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        conn.commit()
        cursor.close()
        conn.close()
        logging.info("Table predictions created or exists")
    except Error as e:
        logging.error(f"Failed to create databases or tables: {e}")
        raise

def save_config(password):
    """Save MySQL configuration to config.json."""
    config = {
        'mysql': {
            'host': MYSQL_CONFIG['host'],
            'user': MYSQL_CONFIG['user'],
            'password': password,
            'database': 'weather_predict'
        },
        'tcp_port': 9000,
        'expected_coords': [13.0, 77.625]
    }
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        logging.info(f"Configuration saved to {CONFIG_FILE}")
    except Exception as e:
        logging.error(f"Failed to save config.json: {e}")
        raise

def main():
    """Main function to set up MySQL and databases."""
    try:
        # Check if MySQL is installed
        if not check_mysql_installed():
            logging.info("Installing MySQL...")
            install_mysql()
        else:
            logging.info("MySQL already installed")
        
        # Configure MySQL root password
        password = configure_mysql()
        
        # Create databases and tables
        create_databases(password)
        
        # Save configuration
        save_config(password)
        
        logging.info("MySQL setup completed successfully")
        print("Setup complete. Configuration saved to config.json")
        print("You can now run MAVSDK_logger.py and predict.py")
    except Exception as e:
        logging.error(f"Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)
