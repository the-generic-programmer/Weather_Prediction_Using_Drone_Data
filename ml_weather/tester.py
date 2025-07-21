#!/usr/bin/env python3
"""
Telemetry Connection and Data Flow Tester
=========================================
This script will help diagnose the exact issue with your wind calculator
"""

import asyncio
import json
import socket
import time
import logging

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s')

TELEMETRY_HOST = "127.0.0.1"
TELEMETRY_PORT = 9000

class TelemetryTester:
    def __init__(self):
        self.lines_received = 0
        self.valid_json_count = 0
        self.data_samples = []
        
    async def test_connection(self):
        """Test if we can connect to the telemetry port"""
        print("🔌 TESTING TELEMETRY CONNECTION")
        print("=" * 50)
        
        try:
            # Test basic socket connection first
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((TELEMETRY_HOST, TELEMETRY_PORT))
            sock.close()
            
            if result == 0:
                print(f"✅ Socket connection to {TELEMETRY_HOST}:{TELEMETRY_PORT} is successful")
            else:
                print(f"❌ Cannot connect to {TELEMETRY_HOST}:{TELEMETRY_PORT} (Error code: {result})")
                return False
                
        except Exception as e:
            print(f"❌ Socket test failed: {e}")
            return False
            
        # Test asyncio connection
        try:
            print("🔄 Testing asyncio connection...")
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(TELEMETRY_HOST, TELEMETRY_PORT), 
                timeout=5.0
            )
            print("✅ Asyncio connection successful")
            writer.close()
            await writer.wait_closed()
            return True
            
        except asyncio.TimeoutError:
            print(f"❌ Connection timeout to {TELEMETRY_HOST}:{TELEMETRY_PORT}")
            return False
        except Exception as e:
            print(f"❌ Asyncio connection failed: {e}")
            return False
    
    async def monitor_data_stream(self, duration=30):
        """Monitor the data stream for a specific duration"""
        print(f"\n📡 MONITORING DATA STREAM FOR {duration} SECONDS")
        print("=" * 50)
        
        try:
            reader, writer = await asyncio.open_connection(TELEMETRY_HOST, TELEMETRY_PORT)
            print("✅ Connected to telemetry stream")
            
            start_time = time.time()
            
            while time.time() - start_time < duration:
                try:
                    # Set a timeout for readline
                    line = await asyncio.wait_for(reader.readline(), timeout=2.0)
                    
                    if not line:
                        print("⚠️  Connection closed by remote")
                        break
                        
                    self.lines_received += 1
                    line_str = line.decode('utf-8').strip()
                    
                    if not line_str:
                        continue
                        
                    # Try to parse JSON
                    try:
                        data = json.loads(line_str)
                        self.valid_json_count += 1
                        
                        # Store first few samples for analysis
                        if len(self.data_samples) < 5:
                            self.data_samples.append(data)
                        
                        # Log every 10th line
                        if self.lines_received % 10 == 0:
                            print(f"📊 Received {self.lines_received} lines, {self.valid_json_count} valid JSON")
                            
                    except json.JSONDecodeError:
                        if self.lines_received <= 5:  # Show first few decode errors
                            print(f"❌ JSON decode error on line {self.lines_received}: {line_str[:100]}...")
                            
                except asyncio.TimeoutError:
                    print("⏰ No data received in last 2 seconds...")
                    continue
                except Exception as e:
                    print(f"❌ Error reading line: {e}")
                    break
            
            writer.close()
            await writer.wait_closed()
            
        except Exception as e:
            print(f"❌ Error monitoring stream: {e}")
    
    def analyze_data_samples(self):
        """Analyze the collected data samples"""
        print(f"\n🔍 DATA ANALYSIS")
        print("=" * 50)
        print(f"Total lines received: {self.lines_received}")
        print(f"Valid JSON messages: {self.valid_json_count}")
        
        if not self.data_samples:
            print("❌ No valid data samples collected!")
            return
            
        print(f"✅ Collected {len(self.data_samples)} sample(s) for analysis")
        
        for i, sample in enumerate(self.data_samples):
            print(f"\n📋 SAMPLE {i+1}:")
            print(f"   Keys: {list(sample.keys())}")
            
            # Check for required wind calculation fields
            required_fields = ["north_m_s", "east_m_s", "airspeed_m_s", "yaw_deg"]
            missing_fields = []
            present_fields = []
            
            for field in required_fields:
                if field in sample:
                    present_fields.append(field)
                    try:
                        value = float(sample[field])
                        print(f"   ✅ {field}: {value}")
                    except (ValueError, TypeError):
                        print(f"   ❌ {field}: {sample[field]} (cannot convert to float)")
                else:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"   ❌ Missing required fields: {missing_fields}")
            
            # Check field value ranges
            if "airspeed_m_s" in sample:
                try:
                    airspeed = float(sample["airspeed_m_s"])
                    if airspeed < 2.0:
                        print(f"   ⚠️  Airspeed {airspeed:.2f} m/s is below minimum threshold of 2.0 m/s")
                    if airspeed > 200:
                        print(f"   ⚠️  Airspeed {airspeed:.2f} m/s is above maximum threshold of 200 m/s")
                except:
                    pass
            
            if "yaw_deg" in sample:
                try:
                    yaw = float(sample["yaw_deg"])
                    if not (0 <= yaw <= 360):
                        print(f"   ⚠️  Yaw {yaw:.1f}° is outside valid range 0-360°")
                except:
                    pass
            
            # Look for alternative field names
            possible_alternatives = {
                "north_m_s": ["velocity_north", "vn", "vel_north", "north_velocity"],
                "east_m_s": ["velocity_east", "ve", "vel_east", "east_velocity"],
                "airspeed_m_s": ["airspeed", "ias", "tas", "air_speed"],
                "yaw_deg": ["heading", "yaw", "heading_deg", "course"]
            }
            
            for req_field, alternatives in possible_alternatives.items():
                if req_field not in sample:
                    found_alternatives = [alt for alt in alternatives if alt in sample]
                    if found_alternatives:
                        print(f"   💡 Found possible alternative(s) for {req_field}: {found_alternatives}")
    
    def print_recommendations(self):
        """Print recommendations based on findings"""
        print(f"\n💡 RECOMMENDATIONS")
        print("=" * 50)
        
        if self.lines_received == 0:
            print("❌ CRITICAL: No data received at all!")
            print("   1. Check if your telemetry sender is running")
            print("   2. Verify the correct port (9000) is being used")
            print("   3. Check firewall settings")
            print("   4. Confirm telemetry is sending to 127.0.0.1:9000")
            
        elif self.valid_json_count == 0:
            print("❌ CRITICAL: No valid JSON received!")
            print("   1. Check the format of data being sent")
            print("   2. Ensure data is sent as one JSON object per line")
            print("   3. Verify JSON is properly formatted")
            
        elif len(self.data_samples) > 0:
            sample = self.data_samples[0]
            required_fields = ["north_m_s", "east_m_s", "airspeed_m_s", "yaw_deg"]
            missing_fields = [f for f in required_fields if f not in sample]
            
            if missing_fields:
                print(f"❌ CRITICAL: Missing required fields: {missing_fields}")
                print("   1. Update your telemetry sender to include these exact field names")
                print("   2. Or modify the wind calculator to use your field names")
                
            # Check airspeed threshold
            if "airspeed_m_s" in sample:
                try:
                    airspeed = float(sample["airspeed_m_s"])
                    if airspeed < 2.0:
                        print(f"⚠️  Airspeed {airspeed:.2f} m/s is too low")
                        print("   1. Lower MIN_AIRSPEED in wind calculator")
                        print("   2. Or increase drone airspeed above 2.0 m/s")
                except:
                    pass
        
        print("\n🔧 QUICK FIXES TO TRY:")
        print("1. Lower sample requirement: Change 'if len(self.raw_measurements) >= 8:' to '>= 1'")
        print("2. Lower airspeed threshold: Change 'MIN_AIRSPEED = 2.0' to '0.5'")
        print("3. Add debug prints in wind calculator validation function")

async def main():
    tester = TelemetryTester()
    
    # Test connection
    if not await tester.test_connection():
        print("\n❌ Cannot establish connection. Check if telemetry sender is running!")
        return
    
    # Monitor data stream
    await tester.monitor_data_stream(duration=15)  # Monitor for 15 seconds
    
    # Analyze results
    tester.analyze_data_samples()
    tester.print_recommendations()

if __name__ == "__main__":
    print("🚀 TELEMETRY DIAGNOSTIC TOOL")
    print("This will test your telemetry connection and analyze the data format\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Diagnostic interrupted by user")
    except Exception as e:
        print(f"\n💥 Diagnostic error: {e}")
        logging.error("Diagnostic failed", exc_info=True)