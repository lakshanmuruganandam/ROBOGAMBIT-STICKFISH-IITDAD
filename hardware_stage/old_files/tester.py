import subprocess
import requests
import serial
import time
import os
import sys

# Import the actual configuration being used by the main game
try:
    import config_l1
    print("✅ Found config_l1.py successfully.")
except ImportError:
    print("❌ ERROR: config_l1.py not found in this folder!")
    sys.exit(1)

def run_test(name, success_msg, fail_msg, func):
    print(f"\n[ TESTING: {name} ]")
    try:
        result = func()
        if result:
            print(f"✅ SUCCESS: {success_msg}")
            return True
        else:
            print(f"❌ FAILED: {fail_msg}")
    except Exception as e:
        print(f"❌ ERROR: {e}")
    return False

def test_ping():
    # Tests the IP address defined in config_l1
    target_ip = config_l1.ARM_IP
    print(f"   Target IP: {target_ip}")
    res = subprocess.run(['ping', '-c', '2', target_ip], capture_output=True)
    return res.returncode == 0

def test_http():
    # Tests the Base URL and Port from config_l1
    # Command T:105 asks the arm for its current firmware/status
    url = f"{config_l1.ARM_BASE_URL}/js?json={{'T':105}}"
    print(f"   Target URL: {url}")
    resp = requests.get(url, timeout=5)
    return resp.status_code == 200

def test_serial():
    # Tests the Serial Port and Baud Rate from config_l1
    port = config_l1.SERIAL_PORT
    baud = config_l1.BAUD_RATE
    print(f"   Target Port: {port} at {baud} baud")
    
    if not os.path.exists(port):
        print(f"   (Device {port} not found on this Ubuntu system)")
        return False
    
    try:
        with serial.Serial(port, baud, timeout=1) as ser:
            return ser.is_open
    except serial.SerialException as e:
        print(f"   (Serial Error: {e})")
        return False

def test_physical_move():
    # Moves the arm using the coordinate logic from config_l1
    # We move to (0, 180, 180) which is a safe 'high' position
    print("⚠️  CAUTION: Command will attempt to move the arm.")
    confirm = input("   Send move command to verify motor power? (y/n): ")
    if confirm.lower() == 'y':
        url = f"{config_l1.ARM_BASE_URL}/js?json={{'T':104,'X':0,'Y':180,'Z':180,'S':{config_l1.SPEED_TRANSIT}}}"
        resp = requests.get(url, timeout=5)
        return resp.status_code == 200
    return False

print("="*50)
print("  ROBOGAMBIT CONFIGURATION DIAGNOSTIC")
print("  Checking config_l1.py against hardware...")
print("="*50)

# 1. Test Network
net_ok = run_test("NETWORK PING", 
                 "Laptop can reach the Arm's IP.", 
                 "IP unreachable. Check if you are on the RoArm-M2 WiFi.", 
                 test_ping)

# 2. Test HTTP API
api_ok = run_test("HTTP API (WIFI)", 
                 "Arm web server is responding.", 
                 "WiFi is connected but Arm timed out. Check 12V power or Arm screen for correct IP.", 
                 test_http)

# 3. Test Serial
ser_ok = run_test("SERIAL PORT (USB)", 
                 "USB/Serial port is accessible.", 
                 "Cannot open Serial. Run 'ls /dev/tty*' to find the right port and check permissions.", 
                 test_serial)

# 4. Final Physical Test
if net_ok and api_ok:
    run_test("PHYSICAL REACH", 
             "Command sent successfully!", 
             "Command accepted but no movement. Check 12V Power Plug.", 
             test_physical_move)

print("\n" + "="*50)
print("DIAGNOSTIC COMPLETE")
if not (net_ok and api_ok):
    print("👉 RECOMMENDATION: Fix WiFi/IP issues before running main_l2.py")
if not ser_ok:
    print("👉 RECOMMENDATION: If using USB for Gripper, fix SERIAL_PORT in config_l1.py")