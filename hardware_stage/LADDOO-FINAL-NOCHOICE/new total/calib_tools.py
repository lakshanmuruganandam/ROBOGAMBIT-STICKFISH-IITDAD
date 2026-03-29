import time
import json
import math
import serial
import numpy as np

# Import YOUR existing logic
import main
import perception

# --- PHYSICAL OVERRIDES (Adjusted for Serial) ---

# Use the GameMaster's Serial Setup
#ser_arm = serial.Serial("COM10", baudrate=115200, timeout=1) # The Arm
# ser_mag = serial.Serial("COM9", baudrate=115200, timeout=1) # The Magnet
ser_arm = main.ser
ser_mag = main.ser2

SAMPLE_BOARD = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0], 
        [0, 0, 0, 8, 8, 0], 
        [0, 0, 0, 0, 8, 0], 
        [0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0]
    ]
)

# --- THE TEST MENU ---

def run_magnet_test():
    print("Testing Magnet on COM3...")
    ser_mag.write(b'1')
    time.sleep(5)
    ser_mag.write(b'0')
    print("Magnet Cycle Complete.")

def run_calibration_test():
    print("\n--- SIDE-PLACEMENT CALIBRATION WITH JOGGING ---")
    print("Point of View: Look at the CAMERA screen.")
    
    points = [
        ("TOP-LEFT (R0,C0)", "TL"),
        ("TOP-RIGHT (R0,C5)", "TR"),
        ("BOTTOM-LEFT (R5,C0)", "BL"),
        ("BOTTOM-RIGHT (R5,C5)", "BR")
    ]
    
    results = {}
    for label, key in points:
        input(f"Move arm to {label} center, touch the board, then press Enter...")
        x, y, z = main.get_feedback_full()
        if x is not None:
            results[key] = (x, y, z)
            print(f"\nCaptured {key}: X={x}, Y={y}, Z={z}")
        else:
            print("\nError: Could not read feedback!")
    
    print("\n--- FINAL CALIBRATION DATA ---")
    print("Paste these into your main_2.py constants:")
    for k, v in results.items():
        print(f"CORNER_{k} = ({v[0]:.2f}, {v[1]:.2f})  # Z={v[2]:.2f}")

# --- KEEPING YOUR OTHER TESTS ---

def run_magnet_test():
    print("Testing Magnet on COM7...")
    ser_mag.write(b'1'); time.sleep(2); ser_mag.write(b'0')
    print("Magnet Cycle Complete.")

def run_perception_test():
    sock = perception.init_perception()
    board, _ = perception.get_stable_board(sock, stability_required=5)
    if board is not None: print(board)
    sock.close()
def run_sample_board_test():
    print("Testing Full Logic on Sample Board...")
    # This uses YOUR functions from main_2.py and perception.py
    # It simulates the entire process on a known board state
    main.execute_turn("1:D2->E3", SAMPLE_BOARD, {1: [(200, 150)]})  # Simulated piece at (250,150)
def run_movement_test():
    print("Testing your linear_move_to logic...")
    # This uses YOUR function from main_2.py
    # Moving to a safe center point
    target_x, target_y = 200, 200
    rx, ry = main.transform_to_robot(target_x, target_y)
    print(f"Moving to {rx}, {ry} at height {main.Z_SAFE}")
    main.linear_move_to(rx, ry, main.Z_SAFE)

if __name__ == "__main__":
    while True:
        print("\n[1] Test Magnet (COM3)")
        print("[2] Calibrate 4 Corners (Capture Robot X,Y,Z)")
        print("[3] Test Perception (Socket)")
        print("[4] Test Your linear_move_to (COM4)")
        print("[5] Execute for Sample Board")
        print("[6] Exit")
        
        choice = input("Select Test: ")
        if choice == '1': run_magnet_test()
        elif choice == '2': run_calibration_test()
        elif choice == '3': run_perception_test()
        elif choice == '4': run_movement_test()
        elif choice == '5': run_sample_board_test() 
        elif choice == '6': break