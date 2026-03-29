import game
import numpy as np
import requests
import serial
import time
import json
import math
from collections import deque
import cv2

import sys

import perception
from typing import Tuple, Optional

# ===========================================================================
#  CONFIG
# ===========================================================================

DEBUG   = True
TESTING = False  # Set False when running on real hardware

ARM_COM_PORT = "COM11"  # Adjust for your laptop
MAGNET_COM_PORT = "COM12"  # Adjust for your laptop

# Extra fold can look like random movement; keep it off until tuned on hardware.
ENABLE_FOLD_AFTER_TURN = False

# --- Communication ---
# Serial port for the arm
ser = None
ser2 = None

if not TESTING:
    ser = serial.Serial(ARM_COM_PORT, baudrate=115200, dsrdtr=None, timeout=1)
    ser.setRTS(False)
    ser.setDTR(False)
    # Serial port for the Solenoid
    ser2 = serial.Serial(MAGNET_COM_PORT, 115200)


# --- Board state ---
BOARD = np.zeros((6, 6), dtype=int)




# ===========================================================================
#  CALIBRATION
# ===========================================================================
# --- Wrist angle ---
# CRITICAL: always keep t = π so the electromagnet stays level with the board.
# Sending t=0 tilts the end-effector and makes pick-up impossible.
EOAT_LEVEL = 0  # 3.14159...

# ===========================================================================
#  CALIBRATION
# ===========================================================================


# --- Z-Heights ---
Z_SAFE  = 100   # Cruise height — clears all pieces
# Z_HOVER = 50    # Just above a piece (optional pre-grip pause) 
Z_GRIP  = -80    # Gripper centred on piece

# --- Graveyard ---
GRAVEYARD = (400, -150)  # * robo coordinates (feedback)

# --- Tuning parameters ---
STEP_SIZE  = 7.0    # mm between ideal waypoints
STEP_DELAY = 0.08   # seconds to wait for arm to move before reading feedback
                    # this is the single biggest tuning knob — too short and
                    # feedback lags behind the command; too long and motion is jerky
KP         = 0.6    # proportional gain on position error (0.0 = open loop, 1.0 = full correction)
                    # start at 0.5 and increase until tracking is tight without oscillating
STEP_TOL   = 10  # mm — if actual error > this after a step, log a warning
                    # (doesn't stop motion — just flags mechanical slip or lag)



PLAYING_WHITE = True  # Runtime prompt in __main__ can override this.

COL_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
REV_COL_MAP = {v: k for k, v in COL_MAP.items()}

# ===========================================================================
#  COORDINATE MAPPING
# ===========================================================================

def transform_to_robot(board_x, board_y):
    """Converts Board (mm) to Robot (mm) using the H-Matrix."""
    # Format the point for cv2.perspectiveTransform
    point = np.array([[[board_x, board_y]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, perception.H_WORLD_TO_ROBOT)
    
    rx, ry = transformed[0][0]
    return rx, ry

def square_to_coords(row, col):
    """Logic for the center of any square (0-5)"""
    # 0,0 is at top-left. Our Board Space has 0,0 at center.
    # A 6x6 board with 60mm squares:
    bw_x = (2.5 - row) * perception.SQUARE_SIZE
    bw_y = (2.5 - col) * perception.SQUARE_SIZE

    return transform_to_robot(bw_x, bw_y)


def cell_to_idx(cell: str) -> Tuple[int, int]:
    col = COL_MAP[cell[0].upper()]
    row = int(cell[1]) - 1
    return row, col


def idx_to_cell(row: int, col: int) -> str:
    return f"{REV_COL_MAP[col]}{row + 1}"

def find_nearest_piece(target_id, start_row, start_col, all_poses):
    """Finds the (x, y) of the piece closest to the expected square."""
    # Get where we THINK the piece should be
    expected_x = (2.5 - start_row) * perception.SQUARE_SIZE
    expected_y = (2.5 - start_col) * perception.SQUARE_SIZE
    
    if target_id not in all_poses:
        debug_print(f"Piece {target_id} not seen by camera!")
        return expected_x, expected_y # Fallback to grid
        
    # Find the detected coordinate with the smallest distance to expected center
    best_pose = min(all_poses[target_id], 
                    key=lambda p: math.hypot(p[0]-expected_x, p[1]-expected_y))
    
    return best_pose
# ===========================================================================
#  COMMUNICATION
# ===========================================================================

def send_cmd(command: str):
    """Send command and wait for the robot's feedback."""

    debug_print(f"Sending command: {command}")
    if TESTING:
        return
        
    ser.write((command + '\n').encode())
    
def read_serial(max_attempts=50):
    for _ in range(max_attempts):
        line = ser.readline().decode('utf-8').strip()
        if not line:
            continue
        if line == '{"T": 105}':
            continue

        if '{' in line and 'x' in line:
            return line
    debug_print("read_serial: no valid response after max attempts")
    return None

def get_feedback_full():
    """
    Request T:105 via Serial and return (x, y, z, s, e).
    Uses ser.readline() to capture the JSON response from the arm.
    """
    debug_print("Requesting feedback from arm...")
    if TESTING:
        return 300.0, 0.0, 120.0, 0.0, 0.0

    try:
        # 1. Clear the input buffer to ensure we aren't reading an old message
        ser.reset_input_buffer()

        # 2. Send the request
        # We use json.dumps to ensure the format is perfect
        command = json.dumps({"T": 105}) + "\n"
        ser.write(command.encode())

        # 3. Read the response
        # ser.readline() waits until a '\n' is received or the timeout is hit
        line = read_serial()  # This will debug_print the line as well
        # debug_print(f"Serial Feedback: {line}")

        if line:
            # The arm might send info strings; we look for the JSON part
            if '{' in line and '}' in line:
                # Clean the line to ensure only JSON is parsed
                json_str = line[line.find('{'):line.rfind('}')+1]
                data = json.loads(json_str)
                
                
                return (
                    data.get('x'),
                    data.get('y'),
                    data.get('z'),
                    data.get('s', 0.0),
                    data.get('e', 0.0),
                )
    except Exception as e:
        debug_print(f"Serial Feedback Error: {e}")
        
    return None, None, None, 0.0, 0.0


def electromagnet_on():
    """Energise the electromagnet to grip a piece."""
    if TESTING:
        debug_print("Electromagnet ON (TESTING mode)")
        return
    ser2.write(b'1')
    debug_print("Electromagnet ON")


def electromagnet_off():
    """De-energise the electromagnet to release a piece."""
    if TESTING:
        debug_print("Electromagnet OFF (TESTING mode)")
        return
    ser2.write(b'0')
    debug_print("Electromagnet OFF")


# ===========================================================================
#  LOW-LEVEL MOTION PRIMITIVES
# ===========================================================================


def linear_move_to(tx: float, ty: float, tz: float,
                   step_size: float = STEP_SIZE,
                   step_delay: float = STEP_DELAY,
                   kp: float = KP,
                   settle_steps: int = 5):
    """
    True closed-loop linear interpolation with an added Home-In Settling Phase.
    """
    # --- Read actual start position ---
    if TESTING:
        debug_print("[LIN] TESTING: no feedback, simulating straight move.")
        sx, sy, sz, s0, e0 = tx, ty, tz, 0.0, 0.0
    else:
        sx, sy, sz, s0, e0 = get_feedback_full()

    if sx is None:
        debug_print("[LIN] ERROR: No feedback at start. Simulating waypoint.")
        sx, sy, sz = tx, ty, tz

    dx = tx - sx
    dy = ty - sy
    dz = tz - sz
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)

    if dist < 0.5:
        debug_print(f"[LIN] Already at target. ({tx:.1f},{ty:.1f},{tz:.1f})")
        return True

    n_steps = max(1, math.ceil(dist / step_size))
    debug_print(f"[LIN] ({sx:.1f},{sy:.1f},{sz:.1f}) → ({tx:.1f},{ty:.1f},{tz:.1f}) | {dist:.1f}mm / {n_steps} steps")

    # --- PART 1: INTERPOLATION LOOP ---
    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        # Ideal waypoint along the line
        ix = sx + alpha * dx
        iy = sy + alpha * dy
        iz = sz + alpha * dz

        # Read actual position + joint angles
        ax, ay, az, s, e = get_feedback_full()
        if ax is None:
            debug_print(f"[LIN] step {i}/{n_steps}: feedback lost, sending ideal waypoint")
            ax, ay, az, s, e = ix, iy, iz, s0, e0

        # Proportional correction toward waypoints
        cx = ix + kp * (ix - ax)
        cy = iy + kp * (iy - ay)
        cz = iz + kp * (iz - az)

        # Wrist levelling from live joint angles
        t_level = math.pi/2 - s + e  

        # Log tracking error (The part I shouldn't have removed!)
        err = math.sqrt((ix-ax)**2 + (iy-ay)**2 + (iz-az)**2)
        if err > STEP_TOL:
            debug_print(f"[LIN] step {i}/{n_steps}: WARNING large tracking error {err:.1f}mm")
        else:
            debug_print(f"[LIN] step {i}/{n_steps}: ideal=({ix:.1f},{iy:.1f},{iz:.1f}) actual=({ax:.1f},{ay:.1f},{az:.1f}) err={err:.1f}mm t={t_level:.3f}")

        cmd = f'{{"T":1041,"x":{cx:.3f},"y":{cy:.3f},"z":{cz:.3f},"t":{t_level:.5f}}}'
        send_cmd(cmd)
        time.sleep(step_delay)

    # --- PART 2: SETTLING PHASE (Home-In) ---
    debug_print(f"[LIN] Interpolation done. Settling for {settle_steps} steps...")
    for j in range(1, settle_steps + 1):
        ax, ay, az, s, e = get_feedback_full()
        if ax is None:
            break 

        # Now we pull toward the FINAL TARGET (tx, ty, tz) instead of a waypoint
        cx = tx + kp * (tx - ax)
        cy = ty + kp * (ty - ay)
        cz = tz + kp * (tz - az)

        t_level = math.pi/2 - s + e 
        
        err = math.sqrt((tx-ax)**2 + (ty-ay)**2 + (tz-az)**2)
        debug_print(f"[SETTLE] step {j}/{settle_steps}: target_err={err:.2f}mm position=({ax:.1f},{ay:.1f},{az:.1f})")

        cmd = f'{{"T":1041,"x":{cx:.3f},"y":{cy:.3f},"z":{cz:.3f},"t":{t_level:.5f}}}'
        send_cmd(cmd)

        if err < 0.3: # Stop early if we are close enough
            debug_print("[SETTLE] Precision target reached.")
            break
            
        time.sleep(step_delay)

    debug_print("[LIN] Move complete.")
    return True

# ===========================================================================
#  HIGH-LEVEL MOTION PRIMITIVES
# ===========================================================================

def go_to_init():
    """
    Reset arm then fold into a tucked pose to clear the camera's field
    of view for the opponent's turn.

    T:100 moves to the default extended pose (base=0, s=0, e=90, h=180).
    The second command folds the arm back and up using T:122 so the arm
    is compact and out of the camera's line of sight.

    Tune FOLD_S and FOLD_E on the real hardware — these are starting guesses.
    """
    FOLD_S = 0   # degrees: pull shoulder backward (away from board)
    FOLD_E = 160    # degrees: fold elbow upward (forearm points up)
    FOLD_H = 180   # degrees: keep wrist level / neutral
    FOLD_SPD = 10 # deg/s — slow enough to be safe
    FOLD_ACC = 10  # smooth start/stop

    debug_print("Returning to home position...")
    send_cmd('{"T":100}')

    time.sleep(1)  # wait for the arm to reach the home position

    if not ENABLE_FOLD_AFTER_TURN:
        debug_print("Fold skipped (ENABLE_FOLD_AFTER_TURN=False).")
        return

    debug_print("Folding arm to clear camera view...")
    fold_cmd = (
        f'{{"T":122,"b":0,"s":{FOLD_S},"e":{FOLD_E},'
        f'"h":{FOLD_H},"spd":{FOLD_SPD},"acc":{FOLD_ACC}}}'
    )
    send_cmd(fold_cmd)




# ===========================================================================
#  PIECE OPERATIONS
# ===========================================================================

def _move_to_xy_at_safe_height(x: float, y: float):
    """Move in straight segments: vertical to Z_SAFE first, then XY at Z_SAFE."""
    ax, ay, az, _s, _e = get_feedback_full()
    if ax is None or ay is None or az is None:
        # Fallback when feedback is unavailable.
        linear_move_to(x, y, Z_SAFE)
        return

    if abs(az - Z_SAFE) > 1.0:
        linear_move_to(ax, ay, Z_SAFE)

    if abs(ax - x) > 0.5 or abs(ay - y) > 0.5:
        linear_move_to(x, y, Z_SAFE)

def pick_up_from_coords(x: float, y: float):
    """Full pick-up sequence at physical coordinates (x, y)."""
    electromagnet_off()          # ensure magnet is off before approach
    _move_to_xy_at_safe_height(x, y)
    linear_move_to(x, y, Z_GRIP) # optional hover step for better grip
    electromagnet_on()
    time.sleep(0.1)              # brief settle so magnet grips before lifting
    linear_move_to(x, y, Z_SAFE)  # lift straight up with piece


def place_down_from_coords(x: float, y: float):
    """Full place-down sequence at physical coordinates (x, y)."""
    _move_to_xy_at_safe_height(x, y)
    linear_move_to(x, y, Z_GRIP) # lower to placement height
    electromagnet_off()
    time.sleep(0.1)
    linear_move_to(x, y, Z_SAFE)  # lift away cleanly


def pick_up(row: int, col: int):
    x, y = square_to_coords(row, col)
    pick_up_from_coords(x, y)


def  place_down(row: int, col: int):
    x, y = square_to_coords(row, col)
    place_down_from_coords(x, y)


def dispose_piece():
    """Move a held piece to the graveyard and release."""
    gx, gy = GRAVEYARD
    _move_to_xy_at_safe_height(gx, gy)
    linear_move_to(gx, gy, Z_GRIP)
    electromagnet_off()
    linear_move_to(gx, gy, Z_SAFE)



    

# ===========================================================================
#  GAME LOGIC
# ===========================================================================
# def get_stable_board(n_samples: int = 7, delay_ms: int = 30) -> np.ndarray: #* idk might work nicely
#     """Sample the board multiple times and return majority-vote result."""
#     _board_history.clear()
    
#     for _ in range(n_samples):
#         _board_history.append(board.copy())  # type: ignore
#         time.sleep(delay_ms / 1000.0)  # wait between samples
    
#     stacked = np.stack(_board_history, axis=0)  # (n_samples, 6, 6)
#     stable = np.apply_along_axis(
#         lambda x: np.bincount(x, minlength=11).argmax(),
#         axis=0,
#         arr=stacked
#     )
#     return stable.astype(int)


# def get_board_state() -> np.ndarray:
#     """Use the perception module to get the current board state."""
#     global BOARD
#     BOARD = get_stable_board(n_samples=7, delay_ms=30)  # ~210ms total
#     return BOARD

def decide_move(board_state: np.ndarray, playing_white: bool = True) -> str:
    """Ask the engine for the best move given the current board state."""
    print("[ENGINE] Deciding move...")
    move = game.get_best_move(board_state, playing_white)
    return move


def parse_move(move_str: str) -> Tuple[int, int, int, int, int, int]:
    """
    Parse the engine's move string into indices.

    Format: "<piece_id>:<source_cell>-><dest_cell>[=<promote_id>]"
    Example: "1:A1->B2"  or  "1:A6->A1=3"

    Returns: (piece_id, src_row, src_col, dst_row, dst_col, promote_id)
    """
    left, right = move_str.split("->")
    piece_id, source_cell = left.split(":")
    if "=" in right:
        destination_cell, promote_piece_id = right.split("=")
    else:
        destination_cell = right
        promote_piece_id = piece_id

    sr, sc = cell_to_idx(source_cell)
    dr, dc = cell_to_idx(destination_cell)

    return int(piece_id), sr, sc, dr, dc, int(promote_piece_id)


def execute_turn(move_str: str, current_board: np.ndarray, all_poses: dict):
    """
    Execute one full turn:
        1. Parse the engine's move.
        2. If capture: remove opponent piece to graveyard first.
        3. If promotion: move pawn to dest, dispose it, human replaces piece.
        4. If normal move: pick up and place down.
        5. Return arm to home/folded position.

    Promotion note: we physically move the pawn to the destination square
    first (so the board reflects the right square), dispose it, then pause
    for a human to place the promoted piece. The engine has already decided
    the promoted piece ID — we just need the square to be correct.
    """
    debug_print(f"[TURN] Executing move: {move_str}")
    p_id, sr, sc, dr, dc, new_p_id = parse_move(move_str)
    cap_p_id = current_board[dr][dc]
    asx, asy = find_nearest_piece(p_id, sr, sc, all_poses)
    ads, ady = find_nearest_piece(cap_p_id, dr, dc, all_poses)
    debug_print(f"[TURN] Source piece {p_id} at ({sr:.1f}, {sc:.1f}), dest square at ({dr:.1f}, {dc:.1f})")
    rsx, rsy = transform_to_robot(asx, asy) 
    rdx, rdy = transform_to_robot(ads, ady) 

    is_capture   = (current_board[dr][dc] != 0)
    is_promotion = (new_p_id != p_id)

    # --- Step 1: Clear the destination square if capture ---
    if is_capture:
        debug_print(f"[TURN] Capture: removing piece {current_board[dr][dc]} at ({idx_to_cell(dr, dc)})")
        pick_up_from_coords(rdx, rdy)  # pick up the piece that's actually there
        dispose_piece()

    # --- Step 2: Move or promote ---
    if is_promotion:
        debug_print(f"[TURN] Promotion: moving pawn from {idx_to_cell(sr, sc)} -> graveyard,(dispose)")
        pick_up_from_coords(rsx, rsy)  # pick up the piece that's actually there
        dispose_piece()

        # input("  Place the promoted piece on the board, then press Enter to continue...")

    else:
        # --- Normal move ---
        debug_print(f"[TURN] Moving piece {p_id} from {idx_to_cell(sr, sc)} to {idx_to_cell(dr, dc)}")
        pick_up_from_coords(rsx, rsy)
        place_down(dr, dc)

    # --- Step 3: Always reset arm at end of turn ---
    go_to_init()
    debug_print("[TURN] Complete.")


# ===========================================================================
#  UTILITIES
# ===========================================================================

def debug_print(message: str):
    if DEBUG:
        print(f"[DEBUG] {message}")


def _prompt_xy(label: str, default_xy):
    """Prompt for an x,y pair. Blank input keeps default."""
    dx, dy = float(default_xy[0]), float(default_xy[1])
    while True:
        raw = input(f"{label} robot x,y [{dx:.1f},{dy:.1f}]: ").strip()
        if raw == "":
            return (dx, dy)
        parts = raw.replace(" ", "").split(",")
        if len(parts) != 2:
            print("[SYSTEM] Invalid format. Enter as x,y")
            continue
        try:
            return (float(parts[0]), float(parts[1]))
        except ValueError:
            print("[SYSTEM] Invalid number. Enter as x,y")


def _prompt_int(label: str, default_value: int, min_value: int = 1):
    """Prompt for integer value. Blank keeps default."""
    while True:
        raw = input(f"{label} [{default_value}]: ").strip()
        if raw == "":
            return default_value
        try:
            val = int(raw)
            if val < min_value:
                print(f"[SYSTEM] Value must be >= {min_value}")
                continue
            return val
        except ValueError:
            print("[SYSTEM] Invalid integer input.")


def _wait_for_external_move(sock, previous_board: np.ndarray, phase_name: str):
    """Wait for a human/opponent move and confirm the board actually changed."""
    while True:
        user_input = input(
            f"\n[{phase_name}] Make the move on board, then press ENTER (or 'q' to quit): "
        ).strip().lower()
        if user_input == "q":
            return None, None, True

        new_board, new_poses = perception.get_stable_board(sock, stability_required=5)
        if new_board is None:
            print("[ERROR] Could not read board. Check camera stream and try again.")
            continue

        if np.array_equal(new_board, previous_board):
            print("[SYNC] No board change detected. Please complete the move and press ENTER again.")
            continue

        print(f"[SYNC] {phase_name} move detected.")
        return new_board, new_poses, False


def calibration_helper():
    """
    Interactive helper to record corner calibration coordinates.

    Move the arm to each corner square manually (using the web UI),
    then call this function — it reads T:1051 and prints the values to
    paste into the CORNER_* constants above.
    """
    corners = [
        ("TL", 0, 0),
        ("TR", 0, 5),
        ("BL", 5, 0),
        ("BR", 5, 5),
    ]
    debug_print("\n=== CALIBRATION HELPER ===")
    debug_print("Move arm to each corner square at Z_GRIP height, then press Enter.\n")
    for name, row, col in corners:
        input(f"  Position arm at corner {name} (row={row}, col={col}), then press Enter...")
        x, y, z, _, _ = get_feedback_full()
        if x is not None:
            debug_print(f"  CORNER_{name} = ({x:.2f}, {y:.2f})   [z={z:.2f}]")
        else:
            debug_print(f"  [!] Could not read feedback — is TESTING=False and arm connected?")
    debug_print("\nPaste the values above into the CORNER_* constants at the top of this file.")


# ===========================================================================
#  MAIN LOOP
# ===========================================================================

if __name__ == "__main__":

    print("\n[SYSTEM] Enter corner calibration (press Enter to keep defaults).")
    default_rr = perception.ROBOT_REALITY
    a1_xy = _prompt_xy("A1", default_rr[21])
    a6_xy = _prompt_xy("A6", default_rr[22])
    f1_xy = _prompt_xy("F1", default_rr[24])
    f6_xy = _prompt_xy("F6", default_rr[23])
    perception.set_robot_reality_from_cells(a1_xy, a6_xy, f6_xy, f1_xy)
    print(f"[CALIB] A1={a1_xy}  A6={a6_xy}  F1={f1_xy}  F6={f6_xy}")

    while True:
        color_choice = input("Bot color? Enter 'w' for White or 'b' for Black [w]: ").strip().lower()
        if color_choice in ("w", "b", ""):
            break
        print("[SYSTEM] Invalid input. Please enter only 'w' or 'b'.")

    PLAYING_WHITE = (color_choice != "b")
    print(f"[SYSTEM] Bot will play {'WHITE' if PLAYING_WHITE else 'BLACK'}.")

    frame_every_n = _prompt_int(
        "Save every i-th camera frame (1 = save all)",
        default_value=1,
        min_value=1,
    )
    perception.configure_frame_dump(enabled=True, every_n=frame_every_n)
    print(f"[SYSTEM] Camera frames will be saved in: {perception.FRAME_DUMP_DIR}")

    # 1. Setup connection once at the start
    sock = perception.init_perception()

    try:
        debug_print("--- Robot Game Started ---")
        board, poses = perception.get_stable_board(sock, stability_required=5)
        if board is None:
            print("[ERROR] Initial board read failed. Exiting.")
            raise SystemExit(1)

        if PLAYING_WHITE:
            phase_cycle = ["OUR_BOT", "OPPONENT", "OUR_HUMAN", "OPPONENT"]
        else:
            phase_cycle = ["OPPONENT", "OUR_BOT", "OPPONENT", "OUR_HUMAN"]

        phase_idx = 0
        print(f"[PHASE] Active cycle: {' -> '.join(phase_cycle)}")

        while True:
            phase = phase_cycle[phase_idx]
            print(f"\n[PHASE] {phase}")

            if phase == "OUR_BOT":
                debug_print("[PERCEPTION] Board before bot move:")
                debug_print(board)
                move_str = decide_move(board, playing_white=PLAYING_WHITE)
                print(f"[ROBOT] Executing move: {move_str}")
                if move_str:
                    execute_turn(move_str, board, poses)
                    debug_print("[ROBOT] Move completed.")

                board_after, poses_after = perception.get_stable_board(sock, stability_required=5)
                if board_after is not None:
                    board, poses = board_after, poses_after
                else:
                    print("[WARN] Could not refresh board after bot move. Keeping previous board state.")
            else:
                board_new, poses_new, should_quit = _wait_for_external_move(sock, board, phase)
                if should_quit:
                    debug_print("[SYSTEM] Quitting game...")
                    break
                board, poses = board_new, poses_new

            phase_idx = (phase_idx + 1) % len(phase_cycle)
            debug_print("-" * 30)

    except KeyboardInterrupt:
        debug_print("\n[SYSTEM] Manual stop detected.")

    finally:
        # Always close the socket so you don't hang the camera server
        debug_print("[SYSTEM] Closing connection.")
        sock.close()