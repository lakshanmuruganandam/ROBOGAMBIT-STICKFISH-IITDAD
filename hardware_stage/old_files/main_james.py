"""
Main control script for RoboGambit Hardware Stage.
Orchestrates perception, game logic, and robotic arm commands.

Two phases:
  Phase 1 (Setup): Place pieces from collection area onto the board
  Phase 2 (Game):  Alternate human/robot turns in collaborative chess
"""

import game
import numpy as np
import requests
import serial
import time
import perception

# ── Configuration ─────────────────────────────────────────────────────────────
SERIAL_PORT  = 'COM3'
SERIAL_BAUD  = 115200
ROBOT_IP     = "192.168.4.1"

# Playing side: True = white, False = black
PLAYING_WHITE = True

# Turn alternation: True = robot goes first, False = human goes first
ROBOT_STARTS = True

# Waveshare RoArm M2 command types
CMD_MOVE_XYZ = 104   # T:104 — move to (x, y, z) with clamp angle t (rad)
Z_UP         = 80    # height to clear pieces (mm)
Z_DOWN       = 10    # height to grab/release pieces (mm)
MOVE_SPD     = 0.25  # movement speed
CLAMP_ANGLE  = 3.14  # end-effector clamp angle in radians

# Board cell centers in world coords (mm) — same as perception module
# Row/Col 0..5 map to centers 250, 150, 50, -50, -150, -250
CELL_CENTERS = [250, 150, 50, -50, -150, -250]

# Graveyard position for captured pieces (off-board)
GRAVEYARD_X = -400
GRAVEYARD_Y = 0

# Collection area positions (mm) — UPDATE with actual positions on-site
# Each entry: piece_id -> (x, y) world coordinates in collection zone
COLLECTION_POSITIONS = {}  # populated by scan or hardcoded on-site

# File letters to column index
FILE_TO_COL = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}

# ── Serial setup ──────────────────────────────────────────────────────────────
ser = None


def init_serial():
    """Initialize serial connection to robotic arm."""
    global ser
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD)
        print(f"Serial connected on {SERIAL_PORT}")
    except serial.SerialException as e:
        print(f"WARNING: Could not open serial port {SERIAL_PORT}: {e}")
        print("Running without serial connection.")


def pick():
    """Activate electromagnet (grab piece)."""
    if ser:
        ser.write(b'1')
        time.sleep(0.5)


def place():
    """Deactivate electromagnet (release piece)."""
    if ser:
        ser.write(b'0')
        time.sleep(0.5)


# ── Robot communication ───────────────────────────────────────────────────────

def send_cmd(command):
    """
    Send a Waveshare RoArm M2 JSON command via HTTP.
    Uses T:104 (linear move to XYZ with clamp angle and speed).
    """
    json_str = (
        f'{{"T":{CMD_MOVE_XYZ},'
        f'"x":{command["x"]},"y":{command["y"]},"z":{command["z"]},'
        f'"t":{CLAMP_ANGLE},"spd":{MOVE_SPD}}}'
    )
    url = f"http://{ROBOT_IP}/js?json={json_str}"
    print(f"Sending: {url}")
    try:
        response = requests.get(url, timeout=10)
        print(f"Response: {response.text}")
        return True
    except requests.RequestException as e:
        print(f"ERROR sending command: {e}")
        return False


def move_arm_to(x, y, z):
    """Move arm to (x, y, z) position. T:104 may block; extra sleep for safety."""
    send_cmd({"x": x, "y": y, "z": z})
    time.sleep(2.5)


def pick_and_place(from_x, from_y, to_x, to_y):
    """
    Pick a piece at (from_x, from_y) and place it at (to_x, to_y).
    Full sequence: hover -> lower -> magnet on -> raise -> move -> lower -> magnet off -> raise
    """
    # Pick
    move_arm_to(from_x, from_y, Z_UP)
    move_arm_to(from_x, from_y, Z_DOWN)
    pick()
    move_arm_to(from_x, from_y, Z_UP)

    # Place
    move_arm_to(to_x, to_y, Z_UP)
    move_arm_to(to_x, to_y, Z_DOWN)
    place()
    move_arm_to(to_x, to_y, Z_UP)


# ── Move parsing ──────────────────────────────────────────────────────────────

def parse_move(move_str):
    """
    Parse a move string from the game engine.
    Format: "piece_id:FILE_RANK->FILE_RANK" e.g. "4:D2->D3" or "1:A2->A3=4" (promotion)

    Returns: (from_row, from_col, to_row, to_col, promotion)
    """
    promotion = None
    if '=' in move_str:
        move_str, promotion = move_str.rsplit('=', 1)

    _, coords = move_str.split(':')
    from_sq, to_sq = coords.split('->')

    from_col = FILE_TO_COL[from_sq[0]]
    from_row = int(from_sq[1]) - 1
    to_col   = FILE_TO_COL[to_sq[0]]
    to_row   = int(to_sq[1]) - 1

    return from_row, from_col, to_row, to_col, promotion


def cell_to_world(row, col):
    """Convert board cell (row, col) to world coordinates (x_mm, y_mm)."""
    return CELL_CENTERS[col], CELL_CENTERS[row]


def execute_move(move_str):
    """
    Execute a game move on the physical board.
    Handles captures by removing the captured piece to graveyard first.
    """
    print(f"\nExecuting move: {move_str}")
    from_row, from_col, to_row, to_col, promotion = parse_move(move_str)

    from_x, from_y = cell_to_world(from_row, from_col)
    to_x, to_y     = cell_to_world(to_row, to_col)

    # If target square is occupied (capture), remove that piece first
    target_piece = perception.board[to_row][to_col]
    if target_piece != 0:
        print(f"  Capturing piece {target_piece} at [{to_row},{to_col}]")
        pick_and_place(to_x, to_y, GRAVEYARD_X, GRAVEYARD_Y)

    # Move our piece
    pick_and_place(from_x, from_y, to_x, to_y)

    print(f"Move {move_str} completed.")


# ── Phase 1: Setup ────────────────────────────────────────────────────────────

def setup_phase(target_layout):
    """
    Phase 1: Place pieces from collection area onto the board.

    target_layout: 6x6 numpy array with piece IDs at target positions.
                   Revealed at match start.

    Strategy:
      1. Scan collection area to find piece positions (via perception)
      2. For each non-zero cell in target_layout, pick the piece from
         collection and place it on the correct board cell
    """
    print("=" * 50)
    print("PHASE 1: Board Setup")
    print("=" * 50)
    print(f"Target layout:\n{target_layout}\n")

    # Scan the collection area to locate pieces
    print("Scanning for pieces in collection area...")
    board_state = perception.get_board_state()

    # Build list of pieces to place: (piece_id, target_row, target_col)
    placements = []
    for r in range(6):
        for c in range(6):
            pid = int(target_layout[r][c])
            if pid != 0:
                placements.append((pid, r, c))

    print(f"Need to place {len(placements)} pieces.")

    for pid, target_row, target_col in placements:
        target_x, target_y = cell_to_world(target_row, target_col)

        # Find where this piece currently is
        # Option A: Use COLLECTION_POSITIONS if hardcoded
        if pid in COLLECTION_POSITIONS:
            src_x, src_y = COLLECTION_POSITIONS[pid]
        else:
            # Option B: Scan perception to find the piece
            board_state = perception.get_board_state()
            found = False
            for r in range(6):
                for c in range(6):
                    if int(board_state[r][c]) == pid:
                        src_x, src_y = cell_to_world(r, c)
                        found = True
                        break
                if found:
                    break

            if not found:
                print(f"  WARNING: Piece {pid} not found! Skipping.")
                continue

        print(f"  Placing piece {pid}: ({src_x},{src_y}) -> board [{target_row},{target_col}]")
        pick_and_place(src_x, src_y, target_x, target_y)

    # Verify placement
    print("\nVerifying board setup...")
    final_board = perception.get_board_state()
    mismatches = 0
    for r in range(6):
        for c in range(6):
            expected = int(target_layout[r][c])
            actual   = int(final_board[r][c])
            if expected != actual:
                mismatches += 1
                print(f"  MISMATCH at [{r},{c}]: expected {expected}, got {actual}")

    if mismatches == 0:
        print("Board setup verified successfully!")
    else:
        print(f"WARNING: {mismatches} mismatches detected.")

    print("Phase 1 complete.\n")


# ── Phase 2: Collaborative Game ───────────────────────────────────────────────

def wait_for_human_move():
    """
    Wait for the human player to physically make their move.
    Detects when the board state changes from the current snapshot.
    """
    print("\n--- HUMAN TURN ---")
    print("Waiting for human to make a move...")

    snapshot = perception.get_board_state()
    while True:
        time.sleep(1)
        current = perception.get_board_state()
        if not np.array_equal(current, snapshot):
            # Wait a moment for the human to fully complete the move
            time.sleep(1.5)
            final = perception.get_board_state()
            print("Human move detected!")
            print(f"Board:\n{final}")
            return final


def robot_turn(move_number):
    """
    Robot's autonomous turn: perceive, think, execute.
    No human intervention allowed.
    """
    print(f"\n--- ROBOT TURN (move {move_number}) ---")

    board_state = perception.get_board_state()
    print(f"Board:\n{board_state}")

    move_str = game.get_best_move(board_state, PLAYING_WHITE)

    if move_str is None:
        print("No valid move found — game may be over.")
        return False

    print(f"Best move: {move_str}")
    execute_move(move_str)
    return True


def game_phase():
    """
    Phase 2: Collaborative chess game.
    Alternates between human and robot turns.
    """
    print("=" * 50)
    print("PHASE 2: Collaborative Game")
    print(f"Playing as: {'WHITE' if PLAYING_WHITE else 'BLACK'}")
    print(f"{'Robot' if ROBOT_STARTS else 'Human'} goes first")
    print("=" * 50)

    move_number = 1
    robot_turn_next = ROBOT_STARTS

    while True:
        if robot_turn_next:
            if not robot_turn(move_number):
                break
            move_number += 1
        else:
            wait_for_human_move()

        # Alternate turns
        robot_turn_next = not robot_turn_next


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("RoboGambit Hardware Stage")
    print("=" * 50)

    # Initialize hardware
    init_serial()
    perception.connect()

    try:
        # Phase 1: Setup — place pieces on board
        # The target layout is revealed at match start.
        # Enter it here or load from a file/input.
        print("\nEnter target board layout (or press Enter to skip setup phase):")
        layout_input = input().strip()

        if layout_input:
            # Parse layout — expects 6 lines of 6 space-separated piece IDs
            rows = layout_input.split(';')
            target_layout = np.zeros((6, 6), dtype=int)
            for r, row_str in enumerate(rows[:6]):
                vals = row_str.strip().split()
                for c, v in enumerate(vals[:6]):
                    target_layout[r][c] = int(v)
            setup_phase(target_layout)
        else:
            print("Skipping setup phase.")

        # Phase 2: Collaborative game
        game_phase()

    except KeyboardInterrupt:
        print("\nGame interrupted by user.")

    finally:
        perception.cleanup()
        if ser:
            ser.close()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
