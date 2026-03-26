"""
RoboGambit Hardware Stage - Main Control Pipeline
===================================================
Orchestrates: Perception -> Game Engine -> Robot Arm Control

CALIBRATION REQUIRED (marked with *):
  Run:  python main.py --calibrate
  Then update the constants below with the printed values.

Usage:
  python main.py --white --manual     # play as white, manual trigger
  python main.py --black --manual     # play as black, manual trigger
  python main.py --white              # play as white, auto-detect turns
  python main_e1.py --calibrate          # interactive calibration wizard
"""

import game
import numpy as np
import requests
import requests.utils
import argparse
import time
import threading
import json
import math
import sys

# Perception imports (same libs as perception.py - we reimplement the loop
# in a background thread so we can read board state without blocking main.
# perception.py runs its main loop at module level so importing it blocks.)
import cv2
import cv2.aruco as aruco
import socket
import struct


# =============================================================================
# * CALIBRATION CONSTANTS - FILL THESE ON-SITE
# =============================================================================

# -- Network --
ARM_IP      = "192.168.4.1"
CAMERA_IP   = "10.168.70.199"    # update to actual camera server IP
CAMERA_PORT = 9999

# -- Arm <-> board coordinate mapping --
# Board world coords (from perception): A1 center = (250, 250) mm
# Run --calibrate to compute these from two measured points.
ARM_X_OFFSET = 0.0      # * CALIBRATE
ARM_Y_OFFSET = 0.0      # * CALIBRATE
ARM_X_SCALE  = 1.0      # * CALIBRATE (scale+sign, typically ~1.0 or ~-1.0)
ARM_Y_SCALE  = 1.0      # * CALIBRATE (scale+sign, typically ~1.0 or ~-1.0)

# -- Z heights (mm, in arm coordinate frame) --
Z_SAFE   = 200.0        # * safe travel height (clears all pieces)
Z_HOVER  = 120.0        # * hover above piece before grab
Z_GRAB   = 40.0         # * height to close gripper on a piece

# -- Gripper --
# RoArm-M2-S EOAT: SMALLER angle = MORE OPEN, LARGER angle = MORE CLOSED
# Default 3.14 rad = fully closed, 1.08 rad = fully open
GRIPPER_OPEN   = 1.08   # fully open  (radians) — small angle = open
GRIPPER_CLOSE  = 2.5    # * CALIBRATE for piece diameter (3.14=max close)
GRIPPER_TORQUE = 300     # grip strength (200=20%, 1000=100%)

# -- Graveyard: where to drop captured pieces (arm coordinates) --
GRAVEYARD_X = -300.0    # * CALIBRATE
GRAVEYARD_Y = 0.0       # * CALIBRATE
GRAVEYARD_Z = 150.0     # * height to release over graveyard
GRAVEYARD_SPACING = 30.0  # mm between stacked captures

# -- Outside arena: for promotion (drop pawn here, human places promoted piece) --
OUTSIDE_X = -350.0      # * CALIBRATE
OUTSIDE_Y = 0.0         # * CALIBRATE

# -- Speeds --
MOVE_SPD      = 0.25    # arm speed for precise moves
MOVE_SPD_FAST = 0.50    # arm speed for safe-height travel

# -- Timing --
SETTLE_TIME      = 1.5  # seconds to wait after arm movement (conservative)
GRIP_TIME        = 0.8  # seconds to wait after gripper open/close
BOARD_STABLE_SEC = 2.0  # seconds board must be stable before reading

# -- State --
_capture_count = 0       # incremented per capture for graveyard offset


# =============================================================================
# PERCEPTION  (background thread - same logic as perception.py)
# =============================================================================

# Camera intrinsics (from perception.py)
_CAM_MTX = np.array([
    [1030.4890823364258, 0, 960],
    [0, 1030.489103794098, 540],
    [0, 0, 1],
], dtype=np.float32)
_DIST = np.zeros((1, 5), dtype=np.float32)

# Board geometry (from perception.py)
_CORNER_WORLD = {21: (350, 350), 22: (350, -350), 23: (-350, -350), 24: (-350, 350)}
_SQ  = 100      # square size mm
_TLX = 300      # top-left X
_TLY = 300      # top-left Y
_BSZ = 6        # board size
_PIECE_IDS = set(range(1, 11))

# ArUco detector (same params as perception.py)
_adict  = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
_aparam = aruco.DetectorParameters()
_aparam.cornerRefinementMethod      = aruco.CORNER_REFINE_SUBPIX
_aparam.adaptiveThreshWinSizeMin    = 3
_aparam.adaptiveThreshWinSizeMax    = 35
_aparam.adaptiveThreshWinSizeStep   = 10
_aparam.minMarkerPerimeterRate      = 0.03
_aparam.maxMarkerPerimeterRate      = 4.0
_aparam.polygonalApproxAccuracyRate = 0.03
_aparam.minCornerDistanceRate       = 0.05
_aparam.minDistanceToBorder         = 1
_adet = aruco.ArucoDetector(_adict, _aparam)

# Shared state
_board_lock  = threading.Lock()
_cur_board   = np.zeros((6, 6), dtype=int)
_board_event = threading.Event()      # set whenever board changes
_perc_ready  = threading.Event()      # set once homography locked
_perc_alive  = True                   # False when perception thread dies


def _px2world(H, px, py):
    pt = cv2.perspectiveTransform(np.array([[[px, py]]], dtype=np.float32), H)
    return float(pt[0][0][0]), float(pt[0][0][1])


def _world2cell(wx, wy):
    best_r, best_c, md = None, None, float('inf')
    for r in range(_BSZ):
        for c in range(_BSZ):
            cx = _TLX - (r * _SQ + _SQ / 2)
            cy = _TLY - (c * _SQ + _SQ / 2)
            d = math.hypot(wx - cx, wy - cy)
            if d < md:
                md, best_r, best_c = d, r, c
    return best_r, best_c


def _build_board(ids, corners, H):
    board = np.zeros((_BSZ, _BSZ), dtype=int)
    for i, mid in enumerate(ids.flatten()):
        if mid not in _PIECE_IDS:
            continue
        c  = corners[i][0]
        px = float(np.mean(c[:, 0]))
        py = float(np.mean(c[:, 1]))
        wx, wy = _px2world(H, px, py)
        r, c2  = _world2cell(wx, wy)
        if r is not None:
            board[r][c2] = mid
    return board


def _recv_frame(sock, data, psz):
    while len(data) < psz:
        pkt = sock.recv(4096)
        if not pkt:
            return None, data
        data += pkt
    packed = data[:psz]
    data   = data[psz:]
    msz    = struct.unpack("Q", packed)[0]
    while len(data) < msz:
        pkt = sock.recv(4096)
        if not pkt:
            return None, data
        data += pkt
    fdata = data[:msz]
    data  = data[msz:]
    frame = cv2.imdecode(np.frombuffer(fdata, dtype=np.uint8), cv2.IMREAD_COLOR)
    return frame, data


def _perception_thread():
    """Background thread: continuously reads camera and updates board state."""
    global _cur_board, _perc_alive

    print("[Perception] Connecting to camera ...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((CAMERA_IP, CAMERA_PORT))
    except (ConnectionRefusedError, OSError) as e:
        print(f"[Perception] ERROR: cannot connect to camera server: {e}")
        _perc_alive = False
        return
    print("[Perception] Connected")

    psz  = struct.calcsize("Q")
    dbuf = b""
    H    = None
    cpix = {}
    prev = None

    try:
        while True:
            frame, dbuf = _recv_frame(sock, dbuf, psz)
            if frame is None:
                print("[Perception] Stream lost!")
                break

            frame = cv2.undistort(frame, _CAM_MTX, _DIST, None, _CAM_MTX)
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = _adet.detectMarkers(gray)

            board = np.zeros((_BSZ, _BSZ), dtype=int)

            if ids is not None:
                for i, mid in enumerate(ids.flatten()):
                    if mid in _CORNER_WORLD:
                        cpix[mid] = np.mean(corners[i][0], axis=0)

                if H is None and len(cpix) == 4:
                    pp = np.array([cpix[m] for m in [21, 22, 23, 24]], dtype=np.float32)
                    wp = np.array([_CORNER_WORLD[m] for m in [21, 22, 23, 24]], dtype=np.float32)
                    H, _ = cv2.findHomography(pp, wp)
                    print("[Perception] Homography locked")
                    _perc_ready.set()

                if H is not None:
                    board = _build_board(ids, corners, H)

            if prev is None or not np.array_equal(board, prev):
                with _board_lock:
                    _cur_board = board.copy()
                _board_event.set()
                prev = board.copy()
                print(f"[Perception] Board updated:\n{board}")

    except Exception as e:
        print(f"[Perception] Error: {e}")
    finally:
        _perc_alive = False
        _board_event.set()  # wake up anyone waiting
        sock.close()
        print("[Perception] Thread exited!")


def get_board_state() -> np.ndarray:
    """Return latest board state from perception."""
    with _board_lock:
        return _cur_board.copy()


def wait_for_board_change(old: np.ndarray, timeout: float = 600) -> np.ndarray:
    """Block until board differs from old. Returns new stable board.
    Verifies stability by checking board is the same across two reads."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not _perc_alive:
            raise RuntimeError("Perception thread died!")
        _board_event.wait(timeout=1.0)
        _board_event.clear()
        new = get_board_state()
        if not np.array_equal(new, old):
            # Verify stability: wait, then confirm board hasn't changed again
            time.sleep(BOARD_STABLE_SEC)
            stable = get_board_state()
            if np.array_equal(new, stable) and not np.array_equal(stable, old):
                return stable
            # Board still changing — keep waiting
    raise TimeoutError("Board did not change within timeout")


# =============================================================================
# ROBOT ARM CONTROL
# =============================================================================

def send_cmd(cmd: dict) -> str:
    """Send a JSON command to the arm via HTTP GET. Returns response text.
    Raises RuntimeError on failure so callers know the command didn't execute."""
    js = json.dumps(cmd)
    url = f"http://{ARM_IP}/js"
    try:
        r = requests.get(url, params={"json": js}, timeout=15)
        return r.text
    except requests.RequestException as e:
        print(f"[Arm] ERROR: command {cmd.get('T','')} failed: {e}")
        raise RuntimeError(f"Arm command failed: {e}") from e


def arm_move(x, y, z, spd=MOVE_SPD):
    """Move arm end-effector to (x, y, z) in mm. Blocking.
    Does NOT set the EOAT angle — gripper state is preserved."""
    # T:104 with spd for interpolated blocking move
    # Omit 't' to avoid overriding gripper angle set by T:106
    send_cmd({"T": 104, "x": x, "y": y, "z": z, "spd": spd})
    time.sleep(SETTLE_TIME)


def arm_home():
    """Return arm to initial position."""
    send_cmd({"T": 100})
    time.sleep(1.5)


def gripper_open():
    send_cmd({"T": 106, "cmd": GRIPPER_OPEN, "spd": 0, "acc": 0})
    time.sleep(GRIP_TIME)


def gripper_close():
    send_cmd({"T": 106, "cmd": GRIPPER_CLOSE, "spd": 0, "acc": 0})
    time.sleep(GRIP_TIME)


def arm_feedback() -> dict:
    """Read current arm position (T:105 -> T:1051 response)."""
    try:
        resp = send_cmd({"T": 105})
        return json.loads(resp)
    except (json.JSONDecodeError, ValueError, RuntimeError):
        return {}


def cell_to_arm_xy(row: int, col: int):
    """Convert board cell (row, col) to arm (x, y) in mm.
    Board world coords: row 0 col 0 = A1 = (250, 250), row 5 col 5 = F6 = (-250, -250)
    """
    bx = 250.0 - row * 100.0
    by = 250.0 - col * 100.0
    ax = ARM_X_OFFSET + ARM_X_SCALE * bx
    ay = ARM_Y_OFFSET + ARM_Y_SCALE * by
    return ax, ay


# =============================================================================
# PICK AND PLACE
# =============================================================================

def pick_piece(row, col):
    """Pick up a piece from board[row][col]."""
    x, y = cell_to_arm_xy(row, col)
    print(f"  [pick] ({row},{col}) -> arm ({x:.0f},{y:.0f})")
    gripper_open()
    arm_move(x, y, Z_SAFE,  spd=MOVE_SPD_FAST)
    arm_move(x, y, Z_HOVER, spd=MOVE_SPD)
    arm_move(x, y, Z_GRAB,  spd=MOVE_SPD)
    gripper_close()
    arm_move(x, y, Z_SAFE,  spd=MOVE_SPD_FAST)


def place_piece(row, col):
    """Place the held piece onto board[row][col]."""
    x, y = cell_to_arm_xy(row, col)
    print(f"  [place] ({row},{col}) -> arm ({x:.0f},{y:.0f})")
    arm_move(x, y, Z_SAFE,  spd=MOVE_SPD_FAST)
    arm_move(x, y, Z_HOVER, spd=MOVE_SPD)
    arm_move(x, y, Z_GRAB,  spd=MOVE_SPD)
    gripper_open()
    arm_move(x, y, Z_SAFE,  spd=MOVE_SPD_FAST)


def drop_to_graveyard():
    """Drop held piece at graveyard. Each capture is offset to avoid stacking."""
    global _capture_count
    gy = GRAVEYARD_Y + _capture_count * GRAVEYARD_SPACING
    _capture_count += 1
    print(f"  [graveyard] dropping captured piece #{_capture_count}")
    arm_move(GRAVEYARD_X, gy, GRAVEYARD_Z, spd=MOVE_SPD_FAST)
    gripper_open()
    arm_move(GRAVEYARD_X, gy, Z_SAFE, spd=MOVE_SPD_FAST)


def drop_outside():
    """Drop pawn outside arena for promotion. Human places promoted piece."""
    print("  [promotion] dropping pawn outside arena")
    arm_move(OUTSIDE_X, OUTSIDE_Y, Z_SAFE,  spd=MOVE_SPD_FAST)
    arm_move(OUTSIDE_X, OUTSIDE_Y, Z_GRAB,  spd=MOVE_SPD)
    gripper_open()
    arm_move(OUTSIDE_X, OUTSIDE_Y, Z_SAFE,  spd=MOVE_SPD_FAST)


# =============================================================================
# MOVE PARSING AND EXECUTION
# =============================================================================

def parse_move(move_str: str) -> dict:
    """Parse '1:B2->B3' or '1:A5->A6=4' into components."""
    promo = None
    s = move_str
    if '=' in s:
        s, p = s.rsplit('=', 1)
        promo = int(p)

    piece_part, move_part = s.split(':')
    src, dst = move_part.split('->')

    def sq(cell):
        col = ord(cell[0].upper()) - ord('A')   # A=0 ... F=5
        row = int(cell[1]) - 1                   # 1->0 ... 6->5
        return row, col

    sr, sc = sq(src)
    dr, dc = sq(dst)
    return dict(piece=int(piece_part), sr=sr, sc=sc, dr=dr, dc=dc, promo=promo)


def execute_move(move_str: str, board: np.ndarray):
    """Execute a chess move physically on the board.
    On failure, attempts emergency cleanup (open gripper + home)."""
    m  = parse_move(move_str)
    sr, sc, dr, dc = m['sr'], m['sc'], m['dr'], m['dc']
    capture   = board[dr][dc] != 0
    promotion = m['promo'] is not None

    print(f"\n[Exec] {move_str}  capture={capture}  promotion={promotion}")

    try:
        # 1. Remove captured piece first
        if capture:
            print(f"  Removing captured piece at ({dr},{dc})")
            pick_piece(dr, dc)
            drop_to_graveyard()

        # 2. Move our piece
        if promotion:
            pick_piece(sr, sc)
            drop_outside()
            print("\n  >>> HUMAN: place the promoted piece on the board <<<")
            input("  Press ENTER when done ... ")
        else:
            pick_piece(sr, sc)
            place_piece(dr, dc)

        arm_home()
        print(f"[Exec] Done: {move_str}\n")

    except (RuntimeError, Exception) as e:
        print(f"[Exec] ERROR during move: {e}")
        print("[Exec] Emergency cleanup: opening gripper and homing ...")
        try:
            gripper_open()
            arm_home()
        except Exception:
            pass
        raise


# =============================================================================
# GAME LOOP
# =============================================================================

def _emergency_cleanup():
    """Best-effort: open gripper and home arm."""
    try:
        gripper_open()
    except Exception:
        pass
    try:
        arm_home()
    except Exception:
        pass


def run_game(playing_white: bool, manual: bool):
    """Main game loop."""

    # --- Start perception ---
    t = threading.Thread(target=_perception_thread, daemon=True)
    t.start()

    print("[Main] Waiting for perception (need 4 corner markers) ...")
    if not _perc_ready.wait(timeout=30):
        print("[Main] ERROR: perception did not initialise. Check camera + markers.")
        return
    print("[Main] Perception ready")

    # --- Initialise arm ---
    print("[Main] Initialising arm ...")
    send_cmd({"T": 107, "tor": GRIPPER_TORQUE})
    arm_home()
    gripper_open()
    print("[Main] Arm ready\n")

    time.sleep(2)
    board = get_board_state()
    print(f"[Main] Starting board:\n{board}\n")

    bot_moves = 0

    try:
        # ----- MANUAL TRIGGER MODE -----
        if manual:
            print("=== MANUAL MODE ===")
            print("Press ENTER when it is the bot's turn.")
            print("Type 'q' to quit, 's' to show board.\n")

            while True:
                cmd = input("[bot turn?] ").strip().lower()
                if cmd == 'q':
                    break
                if cmd == 's':
                    print(get_board_state())
                    continue
                if cmd not in ('', 'y', 'yes'):
                    continue

                board = get_board_state()
                print(f"Board:\n{board}\n")

                print("Computing best move ...")
                try:
                    move_str = game.get_best_move(board, playing_white)
                except Exception as e:
                    print(f"[Main] Game engine error: {e}")
                    continue

                if move_str is None:
                    print("No legal moves - game over!")
                    break

                print(f"Best move: {move_str}")
                ok = input(f"Execute? (y/n) ").strip().lower()
                if ok not in ('y', 'yes', ''):
                    print("Skipped.\n")
                    continue

                execute_move(move_str, board)
                bot_moves += 1
                print(f"Bot has played {bot_moves} moves.\n")

        # ----- AUTO-DETECT MODE -----
        else:
            print("=== AUTO MODE ===")
            print("Watching for board changes. Press Ctrl-C to stop.\n")

            while True:
                print("[Main] Waiting for board change ...")
                try:
                    board = wait_for_board_change(board)
                except TimeoutError:
                    print("Timeout - no change detected.")
                    continue
                except RuntimeError as e:
                    print(f"[Main] {e}")
                    break

                print(f"Board changed:\n{board}\n")

                ans = input("Bot's turn? (y/n/q) ").strip().lower()
                if ans == 'q':
                    break
                if ans not in ('y', 'yes'):
                    continue

                print("Computing best move ...")
                try:
                    move_str = game.get_best_move(board, playing_white)
                except Exception as e:
                    print(f"[Main] Game engine error: {e}")
                    continue

                if move_str is None:
                    print("No legal moves - game over!")
                    break

                print(f"Best move: {move_str}")
                execute_move(move_str, board)
                bot_moves += 1

                # Update board to post-move state so we don't re-trigger
                # on our own physical move
                time.sleep(BOARD_STABLE_SEC)
                board = get_board_state()

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\n[Main] Unexpected error: {e}")
    finally:
        print("[Main] Cleaning up ...")
        _emergency_cleanup()
        print(f"Game ended. Bot played {bot_moves} moves.")


# =============================================================================
# CALIBRATION WIZARD
# =============================================================================

def calibrate():
    """Interactive wizard: move arm to board squares, compute coordinate mapping."""
    print("\n" + "=" * 55)
    print("  CALIBRATION WIZARD")
    print("  Move the arm manually to board squares, then record.")
    print("=" * 55)

    def read_pos(label):
        input(f"\n  Move arm to {label}, then press ENTER ...")
        fb = arm_feedback()
        if fb and 'x' in fb:
            print(f"    Read: x={fb['x']:.1f}  y={fb['y']:.1f}  z={fb['z']:.1f}")
            return fb['x'], fb['y'], fb.get('z', 0)
        print("    Could not read arm. Enter manually:")
        try:
            x = float(input("    x = "))
            y = float(input("    y = "))
            z = float(input("    z = "))
        except ValueError:
            print("    Invalid input, using 0.")
            return 0.0, 0.0, 0.0
        return x, y, z

    a1x, a1y, _ = read_pos("center of A1 (bottom-left, row 0 col 0)")
    f6x, f6y, _ = read_pos("center of F6 (top-right, row 5 col 5)")

    # Sanity check
    if abs(a1x - f6x) < 50 and abs(a1y - f6y) < 50:
        print("\n  WARNING: A1 and F6 positions are very close!")
        print("  Are you sure both measurements are correct?")

    # A1 board world = (250, 250), F6 = (-250, -250)
    # arm = offset + scale * board_world
    sx = (a1x - f6x) / 500.0
    sy = (a1y - f6y) / 500.0
    ox = a1x - sx * 250.0
    oy = a1y - sy * 250.0

    _, _, z_grab = read_pos("piece grab height (gripper on piece)")
    _, _, z_safe = read_pos("safe travel height (above all pieces)")

    print(f"\n{'=' * 55}")
    print(f"  PASTE THESE INTO main.py:")
    print(f"  ARM_X_OFFSET = {ox:.1f}")
    print(f"  ARM_Y_OFFSET = {oy:.1f}")
    print(f"  ARM_X_SCALE  = {sx:.3f}")
    print(f"  ARM_Y_SCALE  = {sy:.3f}")
    print(f"  Z_GRAB       = {z_grab:.1f}")
    print(f"  Z_SAFE       = {z_safe:.1f}")
    print(f"  Z_HOVER      = {z_grab + 80:.1f}")
    print(f"{'=' * 55}")

    # Verify — test a THIRD point (D3 = row 2 col 3, center-ish) in addition to A1/F6
    ok = input("\nTest? Move arm to A1, D3, F6? (y/n) ").strip().lower()
    if ok in ('y', 'yes', ''):
        global ARM_X_OFFSET, ARM_Y_OFFSET, ARM_X_SCALE, ARM_Y_SCALE
        ARM_X_OFFSET, ARM_Y_OFFSET = ox, oy
        ARM_X_SCALE, ARM_Y_SCALE = sx, sy

        for label, r, c in [("A1", 0, 0), ("D3", 2, 3), ("F6", 5, 5)]:
            x, y = cell_to_arm_xy(r, c)
            print(f"  Moving to {label} ({r},{c}) -> ({x:.0f}, {y:.0f}) ...")
            arm_move(x, y, z_safe, spd=MOVE_SPD)
            input(f"  Check {label}. Press ENTER for next ...")

        arm_home()

    print("Calibration done.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RoboGambit Hardware Controller")
    ap.add_argument("--white",     action="store_true", help="Bot plays white")
    ap.add_argument("--black",     action="store_true", help="Bot plays black")
    ap.add_argument("--manual",    action="store_true", help="Manual trigger each turn")
    ap.add_argument("--calibrate", action="store_true", help="Run calibration wizard")
    args = ap.parse_args()

    if args.calibrate:
        calibrate()
        sys.exit(0)

    if args.black:
        playing_white = False
    elif args.white:
        playing_white = True
    else:
        c = input("Play as (w)hite or (b)lack? ").strip().lower()
        playing_white = c != 'b'

    print(f"\n  Playing as: {'WHITE' if playing_white else 'BLACK'}")
    print(f"  Mode: {'manual' if args.manual else 'auto-detect'}\n")

    run_game(playing_white, args.manual)
