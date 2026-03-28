"""LADDOO-FINAL-NOCHOICE main control loop.

Coordinates team-phase gameplay (bot + human), shared clock accounting,
perception polling, and arm command execution.
"""

import game
import numpy as np
import requests
import argparse
import serial
import time
import perception
from perception import BoardPerception


import json

# Board geometry mapping; must stay consistent with perception grid conventions.
_COL_MAP     = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5}

# Arm Z heights in mm — tune these to your physical setup
Z_HOVER = 200   # safe travel height above the board
Z_PICK  =  20   # surface level to engage the piece

ARM_SPD = 0.3   # speed for T=104 command
T_ANGLE = 3.14  # wrist angle (rad) — straight down

# Off-board discard zone for captured/promoted-out pieces (world mm)
DISCARD_X, DISCARD_Y = -150, 0

# Reserve coordinates are kept for setup reference; promotion replacement is human-inserted.
RESERVE_POS = {
    4: (-150, -100),   # White Queen  reserve
    3: (-150, -200),   # White Bishop reserve
    2: (-150, -300),   # White Knight reserve
    9:  (450, -100),   # Black Queen  reserve
    8:  (450, -200),   # Black Bishop reserve
    7:  (450, -300),   # Black Knight reserve
}



BOARD = np.zeros((6, 6), dtype=int)
# Arm controller serial port.
ser = serial.Serial("COM4", baudrate=115200, dsrdtr=None)
ser.setRTS(False)
ser.setDTR(False)
# Electromagnet (solenoid) serial port.
ser2 = serial.Serial('COM3', 115200) 
POSES = {}
vision_system = BoardPerception()
TEAM_TIME_REMAINING = 0.0
CLOCK_OFFSET_SEC = 10.0

WHITE_PIECES = {1, 2, 3, 4, 5}
BLACK_PIECES = {6, 7, 8, 9, 10}


def _parse_move_string(move: str):
    """Return (piece_id, src_cell, dst_cell, promo_id_or_none) from engine move string."""
    raw = move.strip()
    piece_id = None
    promo_id = None
    if ":" in raw:
        pfx, raw = raw.split(":", 1)
        try:
            piece_id = int(pfx)
        except ValueError:
            piece_id = None
    if "=" in raw:
        raw, promo = raw.split("=", 1)
        try:
            promo_id = int(promo)
        except ValueError:
            promo_id = None
    src_cell, dst_cell = raw.split("->")
    return piece_id, src_cell, dst_cell, promo_id


def _cell_to_world(cell: str):
    col = _COL_MAP[cell[0].upper()]
    row = int(cell[1]) - 1
    wx = perception.TOP_LEFT_X - (row * perception.SQUARE_SIZE + perception.SQUARE_SIZE / 2)
    wy = perception.TOP_LEFT_Y - (col * perception.SQUARE_SIZE + perception.SQUARE_SIZE / 2)
    return wx, wy


def _nearest_pose_for_piece(piece_id: int, target_xy):
    """Use detected piece poses when available; fallback to target cell center."""
    if piece_id is None:
        return target_xy
    pvals = POSES.get(piece_id)
    if pvals is None:
        return target_xy
    if isinstance(pvals, tuple):
        return pvals
    if isinstance(pvals, list) and pvals:
        tx, ty = target_xy
        best = min(pvals, key=lambda p: (p[0] - tx) ** 2 + (p[1] - ty) ** 2)
        return best
    return target_xy

def get_board_state() -> np.ndarray:
    """Use the perception module to get the current board state."""
    global BOARD, POSES
    latest_board, latest_poses = vision_system.get_latest_state()
    if latest_board is not None:
        BOARD = latest_board
        POSES = latest_poses
    return BOARD

def move(playing_white, time_budget_sec: float = None, remaining_time_sec: float = None) -> str:
    """Determine the best move using the game module."""
    board = get_board_state()
    if time_budget_sec is not None:
        best = game.get_best_move(
            board,
            playing_white,
            time_budget_sec=time_budget_sec,
            remaining_time_sec=remaining_time_sec,
        )
    else:
        best = game.get_best_move(board, playing_white, remaining_time_sec=remaining_time_sec)
    if best is not None:
        return best
    if hasattr(game, "get_move"):
        return game.get_move(board, 1 if playing_white else 0)
    return None


def movetocmd(move: str) -> list:
    """
    Convert a move string into an ordered list of robot steps.

    Each element is either:
      - A JSON string  → pass to send_cmd()
      - "PICK"         → call pick()
      - "PLACE"        → call place()

    """

    # ── Internal helpers ──────────────────────────────────────────────────────

    def arm_goto(x, y, z):
        """Build a Waveshare T=104 (CMD_XYZT_GOAL_CTRL) JSON command."""
        return json.dumps({
            "T": 104,
            "x": round(x, 1),
            "y": round(y, 1),
            "z": round(z, 1),
            "t": T_ANGLE,
            "spd": ARM_SPD
        })

    def pick_from(wx, wy):
        """Steps to hover → descend → PICK → raise at world position."""
        return [
            arm_goto(wx, wy, Z_HOVER),   # hover above cell
            arm_goto(wx, wy, Z_PICK),    # descend to piece
            "PICK",                       # electromagnet ON
            arm_goto(wx, wy, Z_HOVER),   # raise back up
        ]

    def place_at(wx, wy):
        """Steps to hover → descend → PLACE → raise at world position."""
        return [
            arm_goto(wx, wy, Z_HOVER),   # hover above target
            arm_goto(wx, wy, Z_PICK),    # descend to surface
            "PLACE",                      # electromagnet OFF
            arm_goto(wx, wy, Z_HOVER),   # raise back up
        ]

    # ── Parse move string ─────────────────────────────────────────────────────
    mover_piece, src_cell, dst_cell, promo_id = _parse_move_string(move)
    src_x, src_y = _cell_to_world(src_cell)
    dst_x, dst_y = _cell_to_world(dst_cell)

    # Use live pose if available to improve pick accuracy.
    src_x, src_y = _nearest_pose_for_piece(mover_piece, (src_x, src_y))

    # ── Check for capture (enemy piece sitting at destination?) ───────────────
    board = get_board_state()
    dst_row = int(dst_cell[1]) - 1
    dst_col = _COL_MAP[dst_cell[0].upper()]
    is_capture = board[dst_row][dst_col] != 0

    steps = []

    # ── 1. Capture: clear the destination square first ────────────────────────
    if is_capture:
        captured_id = int(board[dst_row][dst_col])
        cap_x, cap_y = _nearest_pose_for_piece(captured_id, (dst_x, dst_y))
        steps += pick_from(cap_x, cap_y)
        steps += place_at(DISCARD_X, DISCARD_Y)

    # ── 2. Main move: lift our piece from source, set it at destination ───────
    steps += pick_from(src_x, src_y)
    steps += place_at(dst_x, dst_y)

    # ── 3. Promotion: robot removes pawn, human places the chosen replacement piece ──
    if promo_id is not None:
        # Remove pawn from promotion square → discard
        steps += pick_from(dst_x, dst_y)
        steps += place_at(DISCARD_X, DISCARD_Y)
        steps.append(f"HUMAN_PROMO:{promo_id}:{dst_cell}")

    return steps


def pick():
    """Activate the electromagnet to grip a piece."""
    ser2.write(b'1')



def place():
    """Deactivate the electromagnet to release a piece."""
    ser2.write(b'0')



def send_cmd(command: str):
    """Send a single JSON command to the robot arm via HTTP."""
    print(f"Sending command: {command}")
    ser.write(command.encode() + b'\n')

COL_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F']
LOG_FILE = "game_log.txt"

def _idx_to_cell(row: int, col: int) -> str:
    return f"{COL_LETTERS[col]}{row + 1}"


def log_move(prev_board: np.ndarray, curr_board: np.ndarray, log_file: str = LOG_FILE):
    vacated  = []   # had a piece, now empty  → this is the source square
    arrived  = []   # was empty, now has piece → destination (no capture)
    replaced = []   # had one piece, now different piece → capture destination

    for r in range(6):
        for c in range(6):
            pv, cv = int(prev_board[r][c]), int(curr_board[r][c])
            if pv == cv:
                continue
            if pv != 0 and cv == 0:
                vacated.append((r, c, pv))
            elif pv == 0 and cv != 0:
                arrived.append((r, c, cv))
            elif pv != 0 and cv != 0:
                replaced.append((r, c, pv, cv))

    if not vacated:
        return 

    # For perception glitches, confirming source of moves
    src_r, src_c, moved_piece = vacated[0]
    src_cell = _idx_to_cell(src_r, src_c)

    promo_id = None

    if arrived:                                # piece landed on an empty square
        dst_r, dst_c, dst_piece = arrived[0]
        if dst_piece != moved_piece:           # different piece ID → promotion
            promo_id = dst_piece
    elif replaced:                             # piece landed on an occupied square → capture
        dst_r, dst_c, _, dst_piece = replaced[0]
        if dst_piece != moved_piece:           # different piece ID → capture + promotion (rare)
            promo_id = dst_piece
    else:
        return  # Can't determine destination; skip

    dst_cell = _idx_to_cell(dst_r, dst_c)

    
    move_str = f"{moved_piece}:{src_cell}->{dst_cell}"
    if promo_id is not None:
        move_str += f"={promo_id}"

    with open(log_file, 'a') as f:
        f.write(move_str + '\n')
    return (moved_piece, src_r, src_c, dst_r, dst_c, promo_id)


def log_result(result: str, log_file: str = LOG_FILE):
    with open(log_file, 'a') as f:
        f.write(f"Result: {result}\n\n")

def check_legal(prev_board, move_tuple):
    if move_tuple is None:
        return False

    moved_piece, sr, sc, dr, dc, _promo = move_tuple
    if not (0 <= sr < 6 and 0 <= sc < 6 and 0 <= dr < 6 and 0 <= dc < 6):
        return False
    if (sr, sc) == (dr, dc):
        return False
    if int(prev_board[sr][sc]) != int(moved_piece):
        return False

    src_white = moved_piece in WHITE_PIECES
    dst_piece = int(prev_board[dr][dc])
    if src_white and dst_piece in WHITE_PIECES:
        return False
    if (not src_white) and dst_piece in BLACK_PIECES:
        return False

    # Backward compatibility path for older game APIs that expose full legal move gen.
    has_legacy = all(hasattr(game, name) for name in (
        "is_white", "get_offboard_pieces", "WHITE_KING", "BLACK_KING", "get_all_moves"
    ))
    if has_legacy:
        try:
            is_white_move = game.is_white(moved_piece)
            offboard = game.get_offboard_pieces(prev_board)
            offboard["Total"] = sum(offboard.values())
            wk = np.where(prev_board == game.WHITE_KING)
            bk = np.where(prev_board == game.BLACK_KING)
            king = {
                game.WHITE_KING: (int(wk[0][0]), int(wk[1][0])),
                game.BLACK_KING: (int(bk[0][0]), int(bk[1][0])),
            }
            return move_tuple in game.get_all_moves(prev_board, is_white_move, offboard, king)
        except Exception:
            # Fall back to structural check when legacy API exists but fails.
            pass

    return True

def get_stable_board_state(required_frames=10):
    """Return a majority-vote board from recent reads for noise resistance."""
    frames = []
    timeout_s = max(1.5, required_frames * 0.12)
    t0 = time.time()

    while len(frames) < required_frames and (time.time() - t0) < timeout_s:
        current_board = get_board_state()
        if current_board is not None and current_board.shape == (6, 6):
            frames.append(current_board.astype(np.int32, copy=True))
        time.sleep(0.05)

    if not frames:
        return BOARD

    stack = np.stack(frames, axis=0)
    stable = np.zeros((6, 6), dtype=np.int32)
    for r in range(6):
        for c in range(6):
            stable[r, c] = int(np.bincount(stack[:, r, c], minlength=11).argmax())
    return stable


def _deduct_team_time(elapsed: float, reason: str):
    global TEAM_TIME_REMAINING
    TEAM_TIME_REMAINING = max(0.0, TEAM_TIME_REMAINING - max(0.0, elapsed))
    print(
        f"[CLOCK] -{elapsed:.2f}s for {reason}; "
        f"raw_remaining={TEAM_TIME_REMAINING:.2f}s "
        f"effective_remaining={_effective_team_time_remaining():.2f}s"
    )


def _phase_name(phase: int) -> str:
    names = {
        0: "OUR_BOT",
        1: "OPPONENT",
        2: "OUR_HUMAN",
        3: "OPPONENT",
    }
    return names.get(phase % 4, "UNKNOWN")


def _effective_team_time_remaining() -> float:
    return max(0.0, TEAM_TIME_REMAINING + CLOCK_OFFSET_SEC)


def _bot_time_budget_seconds() -> float:
    """Compute bot think-time cap from raw team clock."""
    if TEAM_TIME_REMAINING <= 2.0:
        return 0.5
    return max(0.5, min(12.0, TEAM_TIME_REMAINING * 0.12))


def _print_startup_connections() -> None:
    """Print serial and camera connectivity status before starting the game loop."""
    arm_ok = bool(getattr(ser, "is_open", False))
    solenoid_ok = bool(getattr(ser2, "is_open", False))

    cam_ok = False
    cam_socket = getattr(vision_system, "client_socket", None)
    if cam_socket is not None:
        try:
            cam_ok = cam_socket.fileno() != -1
        except Exception:
            cam_ok = False

    print(f"[STARTUP] arm_serial={('CONNECTED' if arm_ok else 'DISCONNECTED')} port={getattr(ser, 'port', 'unknown')}")
    print(
        f"[STARTUP] solenoid_serial={('CONNECTED' if solenoid_ok else 'DISCONNECTED')} "
        f"port={getattr(ser2, 'port', 'unknown')}"
    )
    print(
        f"[STARTUP] camera_socket={('CONNECTED' if cam_ok else 'DISCONNECTED')} "
        f"endpoint={perception.SERVER_IP}:{perception.SERVER_PORT}"
    )

   
if __name__ == "__main__":
    color=input("Which color is the bot playing?(w/b): ")
    TEAM_TIME_REMAINING=float(int(input("Enter the Time control(10/15): "))*60)
    playing_white=(color=='w')
    if playing_white==True: t=0
    else: t=-1
    _print_startup_connections()
    print(
        f"[CLOCK] Team clock initialized: raw={TEAM_TIME_REMAINING:.0f}s "
        f"offset={CLOCK_OFFSET_SEC:+.1f}s "
        f"effective={_effective_team_time_remaining():.1f}s"
    )
    print("[PHASE] Cycle: OUR_BOT -> OPPONENT -> OUR_HUMAN -> OPPONENT")
    print("[RULE] Coordinator mode active:")
    print("[RULE] 1) Team alternates robot and human turns.")
    print("[RULE] 2) Robot executes captures as OUT-then-IN.")
    print("[RULE] 3) On promotion, robot removes pawn; human places selected replacement piece.")
    BOARD= get_stable_board_state()
    phase_started_at = time.time()
    last_phase = None

    while True:
        curr = get_stable_board_state()
        phase = t % 4
        if _effective_team_time_remaining() <= 0.0:
            print(
                f"[CLOCK] Team clock exhausted "
                f"(raw={TEAM_TIME_REMAINING:.2f}s effective={_effective_team_time_remaining():.2f}s)."
            )
            log_result("TIMEOUT", 'rglog.txt')
            break

        if phase != last_phase:
            print(f"[PHASE] active={_phase_name(phase)}")
            last_phase = phase

        if phase==0:
            bot_turn_start = time.time()
            send_cmd(json.dumps({'T':100}))
            budget = _bot_time_budget_seconds()
            best_move = move(
                playing_white,
                time_budget_sec=budget,
                remaining_time_sec=_effective_team_time_remaining(),
            )
            if best_move is None:
                print("No moves available — game over.")
                log_result(input("Enter result"),'rglog.txt')
                break
            print(
                f"[BOT] move={best_move} budget={budget:.2f}s "
                f"raw_remaining={TEAM_TIME_REMAINING:.2f}s "
                f"effective_remaining={_effective_team_time_remaining():.2f}s"
            )
            for step in movetocmd(best_move):
                if step == "PICK":  pick()
                elif step == "PLACE": place()
                elif step.startswith("HUMAN_PROMO:"):
                    _tag, pid, cell = step.split(":", 2)
                    print(f"[PROMOTION] Bot selected piece id {pid} for {cell}.")
                    input("Place promoted piece by hand, then press Enter to continue...")
                else:
                    send_cmd(step)
                time.sleep(0.25)
            send_cmd(json.dumps({'T':100}))
            BOARD=get_stable_board_state()
            t+=1
            _deduct_team_time(time.time() - bot_turn_start, "our bot turn (think + arm + promotion handling)")
            phase_started_at = time.time()

        elif not np.array_equal(curr, BOARD):
            move_tuple = log_move(BOARD, curr, 'rglog.txt')
            if phase != 0 and not check_legal(BOARD, move_tuple):
                with open('rglog.txt', 'a') as f:
                    f.write("previous move was illegal\n")
                continue

            if phase == 2:
                # Time from start of OUR_HUMAN phase until board changes counts to our team.
                _deduct_team_time(time.time() - phase_started_at, "our human turn")

            BOARD=get_stable_board_state()
            t+=1

            phase_started_at = time.time()
            print(f"[PHASE] observed external move; next phase={_phase_name(t % 4)}")

        time.sleep(0.1)   # brief pause before sensing the next board state
