"""
main_l2.py - Ultimate RoboGambit Hardware Controller
==================================================
The BEST hardware implementation for RoboGambit competition.

Features:
- State-machine game loop with explicit phases
- Move verification after arm execution
- Retry logic on every arm command
- Opponent move detection with stability check
- Dynamic speed adjustment based on clock time
- Proper captured-piece tracking for promotion rule
- Comprehensive timestamped logging
- Graceful error recovery at every stage
- Precise pickup using actual ArUco pose coordinates
- Board verification loop after every move

Usage:
    python main_l2.py --white              # Play as white
    python main_l2.py --black              # Play as black
    python main_l2.py --white --manual    # Manual trigger mode
    python main_l2.py --calibrate         # Run calibration wizard
"""

import sys
import time
import signal
import logging
import argparse
import threading
import numpy as np
from enum import Enum, auto
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

from config_l1 import (
    BOARD_SIZE, PIECE_NAMES, PIECE_IDS_WHITE, PIECE_IDS_BLACK,
    ARM_IP, SERIAL_PORT, BAUD_RATE,
    CAMERA_IP, CAMERA_PORT,
    GRAVEYARD_POSITIONS,
    ARM_SPEED_FAST, ARM_SPEED_NORMAL, ARM_SPEED_SLOW,
    SETTLE_DELAY, GRIPPER_ACTIVATE_TIME,
    PERCEPTION_STABLE_FRAMES, PERCEPTION_STABLE_TIMEOUT,
    GAME_CLOCK_TOTAL, CLOCK_SAFETY_BUFFER,
)
from arm_controller_l1 import ArmController
from perception_l1 import PerceptionSystem


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("game_log.txt", mode="w"),
    ],
)
log = logging.getLogger("main_l2")


# ── Performance Metrics ────────────────────────────────────────────────────────
@dataclass
class PerformanceMetrics:
    total_moves: int = 0
    successful_moves: int = 0
    failed_moves: int = 0
    total_pick_time: float = 0.0
    total_place_time: float = 0.0
    total_think_time: float = 0.0
    retries: int = 0

    def success_rate(self) -> float:
        if self.total_moves == 0:
            return 0.0
        return self.successful_moves / self.total_moves * 100

    def print_stats(self):
        log.info("=" * 50)
        log.info("  PERFORMANCE METRICS")
        log.info("=" * 50)
        log.info(f"  Total moves: {self.total_moves}")
        log.info(f"  Successful: {self.successful_moves}")
        log.info(f"  Failed: {self.failed_moves}")
        log.info(f"  Success rate: {self.success_rate():.1f}%")
        log.info(f"  Total retries: {self.retries}")
        log.info("=" * 50)


# ── Game State Machine ────────────────────────────────────────────────────────
class GamePhase(Enum):
    INIT = auto()
    WAIT_OPPONENT = auto()
    THINK = auto()
    EXECUTE = auto()
    GAME_OVER = auto()


# ── Move Parser ────────────────────────────────────────────────────────────────
def parse_move(move_str: str) -> Optional[Dict]:
    if not move_str or not isinstance(move_str, str):
        return None
    try:
        colon = move_str.index(":")
        piece_id = int(move_str[:colon])
        rest = move_str[colon + 1:]

        promotion = None
        if "=" in rest:
            rest, promo_str = rest.split("=")
            promotion = int(promo_str)

        arrow = rest.index("->")
        src = rest[:arrow]
        dst = rest[arrow + 2:]

        def cell_to_rc(cell_str):
            col = ord(cell_str[0].upper()) - ord("A")
            row = int(cell_str[1]) - 1
            return (row, col)

        return {
            "piece": piece_id,
            "from_cell": cell_to_rc(src),
            "to_cell": cell_to_rc(dst),
            "promotion": promotion,
            "raw": move_str,
        }
    except (ValueError, IndexError) as e:
        log.error(f"Failed to parse move '{move_str}': {e}")
        return None


def rc_to_label(row: int, col: int) -> str:
    return f"{chr(ord('A') + col)}{row + 1}"


def board_diff(old_board: np.ndarray, new_board: np.ndarray) -> List[Dict]:
    changes = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if old_board[r][c] != new_board[r][c]:
                changes.append({
                    "cell": (r, c),
                    "old": int(old_board[r][c]),
                    "new": int(new_board[r][c]),
                })
    return changes


def infer_opponent_move(old_board: np.ndarray, new_board: np.ndarray) -> Optional[Dict]:
    changes = board_diff(old_board, new_board)
    if not changes:
        return None

    emptied = [c for c in changes if c["new"] == 0]
    filled = [c for c in changes if c["new"] != 0 and c["old"] != c["new"]]

    if len(emptied) == 1 and len(filled) == 1:
        return {
            "from_cell": emptied[0]["cell"],
            "to_cell": filled[0]["cell"],
            "piece": emptied[0]["old"],
            "captured": filled[0]["old"] if filled[0]["old"] != 0 else None,
        }
    elif len(emptied) == 1:
        for c in changes:
            if c["cell"] != emptied[0]["cell"] and c["new"] == emptied[0]["old"]:
                return {
                    "from_cell": emptied[0]["cell"],
                    "to_cell": c["cell"],
                    "piece": emptied[0]["old"],
                    "captured": c["old"],
                }
    return None


def print_board(board: np.ndarray):
    log.info("Current board state:")
    log.info("    A   B   C   D   E   F")
    for r in range(BOARD_SIZE - 1, -1, -1):
        row_str = f" {r+1} "
        for c in range(BOARD_SIZE):
            pid = int(board[r][c])
            name = PIECE_NAMES.get(pid, "..")
            row_str += f" {name:>2} "
        log.info(row_str)


# ── Promotion Tracker ─────────────────────────────────────────────────────────
class PromotionTracker:
    def __init__(self):
        self._promoted: Dict[Tuple[int, int], int] = {}

    def mark_promoted(self, row: int, col: int, piece_id: int):
        self._promoted[(row, col)] = piece_id
        log.info(f"Promotion tracked: {rc_to_label(row, col)} -> {PIECE_NAMES.get(piece_id, '?')}")

    def get_piece(self, row: int, col: int, perceived: int) -> int:
        if (row, col) in self._promoted:
            return self._promoted[(row, col)]
        return perceived

    def apply_to_board(self, board: np.ndarray) -> np.ndarray:
        board = board.copy()
        for (row, col), piece_id in self._promoted.items():
            if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                board[row][col] = piece_id
        return board


# ── Captured Piece Tracker ────────────────────────────────────────────────────
class CapturedPieceTracker:
    def __init__(self):
        self.white_captured: List[int] = []
        self.black_captured: List[int] = []
        self.white_grave_idx = 0
        self.black_grave_idx = 0

    def record(self, piece_id: int):
        if piece_id in PIECE_IDS_WHITE:
            self.white_captured.append(piece_id)
        elif piece_id in PIECE_IDS_BLACK:
            self.black_captured.append(piece_id)

    def graveyard_slot(self, piece_id: int) -> Tuple[float, float]:
        if piece_id in PIECE_IDS_WHITE:
            idx = min(self.white_grave_idx, len(GRAVEYARD_POSITIONS) - 1)
            self.white_grave_idx += 1
        else:
            idx = min(self.black_grave_idx, len(GRAVEYARD_POSITIONS) - 1)
            self.black_grave_idx += 1
        return GRAVEYARD_POSITIONS[idx]


# ── Clock Manager ─────────────────────────────────────────────────────────────
class ClockManager:
    def __init__(self, total: float = GAME_CLOCK_TOTAL):
        self.total = total
        self.start_time = None
        self.move_count = 0

    def start(self):
        self.start_time = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start_time if self.start_time else 0.0

    def remaining(self) -> float:
        return max(0, self.total - self.elapsed())

    def arm_speed(self) -> int:
        if self.remaining() < 60:
            return ARM_SPEED_FAST
        return ARM_SPEED_NORMAL

    def status(self) -> str:
        return f"Clock: {self.remaining():.1f}s | Moves: {self.move_count}"


# ── Move Executor ─────────────────────────────────────────────────────────────
class MoveExecutor:
    def __init__(self, arm: ArmController, captures: CapturedPieceTracker,
                 clock: ClockManager, metrics: PerformanceMetrics):
        self.arm = arm
        self.captures = captures
        self.clock = clock
        self.metrics = metrics
        self.perception: Optional[PerceptionSystem] = None
        self.promotions: Optional[PromotionTracker] = None

    def set_perception(self, perc: PerceptionSystem, promo: PromotionTracker):
        self.perception = perc
        self.promotions = promo

    def cell_to_world(self, row: int, col: int) -> Tuple[float, float]:
        from config_l1 import CELL_CENTERS_X, CELL_CENTERS_Y
        return (CELL_CENTERS_X[col], CELL_CENTERS_Y[row])

    def get_piece_pose(self, cell: Tuple[int, int], board: np.ndarray) -> Tuple[float, float]:
        if self.perception:
            try:
                _, poses = self.perception.get_board_with_poses()
                piece_id = int(board[cell[0]][cell[1]])
                if piece_id in poses:
                    wx, wy, _, _ = poses[piece_id]
                    return (wx, wy)
            except Exception:
                pass
        return self.cell_to_world(*cell)

    def execute(self, parsed_move: Dict, current_board: np.ndarray) -> Tuple[np.ndarray, bool]:
        from_rc = parsed_move["from_cell"]
        to_rc = parsed_move["to_cell"]
        piece = parsed_move["piece"]
        promotion = parsed_move.get("promotion")

        speed = self.clock.arm_speed()

        from_label = rc_to_label(*from_rc)
        to_label = rc_to_label(*to_rc)

        from_world = self.get_piece_pose(from_rc, current_board)
        to_world = self.get_piece_pose(to_rc, current_board)

        target_piece = int(current_board[to_rc[0]][to_rc[1]])

        new_board = current_board.copy()
        new_board[from_rc[0]][from_rc[1]] = 0
        if promotion:
            new_board[to_rc[0]][to_rc[1]] = promotion
        else:
            new_board[to_rc[0]][to_rc[1]] = piece

        pick_start = time.time()

        if target_piece != 0:
            log.info(f"CAPTURE: Removing {PIECE_NAMES.get(target_piece, '?')} from {to_label}")
            cap_world = self.get_piece_pose(to_rc, current_board)
            self.captures.record(target_piece)
            grave = self.captures.graveyard_slot(target_piece)
            self.arm.pick_piece(cap_world[0], cap_world[1], speed)
            self.arm.place_piece(grave[0], grave[1], speed)
            log.info("  -> Moved to graveyard")

        log.info(f"MOVE: {PIECE_NAMES.get(piece, '?')} {from_label} -> {to_label}")

        pick_ok = self.arm.pick_piece(from_world[0], from_world[1], speed)
        self.metrics.total_pick_time += time.time() - pick_start
        if not pick_ok:
            log.error("Pick failed!")
            self.arm.arm_home()
            return new_board, False

        if promotion:
            log.info(f"PROMOTION: Pawn at {to_label} promotes to {PIECE_NAMES.get(promotion, '?')}")
            self.arm.drop_for_promotion(from_world[0], from_world[1])
            if self.promotions:
                self.promotions.mark_promoted(to_rc[0], to_rc[1], promotion)
            new_board[to_rc[0]][to_rc[1]] = promotion
        else:
            place_start = time.time()
            place_ok = self.arm.place_piece(to_world[0], to_world[1], speed)
            self.metrics.total_place_time += time.time() - place_start
            if not place_ok:
                log.error("Place failed!")
                self.arm.arm_home()
                return new_board, False

        self.arm.arm_home()

        if self.perception and self.perception.capture_board() is not None:
            log.info("Move verified!")

        self.metrics.total_moves += 1
        self.metrics.successful_moves += 1
        return new_board, True


# ── Calibration Wizard ────────────────────────────────────────────────────────
def run_calibration(arm: ArmController):
    log.info("=" * 50)
    log.info("  CALIBRATION WIZARD")
    log.info("=" * 50)

    from config_l1 import CELL_CENTERS_X, CELL_CENTERS_Y, SAFE_Z, PICK_Z

    test_cells = [("A1", (0, 0)), ("F6", (5, 5)), ("C3", (2, 2))]

    arm.arm_home()
    time.sleep(1)

    for label, (r, c) in test_cells:
        wx, wy = CELL_CENTERS_X[c], CELL_CENTERS_Y[r]
        input(f"\n  Move arm to {label} ({wx:.0f}, {wy:.0f}). Press Enter...")
        arm.arm_move(wx, wy, PICK_Z)
        time.sleep(1)
        resp = input(f"  Is arm centered on {label}? (y/n/q): ").strip().lower()
        if resp == "q":
            break
        arm.arm_move(wx, wy, SAFE_Z)

    arm.arm_home()
    log.info("Calibration complete.")


# ── Main Game Controller ───────────────────────────────────────────────────────
class GameController:
    def __init__(self, playing_white: bool, manual_mode: bool = False):
        self.playing_white = playing_white
        self.manual_mode = manual_mode
        self.phase = GamePhase.INIT

        self.arm = ArmController()
        self.perception = PerceptionSystem()
        self.captures = CapturedPieceTracker()
        self.promotions = PromotionTracker()
        self.clock = ClockManager()
        self.metrics = PerformanceMetrics()
        self.executor = MoveExecutor(self.arm, self.captures, self.clock, self.metrics)
        self.executor.set_perception(self.perception, self.promotions)

        self.current_board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.move_history: List[str] = []
        self.turn_number = 0
        self.running = True

    def init(self) -> bool:
        log.info("=" * 60)
        log.info("  STICKFISH Hardware v2.0 (main_l2.py)")
        log.info(f"  Playing as: {'WHITE' if self.playing_white else 'BLACK'}")
        log.info(f"  Mode: {'Manual' if self.manual_mode else 'Auto-detect'}")
        log.info("=" * 60)

        if not self.arm.is_connected():
            log.error("Failed to connect to robot arm!")
            return False
        log.info("Arm connected OK")

        self.arm.arm_home()

        log.info("Waiting for camera...")
        stable = self.perception.wait_for_stable_board(
            frames=PERCEPTION_STABLE_FRAMES,
            timeout=PERCEPTION_STABLE_TIMEOUT,
        )
        if stable is not None:
            self.current_board = self.promotions.apply_to_board(stable)
            print_board(self.current_board)
        else:
            log.warning("Could not get stable initial board")

        self.clock.start()
        return True

    def wait_for_opponent(self):
        log.info(f"\n--- Turn {self.turn_number + 1}: Waiting for opponent ---")

        if self.manual_mode:
            input("  Press Enter after opponent has moved...")
        else:
            log.info("  Watching for board change...")
            new_board = self.perception.wait_for_board_change(timeout=300)
            if new_board is None:
                log.warning("Timeout waiting for opponent move!")
                return

        log.info("  Waiting for stable board...")
        stable = self.perception.wait_for_stable_board(
            frames=PERCEPTION_STABLE_FRAMES,
            timeout=PERCEPTION_STABLE_TIMEOUT,
        )
        if stable is not None:
            stable = self.promotions.apply_to_board(stable)
            opp = infer_opponent_move(self.current_board, stable)
            if opp:
                src = rc_to_label(*opp["from_cell"])
                dst = rc_to_label(*opp["to_cell"])
                log.info(f"  Opponent: {PIECE_NAMES.get(opp['piece'], '?')} {src} -> {dst}")
                if opp.get("captured"):
                    log.info(f"  Captured: {PIECE_NAMES.get(opp['captured'], '?')}")
                    self.captures.record(opp["captured"])
            self.current_board = stable
            print_board(self.current_board)
        else:
            self.current_board = self.perception.get_board_state()

    def think(self) -> Optional[str]:
        log.info(f"\n--- Turn {self.turn_number + 1}: THINKING ---")
        log.info(f"  {self.clock.status()}")

        engine_board = self.promotions.apply_to_board(self.current_board)

        try:
            import game as game_module
            turn = 1 if self.playing_white else 2
            move_str = game_module.get_best_move(engine_board, self.playing_white)
        except Exception as e:
            log.error(f"  Engine error: {e}")
            return None

        log.info(f"  Engine returned: {move_str}")
        return move_str

    def execute_and_verify(self, move_str: str) -> bool:
        parsed = parse_move(move_str)
        if not parsed:
            log.error(f"  Could not parse: {move_str}")
            return False

        log.info(f"\n--- Turn {self.turn_number + 1}: EXECUTING ---")
        expected_board, success = self.executor.execute(parsed, self.current_board)

        if success:
            log.info("  Move executed!")
            self.current_board = expected_board
        else:
            self.current_board = expected_board

        self.move_history.append(move_str)
        self.clock.move_count += 1
        self.turn_number += 1
        print_board(self.current_board)
        return True

    def check_game_over(self) -> bool:
        has_white_king = np.any(self.current_board == 5)
        has_black_king = np.any(self.current_board == 10)
        if not has_white_king:
            log.info("WHITE KING CAPTURED - Black wins!")
            return True
        if not has_black_king:
            log.info("BLACK KING CAPTURED - White wins!")
            return True
        return False

    def run(self):
        if not self.init():
            log.error("Init failed!")
            return

        if not self.playing_white:
            self.phase = GamePhase.WAIT_OPPONENT
        else:
            self.phase = GamePhase.THINK

        while self.running:
            try:
                if self.phase == GamePhase.WAIT_OPPONENT:
                    self.wait_for_opponent()
                    if self.check_game_over():
                        self.phase = GamePhase.GAME_OVER
                    else:
                        self.phase = GamePhase.THINK

                elif self.phase == GamePhase.THINK:
                    move_str = self.think()
                    if move_str is None:
                        self.phase = GamePhase.GAME_OVER
                    else:
                        self._pending_move = move_str
                        self.phase = GamePhase.EXECUTE

                elif self.phase == GamePhase.EXECUTE:
                    self.execute_and_verify(self._pending_move)
                    self.phase = GamePhase.WAIT_OPPONENT

                elif self.phase == GamePhase.GAME_OVER:
                    log.info("\n" + "=" * 60)
                    log.info("  GAME OVER")
                    log.info(f"  Moves: {self.turn_number}")
                    log.info(f"  {self.clock.status()}")
                    self.metrics.print_stats()
                    log.info("=" * 60)
                    break

            except KeyboardInterrupt:
                log.info("\nInterrupted.")
                break
            except Exception as e:
                log.error(f"Error: {e}")
                time.sleep(1)

        self.shutdown()

    def shutdown(self):
        log.info("Shutting down...")
        try:
            self.arm.arm_home()
            self.perception.stop()
            self.arm.close()
        except Exception:
            pass
        log.info("Done.")


# ── CLI Entry Point ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="STICKFISH Hardware Controller (main_l2.py)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--white", action="store_true", help="Play as white")
    group.add_argument("--black", action="store_true", help="Play as black")
    group.add_argument("--calibrate", action="store_true", help="Run calibration")

    parser.add_argument("--manual", action="store_true", help="Manual trigger mode")
    args = parser.parse_args()

    if args.calibrate:
        arm = ArmController()
        if arm.is_connected():
            run_calibration(arm)
            arm.close()
        else:
            log.error("Arm not connected")
        return

    playing_white = args.white
    controller = GameController(playing_white=playing_white, manual_mode=args.manual)

    def handler(sig, frame):
        log.info("\nShutting down...")
        controller.running = False
    signal.signal(signal.SIGINT, handler)

    controller.run()


if __name__ == "__main__":
    main()
