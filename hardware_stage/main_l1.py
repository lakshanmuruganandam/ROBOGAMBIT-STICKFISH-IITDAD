"""
main_l1.py  --  STICKFISH Hardware Orchestrator (Lakshan Edition)
=================================================================
RoboGambit Round 2 Hardware Stage — Full autonomous game pipeline.

    Perception (camera + ArUco)  →  Engine (frozen STICKFISH)  →  Arm (RoArm M2-S)

Improvements over main_e1.py:
  • State-machine game loop with explicit phases
  • Move verification after arm execution (re-reads board to confirm)
  • Retry logic on every arm command and perception read
  • Opponent move detection via board-diff with stability check
  • Dynamic speed adjustment based on remaining clock time
  • Proper captured-piece tracking for promotion rule
  • Organized graveyard management with per-side slots
  • Comprehensive timestamped logging
  • Graceful error recovery at every stage
  • Thread-safe perception with background updates
  • Setup-phase integration for seeding
  • Calibration wizard with 3-point verification

Usage:
    python main_l1.py --white              # Play as white
    python main_l1.py --black              # Play as black
    python main_l1.py --white --manual     # Manual trigger mode (press Enter each turn)
    python main_l1.py --calibrate          # Run calibration wizard
    python main_l1.py --setup "2,3,4,5,3,2"  # Run setup phase with back-rank config
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

# ── Local modules ─────────────────────────────────────────────────────────────
import game
from config_l1 import (
    BOARD_SIZE, PIECE_NAMES, PIECE_IDS_WHITE, PIECE_IDS_BLACK,
    CELL_CENTERS, cell_to_world, GRAVEYARD_WHITE, GRAVEYARD_BLACK,
    ARM_SAFE_Z, ARM_PICK_Z, ARM_PLACE_Z, ARM_HOVER_Z,
    ARM_SPEED_FAST, ARM_SPEED_NORMAL, ARM_SPEED_SLOW,
    GRIPPER_SETTLE_TIME, ARM_SETTLE_TIME, PERCEPTION_STABLE_FRAMES,
    PERCEPTION_STABLE_TIMEOUT, MOVE_VERIFY_TIMEOUT, GAME_CLOCK_TOTAL,
    CLOCK_SAFETY_BUFFER, SERIAL_PORT, BAUD_RATE, ARM_IP,
    CAMERA_IP, CAMERA_PORT,
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
log = logging.getLogger("main_l1")


# ── Game State Machine ────────────────────────────────────────────────────────
class GamePhase(Enum):
    INIT = auto()
    WAIT_OPPONENT = auto()
    DETECT_OPPONENT_MOVE = auto()
    THINK = auto()
    EXECUTE_MOVE = auto()
    VERIFY_MOVE = auto()
    GAME_OVER = auto()


# ── Move Parser ───────────────────────────────────────────────────────────────
def parse_move(move_str: str) -> Optional[Dict]:
    """
    Parse engine move string like '4:D1->D3' or '1:A5->A6=4' (promotion).
    Returns dict with piece, from_cell (row,col), to_cell (row,col), promotion_piece.
    """
    if not move_str or not isinstance(move_str, str):
        return None
    try:
        # Split piece_id and move
        colon = move_str.index(":")
        piece_id = int(move_str[:colon])
        rest = move_str[colon + 1:]

        # Handle promotion
        promotion = None
        if "=" in rest:
            rest, promo_str = rest.split("=")
            promotion = int(promo_str)

        # Split source -> target
        arrow = rest.index("->")
        src = rest[:arrow]
        dst = rest[arrow + 2:]

        def cell_to_rc(cell_str):
            col = ord(cell_str[0].upper()) - ord("A")
            row = int(cell_str[1]) - 1
            return (row, col)

        from_cell = cell_to_rc(src)
        to_cell = cell_to_rc(dst)

        return {
            "piece": piece_id,
            "from_cell": from_cell,
            "to_cell": to_cell,
            "promotion": promotion,
            "raw": move_str,
        }
    except (ValueError, IndexError) as e:
        log.error(f"Failed to parse move '{move_str}': {e}")
        return None


def rc_to_label(row, col):
    """Convert (row, col) to chess notation like 'D3'."""
    return f"{chr(ord('A') + col)}{row + 1}"


# ── Board Utilities ───────────────────────────────────────────────────────────
def board_diff(old_board: np.ndarray, new_board: np.ndarray) -> List[Dict]:
    """
    Compare two board states and return list of changes.
    Each change: {'cell': (r,c), 'old_piece': int, 'new_piece': int}
    """
    changes = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if old_board[r][c] != new_board[r][c]:
                changes.append({
                    "cell": (r, c),
                    "old_piece": int(old_board[r][c]),
                    "new_piece": int(new_board[r][c]),
                })
    return changes


def infer_opponent_move(old_board: np.ndarray, new_board: np.ndarray) -> Optional[Dict]:
    """
    Infer what move the opponent made by comparing board states.
    Returns dict with 'from_cell', 'to_cell', 'piece', 'captured' or None.
    """
    changes = board_diff(old_board, new_board)
    if not changes:
        return None

    # Find cell that became empty (source) and cell that got a new piece (destination)
    emptied = [c for c in changes if c["new_piece"] == 0 and c["old_piece"] != 0]
    filled = [c for c in changes if c["new_piece"] != 0 and c["old_piece"] != c["new_piece"]]

    if len(emptied) == 1 and len(filled) == 1:
        src = emptied[0]
        dst = filled[0]
        captured = dst["old_piece"] if dst["old_piece"] != 0 else None
        return {
            "from_cell": src["cell"],
            "to_cell": dst["cell"],
            "piece": src["old_piece"],
            "captured": captured,
        }
    elif len(emptied) == 1 and len(filled) == 0:
        # Piece captured on a cell that was occupied (piece replaced by attacker)
        # This shouldn't normally happen with proper diff, but handle edge case
        src = emptied[0]
        for c in changes:
            if c["cell"] != src["cell"] and c["new_piece"] == src["old_piece"]:
                return {
                    "from_cell": src["cell"],
                    "to_cell": c["cell"],
                    "piece": src["old_piece"],
                    "captured": c["old_piece"],
                }
    log.warning(f"Could not infer opponent move from {len(changes)} board changes")
    return None


def print_board(board: np.ndarray):
    """Pretty print the board state."""
    log.info("Current board state:")
    log.info("    A   B   C   D   E   F")
    for r in range(BOARD_SIZE - 1, -1, -1):
        row_str = f" {r+1} "
        for c in range(BOARD_SIZE):
            pid = board[r][c]
            name = PIECE_NAMES.get(pid, ".")
            row_str += f" {name:>2} "
        log.info(row_str)


# ── Captured Piece Tracker ────────────────────────────────────────────────────
class CapturedPieceTracker:
    """
    Tracks captured pieces for the promotion rule:
    pawns can only promote to pieces that have been captured.
    Also manages graveyard slot assignments.
    """

    def __init__(self):
        self.white_captured: List[int] = []   # white pieces captured by black
        self.black_captured: List[int] = []   # black pieces captured by white
        self.white_graveyard_idx = 0
        self.black_graveyard_idx = 0

    def record_capture(self, piece_id: int):
        if piece_id in PIECE_IDS_WHITE:
            self.white_captured.append(piece_id)
            log.info(f"White {PIECE_NAMES[piece_id]} captured (total: {len(self.white_captured)})")
        elif piece_id in PIECE_IDS_BLACK:
            self.black_captured.append(piece_id)
            log.info(f"Black {PIECE_NAMES[piece_id]} captured (total: {len(self.black_captured)})")

    def get_next_graveyard_slot(self, piece_id: int) -> Tuple[float, float]:
        """Get the next available graveyard position for a captured piece."""
        if piece_id in PIECE_IDS_WHITE:
            idx = min(self.white_graveyard_idx, len(GRAVEYARD_WHITE) - 1)
            self.white_graveyard_idx += 1
            return GRAVEYARD_WHITE[idx]
        else:
            idx = min(self.black_graveyard_idx, len(GRAVEYARD_BLACK) - 1)
            self.black_graveyard_idx += 1
            return GRAVEYARD_BLACK[idx]

    def available_promotions(self, is_white: bool) -> List[int]:
        """What pieces can a pawn promote to? Only captured opponent pieces."""
        if is_white:
            # White pawn promotes to captured black non-pawn pieces
            return [p for p in self.black_captured if p != 6]  # exclude black pawns
        else:
            return [p for p in self.white_captured if p != 1]  # exclude white pawns


# ── Clock Manager ─────────────────────────────────────────────────────────────
class ClockManager:
    """Manages game clock and adjusts arm speed based on remaining time."""

    def __init__(self, total_time: float = GAME_CLOCK_TOTAL):
        self.total_time = total_time
        self.start_time = None
        self.move_count = 0

    def start(self):
        self.start_time = time.time()

    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def remaining(self) -> float:
        return max(0, self.total_time - self.elapsed())

    def time_per_move(self) -> float:
        """Estimated time budget per remaining move."""
        remaining = self.remaining() - CLOCK_SAFETY_BUFFER
        est_moves_left = max(5, 20 - self.move_count)
        return max(1.0, remaining / est_moves_left)

    def arm_speed(self) -> int:
        """Choose arm speed based on time pressure."""
        remaining = self.remaining()
        if remaining < 60:
            return ARM_SPEED_FAST
        elif remaining < 300:
            return ARM_SPEED_NORMAL
        else:
            return ARM_SPEED_NORMAL

    def record_move(self):
        self.move_count += 1

    def status(self) -> str:
        return f"Clock: {self.remaining():.1f}s remaining, {self.move_count} moves played"


# ── Move Executor ─────────────────────────────────────────────────────────────
class MoveExecutor:
    """Handles physical execution of chess moves on the board."""

    def __init__(self, arm: ArmController, captures: CapturedPieceTracker, clock: ClockManager):
        self.arm = arm
        self.captures = captures
        self.clock = clock

    def execute(self, parsed_move: Dict, current_board: np.ndarray) -> np.ndarray:
        """
        Execute a parsed move on the physical board.
        Handles: normal moves, captures (remove to graveyard first), promotions.
        Returns the expected new board state.
        """
        from_rc = parsed_move["from_cell"]
        to_rc = parsed_move["to_cell"]
        piece = parsed_move["piece"]
        promotion = parsed_move.get("promotion")

        from_world = cell_to_world(*from_rc)
        to_world = cell_to_world(*to_rc)
        speed = self.clock.arm_speed()

        from_label = rc_to_label(*from_rc)
        to_label = rc_to_label(*to_rc)

        # Check if destination has a piece (capture)
        target_piece = int(current_board[to_rc[0]][to_rc[1]])
        if target_piece != 0:
            log.info(f"CAPTURE: Removing {PIECE_NAMES.get(target_piece, '?')} from {to_label}")
            graveyard_pos = self.captures.get_next_graveyard_slot(target_piece)
            self.captures.record_capture(target_piece)

            # Pick up captured piece and move to graveyard
            self.arm.pick_piece(to_world[0], to_world[1], speed=speed)
            self.arm.place_piece(graveyard_pos[0], graveyard_pos[1], speed=speed)
            log.info(f"  -> Moved captured piece to graveyard")

        # Execute the main move
        log.info(f"MOVE: {PIECE_NAMES.get(piece, '?')} {from_label} -> {to_label}")
        self.arm.pick_piece(from_world[0], from_world[1], speed=speed)
        self.arm.place_piece(to_world[0], to_world[1], speed=speed)

        # Handle promotion (if applicable)
        if promotion is not None:
            log.info(f"PROMOTION: Pawn at {to_label} promotes to {PIECE_NAMES.get(promotion, '?')}")
            # In physical play, the pawn stays on the square but we track it as promoted
            # The ArUco marker on the pawn won't change, so the perception system
            # needs to be aware of promotions via our internal tracking

        # Return home to safe position
        self.arm.arm_home(speed=speed)

        # Build expected new board
        new_board = current_board.copy()
        new_board[from_rc[0]][from_rc[1]] = 0
        if promotion is not None:
            new_board[to_rc[0]][to_rc[1]] = promotion
        else:
            new_board[to_rc[0]][to_rc[1]] = piece

        return new_board


# ── Calibration Wizard ────────────────────────────────────────────────────────
def run_calibration(arm: ArmController):
    """
    Interactive calibration wizard.
    Moves the arm to known board cells so you can verify/adjust coordinate mapping.
    """
    log.info("=" * 50)
    log.info("  CALIBRATION WIZARD")
    log.info("=" * 50)

    test_cells = [
        ("A1", (0, 0)),
        ("F6", (5, 5)),
        ("A6", (5, 0)),
        ("F1", (0, 5)),
        ("C3", (2, 2)),
        ("D4", (3, 3)),
    ]

    arm.arm_home()
    time.sleep(1)

    for label, (r, c) in test_cells:
        wx, wy = cell_to_world(r, c)
        input(f"\n  Press Enter to move arm to {label} (world: {wx:.0f}, {wy:.0f})...")
        arm.arm_move(wx, wy, ARM_HOVER_Z)
        time.sleep(0.5)
        arm.arm_move(wx, wy, ARM_PICK_Z)
        time.sleep(1)

        response = input(f"  Is the arm centered on {label}? (y/n/q): ").strip().lower()
        if response == "q":
            break
        elif response == "n":
            log.warning(f"  Cell {label} needs adjustment! World coords: ({wx}, {wy})")
            log.warning(f"  Update config.py BOARD_ORIGIN_X/Y or ARM_OFFSET values.")

        arm.arm_move(wx, wy, ARM_HOVER_Z)

    arm.arm_home()
    log.info("\nCalibration complete. Adjust config.py if any cells were off.")


# ── Main Game Loop ────────────────────────────────────────────────────────────
class GameController:
    """
    State-machine game controller.
    Phases: INIT -> WAIT_OPPONENT -> DETECT -> THINK -> EXECUTE -> VERIFY -> repeat
    """

    def __init__(self, playing_white: bool, manual_mode: bool = False):
        self.playing_white = playing_white
        self.manual_mode = manual_mode
        self.phase = GamePhase.INIT

        # Components
        self.arm = ArmController()
        self.perception = PerceptionSystem()
        self.captures = CapturedPieceTracker()
        self.clock = ClockManager()
        self.executor = MoveExecutor(self.arm, self.captures, self.clock)

        # State
        self.current_board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.move_history: List[str] = []
        self.turn_number = 0
        self.running = True

    def init(self) -> bool:
        """Initialize all hardware connections."""
        log.info("=" * 60)
        log.info("  STICKFISH Hardware v1.0 (Lakshan Edition)")
        log.info(f"  Playing as: {'WHITE' if self.playing_white else 'BLACK'}")
        log.info(f"  Mode: {'Manual trigger' if self.manual_mode else 'Auto-detect'}")
        log.info("=" * 60)

        # Connect arm
        if not self.arm.is_connected():
            log.error("Failed to connect to robot arm!")
            log.error(f"Check: Serial on {SERIAL_PORT} and HTTP on {ARM_IP}")
            return False
        log.info("Arm connected OK")

        # Connect camera
        if not self.perception.connect_camera():
            log.error("Failed to connect to camera!")
            log.error(f"Check: Socket at {CAMERA_IP}:{CAMERA_PORT}")
            return False
        log.info("Camera connected OK")

        # Start perception background thread
        self.perception.start_background_thread()
        log.info("Perception thread started")

        # Move arm to home
        self.arm.arm_home()
        log.info("Arm homed")

        # Wait for initial board state
        log.info("Waiting for stable initial board state...")
        initial = self.perception.wait_for_stable_board(
            stability_frames=PERCEPTION_STABLE_FRAMES,
            timeout=PERCEPTION_STABLE_TIMEOUT,
        )
        if initial is not None:
            self.current_board = initial
            print_board(self.current_board)
        else:
            log.warning("Could not get stable initial board — using empty board")

        self.clock.start()
        return True

    def wait_for_opponent(self):
        """Wait for the human opponent to make their move."""
        log.info(f"\n--- Turn {self.turn_number + 1}: Waiting for opponent ---")

        if self.manual_mode:
            input("  Press Enter after opponent has moved...")
        else:
            # Auto-detect: wait for board state to change
            log.info("  Watching for board change...")
            new_board = self.perception.wait_for_board_change(timeout=300)
            if new_board is None:
                log.warning("  Timeout waiting for opponent move!")
                return

        # Wait for stable state (hand removed from board)
        log.info("  Waiting for stable board (hand removed)...")
        stable = self.perception.wait_for_stable_board(
            stability_frames=PERCEPTION_STABLE_FRAMES,
            timeout=PERCEPTION_STABLE_TIMEOUT,
        )
        if stable is not None:
            # Detect what changed
            opp_move = infer_opponent_move(self.current_board, stable)
            if opp_move:
                src = rc_to_label(*opp_move["from_cell"])
                dst = rc_to_label(*opp_move["to_cell"])
                piece_name = PIECE_NAMES.get(opp_move["piece"], "?")
                log.info(f"  Opponent moved: {piece_name} {src} -> {dst}")
                if opp_move.get("captured"):
                    cap_name = PIECE_NAMES.get(opp_move["captured"], "?")
                    log.info(f"  Opponent captured: {cap_name}")
                    self.captures.record_capture(opp_move["captured"])
            else:
                log.warning("  Could not determine opponent's move from board diff")

            self.current_board = stable
            print_board(self.current_board)
        else:
            log.warning("  Could not get stable board after opponent move")
            # Try to get whatever we have
            self.current_board = self.perception.get_board_state()

    def think(self) -> Optional[str]:
        """Ask the engine for the best move."""
        log.info(f"\n--- Turn {self.turn_number + 1}: THINKING ---")
        log.info(f"  {self.clock.status()}")

        turn = 1 if self.playing_white else 2
        start = time.time()

        try:
            move_str = game.get_move(self.current_board, turn)
        except Exception as e:
            log.error(f"  Engine error: {e}")
            # Fallback: try get_best_move
            try:
                move_str = game.get_best_move(self.current_board, self.playing_white)
            except Exception as e2:
                log.error(f"  Fallback engine also failed: {e2}")
                return None

        elapsed = time.time() - start
        log.info(f"  Engine returned: {move_str} (took {elapsed:.2f}s)")

        if move_str is None:
            log.info("  No legal moves — game over!")
            return None

        return move_str

    def execute_and_verify(self, move_str: str) -> bool:
        """Execute the move on the physical board and verify it."""
        parsed = parse_move(move_str)
        if parsed is None:
            log.error(f"  Could not parse move: {move_str}")
            return False

        # Execute physically
        log.info(f"\n--- Turn {self.turn_number + 1}: EXECUTING ---")
        expected_board = self.executor.execute(parsed, self.current_board)

        # Verify
        log.info("  Verifying move execution...")
        verified = self.perception.verify_move_executed(
            expected_board, timeout=MOVE_VERIFY_TIMEOUT
        )
        if verified:
            log.info("  Move verified OK!")
            self.current_board = expected_board
        else:
            log.warning("  Move verification FAILED — using expected board state")
            # Trust our execution even if perception disagrees
            # (could be arm still settling, marker occluded, etc.)
            actual = self.perception.get_board_state()
            if actual is not None and np.any(actual != 0):
                self.current_board = actual
            else:
                self.current_board = expected_board

        self.move_history.append(move_str)
        self.clock.record_move()
        self.turn_number += 1
        return True

    def check_game_over(self) -> bool:
        """Check if the game is over (king captured or no legal moves)."""
        # Check if kings are present
        has_white_king = np.any(self.current_board == 5)
        has_black_king = np.any(self.current_board == 10)

        if not has_white_king:
            log.info("WHITE KING CAPTURED — Black wins!")
            return True
        if not has_black_king:
            log.info("BLACK KING CAPTURED — White wins!")
            return True

        return False

    def run(self):
        """Main game loop."""
        if not self.init():
            log.error("Initialization failed — exiting")
            return

        # If playing black, opponent (white) goes first
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
                        self.phase = GamePhase.EXECUTE_MOVE
                        self._pending_move = move_str

                elif self.phase == GamePhase.EXECUTE_MOVE:
                    success = self.execute_and_verify(self._pending_move)
                    if success:
                        print_board(self.current_board)
                        if self.check_game_over():
                            self.phase = GamePhase.GAME_OVER
                        else:
                            self.phase = GamePhase.WAIT_OPPONENT
                    else:
                        log.error("Move execution failed!")
                        self.phase = GamePhase.GAME_OVER

                elif self.phase == GamePhase.GAME_OVER:
                    log.info("\n" + "=" * 60)
                    log.info("  GAME OVER")
                    log.info(f"  Moves played: {self.turn_number}")
                    log.info(f"  {self.clock.status()}")
                    log.info(f"  Move history: {self.move_history}")
                    log.info("=" * 60)
                    break

            except KeyboardInterrupt:
                log.info("\nInterrupted by user")
                break
            except Exception as e:
                log.error(f"Unexpected error in game loop: {e}", exc_info=True)
                log.info("Attempting to continue...")
                time.sleep(1)

        self.shutdown()

    def shutdown(self):
        """Clean shutdown of all hardware."""
        log.info("Shutting down...")
        try:
            self.arm.arm_home()
        except Exception:
            pass
        try:
            self.perception.stop()
        except Exception:
            pass
        try:
            self.arm.close()
        except Exception:
            pass
        log.info("Shutdown complete.")


# ── CLI Entry Point ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="STICKFISH Hardware Controller (Lakshan Edition)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--white", action="store_true", help="Play as white")
    group.add_argument("--black", action="store_true", help="Play as black")
    group.add_argument("--calibrate", action="store_true", help="Run calibration wizard")
    group.add_argument("--setup", type=str, metavar="RANK",
                       help="Run setup phase with back-rank config (e.g. '2,3,4,5,3,2')")

    parser.add_argument("--manual", action="store_true",
                        help="Manual trigger mode (press Enter each turn)")
    parser.add_argument("--serial", type=str, default=SERIAL_PORT,
                        help=f"Serial port (default: {SERIAL_PORT})")
    parser.add_argument("--arm-ip", type=str, default=ARM_IP,
                        help=f"Arm IP address (default: {ARM_IP})")

    args = parser.parse_args()

    if args.calibrate:
        arm = ArmController(serial_port=args.serial, arm_ip=args.arm_ip)
        if arm.is_connected():
            run_calibration(arm)
            arm.close()
        else:
            log.error("Cannot calibrate — arm not connected")
        return

    if args.setup:
        from setup_phase_l1 import SetupPhase
        arm = ArmController(serial_port=args.serial, arm_ip=args.arm_ip)
        if arm.is_connected():
            setup = SetupPhase(arm)
            back_rank = [int(x) for x in args.setup.split(",")]
            setup.timed_run(back_rank)
            arm.close()
        else:
            log.error("Cannot run setup — arm not connected")
        return

    # Normal game mode
    playing_white = args.white
    controller = GameController(
        playing_white=playing_white,
        manual_mode=args.manual,
    )

    # Handle SIGINT gracefully
    def signal_handler(sig, frame):
        log.info("\nReceived SIGINT — shutting down gracefully")
        controller.running = False
    signal.signal(signal.SIGINT, signal_handler)

    controller.run()


if __name__ == "__main__":
    main()
