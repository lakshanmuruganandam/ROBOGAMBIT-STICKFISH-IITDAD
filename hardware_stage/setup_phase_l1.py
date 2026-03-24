"""
RoboGambit Hardware Setup Phase
================================
Optimized board setup script for the seeding phase of the RoboGambit hardware
tournament. Teams use a robotic arm to place pieces onto a 6x6 board as fast as
possible — completion time determines tournament bracket seeding.

Key optimization: nearest-neighbor greedy ordering of piece placements to
minimize total arm travel distance.

Usage:
    python setup_phase.py --config "2,3,4,5,3,2"

The --config argument specifies the back-rank piece order (Fischer Random) as
comma-separated piece IDs. Default is "2,3,4,5,3,2" (Knight, Bishop, Queen,
King, Bishop, Knight).
"""

import argparse
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from config_l1 import (
    BOARD_SIZE,
    BOARD_ORIGIN,
    CELL_SIZE,
    HOME_POSITION,
    STAGING_AREA_ORIGIN,
    STAGING_ROW_SPACING,
    STAGING_COL_SPACING,
    ARM_SPEED_FAST,
    ARM_SPEED_SAFE,
    PICKUP_HEIGHT,
    PLACE_HEIGHT,
    TRAVEL_HEIGHT,
)
from arm_controller_l1 import ArmController

# ---------------------------------------------------------------------------
# Piece ID constants (mirrors game_engine/game_state.py)
# ---------------------------------------------------------------------------
EMPTY = 0
W_PAWN = 1
W_KNIGHT = 2
W_BISHOP = 3
W_QUEEN = 4
W_KING = 5
B_PAWN = 6
B_KNIGHT = 7
B_BISHOP = 8
B_QUEEN = 9
B_KING = 10

PIECE_NAMES = {
    0: ".", 1: "P", 2: "N", 3: "B", 4: "Q", 5: "K",
    6: "p", 7: "n", 8: "b", 9: "q", 10: "k",
}

# ---------------------------------------------------------------------------
# Default staging-area positions
# ---------------------------------------------------------------------------
# Pieces wait in a staging area to the right of the board.  Layout:
#   Row 0  (closest to board): White pawns  (up to 6)
#   Row 1:                     White officers (N, B, Q, K — up to 6)
#   Row 2:                     Black pawns  (up to 6)
#   Row 3:                     Black officers
#
# Each slot is addressed as (row, col) within the staging grid; the real-world
# XYZ is computed from STAGING_AREA_ORIGIN + offsets.
# ---------------------------------------------------------------------------

def _build_staging_positions() -> Dict[int, Tuple[float, float, float]]:
    """Return a dict mapping piece_id -> (x, y, z) in the staging area.

    Multiple copies of the same piece type get consecutive columns in their
    row so they never overlap.
    """
    # Counts track how many of each piece have been assigned a slot so far.
    # Max counts per the 6x6 variant: 6 pawns, 2 knights, 2 bishops, 1 queen,
    # 1 king per side.
    piece_row = {
        W_PAWN: 0, W_KNIGHT: 1, W_BISHOP: 1, W_QUEEN: 1, W_KING: 1,
        B_PAWN: 2, B_KNIGHT: 3, B_BISHOP: 3, B_QUEEN: 3, B_KING: 3,
    }
    # Column order within the officer rows: N N B B Q K
    piece_col_order = {
        W_KNIGHT: 0, W_BISHOP: 2, W_QUEEN: 4, W_KING: 5,
        B_KNIGHT: 0, B_BISHOP: 2, B_QUEEN: 4, B_KING: 5,
    }

    ox, oy, oz = STAGING_AREA_ORIGIN
    positions: Dict[int, List[Tuple[float, float, float]]] = {}

    # --- Pawns: 6 per side, columns 0-5 in their row ---
    for side_pawn, row_idx in [(W_PAWN, 0), (B_PAWN, 2)]:
        positions[side_pawn] = []
        for col in range(6):
            x = ox + col * STAGING_COL_SPACING
            y = oy + row_idx * STAGING_ROW_SPACING
            positions[side_pawn].append((x, y, oz))

    # --- Officers: laid out in fixed column slots ---
    for piece_id, base_col in piece_col_order.items():
        row_idx = piece_row[piece_id]
        count = 2 if piece_id in (W_KNIGHT, W_BISHOP, B_KNIGHT, B_BISHOP) else 1
        positions[piece_id] = []
        for i in range(count):
            x = ox + (base_col + i) * STAGING_COL_SPACING
            y = oy + row_idx * STAGING_ROW_SPACING
            positions[piece_id].append((x, y, oz))

    return positions


DEFAULT_STAGING_POSITIONS: Dict[int, List[Tuple[float, float, float]]] = (
    _build_staging_positions()
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def board_cell_xyz(col: int, row: int) -> Tuple[float, float, float]:
    """Convert board (col, row) to real-world (x, y, z) at placement height."""
    ox, oy, oz = BOARD_ORIGIN
    x = ox + col * CELL_SIZE
    y = oy + row * CELL_SIZE
    return (x, y, oz)


def _distance(a: Tuple[float, float, float],
              b: Tuple[float, float, float]) -> float:
    """Euclidean distance between two 3-D points."""
    return math.sqrt((a[0] - b[0]) ** 2
                     + (a[1] - b[1]) ** 2
                     + (a[2] - b[2]) ** 2)


# ---------------------------------------------------------------------------
# SetupPhase
# ---------------------------------------------------------------------------

class SetupPhase:
    """Manages the timed board-setup phase for tournament seeding.

    Parameters
    ----------
    arm_controller : ArmController
        Initialised controller for the robotic arm.
    staging_positions : dict, optional
        Override the default staging-area layout.  Maps piece_id to a list of
        (x, y, z) positions (one per physical copy of that piece).
    """

    def __init__(
        self,
        arm_controller: ArmController,
        staging_positions: Optional[Dict[int, List[Tuple[float, float, float]]]] = None,
    ) -> None:
        self.arm = arm_controller
        self.piece_source_positions: Dict[int, List[Tuple[float, float, float]]] = (
            staging_positions if staging_positions is not None
            else dict(DEFAULT_STAGING_POSITIONS)
        )
        self.target_positions: Optional[np.ndarray] = None

        # Per-piece timing log: list of (piece_id, source, target, elapsed_s)
        self._placement_log: List[Tuple[int, Tuple, Tuple, float]] = []

    # ------------------------------------------------------------------
    # Placement-order optimisation (nearest-neighbour greedy)
    # ------------------------------------------------------------------

    @staticmethod
    def optimize_placement_order(
        pieces_to_place: List[Tuple[int, Tuple[float, float, float], Tuple[float, float, float]]],
        start_pos: Tuple[float, float, float] = HOME_POSITION,
    ) -> List[Tuple[int, Tuple[float, float, float], Tuple[float, float, float]]]:
        """Re-order placements to minimise total arm travel distance.

        Uses a nearest-neighbour greedy heuristic: starting from *start_pos*
        the arm always picks up the piece whose source position is closest to
        the arm's current location (which is the *target* of the previous
        placement, or home for the first move).

        Parameters
        ----------
        pieces_to_place : list of (piece_id, source_xyz, target_xyz)
        start_pos : arm starting position (default: HOME_POSITION from config)

        Returns
        -------
        list of (piece_id, source_xyz, target_xyz) in optimised order.
        """
        if len(pieces_to_place) <= 1:
            return list(pieces_to_place)

        remaining = list(pieces_to_place)
        ordered: List[Tuple[int, Tuple, Tuple]] = []
        current_pos = start_pos

        while remaining:
            # Find the piece whose *source* (pickup) position is nearest to
            # where the arm currently is.
            best_idx = 0
            best_dist = _distance(current_pos, remaining[0][1])
            for i in range(1, len(remaining)):
                d = _distance(current_pos, remaining[i][1])
                if d < best_dist:
                    best_dist = d
                    best_idx = i

            chosen = remaining.pop(best_idx)
            ordered.append(chosen)
            # After placing, the arm is at the target position.
            current_pos = chosen[2]

        return ordered

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def execute_setup(self, board_config: np.ndarray) -> float:
        """Place every piece onto the board as fast as possible.

        Parameters
        ----------
        board_config : np.ndarray
            6x6 target board (piece IDs, 0 = empty).

        Returns
        -------
        float
            Total elapsed time in seconds.
        """
        self.target_positions = board_config.copy()
        self._placement_log.clear()

        # --- Build the work list: (piece_id, source_xyz, target_xyz) ---
        # Track how many of each piece type we have consumed from staging.
        staging_index: Dict[int, int] = {}
        work_list: List[Tuple[int, Tuple, Tuple]] = []

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece_id = int(board_config[row, col])
                if piece_id == EMPTY:
                    continue

                idx = staging_index.get(piece_id, 0)
                sources = self.piece_source_positions.get(piece_id, [])
                if idx >= len(sources):
                    raise ValueError(
                        f"Not enough staged pieces for ID {piece_id} "
                        f"({PIECE_NAMES.get(piece_id, '?')}): need #{idx + 1} "
                        f"but only {len(sources)} available."
                    )
                source_xyz = sources[idx]
                staging_index[piece_id] = idx + 1

                target_xyz = board_cell_xyz(col, row)
                work_list.append((piece_id, source_xyz, target_xyz))

        # --- Optimise order ---
        work_list = self.optimize_placement_order(work_list)

        # --- Execute ---
        self.arm.set_speed(ARM_SPEED_FAST)
        total_start = time.perf_counter()

        for piece_id, source_xyz, target_xyz in work_list:
            t0 = time.perf_counter()
            self._pick_and_place(source_xyz, target_xyz)
            elapsed = time.perf_counter() - t0
            self._placement_log.append((piece_id, source_xyz, target_xyz, elapsed))
            print(
                f"  [{PIECE_NAMES.get(piece_id, '?')}] "
                f"staging -> ({target_xyz[0]:.1f}, {target_xyz[1]:.1f})  "
                f"{elapsed:.3f}s"
            )

        total_elapsed = time.perf_counter() - total_start

        # Return home
        self.arm.move_to(*HOME_POSITION, speed=ARM_SPEED_SAFE)

        print(f"\n  Pieces placed: {len(self._placement_log)}")
        print(f"  Total setup time: {total_elapsed:.3f}s")
        return total_elapsed

    def _pick_and_place(
        self,
        source: Tuple[float, float, float],
        target: Tuple[float, float, float],
    ) -> None:
        """Execute a single pick-and-place cycle at maximum safe speed."""
        sx, sy, sz = source
        tx, ty, tz = target

        # Raise to travel height
        self.arm.move_to(self.arm.x, self.arm.y, TRAVEL_HEIGHT)
        # Move above source
        self.arm.move_to(sx, sy, TRAVEL_HEIGHT)
        # Lower to pickup
        self.arm.move_to(sx, sy, PICKUP_HEIGHT)
        self.arm.grip()
        # Lift
        self.arm.move_to(sx, sy, TRAVEL_HEIGHT)
        # Move above target
        self.arm.move_to(tx, ty, TRAVEL_HEIGHT)
        # Lower to place
        self.arm.move_to(tx, ty, PLACE_HEIGHT)
        self.arm.release()
        # Lift clear of the piece
        self.arm.move_to(tx, ty, TRAVEL_HEIGHT)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    @staticmethod
    def verify_setup(
        expected_board: np.ndarray,
        perception,
    ) -> List[Tuple[int, int, int, int]]:
        """Compare the physical board against the expected configuration.

        Parameters
        ----------
        expected_board : np.ndarray (6x6)
        perception : object
            Must expose ``capture_board() -> np.ndarray`` returning a 6x6
            array of detected piece IDs.

        Returns
        -------
        list of (col, row, expected_piece, detected_piece)
            One entry per cell that does not match.
        """
        detected = perception.capture_board()
        misplaced: List[Tuple[int, int, int, int]] = []

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                exp = int(expected_board[row, col])
                det = int(detected[row, col])
                if exp != det:
                    misplaced.append((col, row, exp, det))

        if misplaced:
            print(f"\n  Verification: {len(misplaced)} misplaced piece(s)")
            for col, row, exp, det in misplaced:
                print(
                    f"    ({col},{row}): expected "
                    f"{PIECE_NAMES.get(exp, '?')} got "
                    f"{PIECE_NAMES.get(det, '?')}"
                )
        else:
            print("\n  Verification: all pieces correct.")

        return misplaced

    # ------------------------------------------------------------------
    # Fix misplacements
    # ------------------------------------------------------------------

    def fix_misplacements(
        self,
        misplaced: List[Tuple[int, int, int, int]],
    ) -> float:
        """Re-place pieces that were not placed correctly.

        For each misplaced cell the arm removes whatever is there (if any) and
        places the correct piece from staging.

        Parameters
        ----------
        misplaced : list of (col, row, expected_piece, detected_piece)

        Returns
        -------
        float
            Elapsed time for the fix pass.
        """
        if not misplaced:
            return 0.0

        staging_index: Dict[int, int] = {}
        # Count how many of each piece were already used in the main pass so
        # we pull from the correct staging slot.
        for piece_id, _, _, _ in self._placement_log:
            staging_index[piece_id] = staging_index.get(piece_id, 0) + 1

        work_list: List[Tuple[int, Tuple, Tuple]] = []
        for col, row, expected_piece, detected_piece in misplaced:
            if expected_piece == EMPTY:
                # Need to remove a piece; skip for now (no discard bin defined).
                continue

            idx = staging_index.get(expected_piece, 0)
            sources = self.piece_source_positions.get(expected_piece, [])
            if idx < len(sources):
                source_xyz = sources[idx]
                staging_index[expected_piece] = idx + 1
            else:
                # Fallback: try the cell the piece should have come from
                # originally (it may still be sitting in the wrong cell).
                print(
                    f"  Warning: no spare staging slot for "
                    f"{PIECE_NAMES.get(expected_piece, '?')} — skipping."
                )
                continue

            target_xyz = board_cell_xyz(col, row)
            work_list.append((expected_piece, source_xyz, target_xyz))

        work_list = self.optimize_placement_order(work_list)
        self.arm.set_speed(ARM_SPEED_SAFE)

        t0 = time.perf_counter()
        for piece_id, source_xyz, target_xyz in work_list:
            self._pick_and_place(source_xyz, target_xyz)
            self._placement_log.append(
                (piece_id, source_xyz, target_xyz, time.perf_counter() - t0)
            )
        elapsed = time.perf_counter() - t0

        self.arm.move_to(*HOME_POSITION, speed=ARM_SPEED_SAFE)
        print(f"  Fix pass: {len(work_list)} piece(s) re-placed in {elapsed:.3f}s")
        return elapsed

    # ------------------------------------------------------------------
    # Full timed run
    # ------------------------------------------------------------------

    def timed_run(
        self,
        board_config: np.ndarray,
        perception=None,
    ) -> float:
        """Execute a complete timed setup with optional verification and fix.

        Parameters
        ----------
        board_config : np.ndarray (6x6)
        perception : optional
            If provided, the board is verified after placement and any
            misplaced pieces are corrected.

        Returns
        -------
        float
            Total wall-clock time (placement + any fix passes).
        """
        print("=" * 52)
        print("  ROBOGAMBIT HARDWARE — SETUP PHASE")
        print("=" * 52)

        run_start = time.perf_counter()

        # --- Primary placement ---
        print("\n--- Placement pass ---")
        self.execute_setup(board_config)

        # --- Verification & fix ---
        if perception is not None:
            print("\n--- Verification ---")
            misplaced = self.verify_setup(board_config, perception)
            if misplaced:
                print("\n--- Fix pass ---")
                self.fix_misplacements(misplaced)
                # Re-verify
                print("\n--- Re-verification ---")
                remaining = self.verify_setup(board_config, perception)
                if remaining:
                    print(
                        f"  WARNING: {len(remaining)} piece(s) still wrong "
                        f"after fix pass."
                    )

        total = time.perf_counter() - run_start

        print("\n" + "=" * 52)
        print(f"  TOTAL SETUP TIME:  {total:.3f} s")
        print("=" * 52)

        return total


# ---------------------------------------------------------------------------
# Board-config builders (Fischer Random support)
# ---------------------------------------------------------------------------

def build_board_from_back_rank(
    back_rank_ids: List[int],
) -> np.ndarray:
    """Construct a full 6x6 starting board from a Fischer Random back rank.

    Parameters
    ----------
    back_rank_ids : list of int
        Six piece IDs for the white back rank (rank 1), e.g.
        [W_KNIGHT, W_BISHOP, W_QUEEN, W_KING, W_BISHOP, W_KNIGHT].
        The black back rank is mirrored automatically (+5 to each ID).

    Returns
    -------
    np.ndarray  (6, 6) int32
    """
    if len(back_rank_ids) != BOARD_SIZE:
        raise ValueError(
            f"Back rank must have exactly {BOARD_SIZE} pieces, "
            f"got {len(back_rank_ids)}."
        )

    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)

    # White back rank (row 0) and pawns (row 1)
    for col, pid in enumerate(back_rank_ids):
        board[0, col] = pid
    board[1, :] = W_PAWN

    # Black pawns (row 4) and back rank (row 5)
    board[4, :] = B_PAWN
    for col, pid in enumerate(back_rank_ids):
        board[5, col] = pid + 5  # white -> black offset

    return board


def parse_back_rank(config_str: str) -> List[int]:
    """Parse a comma-separated back-rank config string into piece IDs.

    Example: "2,3,4,5,3,2" -> [W_KNIGHT, W_BISHOP, W_QUEEN, W_KING,
                                 W_BISHOP, W_KNIGHT]
    """
    ids = [int(x.strip()) for x in config_str.split(",")]
    if len(ids) != BOARD_SIZE:
        raise ValueError(
            f"Expected {BOARD_SIZE} piece IDs, got {len(ids)}."
        )
    for pid in ids:
        if pid not in (W_KNIGHT, W_BISHOP, W_QUEEN, W_KING):
            raise ValueError(
                f"Invalid back-rank piece ID {pid}. "
                f"Allowed: 2 (N), 3 (B), 4 (Q), 5 (K)."
            )
    return ids


def print_board(board: np.ndarray) -> None:
    """Pretty-print a 6x6 board to the console."""
    print("    A  B  C  D  E  F")
    print("  +" + "--+" * BOARD_SIZE)
    for row in range(BOARD_SIZE - 1, -1, -1):
        rank = row + 1
        cells = "  ".join(
            PIECE_NAMES.get(int(board[row, col]), "?")
            for col in range(BOARD_SIZE)
        )
        print(f"{rank} | {cells} |")
    print("  +" + "--+" * BOARD_SIZE)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RoboGambit hardware setup-phase runner.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="2,3,4,5,3,2",
        help=(
            "Comma-separated white back-rank piece IDs "
            "(2=N, 3=B, 4=Q, 5=K). Default: 2,3,4,5,3,2"
        ),
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run perception verification after placement.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the board and optimised order without moving the arm.",
    )
    args = parser.parse_args()

    back_rank = parse_back_rank(args.config)
    board = build_board_from_back_rank(back_rank)

    print("\nTarget board configuration:")
    print_board(board)

    if args.dry_run:
        # Show optimised order without hardware
        staging = DEFAULT_STAGING_POSITIONS
        staging_index: Dict[int, int] = {}
        work_list = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                pid = int(board[row, col])
                if pid == EMPTY:
                    continue
                idx = staging_index.get(pid, 0)
                sources = staging.get(pid, [])
                if idx < len(sources):
                    src = sources[idx]
                else:
                    src = (0.0, 0.0, 0.0)
                staging_index[pid] = idx + 1
                work_list.append((pid, src, board_cell_xyz(col, row)))

        ordered = SetupPhase.optimize_placement_order(work_list)
        print("\nOptimised placement order:")
        for i, (pid, src, tgt) in enumerate(ordered, 1):
            print(
                f"  {i:2d}. {PIECE_NAMES.get(pid, '?')}  "
                f"staging({src[0]:.1f},{src[1]:.1f}) -> "
                f"board({tgt[0]:.1f},{tgt[1]:.1f})"
            )

        total_dist = 0.0
        pos = HOME_POSITION
        for _, src, tgt in ordered:
            total_dist += _distance(pos, src) + _distance(src, tgt)
            pos = tgt
        total_dist += _distance(pos, HOME_POSITION)
        print(f"\n  Estimated total arm travel: {total_dist:.1f} units")
        return

    # --- Live run ---
    arm = ArmController()
    arm.initialize()

    setup = SetupPhase(arm)
    perception = None
    if args.verify:
        # Lazy import — perception may not be available on all setups.
        from perception import BoardPerception  # type: ignore[import-not-found]
        perception = BoardPerception()

    setup.timed_run(board, perception=perception)

    arm.shutdown()


if __name__ == "__main__":
    main()
