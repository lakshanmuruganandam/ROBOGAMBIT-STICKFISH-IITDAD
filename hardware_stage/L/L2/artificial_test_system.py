"""
Artificial testing system for Team L2 hardware stack.

This script validates core logic without real camera or arm hardware.
It tests:
- Move parsing and validation
- Board update logic (normal, capture, promotion)
- Move execution sequence using a simulated arm
- One mocked run-loop cycle with simulated perception and game move

Run:
  python artificial_test_system.py
  python artificial_test_system.py --verbose
"""

from __future__ import annotations

import argparse
import traceback
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

import main


@dataclass
class TestResult:
    name: str
    passed: bool
    details: str = ""


class SimArm:
    """Minimal arm simulator implementing the methods used by main.py."""

    def __init__(self) -> None:
        self.moves: List[Tuple[float, float, float, int]] = []
        self.pick_calls = 0
        self.place_calls = 0
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def arm_home(self) -> bool:
        return True

    def move_xyz(self, x: float, y: float, z: float, speed: int) -> bool:
        self.moves.append((x, y, z, speed))
        return True

    def magnet_on(self) -> bool:
        return True

    def magnet_off(self) -> bool:
        return True

    def pick_piece(self, x: float, y: float, speed: int) -> bool:
        self.pick_calls += 1
        return True

    def place_piece(self, x: float, y: float, speed: int) -> bool:
        self.place_calls += 1
        return True


class SimPerception:
    """Perception simulator implementing the methods used by run_game."""

    def __init__(self, initial_board: np.ndarray, next_board: Optional[np.ndarray] = None) -> None:
        self.initial_board = initial_board.copy()
        self.current_board = initial_board.copy()
        self.next_board = None if next_board is None else next_board.copy()
        self.closed = False

    def start_background(self) -> bool:
        return True

    def close(self) -> None:
        self.closed = True

    def wait_for_stable_board(self, timeout: float = 0.0, stable_frames: int = 0) -> Optional[np.ndarray]:
        _ = timeout
        _ = stable_frames
        return self.current_board.copy()

    def wait_for_board_change(self, reference: np.ndarray, timeout: float = 0.0) -> Optional[np.ndarray]:
        _ = timeout
        if self.next_board is not None and not np.array_equal(self.next_board, reference):
            self.current_board = self.next_board.copy()
            self.next_board = None
            return self.current_board.copy()
        return None

    def verify_board(self, expected: np.ndarray, timeout: float = 0.0) -> bool:
        _ = timeout
        self.current_board = expected.copy()
        return True


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_parse_move() -> None:
    valid = main.parse_move("1:A2->A3")
    _assert(valid is not None, "valid move was not parsed")
    _assert(valid.piece == 1, "piece id mismatch")
    _assert(valid.from_cell == (1, 0), "source cell mismatch")
    _assert(valid.to_cell == (2, 0), "destination cell mismatch")
    _assert(valid.promotion is None, "promotion should be None")

    promo = main.parse_move("1:A5->A6=4")
    _assert(promo is not None and promo.promotion == 4, "promotion move parsing failed")

    invalid = main.parse_move("BAD_MOVE")
    _assert(invalid is None, "invalid move should return None")


def test_apply_move() -> None:
    board = np.zeros((6, 6), dtype=int)
    board[1, 0] = 1

    parsed = main.parse_move("1:A2->A3")
    _assert(parsed is not None, "parsed move is None")

    new_board, captured = main.apply_move(board, parsed)
    _assert(captured == 0, "capture should be empty")
    _assert(int(new_board[1, 0]) == 0, "source not cleared")
    _assert(int(new_board[2, 0]) == 1, "destination not updated")


def test_execute_move_capture() -> None:
    arm = SimArm()
    board = np.zeros((6, 6), dtype=int)
    board[1, 0] = 1
    board[2, 0] = 6

    parsed = main.parse_move("1:A2->A3")
    _assert(parsed is not None, "parsed move is None")

    new_board, wg, bg, ok = main.execute_move(
        arm=arm,
        board=board,
        move=parsed,
        white_grave_idx=0,
        black_grave_idx=0,
        speed=500,
    )

    _assert(ok, "execute_move returned failure")
    _assert(int(new_board[1, 0]) == 0 and int(new_board[2, 0]) == 1, "board state incorrect")
    _assert(wg == 0 and bg == 1, "graveyard index update incorrect for black capture")
    _assert(arm.pick_calls == 2, "capture move should pick twice (captured + moving piece)")
    _assert(arm.place_calls == 2, "capture move should place twice")


def test_mocked_run_game_once() -> None:
    initial = np.zeros((6, 6), dtype=int)
    initial[1, 0] = 1

    sim_perception = SimPerception(initial_board=initial, next_board=None)

    original_arm_cls = main.ArmController
    original_perception_cls = main.PerceptionSystem
    original_get_move = main.game.get_move

    try:
        main.ArmController = SimArm  # type: ignore[assignment]
        main.PerceptionSystem = lambda: sim_perception  # type: ignore[assignment]
        main.game.get_move = lambda board, turn: "1:A2->A3"  # type: ignore[assignment]

        code = main.run_game(play_white=True)
        _assert(code == 0, "run_game should exit with code 0 in simulation")
    finally:
        main.ArmController = original_arm_cls  # type: ignore[assignment]
        main.PerceptionSystem = original_perception_cls  # type: ignore[assignment]
        main.game.get_move = original_get_move  # type: ignore[assignment]


def run_tests(verbose: bool = False) -> int:
    tests: List[Tuple[str, Callable[[], None]]] = [
        ("parse move", test_parse_move),
        ("apply move", test_apply_move),
        ("execute move with capture", test_execute_move_capture),
        ("mocked run loop", test_mocked_run_game_once),
    ]

    results: List[TestResult] = []

    for name, fn in tests:
        try:
            fn()
            results.append(TestResult(name=name, passed=True))
            print(f"[PASS] {name}")
        except Exception as exc:
            details = str(exc)
            results.append(TestResult(name=name, passed=False, details=details))
            print(f"[FAIL] {name}: {details}")
            if verbose:
                traceback.print_exc()

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    print("\n=== Artificial Test Summary ===")
    print(f"Total: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed:
        return 1
    return 0


def main_cli() -> int:
    parser = argparse.ArgumentParser(description="Artificial tester for Team L2 hardware stage")
    parser.add_argument("--verbose", action="store_true", help="print tracebacks on failure")
    args = parser.parse_args()
    return run_tests(verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main_cli())
