"""
Full-system realistic arm test for Team L2.

This test uses a local ESP32 API emulator and runs the real L2 ArmController
and move-execution flow against it.

Run:
  python full_system_arm_test.py
  python full_system_arm_test.py --port 18080 --verbose
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import math
import os
import sys
from typing import Tuple

import numpy as np

from esp32_arm_emulator import ArmEmulatorServer


def _load_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create module spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def run_full_system_test(port: int, verbose: bool = False) -> int:
    emulator = ArmEmulatorServer(host="127.0.0.1", port=port, block_seconds=3.2)
    emulator.start()

    try:
        # Configure L2 stack to use local ESP32 emulator.
        os.environ["ROBO_ARM_IP"] = "127.0.0.1"
        os.environ["ROBO_ARM_PORT"] = str(port)
        os.environ["ROBO_SERIAL_PORT"] = "COM_DOES_NOT_EXIST"

        # Ensure local folder imports win.
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)

        # Force-load exact L2 modules by path to avoid name collisions.
        config_path = os.path.join(here, "config.py")
        main_path = os.path.join(here, "main.py")

        for mod_name in ("config", "main"):
            if mod_name in sys.modules:
                del sys.modules[mod_name]

        _ = _load_module_from_path("config", config_path)
        main = _load_module_from_path("main", main_path)

        arm = main.ArmController()
        if getattr(arm, "serial_conn", None) is None:
            arm.magnet_on = lambda: True
            arm.magnet_off = lambda: True

        # 1) Basic movement test with realistic timeout behavior on command send.
        ok_home = arm.arm_home()
        ok_a2 = arm.move_xyz(-150.0, -90.0, 180.0, 500)
        ok_a3 = arm.move_xyz(-150.0, -30.0, 180.0, 500)

        if not (ok_home and ok_a2 and ok_a3):
            print("[FAIL] basic move sequence failed")
            arm.close()
            return 1

        # 2) Real execute_move path (capture + piece move) using main.execute_move.
        board = np.zeros((6, 6), dtype=int)
        board[1, 0] = 1   # white pawn at A2
        board[2, 0] = 6   # black pawn at A3 to be captured

        parsed = main.parse_move("1:A2->A3")
        if parsed is None:
            print("[FAIL] parse_move returned None")
            arm.close()
            return 1

        out_board, wg, bg, ok_exec = main.execute_move(
            arm=arm,
            board=board,
            move=parsed,
            white_grave_idx=0,
            black_grave_idx=0,
            speed=500,
        )

        if not ok_exec:
            print("[FAIL] execute_move failed")
            arm.close()
            return 1

        if int(out_board[2, 0]) != 1 or int(out_board[1, 0]) != 0:
            print("[FAIL] board result incorrect after execute_move")
            arm.close()
            return 1

        if wg != 0 or bg != 1:
            print("[FAIL] graveyard indices incorrect")
            arm.close()
            return 1

        snap = emulator.snapshot()
        target = snap["target"]
        pos = (snap["x"], snap["y"], snap["z"])
        dist = _distance(pos, target)

        if verbose:
            print("[INFO] emulator snapshot:", snap)

        # Allow some tolerance because command/feedback are asynchronous.
        if dist > 25.0:
            print(f"[FAIL] emulator final position too far from target: {dist:.2f}mm")
            arm.close()
            return 1

        if snap["command_count"] < 3:
            print("[FAIL] too few emulator commands observed")
            arm.close()
            return 1

        if snap["feedback_count"] < 1:
            print("[FAIL] no feedback polling observed")
            arm.close()
            return 1

        arm.close()

        print("[PASS] full system arm test")
        print(f"[PASS] commands={snap['command_count']} feedback={snap['feedback_count']} final_error_mm={dist:.2f}")
        return 0

    finally:
        emulator.stop()


def main_cli() -> int:
    parser = argparse.ArgumentParser(description="Realistic full-system L2 arm test")
    parser.add_argument("--port", type=int, default=18080, help="local emulator port")
    parser.add_argument("--verbose", action="store_true", help="print detailed snapshot")
    args = parser.parse_args()
    return run_full_system_test(port=args.port, verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main_cli())
