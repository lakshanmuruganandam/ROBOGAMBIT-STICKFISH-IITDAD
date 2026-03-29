"""Interactive arm + board calibration workflow for LADDOO-FINAL-NOCHOICE.

What this script does:
1) Sends arm to a stable start pose (default T=100).
2) Lets you manually position the arm tip on each corner marker center and records robot XY.
3) Computes world->robot homography from those 4 corner pairs.
4) Prints paste-ready ROBOT_REALITY constants for perception.py.
5) Runs quick motion validation tests (center, corner, capture-path style waypoints).
6) Saves a JSON report with all captured values.

Usage:
  python calibration_workflow.py
  python calibration_workflow.py --arm-port COM8 --magnet-port COM10 --stable folded
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import serial as pyserial
except Exception as exc:
    raise RuntimeError("pyserial import failed. Install with: pip install pyserial") from exc

try:
    from serial import Serial as PySerial
except Exception:
    try:
        from serial.serialwin32 import Serial as PySerial
    except Exception as exc:
        raise RuntimeError(
            "Imported 'serial' module does not provide Serial. "
            f"Loaded module file: {getattr(pyserial, '__file__', 'unknown')}."
        ) from exc

import perception


CORNER_IDS = [21, 22, 23, 24]

PIECE_LABELS = {
    0: "..",
    1: "W_P",
    2: "W_N",
    3: "W_B",
    4: "W_Q",
    5: "W_K",
    6: "B_P",
    7: "B_N",
    8: "B_B",
    9: "B_Q",
    10: "B_K",
}

COL_LABELS = ["A", "B", "C", "D", "E", "F"]


@dataclass
class ArmConfig:
    arm_port: str
    magnet_port: str
    baud: int
    z_hover: float
    z_touch: float
    stable_mode: str


class ArmController:
    def __init__(self, cfg: ArmConfig):
        self.cfg = cfg
        self.ser: Optional[PySerial] = None
        self.mag: Optional[PySerial] = None

    def connect(self) -> None:
        self.ser = PySerial(self.cfg.arm_port, baudrate=self.cfg.baud, dsrdtr=None, timeout=1)
        self.ser.setRTS(False)
        self.ser.setDTR(False)
        self.mag = PySerial(self.cfg.magnet_port, baudrate=self.cfg.baud, timeout=1)
        print(f"[OK] Arm serial connected: {self.cfg.arm_port}")
        print(f"[OK] Magnet serial connected: {self.cfg.magnet_port}")

    def close(self) -> None:
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
        if self.mag is not None:
            try:
                self.mag.close()
            except Exception:
                pass
            self.mag = None

    def _send_json(self, payload: dict) -> None:
        if self.ser is None:
            raise RuntimeError("Arm serial not connected.")
        msg = json.dumps(payload)
        self.ser.write(msg.encode() + b"\n")

    def _read_json_line(self, attempts: int = 50) -> Optional[dict]:
        if self.ser is None:
            return None
        for _ in range(attempts):
            line = self.ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            if line == '{"T": 105}':
                continue
            if "{" in line and "}" in line:
                try:
                    obj = json.loads(line[line.find("{") : line.rfind("}") + 1])
                    return obj
                except Exception:
                    continue
        return None

    def get_feedback(self) -> Tuple[Optional[float], Optional[float], Optional[float], float, float]:
        if self.ser is None:
            raise RuntimeError("Arm serial not connected.")
        try:
            self.ser.reset_input_buffer()
        except Exception:
            pass
        self._send_json({"T": 105})
        data = self._read_json_line()
        if not data:
            return None, None, None, 0.0, 0.0
        return (
            data.get("x"),
            data.get("y"),
            data.get("z"),
            float(data.get("s", 0.0)),
            float(data.get("e", 0.0)),
        )

    def move_xyz(self, x: float, y: float, z: float, spd: float = 0.3, t: float = 3.14) -> None:
        self._send_json({"T": 104, "x": round(x, 2), "y": round(y, 2), "z": round(z, 2), "t": t, "spd": spd})

    def init_pose(self) -> None:
        self._send_json({"T": 100})

    def folded_pose(self) -> None:
        self._send_json({"T": 122, "b": 0, "s": 0, "e": 15, "h": 180, "spd": 10, "acc": 10})

    def set_magnet(self, on: bool) -> None:
        if self.mag is None:
            raise RuntimeError("Magnet serial not connected.")
        self.mag.write(b"1" if on else b"0")


class CalibrationWorkflow:
    def __init__(self, arm: ArmController):
        self.arm = arm
        self.corner_robot_xyz: Dict[int, Tuple[float, float, float]] = {}
        self.tests: List[dict] = []
        self.h_world_to_robot: Optional[np.ndarray] = None

    def preflight(self) -> None:
        print("\n=== PRE-FLIGHT ===")
        print("1) Tape board sheet flat (no wrinkles).")
        print("2) Keep all 4 corner markers fully visible.")
        print("3) Keep wires/arm outside camera view while detecting.")
        print("4) Confirm camera stream is running.")
        input("Press Enter when pre-flight is done...")

    def _print_board_human(self, board: np.ndarray, title: str = "PERCEPTION BOARD") -> None:
        print(f"\n=== {title} ===")
        top = "      " + "  ".join(COL_LABELS)
        print(top)
        for r in range(6):
            row_tokens = [PIECE_LABELS.get(int(board[r][c]), f"{int(board[r][c]):02d}") for c in range(6)]
            left = str(r + 1)
            right = str(r + 1)
            print(f"  {left} | " + " ".join(f"{t:>3}" for t in row_tokens) + f" | {right}")
        print(top)
        print("Legend: W_* = white piece, B_* = black piece, .. = empty")

    def verify_perception_board_loop(self) -> None:
        """Continuously print perceived board until user chooses to proceed."""
        print("\n=== PERCEPTION VISIBILITY CHECK ===")
        print("Place your 6-7 test pieces and verify recognition.")
        print("Control: y = looks good, keep checking | n = done, continue calibration")

        vision = perception.BoardPerception(connect_socket=True)
        try:
            while True:
                board, _poses = vision.get_latest_state()
                if board is None:
                    print("[WAIT] No frame/board yet. Check camera stream and corner markers.")
                    ans = input("Continue waiting? (y/n): ").strip().lower()
                    if ans == "n":
                        raise RuntimeError("Perception not ready; cannot continue calibration safely.")
                    continue

                self._print_board_human(board, "PERCEPTION BOARD (LIVE)")
                ans = input("Board correct? (y=continue checking, n=proceed): ").strip().lower()
                if ans == "n":
                    print("[OK] Perception check accepted. Proceeding to calibration.")
                    break
        finally:
            vision.cleanup()

    def set_stable_start_pose(self) -> None:
        print("\n=== STABLE ARM START POSE ===")
        self.arm.init_pose()
        time.sleep(1.2)
        if self.arm.cfg.stable_mode == "folded":
            self.arm.folded_pose()
            time.sleep(1.0)
        print(f"[OK] Stable pose sent (mode={self.arm.cfg.stable_mode}).")

    def capture_corner_points(self) -> None:
        print("\n=== CAPTURE CORNER POINTS ===")
        print("Use your normal manual jog method to place arm tip at each marker center.")
        print("Order: 21 (TL), 22 (TR), 23 (BR), 24 (BL) based on your board definition.")

        for cid in CORNER_IDS:
            input(f"Position tip at marker {cid} center, then press Enter to capture...")
            x, y, z, _s, _e = self.arm.get_feedback()
            if x is None or y is None or z is None:
                raise RuntimeError(f"Failed to read feedback for marker {cid}.")
            self.corner_robot_xyz[cid] = (float(x), float(y), float(z))
            print(f"[CAPTURED] {cid}: x={x:.2f}, y={y:.2f}, z={z:.2f}")

    def compute_homography(self) -> None:
        print("\n=== COMPUTE WORLD->ROBOT HOMOGRAPHY ===")
        world_pts = np.array([perception.CORNER_WORLD[c] for c in CORNER_IDS], dtype=np.float32)
        robot_pts = np.array([(self.corner_robot_xyz[c][0], self.corner_robot_xyz[c][1]) for c in CORNER_IDS], dtype=np.float32)
        h, _ = cv2.findHomography(world_pts, robot_pts)
        if h is None:
            raise RuntimeError("Homography solve failed. Recapture corner points.")
        self.h_world_to_robot = h

        print("[OK] Homography computed.")
        print("Paste-ready ROBOT_REALITY for perception.py:")
        print("ROBOT_REALITY = {")
        for cid in CORNER_IDS:
            x, y, _z = self.corner_robot_xyz[cid]
            print(f"    {cid}: ({x:.2f}, {y:.2f}),")
        print("}")

    def _cell_to_world(self, cell: str) -> Tuple[float, float]:
        col_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
        col = col_map[cell[0].upper()]
        row = int(cell[1]) - 1
        wx = perception.TOP_LEFT_X - (row * perception.SQUARE_SIZE + perception.SQUARE_SIZE / 2)
        wy = perception.TOP_LEFT_Y - (col * perception.SQUARE_SIZE + perception.SQUARE_SIZE / 2)
        return float(wx), float(wy)

    def _world_to_robot(self, wx: float, wy: float) -> Tuple[float, float]:
        if self.h_world_to_robot is None:
            raise RuntimeError("Homography not ready.")
        pt = cv2.perspectiveTransform(np.array([[[wx, wy]]], dtype=np.float32), self.h_world_to_robot)
        return float(pt[0][0][0]), float(pt[0][0][1])

    def _run_waypoint_test(self, label: str, xy_points: List[Tuple[float, float]]) -> bool:
        print(f"\n=== TEST: {label} ===")
        try:
            for i, (x, y) in enumerate(xy_points, start=1):
                print(f"[MOVE {i}] x={x:.2f}, y={y:.2f}, z={self.arm.cfg.z_hover:.2f}")
                self.arm.move_xyz(x, y, self.arm.cfg.z_hover)
                time.sleep(1.0)
                print(f"[MOVE {i}] x={x:.2f}, y={y:.2f}, z={self.arm.cfg.z_touch:.2f}")
                self.arm.move_xyz(x, y, self.arm.cfg.z_touch)
                time.sleep(0.8)
                self.arm.move_xyz(x, y, self.arm.cfg.z_hover)
                time.sleep(0.8)
        except Exception as exc:
            print(f"[ERROR] Test failed during movement: {exc}")
            return False

        ans = input("Did this test land correctly? (y/n): ").strip().lower()
        ok = ans == "y"
        self.tests.append({"name": label, "pass": ok})
        return ok

    def run_validation_tests(self) -> None:
        print("\n=== MOTION VALIDATION TESTS ===")
        print("These tests move only the tip path (no magnet toggles).")

        c3_w = self._cell_to_world("C3")
        c3_r = self._world_to_robot(*c3_w)
        self._run_waypoint_test("center_C3", [c3_r])

        a1_w = self._cell_to_world("A1")
        a1_r = self._world_to_robot(*a1_w)
        f6_w = self._cell_to_world("F6")
        f6_r = self._world_to_robot(*f6_w)
        self._run_waypoint_test("corner_A1_then_F6", [a1_r, f6_r])

        src_w = self._cell_to_world("B2")
        dst_w = self._cell_to_world("C2")
        discard = (-150.0, 0.0)
        src_r = self._world_to_robot(*src_w)
        dst_r = self._world_to_robot(*dst_w)
        self._run_waypoint_test("capture_path_src_discard_dst", [src_r, discard, dst_r])

    def save_report(self) -> str:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(os.path.dirname(__file__), f"calibration_report_{ts}.json")

        h_list = self.h_world_to_robot.tolist() if self.h_world_to_robot is not None else None
        report = {
            "timestamp": ts,
            "corner_robot_xyz": {str(k): [v[0], v[1], v[2]] for k, v in self.corner_robot_xyz.items()},
            "corner_robot_xy_for_perception": {str(k): [v[0], v[1]] for k, v in self.corner_robot_xyz.items()},
            "h_world_to_robot": h_list,
            "tests": self.tests,
            "recommended_steps": [
                "Update ROBOT_REALITY in perception.py with captured XY values.",
                "Run one live non-capture move.",
                "Run one live capture move.",
                "Tune Z_PICK in main.py by +/-2 mm if pick/place misses.",
            ],
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arm + board calibration workflow")
    parser.add_argument("--arm-port", default=os.getenv("ROBO_ARM_SERIAL_PORT", "COM8"))
    parser.add_argument("--magnet-port", default=os.getenv("ROBO_SOLENOID_SERIAL_PORT", "COM10"))
    parser.add_argument("--baud", type=int, default=int(os.getenv("ROBO_SERIAL_BAUD", "115200")))
    parser.add_argument("--z-hover", type=float, default=200.0)
    parser.add_argument("--z-touch", type=float, default=20.0)
    parser.add_argument("--stable", choices=["init", "folded"], default="init")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ArmConfig(
        arm_port=args.arm_port,
        magnet_port=args.magnet_port,
        baud=args.baud,
        z_hover=args.z_hover,
        z_touch=args.z_touch,
        stable_mode=args.stable,
    )

    arm = ArmController(cfg)
    flow = CalibrationWorkflow(arm)

    print("\n=== CALIBRATION WORKFLOW START ===")
    print(f"arm_port={cfg.arm_port}, magnet_port={cfg.magnet_port}, baud={cfg.baud}")

    try:
        arm.connect()
        flow.preflight()
        flow.set_stable_start_pose()
        flow.capture_corner_points()
        flow.compute_homography()
        flow.run_validation_tests()
        report_path = flow.save_report()
        print(f"\n[OK] Calibration report saved: {report_path}")
    finally:
        arm.close()
        print("[DONE] Serial ports closed.")


if __name__ == "__main__":
    main()
