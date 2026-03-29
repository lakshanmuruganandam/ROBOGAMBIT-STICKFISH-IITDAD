"""Pre-match calibration command center.

Single file to run before every match:
1) Stable arm pose
2) Capture 4 corner robot coordinates
3) Compute world->robot homography
4) Auto-update ROBOT_REALITY in perception.py
5) Run motion validation tests
6) Save calibration report
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
from typing import Dict, Tuple

from calibration_workflow import ArmConfig, ArmController, CalibrationWorkflow


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-match calibration tools")
    parser.add_argument("--arm-port", default=os.getenv("ROBO_ARM_SERIAL_PORT", "COM8"))
    parser.add_argument("--magnet-port", default=os.getenv("ROBO_SOLENOID_SERIAL_PORT", "COM10"))
    parser.add_argument("--baud", type=int, default=int(os.getenv("ROBO_SERIAL_BAUD", "115200")))
    parser.add_argument("--z-hover", type=float, default=200.0)
    parser.add_argument("--z-touch", type=float, default=20.0)
    parser.add_argument("--stable", choices=["init", "folded"], default="init")
    parser.add_argument(
        "--perception-file",
        default=os.path.join(os.path.dirname(__file__), "perception.py"),
        help="Path to perception.py where ROBOT_REALITY should be updated",
    )
    return parser.parse_args()


def _format_robot_reality(corner_robot_xyz: Dict[int, Tuple[float, float, float]]) -> str:
    lines = ["ROBOT_REALITY = {"]
    for cid in [21, 22, 23, 24]:
        x, y, _z = corner_robot_xyz[cid]
        lines.append(f"    {cid}: ({x:.2f}, {y:.2f}),")
    lines.append("}")
    return "\n".join(lines)


def _update_perception_robot_reality(perception_file: str, corner_robot_xyz: Dict[int, Tuple[float, float, float]]) -> bool:
    if not os.path.exists(perception_file):
        print(f"[WARN] perception file not found: {perception_file}")
        return False

    with open(perception_file, "r", encoding="utf-8") as f:
        content = f.read()

    replacement_block = _format_robot_reality(corner_robot_xyz)
    pattern = r"ROBOT_REALITY\s*=\s*\{[^\}]*\}"
    new_content, count = re.subn(pattern, replacement_block, content, flags=re.DOTALL)

    if count == 0:
        print("[WARN] ROBOT_REALITY block not found in perception.py; no update applied.")
        return False

    with open(perception_file, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"[OK] Updated ROBOT_REALITY in {perception_file}")
    return True


def _run_prematch(flow: CalibrationWorkflow, perception_file: str) -> None:
    flow.preflight()
    flow.set_stable_start_pose()
    flow.capture_corner_points()
    flow.compute_homography()

    print("\n=== APPLY CALIBRATION TO PERCEPTION ===")
    _update_perception_robot_reality(perception_file, flow.corner_robot_xyz)

    flow.run_validation_tests()
    report = flow.save_report()

    passed = sum(1 for t in flow.tests if t.get("pass"))
    total = len(flow.tests)
    print(f"\n=== PRE-MATCH RESULT: {passed}/{total} tests passed ===")
    if passed < total:
        print("[ACTION] Re-run calibration before match. Do not proceed with live game yet.")
    else:
        print("[OK] Calibration validated. Safe to proceed to match startup.")
    print(f"[OK] Report: {report}")


def _menu() -> str:
    print("\n=== CALIB TOOLS ===")
    print("1) Full pre-match calibration (recommended)")
    print("2) Exit")
    return input("Select option: ").strip()


def main() -> None:
    args = _parse_args()
    cfg = ArmConfig(
        arm_port=args.arm_port,
        magnet_port=args.magnet_port,
        baud=args.baud,
        z_hover=args.z_hover,
        z_touch=args.z_touch,
        stable_mode=args.stable,
    )

    print("\n=== PRE-MATCH CALIBRATION TOOL ===")
    print(f"arm_port={cfg.arm_port}, magnet_port={cfg.magnet_port}, baud={cfg.baud}")
    print(f"stable_mode={cfg.stable_mode}, z_hover={cfg.z_hover}, z_touch={cfg.z_touch}")
    print(f"perception_file={args.perception_file}")

    choice = _menu()
    if choice != "1":
        print("Exit.")
        return

    arm = ArmController(cfg)
    flow = CalibrationWorkflow(arm)

    try:
        arm.connect()
        _run_prematch(flow, args.perception_file)
    finally:
        arm.close()
        print(f"[DONE] {dt.datetime.now().isoformat(timespec='seconds')} Serial ports closed.")


if __name__ == "__main__":
    main()
