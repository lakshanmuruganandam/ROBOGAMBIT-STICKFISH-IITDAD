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
    parser.add_argument(
        "--main-file",
        default=os.path.join(os.path.dirname(__file__), "main.py"),
        help="Path to main.py where Z_HOVER and Z_PICK should be updated",
    )
    parser.add_argument(
        "--skip-z-capture",
        action="store_true",
        help="Skip manual Z capture and keep current --z-hover/--z-touch values",
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


def _capture_z_levels(arm: ArmController, default_hover: float, default_touch: float) -> Tuple[float, float]:
    print("\n=== CAPTURE Z LEVELS ===")
    print("Step 1: Move arm tip to SAFE HOVER height above a center piece (no contact).")
    input("When stable at hover height, press Enter to capture...")
    hx, hy, hz, _hs, _he = arm.get_feedback()
    if hz is None:
        print("[WARN] Could not read hover Z; using current value.")
        hover_z = float(default_hover)
    else:
        hover_z = float(hz)
        print(f"[CAPTURED] Hover point x={hx:.2f}, y={hy:.2f}, z={hover_z:.2f}")

    print("\nStep 2: Move arm tip to PICK/TOUCH height (just touching piece top).")
    input("When stable at touch height, press Enter to capture...")
    tx, ty, tz, _ts, _te = arm.get_feedback()
    if tz is None:
        print("[WARN] Could not read touch Z; using current value.")
        touch_z = float(default_touch)
    else:
        touch_z = float(tz)
        print(f"[CAPTURED] Touch point x={tx:.2f}, y={ty:.2f}, z={touch_z:.2f}")

    if touch_z >= hover_z:
        print("[WARN] Touch Z is not below hover Z. Keeping previous configured Z values.")
        return float(default_hover), float(default_touch)

    print(f"[OK] Using calibrated Z_HOVER={hover_z:.2f}, Z_PICK={touch_z:.2f}")
    return hover_z, touch_z


def _update_main_z_levels(main_file: str, z_hover: float, z_pick: float) -> bool:
    if not os.path.exists(main_file):
        print(f"[WARN] main file not found: {main_file}")
        return False

    with open(main_file, "r", encoding="utf-8") as f:
        content = f.read()

    content2, count_hover = re.subn(
        r"Z_HOVER\s*=\s*[-+]?\d+(?:\.\d+)?",
        f"Z_HOVER = {z_hover:.1f}",
        content,
        count=1,
    )
    content3, count_pick = re.subn(
        r"Z_PICK\s*=\s*[-+]?\d+(?:\.\d+)?",
        f"Z_PICK  = {z_pick:.1f}",
        content2,
        count=1,
    )

    if count_hover == 0 or count_pick == 0:
        print("[WARN] Could not find Z_HOVER/Z_PICK in main.py; no update applied.")
        return False

    with open(main_file, "w", encoding="utf-8") as f:
        f.write(content3)

    print(f"[OK] Updated Z_HOVER and Z_PICK in {main_file}")
    return True


def _run_prematch(
    flow: CalibrationWorkflow,
    perception_file: str,
    main_file: str,
    skip_z_capture: bool,
) -> None:
    flow.preflight()
    flow.set_stable_start_pose()
    flow.capture_corner_points()
    flow.compute_homography()

    print("\n=== APPLY CALIBRATION TO PERCEPTION ===")
    _update_perception_robot_reality(perception_file, flow.corner_robot_xyz)

    if not skip_z_capture:
        z_hover, z_touch = _capture_z_levels(flow.arm, flow.arm.cfg.z_hover, flow.arm.cfg.z_touch)
        flow.arm.cfg.z_hover = z_hover
        flow.arm.cfg.z_touch = z_touch

    print("\n=== APPLY Z LEVELS TO MAIN ===")
    _update_main_z_levels(main_file, flow.arm.cfg.z_hover, flow.arm.cfg.z_touch)

    print(
        f"[CALIB] Active test Z levels: Z_HOVER={flow.arm.cfg.z_hover:.2f}, "
        f"Z_PICK={flow.arm.cfg.z_touch:.2f}"
    )

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
    print(f"main_file={args.main_file}")

    choice = _menu()
    if choice != "1":
        print("Exit.")
        return

    arm = ArmController(cfg)
    flow = CalibrationWorkflow(arm)

    try:
        arm.connect()
        _run_prematch(
            flow,
            args.perception_file,
            args.main_file,
            args.skip_z_capture,
        )
    finally:
        arm.close()
        print(f"[DONE] {dt.datetime.now().isoformat(timespec='seconds')} Serial ports closed.")


if __name__ == "__main__":
    main()
