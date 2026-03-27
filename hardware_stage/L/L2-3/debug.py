"""
Comprehensive setup checker for Team L2-3.

This script validates system environment, network, camera stream quality,
arm feedback, serial connectivity, and perception stability.

Usage:
  python debug.py
  python debug.py --full --verbose
  python debug.py --network --arm --camera --serial --perception
  python debug.py --camera-stream-seconds 8 --save-frame frame.jpg
  python debug.py --arm-motion-check --serial-write-test --json-out report.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import platform
import re
import shutil
import socket
import struct
import subprocess
import sys
import time
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import cv2
import numpy as np
import requests

from config import (
    ARM_BASE_URL,
    ARM_ARRIVAL_TIMEOUT,
    ARM_IP,
    ARM_POLL_INTERVAL,
    ARM_PORT,
    ARM_SETTLE_TIME,
    BAUD_RATE,
    BOARD_SIZE,
    ARUCO_CORNER_IDS,
    ARUCO_PIECE_IDS,
    CAMERA_CONNECT_TIMEOUT,
    CAMERA_IP,
    CAMERA_MATRIX,
    CAMERA_PORT,
    CAMERA_RECV_TIMEOUT,
    CELL_CENTERS_X,
    CELL_CENTERS_Y,
    CORNER_WORLD_COORDS,
    DIST_COEFFS,
    GRAVEYARD_BLACK,
    GRAVEYARD_WHITE,
    GRIPPER_ACTIVATE_TIME,
    HOME_X,
    HOME_Y,
    HOME_Z,
    HTTP_FEEDBACK_TIMEOUT,
    MOVE_REGEX,
    POSITION_TOLERANCE_MM,
    SERIAL_PORT,
    SERIAL_TIMEOUT,
    SPEED_NORMAL,
    Z_PICK,
    Z_PLACE,
    Z_SAFE,
)
from perception import PerceptionSystem

try:
    import serial  # type: ignore
except Exception:
    serial = None


@dataclass
class CheckResult:
    category: str
    name: str
    status: str
    message: str
    elapsed_s: float


class Report:
    def __init__(self) -> None:
        self.start = time.monotonic()
        self.rows: List[CheckResult] = []

    def add(self, category: str, name: str, status: str, message: str = "", elapsed_s: float = 0.0) -> None:
        self.rows.append(CheckResult(category, name, status, message, elapsed_s))

    def to_json(self) -> Dict[str, object]:
        totals = {"PASS": 0, "FAIL": 0, "WARN": 0, "SKIP": 0}
        for row in self.rows:
            totals[row.status] = totals.get(row.status, 0) + 1
        return {
            "elapsed_s": time.monotonic() - self.start,
            "totals": totals,
            "rows": [asdict(r) for r in self.rows],
        }

    def print(self) -> bool:
        print("\n" + "=" * 96)
        print("L2-3 DEBUG REPORT")
        print("=" * 96)

        grouped: Dict[str, List[CheckResult]] = {}
        for row in self.rows:
            grouped.setdefault(row.category, []).append(row)

        counts = {"PASS": 0, "FAIL": 0, "WARN": 0, "SKIP": 0}
        for category in sorted(grouped):
            print(f"\n[{category.upper()}]")
            for row in grouped[category]:
                counts[row.status] = counts.get(row.status, 0) + 1
                dur = f" ({row.elapsed_s:.2f}s)" if row.elapsed_s > 0 else ""
                msg = f" - {row.message}" if row.message else ""
                print(f"  {row.status:5s} {row.name}{dur}{msg}")

        print("\n" + "-" * 96)
        print(
            f"Totals: PASS={counts['PASS']} FAIL={counts['FAIL']} WARN={counts['WARN']} SKIP={counts['SKIP']} | "
            f"elapsed={time.monotonic() - self.start:.2f}s"
        )
        print("-" * 96)
        return counts["FAIL"] == 0


def _recv_exact(sock: socket.socket, nbytes: int) -> Optional[bytes]:
    buf = b""
    while len(buf) < nbytes:
        chunk = sock.recv(nbytes - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def _http_json_cmd(cmd: Dict[str, object], timeout: float) -> Dict[str, object]:
    url = f"{ARM_BASE_URL}/js?json={quote(json.dumps(cmd))}"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.ReadTimeout:
        # Some firmwares accept T:104 move commands but do not reply quickly.
        if int(cmd.get("T", -1)) == 104:
            return {"accepted": True, "timeout": True}
        raise
    try:
        payload = resp.json()
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _arm_feedback(timeout: float) -> Tuple[float, float, float]:
    data = _http_json_cmd({"T": 105}, timeout=timeout)
    return float(data["x"]), float(data["y"]), float(data["z"])


def _distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _wait_arrival(target: Tuple[float, float, float], timeout: float) -> Tuple[bool, Optional[Tuple[float, float, float]], int]:
    deadline = time.monotonic() + timeout
    last = None
    polls = 0
    while time.monotonic() < deadline:
        polls += 1
        try:
            last = _arm_feedback(timeout=HTTP_FEEDBACK_TIMEOUT)
            if _distance(last, target) <= POSITION_TOLERANCE_MM:
                return True, last, polls
        except Exception:
            pass
        time.sleep(ARM_POLL_INTERVAL)
    return False, last, polls


def _parse_ping_loss(output: str) -> Optional[float]:
    m = re.search(r"(\d+)%\s*(packet\s+)?loss", output, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"Lost\s*=\s*\d+\s*\((\d+)%\)", output, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def _load_game_module() -> Optional[object]:
    # L2-3 intentionally excludes game.py; probe likely sibling/workspace locations.
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "L2" / "game.py",
        here.parent / "game.py",
        Path.cwd() / "hardware_stage" / "L" / "L2" / "game.py",
        Path.cwd() / "L2" / "game.py",
    ]
    for game_path in candidates:
        if not game_path.exists():
            continue
        spec = importlib.util.spec_from_file_location("l2_game_debug", str(game_path))
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return None


def _tcp_reachable(ip: str, port: int, timeout: float) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        return sock.connect_ex((ip, port)) == 0
    except Exception:
        return False
    finally:
        try:
            sock.close()
        except Exception:
            pass


def _open_serial_with_retry(retries: int = 3, delay_s: float = 0.2):
    if serial is None:
        raise RuntimeError("pyserial unavailable")
    last_exc: Optional[Exception] = None
    for _ in range(max(1, retries)):
        try:
            return serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
        except Exception as exc:
            last_exc = exc
            time.sleep(delay_s)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("failed to open serial port")


def check_system(report: Report) -> None:
    t0 = time.monotonic()
    report.add(
        "system",
        "python/runtime",
        "PASS",
        f"python={platform.python_version()} platform={platform.platform()}",
        time.monotonic() - t0,
    )

    t0 = time.monotonic()
    try:
        root = Path(__file__).resolve().parent
        needed = ["config.py", "perception.py", "debug.py"]
        missing = [f for f in needed if not (root / f).exists()]
        if missing:
            report.add("system", "required files", "FAIL", f"missing={missing}", time.monotonic() - t0)
        else:
            report.add("system", "required files", "PASS", f"cwd={Path.cwd()}", time.monotonic() - t0)
    except Exception as exc:
        report.add("system", "required files", "FAIL", str(exc), time.monotonic() - t0)

    t0 = time.monotonic()
    try:
        versions = [
            f"opencv={cv2.__version__}",
            f"numpy={np.__version__}",
            f"requests={requests.__version__}",
            f"pyserial={'ok' if serial is not None else 'missing'}",
        ]
        report.add("system", "dependency versions", "PASS", ", ".join(versions), time.monotonic() - t0)
    except Exception as exc:
        report.add("system", "dependency versions", "WARN", str(exc), time.monotonic() - t0)


def check_config_integrity(report: Report) -> None:
    t0 = time.monotonic()
    issues: List[str] = []

    if len(CELL_CENTERS_X) != BOARD_SIZE:
        issues.append(f"CELL_CENTERS_X len={len(CELL_CENTERS_X)} expected={BOARD_SIZE}")
    if len(CELL_CENTERS_Y) != BOARD_SIZE:
        issues.append(f"CELL_CENTERS_Y len={len(CELL_CENTERS_Y)} expected={BOARD_SIZE}")
    if len(GRAVEYARD_WHITE) < 6:
        issues.append(f"GRAVEYARD_WHITE too small ({len(GRAVEYARD_WHITE)})")
    if len(GRAVEYARD_BLACK) < 6:
        issues.append(f"GRAVEYARD_BLACK too small ({len(GRAVEYARD_BLACK)})")
    if not (Z_PICK < Z_SAFE):
        issues.append(f"Z_PICK({Z_PICK}) must be < Z_SAFE({Z_SAFE})")
    if SPEED_NORMAL <= 0:
        issues.append(f"SPEED_NORMAL invalid ({SPEED_NORMAL})")
    if CAMERA_MATRIX.shape != (3, 3):
        issues.append(f"CAMERA_MATRIX shape {CAMERA_MATRIX.shape}")
    if DIST_COEFFS.size < 4:
        issues.append(f"DIST_COEFFS too small ({DIST_COEFFS.size})")
    if len(CORNER_WORLD_COORDS) < 4:
        issues.append(f"CORNER_WORLD_COORDS has {len(CORNER_WORLD_COORDS)} entries")

    if issues:
        report.add("config", "integrity", "FAIL", "; ".join(issues), time.monotonic() - t0)
    else:
        report.add("config", "integrity", "PASS", "all config sanity checks passed", time.monotonic() - t0)


def check_disk_space(report: Report, min_mb: float) -> None:
    t0 = time.monotonic()
    try:
        usage = shutil.disk_usage(Path(__file__).resolve().parent)
        free_mb = usage.free / (1024 * 1024)
        status = "PASS" if free_mb >= min_mb else "WARN"
        report.add("system", "disk space", status, f"free_mb={free_mb:.1f} required_mb={min_mb:.1f}", time.monotonic() - t0)
    except Exception as exc:
        report.add("system", "disk space", "WARN", str(exc), time.monotonic() - t0)


def check_wifi_ping(report: Report, ping_count: int, timeout: float) -> None:
    t0 = time.monotonic()
    count = max(1, ping_count)
    if sys.platform.startswith("win"):
        cmd = ["ping", "-n", str(count), "-w", str(int(timeout * 1000)), ARM_IP]
    else:
        cmd = ["ping", "-c", str(count), "-W", str(max(1, int(timeout))), ARM_IP]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=max(5, int(count * timeout) + 2))
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        loss = _parse_ping_loss(out)
        if loss is None:
            report.add("network", "wifi packet loss", "WARN", "unable to parse ping loss", time.monotonic() - t0)
        elif loss <= 10.0:
            report.add("network", "wifi packet loss", "PASS", f"loss={loss:.0f}%", time.monotonic() - t0)
        else:
            report.add("network", "wifi packet loss", "WARN", f"high loss={loss:.0f}%", time.monotonic() - t0)
    except FileNotFoundError:
        report.add("network", "wifi packet loss", "SKIP", "ping command not available", time.monotonic() - t0)
    except Exception as exc:
        report.add("network", "wifi packet loss", "WARN", str(exc), time.monotonic() - t0)


def check_network(report: Report, timeout: float, probes: int) -> None:
    t0 = time.monotonic()
    try:
        hostname = socket.gethostname()
        host_ip = socket.gethostbyname(hostname)
        report.add("network", "hostname resolution", "PASS", f"{hostname}->{host_ip}", time.monotonic() - t0)
    except Exception as exc:
        report.add("network", "hostname resolution", "FAIL", str(exc), time.monotonic() - t0)

    for name, ip, port in (
        ("arm tcp reachability", ARM_IP, ARM_PORT),
        ("camera tcp reachability", CAMERA_IP, CAMERA_PORT),
    ):
        times: List[float] = []
        failures = 0
        for _ in range(max(1, probes)):
            t0 = time.monotonic()
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                code = sock.connect_ex((ip, port))
                sock.close()
                if code == 0:
                    times.append((time.monotonic() - t0) * 1000.0)
                else:
                    failures += 1
            except Exception:
                failures += 1

        if failures == 0 and times:
            report.add(
                "network",
                name,
                "PASS",
                f"{ip}:{port} probes={len(times)} avg_ms={sum(times)/len(times):.1f} max_ms={max(times):.1f}",
                sum(times) / 1000.0,
            )
        elif times:
            report.add(
                "network",
                name,
                "WARN",
                f"{ip}:{port} success={len(times)} fail={failures} avg_ms={sum(times)/len(times):.1f}",
                0.0,
            )
        else:
            report.add("network", name, "FAIL", f"{ip}:{port} all probes failed", 0.0)


def check_arm(report: Report, timeout: float, feedback_samples: int, motion_check: bool) -> None:
    t0 = time.monotonic()
    try:
        resp = requests.get(f"{ARM_BASE_URL}/", timeout=timeout)
        report.add("arm", "http root reachable", "PASS", f"status={resp.status_code}", time.monotonic() - t0)
    except Exception as exc:
        report.add("arm", "http root reachable", "FAIL", str(exc), time.monotonic() - t0)
        return

    sample_positions: List[Tuple[float, float, float]] = []
    sample_ms: List[float] = []
    fail_count = 0
    for _ in range(max(1, feedback_samples)):
        t0 = time.monotonic()
        try:
            cmd = {"T": 105}
            url = f"{ARM_BASE_URL}/js?json={quote(json.dumps(cmd))}"
            resp = requests.get(url, timeout=HTTP_FEEDBACK_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            x = float(data["x"])
            y = float(data["y"])
            z = float(data["z"])
            sample_positions.append((x, y, z))
            sample_ms.append((time.monotonic() - t0) * 1000.0)
        except Exception:
            fail_count += 1

    if sample_positions:
        first = sample_positions[0]
        last = sample_positions[-1]
        report.add(
            "arm",
            "feedback sampling",
            "PASS" if fail_count == 0 else "WARN",
            (
                f"samples={len(sample_positions)} fail={fail_count} "
                f"avg_ms={sum(sample_ms)/len(sample_ms):.1f} first={first} last={last}"
            ),
            sum(sample_ms) / 1000.0,
        )
    else:
        report.add("arm", "feedback sampling", "FAIL", "no successful feedback samples", 0.0)

    if motion_check and sample_positions:
        t0 = time.monotonic()
        try:
            x, y, z = sample_positions[-1]
            cmd = {"T": 104, "x": x, "y": y, "z": z, "t": 400}
            url = f"{ARM_BASE_URL}/js?json={quote(json.dumps(cmd))}"
            requests.get(url, timeout=timeout)
            report.add("arm", "motion command echo", "PASS", "sent T:104 to current pose", time.monotonic() - t0)
        except Exception as exc:
            report.add("arm", "motion command echo", "WARN", str(exc), time.monotonic() - t0)


def check_arm_arrival_pipeline(report: Report, timeout: float) -> None:
    t0 = time.monotonic()
    if not _tcp_reachable(ARM_IP, ARM_PORT, timeout=max(0.8, timeout)):
        report.add("arm", "move+arrival pipeline", "SKIP", f"arm unreachable at {ARM_IP}:{ARM_PORT}", time.monotonic() - t0)
        return
    try:
        cmd = {"T": 104, "x": HOME_X, "y": HOME_Y, "z": HOME_Z, "t": SPEED_NORMAL}
        _ = _http_json_cmd(cmd, timeout=timeout)
        arrived, last, polls = _wait_arrival((HOME_X, HOME_Y, HOME_Z), timeout=ARM_ARRIVAL_TIMEOUT)
        if arrived:
            report.add("arm", "move+arrival pipeline", "PASS", f"arrived polls={polls} pos={last}", time.monotonic() - t0)
        else:
            report.add("arm", "move+arrival pipeline", "WARN", f"timeout polls={polls} last={last}", time.monotonic() - t0)
    except Exception as exc:
        report.add("arm", "move+arrival pipeline", "FAIL", str(exc), time.monotonic() - t0)


def check_pick_place_dry_run(report: Report, timeout: float) -> None:
    t0 = time.monotonic()
    if not _tcp_reachable(ARM_IP, ARM_PORT, timeout=max(0.8, timeout)):
        report.add("arm", "pick/place dry run", "SKIP", f"arm unreachable at {ARM_IP}:{ARM_PORT}", time.monotonic() - t0)
        return
    # Corner A1 and first black graveyard slot for safe deterministic path.
    pick_x = float(CELL_CENTERS_X[0])
    pick_y = float(CELL_CENTERS_Y[0])
    place_x, place_y = GRAVEYARD_BLACK[0]
    try:
        steps = [
            {"T": 104, "x": HOME_X, "y": HOME_Y, "z": Z_SAFE, "t": SPEED_NORMAL},
            {"T": 104, "x": pick_x, "y": pick_y, "z": Z_SAFE, "t": SPEED_NORMAL},
            {"T": 104, "x": pick_x, "y": pick_y, "z": Z_PICK, "t": SPEED_NORMAL},
            {"T": 104, "x": pick_x, "y": pick_y, "z": Z_SAFE, "t": SPEED_NORMAL},
            {"T": 104, "x": place_x, "y": place_y, "z": Z_SAFE, "t": SPEED_NORMAL},
            {"T": 104, "x": place_x, "y": place_y, "z": Z_PLACE, "t": SPEED_NORMAL},
            {"T": 104, "x": place_x, "y": place_y, "z": Z_SAFE, "t": SPEED_NORMAL},
            {"T": 104, "x": HOME_X, "y": HOME_Y, "z": Z_SAFE, "t": SPEED_NORMAL},
        ]

        for step in steps:
            _ = _http_json_cmd(step, timeout=timeout)
            target = (float(step["x"]), float(step["y"]), float(step["z"]))
            arrived, last, polls = _wait_arrival(target, timeout=ARM_ARRIVAL_TIMEOUT)
            if not arrived:
                raise RuntimeError(f"step arrival timeout target={target} polls={polls} last={last}")

        # Magnet cycle embedded in dry run for full action path.
        if serial is not None:
            conn = _open_serial_with_retry()
            conn.write(b"1")
            conn.flush()
            time.sleep(GRIPPER_ACTIVATE_TIME)
            conn.write(b"0")
            conn.flush()
            conn.close()
            time.sleep(0.15)

        arrived, last, polls = _wait_arrival((HOME_X, HOME_Y, Z_SAFE), timeout=ARM_ARRIVAL_TIMEOUT)
        if arrived:
            report.add("arm", "pick/place dry run", "PASS", f"completed polls={polls} final={last}", time.monotonic() - t0)
        else:
            report.add("arm", "pick/place dry run", "WARN", f"sequence sent, final uncertain last={last}", time.monotonic() - t0)
    except Exception as exc:
        report.add("arm", "pick/place dry run", "FAIL", str(exc), time.monotonic() - t0)


def check_magnet_cycle(report: Report) -> None:
    t0 = time.monotonic()
    if serial is None:
        report.add("serial", "magnet ON/OFF cycle", "WARN", "pyserial unavailable", time.monotonic() - t0)
        return
    conn = None
    try:
        conn = _open_serial_with_retry()
        conn.write(b"1")
        conn.flush()
        time.sleep(GRIPPER_ACTIVATE_TIME)
        conn.write(b"0")
        conn.flush()
        report.add("serial", "magnet ON/OFF cycle", "PASS", "sent '1' then '0'", time.monotonic() - t0)
    except Exception as exc:
        report.add("serial", "magnet ON/OFF cycle", "FAIL", str(exc), time.monotonic() - t0)
    finally:
        if conn is not None:
            try:
                conn.close()
                time.sleep(0.15)
            except Exception:
                pass


def check_servo_cycle(report: Report, timeout: float) -> None:
    t0 = time.monotonic()
    use_servo = os.getenv("ROBO_USE_SERVO_GRIPPER", "0") == "1"
    if not use_servo:
        report.add("arm", "servo gripper cycle", "SKIP", "ROBO_USE_SERVO_GRIPPER!=1", time.monotonic() - t0)
        return
    try:
        sid = int(os.getenv("ROBO_GRIPPER_SERVO_ID", "5"))
        open_angle = int(os.getenv("ROBO_GRIPPER_OPEN_ANGLE", "90"))
        close_angle = int(os.getenv("ROBO_GRIPPER_CLOSE_ANGLE", "20"))
        spd = int(os.getenv("ROBO_GRIPPER_SERVO_SPEED", "250"))
        _ = _http_json_cmd({"T": 11, "id": sid, "angle": close_angle, "t": spd}, timeout=timeout)
        _ = _http_json_cmd({"T": 11, "id": sid, "angle": open_angle, "t": spd}, timeout=timeout)
        report.add("arm", "servo gripper cycle", "PASS", f"id={sid} close={close_angle} open={open_angle}", time.monotonic() - t0)
    except Exception as exc:
        report.add("arm", "servo gripper cycle", "WARN", str(exc), time.monotonic() - t0)


def check_serial_loopback(report: Report) -> None:
    t0 = time.monotonic()
    if serial is None:
        report.add("serial", "serial loopback", "SKIP", "pyserial unavailable", time.monotonic() - t0)
        return
    conn = None
    try:
        conn = _open_serial_with_retry()
        conn.reset_input_buffer()
        conn.write(b"1")
        conn.flush()
        time.sleep(0.15)
        data = conn.read(64)
        if data:
            report.add("serial", "serial loopback", "PASS", f"echo_bytes={len(data)}", time.monotonic() - t0)
        else:
            report.add("serial", "serial loopback", "WARN", "no echo bytes (may be normal for one-way firmware)", time.monotonic() - t0)
    except Exception as exc:
        report.add("serial", "serial loopback", "WARN", str(exc), time.monotonic() - t0)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def check_serial(report: Report, write_test: bool) -> None:
    if serial is None:
        report.add("serial", "pyserial import", "SKIP", "pyserial not installed", 0.0)
        return

    conn = None
    t0 = time.monotonic()
    try:
        conn = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
        report.add("serial", "port open", "PASS", f"{SERIAL_PORT} @ {BAUD_RATE}", time.monotonic() - t0)
        if write_test:
            t1 = time.monotonic()
            conn.write(b"0")
            conn.flush()
            report.add("serial", "safe write test", "PASS", "wrote safe payload '0'", time.monotonic() - t1)
    except Exception as exc:
        report.add("serial", "port open", "WARN", str(exc), time.monotonic() - t0)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def check_camera(report: Report, timeout: float, stream_seconds: float, save_frame: str) -> None:
    t0 = time.monotonic()
    if not _tcp_reachable(CAMERA_IP, CAMERA_PORT, timeout=max(0.8, timeout)):
        report.add("camera", "frame receive/decode", "SKIP", f"camera unreachable at {CAMERA_IP}:{CAMERA_PORT}", time.monotonic() - t0)
        return
    sock = None
    frame = None
    first_size = 0
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(max(timeout, CAMERA_CONNECT_TIMEOUT))
        sock.connect((CAMERA_IP, CAMERA_PORT))

        sock.settimeout(max(timeout, CAMERA_RECV_TIMEOUT))
        header = _recv_exact(sock, 8)
        if header is None:
            raise RuntimeError("camera stream closed before header")

        size = struct.unpack("Q", header)[0]
        if size <= 0 or size > 20_000_000:
            raise RuntimeError(f"invalid frame size: {size}")

        payload = _recv_exact(sock, size)
        if payload is None:
            raise RuntimeError("camera stream closed before payload")

        arr = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("failed to decode jpeg frame")
        first_size = int(size)

        report.add("camera", "frame receive/decode", "PASS", f"shape={frame.shape} bytes={size}", time.monotonic() - t0)
    except Exception as exc:
        report.add("camera", "frame receive/decode", "FAIL", str(exc), time.monotonic() - t0)
        if sock is not None:
            sock.close()
        return

    if save_frame:
        t0 = time.monotonic()
        try:
            ok = cv2.imwrite(save_frame, frame)
            if ok:
                report.add("camera", "save frame", "PASS", f"saved={save_frame}", time.monotonic() - t0)
            else:
                report.add("camera", "save frame", "WARN", f"cv2.imwrite returned False for {save_frame}", time.monotonic() - t0)
        except Exception as exc:
            report.add("camera", "save frame", "WARN", str(exc), time.monotonic() - t0)

    # Stream quality probe
    frames = 1
    sizes = [first_size]
    start = time.monotonic()
    try:
        while (time.monotonic() - start) < max(0.5, stream_seconds):
            header = _recv_exact(sock, 8)
            if header is None:
                break
            size = struct.unpack("Q", header)[0]
            if size <= 0 or size > 20_000_000:
                break
            payload = _recv_exact(sock, size)
            if payload is None:
                break
            frames += 1
            sizes.append(int(size))
    except Exception:
        pass
    finally:
        if sock is not None:
            sock.close()

    elapsed = max(0.001, time.monotonic() - start)
    fps = frames / elapsed
    avg_size = sum(sizes) / max(1, len(sizes))
    report.add(
        "camera",
        "stream quality",
        "PASS" if fps >= 4.0 else "WARN",
        f"frames={frames} duration_s={elapsed:.2f} fps={fps:.2f} avg_bytes={avg_size:.0f}",
        elapsed,
    )


def check_perception(report: Report, timeout: float, samples: int) -> None:
    t0 = time.monotonic()
    if not _tcp_reachable(CAMERA_IP, CAMERA_PORT, timeout=max(0.8, timeout)):
        report.add("perception", "camera connect", "SKIP", f"camera unreachable at {CAMERA_IP}:{CAMERA_PORT}", time.monotonic() - t0)
        return
    perc = PerceptionSystem()
    try:
        # Direct frame + marker inventory check.
        if not perc.connect_camera():
            report.add("perception", "camera connect", "FAIL", "connect_camera failed", time.monotonic() - t0)
            return

        frame = perc.recv_frame()
        if frame is not None:
            corners, ids = perc.detect_markers(frame)
            detected = set() if ids is None else {int(x) for x in ids.flatten()}
            expected = set(ARUCO_PIECE_IDS) | set(ARUCO_CORNER_IDS)
            missing = sorted(expected - detected)
            extras = sorted(detected - expected)
            if missing:
                report.add(
                    "perception",
                    "marker inventory",
                    "WARN",
                    f"detected={sorted(detected)} missing={missing[:10]} extras={extras[:10]}",
                    0.0,
                )
            else:
                report.add("perception", "marker inventory", "PASS", f"detected={sorted(detected)}", 0.0)

            # Homography quality check surfaced in report.
            if ids is not None and len(ids) >= 4 and perc.compute_homography(corners, ids):
                pixel_pts = []
                world_pts = []
                for idx, marker_id in enumerate(ids.flatten()):
                    marker_id = int(marker_id)
                    if marker_id in ARUCO_CORNER_IDS:
                        mc = corners[idx][0]
                        pixel_pts.append([float(np.mean(mc[:, 0])), float(np.mean(mc[:, 1]))])
                        world_pts.append(list(CORNER_WORLD_COORDS[marker_id]))
                if len(pixel_pts) >= 4 and perc.H_matrix is not None:
                    pixel_arr = np.array(pixel_pts, dtype=np.float32)
                    world_arr = np.array(world_pts, dtype=np.float32)
                    reproj = cv2.perspectiveTransform(pixel_arr.reshape(-1, 1, 2), perc.H_matrix).reshape(-1, 2)
                    err = float(np.mean(np.linalg.norm(reproj - world_arr, axis=1)))
                    status = "PASS" if err <= 50.0 else "WARN"
                    report.add("perception", "homography reprojection", status, f"error_mm={err:.2f}", 0.0)
                else:
                    report.add("perception", "homography reprojection", "WARN", "insufficient corner markers", 0.0)
            else:
                report.add("perception", "homography reprojection", "WARN", "homography unavailable", 0.0)
        else:
            report.add("perception", "marker inventory", "WARN", "no frame for marker scan", 0.0)

        ok = perc.start_background()
        if not ok:
            report.add("perception", "start background", "FAIL", "camera connection failed", time.monotonic() - t0)
            return
        report.add("perception", "start background", "PASS", elapsed_s=time.monotonic() - t0)

        t1 = time.monotonic()
        stable = perc.wait_for_stable_board(timeout=max(timeout * 2, 6.0))
        if stable is None:
            report.add("perception", "stable board", "FAIL", "no stable board", time.monotonic() - t1)
            return

        nonzero = int(np.count_nonzero(stable))
        report.add("perception", "stable board", "PASS", f"pieces={nonzero} shape={stable.shape}", time.monotonic() - t1)
        if nonzero == 12:
            report.add("perception", "initial piece count", "PASS", "exactly 12 detected", 0.0)
        else:
            report.add("perception", "initial piece count", "WARN", f"expected 12 got {nonzero}", 0.0)

        # Consistency sampling
        t2 = time.monotonic()
        last = stable
        diffs = []
        observed = 0
        for _ in range(max(1, samples)):
            board = perc.wait_for_stable_board(timeout=max(1.5, timeout))
            if board is None:
                continue
            observed += 1
            diffs.append(int(np.count_nonzero(board != last)))
            last = board
        if observed == 0:
            report.add("perception", "consistency sampling", "WARN", "no additional stable boards", time.monotonic() - t2)
        else:
            max_diff = max(diffs) if diffs else 0
            avg_diff = (sum(diffs) / len(diffs)) if diffs else 0.0
            status = "PASS" if max_diff <= 6 else "WARN"
            report.add(
                "perception",
                "consistency sampling",
                status,
                f"observed={observed} max_diff={max_diff} avg_diff={avg_diff:.2f}",
                time.monotonic() - t2,
            )
    except Exception as exc:
        report.add("perception", "pipeline", "FAIL", str(exc), time.monotonic() - t0)
    finally:
        try:
            perc.close()
        except Exception:
            pass


def check_engine_sanity(report: Report) -> None:
    t0 = time.monotonic()
    try:
        game_mod = _load_game_module()
        if game_mod is None:
            report.add("engine", "load game module", "WARN", "L2 game.py not found", time.monotonic() - t0)
            return

        initial = np.array([
            [2, 3, 4, 5, 3, 2],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [6, 6, 6, 6, 6, 6],
            [7, 8, 9, 10, 8, 7],
        ], dtype=int)

        get_best = getattr(game_mod, "get_best_move", None)
        if get_best is None:
            report.add("engine", "get_best_move", "FAIL", "missing function", time.monotonic() - t0)
            return

        mv = get_best(initial, True)
        if mv is None:
            report.add("engine", "move generation", "FAIL", "returned None", time.monotonic() - t0)
            return

        ok = bool(MOVE_REGEX.match(str(mv).strip()))
        if ok:
            report.add("engine", "move generation", "PASS", f"move={mv}", time.monotonic() - t0)
        else:
            report.add("engine", "move generation", "FAIL", f"invalid format move={mv}", time.monotonic() - t0)
    except Exception as exc:
        report.add("engine", "move generation", "FAIL", str(exc), time.monotonic() - t0)


def write_json_report(report: Report, json_out: str) -> None:
    if not json_out:
        return
    path = Path(json_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_json(), ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"JSON report written: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="L2-3 full setup debugger")
    parser.add_argument("--full", action="store_true", help="run all checks")
    parser.add_argument("--system", action="store_true", help="run system/env checks")
    parser.add_argument("--config-check", action="store_true", help="run config integrity checks")
    parser.add_argument("--network", action="store_true", help="run network checks")
    parser.add_argument("--wifi-check", action="store_true", help="run ping packet-loss check to arm IP")
    parser.add_argument("--arm", action="store_true", help="run arm checks")
    parser.add_argument("--magnet-check", action="store_true", help="run electromagnet ON/OFF cycle")
    parser.add_argument("--arm-arrival-check", action="store_true", help="run arm move+arrival polling validation")
    parser.add_argument("--pick-place-dry-run", action="store_true", help="run full pick/place dry sequence")
    parser.add_argument("--servo-check", action="store_true", help="run servo open/close check if enabled")
    parser.add_argument("--camera", action="store_true", help="run camera checks")
    parser.add_argument("--serial", action="store_true", help="run serial checks")
    parser.add_argument("--serial-loopback", action="store_true", help="attempt serial loopback read-after-write")
    parser.add_argument("--perception", action="store_true", help="run perception checks")
    parser.add_argument("--engine-check", action="store_true", help="run game engine sanity check")
    parser.add_argument("--timeout", type=float, default=4.0, help="timeout seconds for network operations")
    parser.add_argument("--network-probes", type=int, default=3, help="number of tcp probes per endpoint")
    parser.add_argument("--ping-count", type=int, default=5, help="number of ICMP pings for wifi check")
    parser.add_argument("--arm-feedback-samples", type=int, default=3, help="number of arm feedback samples")
    parser.add_argument("--camera-stream-seconds", type=float, default=5.0, help="camera stream quality duration")
    parser.add_argument("--perception-samples", type=int, default=3, help="number of extra perception stable board samples")
    parser.add_argument("--arm-motion-check", action="store_true", help="send a safe T:104 command to current reported pose")
    parser.add_argument("--serial-write-test", action="store_true", help="send safe serial payload '0' after opening port")
    parser.add_argument("--min-free-mb", type=float, default=20.0, help="minimum free disk MB warning threshold")
    parser.add_argument("--save-frame", type=str, default="", help="save first decoded camera frame to this path")
    parser.add_argument("--json-out", type=str, default="", help="write structured json report")
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_any_specific = any(
        (
            args.system,
            args.config_check,
            args.network,
            args.wifi_check,
            args.arm,
            args.magnet_check,
            args.arm_arrival_check,
            args.pick_place_dry_run,
            args.servo_check,
            args.camera,
            args.serial,
            args.serial_loopback,
            args.perception,
            args.engine_check,
        )
    )
    run_all = args.full or (not run_any_specific)

    report = Report()

    print("=" * 96)
    print("L2-3 HARDWARE DEBUG CHECKER")
    print(f"ARM: {ARM_BASE_URL}")
    print(f"CAMERA: {CAMERA_IP}:{CAMERA_PORT}")
    print(f"SERIAL: {SERIAL_PORT} @ {BAUD_RATE}")
    print(f"PYTHON: {sys.version.split()[0]}")
    print("=" * 96)

    if run_all or args.system:
        check_system(report)
        check_disk_space(report, min_mb=args.min_free_mb)
    if run_all or args.config_check:
        check_config_integrity(report)
    if run_all or args.network:
        check_network(report, timeout=args.timeout, probes=args.network_probes)
    if run_all or args.wifi_check:
        check_wifi_ping(report, ping_count=args.ping_count, timeout=args.timeout)
    if run_all or args.arm:
        check_arm(
            report,
            timeout=args.timeout,
            feedback_samples=args.arm_feedback_samples,
            motion_check=args.arm_motion_check,
        )
    if run_all or args.magnet_check:
        check_magnet_cycle(report)
    if run_all or args.servo_check:
        check_servo_cycle(report, timeout=args.timeout)
    if run_all or args.arm_arrival_check:
        check_arm_arrival_pipeline(report, timeout=args.timeout)
    if run_all or args.pick_place_dry_run:
        check_pick_place_dry_run(report, timeout=args.timeout)
    if run_all or args.camera:
        check_camera(report, timeout=args.timeout, stream_seconds=args.camera_stream_seconds, save_frame=args.save_frame)
    if run_all or args.serial:
        check_serial(report, write_test=args.serial_write_test)
    if run_all or args.serial_loopback:
        check_serial_loopback(report)
    if run_all or args.perception:
        check_perception(report, timeout=args.timeout, samples=args.perception_samples)
    if run_all or args.engine_check:
        check_engine_sanity(report)

    ok = report.print()
    write_json_report(report, args.json_out)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
