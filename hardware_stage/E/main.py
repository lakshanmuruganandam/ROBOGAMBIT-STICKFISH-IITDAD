"""
Simple E-stage main script for bring-up and debugging.

Goals:
- Keep behavior easy to reason about.
- Validate arm/camera/serial first.
- Use E folder game.py and perception.py directly.

Typical usage:
  python main.py --checks-only
  python main.py --one-move --white
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests

import game
from config import (
    ARM_BASE_URL,
    ARM_IP,
    ARM_PORT,
    BAUD_RATE,
    CAMERA_IP,
    CAMERA_PORT,
    HTTP_FEEDBACK_TIMEOUT,
    MOVE_REGEX,
    SERIAL_PORT,
    SERIAL_TIMEOUT,
)
from perception import PerceptionSystem

try:
    import serial  # type: ignore
except Exception:
    serial = None


@dataclass
class CheckRow:
    name: str
    status: str
    message: str
    elapsed_s: float


def _tcp_probe(ip: str, port: int, timeout: float) -> Dict[str, object]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    t0 = time.monotonic()
    try:
        code = sock.connect_ex((ip, port))
        return {
            "ok": code == 0,
            "code": int(code),
            "elapsed_s": round(time.monotonic() - t0, 3),
        }
    except Exception as exc:
        return {"ok": False, "code": -1, "error": str(exc), "elapsed_s": round(time.monotonic() - t0, 3)}
    finally:
        try:
            sock.close()
        except Exception:
            pass


def _arm_http_check(timeout: float) -> Dict[str, object]:
    t0 = time.monotonic()
    try:
        r = requests.get(f"{ARM_BASE_URL}/", timeout=timeout)
        return {
            "ok": True,
            "status": int(r.status_code),
            "elapsed_s": round(time.monotonic() - t0, 3),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "elapsed_s": round(time.monotonic() - t0, 3)}


def _arm_feedback_check(timeout: float) -> Dict[str, object]:
    t0 = time.monotonic()
    try:
        payload = {"T": 105}
        r = requests.get(f"{ARM_BASE_URL}/js", params={"json": json.dumps(payload)}, timeout=timeout)
        r.raise_for_status()
        data = r.json() if r.text else {}
        x = float(data.get("x", 0.0))
        y = float(data.get("y", 0.0))
        z = float(data.get("z", 0.0))
        return {
            "ok": True,
            "pose": (x, y, z),
            "elapsed_s": round(time.monotonic() - t0, 3),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "elapsed_s": round(time.monotonic() - t0, 3)}


def _serial_check() -> Dict[str, object]:
    t0 = time.monotonic()
    if serial is None:
        return {"ok": False, "error": "pyserial not installed", "elapsed_s": round(time.monotonic() - t0, 3)}
    conn = None
    try:
        conn = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
        conn.write(b"0")
        conn.flush()
        return {"ok": True, "message": f"opened {SERIAL_PORT} and sent safe byte", "elapsed_s": round(time.monotonic() - t0, 3)}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "elapsed_s": round(time.monotonic() - t0, 3)}
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _camera_check() -> Dict[str, object]:
    t0 = time.monotonic()
    p = PerceptionSystem()
    try:
        if not p.connect_camera():
            return {"ok": False, "error": "connect_camera failed", "elapsed_s": round(time.monotonic() - t0, 3)}
        frame = p.recv_frame()
        if frame is None:
            return {"ok": False, "error": "connected but no frame", "elapsed_s": round(time.monotonic() - t0, 3)}
        return {
            "ok": True,
            "shape": tuple(frame.shape),
            "elapsed_s": round(time.monotonic() - t0, 3),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "elapsed_s": round(time.monotonic() - t0, 3)}
    finally:
        try:
            p.close()
        except Exception:
            pass


def run_preflight(timeout: float) -> List[CheckRow]:
    rows: List[CheckRow] = []

    a_tcp = _tcp_probe(ARM_IP, ARM_PORT, timeout=timeout)
    rows.append(
        CheckRow(
            name="arm tcp",
            status="PASS" if a_tcp.get("ok") else "FAIL",
            message=str(a_tcp),
            elapsed_s=float(a_tcp.get("elapsed_s", 0.0)),
        )
    )

    a_http = _arm_http_check(timeout=timeout)
    rows.append(
        CheckRow(
            name="arm http",
            status="PASS" if a_http.get("ok") else "FAIL",
            message=str(a_http),
            elapsed_s=float(a_http.get("elapsed_s", 0.0)),
        )
    )

    a_fb = _arm_feedback_check(timeout=max(timeout, HTTP_FEEDBACK_TIMEOUT))
    rows.append(
        CheckRow(
            name="arm feedback",
            status="PASS" if a_fb.get("ok") else "WARN",
            message=str(a_fb),
            elapsed_s=float(a_fb.get("elapsed_s", 0.0)),
        )
    )

    c_tcp = _tcp_probe(CAMERA_IP, CAMERA_PORT, timeout=timeout)
    rows.append(
        CheckRow(
            name="camera tcp",
            status="PASS" if c_tcp.get("ok") else "FAIL",
            message=str(c_tcp),
            elapsed_s=float(c_tcp.get("elapsed_s", 0.0)),
        )
    )

    cam = _camera_check()
    rows.append(
        CheckRow(
            name="camera frame",
            status="PASS" if cam.get("ok") else "FAIL",
            message=str(cam),
            elapsed_s=float(cam.get("elapsed_s", 0.0)),
        )
    )

    ser = _serial_check()
    rows.append(
        CheckRow(
            name="serial",
            status="PASS" if ser.get("ok") else "WARN",
            message=str(ser),
            elapsed_s=float(ser.get("elapsed_s", 0.0)),
        )
    )

    return rows


def print_preflight(rows: List[CheckRow]) -> bool:
    print("=" * 88)
    print("E MAIN PREFLIGHT")
    print("=" * 88)
    print(f"ARM: {ARM_IP}:{ARM_PORT}  CAMERA: {CAMERA_IP}:{CAMERA_PORT}  SERIAL: {SERIAL_PORT}")
    print("-" * 88)
    for r in rows:
        print(f"{r.status:>5}  {r.name:14s} ({r.elapsed_s:.2f}s) - {r.message}")

    fail_count = sum(1 for r in rows if r.status == "FAIL")
    warn_count = sum(1 for r in rows if r.status == "WARN")
    print("-" * 88)
    print(f"Totals: FAIL={fail_count} WARN={warn_count}")

    # Hard gate for safe progression.
    needed = {"arm tcp", "arm http", "camera tcp", "camera frame"}
    ok_map = {r.name: (r.status == "PASS") for r in rows}
    go = all(ok_map.get(name, False) for name in needed)
    print("VERDICT:", "GO" if go else "NO-GO")
    return go


def get_board_state(timeout_s: float) -> Optional[np.ndarray]:
    p = PerceptionSystem()
    try:
        if not p.connect_camera():
            return None
        if not p.start_background():
            return None
        return p.wait_for_stable_board(timeout=timeout_s)
    finally:
        try:
            p.close()
        except Exception:
            pass


def append_move_log(move: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(move.strip() + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple E main runner")
    parser.add_argument("--checks-only", action="store_true", help="run preflight only")
    parser.add_argument("--one-move", action="store_true", help="run preflight + infer one move from stable board")
    parser.add_argument("--white", action="store_true", help="play as white when inferring one move")
    parser.add_argument("--black", action="store_true", help="play as black when inferring one move")
    parser.add_argument("--timeout", type=float, default=4.0, help="network timeout for probes")
    parser.add_argument("--board-timeout", type=float, default=10.0, help="stable board wait timeout")
    parser.add_argument("--log-file", type=str, default="game_log.txt", help="move log path")
    args = parser.parse_args()

    do_one_move = args.one_move or (not args.checks_only)
    playing_white = True
    if args.black:
        playing_white = False
    elif args.white:
        playing_white = True

    rows = run_preflight(timeout=max(0.5, args.timeout))
    go = print_preflight(rows)
    if not go:
        return 1

    if not do_one_move:
        return 0

    board = get_board_state(timeout_s=max(2.0, args.board_timeout))
    if board is None:
        print("No stable board received from perception.")
        return 2

    print("Stable board received:")
    print(board)

    move = game.get_best_move(board, playing_white=playing_white)
    if not move:
        print("Engine returned no move.")
        return 3

    move = str(move).strip()
    if MOVE_REGEX.match(move) is None:
        print(f"Engine move format invalid: {move}")
        return 4

    append_move_log(move, Path(args.log_file))
    print(f"Suggested move: {move}")
    print(f"Logged to: {Path(args.log_file).resolve()}")
    print("Execution step intentionally omitted in this simple script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
