"""
Team L2 hardware preflight diagnostic.

Purpose:
- Validate connectivity before running main.py
- Distinguish expected move-send timeout behavior from true communication failure
- Produce a clear report for quick debugging

Run:
  python hardware_preflight_l2.py
  python hardware_preflight_l2.py --move-test
  python hardware_preflight_l2.py --json
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from dataclasses import dataclass, asdict
from typing import Optional
from urllib.parse import quote
from urllib.request import urlopen

from config import (
    ARM_BASE_URL,
    ARM_IP,
    ARM_PORT,
    BAUD_RATE,
    HOME_X,
    HOME_Y,
    HOME_Z,
    HTTP_FEEDBACK_TIMEOUT,
    HTTP_MOVE_SEND_TIMEOUT,
    SERIAL_PORT,
    SERIAL_TIMEOUT,
    SPEED_FAST,
)

try:
    import serial  # type: ignore
except ImportError:
    serial = None


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def check_tcp_ping(timeout: float = 2.0) -> CheckResult:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    t0 = time.monotonic()
    try:
        s.connect((ARM_IP, ARM_PORT))
        dt = (time.monotonic() - t0) * 1000.0
        return CheckResult("network_tcp", True, f"connect ok in {dt:.1f} ms")
    except OSError as exc:
        return CheckResult("network_tcp", False, f"connect failed: {exc}")
    finally:
        s.close()


def http_cmd(payload: dict, timeout: float, accept_timeout: bool = False) -> CheckResult:
    url = f"{ARM_BASE_URL}/js?json={quote(json.dumps(payload))}"
    t0 = time.monotonic()
    try:
        with urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace").strip()
        dt = (time.monotonic() - t0) * 1000.0
        return CheckResult("http_cmd", True, f"ok in {dt:.1f} ms, body={body[:140]}")
    except socket.timeout:
        if accept_timeout:
            dt = (time.monotonic() - t0) * 1000.0
            return CheckResult("http_cmd", True, f"timeout after {dt:.1f} ms (accepted for move-send)")
        return CheckResult("http_cmd", False, "socket timeout")
    except Exception as exc:
        text = str(exc)
        if accept_timeout and "timed out" in text.lower():
            return CheckResult("http_cmd", True, f"timeout accepted for move-send: {text}")
        return CheckResult("http_cmd", False, text)


def check_feedback() -> CheckResult:
    res = http_cmd({"T": 105}, timeout=HTTP_FEEDBACK_TIMEOUT, accept_timeout=False)
    res.name = "http_feedback"
    return res


def check_move_send() -> CheckResult:
    payload = {"T": 104, "x": HOME_X, "y": HOME_Y, "z": HOME_Z, "t": SPEED_FAST}
    res = http_cmd(payload, timeout=HTTP_MOVE_SEND_TIMEOUT, accept_timeout=True)
    res.name = "http_move_send"
    return res


def check_serial() -> CheckResult:
    if serial is None:
        return CheckResult("serial", False, "pyserial not installed")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
        ser.close()
        return CheckResult("serial", True, f"opened {SERIAL_PORT} @ {BAUD_RATE}")
    except Exception as exc:
        return CheckResult("serial", False, str(exc))


def run_preflight(move_test: bool = False) -> list[CheckResult]:
    results: list[CheckResult] = []
    results.append(check_tcp_ping())
    results.append(check_feedback())
    results.append(check_serial())
    if move_test:
        results.append(check_move_send())
    return results


def print_report(results: list[CheckResult]) -> int:
    print("=" * 58)
    print("TEAM L2 HARDWARE PREFLIGHT")
    print("=" * 58)
    failed = 0
    for r in results:
        mark = "PASS" if r.ok else "FAIL"
        print(f"[{mark}] {r.name:15s} | {r.detail}")
        if not r.ok:
            failed += 1

    print("=" * 58)
    if failed == 0:
        print("RESULT: READY")
    else:
        print(f"RESULT: NOT READY ({failed} checks failed)")
    return 0 if failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Team L2 preflight hardware diagnostics")
    parser.add_argument("--move-test", action="store_true", help="send one move command and classify timeout behavior")
    parser.add_argument("--json", action="store_true", help="print machine-readable json")
    args = parser.parse_args()

    results = run_preflight(move_test=args.move_test)
    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
        return 0 if all(r.ok for r in results) else 1

    return print_report(results)


if __name__ == "__main__":
    raise SystemExit(main())
