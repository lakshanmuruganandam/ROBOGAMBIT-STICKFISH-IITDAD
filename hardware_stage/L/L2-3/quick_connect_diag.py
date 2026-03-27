"""
Quick connectivity diagnostics for L2-3 hardware debugging.

Run:
  python quick_connect_diag.py
  python quick_connect_diag.py --json-out diag.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from config import ARM_BASE_URL, ARM_IP, ARM_PORT, CAMERA_IP, CAMERA_PORT, SERIAL_PORT  # noqa: E402


def _run_cmd(args: List[str], timeout: float = 4.0) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as exc:
        return 999, "", str(exc)


def _tcp_probe(ip: str, port: int, timeout: float) -> Dict[str, Any]:
    t0 = time.monotonic()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        code = sock.connect_ex((ip, port))
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        return {
            "ok": code == 0,
            "code": int(code),
            "elapsed_ms": round(elapsed_ms, 2),
        }
    except Exception as exc:
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        return {
            "ok": False,
            "code": -1,
            "elapsed_ms": round(elapsed_ms, 2),
            "error": str(exc),
        }
    finally:
        try:
            sock.close()
        except Exception:
            pass


def _http_probe(url: str, timeout: float) -> Dict[str, Any]:
    try:
        import requests

        t0 = time.monotonic()
        r = requests.get(url, timeout=timeout)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        return {
            "ok": True,
            "status": int(r.status_code),
            "elapsed_ms": round(elapsed_ms, 2),
            "body_prefix": (r.text or "")[:120],
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
        }


def _camera_header_probe(ip: str, port: int, timeout: float) -> Dict[str, Any]:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        t0 = time.monotonic()
        s.connect((ip, port))
        hdr = s.recv(8)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        return {
            "ok": len(hdr) == 8,
            "header_len": len(hdr),
            "header_hex": hdr.hex(),
            "elapsed_ms": round(elapsed_ms, 2),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
        }
    finally:
        try:
            s.close()
        except Exception:
            pass


def _route_get(ip: str) -> Dict[str, Any]:
    if platform.system().lower().startswith("linux"):
        code, out, err = _run_cmd(["ip", "route", "get", ip], timeout=3.0)
        return {"code": code, "out": out, "err": err}
    return {"code": -1, "out": "", "err": "route check only implemented for Linux"}


def _list_serial_ports() -> Dict[str, Any]:
    out: Dict[str, Any] = {"configured": SERIAL_PORT, "detected": []}
    try:
        import serial.tools.list_ports

        ports = []
        for p in serial.tools.list_ports.comports():
            ports.append({"device": p.device, "description": p.description})
        out["detected"] = ports
    except Exception as exc:
        out["error"] = str(exc)
    return out


def build_report(timeout: float, arm_ip: str, arm_port: int, camera_ip: str, camera_port: int) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "hostname": socket.gethostname(),
            "cwd": str(Path.cwd()),
        },
        "targets": {
            "arm_ip": arm_ip,
            "arm_port": arm_port,
            "camera_ip": camera_ip,
            "camera_port": camera_port,
            "arm_base_url": f"http://{arm_ip}:{arm_port}",
            "configured_arm_base_url": ARM_BASE_URL,
            "configured_serial_port": SERIAL_PORT,
            "env": {
                "ROBO_ARM_IP": os.getenv("ROBO_ARM_IP"),
                "ROBO_ARM_PORT": os.getenv("ROBO_ARM_PORT"),
                "ROBO_CAMERA_IP": os.getenv("ROBO_CAMERA_IP"),
                "ROBO_CAMERA_PORT": os.getenv("ROBO_CAMERA_PORT"),
                "ROBO_SERIAL_PORT": os.getenv("ROBO_SERIAL_PORT"),
            },
        },
        "network": {
            "arm_route": _route_get(arm_ip),
            "camera_route": _route_get(camera_ip),
            "arm_tcp": _tcp_probe(arm_ip, arm_port, timeout),
            "camera_tcp": _tcp_probe(camera_ip, camera_port, timeout),
            "arm_http_root": _http_probe(f"http://{arm_ip}:{arm_port}/", timeout),
            "camera_header": _camera_header_probe(camera_ip, camera_port, timeout),
        },
        "serial": _list_serial_ports(),
    }
    return report


def print_report(rep: Dict[str, Any]) -> None:
    print("=" * 90)
    print("QUICK CONNECT DIAGNOSTICS")
    print("=" * 90)
    print(f"Host: {rep['host']['hostname']} | {rep['host']['platform']} | py={rep['host']['python']}")
    print(f"ARM:  {rep['targets']['arm_ip']}:{rep['targets']['arm_port']}")
    print(f"CAM:  {rep['targets']['camera_ip']}:{rep['targets']['camera_port']}")
    print(f"SER:  configured={rep['targets']['configured_serial_port']}")
    print("-" * 90)

    net = rep["network"]
    print(f"arm tcp:      {net['arm_tcp']}")
    print(f"camera tcp:   {net['camera_tcp']}")
    print(f"arm http:     {net['arm_http_root']}")
    print(f"camera hdr:   {net['camera_header']}")
    print(f"arm route:    {net['arm_route']}")
    print(f"camera route: {net['camera_route']}")
    print("-" * 90)
    print(f"serial ports: {rep['serial']}")
    print("=" * 90)


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick connectivity diagnostics for L2-3")
    parser.add_argument("--timeout", type=float, default=4.0, help="socket/http timeout seconds")
    parser.add_argument("--arm-ip", type=str, default=ARM_IP, help="override arm IP")
    parser.add_argument("--arm-port", type=int, default=ARM_PORT, help="override arm port")
    parser.add_argument("--camera-ip", type=str, default=CAMERA_IP, help="override camera IP")
    parser.add_argument("--camera-port", type=int, default=CAMERA_PORT, help="override camera port")
    parser.add_argument("--json-out", type=str, default="", help="save full report as json")
    args = parser.parse_args()

    rep = build_report(
        timeout=max(0.5, args.timeout),
        arm_ip=args.arm_ip,
        arm_port=args.arm_port,
        camera_ip=args.camera_ip,
        camera_port=args.camera_port,
    )
    print_report(rep)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rep, indent=2, ensure_ascii=True), encoding="utf-8")
        print(f"JSON saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
