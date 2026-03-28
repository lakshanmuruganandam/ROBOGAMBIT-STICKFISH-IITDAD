"""
Full preflight checker for hardware bring-up from E folder.

Purpose:
- Run all critical checks before attempting main gameplay scripts.
- Diagnose camera IP confusion (especially .2 vs .6 issues).
- Print a strict GO/NO-GO verdict with concrete remediation steps.

Usage examples:
  python full_preflight.py
  python full_preflight.py --camera-ip 192.168.4.2
  python full_preflight.py --camera-candidates 192.168.4.2,192.168.4.6,192.168.4.10
  python full_preflight.py --json-out preflight.json
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_ARM_IP = os.getenv("ROBO_ARM_IP", "192.168.4.1")
DEFAULT_ARM_PORT = int(os.getenv("ROBO_ARM_PORT", "80"))
DEFAULT_CAMERA_IP = os.getenv("ROBO_CAMERA_IP", "192.168.4.2")
DEFAULT_CAMERA_PORT = int(os.getenv("ROBO_CAMERA_PORT", "9994"))
DEFAULT_SERIAL_PORT = os.getenv("ROBO_SERIAL_PORT", "/dev/ttyUSB0")


@dataclass
class CheckRow:
    name: str
    status: str  # PASS | FAIL | WARN | SKIP
    message: str
    elapsed_s: float


def _run_cmd(args: Sequence[str], timeout: float = 4.0) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()
    except Exception as exc:
        return 999, "", str(exc)


def _route_get(ip: str) -> Dict[str, Any]:
    if platform.system().lower().startswith("linux"):
        code, out, err = _run_cmd(["ip", "route", "get", ip], timeout=3.0)
        return {"code": code, "out": out, "err": err}
    return {"code": -1, "out": "", "err": "route check only implemented for Linux"}


def _http_probe(ip: str, port: int, timeout: float) -> Dict[str, Any]:
    try:
        import requests

        t0 = time.monotonic()
        r = requests.get(f"http://{ip}:{port}/", timeout=timeout)
        return {
            "ok": True,
            "status": int(r.status_code),
            "elapsed_s": round(time.monotonic() - t0, 3),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _ping_host(ip: str, timeout: float) -> Dict[str, Any]:
    if not platform.system().lower().startswith("linux"):
        return {"ok": False, "skip": True, "error": "ping helper only implemented for Linux"}
    wait_s = max(1, int(timeout))
    code, out, err = _run_cmd(["ping", "-c", "1", "-W", str(wait_s), ip], timeout=timeout + 2.0)
    return {
        "ok": code == 0,
        "code": code,
        "out": out,
        "err": err,
    }


def _tcp_probe(ip: str, port: int, timeout: float, attempts: int = 2) -> Dict[str, Any]:
    last_error = ""
    for i in range(max(1, attempts)):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        t0 = time.monotonic()
        try:
            code = sock.connect_ex((ip, port))
            elapsed = time.monotonic() - t0
            if code == 0:
                return {
                    "ok": True,
                    "code": 0,
                    "elapsed_s": round(elapsed, 3),
                    "attempt": i + 1,
                }
            err_name = os.strerror(code) if code > 0 else "unknown"
            last_error = f"connect_ex={code} ({err_name})"
        except Exception as exc:
            last_error = str(exc)
        finally:
            try:
                sock.close()
            except Exception:
                pass
    return {"ok": False, "code": 1, "error": last_error, "attempts": max(1, attempts)}


def _camera_header_probe(ip: str, port: int, timeout: float) -> Dict[str, Any]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        t0 = time.monotonic()
        sock.connect((ip, port))
        hdr = sock.recv(8)
        elapsed = time.monotonic() - t0
        return {
            "ok": len(hdr) == 8,
            "header_len": len(hdr),
            "elapsed_s": round(elapsed, 3),
            "header_hex": hdr.hex(),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        try:
            sock.close()
        except Exception:
            pass


def _camera_header_probe_via_create_connection(ip: str, port: int, timeout: float) -> Dict[str, Any]:
    try:
        t0 = time.monotonic()
        with socket.create_connection((ip, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            hdr = sock.recv(8)
        return {
            "ok": len(hdr) == 8,
            "header_len": len(hdr),
            "elapsed_s": round(time.monotonic() - t0, 3),
            "header_hex": hdr.hex(),
            "method": "create_connection",
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "method": "create_connection"}


def _camera_handshake_probe(ip: str, port: int, timeout: float) -> Dict[str, Any]:
    """
    Try multiple camera stream handshake styles:
    - 8-byte header using direct socket connect
    - 8-byte header using socket.create_connection
    - fallback 4-byte header read (some custom streamers)
    """
    first = _camera_header_probe(ip, port, timeout)
    if first.get("ok"):
        first["method"] = "connect+recv8"
        return first

    second = _camera_header_probe_via_create_connection(ip, port, timeout)
    if second.get("ok"):
        return second

    # Fallback: attempt a 4-byte header read. This is not a hard success for this pipeline,
    # but it helps detect protocol mismatch where service is alive.
    try:
        t0 = time.monotonic()
        with socket.create_connection((ip, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            hdr4 = sock.recv(4)
        if len(hdr4) == 4:
            return {
                "ok": False,
                "proto_mismatch": True,
                "header4_hex": hdr4.hex(),
                "elapsed_s": round(time.monotonic() - t0, 3),
                "method": "connect+recv4",
                "error": "received 4-byte header; expected 8-byte frame length for current perception pipeline",
            }
    except Exception:
        pass

    return {
        "ok": False,
        "method": "multi-handshake",
        "error": f"all handshake methods failed; direct={first}; alt={second}",
    }


def _parse_route_src(route_out: str) -> Optional[str]:
    parts = route_out.split()
    for i, token in enumerate(parts):
        if token == "src" and i + 1 < len(parts):
            return parts[i + 1]
    return None


def _parse_route_dev(route_out: str) -> Optional[str]:
    parts = route_out.split()
    for i, token in enumerate(parts):
        if token == "dev" and i + 1 < len(parts):
            return parts[i + 1]
    return None


def _is_loopback_route(route_out: str) -> bool:
    return " dev lo " in f" {route_out} " or "<local>" in route_out


def _list_serial_ports() -> Dict[str, Any]:
    out: Dict[str, Any] = {"configured": DEFAULT_SERIAL_PORT, "detected": []}
    try:
        import serial.tools.list_ports

        ports = []
        for p in serial.tools.list_ports.comports():
            ports.append({"device": p.device, "description": p.description})
        out["detected"] = ports
    except Exception as exc:
        out["error"] = str(exc)
    return out


def _check_serial_open(timeout: float = 1.0) -> Dict[str, Any]:
    try:
        import serial

        conn = serial.Serial(DEFAULT_SERIAL_PORT, 115200, timeout=timeout)
        try:
            conn.write(b"0")
            conn.flush()
        finally:
            conn.close()
        return {"ok": True, "message": f"opened {DEFAULT_SERIAL_PORT} and sent safe byte '0'"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _candidate_camera_ips(primary: str, user_candidates: str) -> List[str]:
    seen = set()
    out: List[str] = []

    def add(ip: str) -> None:
        ip = ip.strip()
        if not ip:
            return
        if ip in seen:
            return
        seen.add(ip)
        out.append(ip)

    add(primary)
    add("127.0.0.1")
    add("192.168.4.2")
    add("192.168.4.6")
    add("192.168.4.3")
    add("192.168.4.4")
    add("192.168.4.5")
    add("192.168.4.10")
    if user_candidates:
        for ip in user_candidates.split(","):
            add(ip)
    return out


def _candidate_camera_ports(primary_port: int, user_candidates: str) -> List[int]:
    out: List[int] = []
    seen = set()

    def add(port: int) -> None:
        if port <= 0 or port > 65535:
            return
        if port in seen:
            return
        seen.add(port)
        out.append(port)

    add(primary_port)
    add(9994)
    add(9999)
    if user_candidates:
        for raw in user_candidates.split(","):
            raw = raw.strip()
            if not raw:
                continue
            try:
                add(int(raw))
            except ValueError:
                continue
    return out


def _build_fix_playbook(
    arm_tcp_ok: bool,
    arm_http_ok: bool,
    cam_route_loopback: bool,
    camera_target_is_localhost: bool,
    cam_route_dev: Optional[str],
    cam_route_src: Optional[str],
    cam_tcp_ok: bool,
    cam_hdr_ok: bool,
    best_camera_ip: Optional[str],
    configured_camera_ip: str,
    serial_detect_ok: bool,
    camera_probe_details: Dict[str, Dict[str, Any]],
) -> List[str]:
    steps: List[str] = []

    if not arm_tcp_ok:
        steps.append("Arm unreachable: verify robot power, Wi-Fi AP join, and ROBO_ARM_IP/ROBO_ARM_PORT.")
    if arm_tcp_ok and not arm_http_ok:
        steps.append("Arm TCP works but HTTP fails: check arm web service readiness on port 80.")

    if cam_route_loopback and not camera_target_is_localhost:
        steps.append("Camera route is loopback: configured camera IP points to local machine; set ROBO_CAMERA_IP to actual camera host.")
    if (not camera_target_is_localhost) and cam_route_src and cam_route_src == configured_camera_ip:
        steps.append("Camera IP equals local source IP: this is typically misconfiguration; choose camera device IP, not laptop IP.")
    if (not camera_target_is_localhost) and cam_route_dev and cam_route_dev == "lo":
        steps.append("Camera route uses loopback interface: clear wrong static route or wrong /etc/hosts mapping on remote machine.")

    if not cam_tcp_ok:
        steps.append("Camera TCP unreachable: confirm camera stream process is running and target port matches ROBO_CAMERA_PORT.")
        steps.append("If camera can ping but TCP fails, firewall/service binding is likely wrong; bind stream server to 0.0.0.0 and open port.")
        # Common on overloaded or stale local stream process: errno 11.
        if any("connect_ex=11" in str(v.get("error", "")) for v in camera_probe_details.values()):
            steps.append("Detected connect_ex=11 (Resource temporarily unavailable). Hard-restart camera server process and verify fresh LISTEN socket.")
            steps.append("Run: pkill -f 'python3.*9994' ; sleep 1 ; restart server ; ss -lntp | grep 9994 ; nc -vz -w 2 127.0.0.1 9994")
    if cam_tcp_ok and not cam_hdr_ok:
        steps.append("Camera TCP connects but no frame header: protocol mismatch or stalled stream sender.")
        steps.append("Verify stream header is 8 bytes for this pipeline using a direct localhost probe script.")
    if best_camera_ip and best_camera_ip != configured_camera_ip:
        steps.append(f"Detected alternate reachable camera IP {best_camera_ip}; export ROBO_CAMERA_IP={best_camera_ip}.")

    if not serial_detect_ok:
        steps.append("Serial port not detected: replug USB-UART and set ROBO_SERIAL_PORT to detected device path.")

    if not steps:
        steps.append("All critical checks passed. Proceed to main script run.")

    return steps


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict preflight checker from E folder")
    parser.add_argument("--timeout", type=float, default=3.5, help="socket timeout seconds")
    parser.add_argument("--arm-ip", type=str, default=DEFAULT_ARM_IP, help="override arm IP")
    parser.add_argument("--arm-port", type=int, default=DEFAULT_ARM_PORT, help="override arm port")
    parser.add_argument("--camera-ip", type=str, default=DEFAULT_CAMERA_IP, help="override camera IP")
    parser.add_argument("--camera-port", type=int, default=DEFAULT_CAMERA_PORT, help="override camera port")
    parser.add_argument("--camera-candidates", type=str, default="", help="comma-separated fallback camera IPs")
    parser.add_argument("--camera-port-candidates", type=str, default="", help="comma-separated fallback camera ports")
    parser.add_argument("--json-out", type=str, default="", help="write full result as JSON")
    args = parser.parse_args()

    checks: List[CheckRow] = []
    notes: List[str] = []
    start = time.monotonic()

    host = {
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
    }

    t0 = time.monotonic()
    arm_tcp = _tcp_probe(args.arm_ip, args.arm_port, timeout=args.timeout, attempts=2)
    if arm_tcp.get("ok"):
        checks.append(CheckRow("arm tcp", "PASS", f"{args.arm_ip}:{args.arm_port} reachable", time.monotonic() - t0))
    else:
        checks.append(CheckRow("arm tcp", "FAIL", f"{args.arm_ip}:{args.arm_port} unreachable ({arm_tcp})", time.monotonic() - t0))
        notes.append("Arm endpoint is unreachable. Check arm power, Wi-Fi AP, and ROBO_ARM_IP/ROBO_ARM_PORT.")

    t0 = time.monotonic()
    arm_http = _http_probe(args.arm_ip, args.arm_port, timeout=max(args.timeout, 2.0))
    if arm_http.get("ok"):
        checks.append(CheckRow("arm http", "PASS", str(arm_http), time.monotonic() - t0))
    else:
        status = "WARN" if arm_tcp.get("ok") else "SKIP"
        checks.append(CheckRow("arm http", status, str(arm_http), time.monotonic() - t0))

    t0 = time.monotonic()
    arm_ping = _ping_host(args.arm_ip, timeout=max(args.timeout, 2.0))
    if arm_ping.get("skip"):
        checks.append(CheckRow("arm ping", "SKIP", arm_ping.get("error", "unsupported"), time.monotonic() - t0))
    else:
        checks.append(CheckRow("arm ping", "PASS" if arm_ping.get("ok") else "WARN", f"code={arm_ping.get('code')}", time.monotonic() - t0))

    route = _route_get(args.camera_ip)
    route_out = route.get("out", "") if isinstance(route, dict) else ""
    route_src = _parse_route_src(route_out)
    route_dev = _parse_route_dev(route_out)
    cam_route_loopback = _is_loopback_route(route_out)
    camera_target_is_localhost = args.camera_ip in {"127.0.0.1", "localhost"}
    if cam_route_loopback:
        if camera_target_is_localhost:
            checks.append(CheckRow("camera route", "WARN", f"local mode route: {route_out}", 0.0))
            notes.append("Camera target is localhost; this is valid only if camera streamer runs on same machine.")
        else:
            checks.append(CheckRow("camera route", "FAIL", f"loopback route detected: {route_out}", 0.0))
            notes.append("Camera IP resolves to local loopback. Change ROBO_CAMERA_IP to actual camera device IP.")
    else:
        checks.append(CheckRow("camera route", "PASS", f"route={route_out or 'unknown'}", 0.0))
        if route_src and route_src == args.camera_ip:
            checks.append(CheckRow("camera route sanity", "WARN", f"camera IP equals local src ({route_src})", 0.0))
            notes.append("Configured camera IP matches host source IP. Likely wrong camera IP.")

    checks.append(CheckRow("camera route interface", "PASS" if route_dev and route_dev != "lo" else "WARN", f"dev={route_dev or 'unknown'} src={route_src or 'unknown'}", 0.0))

    t0 = time.monotonic()
    cam_ping = _ping_host(args.camera_ip, timeout=max(args.timeout, 2.0))
    if cam_ping.get("skip"):
        checks.append(CheckRow("camera ping", "SKIP", cam_ping.get("error", "unsupported"), time.monotonic() - t0))
    else:
        checks.append(CheckRow("camera ping", "PASS" if cam_ping.get("ok") else "WARN", f"code={cam_ping.get('code')}", time.monotonic() - t0))

    candidates = _candidate_camera_ips(args.camera_ip, args.camera_candidates)
    port_candidates = _candidate_camera_ports(args.camera_port, args.camera_port_candidates)
    best_camera_ip = None
    best_camera_port = None
    camera_probe_details: Dict[str, Dict[str, Any]] = {}
    endpoint_attempts: List[Dict[str, Any]] = []

    for ip in candidates:
        for port in port_candidates:
            probe = _tcp_probe(ip, port, timeout=args.timeout, attempts=2)
            key = f"{ip}:{port}"
            camera_probe_details[key] = probe
            endpoint_attempts.append({"ip": ip, "port": port, "tcp": probe})
            if probe.get("ok") and best_camera_ip is None:
                best_camera_ip = ip
                best_camera_port = port

    t0 = time.monotonic()
    if best_camera_ip is None:
        checks.append(CheckRow("camera tcp", "FAIL", f"no reachable camera endpoint; probes={camera_probe_details}", time.monotonic() - t0))
        notes.append("No route/host for camera stream. Verify camera process is running and IP is correct.")
    else:
        status = "PASS" if (best_camera_ip == args.camera_ip and best_camera_port == args.camera_port) else "WARN"
        msg = f"reachable at {best_camera_ip}:{best_camera_port}"
        if status == "WARN":
            msg += (
                f" (configured={args.camera_ip}:{args.camera_port}; "
                f"use ROBO_CAMERA_IP={best_camera_ip} ROBO_CAMERA_PORT={best_camera_port})"
            )
            notes.append(
                f"Camera works at {best_camera_ip}:{best_camera_port}, "
                f"not configured endpoint {args.camera_ip}:{args.camera_port}."
            )
        checks.append(CheckRow("camera tcp", status, msg, time.monotonic() - t0))

    t0 = time.monotonic()
    if best_camera_ip is not None and best_camera_port is not None:
        hdr = _camera_handshake_probe(best_camera_ip, best_camera_port, timeout=args.timeout)
        if hdr.get("ok"):
            checks.append(
                CheckRow(
                    "camera header",
                    "PASS",
                    f"8-byte header received from {best_camera_ip}:{best_camera_port} via {hdr.get('method')}",
                    time.monotonic() - t0,
                )
            )
        elif hdr.get("proto_mismatch"):
            checks.append(CheckRow("camera header", "WARN", str(hdr), time.monotonic() - t0))
            notes.append("Camera stream reachable but protocol differs (likely 4-byte header sender).")
        else:
            checks.append(CheckRow("camera header", "FAIL", f"connect ok but no valid header: {hdr}", time.monotonic() - t0))
            notes.append("Camera TCP accepts connection but does not provide frame header. Check stream server implementation.")
    else:
        checks.append(CheckRow("camera header", "SKIP", "skipped (camera tcp not reachable)", time.monotonic() - t0))

    t0 = time.monotonic()
    ports = _list_serial_ports()
    detected_devices = [p.get("device") for p in ports.get("detected", []) if isinstance(p, dict)]
    serial_detect_ok = DEFAULT_SERIAL_PORT in detected_devices
    if serial_detect_ok:
        checks.append(CheckRow("serial detect", "PASS", f"configured port present: {DEFAULT_SERIAL_PORT}", time.monotonic() - t0))
    else:
        checks.append(CheckRow("serial detect", "WARN", f"configured port not found: {DEFAULT_SERIAL_PORT}; detected={detected_devices[:8]}", time.monotonic() - t0))
        notes.append("Serial configured port missing. Set ROBO_SERIAL_PORT to detected USB-UART device.")

    t0 = time.monotonic()
    serial_open = _check_serial_open(timeout=1.0)
    if serial_open.get("ok"):
        checks.append(CheckRow("serial open/write", "PASS", serial_open.get("message", "ok"), time.monotonic() - t0))
    else:
        checks.append(CheckRow("serial open/write", "WARN", serial_open.get("error", "unknown error"), time.monotonic() - t0))

    arm_tcp_ok = bool(arm_tcp.get("ok"))
    arm_http_ok = bool(arm_http.get("ok"))
    cam_tcp_ok = (
        best_camera_ip is not None
        and best_camera_port is not None
        and best_camera_ip == args.camera_ip
        and best_camera_port == args.camera_port
    )
    cam_hdr_ok = any(c.name == "camera header" and c.status == "PASS" for c in checks)

    fix_playbook = _build_fix_playbook(
        arm_tcp_ok=arm_tcp_ok,
        arm_http_ok=arm_http_ok,
        cam_route_loopback=cam_route_loopback,
        camera_target_is_localhost=camera_target_is_localhost,
        cam_route_dev=route_dev,
        cam_route_src=route_src,
        cam_tcp_ok=cam_tcp_ok,
        cam_hdr_ok=cam_hdr_ok,
        best_camera_ip=best_camera_ip,
        configured_camera_ip=args.camera_ip,
        serial_detect_ok=serial_detect_ok,
        camera_probe_details=camera_probe_details,
    )

    fail_count = sum(1 for c in checks if c.status == "FAIL")
    warn_count = sum(1 for c in checks if c.status == "WARN")

    hard_requirements = {
        "arm tcp": False,
        "camera tcp": False,
        "camera header": False,
    }
    for c in checks:
        if c.name in hard_requirements:
            hard_requirements[c.name] = c.status == "PASS"

    go = all(hard_requirements.values())

    print("=" * 92)
    print("E FOLDER FULL PREFLIGHT")
    print("=" * 92)
    print(f"Host: {host['hostname']} | {host['platform']} | py={host['python']}")
    print(f"ARM target:    {args.arm_ip}:{args.arm_port}")
    print(f"Camera target: {args.camera_ip}:{args.camera_port}")
    print(f"Serial target: {DEFAULT_SERIAL_PORT}")
    print("-" * 92)

    for row in checks:
        dur = f" ({row.elapsed_s:.2f}s)" if row.elapsed_s > 0 else ""
        print(f"{row.status:>5}  {row.name}{dur} - {row.message}")

    print("-" * 92)
    print(f"Totals: FAIL={fail_count} WARN={warn_count}")
    print("VERDICT:", "GO" if go else "NO-GO")

    if not go:
        print("\nACTIONABLE FIXES:")
        if not hard_requirements["arm tcp"]:
            print("1. Fix arm connectivity first (power/AP/IP).")
        if not hard_requirements["camera tcp"] or not hard_requirements["camera header"]:
            print("2. Fix camera stream endpoint. Confirm camera server is running and reachable.")
            if best_camera_ip and best_camera_ip != args.camera_ip:
                print(f"3. Set ROBO_CAMERA_IP={best_camera_ip} and ROBO_CAMERA_PORT={best_camera_port} before running debug/main.")
        if notes:
            print("4. Notes:")
            for n in notes:
                print(f"   - {n}")
        print("5. Playbook:")
        for i, step in enumerate(fix_playbook, start=1):
            print(f"   {i}. {step}")
        if best_camera_ip and best_camera_port and (best_camera_ip != args.camera_ip or best_camera_port != args.camera_port):
            print("6. Suggested exports:")
            print(f"   export ROBO_CAMERA_IP={best_camera_ip}")
            print(f"   export ROBO_CAMERA_PORT={best_camera_port}")
    else:
        print("\nAll hard requirements passed. It is safe to run main gameplay script.")

    payload = {
        "host": host,
        "targets": {
            "arm_ip": args.arm_ip,
            "arm_port": args.arm_port,
            "camera_ip": args.camera_ip,
            "camera_port": args.camera_port,
            "serial_port": DEFAULT_SERIAL_PORT,
            "env": {
                "ROBO_ARM_IP": os.getenv("ROBO_ARM_IP"),
                "ROBO_ARM_PORT": os.getenv("ROBO_ARM_PORT"),
                "ROBO_CAMERA_IP": os.getenv("ROBO_CAMERA_IP"),
                "ROBO_CAMERA_PORT": os.getenv("ROBO_CAMERA_PORT"),
                "ROBO_SERIAL_PORT": os.getenv("ROBO_SERIAL_PORT"),
            },
        },
        "checks": [asdict(c) for c in checks],
        "camera_candidates": candidates,
        "camera_port_candidates": port_candidates,
        "camera_probe_details": camera_probe_details,
        "best_camera_ip": best_camera_ip,
        "best_camera_port": best_camera_port,
        "endpoint_attempts": endpoint_attempts,
        "route": route,
        "arm_ping": arm_ping,
        "camera_ping": cam_ping,
        "notes": notes,
        "playbook": fix_playbook,
        "verdict": "GO" if go else "NO-GO",
        "elapsed_s": time.monotonic() - start,
    }

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"JSON saved: {out_path}")

    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
