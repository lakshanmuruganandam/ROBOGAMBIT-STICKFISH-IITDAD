"""
ESP32 arm API emulator for local testing.

Imitates Waveshare-style HTTP endpoint:
  GET /js?json={...}

Implemented commands:
- T:104 -> movement command (x, y, z, t)
- T:105 -> feedback query
- T:11  -> servo command (ack only)

The emulator intentionally blocks movement command responses for a short period
so client timeout handling can be tested realistically.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Tuple
from urllib.parse import parse_qs, urlparse


@dataclass
class ArmState:
    x: float = 0.0
    y: float = 0.0
    z: float = 180.0

    target_x: float = 0.0
    target_y: float = 0.0
    target_z: float = 180.0

    move_start: float = 0.0
    move_duration: float = 0.0
    moving: bool = False

    command_count: int = 0
    feedback_count: int = 0
    last_payload: Dict = field(default_factory=dict)

    lock: threading.Lock = field(default_factory=threading.Lock)


class ArmEmulatorServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 18080, block_seconds: float = 3.2):
        self.host = host
        self.port = port
        self.block_seconds = block_seconds
        self.state = ArmState()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def _current_position_unlocked(self) -> Tuple[float, float, float]:
        if not self.state.moving:
            return self.state.x, self.state.y, self.state.z

        now = time.monotonic()
        elapsed = now - self.state.move_start
        if elapsed >= self.state.move_duration:
            self.state.x = self.state.target_x
            self.state.y = self.state.target_y
            self.state.z = self.state.target_z
            self.state.moving = False
            return self.state.x, self.state.y, self.state.z

        ratio = max(0.0, min(1.0, elapsed / max(self.state.move_duration, 1e-6)))
        x = self.state.x + (self.state.target_x - self.state.x) * ratio
        y = self.state.y + (self.state.target_y - self.state.y) * ratio
        z = self.state.z + (self.state.target_z - self.state.z) * ratio
        return x, y, z

    def _handle_payload(self, payload: Dict) -> Tuple[int, Dict, float]:
        cmd = int(payload.get("T", -1))
        response_delay = 0.0

        with self.state.lock:
            self.state.last_payload = dict(payload)

            if cmd == 104:
                tx = float(payload.get("x", self.state.x))
                ty = float(payload.get("y", self.state.y))
                tz = float(payload.get("z", self.state.z))
                speed = max(1.0, float(payload.get("t", 500.0)))

                cx, cy, cz = self._current_position_unlocked()
                self.state.x, self.state.y, self.state.z = cx, cy, cz

                dist = math.sqrt((tx - cx) ** 2 + (ty - cy) ** 2 + (tz - cz) ** 2)
                duration = max(0.3, dist / speed * 4.0)

                self.state.target_x = tx
                self.state.target_y = ty
                self.state.target_z = tz
                self.state.move_start = time.monotonic()
                self.state.move_duration = duration
                self.state.moving = True
                self.state.command_count += 1

                response_delay = self.block_seconds
                return 200, {"ok": 1, "cmd": 104, "accepted": True}, response_delay

            if cmd == 105:
                px, py, pz = self._current_position_unlocked()
                self.state.feedback_count += 1
                return 200, {"T": 105, "x": px, "y": py, "z": pz}, 0.0

            if cmd == 11:
                self.state.command_count += 1
                return 200, {"ok": 1, "cmd": 11}, 0.0

            self.state.command_count += 1
            return 200, {"ok": 1, "cmd": cmd}, 0.0

    def start(self) -> None:
        emulator = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path != "/js":
                    self.send_response(404)
                    self.end_headers()
                    return

                params = parse_qs(parsed.query)
                raw = params.get("json", ["{}"]) [0]
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"error":"invalid json"}')
                    return

                status, body, delay = emulator._handle_payload(payload)
                if delay > 0:
                    time.sleep(delay)

                out = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                try:
                    self.wfile.write(out)
                except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                    # Expected when the client intentionally times out on long-running move sends.
                    return

            def log_message(self, fmt: str, *args):
                return

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def snapshot(self) -> Dict:
        with self.state.lock:
            x, y, z = self._current_position_unlocked()
            return {
                "x": x,
                "y": y,
                "z": z,
                "moving": self.state.moving,
                "target": (self.state.target_x, self.state.target_y, self.state.target_z),
                "command_count": self.state.command_count,
                "feedback_count": self.state.feedback_count,
                "last_payload": dict(self.state.last_payload),
            }
