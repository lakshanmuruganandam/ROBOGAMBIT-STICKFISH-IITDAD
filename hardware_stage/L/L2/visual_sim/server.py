"""
Web visual simulator for RoboGambit Team L2.

Launch:
  python server.py
Then open:
  http://127.0.0.1:8090
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
L2_DIR = ROOT.parent

if str(L2_DIR) not in sys.path:
    sys.path.insert(0, str(L2_DIR))

import game  # noqa: E402

MOVE_REGEX = re.compile(r"^(\d+):([A-F][1-6])->([A-F][1-6])(?:=(\d+))?$")

BOARD_SIZE = 6
CELL_MM = 60.0
HOME_Z = 180.0
SAFE_Z = 180.0
PICK_Z = 15.0
PLACE_Z = 15.0
ARM_SPEED_MM_S = 220.0
GRAVE_WHITE = [(250.0, -150.0 + i * 60.0) for i in range(12)]
GRAVE_BLACK = [(-250.0, -150.0 + i * 60.0) for i in range(12)]


def initial_board() -> np.ndarray:
    return np.array(
        [
            [2, 3, 4, 5, 3, 2],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [6, 6, 6, 6, 6, 6],
            [7, 8, 9, 10, 8, 7],
        ],
        dtype=int,
    )


def rc_to_world(row: int, col: int) -> Tuple[float, float]:
    return (-150.0 + col * CELL_MM, -150.0 + row * CELL_MM)


def cell_to_rc(cell: str) -> Tuple[int, int]:
    col = ord(cell[0].upper()) - ord("A")
    row = int(cell[1]) - 1
    return row, col


@dataclass
class ArmState:
    x: float = 0.0
    y: float = 0.0
    z: float = HOME_Z
    holding: int = 0


@dataclass
class Segment:
    target: Tuple[float, float, float]
    label: str
    on_reach: Optional[str] = None


@dataclass
class FaultConfig:
    sim_speed: float = 1.0
    api_delay_ms: int = 0
    plan_delay_ms: int = 0
    command_drop_prob: float = 0.0
    invalid_move_prob: float = 0.0
    arm_stall_prob: float = 0.0
    camera_lag_ms: int = 0
    camera_noise: float = 0.0
    feedback_jitter_mm: float = 0.0


@dataclass
class SimState:
    board: np.ndarray = field(default_factory=initial_board)
    arm: ArmState = field(default_factory=ArmState)
    turn_white: bool = True
    running: bool = False
    move_count: int = 0
    white_grave_idx: int = 0
    black_grave_idx: int = 0
    last_move: str = ""
    current_phase: str = "idle"
    log: Deque[str] = field(default_factory=lambda: deque(maxlen=80))

    pending_segments: Deque[Segment] = field(default_factory=deque)
    pending_board_commit: Optional[np.ndarray] = None
    config: FaultConfig = field(default_factory=FaultConfig)

    def reset(self) -> None:
        self.board = initial_board()
        self.arm = ArmState()
        self.turn_white = True
        self.running = False
        self.move_count = 0
        self.white_grave_idx = 0
        self.black_grave_idx = 0
        self.last_move = ""
        self.current_phase = "idle"
        self.pending_segments.clear()
        self.pending_board_commit = None
        self.config = FaultConfig()
        self.log.clear()
        self.push_log("Simulator reset")

    def push_log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log.appendleft(f"[{ts}] {msg}")


class Simulator:
    def __init__(self) -> None:
        self.state = SimState()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.state.push_log("Visual simulator online")

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def snapshot(self) -> Dict:
        with self._lock:
            s = self.state
            return {
                "board": s.board.tolist(),
                "arm": {"x": s.arm.x, "y": s.arm.y, "z": s.arm.z, "holding": s.arm.holding},
                "turn_white": s.turn_white,
                "running": s.running,
                "move_count": s.move_count,
                "last_move": s.last_move,
                "current_phase": s.current_phase,
                "log": list(s.log),
                "pending_segments": len(s.pending_segments),
                "config": {
                    "sim_speed": s.config.sim_speed,
                    "api_delay_ms": s.config.api_delay_ms,
                    "plan_delay_ms": s.config.plan_delay_ms,
                    "command_drop_prob": s.config.command_drop_prob,
                    "invalid_move_prob": s.config.invalid_move_prob,
                    "arm_stall_prob": s.config.arm_stall_prob,
                    "camera_lag_ms": s.config.camera_lag_ms,
                    "camera_noise": s.config.camera_noise,
                    "feedback_jitter_mm": s.config.feedback_jitter_mm,
                },
            }

    def set_config(self, updates: Dict) -> None:
        with self._lock:
            c = self.state.config
            c.sim_speed = float(max(0.1, min(4.0, float(updates.get("sim_speed", c.sim_speed)))))
            c.api_delay_ms = int(max(0, min(1500, int(updates.get("api_delay_ms", c.api_delay_ms)))))
            c.plan_delay_ms = int(max(0, min(5000, int(updates.get("plan_delay_ms", c.plan_delay_ms)))))
            c.command_drop_prob = float(max(0.0, min(1.0, float(updates.get("command_drop_prob", c.command_drop_prob)))))
            c.invalid_move_prob = float(max(0.0, min(1.0, float(updates.get("invalid_move_prob", c.invalid_move_prob)))))
            c.arm_stall_prob = float(max(0.0, min(1.0, float(updates.get("arm_stall_prob", c.arm_stall_prob)))))
            c.camera_lag_ms = int(max(0, min(3000, int(updates.get("camera_lag_ms", c.camera_lag_ms)))))
            c.camera_noise = float(max(0.0, min(1.0, float(updates.get("camera_noise", c.camera_noise)))))
            c.feedback_jitter_mm = float(max(0.0, min(25.0, float(updates.get("feedback_jitter_mm", c.feedback_jitter_mm)))))
            self.state.push_log("Fault config updated")

    def set_running(self, running: bool) -> None:
        with self._lock:
            self.state.running = running
            self.state.push_log("Simulation started" if running else "Simulation paused")

    def reset(self) -> None:
        with self._lock:
            self.state.reset()

    def step_once(self) -> None:
        with self._lock:
            if self.state.pending_segments:
                return
            self._plan_next_move_locked()

    def _loop(self) -> None:
        last = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            dt = max(0.001, now - last)
            last = now
            should_plan = False
            with self._lock:
                dt *= self.state.config.sim_speed
                if self.state.pending_segments:
                    self._advance_arm_locked(dt)
                elif self.state.running:
                    should_plan = True
            if should_plan:
                self._plan_next_move_unlocked()
            time.sleep(0.02)

    def _plan_next_move_unlocked(self) -> None:
        with self._lock:
            if self.state.pending_segments or not self.state.running:
                return
            board = self.state.board.copy()
            turn_white = self.state.turn_white
            plan_delay = self.state.config.plan_delay_ms

        if plan_delay > 0:
            time.sleep(plan_delay / 1000.0)

        turn = 1 if turn_white else 0
        move_str = game.get_move(board, turn)

        with self._lock:
            if self.state.pending_segments or not self.state.running:
                return
            self._apply_planned_move_locked(move_str)

    def _advance_arm_locked(self, dt: float) -> None:
        if random.random() < self.state.config.arm_stall_prob * min(1.0, dt * 8.0):
            self.state.current_phase = "arm-stall"
            return

        seg = self.state.pending_segments[0]
        ax, ay, az = self.state.arm.x, self.state.arm.y, self.state.arm.z
        tx, ty, tz = seg.target
        dx, dy, dz = tx - ax, ty - ay, tz - az
        dist = (dx * dx + dy * dy + dz * dz) ** 0.5
        step = ARM_SPEED_MM_S * dt
        if dist <= max(1e-6, step):
            self.state.arm.x, self.state.arm.y, self.state.arm.z = tx, ty, tz
            self.state.current_phase = seg.label
            self.state.pending_segments.popleft()
            if seg.on_reach == "pick":
                self.state.arm.holding = 1
            elif seg.on_reach == "drop":
                self.state.arm.holding = 0

            if not self.state.pending_segments and self.state.pending_board_commit is not None:
                self.state.board = self.state.pending_board_commit
                self.state.pending_board_commit = None
                self.state.turn_white = not self.state.turn_white
                self.state.move_count += 1
                self.state.current_phase = "idle"
                self.state.push_log(f"Move committed. Next: {'white' if self.state.turn_white else 'black'}")
            return

        ratio = step / dist
        self.state.arm.x = ax + dx * ratio
        self.state.arm.y = ay + dy * ratio
        self.state.arm.z = az + dz * ratio
        self.state.current_phase = seg.label

    def _plan_next_move_locked(self) -> None:
        turn = 1 if self.state.turn_white else 0
        move_str = game.get_move(self.state.board.copy(), turn)
        self._apply_planned_move_locked(move_str)

    def _apply_planned_move_locked(self, move_str: str) -> None:
        if random.random() < self.state.config.invalid_move_prob:
            move_str = "BAD:MOVE"

        m = MOVE_REGEX.match(move_str or "")
        if m is None:
            self.state.push_log(f"Engine produced invalid move: {move_str}")
            self.state.running = False
            return

        piece = int(m.group(1))
        fr = cell_to_rc(m.group(2))
        to = cell_to_rc(m.group(3))
        promo = int(m.group(4)) if m.group(4) else None

        sr, sc = fr
        tr, tc = to
        board = self.state.board.copy()
        captured = int(board[tr, tc])

        fx, fy = rc_to_world(sr, sc)
        tx, ty = rc_to_world(tr, tc)

        if random.random() < self.state.config.command_drop_prob:
            self.state.current_phase = "command-dropped"
            self.state.push_log(f"Command dropped (simulated): {move_str}")
            return

        segments: List[Segment] = []

        if captured != 0:
            gx, gy = self._next_grave_locked(captured)
            segments.extend(
                [
                    Segment((tx, ty, SAFE_Z), "capture-hover"),
                    Segment((tx, ty, PICK_Z), "capture-pick", "pick"),
                    Segment((tx, ty, SAFE_Z), "capture-lift"),
                    Segment((gx, gy, SAFE_Z), "grave-hover"),
                    Segment((gx, gy, PLACE_Z), "grave-drop", "drop"),
                    Segment((gx, gy, SAFE_Z), "grave-lift"),
                ]
            )

        segments.extend(
            [
                Segment((fx, fy, SAFE_Z), "move-source-hover"),
                Segment((fx, fy, PICK_Z), "move-pick", "pick"),
                Segment((fx, fy, SAFE_Z), "move-lift"),
                Segment((tx, ty, SAFE_Z), "move-target-hover"),
                Segment((tx, ty, PLACE_Z), "move-drop", "drop"),
                Segment((tx, ty, SAFE_Z), "move-finish"),
                Segment((0.0, 0.0, SAFE_Z), "home"),
            ]
        )

        board[sr, sc] = 0
        board[tr, tc] = promo if promo is not None else piece

        self.state.pending_segments = deque(segments)
        self.state.pending_board_commit = board
        self.state.last_move = move_str
        self.state.push_log(f"Planned move: {move_str}")

    def _next_grave_locked(self, captured_piece: int) -> Tuple[float, float]:
        if captured_piece <= 5:
            idx = min(self.state.white_grave_idx, len(GRAVE_WHITE) - 1)
            self.state.white_grave_idx += 1
            return GRAVE_WHITE[idx]
        idx = min(self.state.black_grave_idx, len(GRAVE_BLACK) - 1)
        self.state.black_grave_idx += 1
        return GRAVE_BLACK[idx]


SIM = Simulator()


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj: Dict, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return

        if parsed.path == "/static/app.js":
            self._send_file(STATIC_DIR / "app.js", "application/javascript; charset=utf-8")
            return

        if parsed.path == "/static/styles.css":
            self._send_file(STATIC_DIR / "styles.css", "text/css; charset=utf-8")
            return

        if parsed.path == "/api/state":
            delay_ms = SIM.state.config.api_delay_ms
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)
            self._send_json(SIM.snapshot())
            return

        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        _ = parse_qs(parsed.query)

        content_len = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_len) if content_len > 0 else b""
        payload = {}
        if raw:
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                payload = {}

        if parsed.path == "/api/start":
            SIM.set_running(True)
            self._send_json({"ok": True})
            return

        if parsed.path == "/api/pause":
            SIM.set_running(False)
            self._send_json({"ok": True})
            return

        if parsed.path == "/api/reset":
            SIM.reset()
            self._send_json({"ok": True})
            return

        if parsed.path == "/api/step":
            SIM.step_once()
            self._send_json({"ok": True})
            return

        if parsed.path == "/api/config":
            SIM.set_config(payload)
            self._send_json({"ok": True})
            return

        self.send_error(404)

    def log_message(self, fmt: str, *args) -> None:
        return


def run(host: str = "127.0.0.1", port: int = 8090) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Visual simulator running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        SIM.stop()
        server.server_close()


if __name__ == "__main__":
    run()
