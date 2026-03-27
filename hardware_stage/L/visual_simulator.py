"""
visual_simulator.py — Full Visual Hardware Mock for RoboGambit
================================================================
A web-based visual simulator that replicates the ENTIRE hardware
environment with a stunning visual dashboard:

  ┌──────────────────────────────────────────────┐
  │  6×6 Board (animated pieces)                 │
  │  Robotic Arm 3D Position                     │
  │  Simulated Camera Feed (ArUco markers)       │
  │  Electromagnet / Gripper State               │
  │  Game Log (real-time)                        │
  │  AI vs AI automated play                     │
  └──────────────────────────────────────────────┘

Usage:
    python visual_simulator.py
    → Open http://localhost:5050 in browser

Requirements:
    pip install flask numpy opencv-python
"""

import os
import sys
import json
import math
import time
import struct
import base64
import threading
import logging
from collections import deque

import cv2
import numpy as np
from flask import Flask, render_template_string, jsonify, request

# Ensure local imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import game as game_module

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("visual_sim")

app = Flask(__name__)

# ═════════════════════════════════════════════════════════════════════════
#  GAME STATE
# ═════════════════════════════════════════════════════════════════════════

INITIAL_BOARD = [
    [2, 3, 4, 5, 3, 2],
    [1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [6, 6, 6, 6, 6, 6],
    [7, 8, 9, 10, 8, 7],
]

PIECE_NAMES = {
    0: ".", 1: "wP", 2: "wN", 3: "wB", 4: "wQ", 5: "wK",
    6: "bP", 7: "bN", 8: "bB", 9: "bQ", 10: "bK",
}

PIECE_UNICODE = {
    0: "", 1: "♙", 2: "♘", 3: "♗", 4: "♕", 5: "♔",
    6: "♟", 7: "♞", 8: "♝", 9: "♛", 10: "♚",
}

CELL_X = [-150.0, -90.0, -30.0, 30.0, 90.0, 150.0]
CELL_Y = [-150.0, -90.0, -30.0, 30.0, 90.0, 150.0]

class GameState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [row[:] for row in INITIAL_BOARD]
        self.move_history = []
        self.log_messages = deque(maxlen=200)
        self.white_turn = True
        self.move_number = 0
        self.game_over = False
        self.game_over_msg = ""
        self.arm_x = 0.0
        self.arm_y = 0.0
        self.arm_z = 180.0
        self.arm_speed = 500
        self.magnet_on = False
        self.gripper_open = True
        self.held_piece = 0
        self.arm_moving = False
        self.arm_target = (0, 0, 180)
        self.arm_trail = []
        self.computing = False
        self.last_move = None
        self.captured_white = []
        self.captured_black = []
        self._lock = threading.Lock()
        self.add_log("System initialized. Ready to play.")

    def add_log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_messages.append(f"[{ts}] {msg}")

    def to_dict(self):
        with self._lock:
            return {
                "board": self.board,
                "move_history": self.move_history[-20:],
                "log": list(self.log_messages)[-30:],
                "white_turn": self.white_turn,
                "move_number": self.move_number,
                "game_over": self.game_over,
                "game_over_msg": self.game_over_msg,
                "arm": {
                    "x": round(self.arm_x, 1),
                    "y": round(self.arm_y, 1),
                    "z": round(self.arm_z, 1),
                    "speed": self.arm_speed,
                    "magnet": self.magnet_on,
                    "gripper_open": self.gripper_open,
                    "moving": self.arm_moving,
                    "held_piece": self.held_piece,
                    "target": list(self.arm_target),
                },
                "computing": self.computing,
                "last_move": self.last_move,
                "captured_white": self.captured_white,
                "captured_black": self.captured_black,
            }


state = GameState()

# ArUco frame generation
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker_cache = {}

CORNERS = {21: (212.5, 212.5), 22: (212.5, -212.5),
           23: (-212.5, -212.5), 24: (-212.5, 212.5)}
IMG_W, IMG_H = 640, 360
IMG_CX, IMG_CY = IMG_W // 2, IMG_H // 2
CAM_SCALE = 0.65


def get_marker(mid, size=20):
    if (mid, size) not in marker_cache:
        marker_cache[(mid, size)] = cv2.aruco.generateImageMarker(aruco_dict, mid, size)
    return marker_cache[(mid, size)]


def world_to_px(wx, wy):
    return int(IMG_CX + wx * CAM_SCALE), int(IMG_CY - wy * CAM_SCALE)


def generate_camera_frame():
    """Generate synthetic camera frame."""
    frame = np.full((IMG_H, IMG_W, 3), (35, 55, 35), dtype=np.uint8)

    # Grid
    for i in range(7):
        v = -180 + i * 60
        p1 = world_to_px(-180, v)
        p2 = world_to_px(180, v)
        cv2.line(frame, p1, p2, (60, 80, 60), 1)
        p1 = world_to_px(v, -180)
        p2 = world_to_px(v, 180)
        cv2.line(frame, p1, p2, (60, 80, 60), 1)

    # Cells
    for r in range(6):
        for c in range(6):
            if (r + c) % 2 == 0:
                w1, w2 = -180 + c * 60, -180 + (c + 1) * 60
                h1, h2 = -180 + r * 60, -180 + (r + 1) * 60
                cv2.rectangle(frame, world_to_px(w1, h2), world_to_px(w2, h1),
                              (50, 70, 50), -1)

    # Corner markers
    for mid, (wx, wy) in CORNERS.items():
        _place(frame, mid, wx, wy, 18)

    # Piece markers
    for r in range(6):
        for c in range(6):
            pid = state.board[r][c]
            if pid > 0:
                wx, wy = CELL_X[c], CELL_Y[r]
                _place(frame, pid, wx, wy, 16)

    # Arm indicator
    ax, ay = world_to_px(state.arm_x, state.arm_y)
    color = (0, 200, 255) if state.magnet_on else (200, 200, 200)
    cv2.circle(frame, (ax, ay), 8, color, 2)
    cv2.circle(frame, (ax, ay), 3, color, -1)

    # HUD
    cv2.putText(frame, f"ARM ({state.arm_x:.0f},{state.arm_y:.0f},{state.arm_z:.0f})",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    mag_str = "MAG:ON" if state.magnet_on else "MAG:OFF"
    cv2.putText(frame, mag_str, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 200, 255) if state.magnet_on else (150, 150, 150), 1)

    return frame


def _place(frame, mid, wx, wy, size):
    px, py = world_to_px(wx, wy)
    half = size // 2
    if px - half < 0 or py - half < 0 or px + half >= IMG_W or py + half >= IMG_H:
        return
    marker = get_marker(mid, size)
    mbgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    y1, y2 = py - half, py - half + size
    x1, x2 = px - half, px - half + size
    if 0 <= y1 and y2 <= IMG_H and 0 <= x1 and x2 <= IMG_W:
        frame[y1:y2, x1:x2] = mbgr


# ═════════════════════════════════════════════════════════════════════════
#  ARM SIMULATION
# ═════════════════════════════════════════════════════════════════════════

def sim_arm_move(x, y, z, speed=500):
    """Simulate arm movement with animation steps."""
    state.arm_moving = True
    state.arm_target = (x, y, z)
    state.arm_speed = speed

    dx = x - state.arm_x
    dy = y - state.arm_y
    dz = z - state.arm_z
    dist = math.sqrt(dx**2 + dy**2 + dz**2)
    steps = max(5, int(dist / 10))

    for i in range(1, steps + 1):
        t = i / steps
        state.arm_x = state.arm_x + dx / steps
        state.arm_y = state.arm_y + dy / steps
        state.arm_z = state.arm_z + dz / steps
        time.sleep(0.02)

    state.arm_x, state.arm_y, state.arm_z = x, y, z
    state.arm_moving = False


def sim_pick(r, c):
    """Simulate picking a piece."""
    wx, wy = CELL_X[c], CELL_Y[r]
    state.add_log(f"ARM → hover ({wx:.0f}, {wy:.0f}, 180)")
    sim_arm_move(wx, wy, 180)
    time.sleep(0.1)

    state.gripper_open = True
    state.add_log(f"ARM → lower to pick ({wx:.0f}, {wy:.0f}, 8)")
    sim_arm_move(wx, wy, 8)
    time.sleep(0.1)

    state.gripper_open = False
    state.magnet_on = True
    state.held_piece = state.board[r][c]
    state.add_log(f"MAGNET ON — grabbed {PIECE_NAMES.get(state.held_piece, '?')}")
    time.sleep(0.1)

    state.add_log(f"ARM → lift ({wx:.0f}, {wy:.0f}, 180)")
    sim_arm_move(wx, wy, 180)


def sim_place(r, c):
    """Simulate placing a piece."""
    wx, wy = CELL_X[c], CELL_Y[r]
    state.add_log(f"ARM → hover ({wx:.0f}, {wy:.0f}, 180)")
    sim_arm_move(wx, wy, 180)
    time.sleep(0.1)

    state.add_log(f"ARM → lower to place ({wx:.0f}, {wy:.0f}, 10)")
    sim_arm_move(wx, wy, 10)
    time.sleep(0.1)

    state.magnet_on = False
    state.gripper_open = True
    state.add_log(f"MAGNET OFF — released {PIECE_NAMES.get(state.held_piece, '?')}")
    state.held_piece = 0
    time.sleep(0.1)

    state.add_log(f"ARM → lift ({wx:.0f}, {wy:.0f}, 180)")
    sim_arm_move(wx, wy, 180)


def sim_home():
    state.add_log("ARM → HOME (0, 0, 180)")
    sim_arm_move(0, 0, 180)


# ═════════════════════════════════════════════════════════════════════════
#  MOVE EXECUTION
# ═════════════════════════════════════════════════════════════════════════

def parse_move(move_str):
    try:
        colon = move_str.index(":")
        piece = int(move_str[:colon])
        rest = move_str[colon + 1:]
        promo = None
        if "=" in rest:
            rest, ps = rest.split("=")
            promo = int(ps)
        arrow = rest.index("->")
        src, dst = rest[:arrow], rest[arrow + 2:]
        sc, sr = ord(src[0]) - ord("A"), int(src[1]) - 1
        dc, dr = ord(dst[0]) - ord("A"), int(dst[1]) - 1
        return piece, (sr, sc), (dr, dc), promo
    except:
        return None


def execute_one_move():
    """Execute one move (for current side), return True if game continues."""
    if state.game_over:
        return False

    side = "WHITE" if state.white_turn else "BLACK"
    state.computing = True
    state.add_log(f"--- Move {state.move_number + 1} ({side}) --- THINKING...")

    board_np = np.array(state.board, dtype=int)
    t0 = time.time()
    move_str = game_module.get_best_move(board_np, state.white_turn)
    elapsed = time.time() - t0

    state.computing = False

    if move_str is None:
        state.game_over = True
        state.game_over_msg = f"{side} has no legal moves!"
        state.add_log(f"GAME OVER: {state.game_over_msg}")
        return False

    state.add_log(f"ENGINE: {move_str} ({elapsed:.2f}s)")
    parsed = parse_move(move_str)
    if not parsed:
        state.add_log(f"ERROR: could not parse {move_str}")
        return False

    piece_id, (sr, sc), (dr, dc), promo = parsed
    target = state.board[dr][dc]

    # Handle capture
    if target != 0:
        state.add_log(f"CAPTURE: {PIECE_NAMES.get(target, '?')} at {chr(65+dc)}{dr+1}")
        sim_pick(dr, dc)
        # Move to graveyard
        gx = 250 if target in {1,2,3,4,5} else -250
        gy = -150 + len(state.captured_white if target in {1,2,3,4,5} else state.captured_black) * 60
        state.add_log(f"ARM → graveyard ({gx}, {gy})")
        sim_arm_move(gx, gy, 180)
        sim_arm_move(gx, gy, 10)
        state.magnet_on = False
        state.held_piece = 0
        sim_arm_move(gx, gy, 180)
        if target in {1,2,3,4,5}:
            state.captured_white.append(target)
        else:
            state.captured_black.append(target)
        state.board[dr][dc] = 0

    # Move piece
    state.add_log(f"MOVE: {PIECE_NAMES.get(piece_id, '?')} {chr(65+sc)}{sr+1} → {chr(65+dc)}{dr+1}")
    sim_pick(sr, sc)
    state.board[sr][sc] = 0

    final_piece = promo if promo else piece_id
    sim_place(dr, dc)
    state.board[dr][dc] = final_piece

    if promo:
        state.add_log(f"PROMOTION → {PIECE_NAMES.get(promo, '?')}")

    sim_home()

    state.last_move = {
        "from": [sr, sc], "to": [dr, dc],
        "piece": piece_id, "captured": target,
        "move_str": move_str, "time": round(elapsed, 2),
    }
    state.move_history.append(f"{'W' if state.white_turn else 'B'}{state.move_number+1}: {move_str}")
    state.move_number += 1
    state.white_turn = not state.white_turn

    # Check game over
    has_wk = any(state.board[r][c] == 5 for r in range(6) for c in range(6))
    has_bk = any(state.board[r][c] == 10 for r in range(6) for c in range(6))
    if not has_wk:
        state.game_over = True
        state.game_over_msg = "BLACK WINS — White King captured!"
        state.add_log(f"★ {state.game_over_msg}")
    elif not has_bk:
        state.game_over = True
        state.game_over_msg = "WHITE WINS — Black King captured!"
        state.add_log(f"★ {state.game_over_msg}")

    return not state.game_over


# Auto-play thread
auto_play_active = False
auto_play_thread = None

def auto_play_loop():
    global auto_play_active
    while auto_play_active and not state.game_over:
        execute_one_move()
        time.sleep(0.3)
    auto_play_active = False


# ═════════════════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ═════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/state")
def api_state():
    return jsonify(state.to_dict())


@app.route("/api/camera")
def api_camera():
    frame = generate_camera_frame()
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    b64 = base64.b64encode(buf.tobytes()).decode()
    return jsonify({"image": f"data:image/jpeg;base64,{b64}"})


@app.route("/api/move", methods=["POST"])
def api_move():
    if state.game_over:
        return jsonify({"ok": False, "msg": "Game over"})
    if state.computing or state.arm_moving:
        return jsonify({"ok": False, "msg": "Busy"})

    def do_move():
        execute_one_move()

    threading.Thread(target=do_move, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/autoplay", methods=["POST"])
def api_autoplay():
    global auto_play_active, auto_play_thread
    action = request.json.get("action", "start")
    if action == "start" and not auto_play_active:
        auto_play_active = True
        auto_play_thread = threading.Thread(target=auto_play_loop, daemon=True)
        auto_play_thread.start()
        return jsonify({"ok": True, "msg": "Auto-play started"})
    elif action == "stop":
        auto_play_active = False
        return jsonify({"ok": True, "msg": "Auto-play stopped"})
    return jsonify({"ok": False})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    global auto_play_active
    auto_play_active = False
    time.sleep(0.5)
    state.reset()
    return jsonify({"ok": True})


# ═════════════════════════════════════════════════════════════════════════
#  HTML TEMPLATE (Full Visual Dashboard)
# ═════════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RoboGambit — Full Hardware Simulator</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#0a0e17;--bg2:#111827;--bg3:#1a2035;
  --border:#2a3555;--accent:#00d4ff;--accent2:#7c3aed;
  --green:#10b981;--red:#ef4444;--amber:#f59e0b;
  --text:#e2e8f0;--text2:#94a3b8;
}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);
  min-height:100vh;overflow-x:hidden}
.header{background:linear-gradient(135deg,#0f172a,#1e1b4b);
  border-bottom:1px solid var(--border);padding:16px 32px;
  display:flex;align-items:center;gap:16px}
.header h1{font-size:22px;font-weight:700;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.header .badge{background:var(--accent2);color:#fff;padding:3px 10px;
  border-radius:12px;font-size:11px;font-weight:600}
.status-bar{display:flex;gap:16px;margin-left:auto;font-size:13px}
.status-item{display:flex;align-items:center;gap:6px}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block}
.dot.on{background:var(--green);box-shadow:0 0 8px var(--green)}
.dot.off{background:#555}
.dot.warn{background:var(--amber);box-shadow:0 0 8px var(--amber)}

.main{display:grid;grid-template-columns:1fr 1fr 380px;
  grid-template-rows:auto 1fr;gap:16px;padding:20px;max-height:calc(100vh - 70px)}

/* Board Panel */
.panel{background:var(--bg2);border:1px solid var(--border);border-radius:12px;
  padding:16px;position:relative;overflow:hidden}
.panel::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--accent),var(--accent2))}
.panel-title{font-size:13px;font-weight:600;color:var(--accent);
  text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;
  display:flex;align-items:center;gap:8px}
.panel-title .icon{font-size:16px}

/* Chess board */
.board-container{display:flex;flex-direction:column;align-items:center}
.board{display:grid;grid-template-columns:30px repeat(6,60px);
  grid-template-rows:repeat(6,60px) 30px;gap:0;
  border:2px solid var(--border);border-radius:4px;overflow:hidden}
.cell{width:60px;height:60px;display:flex;align-items:center;
  justify-content:center;font-size:32px;cursor:default;
  position:relative;transition:all 0.3s ease}
.cell.light{background:#2a3a2a}
.cell.dark{background:#1a2a1a}
.cell.highlight{box-shadow:inset 0 0 0 3px var(--accent);
  background:rgba(0,212,255,0.15)}
.cell.from-cell{box-shadow:inset 0 0 0 3px var(--amber);
  background:rgba(245,158,11,0.1)}
.cell.to-cell{box-shadow:inset 0 0 0 3px var(--green);
  background:rgba(16,185,129,0.15)}
.rank-label,.file-label{display:flex;align-items:center;
  justify-content:center;font-size:12px;color:var(--text2);
  font-family:'JetBrains Mono',monospace}
.rank-label{width:30px}
.file-label{height:30px}

/* Captured */
.captured{display:flex;gap:4px;margin-top:8px;min-height:32px;
  flex-wrap:wrap;align-items:center}
.captured-piece{font-size:20px;opacity:0.6}

/* Controls */
.controls{display:flex;gap:10px;margin-top:14px;flex-wrap:wrap;justify-content:center}
.btn{padding:8px 18px;border:1px solid var(--border);border-radius:8px;
  font-family:'Inter',sans-serif;font-size:13px;font-weight:600;
  cursor:pointer;transition:all 0.2s;color:var(--text);background:var(--bg3)}
.btn:hover{border-color:var(--accent);transform:translateY(-1px);
  box-shadow:0 4px 12px rgba(0,212,255,0.15)}
.btn.primary{background:linear-gradient(135deg,#0059b3,#7c3aed);border:none;color:#fff}
.btn.primary:hover{box-shadow:0 4px 16px rgba(124,58,237,0.4)}
.btn.danger{border-color:var(--red);color:var(--red)}
.btn.danger:hover{background:rgba(239,68,68,0.1)}
.btn.success{border-color:var(--green);color:var(--green)}
.btn:disabled{opacity:0.4;cursor:not-allowed;transform:none}

/* Camera feed */
.camera-feed{width:100%;border-radius:8px;border:1px solid var(--border);
  image-rendering:pixelated}

/* Arm status */
.arm-status{font-family:'JetBrains Mono',monospace;font-size:12px}
.arm-row{display:flex;justify-content:space-between;padding:4px 0;
  border-bottom:1px solid rgba(255,255,255,0.05)}
.arm-label{color:var(--text2)}
.arm-val{color:var(--accent);font-weight:600}
.arm-val.on{color:var(--green)}
.arm-val.off{color:var(--text2)}

/* 3D Arm Viz */
.arm-viz{width:100%;height:180px;background:var(--bg);
  border-radius:8px;border:1px solid var(--border);position:relative;overflow:hidden}
.arm-canvas{width:100%;height:100%}

/* Log */
.log-panel{grid-column:3;grid-row:1/3;overflow:hidden;display:flex;flex-direction:column}
.log-scroll{flex:1;overflow-y:auto;font-family:'JetBrains Mono',monospace;
  font-size:11px;line-height:1.6;padding:8px;background:var(--bg);
  border-radius:6px;border:1px solid var(--border)}
.log-entry{padding:2px 4px;border-radius:3px}
.log-entry:hover{background:rgba(255,255,255,0.03)}

/* Move History */
.move-list{max-height:120px;overflow-y:auto;font-family:'JetBrains Mono',monospace;
  font-size:12px;padding:8px;background:var(--bg);border-radius:6px;
  border:1px solid var(--border)}
.move-item{padding:2px 6px;display:inline-block;border-radius:4px;margin:2px}
.move-item.white{background:rgba(255,255,255,0.05)}
.move-item.black{background:rgba(124,58,237,0.1)}

/* Scrollbar */
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}

/* Game over overlay */
.game-over-overlay{position:fixed;top:0;left:0;right:0;bottom:0;
  background:rgba(0,0,0,0.7);display:flex;align-items:center;
  justify-content:center;z-index:100;backdrop-filter:blur(4px)}
.game-over-card{background:var(--bg2);border:2px solid var(--accent2);
  border-radius:16px;padding:40px;text-align:center;
  animation:popIn 0.3s ease}
.game-over-card h2{font-size:28px;margin-bottom:12px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
@keyframes popIn{from{transform:scale(0.8);opacity:0}to{transform:scale(1);opacity:1}}

/* Responsive */
@media(max-width:1200px){.main{grid-template-columns:1fr 1fr;}.log-panel{grid-column:1/3;grid-row:auto}}
@media(max-width:768px){.main{grid-template-columns:1fr}.log-panel{grid-column:1}}

/* Pulse animation for computing */
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.5}}
.computing{animation:pulse 1s infinite}

/* Turn indicator */
.turn-badge{padding:4px 12px;border-radius:6px;font-size:12px;font-weight:600;
  display:inline-flex;align-items:center;gap:6px}
.turn-badge.white{background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.2)}
.turn-badge.black{background:rgba(124,58,237,0.15);border:1px solid var(--accent2)}
</style>
</head>
<body>

<div class="header">
  <h1>🤖 RoboGambit</h1>
  <span class="badge">HARDWARE SIMULATOR</span>
  <div class="status-bar">
    <div class="status-item"><span class="dot on" id="dot-arm"></span> ARM</div>
    <div class="status-item"><span class="dot on" id="dot-cam"></span> CAMERA</div>
    <div class="status-item"><span class="dot on" id="dot-serial"></span> SERIAL</div>
    <div class="status-item" id="turn-indicator"></div>
  </div>
</div>

<div class="main">
  <!-- Board Panel -->
  <div class="panel">
    <div class="panel-title"><span class="icon">♟</span> Game Board</div>
    <div class="board-container">
      <div class="captured" id="captured-black"></div>
      <div class="board" id="board"></div>
      <div class="captured" id="captured-white"></div>
      <div class="controls">
        <button class="btn primary" id="btn-move" onclick="nextMove()">▶ Next Move</button>
        <button class="btn success" id="btn-auto" onclick="toggleAuto()">⚡ Auto Play</button>
        <button class="btn danger" onclick="resetGame()">↺ Reset</button>
      </div>
      <div class="move-list" id="move-list" style="margin-top:12px;width:100%"></div>
    </div>
  </div>

  <!-- Hardware Panel -->
  <div class="panel">
    <div class="panel-title"><span class="icon">🦾</span> Hardware Status</div>

    <div style="margin-bottom:14px">
      <div style="font-size:12px;color:var(--text2);margin-bottom:6px">SIMULATED CAMERA FEED</div>
      <img class="camera-feed" id="camera-feed" alt="Camera">
    </div>

    <div style="margin-bottom:14px">
      <div style="font-size:12px;color:var(--text2);margin-bottom:6px">ARM VISUALIZATION</div>
      <div class="arm-viz">
        <canvas class="arm-canvas" id="arm-canvas"></canvas>
      </div>
    </div>

    <div class="arm-status" id="arm-status"></div>
  </div>

  <!-- Log Panel -->
  <div class="panel log-panel">
    <div class="panel-title"><span class="icon">📋</span> System Log</div>
    <div class="log-scroll" id="log-scroll"></div>
  </div>
</div>

<div class="game-over-overlay" id="game-over" style="display:none">
  <div class="game-over-card">
    <h2 id="game-over-msg">GAME OVER</h2>
    <p style="color:var(--text2);margin-bottom:20px" id="game-over-detail"></p>
    <button class="btn primary" onclick="resetGame()">Play Again</button>
  </div>
</div>

<script>
const PIECE_UNICODE = {0:'',1:'♙',2:'♘',3:'♗',4:'♕',5:'♔',
  6:'♟',7:'♞',8:'♝',9:'♛',10:'♚'};
const FILES = 'ABCDEF';
let autoPlaying = false;
let lastState = null;
let pollInterval = null;

function buildBoard() {
  const board = document.getElementById('board');
  board.innerHTML = '';
  // Ranks 6 down to 1
  for (let r = 5; r >= 0; r--) {
    // Rank label
    const rl = document.createElement('div');
    rl.className = 'rank-label';
    rl.textContent = r + 1;
    board.appendChild(rl);
    for (let c = 0; c < 6; c++) {
      const cell = document.createElement('div');
      cell.className = 'cell ' + ((r + c) % 2 === 0 ? 'light' : 'dark');
      cell.id = `cell-${r}-${c}`;
      cell.dataset.row = r;
      cell.dataset.col = c;
      board.appendChild(cell);
    }
  }
  // File labels
  const empty = document.createElement('div');
  empty.className = 'file-label';
  board.appendChild(empty);
  for (let c = 0; c < 6; c++) {
    const fl = document.createElement('div');
    fl.className = 'file-label';
    fl.textContent = FILES[c];
    board.appendChild(fl);
  }
}

function updateBoard(data) {
  if (!data) return;
  const board = data.board;
  const lm = data.last_move;

  for (let r = 0; r < 6; r++) {
    for (let c = 0; c < 6; c++) {
      const cell = document.getElementById(`cell-${r}-${c}`);
      if (!cell) continue;
      const pid = board[r][c];
      cell.textContent = PIECE_UNICODE[pid] || '';
      cell.classList.remove('highlight','from-cell','to-cell');
      if (pid >= 6) cell.style.filter = 'brightness(0.7)';
      else cell.style.filter = '';
    }
  }

  // Highlight last move
  if (lm) {
    const fc = document.getElementById(`cell-${lm.from[0]}-${lm.from[1]}`);
    const tc = document.getElementById(`cell-${lm.to[0]}-${lm.to[1]}`);
    if (fc) fc.classList.add('from-cell');
    if (tc) tc.classList.add('to-cell');
  }

  // Captured
  document.getElementById('captured-white').innerHTML =
    (data.captured_white || []).map(p => `<span class="captured-piece">${PIECE_UNICODE[p]}</span>`).join('');
  document.getElementById('captured-black').innerHTML =
    (data.captured_black || []).map(p => `<span class="captured-piece">${PIECE_UNICODE[p]}</span>`).join('');

  // Turn indicator
  const ti = document.getElementById('turn-indicator');
  if (data.computing) {
    ti.innerHTML = `<span class="turn-badge ${data.white_turn?'white':'black'} computing">🧠 THINKING...</span>`;
  } else if (data.arm && data.arm.moving) {
    ti.innerHTML = `<span class="turn-badge ${data.white_turn?'white':'black'}">🦾 MOVING...</span>`;
  } else {
    const side = data.white_turn ? 'WHITE' : 'BLACK';
    ti.innerHTML = `<span class="turn-badge ${data.white_turn?'white':'black'}">${data.white_turn?'♔':'♚'} ${side}</span>`;
  }

  // Move list
  const ml = document.getElementById('move-list');
  ml.innerHTML = (data.move_history || []).map(m => {
    const cls = m.startsWith('W') ? 'white' : 'black';
    return `<span class="move-item ${cls}">${m}</span>`;
  }).join('');
  ml.scrollTop = ml.scrollHeight;

  // Buttons
  const busy = data.computing || (data.arm && data.arm.moving);
  document.getElementById('btn-move').disabled = busy || data.game_over;

  // Game over
  if (data.game_over) {
    document.getElementById('game-over').style.display = 'flex';
    document.getElementById('game-over-msg').textContent = '🏆 GAME OVER';
    document.getElementById('game-over-detail').textContent = data.game_over_msg;
  } else {
    document.getElementById('game-over').style.display = 'none';
  }
}

function updateArmStatus(arm) {
  if (!arm) return;
  const el = document.getElementById('arm-status');
  el.innerHTML = `
    <div class="arm-row"><span class="arm-label">Position</span>
      <span class="arm-val">(${arm.x}, ${arm.y}, ${arm.z})</span></div>
    <div class="arm-row"><span class="arm-label">Speed</span>
      <span class="arm-val">${arm.speed}</span></div>
    <div class="arm-row"><span class="arm-label">Magnet</span>
      <span class="arm-val ${arm.magnet?'on':'off'}">${arm.magnet?'● ON':'○ OFF'}</span></div>
    <div class="arm-row"><span class="arm-label">Gripper</span>
      <span class="arm-val">${arm.gripper_open?'OPEN':'CLOSED'}</span></div>
    <div class="arm-row"><span class="arm-label">Holding</span>
      <span class="arm-val">${arm.held_piece ? PIECE_UNICODE[arm.held_piece]+' (ID '+arm.held_piece+')' : '—'}</span></div>
    <div class="arm-row"><span class="arm-label">Status</span>
      <span class="arm-val ${arm.moving?'on':''}">${arm.moving?'⚡ MOVING':'● IDLE'}</span></div>
  `;
  // Status dots
  document.getElementById('dot-arm').className = 'dot on';
  document.getElementById('dot-cam').className = 'dot on';
  document.getElementById('dot-serial').className = 'dot on';
}

function updateLog(logs) {
  const el = document.getElementById('log-scroll');
  el.innerHTML = (logs || []).map(l => `<div class="log-entry">${escHtml(l)}</div>`).join('');
  el.scrollTop = el.scrollHeight;
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function drawArmViz(arm) {
  const canvas = document.getElementById('arm-canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
  const w = canvas.width, h = canvas.height;

  ctx.fillStyle = '#0a0e17';
  ctx.fillRect(0, 0, w, h);

  // Draw grid
  ctx.strokeStyle = '#1a2a3a';
  ctx.lineWidth = 1;
  for (let i = 0; i < w; i += 30) {
    ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, h); ctx.stroke();
  }
  for (let i = 0; i < h; i += 30) {
    ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(w, i); ctx.stroke();
  }

  if (!arm) return;

  // Map arm XY to canvas (side view: X horizontal, Z vertical)
  const cx = w / 2 + arm.x * 0.8;
  const cy = h - 20 - arm.z * 0.7;

  // Base
  ctx.fillStyle = '#333';
  ctx.fillRect(w/2 - 15, h - 25, 30, 25);

  // Arm segments
  ctx.strokeStyle = '#00d4ff';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(w/2, h - 25);
  ctx.lineTo(w/2, h - 25 - 40);
  ctx.lineTo(cx, cy);
  ctx.stroke();

  // Joints
  ctx.fillStyle = '#7c3aed';
  ctx.beginPath(); ctx.arc(w/2, h - 25 - 40, 5, 0, Math.PI * 2); ctx.fill();

  // End effector
  ctx.fillStyle = arm.magnet ? '#10b981' : '#555';
  ctx.beginPath(); ctx.arc(cx, cy, 7, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = arm.magnet ? '#10b981' : '#555';
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(cx, cy, 11, 0, Math.PI * 2); ctx.stroke();

  // Labels
  ctx.fillStyle = '#94a3b8';
  ctx.font = '10px JetBrains Mono';
  ctx.fillText(`X:${arm.x.toFixed(0)} Y:${arm.y.toFixed(0)} Z:${arm.z.toFixed(0)}`, 8, 14);
}

// Polling
async function poll() {
  try {
    const [stateRes, camRes] = await Promise.all([
      fetch('/api/state'),
      fetch('/api/camera')
    ]);
    const data = await stateRes.json();
    const cam = await camRes.json();

    lastState = data;
    updateBoard(data);
    updateArmStatus(data.arm);
    updateLog(data.log);
    drawArmViz(data.arm);

    if (cam.image) {
      document.getElementById('camera-feed').src = cam.image;
    }
  } catch(e) {
    console.error('Poll error:', e);
  }
}

async function nextMove() {
  document.getElementById('btn-move').disabled = true;
  await fetch('/api/move', {method: 'POST'});
}

async function toggleAuto() {
  autoPlaying = !autoPlaying;
  const btn = document.getElementById('btn-auto');
  if (autoPlaying) {
    btn.textContent = '⏸ Stop Auto';
    btn.classList.remove('success');
    btn.classList.add('danger');
    await fetch('/api/autoplay', {method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action: 'start'})});
  } else {
    btn.textContent = '⚡ Auto Play';
    btn.classList.remove('danger');
    btn.classList.add('success');
    await fetch('/api/autoplay', {method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action: 'stop'})});
  }
}

async function resetGame() {
  autoPlaying = false;
  const btn = document.getElementById('btn-auto');
  btn.textContent = '⚡ Auto Play';
  btn.classList.remove('danger');
  btn.classList.add('success');
  await fetch('/api/reset', {method: 'POST'});
}

// Init
buildBoard();
poll();
pollInterval = setInterval(poll, 300);
</script>
</body>
</html>
"""


# ═════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  🤖 ROBOGAMBIT — Full Hardware Visual Simulator     ║")
    print("║  Open: http://localhost:5050                        ║")
    print("╚══════════════════════════════════════════════════════╝\n")
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
