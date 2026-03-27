"""
test_simulator.py — Complete Hardware Simulation for RoboGambit
================================================================
Simulates the ENTIRE hardware environment so you can test main.py,
perception.py, config.py, and game.py end-to-end WITHOUT any real
hardware (no arm, no camera, no serial port needed).

Simulates:
  - Waveshare RoArm M2-S HTTP API (T:104 move, T:105 feedback, T:11 servo)
  - Camera TCP server (generates 1920x1080 frames with ArUco markers)
  - Serial electromagnet (mock)
  - Full game: AI (white) vs AI (black) through the hardware stack

Usage:
    python test_simulator.py                 # Full AI vs AI simulation
    python test_simulator.py --unit          # Quick unit tests only
    python test_simulator.py --moves 5       # Run only 5 moves per side

Requirements:
    pip install numpy opencv-python requests
    (pyserial NOT needed for simulation)
"""

import sys
import os
import json
import math
import time
import struct
import socket
import logging
import argparse
import threading
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote
from io import BytesIO
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)-12s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("simulator")

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION OVERRIDES (use localhost instead of real hardware)
# ═════════════════════════════════════════════════════════════════════════════

SIM_ARM_HOST    = "127.0.0.1"
SIM_ARM_PORT    = 18080        # mock arm HTTP server
SIM_CAMERA_HOST = "127.0.0.1"
SIM_CAMERA_PORT = 19999        # mock camera TCP server

# World-to-pixel transform for synthetic camera frames
# pixel_x = IMG_CX + world_x * SCALE
# pixel_y = IMG_CY - world_y * SCALE  (Y flipped)
IMG_W, IMG_H = 1920, 1080
IMG_CX, IMG_CY = IMG_W // 2, IMG_H // 2
SCALE = 2.0  # pixels per mm

MARKER_PX_SIZE = 30  # marker size in pixels for drawing

# ═════════════════════════════════════════════════════════════════════════════
#  MOCK ARM HTTP SERVER
# ═════════════════════════════════════════════════════════════════════════════

class ArmState:
    """Tracks simulated arm position."""
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 180.0
        self.speed = 500
        self.gripper_angle = 30.0
        self.magnet = False
        self.lock = threading.Lock()
        self.move_count = 0
        self.log = logging.getLogger("mock_arm")

    def move_to(self, x, y, z, speed):
        """Simulate movement with realistic delay."""
        with self.lock:
            dist = math.sqrt((x - self.x)**2 + (y - self.y)**2 + (z - self.z)**2)
            travel_time = max(0.05, dist / max(speed, 1) * 0.5)
            # Simulate movement (short delay, not real-time)
            time.sleep(min(travel_time, 0.2))
            self.x, self.y, self.z = x, y, z
            self.speed = speed
            self.move_count += 1
            self.log.debug(f"Moved to ({x:.0f}, {y:.0f}, {z:.0f}) [{self.move_count}]")

    def get_feedback(self):
        with self.lock:
            return {"x": self.x, "y": self.y, "z": self.z,
                    "speed": self.speed, "T": 105}

    def set_servo(self, servo_id, angle, speed):
        with self.lock:
            if servo_id == 5:
                self.gripper_angle = angle
                self.log.debug(f"Servo {servo_id} -> {angle}°")


# Global arm state (shared between server and test runner)
arm_state = ArmState()


class ArmHTTPHandler(BaseHTTPRequestHandler):
    """HTTP handler mimicking the Waveshare RoArm M2-S API."""

    def log_message(self, format, *args):
        pass  # suppress default HTTP logging

    def do_GET(self):
        try:
            parsed = urlparse(self.path)

            if parsed.path == "/js":
                query = parse_qs(parsed.query)
                json_str = unquote(query.get("json", ["{}"])[0])
                cmd = json.loads(json_str)
                cmd_type = cmd.get("T", 0)

                if cmd_type == 104:
                    # Movement command
                    x = float(cmd.get("x", 0))
                    y = float(cmd.get("y", 0))
                    z = float(cmd.get("z", 0))
                    t = int(cmd.get("t", 500))
                    arm_state.move_to(x, y, z, t)
                    self._respond(200, {"status": "ok", "T": 104})

                elif cmd_type == 105:
                    # Feedback query
                    fb = arm_state.get_feedback()
                    self._respond(200, fb)

                elif cmd_type == 11:
                    # Servo control
                    sid = int(cmd.get("id", 0))
                    angle = float(cmd.get("angle", 0))
                    spd = int(cmd.get("t", 200))
                    arm_state.set_servo(sid, angle, spd)
                    self._respond(200, {"status": "ok", "T": 11})

                else:
                    self._respond(200, {"status": "ok"})
            else:
                self._respond(404, {"error": "not found"})

        except Exception as e:
            self._respond(500, {"error": str(e)})

    def _respond(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_arm_server():
    """Start mock arm HTTP server in a daemon thread."""
    server = HTTPServer((SIM_ARM_HOST, SIM_ARM_PORT), ArmHTTPHandler)
    server.timeout = 0.5
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="MockArm")
    thread.start()
    log.info(f"Mock ARM server started on {SIM_ARM_HOST}:{SIM_ARM_PORT}")
    return server


# ═════════════════════════════════════════════════════════════════════════════
#  MOCK CAMERA TCP SERVER (generates ArUco marker frames)
# ═════════════════════════════════════════════════════════════════════════════

class BoardSimulator:
    """Manages the simulated chess board state and generates camera frames."""

    # Initial board layout: same as game.py
    INITIAL_BOARD = np.array([
        [2, 3, 4, 5, 3, 2],   # rank 1: wN wB wQ wK wB wN
        [1, 1, 1, 1, 1, 1],   # rank 2: 6 white pawns
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [6, 6, 6, 6, 6, 6],   # rank 5: 6 black pawns
        [7, 8, 9, 10, 8, 7],  # rank 6: bN bB bQ bK bB bN
    ], dtype=int)

    # Cell centres in world coords (mm)
    CELL_X = [-150.0, -90.0, -30.0, 30.0, 90.0, 150.0]
    CELL_Y = [-150.0, -90.0, -30.0, 30.0, 90.0, 150.0]

    # Corner marker world positions
    CORNERS = {
        21: (212.5, 212.5),
        22: (212.5, -212.5),
        23: (-212.5, -212.5),
        24: (-212.5, 212.5),
    }

    def __init__(self):
        self.board = self.INITIAL_BOARD.copy()
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.lock = threading.Lock()
        self._marker_cache = {}
        self.log = logging.getLogger("mock_cam")

    def set_board(self, board):
        with self.lock:
            self.board = board.copy()

    def get_board(self):
        with self.lock:
            return self.board.copy()

    def _get_marker_img(self, marker_id: int, size: int) -> np.ndarray:
        """Get or generate a cached ArUco marker image."""
        key = (marker_id, size)
        if key not in self._marker_cache:
            marker = cv2.aruco.generateImageMarker(self.aruco_dict, marker_id, size)
            self._marker_cache[key] = marker
        return self._marker_cache[key]

    def world_to_pixel(self, wx: float, wy: float):
        """Convert world (mm) to pixel coordinates."""
        px = int(IMG_CX + wx * SCALE)
        py = int(IMG_CY - wy * SCALE)
        return px, py

    def generate_frame(self) -> np.ndarray:
        """Generate a synthetic camera frame with ArUco markers."""
        # Create dark green background (chess board look)
        frame = np.full((IMG_H, IMG_W, 3), (40, 60, 40), dtype=np.uint8)

        # Draw board grid
        for r in range(7):
            wy = -180.0 + r * 60.0
            px1, py1 = self.world_to_pixel(-180.0, wy)
            px2, py2 = self.world_to_pixel(180.0, wy)
            cv2.line(frame, (px1, py1), (px2, py2), (80, 100, 80), 1)
        for c in range(7):
            wx = -180.0 + c * 60.0
            px1, py1 = self.world_to_pixel(wx, -180.0)
            px2, py2 = self.world_to_pixel(wx, 180.0)
            cv2.line(frame, (px1, py1), (px2, py2), (80, 100, 80), 1)

        # Draw cell shading
        for r in range(6):
            for c in range(6):
                if (r + c) % 2 == 0:
                    wx1, wy1 = -180.0 + c * 60.0, -180.0 + r * 60.0
                    wx2, wy2 = wx1 + 60.0, wy1 + 60.0
                    p1 = self.world_to_pixel(wx1, wy2)
                    p2 = self.world_to_pixel(wx2, wy1)
                    cv2.rectangle(frame, p1, p2, (60, 80, 60), -1)

        # Draw corner markers (IDs 21-24)
        for mid, (wx, wy) in self.CORNERS.items():
            self._place_marker(frame, mid, wx, wy, MARKER_PX_SIZE)

        # Draw piece markers
        with self.lock:
            for r in range(6):
                for c in range(6):
                    pid = int(self.board[r][c])
                    if pid > 0:
                        wx, wy = self.CELL_X[c], self.CELL_Y[r]
                        self._place_marker(frame, pid, wx, wy, MARKER_PX_SIZE)

        return frame

    def _place_marker(self, frame, marker_id, wx, wy, size):
        """Place an ArUco marker at world coordinates in the frame."""
        px, py = self.world_to_pixel(wx, wy)
        half = size // 2

        # Bounds check
        if (px - half < 0 or px + half >= IMG_W or
            py - half < 0 or py + half >= IMG_H):
            return

        marker = self._get_marker_img(marker_id, size)
        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

        y1, y2 = py - half, py - half + size
        x1, x2 = px - half, px - half + size

        if y2 <= IMG_H and x2 <= IMG_W and y1 >= 0 and x1 >= 0:
            frame[y1:y2, x1:x2] = marker_bgr

    def encode_frame(self, frame) -> bytes:
        """Encode frame as JPEG and wrap in the TCP protocol (8-byte header + data)."""
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        jpeg_bytes = jpeg.tobytes()
        header = struct.pack("Q", len(jpeg_bytes))
        return header + jpeg_bytes


# Global board simulator
board_sim = BoardSimulator()


def camera_client_handler(conn: socket.socket, addr):
    """Handle a single camera client connection."""
    cam_log = logging.getLogger("mock_cam")
    cam_log.info(f"Camera client connected: {addr}")
    try:
        while True:
            frame = board_sim.generate_frame()
            data = board_sim.encode_frame(frame)
            try:
                conn.sendall(data)
            except (BrokenPipeError, ConnectionResetError):
                break
            time.sleep(0.033)  # ~30 fps
    except Exception as e:
        cam_log.debug(f"Camera client {addr} disconnected: {e}")
    finally:
        conn.close()


def start_camera_server():
    """Start mock camera TCP server in a daemon thread."""
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((SIM_CAMERA_HOST, SIM_CAMERA_PORT))
    server_sock.listen(2)
    server_sock.settimeout(1.0)

    def accept_loop():
        while True:
            try:
                conn, addr = server_sock.accept()
                t = threading.Thread(target=camera_client_handler, args=(conn, addr),
                                     daemon=True)
                t.start()
            except socket.timeout:
                continue
            except Exception:
                break

    thread = threading.Thread(target=accept_loop, daemon=True, name="MockCamera")
    thread.start()
    log.info(f"Mock CAMERA server started on {SIM_CAMERA_HOST}:{SIM_CAMERA_PORT}")
    return server_sock


# ═════════════════════════════════════════════════════════════════════════════
#  MOCK SERIAL PORT
# ═════════════════════════════════════════════════════════════════════════════

class MockSerial:
    """Mock serial port that simulates the electromagnet controller."""

    def __init__(self, port=None, baudrate=115200, timeout=1, **kwargs):
        self.port = port
        self.baudrate = baudrate
        self.is_open = True
        self._magnet_state = False
        self._log = logging.getLogger("mock_serial")
        self._log.info(f"Mock serial opened: {port} @ {baudrate}")

    def write(self, data: bytes):
        if data == b"1":
            self._magnet_state = True
            self._log.debug("Magnet ON")
        elif data == b"0":
            self._magnet_state = False
            self._log.debug("Magnet OFF")
        return len(data)

    def flush(self):
        pass

    def read(self, size=1):
        return b""

    def close(self):
        self.is_open = False
        self._log.info("Mock serial closed")

    def setRTS(self, val):
        pass

    def setDTR(self, val):
        pass


# ═════════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
# ═════════════════════════════════════════════════════════════════════════════

def run_unit_tests():
    """Quick unit tests for each component."""
    print("\n" + "=" * 60)
    print("  UNIT TESTS")
    print("=" * 60)
    passed = 0
    failed = 0

    def test(name, fn):
        nonlocal passed, failed
        try:
            fn()
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            traceback.print_exc()
            failed += 1

    # ── Config tests ──
    def test_config():
        from config import (cell_to_world, rc_to_label, world_to_cell,
                            CELL_CENTERS_X, CELL_CENTERS_Y, BOARD_SIZE)
        assert BOARD_SIZE == 6
        wx, wy = cell_to_world(0, 0)
        assert wx == -150.0 and wy == -150.0, f"A1 should be (-150,-150), got ({wx},{wy})"
        wx, wy = cell_to_world(5, 5)
        assert wx == 150.0 and wy == 150.0, f"F6 should be (150,150), got ({wx},{wy})"
        assert rc_to_label(0, 0) == "A1"
        assert rc_to_label(5, 5) == "F6"
        r, c = world_to_cell(-150, -150)
        assert (r, c) == (0, 0), f"Expected (0,0) got ({r},{c})"

    test("Config: cell_to_world, rc_to_label, world_to_cell", test_config)

    # ── Game engine tests ──
    def test_game_engine():
        import game
        board = np.array([
            [2, 3, 4, 5, 3, 2],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [6, 6, 6, 6, 6, 6],
            [7, 8, 9, 10, 8, 7],
        ], dtype=int)
        mv_w = game.get_best_move(board, True)
        assert mv_w is not None, "White move should not be None"
        assert ":" in mv_w and "->" in mv_w, f"Bad format: {mv_w}"
        mv_b = game.get_best_move(board, False)
        assert mv_b is not None, "Black move should not be None"

    test("Game engine: get_best_move (white & black)", test_game_engine)

    # ── Move parsing test ──
    def test_move_parsing():
        # Patch serial before importing main
        with patch("serial.Serial", MockSerial):
            # We need to import parse_move from main
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            # Import just the parse function
            from main import parse_move
            p = parse_move("1:A2->A3")
            assert p is not None
            assert p["piece"] == 1
            assert p["from_cell"] == (1, 0)
            assert p["to_cell"] == (2, 0)
            assert p["promotion"] is None

            p2 = parse_move("1:A5->A6=4")
            assert p2 is not None
            assert p2["promotion"] == 4

    test("Move parsing: standard + promotion", test_move_parsing)

    # ── ArUco frame generation test ──
    def test_aruco_generation():
        frame = board_sim.generate_frame()
        assert frame.shape == (IMG_H, IMG_W, 3), f"Wrong shape: {frame.shape}"

        # Try to detect markers in the generated frame
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        detected = set() if ids is None else set(ids.flatten())
        # Should detect at least the 4 corner markers and some piece markers
        for cid in [21, 22, 23, 24]:
            assert cid in detected, f"Corner marker {cid} not detected! Detected: {detected}"

        piece_count = len([mid for mid in detected if 1 <= mid <= 10])
        assert piece_count >= 8, f"Expected >= 8 piece markers, found {piece_count}"

    test("ArUco frame: generation + detection", test_aruco_generation)

    # ── Mock arm HTTP test ──
    def test_mock_arm():
        import requests
        from urllib.parse import quote

        base = f"http://{SIM_ARM_HOST}:{SIM_ARM_PORT}"

        # Test feedback
        cmd = json.dumps({"T": 105})
        resp = requests.get(f"{base}/js?json={quote(cmd)}", timeout=3)
        assert resp.status_code == 200
        fb = resp.json()
        assert "x" in fb and "y" in fb and "z" in fb

        # Test movement
        cmd = json.dumps({"T": 104, "x": 100, "y": 50, "z": 80, "t": 500})
        resp = requests.get(f"{base}/js?json={quote(cmd)}", timeout=5)
        assert resp.status_code == 200

        # Verify position updated
        cmd = json.dumps({"T": 105})
        resp = requests.get(f"{base}/js?json={quote(cmd)}", timeout=3)
        fb = resp.json()
        assert abs(fb["x"] - 100) < 1, f"X should be ~100, got {fb['x']}"
        assert abs(fb["y"] - 50) < 1, f"Y should be ~50, got {fb['y']}"

    test("Mock arm HTTP: feedback + movement", test_mock_arm)

    # ── Mock camera TCP test ──
    def test_mock_camera():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect((SIM_CAMERA_HOST, SIM_CAMERA_PORT))

        # Read one frame
        header = b""
        while len(header) < 8:
            header += sock.recv(8 - len(header))
        size = struct.unpack("Q", header)[0]
        assert 0 < size < 10_000_000, f"Bad frame size: {size}"

        data = b""
        while len(data) < size:
            data += sock.recv(min(4096, size - len(data)))

        # Decode
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        assert frame is not None, "Failed to decode frame"
        assert frame.shape == (IMG_H, IMG_W, 3)

        sock.close()

    test("Mock camera TCP: connect + read frame", test_mock_camera)

    # ── Perception integration test ──
    def test_perception_integration():
        # Override config so connect_camera reads from config at call time
        import config
        orig_cam_ip = config.CAMERA_IP
        orig_cam_port = config.CAMERA_PORT
        config.CAMERA_IP = SIM_CAMERA_HOST
        config.CAMERA_PORT = SIM_CAMERA_PORT

        try:
            from perception import PerceptionSystem
            perc = PerceptionSystem()
            # Pass host/port directly to connect to mock server
            assert perc.connect_camera(host=SIM_CAMERA_HOST, port=SIM_CAMERA_PORT), \
                "Camera connection failed"

            board = perc.get_board_state()
            assert board is not None, "Board state is None"
            assert board.shape == (6, 6), f"Wrong board shape: {board.shape}"

            # Should have pieces on the board
            piece_count = np.count_nonzero(board)
            assert piece_count >= 8, f"Expected >= 8 pieces, found {piece_count}"

            perc.stop()
        finally:
            config.CAMERA_IP = orig_cam_ip
            config.CAMERA_PORT = orig_cam_port

    test("Perception: connect + board detection", test_perception_integration)

    # ── Summary ──
    total = passed + failed
    print(f"\n  Results: {passed}/{total} passed", end="")
    if failed:
        print(f", {failed} FAILED")
    else:
        print(" — ALL PASSED ✓")
    print("=" * 60)
    return failed == 0


# ═════════════════════════════════════════════════════════════════════════════
#  FULL GAME SIMULATION (AI vs AI through the hardware stack)
# ═════════════════════════════════════════════════════════════════════════════

def apply_move_to_sim_board(board, move_str):
    """Apply a move string to the simulated board."""
    if not move_str or ":" not in move_str or "->" not in move_str:
        return board

    try:
        colon = move_str.index(":")
        piece_id = int(move_str[:colon])
        rest = move_str[colon + 1:]

        promotion = None
        if "=" in rest:
            rest, promo_str = rest.split("=")
            promotion = int(promo_str)

        arrow = rest.index("->")
        src = rest[:arrow]
        dst = rest[arrow + 2:]

        sc = ord(src[0]) - ord("A")
        sr = int(src[1]) - 1
        dc = ord(dst[0]) - ord("A")
        dr = int(dst[1]) - 1

        new_board = board.copy()
        new_board[sr][sc] = 0
        new_board[dr][dc] = promotion if promotion else piece_id
        return new_board
    except Exception as e:
        log.error(f"Failed to apply move '{move_str}': {e}")
        return board


def run_full_simulation(max_moves=10):
    """
    Run a full AI vs AI game through the complete hardware stack.
    White uses main.py's game engine, Black uses the engine directly.
    """
    print("\n" + "=" * 60)
    print("  FULL GAME SIMULATION (AI vs AI)")
    print(f"  Max moves per side: {max_moves}")
    print("=" * 60)

    import config
    import game

    # Override config to use simulators
    config.ARM_IP = SIM_ARM_HOST
    config.ARM_PORT = SIM_ARM_PORT
    config.ARM_BASE_URL = f"http://{SIM_ARM_HOST}:{SIM_ARM_PORT}"
    config.CAMERA_IP = SIM_CAMERA_HOST
    config.CAMERA_PORT = SIM_CAMERA_PORT
    config.SERIAL_PORT = "MOCK"

    # Reset board simulator to initial position
    board_sim.set_board(BoardSimulator.INITIAL_BOARD.copy())

    current_board = BoardSimulator.INITIAL_BOARD.copy()
    move_num = 0
    white_turn = True

    print(f"\n  Initial board:")
    _print_sim_board(current_board)

    try:
        for i in range(max_moves * 2):
            side = "WHITE" if white_turn else "BLACK"
            move_num += 1

            print(f"\n  --- Move {(move_num + 1) // 2} ({side}) ---")

            # Get best move from engine
            t0 = time.time()
            move_str = game.get_best_move(current_board, white_turn)
            elapsed = time.time() - t0

            if move_str is None:
                print(f"  {side} has no moves — GAME OVER")
                break

            print(f"  Engine: {move_str}  ({elapsed:.2f}s)")

            # Apply move
            new_board = apply_move_to_sim_board(current_board, move_str)

            # Update the camera simulator's board
            board_sim.set_board(new_board)

            # Simulate arm movement for the current player's move
            _simulate_arm_move(move_str, config)

            current_board = new_board
            _print_sim_board(current_board)

            # Check for game over (king captured)
            has_wk = np.any(current_board == 5)
            has_bk = np.any(current_board == 10)
            if not has_wk:
                print(f"\n  ★ BLACK WINS — White king captured!")
                break
            if not has_bk:
                print(f"\n  ★ WHITE WINS — Black king captured!")
                break

            white_turn = not white_turn

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    except Exception as e:
        print(f"\n  Error: {e}")
        traceback.print_exc()

    print(f"\n  Game ended after {move_num} half-moves.")
    print("=" * 60)


def _simulate_arm_move(move_str, config):
    """Simulate arm picking and placing a piece via the mock HTTP server."""
    import requests
    from urllib.parse import quote

    try:
        colon = move_str.index(":")
        rest = move_str[colon + 1:]
        if "=" in rest:
            rest = rest.split("=")[0]
        arrow = rest.index("->")
        src = rest[:arrow]
        dst = rest[arrow + 2:]

        sc = ord(src[0]) - ord("A")
        sr = int(src[1]) - 1
        dc = ord(dst[0]) - ord("A")
        dr = int(dst[1]) - 1

        from_x, from_y = config.CELL_CENTERS_X[sc], config.CELL_CENTERS_Y[sr]
        to_x, to_y = config.CELL_CENTERS_X[dc], config.CELL_CENTERS_Y[dr]

        base = config.ARM_BASE_URL

        # Pick
        cmd = json.dumps({"T": 104, "x": from_x, "y": from_y, "z": 8.0, "t": 500})
        requests.get(f"{base}/js?json={quote(cmd)}", timeout=5)

        # Place
        cmd = json.dumps({"T": 104, "x": to_x, "y": to_y, "z": 10.0, "t": 500})
        requests.get(f"{base}/js?json={quote(cmd)}", timeout=5)

        # Home
        cmd = json.dumps({"T": 104, "x": 0, "y": 0, "z": 180, "t": 500})
        requests.get(f"{base}/js?json={quote(cmd)}", timeout=5)

    except Exception as e:
        log.debug(f"Arm sim move error: {e}")


PIECE_SYM = {
    0: " . ", 1: "wP ", 2: "wN ", 3: "wB ", 4: "wQ ", 5: "wK ",
    6: "bP ", 7: "bN ", 8: "bB ", 9: "bQ ", 10: "bK ",
}


def _print_sim_board(board):
    print("       A   B   C   D   E   F")
    for r in range(5, -1, -1):
        row = f"    {r+1} "
        for c in range(6):
            row += PIECE_SYM.get(int(board[r][c]), " ? ")
        print(row)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RoboGambit Hardware Test Simulator")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--moves", type=int, default=10,
                        help="Max moves per side in full simulation (default: 10)")
    parser.add_argument("--save-frame", action="store_true",
                        help="Save a sample camera frame as test_frame.png")
    args = parser.parse_args()

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  ROBOGAMBIT HARDWARE TEST SIMULATOR                      ║")
    print("║  Simulates: ARM (HTTP) + CAMERA (TCP) + SERIAL (Mock)    ║")
    print("╚" + "═" * 58 + "╝")

    # Ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    os.chdir(script_dir)

    # Patch serial.Serial BEFORE any imports that use it
    import serial as serial_module
    serial_module.Serial = MockSerial

    # Start mock servers
    arm_server = start_arm_server()
    cam_server = start_camera_server()
    time.sleep(0.5)  # let servers start

    # Save sample frame if requested
    if args.save_frame:
        frame = board_sim.generate_frame()
        cv2.imwrite("test_frame.png", frame)
        print(f"  Saved test_frame.png ({frame.shape[1]}×{frame.shape[0]})")

    try:
        if args.unit:
            success = run_unit_tests()
            sys.exit(0 if success else 1)
        else:
            # Run both unit tests and full simulation
            success = run_unit_tests()
            if success:
                run_full_simulation(max_moves=args.moves)
            else:
                print("\n  ⚠ Unit tests failed — skipping full simulation")
                sys.exit(1)
    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        arm_server.shutdown()
        cam_server.close()
        print("\n  Simulators stopped. Done.")


if __name__ == "__main__":
    main()
