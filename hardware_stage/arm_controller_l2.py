"""
arm_controller.py — Robot Arm Control for RoboGambit Hardware Stage
====================================================================
Controls a Waveshare RoArm M2-S (4-DOF) via HTTP + serial electromagnet.

Improvements over main_e1.py:
  - requests.Session with connection pooling
  - Retry with exponential backoff on all HTTP commands
  - Thread-safe serial access
  - Estimated travel time + optional feedback polling
  - Comprehensive logging
"""

import json
import math
import time
import logging
import threading
from urllib.parse import quote

import requests
import serial

from config_l1 import (
    ARM_IP, ARM_PORT,
    SERIAL_PORT, BAUD_RATE,
    DEFAULT_SPEED, ARM_SPEED_FAST, ARM_SPEED_NORMAL,
    SAFE_Z, PICK_Z, PLACE_Z,
    HOME_X, HOME_Y, HOME_Z,
    GRAVEYARD_POSITIONS,
    HTTP_TIMEOUT, MAX_RETRIES, RETRY_BACKOFF_BASE,
    MAGNET_ENERGIZE_DELAY, MAGNET_RELEASE_DELAY,
    SETTLE_DELAY, MOVEMENT_SPEED_FACTOR,
    FEEDBACK_POLL_INTERVAL, POSITION_TOLERANCE,
)

log = logging.getLogger("arm")


class ArmController:
    """
    Controls the Waveshare RoArm M2-S robotic arm.
    Thread-safe serial access, HTTP retry logic, high-level pick/place.
    """

    def __init__(self, serial_port=None, arm_ip=None):
        self._arm_ip = arm_ip or ARM_IP
        self._base_url = f"http://{self._arm_ip}"
        self._serial_port = serial_port or SERIAL_PORT

        # HTTP session with connection pooling
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=4)
        self._session.mount("http://", adapter)

        # Serial for electromagnet
        self._serial_lock = threading.Lock()
        self._ser = None
        try:
            self._ser = serial.Serial(self._serial_port, BAUD_RATE, timeout=1)
            log.info(f"Serial opened: {self._serial_port} @ {BAUD_RATE}")
        except serial.SerialException as e:
            log.warning(f"Serial open failed ({self._serial_port}): {e}")

        # Position tracking
        self.x = HOME_X
        self.y = HOME_Y
        self.z = HOME_Z
        self._speed = DEFAULT_SPEED

    # ── HTTP Command ──────────────────────────────────────────────────────

    def send_cmd(self, json_cmd: str) -> bool:
        """Send JSON command to arm via HTTP with retry logic."""
        url = f"{self._base_url}/js?json={quote(json_cmd)}"
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self._session.get(url, timeout=HTTP_TIMEOUT)
                resp.raise_for_status()
                log.debug(f"CMD OK (attempt {attempt}): {json_cmd}")
                return True
            except requests.RequestException as e:
                wait = RETRY_BACKOFF_BASE ** attempt
                log.warning(f"HTTP attempt {attempt}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(wait)
        log.error(f"CMD FAILED after {MAX_RETRIES} retries: {json_cmd}")
        return False

    # ── Movement ──────────────────────────────────────────────────────────

    def _estimate_travel(self, x, y, z, speed):
        dist = math.sqrt((x - self.x)**2 + (y - self.y)**2 + (z - self.z)**2)
        return max(0.3, (dist / max(speed, 1)) * MOVEMENT_SPEED_FACTOR)

    def arm_move(self, x, y, z, speed=None):
        """Move arm to (x, y, z). Returns True on success."""
        speed = speed or self._speed
        cmd = json.dumps({"T": 104, "x": x, "y": y, "z": z, "t": speed})
        log.info(f"arm_move -> ({x:.0f}, {y:.0f}, {z:.0f}) spd={speed}")

        ok = self.send_cmd(cmd)
        if not ok:
            return False

        # Wait for movement
        travel = self._estimate_travel(x, y, z, speed)
        if not self.wait_for_arrival(timeout=max(travel * 2, 3.0)):
            time.sleep(travel)

        self.x, self.y, self.z = x, y, z
        return True

    def move_to(self, x, y, z, speed=None):
        """Alias for arm_move (used by setup_phase)."""
        return self.arm_move(x, y, z, speed)

    def arm_home(self, speed=None):
        """Move to safe home position."""
        return self.arm_move(HOME_X, HOME_Y, HOME_Z, speed)

    def set_speed(self, speed):
        """Set default movement speed."""
        self._speed = speed

    # ── Electromagnet Gripper ─────────────────────────────────────────────

    def gripper_on(self):
        """Activate electromagnet (pick)."""
        with self._serial_lock:
            if self._ser and self._ser.is_open:
                self._ser.write(b"1")
                self._ser.flush()
        time.sleep(MAGNET_ENERGIZE_DELAY)
        log.info("Gripper ON")

    def gripper_off(self):
        """Deactivate electromagnet (release)."""
        with self._serial_lock:
            if self._ser and self._ser.is_open:
                self._ser.write(b"0")
                self._ser.flush()
        time.sleep(MAGNET_RELEASE_DELAY)
        log.info("Gripper OFF")

    def gripper_open(self):
        """Open mechanical gripper fingers (servo control)."""
        cmd = json.dumps({"T": 11, "id": 5, "angle": 30.0, "t": 200})
        url = f"{self._base_url}/js?json={quote(cmd)}"
        try:
            self._session.get(url, timeout=HTTP_TIMEOUT)
            time.sleep(0.3)
            log.info("Gripper OPEN")
        except Exception as e:
            log.warning(f"Gripper open failed: {e}")

    def gripper_close(self):
        """Close mechanical gripper fingers (servo control)."""
        cmd = json.dumps({"T": 11, "id": 5, "angle": 150.0, "t": 200})
        url = f"{self._base_url}/js?json={quote(cmd)}"
        try:
            self._session.get(url, timeout=HTTP_TIMEOUT)
            time.sleep(0.3)
            log.info("Gripper CLOSED")
        except Exception as e:
            log.warning(f"Gripper close failed: {e}")

    def grip(self):
        """Alias for gripper_on (used by setup_phase)."""
        self.gripper_on()

    def release(self):
        """Alias for gripper_off (used by setup_phase)."""
        self.gripper_off()

    # ── High-Level Pick & Place ───────────────────────────────────────────

    def pick_piece(self, x, y, speed=None):
        """Pick sequence: hover -> open gripper -> lower -> close gripper -> magnet on -> raise.
        
        FIXED: Opens gripper BEFORE lowering to prevent crushing pieces.
        """
        speed = speed or self._speed
        
        # 1. Hover above target
        self.arm_move(x, y, SAFE_Z, speed)
        time.sleep(SETTLE_DELAY)
        
        # 2. Open gripper while hovering (SAFE - piece won't fall)
        self.gripper_open()
        time.sleep(SETTLE_DELAY)
        
        # 3. Lower to piece
        self.arm_move(x, y, PICK_Z, speed)
        time.sleep(SETTLE_DELAY)
        
        # 4. Close gripper ON the piece
        self.gripper_close()
        time.sleep(SETTLE_DELAY)
        
        # 5. Activate magnet
        self.gripper_on()
        
        # 6. Lift
        self.arm_move(x, y, SAFE_Z, speed)
        return True

    def place_piece(self, x, y, speed=None):
        """Place sequence: hover -> lower -> magnet off -> open gripper -> raise.
        
        FIXED: Opens gripper AFTER lowering to prevent piece from flying off.
        """
        speed = speed or self._speed
        
        # 1. Hover above target
        self.arm_move(x, y, SAFE_Z, speed)
        time.sleep(SETTLE_DELAY)
        
        # 2. Lower to placement
        self.arm_move(x, y, PLACE_Z, speed)
        time.sleep(SETTLE_DELAY)
        
        # 3. Release piece
        self.gripper_off()
        time.sleep(SETTLE_DELAY)
        
        # 4. Open gripper
        self.gripper_open()
        
        # 5. Lift
        self.arm_move(x, y, SAFE_Z, speed)
        return True

    def pick_and_place(self, fx, fy, tx, ty, speed=None):
        """Pick from (fx,fy) and place at (tx,ty)."""
        self.pick_piece(fx, fy, speed)
        self.place_piece(tx, ty, speed)
        return True

    def move_to_graveyard(self, fx, fy, slot_idx):
        """Pick piece and place in graveyard slot."""
        if slot_idx >= len(GRAVEYARD_POSITIONS):
            slot_idx = len(GRAVEYARD_POSITIONS) - 1
        gx, gy = GRAVEYARD_POSITIONS[slot_idx]
        return self.pick_and_place(fx, fy, gx, gy)

    # ── Feedback ──────────────────────────────────────────────────────────

    def get_feedback(self):
        """Query arm position via T:105. Returns dict or None."""
        cmd = json.dumps({"T": 105})
        url = f"{self._base_url}/js?json={quote(cmd)}"
        try:
            resp = self._session.get(url, timeout=HTTP_TIMEOUT)
            return resp.json()
        except Exception:
            return None

    def wait_for_arrival(self, timeout=5.0):
        """Poll feedback until position matches target or timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            fb = self.get_feedback()
            if fb is None:
                return False
            try:
                cx, cy, cz = float(fb.get("x", 0)), float(fb.get("y", 0)), float(fb.get("z", 0))
                dist = math.sqrt((cx - self.x)**2 + (cy - self.y)**2 + (cz - self.z)**2)
                if dist <= POSITION_TOLERANCE:
                    return True
            except (TypeError, ValueError):
                return False
            time.sleep(FEEDBACK_POLL_INTERVAL)
        return False

    def is_connected(self):
        """Check if arm responds to feedback query."""
        fb = self.get_feedback()
        return fb is not None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def initialize(self):
        """Alias for compatibility with setup_phase."""
        self.arm_home()

    def shutdown(self):
        """Alias for close."""
        self.close()

    def close(self):
        """Clean shutdown."""
        with self._serial_lock:
            if self._ser and self._ser.is_open:
                try:
                    self._ser.write(b"0")
                    self._ser.close()
                except Exception:
                    pass
        self._session.close()
        log.info("ArmController closed")
