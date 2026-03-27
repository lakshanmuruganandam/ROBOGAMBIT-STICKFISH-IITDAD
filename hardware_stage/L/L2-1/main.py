"""
Main hardware-stage controller for Team L2.

Usage examples:
  python main.py --white
  python main.py --black
  python main.py --calibrate
"""

import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
import signal
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
import requests
import requests.adapters

import game
from config import (
    ARM_ARRIVAL_TIMEOUT,
    ARM_BASE_URL,
    ARM_IP,
    ARM_POLL_INTERVAL,
    ARM_SETTLE_TIME,
    BAUD_RATE,
    BOARD_SIZE,
    EMPTY,
    GRAVEYARD_BLACK,
    GRAVEYARD_WHITE,
    GRIPPER_ACTIVATE_TIME,
    HOME_X,
    HOME_Y,
    HOME_Z,
    HTTP_FEEDBACK_TIMEOUT,
    HTTP_MOVE_SEND_TIMEOUT,
    MAX_RETRIES,
    MOVE_REGEX,
    BLACK_PIECE_IDS,
    PIECE_NAMES,
    POSITION_TOLERANCE_MM,
    RETRY_DELAY_BASE,
    SERIAL_PORT,
    SERIAL_TIMEOUT,
    Z_PICK,
    Z_PLACE,
    SPEED_FAST,
    SPEED_NORMAL,
    SPEED_SLOW,
    WHITE_PIECE_IDS,
    cell_to_rc,
    rc_to_label,
    rc_to_world,
)
from perception import PerceptionSystem

try:
    import serial  # type: ignore
except ImportError:
    serial = None


log = logging.getLogger("l2.main")
CHECKPOINT_PATH = Path(__file__).resolve().parent / "resume_checkpoint.json"
CHECKPOINT_VERSION = 1


def setup_logging(debug: bool = False, log_file: Optional[str] = None) -> Path:
    level = logging.DEBUG if debug else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    default_dir = Path(__file__).resolve().parent / "logs"
    default_dir.mkdir(parents=True, exist_ok=True)
    logfile_path = Path(log_file) if log_file else (default_dir / "l2_main.log")
    file_handler = RotatingFileHandler(logfile_path, maxBytes=3_000_000, backupCount=3, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    log.info("logging initialized: level=%s, file=%s", logging.getLevelName(level), logfile_path)
    return logfile_path


@dataclass
class ParsedMove:
    piece: int
    from_cell: Tuple[int, int]
    to_cell: Tuple[int, int]
    promotion: Optional[int]


@dataclass
class RuntimeStats:
    move_count: int = 0
    recoveries: int = 0
    invalid_moves: int = 0
    execution_retries: int = 0
    fallback_moves: int = 0
    opponent_rejects: int = 0
    camera_recovers: int = 0
    verify_failures: int = 0
    total_think_time_s: float = 0.0
    total_loop_time_s: float = 0.0


@dataclass
class PromotionTracker:
    promoted: Dict[Tuple[int, int], int]

    def __init__(self) -> None:
        self.promoted = {}

    def mark(self, row: int, col: int, piece_id: int) -> None:
        self.promoted[(row, col)] = piece_id

    def clear_at(self, row: int, col: int) -> None:
        self.promoted.pop((row, col), None)

    def apply(self, board: np.ndarray) -> np.ndarray:
        out = board.copy()
        for (r, c), pid in self.promoted.items():
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                if int(out[r, c]) != EMPTY:
                    out[r, c] = pid
        return out


def _stats_to_dict(stats: RuntimeStats) -> Dict[str, Any]:
    return {
        "move_count": stats.move_count,
        "recoveries": stats.recoveries,
        "invalid_moves": stats.invalid_moves,
        "execution_retries": stats.execution_retries,
        "fallback_moves": stats.fallback_moves,
        "opponent_rejects": stats.opponent_rejects,
        "camera_recovers": stats.camera_recovers,
        "verify_failures": stats.verify_failures,
        "total_think_time_s": stats.total_think_time_s,
        "total_loop_time_s": stats.total_loop_time_s,
    }


def _stats_from_dict(raw: Dict[str, Any]) -> RuntimeStats:
    return RuntimeStats(
        move_count=int(raw.get("move_count", 0)),
        recoveries=int(raw.get("recoveries", 0)),
        invalid_moves=int(raw.get("invalid_moves", 0)),
        execution_retries=int(raw.get("execution_retries", 0)),
        fallback_moves=int(raw.get("fallback_moves", 0)),
        opponent_rejects=int(raw.get("opponent_rejects", 0)),
        camera_recovers=int(raw.get("camera_recovers", 0)),
        verify_failures=int(raw.get("verify_failures", 0)),
        total_think_time_s=float(raw.get("total_think_time_s", 0.0)),
        total_loop_time_s=float(raw.get("total_loop_time_s", 0.0)),
    )


def _checkpoint_write(payload: Dict[str, Any]) -> None:
    payload["version"] = CHECKPOINT_VERSION
    payload["saved_at"] = time.time()
    tmp = CHECKPOINT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp.replace(CHECKPOINT_PATH)


def _checkpoint_load() -> Optional[Dict[str, Any]]:
    if not CHECKPOINT_PATH.exists():
        return None
    try:
        raw = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("failed to load checkpoint: %s", exc)
        return None
    if raw.get("version") != CHECKPOINT_VERSION:
        log.warning("checkpoint version mismatch, ignoring old checkpoint")
        return None
    return raw


def _checkpoint_clear() -> None:
    try:
        if CHECKPOINT_PATH.exists():
            CHECKPOINT_PATH.unlink()
    except Exception as exc:
        log.warning("failed to clear checkpoint: %s", exc)


class ArmController:
    def __init__(self) -> None:
        self.base_url = ARM_BASE_URL
        self.serial_conn = None
        self.x = HOME_X
        self.y = HOME_Y
        self.z = HOME_Z
        self.command_counter = 0
        self.serial_fallback_count = 0
        self.feedback_failures = 0

        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=2, pool_maxsize=4, max_retries=0)
        self._session.mount("http://", adapter)

        self._servo_enabled = os.getenv("ROBO_USE_SERVO_GRIPPER", "0") == "1"
        self._servo_id = int(os.getenv("ROBO_GRIPPER_SERVO_ID", "5"))
        self._servo_open_angle = int(os.getenv("ROBO_GRIPPER_OPEN_ANGLE", "90"))
        self._servo_close_angle = int(os.getenv("ROBO_GRIPPER_CLOSE_ANGLE", "20"))
        self._servo_speed = int(os.getenv("ROBO_GRIPPER_SERVO_SPEED", "250"))

        if serial is not None:
            try:
                self.serial_conn = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
                log.info("serial opened: %s @ %d", SERIAL_PORT, BAUD_RATE)
            except Exception as exc:
                log.warning("serial unavailable: %s", exc)
        else:
            log.warning("pyserial not installed, serial fallback disabled")

    def close(self) -> None:
        if self.serial_conn is not None:
            try:
                self.serial_conn.close()
            except Exception:
                pass
            self.serial_conn = None
        self._session.close()

    def check_connection(self) -> bool:
        payload = {"T": 105}
        for attempt in range(1, MAX_RETRIES + 1):
            if self._send_http(payload, timeout=HTTP_FEEDBACK_TIMEOUT, accept_timeout=False):
                return True
            time.sleep(RETRY_DELAY_BASE * attempt)
        return False

    @staticmethod
    def _distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _send_http(self, payload: Dict, timeout: float, accept_timeout: bool) -> bool:
        self.command_counter += 1
        url = f"{self.base_url}/js?json={quote(json.dumps(payload))}"
        t0 = time.monotonic()
        try:
            response = self._session.get(url, timeout=timeout)
            response.raise_for_status()
            log.debug("http cmd #%d ok in %.3fs: %s", self.command_counter, time.monotonic() - t0, payload)
            return True
        except requests.exceptions.ReadTimeout:
            if accept_timeout:
                log.debug("HTTP read timeout treated as accepted send for payload: %s", payload)
                return True
            log.warning("HTTP read timeout on payload: %s", payload)
            return False
        except requests.exceptions.ConnectTimeout:
            log.warning("HTTP connect timeout on payload: %s", payload)
            return False
        except requests.RequestException as exc:
            log.warning("HTTP error for payload %s: %s", payload, exc)
            return False
        except Exception as exc:
            log.exception("HTTP unexpected error for payload %s: %s", payload, exc)
            return False

    def _send_serial_line(self, text: str) -> bool:
        if self.serial_conn is None:
            log.debug("serial send skipped (no serial): %s", text)
            return False
        try:
            self.serial_conn.write((text + "\n").encode("utf-8"))
            self.serial_conn.flush()
            log.debug("serial send ok: %s", text)
            return True
        except Exception as exc:
            log.exception("serial write failed for %s: %s", text, exc)
            return False

    def _query_position(self) -> Optional[Tuple[float, float, float]]:
        payload = {"T": 105}
        try:
            url = f"{self.base_url}/js?json={quote(json.dumps(payload))}"
            response = self._session.get(url, timeout=HTTP_FEEDBACK_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            x = float(data.get("x", self.x))
            y = float(data.get("y", self.y))
            z = float(data.get("z", self.z))
            log.debug("feedback position: x=%.2f y=%.2f z=%.2f", x, y, z)
            return x, y, z
        except Exception as exc:
            self.feedback_failures += 1
            log.debug("feedback query failed (%d): %s", self.feedback_failures, exc)
            return None

    def move_xyz(self, x: float, y: float, z: float, speed: int = SPEED_NORMAL) -> bool:
        payload = {"T": 104, "x": float(x), "y": float(y), "z": float(z), "t": int(speed)}
        log.info("arm_move start -> x=%.1f y=%.1f z=%.1f speed=%d", x, y, z, speed)

        sent = False
        for attempt in range(1, MAX_RETRIES + 1):
            log.debug("move send attempt %d/%d: %s", attempt, MAX_RETRIES, payload)
            sent = self._send_http(payload, timeout=HTTP_MOVE_SEND_TIMEOUT, accept_timeout=True)
            if sent:
                break
            time.sleep(RETRY_DELAY_BASE * attempt)

        if not sent:
            # Fallback path for cases where WiFi command channel is unreliable.
            self.serial_fallback_count += 1
            log.warning("HTTP send failed for move, trying serial fallback #%d", self.serial_fallback_count)
            sent = self._send_serial_line(json.dumps(payload))

        if not sent:
            log.error("move command send failed: %s", payload)
            return False

        target = (x, y, z)
        dist = self._distance((self.x, self.y, self.z), target)
        estimated = max(0.8, dist / max(1.0, float(speed)) * 7.5)
        effective_timeout = min(ARM_ARRIVAL_TIMEOUT, max(3.0, estimated * 2.5))
        deadline = time.monotonic() + effective_timeout
        polls = 0
        last_pos = None
        while time.monotonic() < deadline:
            polls += 1
            pos = self._query_position()
            if pos is not None:
                last_pos = pos
                if self._distance(pos, target) <= POSITION_TOLERANCE_MM:
                    self.x, self.y, self.z = x, y, z
                    log.info("arm_move arrived in %d polls", polls)
                    return True
            time.sleep(ARM_POLL_INTERVAL)

        # We do not hard-fail here because some firmware builds do not report feedback reliably.
        self.x, self.y, self.z = x, y, z
        log.warning(
            "arrival poll timeout for target (%.1f, %.1f, %.1f), polls=%d, last_feedback=%s",
            x,
            y,
            z,
            polls,
            last_pos,
        )
        return True

    def arm_home(self) -> bool:
        return self.move_xyz(HOME_X, HOME_Y, HOME_Z, SPEED_FAST)

    def magnet_on(self) -> bool:
        if self._servo_enabled:
            self._send_http(
                {"T": 11, "id": self._servo_id, "angle": self._servo_close_angle, "t": self._servo_speed},
                timeout=HTTP_FEEDBACK_TIMEOUT,
                accept_timeout=True,
            )
        ok = self._send_serial_line("1")
        time.sleep(GRIPPER_ACTIVATE_TIME)
        if self.serial_conn is None:
            return True
        log.debug("magnet_on=%s", ok)
        return ok

    def magnet_off(self) -> bool:
        if self._servo_enabled:
            self._send_http(
                {"T": 11, "id": self._servo_id, "angle": self._servo_open_angle, "t": self._servo_speed},
                timeout=HTTP_FEEDBACK_TIMEOUT,
                accept_timeout=True,
            )
        ok = self._send_serial_line("0")
        time.sleep(GRIPPER_ACTIVATE_TIME)
        if self.serial_conn is None:
            return True
        log.debug("magnet_off=%s", ok)
        return ok

    def pick_piece(self, x: float, y: float, speed: int) -> bool:
        return (
            self.move_xyz(x, y, HOME_Z, speed)
            and self.move_xyz(x, y, Z_PICK, speed)
            and self.magnet_on()
            and self.move_xyz(x, y, HOME_Z, speed)
        )

    def place_piece(self, x: float, y: float, speed: int) -> bool:
        ok = (
            self.move_xyz(x, y, HOME_Z, speed)
            and self.move_xyz(x, y, Z_PLACE, speed)
            and self.magnet_off()
            and self.move_xyz(x, y, HOME_Z, speed)
        )
        time.sleep(ARM_SETTLE_TIME)
        return ok


def parse_move(move_str: str) -> Optional[ParsedMove]:
    if not move_str:
        log.error("parse_move got empty move string")
        return None
    match = MOVE_REGEX.match(move_str.strip())
    if not match:
        log.error("parse_move failed for move string: %s", move_str)
        return None

    piece = int(match.group(1))
    src = cell_to_rc(match.group(2))
    dst = cell_to_rc(match.group(3))
    promotion = int(match.group(4)) if match.group(4) else None
    return ParsedMove(piece=piece, from_cell=src, to_cell=dst, promotion=promotion)


def _side_piece_ids(play_white: bool):
    return WHITE_PIECE_IDS if play_white else BLACK_PIECE_IDS


def _is_white_piece(piece_id: int) -> bool:
    return piece_id in WHITE_PIECE_IDS


def _is_black_piece(piece_id: int) -> bool:
    return piece_id in BLACK_PIECE_IDS


def _is_board_plausible(board: np.ndarray) -> bool:
    # A plausible board should have at most one white king and one black king.
    white_king_count = int(np.count_nonzero(board == 5))
    black_king_count = int(np.count_nonzero(board == 10))
    return white_king_count <= 1 and black_king_count <= 1


def validate_move_for_side(board: np.ndarray, move: ParsedMove, play_white: bool) -> Tuple[bool, str]:
    fr, fc = move.from_cell
    tr, tc = move.to_cell

    if not (0 <= fr < BOARD_SIZE and 0 <= fc < BOARD_SIZE and 0 <= tr < BOARD_SIZE and 0 <= tc < BOARD_SIZE):
        return False, "cell out of board"

    if (fr, fc) == (tr, tc):
        return False, "source and target are identical"

    own_ids = _side_piece_ids(play_white)
    src = int(board[fr, fc])
    dst = int(board[tr, tc])

    if src == EMPTY:
        return False, "source square empty"
    if src not in own_ids:
        return False, f"source piece {src} is not ours"
    if dst in own_ids:
        return False, f"target occupied by our piece {dst}"

    if move.promotion is not None and move.promotion not in own_ids:
        return False, f"promotion piece {move.promotion} not valid for side"

    return True, "ok"


def _board_change_count(before: np.ndarray, after: np.ndarray) -> int:
    return int(np.count_nonzero(before != after))


def _board_delta_to_expected(observed: np.ndarray, expected: np.ndarray) -> int:
    return int(np.count_nonzero(observed != expected))


def accept_opponent_board_change(before: np.ndarray, after: np.ndarray) -> Tuple[bool, str]:
    if after is None:
        return False, "missing board"
    if not _is_board_plausible(after):
        return False, "implausible board state"

    delta = _board_change_count(before, after)
    if delta == 0:
        return False, "no board change"
    if delta > 8:
        return False, f"too many changed cells: {delta}"

    return True, "ok"


def accept_board_change_for_side(before: np.ndarray, after: np.ndarray, play_white: bool) -> Tuple[bool, str]:
    accept, reason = accept_opponent_board_change(before, after)
    if not accept:
        return False, reason

    own_ids = _side_piece_ids(play_white)

    for fr in range(BOARD_SIZE):
        for fc in range(BOARD_SIZE):
            src_piece = int(before[fr, fc])
            if src_piece not in own_ids:
                continue
            if int(after[fr, fc]) != EMPTY:
                continue

            for tr in range(BOARD_SIZE):
                for tc in range(BOARD_SIZE):
                    if (fr, fc) == (tr, tc):
                        continue
                    dst_after = int(after[tr, tc])
                    if dst_after not in own_ids:
                        continue

                    promotion = dst_after if dst_after != src_piece else None
                    move = ParsedMove(piece=src_piece, from_cell=(fr, fc), to_cell=(tr, tc), promotion=promotion)
                    ok, _ = validate_move_for_side(before, move, play_white)
                    if not ok:
                        continue

                    expected, _ = apply_move(before, move)
                    if np.array_equal(expected, after):
                        return True, "ok"

    side = "white" if play_white else "black"
    return False, f"board change does not match one legal {side} move"


def _recover_camera_stream(perception: PerceptionSystem, stats: RuntimeStats) -> bool:
    stats.recoveries += 1
    stats.camera_recovers += 1
    perception.close()
    return perception.start_background()


def _reconcile_post_move_board(
    perception: PerceptionSystem,
    expected: np.ndarray,
    fallback_board: np.ndarray,
    stats: RuntimeStats,
    verify_retries: int = 2,
) -> np.ndarray:
    if perception.verify_board(expected):
        return expected

    stats.verify_failures += 1
    best = fallback_board
    best_delta = _board_delta_to_expected(best, expected)

    for _ in range(max(1, verify_retries)):
        latest = perception.wait_for_stable_board(timeout=4.0)
        if latest is None or not _is_board_plausible(latest):
            continue
        delta = _board_delta_to_expected(latest, expected)
        if delta < best_delta:
            best = latest
            best_delta = delta
        if delta == 0:
            break

    if best_delta > 3:
        log.warning("board verify mismatch (delta=%d), using expected state", best_delta)
        stats.recoveries += 1
        return expected

    if best_delta > 0:
        stats.recoveries += 1
    log.warning("board verification mismatch resolved with delta=%d", best_delta)
    return best


def apply_move(board: np.ndarray, move: ParsedMove) -> Tuple[np.ndarray, int]:
    updated = board.copy()
    fr, fc = move.from_cell
    tr, tc = move.to_cell
    captured = int(updated[tr, tc])
    updated[fr, fc] = EMPTY
    updated[tr, tc] = move.promotion if move.promotion is not None else move.piece
    return updated, captured


def board_to_text(board: np.ndarray) -> str:
    rows = []
    rows.append("    A   B   C   D   E   F")
    for r in range(BOARD_SIZE - 1, -1, -1):
        chunks = []
        for c in range(BOARD_SIZE):
            pid = int(board[r, c])
            chunks.append(f"{PIECE_NAMES.get(pid, '..'):>2}")
        rows.append(f"{r + 1} | " + " ".join(chunks))
    return "\n".join(rows)


def board_state_digest(board: np.ndarray) -> str:
    white_total = int(np.isin(board, list(WHITE_PIECE_IDS)).sum())
    black_total = int(np.isin(board, list(BLACK_PIECE_IDS)).sum())
    white_kings = int((board == 5).sum())
    black_kings = int((board == 10).sum())
    checksum = int(np.sum(board * (np.arange(board.size, dtype=np.int64).reshape(board.shape) + 1)))
    return (
        f"white={white_total} black={black_total} "
        f"white_kings={white_kings} black_kings={black_kings} checksum={checksum}"
    )


def execute_move(
    arm: ArmController,
    board: np.ndarray,
    move: ParsedMove,
    white_grave_idx: int,
    black_grave_idx: int,
    speed: int,
) -> Tuple[np.ndarray, int, int, bool]:
    fr, fc = move.from_cell
    tr, tc = move.to_cell
    if (fr, fc) == (tr, tc):
        log.error("invalid no-op move: %s -> %s", rc_to_label(fr, fc), rc_to_label(tr, tc))
        return board, white_grave_idx, black_grave_idx, False

    source_piece = int(board[fr, fc])
    if source_piece == EMPTY:
        log.error("invalid move source is empty at %s", rc_to_label(fr, fc))
        return board, white_grave_idx, black_grave_idx, False

    effective_piece = move.piece
    if source_piece != move.piece:
        # Trust observed board state over engine label to avoid desync crashes.
        log.warning(
            "engine piece mismatch at %s: move piece=%d board piece=%d; using board piece",
            rc_to_label(fr, fc),
            move.piece,
            source_piece,
        )
        effective_piece = source_piece

    destination_piece = int(board[tr, tc])
    if destination_piece != EMPTY:
        same_side_capture = (
            (_is_white_piece(effective_piece) and _is_white_piece(destination_piece))
            or (_is_black_piece(effective_piece) and _is_black_piece(destination_piece))
        )
        if same_side_capture:
            log.error(
                "invalid self-capture: mover=%d target=%d at %s",
                effective_piece,
                destination_piece,
                rc_to_label(tr, tc),
            )
            return board, white_grave_idx, black_grave_idx, False

    log.info(
        "execute_move piece=%d from=%s to=%s promotion=%s",
        effective_piece,
        rc_to_label(*move.from_cell),
        rc_to_label(*move.to_cell),
        move.promotion,
    )
    effective_move = ParsedMove(
        piece=effective_piece,
        from_cell=move.from_cell,
        to_cell=move.to_cell,
        promotion=move.promotion,
    )
    expected, captured = apply_move(board, effective_move)

    from_x, from_y = rc_to_world(*move.from_cell)
    to_x, to_y = rc_to_world(*move.to_cell)

    if captured != EMPTY:
        log.info("capture detected at %s: piece=%d", rc_to_label(*move.to_cell), captured)
        captured_is_white = captured in WHITE_PIECE_IDS
        if captured in WHITE_PIECE_IDS:
            slot_idx = min(white_grave_idx, len(GRAVEYARD_WHITE) - 1)
            slot = GRAVEYARD_WHITE[slot_idx]
        else:
            slot_idx = min(black_grave_idx, len(GRAVEYARD_BLACK) - 1)
            slot = GRAVEYARD_BLACK[slot_idx]

        if not arm.pick_piece(to_x, to_y, speed):
            log.error("failed while picking captured piece")
            return board, white_grave_idx, black_grave_idx, False
        if not arm.place_piece(slot[0], slot[1], speed):
            log.error("failed while placing captured piece to graveyard")
            return board, white_grave_idx, black_grave_idx, False
        if captured_is_white:
            white_grave_idx += 1
        else:
            black_grave_idx += 1

    if not arm.pick_piece(from_x, from_y, speed):
        log.error("failed while picking moving piece")
        return board, white_grave_idx, black_grave_idx, False

    if not arm.place_piece(to_x, to_y, speed):
        log.error("failed while placing moving piece")
        return board, white_grave_idx, black_grave_idx, False

    return expected, white_grave_idx, black_grave_idx, True


def calibration_wizard(arm: ArmController) -> None:
    checks = [
        ("home", (HOME_X, HOME_Y, HOME_Z)),
        ("A1", (*rc_to_world(0, 0), HOME_Z)),
        ("F1", (*rc_to_world(0, 5), HOME_Z)),
        ("A6", (*rc_to_world(5, 0), HOME_Z)),
        ("F6", (*rc_to_world(5, 5), HOME_Z)),
    ]
    for label, (x, y, z) in checks:
        input(f"Press Enter to move to {label} ({x:.1f}, {y:.1f}, {z:.1f})...")
        ok = arm.move_xyz(x, y, z, SPEED_NORMAL)
        print(f"{label}: {'OK' if ok else 'FAILED'}")


def choose_move_speed(move_count: int, think_time_s: float) -> int:
    # Openings prioritize placement precision; speed up when game gets tactical.
    if move_count < 3:
        return SPEED_SLOW
    if move_count >= 18 or think_time_s > 8.0:
        return SPEED_FAST
    return SPEED_NORMAL


def request_engine_move(board: np.ndarray, play_white: bool) -> Tuple[Optional[str], bool]:
    try:
        best = game.get_best_move(board, play_white)
        if best:
            parsed = parse_move(best)
            if parsed is not None:
                ok, reason = validate_move_for_side(board, parsed, play_white)
                if ok:
                    return best, False
                log.warning("get_best_move rejected: %s | %s", best, reason)
        log.warning("get_best_move returned invalid move: %s", best)
    except Exception as exc:
        log.exception("get_best_move raised: %s", exc)

    turn_value = 1 if play_white else 0
    try:
        fallback = game.get_move(board, turn_value)
        if fallback:
            parsed = parse_move(fallback)
            if parsed is not None:
                ok, reason = validate_move_for_side(board, parsed, play_white)
                if ok:
                    log.warning("using get_move fallback")
                    return fallback, True
                log.warning("get_move fallback rejected: %s | %s", fallback, reason)
    except Exception as exc:
        log.exception("get_move fallback raised: %s", exc)

    return None, False


def request_human_move(board: np.ndarray, play_white: bool) -> Tuple[Optional[str], bool]:
    side = "WHITE" if play_white else "BLACK"
    log.info("human move request: side=%s", side)
    log.info("human board snapshot:\n%s", board_to_text(board))
    print(f"\n[HUMAN TURN {side}] format: piece:A1->A2 or piece:A5->A6=promotion")

    while True:
        try:
            move_str = input(f"Enter {side} move: ").strip()
            log.debug("human raw input: %s", move_str)
            if not move_str:
                print("Empty input. Try again.")
                continue

            parsed = parse_move(move_str)
            if parsed is None:
                print("Invalid move format.")
                continue

            ok, reason = validate_move_for_side(board, parsed, play_white)
            if not ok:
                print(f"Illegal move: {reason}")
                continue

            log.info("human move accepted: %s", move_str)
            return move_str, False
        except EOFError:
            log.error("stdin closed while waiting for human move")
            return None, False
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            log.exception("unexpected human input error: %s", exc)
            return None, False


def _prompt_our_turn_mode(default_mode: str) -> str:
    default_mode = default_mode if default_mode in {"engine", "human", "hand"} else "engine"
    print("\n[TEAM MODE] Choose our move source: [e]ngine, [t]yped-human, [h]and")
    print(f"Press Enter for default: {default_mode}")
    while True:
        raw = input("Our turn source [e/t/h]: ").strip().lower()
        if raw == "":
            return default_mode
        if raw in {"e", "engine"}:
            return "engine"
        if raw in {"t", "typed", "human"}:
            return "human"
        if raw in {"h", "hand"}:
            return "hand"
        print("Invalid choice. Use e, t, or h.")


def _is_game_over(board: np.ndarray) -> bool:
    has_wk = bool(np.any(board == 5))
    has_bk = bool(np.any(board == 10))
    if not has_wk:
        log.info("game over: white king missing")
        return True
    if not has_bk:
        log.info("game over: black king missing")
        return True
    return False


def _wait_and_accept_opponent_board(
    perception: PerceptionSystem,
    board: np.ndarray,
    manual_mode: bool,
    stats: RuntimeStats,
    promotions: PromotionTracker,
) -> Optional[np.ndarray]:
    if manual_mode:
        input("Press Enter after opponent move is complete...")
        changed = perception.wait_for_stable_board(timeout=30.0)
    else:
        changed = perception.wait_for_board_change(board)

    if changed is None:
        log.warning("timeout waiting for opponent move, attempting camera recovery")
        if _recover_camera_stream(perception, stats):
            changed = perception.wait_for_board_change(board, timeout=45.0)
    if changed is None:
        return None

    changed = promotions.apply(changed)
    accept, reason = accept_opponent_board_change(board, changed)
    if accept:
        return changed

    stats.opponent_rejects += 1
    log.warning("rejected opponent board change: %s", reason)
    changed_retry = perception.wait_for_stable_board(timeout=4.0)
    if changed_retry is None:
        return None

    changed_retry = promotions.apply(changed_retry)
    accept, reason = accept_opponent_board_change(board, changed_retry)
    if not accept:
        stats.opponent_rejects += 1
        log.warning("rejected retry opponent board change: %s", reason)
        return None
    return changed_retry


def _wait_and_accept_board_for_side(
    perception: PerceptionSystem,
    board: np.ndarray,
    manual_mode: bool,
    stats: RuntimeStats,
    promotions: PromotionTracker,
    play_white_side: bool,
    actor_label: str,
) -> Optional[np.ndarray]:
    if manual_mode:
        input(f"Press Enter after {actor_label} move is complete...")
        changed = perception.wait_for_stable_board(timeout=30.0)
    else:
        changed = perception.wait_for_board_change(board)

    if changed is None:
        log.warning("timeout waiting for %s move, attempting camera recovery", actor_label)
        if _recover_camera_stream(perception, stats):
            changed = perception.wait_for_board_change(board, timeout=45.0)
    if changed is None:
        return None

    changed = promotions.apply(changed)
    accept, reason = accept_board_change_for_side(board, changed, play_white=play_white_side)
    if accept:
        return changed

    stats.opponent_rejects += 1
    log.warning("rejected %s board change: %s", actor_label, reason)
    changed_retry = perception.wait_for_stable_board(timeout=4.0)
    if changed_retry is None:
        return None

    changed_retry = promotions.apply(changed_retry)
    accept, reason = accept_board_change_for_side(board, changed_retry, play_white=play_white_side)
    if not accept:
        stats.opponent_rejects += 1
        log.warning("rejected retry %s board change: %s", actor_label, reason)
        return None
    return changed_retry


def run_game(
    play_white: bool,
    manual_mode: bool = False,
    human_mode: bool = False,
    hand_mode: bool = False,
    team_mode: bool = False,
    team_default_mode: str = "engine",
    max_moves: int = 0,
    resume: bool = True,
) -> int:
    log.info(
        "run_game start: play_white=%s hand_mode=%s arm_ip=%s serial_port=%s",
        play_white,
        hand_mode,
        ARM_IP,
        SERIAL_PORT,
    )
    arm: Optional[ArmController] = ArmController()
    perception = PerceptionSystem()
    stats = RuntimeStats()
    promotions = PromotionTracker()
    clear_checkpoint_on_exit = False

    loaded_checkpoint = _checkpoint_load() if resume else None

    def save_checkpoint(board_state: np.ndarray, white_idx: int, black_idx: int, next_action: str) -> None:
        payload = {
            "play_white": play_white,
            "manual_mode": manual_mode,
            "human_mode": human_mode,
            "hand_mode": hand_mode,
            "team_mode": team_mode,
            "team_default_mode": team_default_mode,
            "max_moves": max_moves,
            "next_action": next_action,
            "board": board_state.tolist(),
            "white_grave_idx": int(white_idx),
            "black_grave_idx": int(black_idx),
            "stats": _stats_to_dict(stats),
            "promotions": [
                {"r": int(r), "c": int(c), "piece": int(pid)}
                for (r, c), pid in promotions.promoted.items()
            ],
        }
        _checkpoint_write(payload)

    if arm is not None:
        if not arm.check_connection():
            log.error("arm preflight failed")
            arm.close()
            return 2

    if not perception.start_background():
        log.error("camera connection failed")
        if arm is not None:
            arm.close()
        return 2

    board = None
    white_grave_idx = 0
    black_grave_idx = 0
    next_action = "our_turn" if play_white else "wait_opponent"

    if loaded_checkpoint is not None:
        if bool(loaded_checkpoint.get("play_white", play_white)) != play_white:
            log.warning("checkpoint side mismatch; ignoring checkpoint")
            loaded_checkpoint = None
        elif bool(loaded_checkpoint.get("manual_mode", manual_mode)) != manual_mode:
            log.warning("checkpoint mode mismatch; ignoring checkpoint")
            loaded_checkpoint = None
        elif bool(loaded_checkpoint.get("human_mode", human_mode)) != human_mode:
            log.warning("checkpoint human-mode mismatch; ignoring checkpoint")
            loaded_checkpoint = None
        elif bool(loaded_checkpoint.get("hand_mode", hand_mode)) != hand_mode:
            log.warning("checkpoint hand-mode mismatch; ignoring checkpoint")
            loaded_checkpoint = None
        elif bool(loaded_checkpoint.get("team_mode", team_mode)) != team_mode:
            log.warning("checkpoint team-mode mismatch; ignoring checkpoint")
            loaded_checkpoint = None
        elif str(loaded_checkpoint.get("team_default_mode", team_default_mode)) != team_default_mode:
            log.warning("checkpoint team-default mismatch; ignoring checkpoint")
            loaded_checkpoint = None

    if loaded_checkpoint is not None:
        try:
            cp_board = np.array(loaded_checkpoint.get("board", []), dtype=int)
            if cp_board.shape != (BOARD_SIZE, BOARD_SIZE):
                raise ValueError(f"bad checkpoint board shape: {cp_board.shape}")
            if not _is_board_plausible(cp_board):
                raise ValueError("checkpoint board not plausible")

            white_grave_idx = int(loaded_checkpoint.get("white_grave_idx", 0))
            black_grave_idx = int(loaded_checkpoint.get("black_grave_idx", 0))
            next_action = str(loaded_checkpoint.get("next_action", next_action))
            stats = _stats_from_dict(loaded_checkpoint.get("stats", {}))
            promotions.promoted.clear()
            for row in loaded_checkpoint.get("promotions", []):
                promotions.mark(int(row["r"]), int(row["c"]), int(row["piece"]))

            live_board = perception.wait_for_stable_board(timeout=8.0)
            if live_board is not None and _is_board_plausible(live_board):
                live_board = promotions.apply(live_board)
                delta = _board_change_count(cp_board, live_board)
                if delta <= 6:
                    board = live_board
                    log.info("resumed from checkpoint with live board sync (delta=%d)", delta)
                else:
                    board = cp_board
                    log.warning("live board differs too much from checkpoint (delta=%d), using checkpoint board", delta)
            else:
                board = cp_board
                log.info("resumed from checkpoint without live sync")
        except Exception as exc:
            log.warning("failed to restore checkpoint, starting fresh: %s", exc)
            board = None

    if board is None:
        board = perception.wait_for_stable_board()
        if board is None:
            log.error("could not read stable initial board")
            perception.close()
            if arm is not None:
                arm.close()
            return 2

    if not _is_board_plausible(board):
        log.warning("initial board not plausible, retrying stabilization")
        retry_board = perception.wait_for_stable_board(timeout=10.0)
        if retry_board is not None and _is_board_plausible(retry_board):
            board = retry_board
        else:
            log.error("initial board remained implausible")
            perception.close()
            if arm is not None:
                arm.close()
            return 2

    board = promotions.apply(board)

    log.info("active board:\n%s", board_to_text(board))
    log.info("initial board digest: %s", board_state_digest(board))
    log.info("checkpoint file: %s", CHECKPOINT_PATH)
    save_checkpoint(board, white_grave_idx, black_grave_idx, next_action)

    try:
        while True:
            if max_moves > 0 and stats.move_count >= max_moves:
                log.info("reached max_moves=%d, stopping cleanly", max_moves)
                clear_checkpoint_on_exit = True
                break

            loop_t0 = time.monotonic()
            log.info(
                "loop state: move_count=%d next_action=%s white_grave_idx=%d black_grave_idx=%d digest=%s",
                stats.move_count,
                next_action,
                white_grave_idx,
                black_grave_idx,
                board_state_digest(board),
            )
            if next_action == "wait_opponent":
                if hand_mode:
                    changed = _wait_and_accept_board_for_side(
                        perception,
                        board,
                        manual_mode,
                        stats,
                        promotions,
                        play_white_side=(not play_white),
                        actor_label="opponent",
                    )
                else:
                    changed = _wait_and_accept_opponent_board(perception, board, manual_mode, stats, promotions)
                if changed is None:
                    log.error("timeout waiting for opponent move")
                    break

                board = changed
                if _is_game_over(board):
                    clear_checkpoint_on_exit = True
                    break
                next_action = "our_turn"
                save_checkpoint(board, white_grave_idx, black_grave_idx, next_action)
                continue

            my_board = board

            active_our_mode = "human" if human_mode else "engine"
            if hand_mode and not team_mode:
                # Auto-alternate our side without prompts: bot first, then hand, then bot...
                active_our_mode = "engine" if (stats.move_count % 2 == 0) else "hand"
            if team_mode:
                active_our_mode = _prompt_our_turn_mode(team_default_mode)
                log.info("team mode selected our source: %s", active_our_mode)
            elif hand_mode and not team_mode:
                log.info("hand mode auto source: %s (our_move_index=%d)", active_our_mode, stats.move_count)

            if active_our_mode == "hand":
                changed = _wait_and_accept_board_for_side(
                    perception,
                    board,
                    manual_mode,
                    stats,
                    promotions,
                    play_white_side=play_white,
                    actor_label="our",
                )
                if changed is None:
                    log.error("timeout waiting for our hand move")
                    break

                board = changed
                stats.move_count += 1
                loop_elapsed = time.monotonic() - loop_t0
                stats.total_loop_time_s += loop_elapsed
                log.info("hand move accepted; loop iteration time: %.3fs", loop_elapsed)
                if _is_game_over(board):
                    clear_checkpoint_on_exit = True
                    break
                next_action = "wait_opponent"
                save_checkpoint(board, white_grave_idx, black_grave_idx, next_action)
                continue

            think_t0 = time.monotonic()
            if active_our_mode == "human":
                move_str, used_fallback = request_human_move(my_board, play_white)
            else:
                move_str, used_fallback = request_engine_move(my_board, play_white)
            think_elapsed = time.monotonic() - think_t0
            if active_our_mode == "human":
                log.info("human input time: %.3fs", think_elapsed)
            else:
                log.info("engine think time: %.3fs", think_elapsed)
            stats.total_think_time_s += think_elapsed
            if not move_str:
                stats.invalid_moves += 1
                if active_our_mode == "human":
                    log.error("human failed to provide a valid move")
                else:
                    log.error("engine failed to provide a valid move")
                break
            if used_fallback:
                stats.fallback_moves += 1

            parsed = parse_move(move_str)
            if parsed is None:
                stats.invalid_moves += 1
                log.error("engine returned invalid move: %s", move_str)
                break

            valid, reason = validate_move_for_side(board, parsed, play_white)
            if not valid:
                stats.invalid_moves += 1
                log.error("engine move rejected: %s | %s", move_str, reason)
                break

            speed = choose_move_speed(stats.move_count, think_elapsed)
            log.info("engine move %d: %s | speed=%d", stats.move_count + 1, move_str, speed)
            if arm is None:
                log.error("arm unavailable while robot execution path is active")
                break
            expected, white_grave_idx, black_grave_idx, ok = execute_move(
                arm,
                board,
                parsed,
                white_grave_idx,
                black_grave_idx,
                speed=speed,
            )
            if not ok:
                stats.execution_retries += 1
                log.warning("arm execution failed, homing and retrying once: %s", move_str)
                arm.arm_home()
                expected, white_grave_idx, black_grave_idx, ok = execute_move(
                    arm,
                    board,
                    parsed,
                    white_grave_idx,
                    black_grave_idx,
                    speed=SPEED_FAST,
                )
                if not ok:
                    log.error("arm execution failed for move after retry: %s", move_str)
                    break

            board = _reconcile_post_move_board(
                perception=perception,
                expected=expected,
                fallback_board=board,
                stats=stats,
                verify_retries=2,
            )

            src_rc = parsed.from_cell
            dst_rc = parsed.to_cell
            carried_promo = promotions.promoted.get(src_rc)
            promotions.clear_at(src_rc[0], src_rc[1])
            if parsed.promotion is not None:
                promotions.mark(dst_rc[0], dst_rc[1], parsed.promotion)
            elif carried_promo is not None:
                # Move existing promoted-piece identity with the piece.
                promotions.mark(dst_rc[0], dst_rc[1], int(carried_promo))
            board = promotions.apply(board)

            log.info("board after move:\n%s", board_to_text(board))
            stats.move_count += 1
            loop_elapsed = time.monotonic() - loop_t0
            stats.total_loop_time_s += loop_elapsed
            log.info("loop iteration time: %.3fs", loop_elapsed)
            if _is_game_over(board):
                clear_checkpoint_on_exit = True
                break
            next_action = "wait_opponent"
            save_checkpoint(board, white_grave_idx, black_grave_idx, next_action)

    except KeyboardInterrupt:
        log.info("stopped by user")
    except Exception as exc:
        log.exception("unhandled exception in run_game: %s", exc)
        return 3
    finally:
        if arm is not None:
            arm.arm_home()
        perception.close()
        if arm is not None:
            arm.close()
        if clear_checkpoint_on_exit:
            _checkpoint_clear()
            log.info("checkpoint cleared after clean stop")
        log.info(
            "run_game cleanup done | commands=%d serial_fallbacks=%d feedback_failures=%d moves=%d recoveries=%d invalid_moves=%d retries=%d fallback_moves=%d opponent_rejects=%d camera_recovers=%d verify_failures=%d avg_think=%.3fs avg_loop=%.3fs",
            0 if arm is None else arm.command_counter,
            0 if arm is None else arm.serial_fallback_count,
            0 if arm is None else arm.feedback_failures,
            stats.move_count,
            stats.recoveries,
            stats.invalid_moves,
            stats.execution_retries,
            stats.fallback_moves,
            stats.opponent_rejects,
            stats.camera_recovers,
            stats.verify_failures,
            stats.total_think_time_s / max(1, stats.move_count),
            stats.total_loop_time_s / max(1, stats.move_count),
        )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="RoboGambit Team L2 controller")
    side = parser.add_mutually_exclusive_group()
    side.add_argument("--white", action="store_true", help="play as white")
    side.add_argument("--black", action="store_true", help="play as black")
    parser.add_argument("--calibrate", action="store_true", help="run calibration wizard")
    parser.add_argument("--manual", action="store_true", help="manual opponent move trigger")
    parser.add_argument("--human", action="store_true", help="use human move input for our side")
    parser.add_argument("--hand", action="store_true", help="our side auto-alternates bot then hand each turn (no prompts)")
    parser.add_argument("--team", action="store_true", help="team mode: choose our source each turn (engine/human/hand)")
    parser.add_argument("--team-default", choices=["engine", "human", "hand"], default="engine", help="default source used on Enter in --team mode")
    parser.add_argument("--max-moves", type=int, default=0, help="stop after N of our own moves (0=unlimited)")
    parser.add_argument("--fresh", action="store_true", help="ignore and clear saved checkpoint before starting")
    parser.add_argument("--no-resume", action="store_true", help="run without loading checkpoint")
    parser.add_argument("--debug", action="store_true", help="enable verbose debug logging")
    parser.add_argument("--log-file", type=str, default="", help="custom path for log file")
    args = parser.parse_args()

    setup_logging(debug=args.debug, log_file=args.log_file or None)
    log.info(
        "main args: white=%s black=%s calibrate=%s manual=%s human=%s hand=%s team=%s team_default=%s max_moves=%d fresh=%s no_resume=%s debug=%s",
        args.white,
        args.black,
        args.calibrate,
        args.manual,
        args.human,
        args.hand,
        args.team,
        args.team_default,
        args.max_moves,
        args.fresh,
        args.no_resume,
        args.debug,
    )

    if args.fresh:
        _checkpoint_clear()
        log.info("cleared checkpoint due to --fresh")

    if not args.white and not args.black:
        args.white = True

    if args.human and args.hand:
        log.error("--human and --hand cannot be used together")
        return 2
    if args.hand and args.team:
        log.error("--hand and --team cannot be used together; use --team --team-default hand")
        return 2

    team_default = args.team_default
    if args.team and args.human and args.team_default == "engine":
        team_default = "human"
        log.info("team mode default set to human because --human was provided")

    arm: Optional[ArmController] = None
    try:
        if args.calibrate:
            arm = ArmController()
            calibration_wizard(arm)
            return 0
    finally:
        if args.calibrate and arm is not None:
            arm.close()

    def _sigint_handler(sig_num, frame):
        log.warning("SIGINT received, stopping now")
        raise KeyboardInterrupt()

    previous_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _sigint_handler)

    resume_enabled = (not args.no_resume) and (not args.fresh)
    try:
        return run_game(
        play_white=args.white,
        manual_mode=args.manual,
        human_mode=args.human,
        hand_mode=args.hand,
        team_mode=args.team,
        team_default_mode=team_default,
        max_moves=max(0, args.max_moves),
        resume=resume_enabled,
        )
    finally:
        signal.signal(signal.SIGINT, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
