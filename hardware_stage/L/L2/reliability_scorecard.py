"""
Head-to-head reliability scorecard for Team L vs Team L2 controllers.

This script runs available simulator/emulator test paths repeatedly and computes
pass rates, timings, and a simple reliability index for each controller.

Usage examples:
  python reliability_scorecard.py
  python reliability_scorecard.py --rounds 3 --stress-rounds 3
  python reliability_scorecard.py --rounds 2 --skip-l-full
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class RunResult:
    ok: bool
    duration_s: float
    command: List[str]
    label: str
    stdout_tail: str


@dataclass
class SuiteSummary:
    label: str
    total: int = 0
    passed: int = 0
    total_time_s: float = 0.0

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total) * 100.0 if self.total else 0.0

    @property
    def avg_time_s(self) -> float:
        return self.total_time_s / self.total if self.total else 0.0


def _run_once(
    command: List[str],
    cwd: Path,
    env_overrides: Dict[str, str],
    timeout_s: int,
    label: str,
) -> RunResult:
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.update(env_overrides)

    t0 = time.monotonic()
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
    )
    elapsed = time.monotonic() - t0

    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    tail = "\n".join(combined.strip().splitlines()[-8:]) if combined.strip() else "(no output)"

    return RunResult(
        ok=(proc.returncode == 0),
        duration_s=elapsed,
        command=command,
        label=label,
        stdout_tail=tail,
    )


def _print_result(r: RunResult) -> None:
    status = "PASS" if r.ok else "FAIL"
    print(f"  [{status}] {r.label} | {r.duration_s:.2f}s")
    if not r.ok:
        print("    Output tail:")
        for ln in r.stdout_tail.splitlines():
            print(f"      {ln}")


def _add_summary(summary: SuiteSummary, result: RunResult) -> None:
    summary.total += 1
    summary.total_time_s += result.duration_s
    if result.ok:
        summary.passed += 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Head-to-head reliability scorecard for L vs L2")
    parser.add_argument("--rounds", type=int, default=2, help="baseline rounds per suite")
    parser.add_argument("--stress-rounds", type=int, default=2, help="stress rounds for L2 suite")
    parser.add_argument("--timeout", type=int, default=180, help="timeout seconds per test invocation")
    parser.add_argument("--skip-l-full", action="store_true", help="skip Team L short full simulation")
    args = parser.parse_args()

    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    py = Path(sys.executable)

    l_unit_cmd = [str(py), "hardware_stage/L/test_simulator.py", "--unit"]
    l_full_cmd = [str(py), "hardware_stage/L/test_simulator.py", "--moves", "2"]
    l2_full_cmd = [str(py), "hardware_stage/L/L2/full_system_arm_test.py"]

    l_summary = SuiteSummary(label="Team L")
    l2_summary = SuiteSummary(label="Team L2")

    print("=" * 74)
    print("RELIABILITY SCORECARD: Team L vs Team L2")
    print(f"repo: {repo_root}")
    print(f"python: {py}")
    print("=" * 74)

    print("\nTeam L baseline unit suite")
    for i in range(max(1, args.rounds)):
        r = _run_once(
            command=l_unit_cmd,
            cwd=repo_root,
            env_overrides={},
            timeout_s=args.timeout,
            label=f"L unit round {i + 1}/{max(1, args.rounds)}",
        )
        _print_result(r)
        _add_summary(l_summary, r)

    if not args.skip_l_full:
        print("\nTeam L short full simulation")
        r = _run_once(
            command=l_full_cmd,
            cwd=repo_root,
            env_overrides={},
            timeout_s=max(args.timeout, 240),
            label="L full short (--moves 2)",
        )
        _print_result(r)
        _add_summary(l_summary, r)

    print("\nTeam L2 baseline full-system emulator")
    for i in range(max(1, args.rounds)):
        r = _run_once(
            command=l2_full_cmd,
            cwd=repo_root,
            env_overrides={},
            timeout_s=args.timeout,
            label=f"L2 full round {i + 1}/{max(1, args.rounds)}",
        )
        _print_result(r)
        _add_summary(l2_summary, r)

    print("\nTeam L2 stress profile (tight HTTP timeouts + fewer retries)")
    for i in range(max(1, args.stress_rounds)):
        r = _run_once(
            command=l2_full_cmd,
            cwd=repo_root,
            env_overrides={
                "ROBO_MAX_RETRIES": "2",
                "ROBO_HTTP_MOVE_TIMEOUT": "0.6",
                "ROBO_HTTP_FEEDBACK_TIMEOUT": "0.8",
                "ROBO_RETRY_DELAY": "0.25",
            },
            timeout_s=args.timeout,
            label=f"L2 stress round {i + 1}/{max(1, args.stress_rounds)}",
        )
        _print_result(r)
        _add_summary(l2_summary, r)

    l_index = (l_summary.pass_rate * 0.85) + (max(0.0, 60.0 - l_summary.avg_time_s) * 0.15)
    l2_index = (l2_summary.pass_rate * 0.85) + (max(0.0, 60.0 - l2_summary.avg_time_s) * 0.15)

    print("\n" + "-" * 74)
    print("SUMMARY")
    print("-" * 74)
    print(
        f"{l_summary.label:10} | passed {l_summary.passed:>2}/{l_summary.total:<2} | "
        f"pass rate {l_summary.pass_rate:6.2f}% | avg time {l_summary.avg_time_s:6.2f}s | score {l_index:6.2f}"
    )
    print(
        f"{l2_summary.label:10} | passed {l2_summary.passed:>2}/{l2_summary.total:<2} | "
        f"pass rate {l2_summary.pass_rate:6.2f}% | avg time {l2_summary.avg_time_s:6.2f}s | score {l2_index:6.2f}"
    )

    winner = "Team L2" if l2_index >= l_index else "Team L"
    print("-" * 74)
    print(f"WINNER: {winner}")
    print("-" * 74)

    return 0 if l2_summary.passed == l2_summary.total else 1


if __name__ == "__main__":
    raise SystemExit(main())
