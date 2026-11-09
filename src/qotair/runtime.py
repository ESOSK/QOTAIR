"""Compatibility commands and the preserved analysis replay boundary."""
from __future__ import annotations

import os
import json
import subprocess
import sys
from pathlib import Path

from .paths import ARCHIVE_ROOT, archive_manifest_path
from .reproducibility import (
    environment_lock_snapshot,
    verify_archive_manifest,
)
from .services.environment import environment_report


def doctor() -> dict[str, object]:
    """Inspect the local package without starting OpenMC transport."""

    report = environment_report()
    report["cwd"] = str(Path.cwd())
    report["archive_manifest"] = verify_archive_manifest(
        ARCHIVE_ROOT,
        manifest=archive_manifest_path(ARCHIVE_ROOT),
    )
    report["runtime"] = environment_lock_snapshot()
    return report


def _archive_environment(archive_root: Path = ARCHIVE_ROOT) -> dict[str, str]:
    environment = os.environ.copy()
    environment["P1_ROOT"] = str(archive_root)
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(archive_root) + (os.pathsep + existing if existing else "")
    return environment


def _archive_preflight() -> dict[str, object]:
    """Validate required paths and, when present, the immutable archive manifest."""

    state = doctor()
    required = state["required_paths"]
    if not state["archive_ready"]:
        missing = [name for name, present in required.items() if not present]
        return {
            "status": "UNSUPPORTED",
            "kind": "analysis_replay_not_full_transport",
            "reason_code": "E_ARCHIVE_INCOMPLETE",
            "message": "QOTAIR archive is incomplete.",
            "missing": missing,
            "archive_manifest": state.get("archive_manifest"),
        }
    manifest = state.get("archive_manifest", {})
    if isinstance(manifest, dict) and manifest.get("status") == "FAIL":
        return {
            "status": "FAIL",
            "kind": "analysis_replay_not_full_transport",
            "reason_code": "E_ARCHIVE_INTEGRITY",
            "message": "The preserved archive failed its SHA-256 manifest check.",
            "archive_manifest": manifest,
        }
    return {"status": "PASS", "archive_manifest": manifest}


def replay() -> int:
    """Run the preserved analysis replay as an isolated child process."""

    preflight = _archive_preflight()
    if preflight["status"] != "PASS":
        print(json.dumps(preflight, indent=2, ensure_ascii=False, default=str), file=sys.stderr)
        return 6 if preflight.get("reason_code") == "E_ARCHIVE_INTEGRITY" else 3
    return subprocess.run(
        [sys.executable, str(ARCHIVE_ROOT / "reproduce.py")],
        cwd=ARCHIVE_ROOT,
        env=_archive_environment(),
        check=False,
    ).returncode


def replay_report() -> dict[str, object]:
    """Run the preserved replay and return a machine-readable summary."""

    preflight = _archive_preflight()
    if preflight["status"] != "PASS":
        return preflight
    completed = subprocess.run(
        [sys.executable, str(ARCHIVE_ROOT / "reproduce.py")],
        cwd=ARCHIVE_ROOT,
        env=_archive_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    parsed: dict[str, object]
    try:
        value = json.loads(completed.stdout)
        parsed = dict(value) if isinstance(value, dict) else {}
    except (json.JSONDecodeError, TypeError):
        parsed = {}
    parsed.update({
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "kind": "analysis_replay_not_full_transport",
        "returncode": completed.returncode,
        "report": parsed.get("report"),
        "archive_manifest": preflight.get("archive_manifest"),
    })
    if completed.stderr:
        parsed["stderr"] = completed.stderr[-20000:]
    return parsed


def run_tests() -> int:
    """Run the preserved freeze-snapshot tests in their original import context."""

    preflight = _archive_preflight()
    if preflight["status"] != "PASS":
        print(json.dumps(preflight, indent=2, ensure_ascii=False, default=str), file=sys.stderr)
        return 6 if preflight.get("reason_code") == "E_ARCHIVE_INTEGRITY" else 3
    return subprocess.run(
        [sys.executable, "-m", "pytest", "derived_stats/tests"],
        cwd=ARCHIVE_ROOT,
        env=_archive_environment(),
        check=False,
    ).returncode


def test_report() -> dict[str, object]:
    """Run the preserved tests and return output suitable for ``--json``."""

    preflight = _archive_preflight()
    if preflight["status"] != "PASS":
        return preflight
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "derived_stats/tests"],
        cwd=ARCHIVE_ROOT,
        env=_archive_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "returncode": completed.returncode,
        "suite": "preserved freeze-snapshot tests",
        "stdout": completed.stdout[-20000:],
        "stderr": completed.stderr[-20000:],
    }
