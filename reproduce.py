#!/usr/bin/env python3
"""One-command analysis replay and optional offline reviewer isolation.

The normal source-checkout path reuses the active Python environment.  A
reviewer bundle can pass ``--isolated`` (or set ``QOTAIR_REPRO_ISOLATED=1``)
to create a temporary standard-library ``venv`` and install only the exact
local wheels listed by ``requirements-lock.txt``.  No network fallback is
implemented: a missing wheel or lock entry is reported as an explicit
unsupported environment.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"


def _venv_python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _failure(status: str, code: str, message: str, **extra: object) -> int:
    payload: dict[str, object] = {"status": status, "reason_code": code, "message": message, **extra}
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 6 if code == "E_ARCHIVE_INTEGRITY" else 3


def _archive_root(args: argparse.Namespace) -> Path:
    if args.archive_root:
        return Path(args.archive_root).expanduser().resolve()
    for candidate in (
        PROJECT_ROOT / "reviewer_bundle" / "archive" / "reproduction_snapshot",
        PROJECT_ROOT / "archive" / "reproduction_snapshot",
        PROJECT_ROOT / "research_archive" / "reproduction_snapshot",
        PROJECT_ROOT / "reproduction_snapshot",
    ):
        if candidate.is_dir():
            return candidate.resolve()
    return PROJECT_ROOT / "reviewer_bundle" / "archive" / "reproduction_snapshot"


def _base_environment(archive_root: Path, manifest: str | None) -> dict[str, str]:
    environment = os.environ.copy()
    environment["QOTAIR_ARCHIVE_ROOT"] = str(archive_root)
    if manifest:
        environment["QOTAIR_ARCHIVE_MANIFEST"] = str(Path(manifest).expanduser().resolve())
    return environment


def _run_direct(args: argparse.Namespace) -> int:
    archive_root = _archive_root(args)
    os.environ.update(_base_environment(archive_root, args.manifest))
    sys.path.insert(0, str(SRC_ROOT))
    from qotair.runtime import replay

    return int(replay())


def _find_qotair_wheel() -> Path | None:
    candidates: list[Path] = []
    for directory in (PROJECT_ROOT / "wheels", PROJECT_ROOT / "wheelhouse", PROJECT_ROOT):
        if directory.is_dir():
            candidates.extend(directory.glob("qotair-*.whl"))
    return sorted(candidates)[-1] if candidates else None


def _run_isolated(args: argparse.Namespace) -> int:
    outer_manifest = PROJECT_ROOT / "reproduction_manifest.json"
    if not outer_manifest.is_file():
        return _failure(
            "UNSUPPORTED",
            "E_MANIFEST_MISSING",
            "reproduction_manifest.json is required for isolated replay.",
        )
    wheel = _find_qotair_wheel()
    lock = PROJECT_ROOT / "requirements-lock.txt"
    wheelhouse = PROJECT_ROOT / "wheelhouse"
    if wheel is None:
        return _failure("UNSUPPORTED", "E_OFFLINE_WHEEL_MISSING", "No local qotair wheel was bundled.")
    if not lock.is_file():
        return _failure(
            "UNSUPPORTED",
            "E_OFFLINE_LOCK_MISSING",
            "requirements-lock.txt is required for isolated offline replay.",
        )
    if not wheelhouse.is_dir():
        return _failure(
            "UNSUPPORTED",
            "E_OFFLINE_WHEELHOUSE_MISSING",
            "wheelhouse/ is required; the runner will not contact PyPI.",
        )

    archive_root = _archive_root(args)
    keep = args.keep_venv
    temp_context = tempfile.TemporaryDirectory(prefix="qotair-repro-") if not keep else None
    venv_root = Path(temp_context.name) if temp_context else PROJECT_ROOT / ".qotair-repro-venv"
    try:
        venv_args = [sys.executable, "-m", "venv", "--without-pip"]
        if venv_root.exists():
            venv_args.append("--clear")
        venv_args.append(str(venv_root))
        created = subprocess.run(
            venv_args,
            capture_output=True,
            text=True,
            check=False,
        )
        if created.returncode:
            return _failure("FAIL", "E_VENV_CREATE", created.stderr[-4000:] or "venv creation failed")
        child_python = _venv_python(venv_root)
        bootstrap = subprocess.run(
            [str(child_python), "-m", "ensurepip", "--upgrade"],
            capture_output=True,
            text=True,
            env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
            check=False,
        )
        if bootstrap.returncode:
            return _failure("FAIL", "E_ENSUREPIP", bootstrap.stderr[-4000:] or "ensurepip failed")
        install = subprocess.run(
            [
                str(child_python),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--disable-pip-version-check",
                "--require-hashes",
                "--find-links",
                str(wheelhouse),
                "--find-links",
                str(wheel.parent),
                "-r",
                str(lock),
            ],
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PIP_NO_INDEX": "1",
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            },
            check=False,
        )
        if install.returncode:
            return _failure(
                "UNSUPPORTED",
                "E_OFFLINE_DEPENDENCY_MISSING",
                "The locked offline dependencies could not be installed.",
                stderr=install.stderr[-8000:],
            )
        binding = subprocess.run(
            [
                str(child_python),
                "-m",
                "qotair",
                "verify-bundle",
                str(PROJECT_ROOT),
                "--json",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if binding.returncode:
            return _failure(
                "FAIL",
                "E_ARCHIVE_INTEGRITY",
                "The reviewer bundle binding manifest failed.",
                binding=binding.stdout[-12000:] or binding.stderr[-12000:],
            )
        child_env = _base_environment(archive_root, args.manifest)
        child_env["QOTAIR_REPRO_ISOLATED"] = "0"
        child_env["QOTAIR_NETWORK_POLICY"] = "deny"
        result = subprocess.run(
            [str(child_python), "-m", "qotair", "reproduce", "--json"],
            cwd=PROJECT_ROOT,
            env=child_env,
            check=False,
        )
        return result.returncode
    finally:
        if not keep and temp_context is not None:
            temp_context.cleanup()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the QOTAIR analysis replay.")
    parser.add_argument("--archive-root", help="extracted reproduction_snapshot directory")
    parser.add_argument("--manifest", help="archive manifest.json path")
    parser.add_argument(
        "--isolated",
        action="store_true",
        default=os.environ.get("QOTAIR_REPRO_ISOLATED") == "1",
        help="create a temporary offline venv from bundled wheels",
    )
    parser.add_argument("--keep-venv", action="store_true", help="keep .qotair-repro-venv for debugging")
    args = parser.parse_args(argv)
    return _run_isolated(args) if args.isolated else _run_direct(args)


if __name__ == "__main__":
    raise SystemExit(main())
