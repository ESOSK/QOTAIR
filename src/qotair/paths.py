"""Stable paths for QOTAIR and an optional external research snapshot.

The reviewer delivery keeps a single snapshot under
``reviewer_bundle/archive/reproduction_snapshot``. Installed wheels do not
contain that artifact, so a wheel install must set ``QOTAIR_ARCHIVE_ROOT``
(or pass the equivalent option) to the extracted snapshot directory.
"""
from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def discover_archive_root(package_root: Path | None = None) -> Path:
    """Return the first existing snapshot directory, preferring the reviewer bundle."""

    root = package_root or PACKAGE_ROOT
    candidates = (
        root / "reviewer_bundle" / "archive" / "reproduction_snapshot",
        root / "archive" / "reproduction_snapshot",
        root / "research_archive" / "reproduction_snapshot",
        root / "reproduction_snapshot",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def discover_bundle_manifest(package_root: Path | None = None) -> Path:
    """Return the outer software/archive binding manifest if present."""

    root = package_root or PACKAGE_ROOT
    configured = os.environ.get("QOTAIR_BUNDLE_MANIFEST")
    if configured:
        return Path(configured).expanduser().resolve()
    for candidate in (
        root / "reviewer_bundle" / "reproduction_manifest.json",
        root / "reproduction_manifest.json",
    ):
        if candidate.is_file():
            return candidate
    return root / "reviewer_bundle" / "reproduction_manifest.json"


DEFAULT_ARCHIVE_ROOT = discover_archive_root()


def _configured_path(name: str, fallback: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        return fallback.resolve()
    return Path(value).expanduser().resolve()


ARCHIVE_ROOT = _configured_path("QOTAIR_ARCHIVE_ROOT", DEFAULT_ARCHIVE_ROOT)


def archive_manifest_path(archive_root: Path | None = None) -> Path:
    """Return the archive manifest, allowing a bundle to override its path."""

    configured = os.environ.get("QOTAIR_ARCHIVE_MANIFEST")
    if configured:
        return Path(configured).expanduser().resolve()
    root = archive_root or ARCHIVE_ROOT
    return root / "manifest.json"


def bundle_manifest_path() -> Path:
    """Return the outer software/archive binding manifest if configured."""

    return discover_bundle_manifest()


def required_archive_paths(archive_root: Path | None = None) -> tuple[Path, ...]:
    """Return the minimum files and directories needed for replay."""

    root = archive_root or ARCHIVE_ROOT
    return (
        root / "reproduce.py",
        root / "derived_stats",
        root / "registry",
        root / "results",
        root / "LICENSE_STATUS.txt",
    )
