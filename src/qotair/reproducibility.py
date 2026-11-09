"""Portable, auditable hashes and offline reproduction helpers.

The reproducibility contract deliberately distinguishes two notions of
identity:

* ``binary-sha256-v1`` hashes the exact bytes delivered to the reviewer.  It
  is the default and is the right choice for proving artifact identity.
* ``canonical-text-lf-utf8-v1`` is an opt-in semantic comparison for text
  files.  It normalises only CRLF/CR to LF and leaves the rest of the bytes
  unchanged after UTF-8 decoding.  A manifest must name this mode explicitly;
  it must never be silently substituted for the binary hash.

The module has no third-party dependencies so that a reviewer-side wrapper can
use it before QOTAIR itself is installed.
"""
from __future__ import annotations

import hashlib
import json
import fnmatch
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ARCHIVE_MANIFEST_SCHEMA = "qotair-archive-manifest-v1"
BUNDLE_MANIFEST_SCHEMA = "qotair-reproduction-bundle-v1"
BINARY_HASH_POLICY = "binary-sha256-v1"
CANONICAL_TEXT_HASH_POLICY = "canonical-text-lf-utf8-v1"
DEFAULT_DYNAMIC_PATHS = ("manifest.json", "manifest.sha256")
VOLATILE_PATHS: tuple[str, ...] = (
    "**/__pycache__/**",
    "__pycache__/**",
    "*.pyc",
    ".pytest_cache/**",
)


class ReproducibilityError(RuntimeError):
    """A user-facing reproducibility failure with a stable reason code."""

    def __init__(self, message: str, code: str = "E_REPRODUCIBILITY"):
        super().__init__(message)
        self.code = code


def _normalise_relative(path: Path | str) -> str:
    """Return a safe, portable manifest path or raise on traversal."""

    value = Path(path)
    if value.is_absolute() or ".." in value.parts:
        raise ReproducibilityError(f"Manifest path is not relative: {path}", "E_MANIFEST_PATH")
    return value.as_posix()


def _matches(path: str, patterns: Iterable[str]) -> bool:
    """Match a POSIX path against exact paths or ``/**`` prefixes."""

    for pattern in patterns:
        if "*" in pattern and fnmatch.fnmatchcase(path, pattern):
            return True
        if pattern.endswith("/**"):
            candidate = pattern[:-3].rstrip("/")
            if path == candidate or path.startswith(candidate + "/"):
                return True
        elif path == pattern.rstrip("/"):
            return True
    return False


def _canonical_text_bytes(path: Path) -> bytes:
    """Return explicitly canonical UTF-8/LF bytes for a text hash mode."""

    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReproducibilityError(
            f"Cannot apply text hash policy to non-UTF-8 file: {path}",
            "E_MANIFEST_ENCODING",
        ) from exc
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def sha256_file(path: Path | str, *, policy: str = BINARY_HASH_POLICY) -> str:
    """Hash a file using a named, reproducible byte policy."""

    target = Path(path)
    if policy == BINARY_HASH_POLICY:
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    if policy == CANONICAL_TEXT_HASH_POLICY:
        return hashlib.sha256(_canonical_text_bytes(target)).hexdigest()
    raise ReproducibilityError(f"Unknown hash policy: {policy}", "E_MANIFEST_POLICY")


def tree_digest(
    root: Path | str,
    *,
    policy: str = BINARY_HASH_POLICY,
    exclude: Iterable[str] = (),
) -> tuple[str, int, list[dict[str, Any]]]:
    """Return a deterministic digest, file count, and per-file records.

    Paths are sorted by POSIX spelling and are included in the digest together
    with their file hash.  This prevents a rename from preserving the same
    tree digest and avoids platform-specific path separators.
    """

    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise ReproducibilityError(f"Archive root is not a directory: {base}", "E_ARCHIVE_INCOMPLETE")
    excluded = tuple(dict.fromkeys((*exclude, *VOLATILE_PATHS)))
    records: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for candidate in sorted(base.rglob("*"), key=lambda item: item.relative_to(base).as_posix()):
        if not candidate.is_file():
            continue
        if candidate.is_symlink():
            raise ReproducibilityError(
                f"Symlinks are not accepted in a reproducibility archive: {candidate}",
                "E_MANIFEST_SYMLINK",
            )
        relative = _normalise_relative(candidate.relative_to(base))
        if _matches(relative, excluded):
            continue
        file_hash = sha256_file(candidate, policy=policy)
        record = {"path": relative, "size": candidate.stat().st_size, "sha256": file_hash}
        records.append(record)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), len(records), records


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_json(path: Path | str, value: Mapping[str, Any]) -> str:
    """Write canonical UTF-8 JSON with LF endings and return its SHA-256."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = _json_bytes(value)
    target.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def write_sidecar(path: Path | str, digest: str) -> Path:
    """Write a conventional detached SHA-256 sidecar next to ``path``."""

    target = Path(path)
    sidecar = target.with_name(target.stem + ".sha256")
    sidecar.write_bytes(f"{digest}  {target.name}\n".encode("utf-8"))
    return sidecar


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReproducibilityError(f"Manifest not found: {path}", "E_MANIFEST_MISSING") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReproducibilityError(f"Manifest cannot be read: {path}", "E_MANIFEST_MALFORMED") from exc
    if not isinstance(value, dict):
        raise ReproducibilityError(f"Manifest must contain a JSON object: {path}", "E_MANIFEST_MALFORMED")
    return value


def _verify_sidecar(path: Path) -> tuple[bool, str | None]:
    sidecar = path.with_name(path.stem + ".sha256")
    if not sidecar.is_file():
        return False, "sidecar_missing"
    try:
        token = sidecar.read_text(encoding="utf-8").strip().split()[0]
    except (OSError, UnicodeError, IndexError):
        return False, "sidecar_malformed"
    actual = sha256_file(path)
    return token == actual, None if token == actual else "sidecar_mismatch"


def build_archive_manifest(
    archive_root: Path | str,
    *,
    output: Path | str | None = None,
    exclude: Sequence[str] | None = None,
    scope: str = "analysis_replay_not_full_transport",
) -> dict[str, Any]:
    """Build a per-file archive manifest without normalising text bytes."""

    root = Path(archive_root).expanduser().resolve()
    if exclude is None:
        existing = root / "manifest.json"
        policy = json.loads(existing.read_text(encoding="utf-8")).get("hash_policy", {}) if existing.is_file() else {}
        exclude = policy.get("excluded_paths", DEFAULT_DYNAMIC_PATHS)
    effective_exclude = tuple(dict.fromkeys((*exclude, *VOLATILE_PATHS)))
    digest, count, records = tree_digest(root, policy=BINARY_HASH_POLICY, exclude=effective_exclude)
    manifest: dict[str, Any] = {
        "schema_version": ARCHIVE_MANIFEST_SCHEMA,
        "scope": scope,
        "root_name": root.name,
        "hash_policy": {
            "name": BINARY_HASH_POLICY,
            "file_bytes": "exact bytes as distributed",
            "text_normalization": "none",
            "path_separator": "/",
            "excluded_paths": list(effective_exclude),
        },
        "tree": {"sha256": digest, "file_count": count},
        "files": records,
    }
    if output is not None:
        target = Path(output)
        manifest_hash = write_json(target, manifest)
        write_sidecar(target, manifest_hash)
    return manifest


def verify_archive_manifest(
    archive_root: Path | str,
    *,
    manifest: Path | str | None = None,
    strict_extra: bool = False,
) -> dict[str, Any]:
    """Verify a binary archive manifest and report every discrepancy."""

    root = Path(archive_root).expanduser().resolve()
    manifest_path = Path(manifest).expanduser().resolve() if manifest else root / "manifest.json"
    base_report: dict[str, Any] = {
        "status": "UNSUPPORTED",
        "reason_code": "E_MANIFEST_MISSING",
        "archive_root": str(root),
        "manifest": str(manifest_path),
        "hash_policy": None,
        "files_checked": 0,
        "missing": [],
        "mismatched": [],
        "extra": [],
        "malformed": [],
    }
    if not manifest_path.is_file():
        return base_report
    try:
        payload = _load_json(manifest_path)
    except ReproducibilityError as exc:
        base_report.update({"status": "FAIL", "reason_code": exc.code, "malformed": [str(exc)]})
        return base_report
    if payload.get("schema_version") != ARCHIVE_MANIFEST_SCHEMA:
        base_report.update({"status": "FAIL", "reason_code": "E_MANIFEST_SCHEMA", "malformed": ["schema_version"]})
        return base_report
    policy = str(payload.get("hash_policy", {}).get("name", ""))
    if policy != BINARY_HASH_POLICY:
        base_report.update({"status": "FAIL", "reason_code": "E_MANIFEST_POLICY", "hash_policy": policy})
        return base_report
    records = payload.get("files")
    if not isinstance(records, list):
        base_report.update({"status": "FAIL", "reason_code": "E_MANIFEST_MALFORMED", "malformed": ["files"]})
        return base_report
    base_report["hash_policy"] = policy
    expected: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str) or not isinstance(record.get("sha256"), str):
            base_report["malformed"].append(str(record))
            continue
        try:
            relative = _normalise_relative(str(record["path"]))
        except ReproducibilityError:
            base_report["malformed"].append(str(record.get("path")))
            continue
        expected.add(relative)
        target = root / Path(relative)
        if not target.is_file():
            base_report["missing"].append(relative)
            continue
        actual = sha256_file(target, policy=policy)
        base_report["files_checked"] += 1
        if actual != record["sha256"]:
            base_report["mismatched"].append(relative)
    excluded = tuple(str(item) for item in payload.get("hash_policy", {}).get("excluded_paths", []))
    if strict_extra:
        for candidate in root.rglob("*"):
            if not candidate.is_file() or candidate.is_symlink():
                continue
            relative = candidate.relative_to(root).as_posix()
            if relative in expected or _matches(relative, excluded) or relative in {"manifest.json", "manifest.sha256"}:
                continue
            base_report["extra"].append(relative)
    sidecar_ok, sidecar_error = _verify_sidecar(manifest_path)
    base_report["manifest_sidecar"] = {"verified": sidecar_ok, "error": sidecar_error}
    bad = base_report["missing"] or base_report["mismatched"] or base_report["malformed"] or (not sidecar_ok)
    base_report.update(
        {
            "status": "FAIL" if bad else "PASS",
            "reason_code": "E_ARCHIVE_INTEGRITY" if bad else None,
            "tree": payload.get("tree", {}),
        }
    )
    return base_report


def _git_binding(root: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        clean = not subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return commit or None, clean
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _source_tree_digest(source_root: Path | None) -> str | None:
    if source_root is None or not source_root.is_dir():
        return None
    return tree_digest(source_root, policy=BINARY_HASH_POLICY)[0]


def build_bundle_manifest(
    bundle_root: Path | str,
    *,
    archive_root: Path | str,
    archive_manifest: Path | str,
    software_root: Path | str | None = None,
    software_version: str = "0.0.0",
    api_version: str | None = None,
    wheel: Path | str | None = None,
    requirements_lock: Path | str | None = None,
    scope: str = "analysis_replay_not_full_transport",
) -> dict[str, Any]:
    """Create the outer binding manifest used by a reviewer bundle."""

    bundle = Path(bundle_root).expanduser().resolve()
    archive = Path(archive_root).expanduser().resolve()
    archive_manifest_path = Path(archive_manifest).expanduser().resolve()
    source = Path(software_root).expanduser().resolve() if software_root else None
    commit, clean = _git_binding(source or bundle)
    archive_payload = _load_json(archive_manifest_path)
    manifest: dict[str, Any] = {
        "schema_version": BUNDLE_MANIFEST_SCHEMA,
        "scope": scope,
        "network_policy": "deny-by-default",
        "entrypoint": "reproduce.py",
        "entrypoint_sha256": sha256_file(bundle / "reproduce.py") if (bundle / "reproduce.py").is_file() else None,
        "hash_policy": {
            "artifact_identity": BINARY_HASH_POLICY,
            "text_normalization": "none for identity hashes",
            "manifest_encoding": "UTF-8 with LF",
        },
        "software": {
            "name": "qotair",
            "version": software_version,
            "api_version": api_version,
            "git_commit": commit,
            "working_tree_clean": clean,
            "source_root": str(source.relative_to(bundle)) if source and source.is_relative_to(bundle) else None,
            "source_tree_sha256": _source_tree_digest(source),
            "wheel": None,
        },
        "archive": {
            "root": str(archive.relative_to(bundle)) if archive.is_relative_to(bundle) else str(archive),
            "manifest": str(archive_manifest_path.relative_to(bundle)) if archive_manifest_path.is_relative_to(bundle) else str(archive_manifest_path),
            "tree_sha256": archive_payload.get("tree", {}).get("sha256"),
            "file_count": archive_payload.get("tree", {}).get("file_count"),
        },
        "environment": {
            "python_requires": ">=3.10",
            "python_runtime": f"{sys.version_info.major}.{sys.version_info.minor}",
            "platform": platform.platform(),
            "machine": platform.machine(),
            "requirements_lock": None,
        },
    }
    if wheel is not None:
        wheel_path = Path(wheel).expanduser().resolve()
        manifest["software"]["wheel"] = {
            "path": str(wheel_path.relative_to(bundle)) if wheel_path.is_relative_to(bundle) else str(wheel_path),
            "sha256": sha256_file(wheel_path),
            "size": wheel_path.stat().st_size,
        }
    if requirements_lock is not None:
        lock_path = Path(requirements_lock).expanduser().resolve()
        manifest["environment"]["requirements_lock"] = {
            "path": str(lock_path.relative_to(bundle)) if lock_path.is_relative_to(bundle) else str(lock_path),
            "sha256": sha256_file(lock_path),
            "size": lock_path.stat().st_size,
        }
    return manifest


def verify_bundle_manifest(bundle_root: Path | str, *, manifest: Path | str | None = None) -> dict[str, Any]:
    """Verify outer software/archive/environment bindings without installing anything."""

    bundle = Path(bundle_root).expanduser().resolve()
    manifest_path = Path(manifest).expanduser().resolve() if manifest else bundle / "reproduction_manifest.json"
    report: dict[str, Any] = {
        "status": "UNSUPPORTED",
        "reason_code": "E_MANIFEST_MISSING",
        "bundle_root": str(bundle),
        "manifest": str(manifest_path),
        "checks": {},
        "mismatched": [],
        "missing": [],
    }
    if not manifest_path.is_file():
        return report
    try:
        payload = _load_json(manifest_path)
    except ReproducibilityError as exc:
        report.update({"status": "FAIL", "reason_code": exc.code, "mismatched": [str(exc)]})
        return report
    if payload.get("schema_version") != BUNDLE_MANIFEST_SCHEMA:
        report.update({"status": "FAIL", "reason_code": "E_MANIFEST_SCHEMA", "mismatched": ["schema_version"]})
        return report
    checks: dict[str, Any] = report["checks"]
    archive_data = payload.get("archive", {})
    archive_root = bundle / str(archive_data.get("root", ""))
    archive_manifest = bundle / str(archive_data.get("manifest", ""))
    archive_report = verify_archive_manifest(archive_root, manifest=archive_manifest)
    checks["archive"] = archive_report
    expected_tree = archive_data.get("tree_sha256")
    if expected_tree and archive_report.get("tree", {}).get("sha256") != expected_tree:
        report["mismatched"].append("archive.tree_sha256")
    software = payload.get("software", {})
    entrypoint_hash = payload.get("entrypoint_sha256")
    entrypoint_path = bundle / str(payload.get("entrypoint", "reproduce.py"))
    if entrypoint_hash:
        if not entrypoint_path.is_file():
            report["missing"].append(str(payload.get("entrypoint", "reproduce.py")))
        elif sha256_file(entrypoint_path) != entrypoint_hash:
            report["mismatched"].append("entrypoint_sha256")
    source_root_value = software.get("source_root")
    if source_root_value and software.get("source_tree_sha256"):
        source_root = bundle / str(source_root_value)
        if not source_root.is_dir():
            report["missing"].append(str(source_root_value))
        elif _source_tree_digest(source_root) != software.get("source_tree_sha256"):
            report["mismatched"].append("software.source_tree_sha256")
    wheel = software.get("wheel")
    if isinstance(wheel, Mapping):
        wheel_path = bundle / str(wheel.get("path", ""))
        if not wheel_path.is_file():
            report["missing"].append(str(wheel.get("path")))
        elif sha256_file(wheel_path) != wheel.get("sha256"):
            report["mismatched"].append("software.wheel.sha256")
    lock = payload.get("environment", {}).get("requirements_lock")
    if isinstance(lock, Mapping):
        lock_path = bundle / str(lock.get("path", ""))
        if not lock_path.is_file():
            report["missing"].append(str(lock.get("path")))
        elif sha256_file(lock_path) != lock.get("sha256"):
            report["mismatched"].append("environment.requirements_lock.sha256")
    expected_python = payload.get("environment", {}).get("python_runtime")
    current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    checks["python_runtime"] = {"expected": expected_python, "current": current_python}
    if expected_python and str(expected_python) != current_python:
        report["mismatched"].append("environment.python_runtime")
    sidecar_ok, sidecar_error = _verify_sidecar(manifest_path)
    checks["manifest_sidecar"] = {"verified": sidecar_ok, "error": sidecar_error}
    bad_archive = archive_report.get("status") == "FAIL"
    bad = bool(report["missing"] or report["mismatched"] or bad_archive or not sidecar_ok)
    report.update({"status": "FAIL" if bad else "PASS", "reason_code": "E_ARCHIVE_INTEGRITY" if bad else None})
    return report


def environment_lock_snapshot() -> dict[str, Any]:
    """Return non-invasive runtime facts for a replay report."""

    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "network_policy": "deny-by-default",
    }
