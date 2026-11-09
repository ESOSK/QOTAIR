"""Read-only inspection and integrity services used by the canonical CLI."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from ..domain.models import ProjectConfig, QuerySpec
from ..storage.projects import (
    ProjectLayout,
    list_runs,
    load_project,
    read_mapping,
    sha256_file,
    sha256_mapping,
)
from .planning import build_plan
from .scientific import compile_query, query_operands


def _query_payload(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = read_mapping(resolved)
    if "query" in payload and isinstance(payload["query"], Mapping):
        payload = dict(payload["query"])
    return resolved, payload


def inspect_query(path: str | Path, *, available_operands: Iterable[str] | None = None) -> dict[str, Any]:
    """Validate one query and expose its frozen compiler result without writing files."""

    resolved, payload = _query_payload(path)
    query = QuerySpec.from_mapping(payload)
    operands = query_operands(payload)
    candidates = tuple(str(item) for item in (available_operands or operands))
    compiled = compile_query(payload, available_operands=candidates)
    return {
        "status": "PASS",
        "reason_code": None,
        "schema_version": "qotair-query-inspection-v1",
        "path": str(resolved),
        "query_id": query.query_id,
        "name": query.name,
        "expression_class": query.expression_class,
        "query_sha256": sha256_mapping(payload),
        "requested_operands": list(operands),
        "available_operands": list(candidates),
        "compiled": compiled.as_dict(),
    }


def inspect_project(path: str | Path) -> dict[str, Any]:
    """Validate a project and compile its plan without publishing a plan file."""

    project = load_project(path)
    plan = build_plan(project)
    ledger = project.ledger_path()
    queries = project.queries()
    ledger_supported = project.mode not in {"ledger_replay", "analysis"} or bool(ledger and ledger.is_file())
    return {
        "status": "PASS" if ledger_supported else "UNSUPPORTED",
        "reason_code": None if ledger_supported else "E_LEDGER_MISSING",
        "schema_version": "qotair-project-inspection-v1",
        "project": str(project.path),
        "project_id": project.project_id,
        "mode": project.mode,
        "model": project.model,
        "ledger": {
            "path": str(ledger) if ledger else None,
            "exists": bool(ledger and ledger.is_file()),
        },
        "queries": [
            {
                "query_id": query.query_id,
                "name": query.name,
                "expression_class": query.expression_class,
                "requested_operands": list(query_operands(query.payload)),
            }
            for query in queries
        ],
        "required_operands": plan["required_operands"],
        "excluded_available_operands": plan["excluded_available_operands"],
        "plan_sha256": plan["plan_sha256"],
        "config_sha256": plan["config_sha256"],
        "support": plan["support"],
    }


def _resolve_run(project: ProjectConfig, run_id: str) -> tuple[str, Path] | tuple[None, None]:
    if run_id != "latest":
        path = ProjectLayout(project).run / run_id
        return (run_id, path) if path.is_dir() else (None, None)
    records = list_runs(project.path)
    if not records:
        return None, None
    latest = records[0]
    raw_path = latest.get("path")
    resolved = Path(str(raw_path)).expanduser().resolve() if raw_path else Path("__missing_run_path__")
    if not raw_path or not resolved.is_dir():
        resolved = ProjectLayout(project).run / str(latest.get("run_id"))
    return str(latest.get("run_id")), resolved


def verify_run(project_path: str | Path, run_id: str = "latest") -> dict[str, Any]:
    """Verify the recorded run manifest and every file in its hash manifest."""

    project = load_project(project_path)
    resolved_id, run_dir = _resolve_run(project, run_id)
    if resolved_id is None or run_dir is None:
        return {
            "status": "FAIL",
            "reason_code": "E_RUN_NOT_FOUND",
            "project": str(project.path),
            "run_id": run_id,
            "message": "No matching run directory exists.",
        }
    manifest_path = run_dir / "run_manifest.json"
    integrity_path = run_dir / "integrity" / "MANIFEST.sha256"
    if not manifest_path.is_file() or not integrity_path.is_file():
        return {
            "status": "FAIL",
            "reason_code": "E_INTEGRITY",
            "project": str(project.path),
            "run_id": resolved_id,
            "run_path": str(run_dir),
            "message": "The run manifest or integrity manifest is missing.",
        }
    try:
        manifest = read_mapping(manifest_path)
    except Exception as exc:
        return {
            "status": "FAIL",
            "reason_code": "E_INTEGRITY",
            "project": str(project.path),
            "run_id": resolved_id,
            "run_path": str(run_dir),
            "message": f"Cannot parse run_manifest.json: {exc}",
        }

    expected: dict[str, str] = {}
    malformed: list[str] = []
    for line in integrity_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2:
            malformed.append(line)
            continue
        digest, relative = parts
        relative_path = Path(relative)
        if len(digest) != 64 or relative_path.is_absolute() or ".." in relative_path.parts:
            malformed.append(line)
            continue
        expected[relative_path.as_posix()] = digest

    missing: list[str] = []
    mismatched: list[str] = []
    for relative, digest in expected.items():
        path = run_dir / relative
        if not path.is_file():
            missing.append(relative)
        elif sha256_file(path) != digest:
            mismatched.append(relative)
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file() and path != integrity_path
    }
    extra = sorted(actual - set(expected))
    ok = not (malformed or missing or mismatched or extra)
    return {
        "status": "PASS" if ok else "FAIL",
        "reason_code": None if ok else "E_INTEGRITY",
        "project": str(project.path),
        "run_id": resolved_id,
        "run_path": str(run_dir),
        "manifest_status": manifest.get("status"),
        "files_checked": len(expected),
        "missing": missing,
        "mismatched": mismatched,
        "extra": extra,
        "malformed": malformed,
    }
