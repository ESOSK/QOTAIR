"""Safe project storage with no-overwrite run commits."""
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..domain.models import ProjectConfig, QuerySpec


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_mapping(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def read_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        value = json.loads(text)
    else:
        try:
            import yaml
        except ImportError:
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError("PyYAML is required for YAML project files") from exc
        else:
            value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return dict(value)


def write_mapping(path: Path, value: Mapping[str, Any], *, overwrite: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    if path.suffix.lower() == ".json":
        text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    else:
        try:
            import yaml
        except ImportError:
            text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        else:
            text = yaml.safe_dump(value, sort_keys=False, allow_unicode=False)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return path


def load_project(path: str | Path) -> ProjectConfig:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = read_mapping(resolved)
    if payload.get("schema_version") not in (None, "qotair-project-v1"):
        raise ValueError(f"Unsupported project schema: {payload.get('schema_version')}")
    payload.setdefault("schema_version", "qotair-project-v1")
    return ProjectConfig(resolved, payload)


def save_query(project_path: str | Path, query: Mapping[str, Any]) -> tuple[Path, Path]:
    """Validate and persist one external query, then attach it to the project.

    The query is written atomically below ``query/`` and the project is updated
    to reference it by a relative path. Existing inline or external queries
    with the same identifier are replaced instead of duplicated.
    """

    validated = QuerySpec.from_mapping(query).as_dict()
    project = load_project(project_path)
    query_id = str(validated["query_id"])
    if not re.fullmatch(r"[A-Za-z0-9._-]+", query_id):
        raise ValueError("query_id may contain only letters, digits, dot, underscore, or hyphen")
    query_dir = project.path.parent / "query"
    query_path = query_dir / f"{query_id}.yaml"
    write_mapping(query_path, validated)

    payload = project.as_dict()
    raw_queries = payload.get("queries", [])
    if isinstance(raw_queries, Mapping):
        raw_queries = [raw_queries]
    if not isinstance(raw_queries, list):
        raw_queries = []
    references: list[Any] = []
    replaced = False
    for item in raw_queries:
        existing_id = item.get("query_id") if isinstance(item, Mapping) else Path(str(item)).stem
        if str(existing_id) == query_id:
            references.append(query_path.relative_to(project.path.parent).as_posix())
            replaced = True
        else:
            references.append(item)
    if not replaced:
        references.append(query_path.relative_to(project.path.parent).as_posix())
    payload["queries"] = references
    write_mapping(project.path, payload)
    return project.path, query_path


DEFAULT_PROJECT: dict[str, Any] = {
    "schema_version": "qotair-project-v1",
    "project_id": "pin-replay-demo",
    "mode": "ledger_replay",
    "model": {
        "kind": "builtin",
        "name": "pin",
        "support": "ledger_replay_only",
    },
    "ledger": {
        "path": "data/batch_operand_ledger.csv",
        "format": "csv",
        "batch_column": "",
    },
    "queries": [
        {
            "schema_version": "p1-query-v1",
            "query_id": "absorption_over_flux",
            "name": "Absorption over flux",
            "expression_class": "ratio",
            "output_unit": "1/cm",
            "operand_units": {"absorption": "1/cm", "flux": "1/cm2"},
            "domain_constraints": ["flux denominator must be stable"],
            "expression": {
                "op": "divide",
                "args": [
                    {"op": "score", "name": "absorption"},
                    {"op": "score", "name": "flux"},
                ],
            },
        }
    ],
    "baselines": ["collect_all", "native_merge", "manual_joint", "compiled_plan"],
    "run": {
        "confidence_level": 0.95,
        "seed": 20260822,
        "batches": 64,
        "inactive": 4,
        "particles_per_batch": 20000,
        "omp_threads": 1,
    },
    "report": {"formats": ["json", "csv", "html"]},
}


def create_project(path: str | Path, *, template: Mapping[str, Any] | None = None) -> Path:
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(target)
    write_mapping(target, dict(template or DEFAULT_PROJECT), overwrite=False)
    for name in ("query", "plan", "run", "data", "result", "integrity"):
        (target.parent / name).mkdir(parents=True, exist_ok=True)
    return target


class ProjectLayout:
    """Paths for one project; all run artifacts stay below its directory."""

    def __init__(self, project: ProjectConfig):
        self.project = project
        self.root = project.path.parent
        self.query = self.root / "query"
        self.plan = self.root / "plan"
        self.run = self.root / "run"
        self.data = self.root / "data"
        self.result = self.root / "result"
        self.integrity = self.root / "integrity"

    def ensure(self) -> None:
        for directory in (self.query, self.plan, self.run, self.data, self.result, self.integrity):
            directory.mkdir(parents=True, exist_ok=True)

    def plan_path(self) -> Path:
        self.ensure()
        return self.plan / "collection_plan.json"

    def new_run(self) -> tuple[str, Path, Path]:
        self.ensure()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{stamp}-{uuid.uuid4().hex[:8]}"
        staging = self.run / f".staging-{run_id}"
        final = self.run / run_id
        staging.mkdir(parents=True, exist_ok=False)
        for name in ("data", "result", "integrity"):
            (staging / name).mkdir()
        return run_id, staging, final

    def commit_run(self, staging: Path, final: Path) -> Path:
        if final.exists():
            raise FileExistsError(final)
        os.replace(staging, final)
        return final

    def record_history(self, summary: Mapping[str, Any]) -> Path:
        self.ensure()
        path = self.run / "index.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(summary), ensure_ascii=False, sort_keys=True) + "\n")
        return path

    def write_integrity_manifest(self, run_dir: Path) -> Path:
        manifest = run_dir / "integrity" / "MANIFEST.sha256"
        lines = []
        for path in sorted(run_dir.rglob("*")):
            if not path.is_file() or path == manifest:
                continue
            lines.append(f"{sha256_file(path)}  {path.relative_to(run_dir).as_posix()}")
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return manifest


def list_runs(project_path: str | Path) -> list[dict[str, Any]]:
    project = load_project(project_path)
    root = ProjectLayout(project).run
    if not root.exists():
        return []
    records = []
    for directory in sorted(item for item in root.iterdir() if item.is_dir() and not item.name.startswith(".")):
        manifest = directory / "run_manifest.json"
        if manifest.exists():
            try:
                records.append(json.loads(manifest.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                records.append({"run_id": directory.name, "status": "CORRUPT_MANIFEST", "path": str(directory)})
    return sorted(records, key=lambda item: str(item.get("run_id", "")), reverse=True)
