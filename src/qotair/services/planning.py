"""Project validation and deterministic collection-plan compilation."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ..domain.models import ProjectConfig, QuerySpec
from ..storage.projects import ProjectLayout, load_project, sha256_mapping, write_mapping
from .scientific import compile_query, query_operands


def _available_operands(project: ProjectConfig) -> tuple[str, ...] | None:
    path = project.ledger_path()
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
    if not header:
        return None
    first = header[0].strip().lower()
    if first in {"batch_id", "active_batch", "batch", "active_index"}:
        header = header[1:]
    return tuple(item.strip() for item in header if item.strip())


def _query_plan(query: QuerySpec, available: tuple[str, ...] | None) -> dict[str, Any]:
    operands = query_operands(query.payload)
    candidates = available or operands
    compiled = compile_query(query.payload, available_operands=candidates)
    payload = compiled.as_dict()
    payload["name"] = query.name
    payload["domain_constraints"] = list(query.payload.get("domain_constraints", []))
    payload["requested_operands"] = list(operands)
    return payload


def build_plan(project: ProjectConfig) -> dict[str, Any]:
    available = _available_operands(project)
    queries = [_query_plan(query, available) for query in project.queries()]
    required = []
    for item in queries:
        for operand in item["operands"]:
            if operand not in required:
                required.append(operand)
    body: dict[str, Any] = {
        "schema_version": "qotair-collection-plan-v1",
        "project_id": project.project_id,
        "mode": project.mode,
        "model": project.model,
        "ledger": {
            "path": str(project.ledger_path()) if project.ledger_path() else None,
            "available_operands": list(available or ()),
        },
        "queries": queries,
        "required_operands": required,
        "excluded_available_operands": [
            operand for operand in (available or ()) if operand not in required
        ],
        "baselines": list(project.baselines),
        "support": {
            "plan_preview": "supported",
            "csv_ledger_replay": "supported" if project.ledger_path() else "requires_ledger",
            "openmc_transport": "explicit_model_adapter_required",
            "statepoint_reduction": "no_evidence",
            "runtime_or_memory_gain": "no_evidence",
        },
    }
    body["config_sha256"] = sha256_mapping(project.as_dict())
    body["plan_sha256"] = sha256_mapping(body)
    return body


def write_plan(project_path: str | Path) -> Path:
    project = load_project(project_path)
    plan = build_plan(project)
    return write_mapping(ProjectLayout(project).plan_path(), plan)
