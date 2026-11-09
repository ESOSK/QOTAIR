"""Project execution, ledger replay, and atomic result publication."""
from __future__ import annotations

import csv
import html
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from ..domain.models import ProjectConfig, RunResult
from ..paths import ARCHIVE_ROOT
from ..storage.projects import (
    ProjectLayout,
    list_runs,
    load_project,
    read_mapping,
    sha256_file,
    sha256_mapping,
    write_mapping,
)
from .environment import environment_report
from .planning import build_plan, write_plan
from .scientific import infer_query, json_safe


@dataclass(frozen=True)
class LedgerData:
    values: np.ndarray
    score_names: tuple[str, ...]
    batch_ids: tuple[int, ...]
    source_path: Path


def load_csv_ledger(path: str | Path, *, batch_column: str | None = None) -> LedgerData:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    if len(rows) < 3 or not rows[0]:
        raise ValueError("CSV ledger must have a header and at least two data rows")
    header = [item.strip() for item in rows[0]]
    if len(set(header)) != len(header) or any(not item for item in header):
        raise ValueError("CSV ledger headers must be unique and non-empty")
    selected_batch = batch_column or next(
        (item for item in header if item.lower() in {"batch_id", "active_batch", "batch", "active_index"}),
        None,
    )
    batch_index = header.index(selected_batch) if selected_batch else None
    score_names = tuple(item for index, item in enumerate(header) if index != batch_index)
    values: list[list[float]] = []
    batch_ids: list[int] = []
    for row_index, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise ValueError(f"CSV row {row_index} has {len(row)} fields; expected {len(header)}")
        try:
            if batch_index is None:
                batch_ids.append(row_index - 1)
            else:
                batch_ids.append(int(row[batch_index]))
            values.append([float(value) for index, value in enumerate(row) if index != batch_index])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"CSV row {row_index} contains a non-numeric ledger value") from exc
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] < 2 or not np.isfinite(array).all():
        raise ValueError("CSV ledger values must be a finite two-dimensional matrix")
    if len(set(batch_ids)) != len(batch_ids):
        raise ValueError("CSV batch identifiers must be unique")
    return LedgerData(array, score_names, tuple(batch_ids), source)


def _load_query_documents(project: ProjectConfig) -> tuple[dict[str, Any], ...]:
    return tuple(query.as_dict() for query in project.queries())


def _write_html(path: Path, result: Mapping[str, Any]) -> None:
    payload = html.escape(json.dumps(json_safe(result), ensure_ascii=False, indent=2))
    status = html.escape(str(result.get("status", "UNKNOWN")))
    document = f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><title>QOTAIR report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#202124}}
.status{{padding:.6rem 1rem;border-radius:.4rem;background:#eef2f7;font-weight:700}}
pre{{white-space:pre-wrap;background:#f6f8fa;padding:1rem;border-radius:.4rem}}</style></head>
<body><h1>QOTAIR analysis report</h1><p class=\"status\">Status: {status}</p>
<h2>Machine-readable result</h2><pre>{payload}</pre></body></html>\n"""
    path.write_text(document, encoding="utf-8")


def _write_run_manifest(path: Path, payload: Mapping[str, Any]) -> Path:
    write_mapping(path, payload)
    return path


def _run_archive_replay() -> dict[str, Any]:
    environment = os.environ.copy()
    environment["P1_ROOT"] = str(ARCHIVE_ROOT)
    current = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(ARCHIVE_ROOT) + (os.pathsep + current if current else "")
    completed = subprocess.run(
        [sys.executable, str(ARCHIVE_ROOT / "reproduce.py")],
        cwd=ARCHIVE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = {}
    report = payload.get("report") if isinstance(payload, dict) else None
    return {
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "kind": "analysis_replay_not_full_transport",
        "returncode": completed.returncode,
        "stdout": completed.stdout[-20000:],
        "stderr": completed.stderr[-20000:],
        "report": report,
    }


def _execute_ledger_project(project: ProjectConfig, plan: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    ledger_path = project.ledger_path()
    if ledger_path is None or not ledger_path.is_file():
        return "UNSUPPORTED", {
            "status": "UNSUPPORTED",
            "reason_code": "E_LEDGER_MISSING",
            "message": "This project requires a local CSV BatchLedger before running inference.",
            "queries": [],
        }
    ledger = load_csv_ledger(ledger_path, batch_column=project.ledger.get("batch_column") or None)
    confidence = float(project.run.get("confidence_level", 0.95))
    query_results = []
    for query in _load_query_documents(project):
        inference = infer_query(query, ledger.values, ledger.score_names, confidence_level=confidence)
        query_results.append({
            "query_id": query["query_id"],
            "name": query.get("name", query["query_id"]),
            "expression_class": query.get("expression_class"),
            "output_unit": query.get("output_unit"),
            "inference": json_safe(inference),
        })
    statuses = [str(item["inference"].get("status", "FAIL")) for item in query_results]
    status = "PASS" if statuses and all(item == "PASS" for item in statuses) else "REFUSE" if any(item == "REFUSE" for item in statuses) else "FAIL"
    result = {
        "schema_version": "qotair-inference-result-v1",
        "status": status,
        "project_id": project.project_id,
        "run_id": None,
        "plan_sha256": plan.get("plan_sha256"),
        "ledger": {
            "path": str(ledger.source_path),
            "sha256": sha256_file(ledger.source_path),
            "n_batches": len(ledger.batch_ids),
            "score_names": list(ledger.score_names),
            "batch_ids": list(ledger.batch_ids),
        },
        "queries": query_results,
        "limitations": [
            "This result replays a CSV ledger; it does not claim OpenMC transport, statepoint, runtime, or memory reduction.",
            "Only the frozen inference router and production expression path are used.",
        ],
    }
    return status, result


def run_project(project_path: str | Path, *, progress: Callable[[str], None] | None = None) -> RunResult:
    """Execute one project while optionally reporting coarse workflow phases."""

    def report(message: str) -> None:
        if progress is not None:
            progress(message)

    report("Loading project")
    project = load_project(project_path)
    layout = ProjectLayout(project)
    report("Compiling collection plan")
    plan = build_plan(project)
    report("Writing collection plan")
    plan_path = write_plan(project.path)
    report("Creating run workspace")
    run_id, staging, final = layout.new_run()
    started = datetime.now(timezone.utc).isoformat()
    try:
        mode = project.mode
        if mode == "archive_replay":
            report("Replaying preserved analysis snapshot")
            status, result = "PASS", _run_archive_replay()
        elif mode in {"ledger_replay", "analysis"}:
            report("Loading CSV ledger and running inference")
            status, result = _execute_ledger_project(project, plan)
        elif mode == "transport":
            report("Checking transport support boundary")
            status, result = "UNSUPPORTED", {
                "status": "UNSUPPORTED",
                "reason_code": "E_TRANSPORT_UNSUPPORTED",
                "message": "OpenMC transport requires a verified model adapter; this release only executes ledger replay.",
            }
        else:
            status, result = "UNSUPPORTED", {
                "status": "UNSUPPORTED",
                "reason_code": "E_PROJECT_MODE",
                "message": f"Unsupported project mode: {mode}",
            }
        result["run_id"] = run_id
        result = json_safe(result)
        report("Writing result and report")
        write_mapping(staging / "result" / "inference.json", result)
        _write_html(staging / "result" / "report.html", result)
        manifest_payload = {
            "schema_version": "qotair-run-manifest-v1",
            "run_id": run_id,
            "project_id": project.project_id,
            "status": status,
            "mode": mode,
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "project_path": str(project.path),
            "project_sha256": sha256_mapping(project.as_dict()),
            "plan_path": str(plan_path),
            "plan_sha256": plan.get("plan_sha256"),
            "environment": environment_report(),
            "result_path": "result/inference.json",
            "report_path": "result/report.html",
        }
        _write_run_manifest(staging / "run_manifest.json", manifest_payload)
        layout.write_integrity_manifest(staging)
        report("Committing run")
        final_path = layout.commit_run(staging, final)
        manifest_payload["path"] = str(final_path)
        layout.record_history(manifest_payload)
        report(f"Completed {status}")
        return RunResult(run_id, status, final_path, str(result.get("message", "Run completed")), final_path / "result" / "inference.json", plan_path)
    except Exception as exc:
        report(f"Run failed: {exc}")
        failure = {
            "schema_version": "qotair-run-manifest-v1",
            "run_id": run_id,
            "project_id": project.project_id,
            "status": "FAIL",
            "reason_code": getattr(exc, "code", "E_RUNTIME"),
            "message": str(exc),
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        write_mapping(staging / "result" / "inference.json", failure)
        _write_html(staging / "result" / "report.html", failure)
        _write_run_manifest(staging / "run_manifest.json", failure)
        layout.write_integrity_manifest(staging)
        final_path = layout.commit_run(staging, final)
        failure["path"] = str(final_path)
        layout.record_history(failure)
        return RunResult(run_id, "FAIL", final_path, str(exc), final_path / "result" / "inference.json", plan_path)


def replay_ledger(
    ledger_path: str | Path,
    query_path: str | Path,
    *,
    batch_column: str | None = None,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    ledger = load_csv_ledger(ledger_path, batch_column=batch_column)
    payload = read_mapping(Path(query_path).expanduser().resolve())
    if "query" in payload and isinstance(payload["query"], Mapping):
        payload = dict(payload["query"])
    confidence = float(payload.get("confidence_level", confidence_level))
    inference = infer_query(payload, ledger.values, ledger.score_names, confidence_level=confidence)
    return json_safe({
        "schema_version": "qotair-inference-result-v1",
        "status": inference.get("status", "FAIL"),
        "query_id": payload.get("query_id"),
        "ledger": {"path": str(ledger.source_path), "sha256": sha256_file(ledger.source_path), "n_batches": len(ledger.batch_ids), "score_names": list(ledger.score_names)},
        "inference": inference,
    })


def history(project_path: str | Path) -> list[dict[str, Any]]:
    return list_runs(project_path)
