from __future__ import annotations

import csv
import json

from qotair.services.execution import run_project
from qotair.services.planning import build_plan
from qotair.storage.projects import load_project, save_query, write_mapping


def _project(tmp_path):
    ledger = tmp_path / "ledger.csv"
    with ledger.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["batch_id", "absorption", "flux"])
        for index in range(1, 25):
            writer.writerow([index, 1.0 + index * 0.01, 4.0 + index * 0.02])
    project_path = tmp_path / "project.yaml"
    write_mapping(
        project_path,
        {
            "schema_version": "qotair-project-v1",
            "project_id": "test-project",
            "mode": "ledger_replay",
            "model": {"kind": "builtin", "name": "pin"},
            "ledger": {"path": "ledger.csv", "format": "csv"},
            "queries": [
                {
                    "schema_version": "p1-query-v1",
                    "query_id": "ratio",
                    "name": "Ratio",
                    "expression_class": "ratio",
                    "output_unit": "1",
                    "operand_units": {"absorption": "1", "flux": "1"},
                    "domain_constraints": ["positive denominator"],
                    "expression": {"op": "divide", "args": [{"op": "score", "name": "absorption"}, {"op": "score", "name": "flux"}]},
                }
            ],
            "run": {"confidence_level": 0.95},
        },
    )
    return project_path


def test_plan_contains_minimal_operands(tmp_path):
    path = _project(tmp_path)
    project = load_project(path)
    plan = build_plan(project)
    assert plan["required_operands"] == ["absorption", "flux"]
    assert plan["excluded_available_operands"] == []
    assert len(plan["plan_sha256"]) == 64


def test_run_writes_atomic_result_and_manifest(tmp_path):
    path = _project(tmp_path)
    result = run_project(path)
    assert result.status in {"PASS", "REFUSE"}
    assert result.path.is_dir()
    assert (result.path / "run_manifest.json").is_file()
    assert (result.path / "result" / "inference.json").is_file()
    assert (result.path / "result" / "report.html").is_file()
    assert (result.path / "integrity" / "MANIFEST.sha256").is_file()
    payload = json.loads((result.path / "result" / "inference.json").read_text(encoding="utf-8"))
    assert payload["run_id"] == result.run_id


def test_save_query_attaches_external_query_atomically(tmp_path):
    path = _project(tmp_path)
    project_path, query_path = save_query(
        path,
        {
            "schema_version": "p1-query-v1",
            "query_id": "new_ratio",
            "name": "New ratio",
            "expression_class": "ratio",
            "output_unit": "1",
            "operand_units": {"absorption": "1", "flux": "1"},
            "domain_constraints": ["stable denominator"],
            "expression": {"op": "divide", "args": [{"op": "score", "name": "absorption"}, {"op": "score", "name": "flux"}]},
        },
    )
    assert project_path == path.resolve()
    assert query_path.is_file()
    assert any(query.query_id == "new_ratio" for query in load_project(path).queries())


def test_product_magnitude_guard_blocks_large_and_spares_small():
    from qotair.services.scientific import apply_product_magnitude_guard

    blocked = apply_product_magnitude_guard(
        {
            "status": "PASS",
            "engineering_finite_release": True,
            "estimate": 22086.0,
            "reason_code": "R_OK",
        }
    )
    assert blocked["engineering_finite_release"] is False
    assert blocked["status"] == "PASS"
    assert blocked["statistical_engineering_finite_release"] is True
    assert blocked["product_magnitude_guard"]["blocked"] is True
    assert blocked["product_reason_code"] == "R_PRODUCT_MAGNITUDE_CAP"

    spared = apply_product_magnitude_guard(
        {
            "status": "PASS",
            "engineering_finite_release": True,
            "estimate": 0.16,
        }
    )
    assert spared["engineering_finite_release"] is True
    assert spared["product_magnitude_guard"]["blocked"] is False


def test_transport_mode_is_explicitly_unsupported(tmp_path):
    path = _project(tmp_path)
    payload = load_project(path).as_dict()
    payload["mode"] = "transport"
    write_mapping(path, payload)
    result = run_project(path)
    assert result.status == "UNSUPPORTED"
    inference = json.loads((result.path / "result" / "inference.json").read_text(encoding="utf-8"))
    assert inference["reason_code"] == "E_TRANSPORT_UNSUPPORTED"
