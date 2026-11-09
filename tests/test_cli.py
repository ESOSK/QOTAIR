from __future__ import annotations

import json
import csv

from qotair.cli import main
from qotair.services.execution import run_project
from qotair.storage.projects import write_mapping


def _example(name: str) -> str:
    from qotair.paths import PACKAGE_ROOT

    return str(PACKAGE_ROOT / "examples" / "projects" / name)


def test_query_and_plan_json_contract(capsys):
    assert main(["query", "operands", _example("query-ratio.yaml"), "--json"]) == 0
    operands = json.loads(capsys.readouterr().out)
    assert operands["status"] == "PASS"
    assert operands["operands"] == ["absorption", "flux"]

    assert main(["plan", _example("pin-replay.yaml"), "--no-write", "--json"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["status"] == "PASS"
    assert plan["required_operands"] == ["absorption", "flux"]


def test_missing_input_has_clean_json_and_input_exit(capsys, tmp_path):
    code = main(["query", "check", str(tmp_path / "missing.yaml"), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["reason_code"] == "E_FILE_NOT_FOUND"


def test_validate_and_verify_report_status(capsys, tmp_path):
    project = tmp_path / "project.yaml"
    write_mapping(
        project,
        {
            "schema_version": "qotair-project-v1",
            "project_id": "empty-project",
            "mode": "archive_replay",
            "model": {"kind": "archive"},
            "queries": {
                "schema_version": "p1-query-v1",
                "query_id": "frozen",
                "name": "Frozen",
                "expression_class": "linear_combination",
                "output_unit": "report",
                "operand_units": {"report": "report"},
                "domain_constraints": ["preserved analysis contract"],
                "expression": {"op": "score", "name": "report"},
            },
        },
    )
    assert main(["validate", str(project), "--json"]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["status"] == "PASS"

    code = main(["verify", str(project), "--json"])
    verified = json.loads(capsys.readouterr().out)
    assert code == 2
    assert verified["reason_code"] == "E_RUN_NOT_FOUND"


def test_direct_replay_json(capsys):
    ledger = _example("../data/batch_operand_ledger.csv")
    query = _example("query-ratio.yaml")
    assert main(["replay", ledger, "--query", query, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PASS"
    assert payload["inference"]["reason_code"] == "R_REGULAR_ASYMPTOTIC"


def test_verify_detects_tampering_with_exit_six(capsys, tmp_path):
    ledger = tmp_path / "ledger.csv"
    with ledger.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["batch_id", "absorption", "flux"])
        for index in range(1, 16):
            writer.writerow([index, 1.0 + index * 0.01, 4.0 + index * 0.02])
    project = tmp_path / "project.yaml"
    write_mapping(
        project,
        {
            "schema_version": "qotair-project-v1",
            "project_id": "tamper-check",
            "mode": "ledger_replay",
            "model": {"kind": "builtin", "name": "pin"},
            "ledger": {"path": "ledger.csv", "format": "csv"},
            "queries": [{
                "schema_version": "p1-query-v1",
                "query_id": "ratio",
                "name": "Ratio",
                "expression_class": "ratio",
                "output_unit": "1",
                "operand_units": {"absorption": "1", "flux": "1"},
                "domain_constraints": ["stable denominator"],
                "expression": {"op": "divide", "args": [{"op": "score", "name": "absorption"}, {"op": "score", "name": "flux"}]},
            }],
        },
    )
    result = run_project(project)
    report = result.path / "result" / "report.html"
    report.write_text(report.read_text(encoding="utf-8") + "tampered", encoding="utf-8")
    code = main(["verify", str(project), result.run_id, "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 6
    assert payload["reason_code"] == "E_INTEGRITY"
    assert "result/report.html" in payload["mismatched"]
