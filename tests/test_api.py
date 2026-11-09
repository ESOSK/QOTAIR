from __future__ import annotations

import csv

import numpy as np
import pytest

from qotair import API_VERSION, build_plan, compile_query, query_operands, run_project
from qotair.api import APIError, history, infer_or_refuse, show_run, verify_run
from qotair.storage.projects import load_project, write_mapping


QUERY = {
    "schema_version": "p1-query-v1",
    "query_id": "ratio",
    "name": "Ratio",
    "expression_class": "ratio",
    "output_unit": "1",
    "operand_units": {"absorption": "1", "flux": "1"},
    "domain_constraints": ["positive denominator"],
    "expression": {
        "op": "divide",
        "args": [
            {"op": "score", "name": "absorption"},
            {"op": "score", "name": "flux"},
        ],
    },
}


def _project(tmp_path):
    ledger = tmp_path / "ledger.csv"
    with ledger.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["batch_id", "absorption", "flux"])
        for index in range(1, 25):
            writer.writerow([index, 1.0 + index * 0.01, 4.0 + index * 0.02])
    project = tmp_path / "project.yaml"
    write_mapping(
        project,
        {
            "schema_version": "qotair-project-v1",
            "project_id": "api-project",
            "mode": "ledger_replay",
            "model": {"kind": "builtin", "name": "pin"},
            "ledger": {"path": "ledger.csv", "format": "csv"},
            "queries": [QUERY],
            "run": {"confidence_level": 0.95},
        },
    )
    return project


def test_public_compile_contract_accepts_mapping_and_path(tmp_path):
    query_path = tmp_path / "query.yaml"
    write_mapping(query_path, QUERY)

    from_path = compile_query(query_path)
    from_mapping = compile_query(QUERY)

    assert API_VERSION == "qotair-api-v1"
    assert from_path["operands"] == ["absorption", "flux"]
    assert from_mapping["operands"] == from_path["operands"]
    assert query_operands(QUERY) == ("absorption", "flux")


def test_public_plan_run_show_history_and_verify(tmp_path):
    project = _project(tmp_path)

    preview = build_plan(project)
    assert preview["required_operands"] == ["absorption", "flux"]
    assert not (tmp_path / "plan" / "collection_plan.json").exists()

    written = build_plan(project, write=True)
    assert (tmp_path / "plan" / "collection_plan.json").is_file()
    assert written["plan_sha256"] == preview["plan_sha256"]

    run = run_project(project)
    manifest = show_run(project, run.run_id)
    result = show_run(project, run.run_id, result=True)
    records = history(project)
    integrity = verify_run(project, run.run_id)

    assert manifest["run_id"] == run.run_id
    assert result["run_id"] == run.run_id
    assert records[0]["run_id"] == run.run_id
    assert integrity["status"] == "PASS"


def test_public_inference_and_missing_run_error(tmp_path):
    ledger = [[1.01, 4.02], [1.02, 4.04], [1.03, 4.06], [1.04, 4.08], [1.05, 4.10], [1.06, 4.12]]
    result = infer_or_refuse(QUERY, ledger, ["absorption", "flux"])
    assert result["status"] in {"PASS", "REFUSE"}

    with pytest.raises(APIError) as error:
        show_run(_project(tmp_path))
    assert error.value.code == "E_RUN_NOT_FOUND"


def test_public_input_errors_have_stable_api_codes(tmp_path):
    with pytest.raises(APIError) as error:
        compile_query({"query_id": "incomplete"})
    assert error.value.code == "E_QUERY_SCHEMA"

    bad_ledger = tmp_path / "bad.csv"
    bad_ledger.write_text("batch_id,absorption,flux\n1,not-a-number,4\n", encoding="utf-8")
    query_path = tmp_path / "query.yaml"
    write_mapping(query_path, QUERY)
    with pytest.raises(APIError) as error:
        from qotair.api import replay_ledger

        replay_ledger(bad_ledger, query_path)
    assert error.value.code == "E_LEDGER_INPUT"


def test_public_refusal_is_a_result_without_a_finite_interval():
    ledger = np.asarray(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
        ],
        dtype=float,
    )
    result = infer_or_refuse(QUERY, ledger, ["absorption", "flux"])
    assert result["status"] == "REFUSE"
    assert result["method"] == "REFUSE"
    assert result["intervals"] == []
    assert result["estimate"] is None
    assert str(result["reason_code"]).startswith("E_")


def test_transport_and_missing_ledger_boundaries_are_explicit(tmp_path):
    project = _project(tmp_path)
    payload = load_project(project).as_dict()
    payload["mode"] = "transport"
    write_mapping(project, payload)
    transport = run_project(project)
    transport_result = show_run(project, transport.run_id, result=True)
    assert transport_result["status"] == "UNSUPPORTED"
    assert transport_result["reason_code"] == "E_TRANSPORT_UNSUPPORTED"

    missing_project = tmp_path / "missing-ledger.yaml"
    payload = load_project(project).as_dict()
    payload["project_id"] = "missing-ledger"
    payload["mode"] = "ledger_replay"
    payload["ledger"] = {"path": "does-not-exist.csv", "format": "csv"}
    write_mapping(missing_project, payload)
    missing = run_project(missing_project)
    missing_result = show_run(missing_project, missing.run_id, result=True)
    assert missing_result["status"] == "UNSUPPORTED"
    assert missing_result["reason_code"] == "E_LEDGER_MISSING"
