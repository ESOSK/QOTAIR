"""Query-contract transformations. Independent layout math does not call bind_region."""
from __future__ import annotations

import numpy as np

from qotair.acquisition.bind import RegionSpec, bind_region
from qotair.acquisition.contract import (
    QueryContract,
    can_reuse_batch_evidence,
    can_reuse_query_definition,
    contract_columns,
    preflight_contract,
)
from qotair.acquisition.diagnose import (
    judge_task,
    no_diagnostic_envelope,
    production_infer,
    simple_gated_envelope,
    simple_labels,
    summarize_evaluation,
)
from qotair.acquisition.planner import RECORDED_SIMPLE_RULES, legal_plans_same_tally, select_plan
from qotair.acquisition.identity import AcquisitionIdentity, FilterSpec
from qotair.acquisition.provenance import RunSource
from qotair.acquisition.select import two_score_column_indices


def _independent_mesh_columns(nx, ny, nz, i, j, k, energy_bin, scores):
    """OpenMC last-filter-fastest, score-fastest. Not bind_region."""

    n_mesh = int(nx) * int(ny) * int(nz)
    filter_index = int(energy_bin) * n_mesh + int(i) + int(nx) * (int(j) + int(ny) * int(k))
    n_scores = len(scores)
    return filter_index * n_scores + scores.index("absorption"), filter_index * n_scores + scores.index("flux")


def _mesh_identity(scores=("flux", "absorption")):
    mesh = FilterSpec.mesh(4, 4, 1, (-2, -2, -2), (2, 2, 2))
    energy = FilterSpec.energy((0.0, 0.625, 2.0e7))
    return AcquisitionIdentity((energy, mesh), ("total",), scores)


def test_independent_layout_matches_binder_on_center() -> None:
    ident = _mesh_identity()
    spec = RegionSpec(kind="mesh_ijk", name="c", mesh_ijk=(2, 2, 0), energy_bin=0)
    q_cols = two_score_column_indices(ident, bind_region(ident, spec)["selection"], "absorption", "flux")
    ind = _independent_mesh_columns(4, 4, 1, 2, 2, 0, 0, ("flux", "absorption"))
    assert q_cols == ind


def test_score_order_change_keeps_meaning_not_integer_columns() -> None:
    a = _mesh_identity(("flux", "absorption"))
    b = _mesh_identity(("absorption", "flux"))
    contract = QueryContract(name="c", kind="mesh_ijk", mesh_ijk=(1, 0, 0), numerator="absorption", denominator="flux")
    cols_a = contract_columns(a, contract)
    cols_b = contract_columns(b, contract)
    assert cols_a != cols_b
    assert can_reuse_query_definition(contract, QueryContract(name="other", kind="mesh_ijk", mesh_ijk=(1, 0, 0), numerator="absorption", denominator="flux"))["reuse"]
    rng = np.random.default_rng(0)
    n, na, nb = 60, a.filter_bin_count() * 2, b.filter_bin_count() * 2
    led_a = rng.random((n, na))
    two_a = led_a[:, list(cols_a)]
    led_b = np.empty((n, nb))
    # meaning-preserving remap: swap score order inside each filter bin
    for filt in range(a.filter_bin_count()):
        led_b[:, filt * 2 + 0] = led_a[:, filt * 2 + 1]  # absorption first on B
        led_b[:, filt * 2 + 1] = led_a[:, filt * 2 + 0]
    two_b = led_b[:, list(cols_b)]
    np.testing.assert_allclose(two_a, two_b)


def test_energy_edges_not_just_bin_index() -> None:
    ident_a = AcquisitionIdentity((FilterSpec.energy((0.0, 0.625, 2.0e7)), FilterSpec.cell((1,))), ("total",), ("flux", "absorption"))
    ident_b = AcquisitionIdentity((FilterSpec.energy((0.0, 100.0, 2.0e7)), FilterSpec.cell((1,))), ("total",), ("flux", "absorption"))
    ca = QueryContract.bind(ident_a, name="c", kind="cell", cell_id=1, energy_bin=0, numerator="absorption", denominator="flux")
    cb = QueryContract.bind(ident_b, name="c", kind="cell", cell_id=1, energy_bin=0, numerator="absorption", denominator="flux")
    assert ca.energy_bin == cb.energy_bin
    assert can_reuse_query_definition(ca, cb)["reuse"] is False
    assert can_reuse_query_definition(ca, cb)["reason_code"] in {"E_ENERGY_MISMATCH", "E_IDENTITY_MISMATCH"}
    report = preflight_contract(ident_b, ca)
    assert report["ok"] is False
    assert any(item["code"] in {"E_ENERGY_EDGES_MISMATCH", "E_IDENTITY_MISMATCH"} for item in report["issues"])
    src = RunSource.bind_files(settings={"batches": 64}, seed=1)
    ev = can_reuse_batch_evidence(src, src, ca, cb, kept_meanings=[ca.meaning_sha256])
    assert ev["reuse"] is False


def test_energy_or_instance_change_blocks_definition_reuse() -> None:
    base = QueryContract(name="c", kind="mesh_ijk", mesh_ijk=(1, 1, 0), energy_bin=0, numerator="absorption", denominator="flux")
    energy = QueryContract(name="c", kind="mesh_ijk", mesh_ijk=(1, 1, 0), energy_bin=1, numerator="absorption", denominator="flux")
    inst = QueryContract(name="f", kind="distribcell", cell_id=52, instance_index=1, expected_instances=17160, numerator="absorption", denominator="flux")
    inst0 = QueryContract(name="f", kind="distribcell", cell_id=52, instance_index=0, expected_instances=17160, numerator="absorption", denominator="flux")
    assert can_reuse_query_definition(base, energy)["reason_code"] == "E_ENERGY_MISMATCH"
    assert can_reuse_query_definition(inst0, inst)["reason_code"] == "E_INSTANCE_MISMATCH"


def test_retained_columns_allow_new_name_same_meaning() -> None:
    src = RunSource.bind_files(settings={"batches": 64}, seed=1)
    ident = _mesh_identity()
    stored = QueryContract.bind(ident, name="center", kind="mesh_ijk", mesh_ijk=(2, 2, 0), numerator="absorption", denominator="flux")
    alias = QueryContract.bind(ident, name="mid", kind="mesh_ijk", mesh_ijk=(2, 2, 0), numerator="absorption", denominator="flux")
    edge = QueryContract.bind(ident, name="edge", kind="mesh_ijk", mesh_ijk=(0, 0, 0), numerator="absorption", denominator="flux")
    kept = [stored.meaning_sha256]
    assert can_reuse_batch_evidence(src, src, stored, alias, kept_meanings=kept)["reuse"]
    miss = can_reuse_batch_evidence(src, src, stored, edge, kept_meanings=kept)
    assert miss["reuse"] is False
    assert miss["reason_code"] == "E_REGION_MISMATCH"


def test_material_change_forbids_evidence_reuse() -> None:
    a = RunSource.bind_files(materials="boc.xml", settings={"p": 1}, seed=1)
    b = RunSource.bind_files(materials="eoc.xml", settings={"p": 1}, seed=1)
    ident = AcquisitionIdentity((FilterSpec.energy((0.0, 0.625, 2e7)), FilterSpec.cell((1,))), ("total",), ("flux", "absorption"))
    q = QueryContract.bind(ident, name="c", kind="cell", cell_id=1, numerator="absorption", denominator="flux")
    rec = can_reuse_batch_evidence(a, b, q, q, kept_meanings=[q.meaning_sha256])
    assert rec["reuse"] is False
    assert rec["reason_code"] == "E_RUN_SOURCE_MISMATCH"


def test_preflight_wrong_cell() -> None:
    ident = AcquisitionIdentity((FilterSpec.cell((1,)),), ("total",), ("flux", "absorption"))
    bad = QueryContract(name="x", kind="cell", cell_id=99, numerator="absorption", denominator="flux")
    report = preflight_contract(ident, bad)
    assert report["ok"] is False
    assert report["would_transport"] is False


def test_diagnose_labels_and_unable() -> None:
    rng = np.random.default_rng(1)
    two = np.column_stack([0.5 + 0.01 * rng.standard_normal(60), 1.0 + 0.02 * rng.standard_normal(60)])
    prod = production_infer(two)
    none = no_diagnostic_envelope(two)
    simple = simple_labels(two)
    assert simple["engineering_interval"] is None
    assert prod["engineering_interval"] is not None
    assert prod.get("envelope_width") == 2.0 * prod["envelope_halfwidth"]
    point = float(two[:, 0].mean() / two[:, 1].mean())
    judged = judge_task(prod, oracle_point=point, oracle_sd=1e-12, no_diag=none)
    assert judged["able_to_judge"] is True
    coarse = judge_task(prod, oracle_point=point, oracle_sd=1.0, no_diag=none)
    assert coarse["able_to_judge"] is False
    summary = summarize_evaluation([judged, coarse])
    assert summary["n_unable"] == 1
    assert summary["n_judged"] == 1
    gated = simple_gated_envelope(two)
    assert gated["engineering_interval"] is not None
    assert gated["rule"] == "simple_gated_envelope"


def test_evaluate_rejects_stale_energy_edges() -> None:
    ident_a = AcquisitionIdentity(
        (FilterSpec.energy((0.0, 0.625, 2.0e7)), FilterSpec.cell((1,))),
        ("total",),
        ("flux", "absorption"),
    )
    ident_b = AcquisitionIdentity(
        (FilterSpec.energy((0.0, 100.0, 2.0e7)), FilterSpec.cell((1,))),
        ("total",),
        ("flux", "absorption"),
    )
    spec = RegionSpec(kind="cell", name="c", cell_id=1)
    stale = QueryContract.bind(
        ident_a, name="c", kind="cell", cell_id=1, energy_bin=0, numerator="absorption", denominator="flux"
    )
    ledger = np.ones((60, 4))
    from qotair.acquisition.pipeline import evaluate_declared_query

    rec = evaluate_declared_query(ledger, ident_b, spec, n_active_batches=60, contract=stale)
    assert rec["execute_status"] == "config_error"
    assert rec["reason_code"] in {"E_ENERGY_EDGES_MISMATCH", "E_IDENTITY_MISMATCH"}


def test_rss_unknown_cannot_claim_limit() -> None:
    measured = [
        {"plan_id": "selected_copy", "legal": True, "file_bytes": 100, "end_to_end_s": 1.0, "process_peak_rss_bytes": None},
    ]
    rec = select_plan(measured, objective="min_storage", rss_limit_bytes=10**9)
    assert rec["selected"] is None
    assert rec["rejected"][0]["reason"] == "rss_unknown_cannot_claim_limit"


def test_select_plan_min_storage() -> None:
    plans = legal_plans_same_tally()
    assert {p.plan_id for p in plans} == {"full_then_slice", "selected_copy"}
    measured = [
        {"plan_id": "full_then_slice", "legal": True, "file_bytes": 555008, "end_to_end_s": 22.0, "process_peak_rss_bytes": 100},
        {"plan_id": "selected_copy", "legal": True, "file_bytes": 1088, "end_to_end_s": 22.5, "process_peak_rss_bytes": 100},
    ]
    rec = select_plan(measured, objective="min_storage")
    assert rec["selected"] == "selected_copy"
    assert rec["matches_simple_rule"] is True


def test_incomplete_contract_refused_on_reuse() -> None:
    src = RunSource.bind_files(settings={"batches": 1}, seed=1)
    incomplete = QueryContract(name="c", kind="cell", cell_id=1, numerator="absorption", denominator="flux")
    rec = can_reuse_batch_evidence(src, src, incomplete, incomplete, kept_meanings=[incomplete.meaning_sha256])
    assert rec["reuse"] is False
    assert rec["reason_code"] in {"E_ENERGY_UNSPECIFIED", "E_IDENTITY_UNSPECIFIED"}


def test_evaluate_and_preflight_reject_incomplete_contract() -> None:
    ident = AcquisitionIdentity(
        (FilterSpec.energy((0.0, 0.625, 2.0e7)), FilterSpec.cell((1,))),
        ("total",),
        ("flux", "absorption"),
    )
    spec = RegionSpec(kind="cell", name="c", cell_id=1)
    incomplete = QueryContract(name="c", kind="cell", cell_id=1, numerator="absorption", denominator="flux")
    report = preflight_contract(ident, incomplete)
    assert report["ok"] is False
    assert report["would_transport"] is False
    assert any(item["code"] in {"E_ENERGY_UNSPECIFIED", "E_IDENTITY_UNSPECIFIED"} for item in report["issues"])
    from qotair.acquisition.pipeline import evaluate_declared_query

    rec = evaluate_declared_query(np.ones((60, 4)), ident, spec, n_active_batches=60, contract=incomplete)
    assert rec["execute_status"] == "config_error"
    assert rec["reason_code"] in {"E_ENERGY_UNSPECIFIED", "E_IDENTITY_UNSPECIFIED"}


def test_recorded_simple_rules_include_share_same_identity() -> None:
    assert "selected_copy if n_kept < n_identity else full_then_slice" in RECORDED_SIMPLE_RULES
    assert "one_transport_per_identity_for_the_declared_query_set" in RECORDED_SIMPLE_RULES
    measured = [
        {"plan_id": "full_then_slice", "legal": True, "file_bytes": 555008, "end_to_end_s": 22.0, "process_peak_rss_bytes": 100},
        {"plan_id": "selected_copy", "legal": True, "file_bytes": 1088, "end_to_end_s": 22.5, "process_peak_rss_bytes": 100},
    ]
    rec = select_plan(measured, objective="min_storage")
    assert rec["recorded_simple_rules"] == list(RECORDED_SIMPLE_RULES)
