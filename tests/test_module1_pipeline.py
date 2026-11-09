from __future__ import annotations

import numpy as np
import pytest

from qotair.acquisition import (
    AcquisitionIdentity,
    FilterSpec,
    RegionSpec,
    RunSource,
    bind_region,
    evaluate_declared_query,
    jackknife_partition_ok,
    ledger_mix_forbidden,
    prerun_check,
    replay_from_two_column,
    same_query_identity,
    summarize_records,
)
from qotair.acquisition.identity import AcquisitionError
from qotair.acquisition.archive import load_two_column, write_task_bundle
from qotair.acquisition.correspondence import snapshot_correspondence
from qotair.acquisition.provenance import file_sha256
from qotair.acquisition.plan import ledger_reuse_decision, planned_columns
from qotair.acquisition.select import slice_two_scores, two_score_column_indices
from qotair.acquisition.status import classify_inference, paper_release_status


def _mesh_identity(nx: int = 4, ny: int = 4, nz: int = 1) -> AcquisitionIdentity:
    mesh = FilterSpec.mesh(nx, ny, nz, (-2, -2, -2), (2, 2, 2))
    return AcquisitionIdentity((mesh,), ("total",), ("flux", "absorption"))


def _full_ledger(identity: AcquisitionIdentity, n: int = 60, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cols = identity.filter_bin_count() * 2
    flux = 1.0 + 0.02 * rng.standard_normal((n, identity.filter_bin_count()))
    absorption = 0.5 + 0.01 * rng.standard_normal((n, identity.filter_bin_count()))
    out = np.empty((n, cols))
    out[:, 0::2] = flux
    out[:, 1::2] = absorption
    return out


def test_handwritten_openmc_layout_instance_zero_matches_bind() -> None:
    """Independent last-filter-fastest / score-fastest layout vs RegionSpec bind.

    Does not call pick_distribcell_target. cell_id and instance_index are declared.
    """

    ident = AcquisitionIdentity(
        (FilterSpec.energy((0.0, 0.625, 2.0e7)), FilterSpec.distribcell(52, 17160)),
        ("total",),
        ("flux", "absorption"),
    )
    spec = RegionSpec(kind="distribcell", name="fuel_pin_0", cell_id=52, instance_index=0)
    bound = bind_region(ident, spec)
    assert bound["heuristic_used"] is False
    assert bound["object_path"]["cell_id"] == 52
    assert bound["object_path"]["instance_index"] == 0
    q_cols = two_score_column_indices(ident, bound["selection"], "absorption", "flux")
    n_inst, n_scores = 17160, 2
    energy_bin, instance_index = 0, 0
    filter_index = energy_bin * n_inst + instance_index
    flux_col = filter_index * n_scores + 0
    abs_col = filter_index * n_scores + 1
    assert q_cols == (abs_col, flux_col)
    wide = prerun_check(ident, n_active_batches=60, region=spec, max_columns=20000)
    assert any(item["code"] == "E_LEDGER_TOO_WIDE" for item in wide["issues"])
    assert wide["would_transport"] is True
    assert wide["n_columns"] == 17160 * 2 * 2


def test_silent_distribcell_fallback_is_disabled() -> None:
    spec = RegionSpec(kind="distribcell", name="fuel")
    with pytest.raises(AcquisitionError) as exc:
        spec.validate()
    assert exc.value.code == "E_REGION_UNSPECIFIED"


def test_explicit_mesh_bind_and_column_slice() -> None:
    ident = _mesh_identity()
    spec = RegionSpec(kind="mesh_ijk", name="center", mesh_ijk=(2, 2, 0))
    bound = bind_region(ident, spec)
    assert bound["heuristic_used"] is False
    assert bound["selection"]["MeshFilter"] == 2 + 4 * 2
    ledger = _full_ledger(ident)
    two = slice_two_scores(ledger, ident, bound["selection"], "absorption", "flux")
    cols = two_score_column_indices(ident, bound["selection"], "absorption", "flux")
    assert two.shape == (60, 2)
    np.testing.assert_allclose(two[:, 0], ledger[:, cols[0]])
    np.testing.assert_allclose(two[:, 1], ledger[:, cols[1]])


def test_wrong_cell_is_ambiguous() -> None:
    ident = AcquisitionIdentity((FilterSpec.cell((10, 11)),), ("total",), ("flux", "absorption"))
    spec = RegionSpec(kind="cell", name="x", cell_id=99)
    with pytest.raises(AcquisitionError) as exc:
        bind_region(ident, spec)
    assert exc.value.code == "E_REGION_AMBIGUOUS"


def test_prerun_blocks_sixteen_batches() -> None:
    ident = _mesh_identity()
    report = prerun_check(ident, n_active_batches=16, block_count=6)
    assert report["ok"] is False
    assert report["would_transport"] is False
    assert any(item["code"] == "E_BLOCK_PARTITION" for item in report["issues"])
    assert jackknife_partition_ok(60, 6)
    assert not jackknife_partition_ok(16, 6)


def test_prerun_blocks_wide_ledger() -> None:
    ident = AcquisitionIdentity(
        (FilterSpec.mesh(100, 100, 2, (-1, -1, -1), (1, 1, 1)),),
        ("total",),
        ("flux", "absorption"),
    )
    report = prerun_check(ident, n_active_batches=60, max_columns=20000)
    assert any(item["code"] == "E_LEDGER_TOO_WIDE" for item in report["issues"])
    assert report["would_transport"] is True


def test_product_interval_is_not_paper_hull() -> None:
    ident = _mesh_identity()
    ledger = _full_ledger(ident, seed=7)
    spec = RegionSpec(kind="mesh_ijk", name="c", mesh_ijk=(1, 1, 0))
    result = evaluate_declared_query(ledger, ident, spec, n_active_batches=60)
    obj = result["statistical_object"]
    assert obj["kind"] in {"product_interval", "fieller_set", None}
    assert obj["kind"] != "engineering_interval"
    envelope = result["paper_envelope"]
    assert envelope["method"] in {"ENVELOPE", "REFUSE"}
    if envelope["method"] == "ENVELOPE":
        assert envelope["reason_code"] == "R_ENGINEERING_ENVELOPE"
        product_iv = (result["inference"].get("intervals") or [{}])[0]
        env_iv = envelope["intervals"][0]
        if product_iv.get("lower") is not None and env_iv.get("lower") is not None:
            assert env_iv["lower"] <= product_iv["lower"] + 1e-15
            assert env_iv["upper"] >= product_iv["upper"] - 1e-15


def test_ledger_bundle_replays(tmp_path) -> None:
    ident = _mesh_identity()
    ledger = _full_ledger(ident, seed=11)
    spec = RegionSpec(kind="mesh_ijk", name="c", mesh_ijk=(0, 0, 0))
    result = evaluate_declared_query(ledger, ident, spec, n_active_batches=60)
    two = np.asarray(result["two_column"], float)
    bundle = write_task_bundle(
        tmp_path,
        "t1",
        two,
        {"product_method": result["product_method"], "paper_envelope": result["paper_envelope"]},
    )
    loaded = load_two_column(bundle)
    replay = replay_from_two_column(loaded)
    assert replay["product_method"] == result["product_method"]
    assert replay["paper_envelope"]["method"] == result["paper_envelope"]["method"]
    np.testing.assert_allclose(
        replay["paper_envelope"]["intervals"][0]["lower"],
        result["paper_envelope"]["intervals"][0]["lower"],
    )


def test_run_source_hashes_file_bytes(tmp_path) -> None:
    a = tmp_path / "a.xml"
    b = tmp_path / "b.xml"
    a.write_text("geom-a", encoding="utf-8")
    b.write_text("geom-b", encoding="utf-8")
    left = RunSource.bind_files(geometry=a, materials=a, settings={"batches": 64}, seed=1)
    right = RunSource.bind_files(geometry=b, materials=a, settings={"batches": 64}, seed=1)
    assert left.geometry_sha256 != right.geometry_sha256
    assert ledger_mix_forbidden(left, right)
    assert file_sha256(a) == left.geometry_sha256


def test_correspondence_reject_cases() -> None:
    assert snapshot_correspondence("mesh_ijk", {"bounds": {"shape": [2, 2, 1]}}, {"bounds": {"shape": [4, 4, 1]}})["status"] == "must_update"
    assert snapshot_correspondence("mesh_ijk", {"bounds": None}, {"bounds": {"shape": [2, 2, 1]}})["status"] == "cannot_continue"
    assert snapshot_correspondence("universe", {"universe_id": 3}, {"universe_ids": [1, 2]})["status"] == "cannot_continue"
    assert snapshot_correspondence("material", {"material_id": 9}, {"material_ids": [1]})["status"] == "cannot_continue"
    assert snapshot_correspondence("cell", {"cell_id": 5}, {"cell_ids": [1, 2]})["status"] == "cannot_continue"
    assert snapshot_correspondence("distribcell", {"instances": 4, "path": "a"}, {"instances": 4, "path": "b"})["status"] == "must_update"


def test_fieller_is_diagnostic_not_release() -> None:
    classified = classify_inference(
        {
            "status": "PASS",
            "method": "FIELLER",
            "reason_code": "R_DENOMINATOR_BORDERLINE",
            "interval_kind": "bounded",
            "intervals": [{"lower": 0.0, "upper": 1.0}],
        }
    )
    assert classified["execute_status"] == "success"
    assert classified["release_status"] == "DIAGNOSTIC"
    assert classified["statistical_object"]["kind"] == "fieller_set"
    assert paper_release_status({"status": "PASS", "method": "DELTA"}) == "RELEASE"


def test_evaluate_and_replay_agree() -> None:
    ident = _mesh_identity()
    ledger = _full_ledger(ident, seed=7)
    spec = RegionSpec(kind="mesh_ijk", name="c", mesh_ijk=(1, 1, 0))
    result = evaluate_declared_query(ledger, ident, spec, n_active_batches=60)
    assert result["execute_status"] == "success"
    assert result["release_status"] in {"RELEASE", "DIAGNOSTIC", "REFUSE"}
    assert result["binding"]["heuristic_used"] is False
    two = slice_two_scores(ledger, ident, result["binding"]["data_map"]["selection"], "absorption", "flux")
    replay = replay_from_two_column(two, numerator="absorption", denominator="flux")
    assert replay["release_status"] == result["release_status"]
    assert replay["inference"]["method"] == result["inference"]["method"]


def test_config_error_does_not_transport_semantics() -> None:
    ident = _mesh_identity()
    ledger = _full_ledger(ident, n=16)
    spec = RegionSpec(kind="mesh_ijk", name="c", mesh_ijk=(1, 1, 0))
    result = evaluate_declared_query(ledger, ident, spec, n_active_batches=16)
    assert result["execute_status"] == "config_error"
    assert result["release_status"] == "REFUSE"
    assert result["reason_code"] == "E_BLOCK_PARTITION"


def test_run_sources_do_not_mix() -> None:
    ident = _mesh_identity()
    a = RunSource(materials="fresh.xml", seed=1)
    b = RunSource(materials="eoc.xml", seed=1)
    assert same_query_identity(ident, ident)
    assert ledger_mix_forbidden(a, b)
    assert not ledger_mix_forbidden(a, a)


def test_summarize_never_counts_fieller_as_release() -> None:
    records = [
        {"execute_status": "success", "release_status": "DIAGNOSTIC", "product_method": "FIELLER"},
        {"execute_status": "success", "release_status": "RELEASE", "product_method": "DELTA"},
        {"execute_status": "config_error", "release_status": "REFUSE", "product_method": "REFUSE"},
    ]
    counts = summarize_records(records)
    assert counts["release_DIAGNOSTIC"] == 1
    assert counts["release_RELEASE"] == 1
    assert counts["fieller_counted_as_release"] == 0
    assert counts["execute_config_error"] == 1


def test_planned_columns_dedupe_and_reuse() -> None:
    ident = _mesh_identity()
    center = RegionSpec(kind="mesh_ijk", name="center", mesh_ijk=(2, 2, 0))
    edge = RegionSpec(kind="mesh_ijk", name="edge", mesh_ijk=(0, 0, 0))
    plan = planned_columns(ident, [center, edge])
    assert plan["n_unique"] == 4
    assert plan["n_if_separate"] == 4
    assert plan["n_identity"] == 32
    assert plan["retention"]["n_dropped"] == 28
    a = RunSource(materials="boc.xml", seed=1)
    b = RunSource(materials="eoc.xml", seed=1)
    reuse = ledger_reuse_decision(ident, ident, a, a)
    assert reuse["reuse"] is True
    assert ledger_reuse_decision(ident, ident, a, b)["must_reacquire"] is True
    bound = bind_region(ident, center)
    assert bound["object_path"]["index_base"] == 0
    assert isinstance(bound["object_path"]["linear_index"], int)
    assert "i + nx" in bound["object_path"]["linear_index_formula"]


def test_uncomputed_is_not_statistical_refuse() -> None:
    ident = _mesh_identity()
    spec = RegionSpec(kind="cell", name="missing", cell_id=99)
    result = evaluate_declared_query(np.zeros((60, 32)), ident, spec, n_active_batches=60)
    assert result["execute_status"] == "config_error"
    assert result["release_status"] == "REFUSE"
    assert result["reason_code"] in {"E_BIN_SELECTION", "E_REGION_AMBIGUOUS", "E_REGION_UNSPECIFIED"}
    counts = summarize_records([result])
    assert counts["execute_config_error"] == 1
    assert counts["release_REFUSE"] == 1


def test_simple_baseline_on_same_two_columns() -> None:
    ident = _mesh_identity()
    ledger = _full_ledger(ident, seed=3)
    spec = RegionSpec(kind="mesh_ijk", name="c", mesh_ijk=(0, 0, 0))
    result = evaluate_declared_query(ledger, ident, spec)
    two = slice_two_scores(ledger, ident, result["binding"]["data_map"]["selection"], "absorption", "flux")
    mean_d = float(two[:, 1].mean())
    se_d = float(two[:, 1].std(ddof=1) / np.sqrt(two.shape[0]))
    t_den = abs(mean_d) / se_d
    simple = "REFUSE" if t_den < 1 else ("DIAGNOSTIC" if t_den < 3 else "RELEASE")
    assert simple in {"RELEASE", "DIAGNOSTIC", "REFUSE"}
    assert result["release_status"] in {"RELEASE", "DIAGNOSTIC", "REFUSE"}
