"""Frozen validation decision rules. No OpenMC."""
from qotair.acquisition.validation import (
    independent_mesh_columns,
    reference_sensitivity,
    source_protocol_decision,
    wilson,
)


def test_source_decision_p4_when_bias_small() -> None:
    paired = [
        {"seed": 1, "r4": 1.00, "r100": 1.001, "hw4": 0.05},
        {"seed": 2, "r4": 1.00, "r100": 0.999, "hw4": 0.05},
        {"seed": 3, "r4": 1.00, "r100": 1.0005, "hw4": 0.05},
    ]
    rec = source_protocol_decision(paired)
    assert rec["protocol"] == "P4"
    assert rec["used_hits"] is False
    assert rec["used_keff"] is False


def test_source_decision_p100_on_relative_bias() -> None:
    paired = [
        {"seed": 1, "r4": 0.12, "r100": 0.15, "hw4": 0.20},
        {"seed": 2, "r4": 0.12, "r100": 0.15, "hw4": 0.20},
        {"seed": 3, "r4": 0.12, "r100": 0.15, "hw4": 0.20},
    ]
    rec = source_protocol_decision(paired)
    assert rec["rel_fail"] is True
    assert rec["protocol"] == "P100"


def test_source_decision_mixed_goes_p100() -> None:
    paired = [
        {"seed": 1, "r4": 1.0, "r100": 1.0, "hw4": 0.01},
        {"seed": 2, "r4": 1.5, "r100": 1.0, "hw4": 0.01},
        {"seed": 3, "r4": 1.0, "r100": 1.0, "hw4": 0.01},
    ]
    rec = source_protocol_decision(paired)
    assert rec["mixed_seeds"] is True
    assert rec["protocol"] == "P100"


def test_wilson_n30_around_p9() -> None:
    rec = wilson(27, 30)
    assert rec["width"] is not None
    assert rec["width"] < 0.28
    assert rec["low"] > 0.7


def test_reference_sensitivity_flip() -> None:
    interval = {"lower": 0.10, "upper": 0.12}
    rec = reference_sensitivity(interval, oracle=0.119, sd=0.002)
    assert rec["center"] is True
    assert rec["plus_2sd"] is False
    assert rec["flip"] is True


def test_independent_mesh_columns_centerish() -> None:
    a, b = independent_mesh_columns(4, 4, 1, 2, 2, 0, 0)
    assert a != b
    assert min(a, b) >= 0


def test_zero_denominator_refuses_not_raises() -> None:
    import numpy as np
    from qotair.acquisition.diagnose import production_infer
    from qotair.acquisition.pipeline import replay_from_two_column

    two = np.column_stack([np.ones(60), np.zeros(60)])
    rec = replay_from_two_column(two)
    assert rec["paper_envelope"]["status"] == "REFUSE"
    assert rec["paper_envelope"]["reason_code"] == "E_ZERO_DENOMINATOR"
    prod = production_infer(two)
    assert prod["engineering_release"] in {"REFUSE", "DIAGNOSTIC"}
    assert prod.get("reason_code") in {"E_ZERO_DENOMINATOR", None} or prod["engineering_release"] == "REFUSE"


def test_magnitude_guard_spares_order_one_ratio() -> None:
    import numpy as np
    from qotair.services.scientific import paper_engineering_envelope

    rng = np.random.default_rng(0)
    den = 1.0 + 0.01 * rng.standard_normal(60)
    num = 0.16 * den + 0.001 * rng.standard_normal(60)
    two = np.column_stack((num, den))
    env = paper_engineering_envelope(two, ["num", "den"], numerator="num", denominator="den")
    assert abs(float(env["estimate"])) < 1.0
    assert env["product_magnitude_guard"]["blocked"] is False
    assert env["engineering_finite_release"] is True
    assert env["status"] == "PASS"
