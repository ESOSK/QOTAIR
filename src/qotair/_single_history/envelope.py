"""Single-history inference: conservative HAC grid, ESS/PSD gates, engineering envelope."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from .covariance import (
    DependenceError,
    MIN_ESS,
    analyze_dependence,
    bartlett_bandwidth,
    covariance_of_mean_hac,
    _windows,
)
from .router import (
    RouterThresholds,
    infer_block_jackknife_ratio,
    ratio_router_diagnostics,
    select_ratio_route,
)
from .expression import infer_delta, minimal_operands
from .fieller import classify_fieller_moments, physical_intervals
from .methods import (
    ConfidenceInterval,
    InferenceResult,
    _evaluate_from_means,
)


PRODUCTION_VERSION = "p1-historical-envelope"
PROTOCOL_VERSION = "historical-envelope"


def conservative_scalar_variance(values: np.ndarray, gradient: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    n = array.shape[0]
    base = bartlett_bandwidth(n)
    lags = _windows(n, base) or [base]
    g = np.asarray(gradient, dtype=np.float64).reshape(-1)
    best_var = -1.0
    best_L = base
    for lag in lags:
        cov = covariance_of_mean_hac(array, bandwidth=int(lag))
        var = float(g @ cov @ g)
        if var > best_var:
            best_var = var
            best_L = int(lag)
    return {
        "variance": max(0.0, best_var),
        "standard_error": math.sqrt(max(0.0, best_var)),
        "L_base": int(base),
        "L_used": int(best_L),
        "L_grid": [int(x) for x in lags],
    }


def dependence_gate(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 32:
        return {"status": "SKIP", "reason": "n<32", "min_ess": None}
    try:
        payload = analyze_dependence(array, case_id="historical-envelope")
    except DependenceError as exc:
        return {"status": "REFUSE", "reason": exc.code, "min_ess": None}
    min_ess = float(payload["min_ess"])
    if payload["decision"] == "REFUSE_UNSTABLE" or min_ess < MIN_ESS:
        return {"status": "REFUSE", "reason": payload["reason_code"], "min_ess": min_ess, "dependence": payload}
    return {"status": "PASS", "reason": payload["reason_code"], "min_ess": min_ess, "dependence": payload}


def _t_critical(level: float, df: int) -> float:
    return float(student_t.ppf(0.5 + 0.5 * level, df))


def _union(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (min(a[0], b[0]), max(a[1], b[1]))


def route_production_ratio(
    ledger: np.ndarray,
    score_names: Sequence[str],
    *,
    numerator: str,
    denominator: str,
    thresholds: RouterThresholds,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    array = np.asarray(ledger, dtype=float)
    selected = np.column_stack(
        (array[:, score_names.index(numerator)], array[:, score_names.index(denominator)])
    )
    diagnostics = ratio_router_diagnostics(
        array, score_names, numerator=numerator, denominator=denominator, thresholds=thresholds
    )
    mean_y = float(selected[:, 1].mean())
    hac_den = conservative_scalar_variance(selected, np.asarray([0.0, 1.0]))
    if hac_den["standard_error"] > 0.0:
        diagnostics["denominator_t"] = abs(mean_y) / hac_den["standard_error"]
        diagnostics["denominator_se_conservative"] = hac_den["standard_error"]
        diagnostics["denominator_L_used"] = hac_den["L_used"]
    gate = dependence_gate(selected)
    diagnostics = {**diagnostics, "dependence_gate": {"status": gate["status"], "reason": gate["reason"], "min_ess": gate["min_ess"]}}
    method, reason = select_ratio_route(diagnostics, thresholds)
    n = selected.shape[0]
    ess = gate.get("min_ess")
    df_delta = int(n - 1)
    if ess is not None:
        df_delta = int(min(n - 1, max(5, math.floor(ess) - 1)))

    base = {
        "router_version": PRODUCTION_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "operands": [numerator, denominator],
        "diagnostics": diagnostics,
        "engineering_finite_release": False,
    }
    if method == "REFUSE" or gate["status"] == "REFUSE":
        code = reason if method == "REFUSE" else "E_DEPENDENCE_GATE"
        if gate["status"] == "REFUSE":
            code = "E_DEPENDENCE_GATE"
        return {
            **base,
            "status": "REFUSE",
            "method": "REFUSE",
            "reason_code": code,
            "interval_kind": None,
            "intervals": [],
            "estimate": None,
            "standard_error": None,
            "message": "Single-history refusal",
        }

    t_den = diagnostics.get("denominator_t")
    if t_den is not None and float(t_den) < thresholds.denominator_t_fieller_below:
        cov = np.atleast_2d(covariance_of_mean_hac(selected))
        q = _t_critical(confidence_level, selected.shape[0] - 1) ** 2
        classified = classify_fieller_moments(
            float(selected[:, 0].mean()),
            float(selected[:, 1].mean()),
            float(cov[0, 0]),
            float(cov[1, 1]),
            float(cov[0, 1]),
            q=q,
        )
        kind = classified["interval_kind"]
        if kind in {"empty", "numerical_failure"}:
            return {
                **base,
                "status": "REFUSE",
                "method": "REFUSE",
                "reason_code": classified.get("reason_code") or "E_FIELLER_EMPTY_SET",
                "interval_kind": kind,
                "intervals": [],
                "estimate": None,
                "standard_error": None,
                "message": "Fieller numerical failure or empty set is not a diagnostic success",
            }
        mean_y = float(selected[:, 1].mean())
        estimate = None if mean_y == 0.0 else float(selected[:, 0].mean()) / mean_y
        return {
            **base,
            "status": "PASS",
            "method": "FIELLER",
            "reason_code": "R_DIAGNOSTIC_FIELLER_NOT_ENGINEERING_RELEASE",
            "interval_kind": kind,
            "intervals": physical_intervals(classified),
            "estimate": estimate,
            "standard_error": None,
            "engineering_finite_release": False,
            "diagnostics": {**diagnostics, "fieller_studentized": classified.get("studentized"), "fieller_coefficients": classified.get("coefficients")},
            "message": "Studentized Fieller set is diagnostic only",
        }

    mean_x = float(selected[:, 0].mean())
    mean_y = float(selected[:, 1].mean())
    estimate = mean_x / mean_y
    gradient = np.asarray([1.0 / mean_y, -mean_x / mean_y**2])
    hac = conservative_scalar_variance(selected, gradient)
    crit_d = _t_critical(confidence_level, df_delta)
    half_d = crit_d * hac["standard_error"]
    delta_iv = (estimate - half_d, estimate + half_d)
    jack = infer_block_jackknife_ratio(
        array, score_names, numerator=numerator, denominator=denominator,
        confidence_level=confidence_level, block_count=thresholds.block_count,
    )
    if jack.status == "PASS" and jack.intervals and jack.intervals[0].lower is not None:
        jack_iv = (float(jack.intervals[0].lower), float(jack.intervals[0].upper))
        lo, hi = _union(delta_iv, jack_iv)
        jack_ok = True
    else:
        lo, hi = delta_iv
        jack_ok = False
    diagnostics = {
        **diagnostics,
        "conservative_hac": hac,
        "delta_df": df_delta,
        "envelope_uses_jackknife": jack_ok,
    }
    return {
        **base,
        "status": "PASS",
        "method": "ENVELOPE",
        "reason_code": "R_ENGINEERING_ENVELOPE",
        "interval_kind": "bounded",
        "intervals": [{"lower": lo, "upper": hi}],
        "estimate": estimate,
        "standard_error": (hi - lo) / (2.0 * crit_d) if crit_d else None,
        "engineering_finite_release": True,
        "diagnostics": diagnostics,
        "message": "Historical envelope of conservative HAC-delta and block jackknife",
    }


def infer_block_jackknife_expression(
    ledger: np.ndarray,
    score_names: Sequence[str],
    expression: Mapping[str, Any],
    *,
    confidence_level: float,
    block_count: int,
) -> InferenceResult:
    array = np.asarray(ledger, dtype=float)
    operands = tuple(minimal_operands(expression))
    selected = np.column_stack([array[:, list(score_names).index(name)] for name in operands])
    n = selected.shape[0]
    if n < 2 * block_count or n % block_count != 0:
        return InferenceResult(
            status="REFUSE", method="JACKKNIFE", estimate=None, standard_error=None,
            confidence_level=confidence_level, interval_kind=None, intervals=(),
            rejection_code="E_BLOCK_PARTITION", message="unequal block partition",
            operands=operands, diagnostics={},
        )
    full = _evaluate_from_means(
        selected.mean(axis=0), operands, expression,
        denominator_abs_floor=1e-14, denominator_relative_floor=1e-10,
    )
    block_size = n // block_count
    estimates = []
    for index in range(block_count):
        start, stop = index * block_size, (index + 1) * block_size
        reduced = np.concatenate((selected[:start], selected[stop:]), axis=0)
        estimates.append(
            _evaluate_from_means(
                reduced.mean(axis=0), operands, expression,
                denominator_abs_floor=1e-14, denominator_relative_floor=1e-10,
            )
        )
    replicates = np.asarray(estimates, dtype=float)
    replicate_mean = float(replicates.mean())
    variance = float((block_count - 1) / block_count * np.square(replicates - replicate_mean).sum())
    se = math.sqrt(max(0.0, variance))
    crit = _t_critical(confidence_level, block_count - 1)
    half = crit * se
    return InferenceResult(
        status="PASS", method="JACKKNIFE", estimate=float(full), standard_error=se,
        confidence_level=confidence_level, interval_kind="bounded",
        intervals=(ConfidenceInterval(float(full) - half, float(full) + half),),
        rejection_code=None, message="contiguous delete-one-block jackknife",
        operands=operands, diagnostics={"block_count": block_count, "df": block_count - 1},
    )


def infer_production_expression(
    ledger: np.ndarray,
    score_names: Sequence[str],
    expression: Mapping[str, Any],
    *,
    thresholds: RouterThresholds,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    array = np.asarray(ledger, dtype=float)
    raw = infer_delta(array, list(score_names), expression)
    operands = list(raw.operands)
    selected = np.column_stack([array[:, list(score_names).index(name)] for name in operands]) if operands else array
    gate = dependence_gate(selected)
    base = {
        "router_version": PRODUCTION_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "engineering_finite_release": False,
        "diagnostics": {"dependence_gate": {"status": gate["status"], "reason": gate["reason"], "min_ess": gate["min_ess"]}},
    }
    if raw.status != "PASS" or raw.estimate is None or not raw.gradient:
        return {
            **base,
            "status": "REFUSE",
            "method": "REFUSE",
            "reason_code": raw.rejection_code or "E_DELTA_FAILED",
            "intervals": [],
        }
    if gate["status"] == "REFUSE":
        return {
            **base,
            "status": "REFUSE",
            "method": "REFUSE",
            "reason_code": "E_DEPENDENCE_GATE",
            "intervals": [],
            "estimate": raw.estimate,
        }
    n = selected.shape[0]
    ess = gate.get("min_ess")
    df_delta = int(n - 1)
    if ess is not None:
        df_delta = int(min(n - 1, max(5, math.floor(ess) - 1)))
    hac = conservative_scalar_variance(selected, np.asarray(raw.gradient, dtype=np.float64))
    crit = _t_critical(confidence_level, df_delta)
    half = crit * hac["standard_error"]
    delta_iv = (float(raw.estimate) - half, float(raw.estimate) + half)
    jack = infer_block_jackknife_expression(
        array, score_names, expression,
        confidence_level=confidence_level, block_count=thresholds.block_count,
    )
    if jack.status == "PASS" and jack.intervals and jack.intervals[0].lower is not None:
        lo, hi = _union(delta_iv, (float(jack.intervals[0].lower), float(jack.intervals[0].upper)))
        jack_ok = True
    else:
        lo, hi = delta_iv
        jack_ok = False
    return {
        "router_version": PRODUCTION_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "status": "PASS",
        "method": "ENVELOPE",
        "reason_code": "R_ENGINEERING_ENVELOPE",
        "interval_kind": "bounded",
        "intervals": [{"lower": lo, "upper": hi}],
        "estimate": raw.estimate,
        "standard_error": (hi - lo) / (2.0 * crit) if crit else None,
        "engineering_finite_release": True,
        "diagnostics": {
            "dependence_gate": {"status": gate["status"], "reason": gate["reason"], "min_ess": gate["min_ess"]},
            "conservative_hac": hac,
            "delta_df": df_delta,
            "envelope_uses_jackknife": jack_ok,
            "jackknife_status": jack.status,
        },
        "message": "Historical envelope of conservative HAC-delta and block jackknife for a registered expression",
    }
