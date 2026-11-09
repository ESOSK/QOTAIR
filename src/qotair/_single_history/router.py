from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from .methods import (
    ConfidenceInterval,
    InferenceResult,
    infer_delta,
    infer_fieller,
)


ROUTER_VERSION = "p1-inference-router-v1"
PROTOCOL_VERSION = "single-history-router"


@dataclass(frozen=True)
class RouterThresholds:
    minimum_batches: int
    denominator_t_refuse_below: float
    denominator_t_fieller_below: float
    absolute_fisher_skew_jackknife_above: float
    block_sensitivity_z_jackknife_above: float
    block_sensitivity_z_refuse_above: float
    jackknife_bias_z_jackknife_above: float
    block_count: int
    comparison_epsilon: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RouterThresholds":
        return cls(
            minimum_batches=int(values["minimum_batches"]),
            denominator_t_refuse_below=float(values["denominator_t_refuse_below"]),
            denominator_t_fieller_below=float(values["denominator_t_fieller_below"]),
            absolute_fisher_skew_jackknife_above=float(
                values["absolute_fisher_skew_jackknife_above"]
            ),
            block_sensitivity_z_jackknife_above=float(
                values["block_sensitivity_z_jackknife_above"]
            ),
            block_sensitivity_z_refuse_above=float(
                values["block_sensitivity_z_refuse_above"]
            ),
            jackknife_bias_z_jackknife_above=float(
                values["jackknife_bias_z_jackknife_above"]
            ),
            block_count=int(values["block_count"]),
            comparison_epsilon=float(values["comparison_epsilon"]),
        )

    def validate(self) -> None:
        if self.minimum_batches < 3:
            raise ValueError("minimum_batches must be at least three")
        if not (
            0.0
            <= self.denominator_t_refuse_below
            < self.denominator_t_fieller_below
        ):
            raise ValueError("denominator thresholds must be ordered")
        if not (
            0.0
            <= self.block_sensitivity_z_jackknife_above
            < self.block_sensitivity_z_refuse_above
        ):
            raise ValueError("block sensitivity thresholds must be ordered")
        if self.block_count < 2:
            raise ValueError("block_count must be at least two")
        if self.comparison_epsilon <= 0.0:
            raise ValueError("comparison_epsilon must be positive")


def _fisher_skew(values: np.ndarray) -> float:
    n = values.size
    centered = values - values.mean()
    m2 = float(np.mean(centered**2))
    if m2 <= 0.0:
        return 0.0
    m3 = float(np.mean(centered**3))
    raw = m3 / (m2**1.5)
    return math.sqrt(n * (n - 1)) / (n - 2) * raw if n > 2 else raw


def ratio_router_diagnostics(
    ledger: np.ndarray,
    score_names: Sequence[str],
    *,
    numerator: str,
    denominator: str,
    thresholds: RouterThresholds,
) -> dict[str, float | int | bool | None]:
    """Compute the frozen Week-08 diagnostics without selecting a method."""

    thresholds.validate()
    array = np.asarray(ledger, dtype=float)
    n_batches = int(array.shape[0]) if array.ndim == 2 else 0
    valid_shape = bool(array.ndim == 2 and array.shape[1] == len(score_names))
    unique_scores = len(set(score_names)) == len(score_names)
    required_scores = numerator in score_names and denominator in score_names
    finite = bool(
        valid_shape and unique_scores and required_scores and np.all(np.isfinite(array))
    )
    if not finite or n_batches < 3:
        return {
            "n_batches": n_batches,
            "finite": finite,
            "denominator_t": None,
            "max_abs_fisher_skew": None,
            "block_sensitivity_z": None,
            "jackknife_bias_z": None,
        }

    selected = np.column_stack(
        (array[:, score_names.index(numerator)], array[:, score_names.index(denominator)])
    )
    numerator_values = selected[:, 0]
    denominator_values = selected[:, 1]
    mean_x = float(numerator_values.mean())
    mean_y = float(denominator_values.mean())
    from .covariance import bartlett_bandwidth, covariance_of_mean_hac
    covariance = np.atleast_2d(covariance_of_mean_hac(selected))
    denominator_se = math.sqrt(max(0.0, float(covariance[1, 1])))
    denominator_t = math.inf if denominator_se == 0.0 else abs(mean_y) / denominator_se
    full_estimate = math.inf if mean_y == 0.0 else mean_x / mean_y
    gradient = (
        np.asarray([1.0 / mean_y, -mean_x / mean_y**2])
        if mean_y != 0.0
        else np.asarray([math.inf, math.inf])
    )
    delta_variance = (
        float(gradient @ covariance @ gradient)
        if np.all(np.isfinite(gradient))
        else math.inf
    )
    delta_se = math.sqrt(max(0.0, delta_variance))

    leave_one_out: list[float] = []
    for index in range(n_batches):
        reduced = np.delete(selected, index, axis=0)
        reduced_denominator = float(reduced[:, 1].mean())
        leave_one_out.append(
            math.inf
            if reduced_denominator == 0.0
            else float(reduced[:, 0].mean()) / reduced_denominator
        )
    loo = np.asarray(leave_one_out, dtype=float)
    if np.all(np.isfinite(loo)) and math.isfinite(full_estimate):
        jackknife_estimate = (
            n_batches * full_estimate - (n_batches - 1) * float(np.mean(loo))
        )
        jackknife_bias_z = abs(jackknife_estimate - full_estimate) / max(
            delta_se, thresholds.comparison_epsilon
        )
    else:
        jackknife_bias_z = math.inf

    sensitivity = 0.0
    if n_batches >= thresholds.minimum_batches and math.isfinite(full_estimate):
        for block in np.array_split(np.arange(n_batches), thresholds.block_count):
            reduced = np.delete(selected, block, axis=0)
            reduced_denominator = float(reduced[:, 1].mean())
            candidate = (
                math.inf
                if reduced_denominator == 0.0
                else float(reduced[:, 0].mean()) / reduced_denominator
            )
            sensitivity = max(
                sensitivity,
                abs(candidate - full_estimate)
                / max(delta_se, thresholds.comparison_epsilon),
            )

    return {
        "n_batches": n_batches,
        "finite": finite,
        "denominator_t": float(denominator_t),
        "max_abs_fisher_skew": float(
            max(abs(_fisher_skew(numerator_values)), abs(_fisher_skew(denominator_values)))
        ),
        "block_sensitivity_z": float(sensitivity),
        "jackknife_bias_z": float(jackknife_bias_z),
        "covariance_estimator": "hac_bartlett",
        "hac_bandwidth": int(bartlett_bandwidth(n_batches)),
    }


def select_ratio_route(
    diagnostics: Mapping[str, float | int | bool | None],
    thresholds: RouterThresholds,
) -> tuple[str, str]:
    """Apply the immutable Week-08 decision order to diagnostics."""

    thresholds.validate()
    if not diagnostics["finite"]:
        return "REFUSE", "E_NONFINITE_INPUT"
    if int(diagnostics["n_batches"]) < thresholds.minimum_batches:
        return "REFUSE", "E_ROUTER_INSUFFICIENT_BATCHES"
    denominator_t = float(diagnostics["denominator_t"])
    if denominator_t < thresholds.denominator_t_refuse_below:
        return "REFUSE", "E_DENOMINATOR_UNIDENTIFIED"
    if denominator_t < thresholds.denominator_t_fieller_below:
        return "FIELLER", "R_DENOMINATOR_BORDERLINE"
    sensitivity = float(diagnostics["block_sensitivity_z"])
    if sensitivity > thresholds.block_sensitivity_z_refuse_above:
        return "REFUSE", "E_BATCH_INSTABILITY"
    if (
        float(diagnostics["max_abs_fisher_skew"])
        > thresholds.absolute_fisher_skew_jackknife_above
        or sensitivity > thresholds.block_sensitivity_z_jackknife_above
        or float(diagnostics["jackknife_bias_z"])
        > thresholds.jackknife_bias_z_jackknife_above
    ):
        return "JACKKNIFE", "R_NONLINEAR_OR_SENSITIVE"
    return "DELTA", "R_REGULAR_ASYMPTOTIC"


def infer_block_jackknife_ratio(
    ledger: np.ndarray,
    score_names: Sequence[str],
    *,
    numerator: str,
    denominator: str,
    confidence_level: float,
    block_count: int,
    denominator_floor: float = 1.0e-14,
) -> InferenceResult:
    """Equal-size contiguous delete-one-block jackknife for a direct ratio."""

    array = np.asarray(ledger, dtype=float)
    operands = (numerator, denominator)
    if array.ndim != 2 or array.shape[1] != len(score_names):
        return _inference_refusal("E_SHAPE", "Ledger shape does not match score_names", operands)
    if numerator not in score_names or denominator not in score_names:
        return _inference_refusal("E_UNKNOWN_OPERAND", "Ratio operand is absent", operands)
    if not np.isfinite(array).all():
        return _inference_refusal("E_NONFINITE", "Ledger contains a non-finite value", operands)
    if not 0.0 < float(confidence_level) < 1.0:
        return _inference_refusal(
            "E_INVALID_CONFIDENCE_LEVEL", "confidence_level must be between zero and one", operands
        )
    if array.shape[0] < 2 * block_count or array.shape[0] % block_count != 0:
        return _inference_refusal(
            "E_BLOCK_PARTITION",
            "Equal contiguous blocks require at least two batches per block and exact division",
            operands,
        )

    selected = np.column_stack(
        (array[:, score_names.index(numerator)], array[:, score_names.index(denominator)])
    )
    mean_denominator = float(selected[:, 1].mean())
    if abs(mean_denominator) <= denominator_floor:
        return _inference_refusal("E_ZERO_DENOMINATOR", "Full-sample denominator is zero", operands)
    full_estimate = float(selected[:, 0].mean()) / mean_denominator
    estimates: list[float] = []
    block_size = selected.shape[0] // block_count
    for block_index in range(block_count):
        start = block_index * block_size
        stop = start + block_size
        reduced = np.concatenate((selected[:start], selected[stop:]), axis=0)
        reduced_denominator = float(reduced[:, 1].mean())
        if abs(reduced_denominator) <= denominator_floor:
            return _inference_refusal(
                "E_JACKKNIFE_UNSTABLE_REPLICATE",
                f"Delete-block replicate {block_index} has a zero denominator",
                operands,
            )
        estimates.append(float(reduced[:, 0].mean()) / reduced_denominator)

    replicates = np.asarray(estimates, dtype=float)
    replicate_mean = float(replicates.mean())
    bias_corrected = float(
        block_count * full_estimate - (block_count - 1) * replicate_mean
    )
    variance = float(
        (block_count - 1)
        / block_count
        * np.square(replicates - replicate_mean).sum()
    )
    standard_error = math.sqrt(max(0.0, variance))
    critical = float(
        student_t.ppf(0.5 + float(confidence_level) / 2.0, block_count - 1)
    )
    half_width = critical * standard_error
    return InferenceResult(
        status="PASS",
        method="JACKKNIFE",
        estimate=bias_corrected,
        standard_error=standard_error,
        confidence_level=float(confidence_level),
        interval_kind="bounded",
        intervals=(ConfidenceInterval(bias_corrected - half_width, bias_corrected + half_width),),
        rejection_code=None,
        message="Contiguous delete-one-block jackknife completed",
        operands=operands,
        diagnostics={
            "jackknife_variant": "contiguous_delete_one_block",
            "block_count": block_count,
            "block_size": block_size,
            "full_sample_estimate": full_estimate,
            "replicate_mean": replicate_mean,
            "replicate_min": float(replicates.min()),
            "replicate_max": float(replicates.max()),
            "degrees_of_freedom": block_count - 1,
            "critical_value": critical,
        },
    )


def _inference_refusal(
    code: str, message: str, operands: tuple[str, str]
) -> InferenceResult:
    return InferenceResult(
        status="REFUSE",
        method="REFUSE",
        estimate=None,
        standard_error=None,
        confidence_level=None,
        interval_kind=None,
        intervals=(),
        rejection_code=code,
        message=message,
        operands=operands,
        diagnostics={},
    )


def route_ratio_inference(
    ledger: np.ndarray,
    score_names: Sequence[str],
    *,
    numerator: str,
    denominator: str,
    thresholds: RouterThresholds,
    confidence_level: float = 0.95,
    protocol_version: str = PROTOCOL_VERSION,
) -> dict[str, Any]:
    """Select and execute one inference method, or return a credible refusal."""

    diagnostics = ratio_router_diagnostics(
        ledger,
        score_names,
        numerator=numerator,
        denominator=denominator,
        thresholds=thresholds,
    )
    method, reason_code = select_ratio_route(diagnostics, thresholds)
    expression = {
        "op": "divide",
        "args": [
            {"op": "score", "name": numerator},
            {"op": "score", "name": denominator},
        ],
    }
    if method == "REFUSE":
        return {
            "router_version": ROUTER_VERSION,
            "protocol_version": protocol_version,
            "status": "REFUSE",
            "method": "REFUSE",
            "reason_code": reason_code,
            "estimate": None,
            "standard_error": None,
            "confidence_level": float(confidence_level),
            "interval_kind": None,
            "intervals": [],
            "operands": [numerator, denominator],
            "diagnostics": diagnostics,
            "message": "Inference refused by the preregistered diagnostic router",
        }

    if method == "DELTA":
        inference = infer_delta(
            ledger, score_names, expression, confidence_level=confidence_level
        )
    elif method == "FIELLER":
        inference = infer_fieller(
            ledger, score_names, expression, confidence_level=confidence_level
        )
    else:
        inference = infer_block_jackknife_ratio(
            ledger,
            score_names,
            numerator=numerator,
            denominator=denominator,
            confidence_level=confidence_level,
            block_count=thresholds.block_count,
        )

    if inference.status != "PASS":
        reason_code = inference.rejection_code or "E_SELECTED_METHOD_FAILED"
        method = "REFUSE"
    combined_diagnostics = {**diagnostics, **dict(inference.diagnostics)}
    return {
        "router_version": ROUTER_VERSION,
        "protocol_version": protocol_version,
        "status": inference.status,
        "method": method,
        "reason_code": reason_code,
        "estimate": inference.estimate,
        "standard_error": inference.standard_error,
        "confidence_level": (
            float(confidence_level)
            if inference.confidence_level is None
            else inference.confidence_level
        ),
        "interval_kind": inference.interval_kind,
        "intervals": [interval.as_dict() for interval in inference.intervals],
        "operands": list(inference.operands or (numerator, denominator)),
        "diagnostics": combined_diagnostics,
        "message": inference.message,
    }
