from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from .expression import (
    ExpressionError,
    infer_delta as expression_delta,
    minimal_operands,
    validate_expression,
    value_gradient,
)


INFERENCE_VERSION = "p1-inference-v1"


@dataclass(frozen=True)
class ConfidenceInterval:
    """One connected component of a confidence set.

    ``None`` represents an open infinite endpoint so serialized evidence remains
    valid JSON rather than relying on non-standard Infinity values.
    """

    lower: float | None
    upper: float | None

    def as_dict(self) -> dict[str, float | None]:
        return {"lower": self.lower, "upper": self.upper}


@dataclass(frozen=True)
class InferenceResult:
    status: str
    method: str
    estimate: float | None
    standard_error: float | None
    confidence_level: float | None
    interval_kind: str | None
    intervals: tuple[ConfidenceInterval, ...]
    rejection_code: str | None
    message: str
    operands: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": INFERENCE_VERSION,
            "status": self.status,
            "method": self.method,
            "estimate": self.estimate,
            "standard_error": self.standard_error,
            "confidence_level": self.confidence_level,
            "interval_kind": self.interval_kind,
            "intervals": [interval.as_dict() for interval in self.intervals],
            "rejection_code": self.rejection_code,
            "message": self.message,
            "operands": list(self.operands),
            "diagnostics": _json_safe(dict(self.diagnostics)),
        }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def refuse_inference(
    code: str,
    message: str,
    *,
    method: str = "REFUSE",
    operands: Sequence[str] = (),
    diagnostics: Mapping[str, Any] | None = None,
) -> InferenceResult:
    """Return a stable, machine-readable refusal instead of a forced estimate."""

    if not isinstance(code, str) or not code.startswith("E_"):
        raise ValueError("Refusal codes must be stable strings beginning with E_")
    return InferenceResult(
        status="REFUSE",
        method=method,
        estimate=None,
        standard_error=None,
        confidence_level=None,
        interval_kind=None,
        intervals=(),
        rejection_code=code,
        message=message,
        operands=tuple(operands),
        diagnostics={} if diagnostics is None else dict(diagnostics),
    )


def _validate_confidence_level(confidence_level: float) -> float:
    level = float(confidence_level)
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        raise ExpressionError(
            "E_INVALID_CONFIDENCE_LEVEL", "confidence_level must be finite and between 0 and 1"
        )
    return level


def _prepare_ledger(
    ledger: np.ndarray,
    score_names: Sequence[str],
    expression: Mapping[str, Any],
    *,
    min_batches: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    validate_expression(dict(expression))
    operands = tuple(minimal_operands(expression))
    array = np.asarray(ledger, dtype=float)
    if array.ndim != 2 or array.shape[1] != len(score_names):
        raise ExpressionError("E_SHAPE", "Ledger must be n-by-len(score_names)")
    if array.shape[0] < min_batches:
        raise ExpressionError(
            "E_INSUFFICIENT_BATCHES", f"At least {min_batches} batches are required"
        )
    if not np.isfinite(array).all():
        raise ExpressionError("E_NONFINITE", "Ledger contains a non-finite value")
    if len(set(score_names)) != len(score_names):
        raise ExpressionError("E_SCHEMA", "score_names must be unique")
    missing = [name for name in operands if name not in score_names]
    if missing:
        raise ExpressionError("E_UNKNOWN_OPERAND", f"Missing operands: {missing}")
    selected = np.column_stack([array[:, score_names.index(name)] for name in operands])
    return selected, operands


def _evaluate_from_means(
    means_array: np.ndarray,
    operands: Sequence[str],
    expression: Mapping[str, Any],
    *,
    denominator_abs_floor: float,
    denominator_relative_floor: float,
) -> float:
    means = {name: float(means_array[index]) for index, name in enumerate(operands)}
    operand_index = {name: index for index, name in enumerate(operands)}
    value, _ = value_gradient(
        expression,
        means,
        operand_index,
        denominator_abs_floor=denominator_abs_floor,
        denominator_relative_floor=denominator_relative_floor,
    )
    if not math.isfinite(value):
        raise ExpressionError("E_NONFINITE", "Expression estimate is non-finite")
    return float(value)


def _student_critical(confidence_level: float, degrees_of_freedom: int) -> float:
    critical = float(student_t.ppf(0.5 + confidence_level / 2.0, degrees_of_freedom))
    if not math.isfinite(critical):
        raise ExpressionError("E_NONFINITE", "Critical value is non-finite")
    return critical


def infer_delta(
    ledger: np.ndarray,
    score_names: Sequence[str],
    expression: Mapping[str, Any],
    *,
    confidence_level: float = 0.95,
    denominator_abs_floor: float = 1.0e-14,
    denominator_relative_floor: float = 1.0e-10,
) -> InferenceResult:
    """Correlated first-order delta inference with a Student-t interval."""

    try:
        level = _validate_confidence_level(confidence_level)
        selected, operands = _prepare_ledger(
            ledger, score_names, expression, min_batches=2
        )
        result = expression_delta(
            ledger,
            list(score_names),
            expression,
            denominator_abs_floor=denominator_abs_floor,
            denominator_relative_floor=denominator_relative_floor,
        )
        if result.status != "PASS":
            return refuse_inference(
                result.rejection_code or "E_DELTA_FAILED",
                "Correlated delta inference rejected the input",
                method="DELTA",
                operands=result.operands,
            )
        critical = _student_critical(level, selected.shape[0] - 1)
        half_width = critical * float(result.standard_error)
        estimate = float(result.estimate)
        return InferenceResult(
            status="PASS",
            method="DELTA",
            estimate=estimate,
            standard_error=float(result.standard_error),
            confidence_level=level,
            interval_kind="bounded",
            intervals=(ConfidenceInterval(estimate - half_width, estimate + half_width),),
            rejection_code=None,
            message="Correlated delta inference completed",
            operands=tuple(operands),
            diagnostics={
                "n_batches": int(selected.shape[0]),
                "degrees_of_freedom": int(selected.shape[0] - 1),
                "critical_value": critical,
                "gradient": list(result.gradient),
            },
        )
    except ExpressionError as exc:
        operands: tuple[str, ...] = ()
        try:
            operands = tuple(minimal_operands(expression))
        except Exception:
            pass
        return refuse_inference(exc.code, str(exc), method="DELTA", operands=operands)


def infer_jackknife(
    ledger: np.ndarray,
    score_names: Sequence[str],
    expression: Mapping[str, Any],
    *,
    confidence_level: float = 0.95,
    denominator_abs_floor: float = 1.0e-14,
    denominator_relative_floor: float = 1.0e-10,
) -> InferenceResult:
    """Delete-one-batch jackknife with explicit unstable-replicate refusal."""

    try:
        level = _validate_confidence_level(confidence_level)
        selected, operands = _prepare_ledger(
            ledger, score_names, expression, min_batches=3
        )
        n_batches = selected.shape[0]
        full_estimate = _evaluate_from_means(
            selected.mean(axis=0),
            operands,
            expression,
            denominator_abs_floor=denominator_abs_floor,
            denominator_relative_floor=denominator_relative_floor,
        )
        total = selected.sum(axis=0)
        leave_one_out: list[float] = []
        for batch_index in range(n_batches):
            means = (total - selected[batch_index]) / (n_batches - 1)
            try:
                estimate = _evaluate_from_means(
                    means,
                    operands,
                    expression,
                    denominator_abs_floor=denominator_abs_floor,
                    denominator_relative_floor=denominator_relative_floor,
                )
            except ExpressionError as exc:
                raise ExpressionError(
                    "E_JACKKNIFE_UNSTABLE_REPLICATE",
                    f"Delete-one batch {batch_index} failed with {exc.code}",
                ) from exc
            leave_one_out.append(estimate)

        replicates = np.asarray(leave_one_out, dtype=float)
        replicate_mean = float(replicates.mean())
        bias_corrected = float(n_batches * full_estimate - (n_batches - 1) * replicate_mean)
        variance = float((n_batches - 1) / n_batches * np.square(replicates - replicate_mean).sum())
        if variance < -1.0e-18:
            raise ExpressionError("E_NEGATIVE_VARIANCE", "Jackknife variance is negative")
        standard_error = math.sqrt(max(0.0, variance))
        critical = _student_critical(level, n_batches - 1)
        half_width = critical * standard_error
        return InferenceResult(
            status="PASS",
            method="JACKKNIFE",
            estimate=bias_corrected,
            standard_error=standard_error,
            confidence_level=level,
            interval_kind="bounded",
            intervals=(
                ConfidenceInterval(bias_corrected - half_width, bias_corrected + half_width),
            ),
            rejection_code=None,
            message="Delete-one-batch jackknife completed",
            operands=tuple(operands),
            diagnostics={
                "n_batches": int(n_batches),
                "full_sample_estimate": full_estimate,
                "jackknife_replicate_mean": replicate_mean,
                "replicate_min": float(replicates.min()),
                "replicate_max": float(replicates.max()),
                "critical_value": critical,
            },
        )
    except ExpressionError as exc:
        operands: tuple[str, ...] = ()
        try:
            operands = tuple(minimal_operands(expression))
        except Exception:
            pass
        return refuse_inference(exc.code, str(exc), method="JACKKNIFE", operands=operands)


def _direct_ratio_operands(expression: Mapping[str, Any]) -> tuple[str, str]:
    validate_expression(dict(expression))
    if expression.get("op") != "divide":
        raise ExpressionError(
            "E_UNSUPPORTED_FIELLER_EXPRESSION",
            "Fieller supports only a direct ratio of two score operands",
        )
    numerator, denominator = expression["args"]
    if numerator.get("op") != "score" or denominator.get("op") != "score":
        raise ExpressionError(
            "E_UNSUPPORTED_FIELLER_EXPRESSION",
            "Fieller supports only a direct ratio of two score operands",
        )
    return str(numerator["name"]), str(denominator["name"])


def infer_fieller(
    ledger: np.ndarray,
    score_names: Sequence[str],
    expression: Mapping[str, Any],
    *,
    confidence_level: float = 0.95,
) -> InferenceResult:
    """Fieller confidence set for a direct ratio of correlated batch means.

    Unbounded, disjoint, and all-real confidence sets are successful diagnostic
    outcomes. They are not silently collapsed to a finite symmetric interval.
    """

    try:
        level = _validate_confidence_level(confidence_level)
        numerator_name, denominator_name = _direct_ratio_operands(expression)
        selected, operands = _prepare_ledger(
            ledger, score_names, expression, min_batches=3
        )
        numerator_index = operands.index(numerator_name)
        denominator_index = operands.index(denominator_name)
        n_batches = selected.shape[0]
        means = selected.mean(axis=0)
        from .covariance import bartlett_bandwidth, covariance_of_mean_hac
        covariance_of_mean = np.atleast_2d(covariance_of_mean_hac(selected))
        mean_x = float(means[numerator_index])
        mean_y = float(means[denominator_index])
        var_x = float(covariance_of_mean[numerator_index, numerator_index])
        var_y = float(covariance_of_mean[denominator_index, denominator_index])
        cov_xy = float(covariance_of_mean[numerator_index, denominator_index])
        if min(var_x, var_y) < -1.0e-18:
            raise ExpressionError("E_NEGATIVE_VARIANCE", "Observed covariance has negative variance")

        critical = _student_critical(level, n_batches - 1)
        q = critical**2
        coefficient_a = mean_y**2 - q * var_y
        coefficient_b = -2.0 * (mean_x * mean_y - q * cov_xy)
        coefficient_c = mean_x**2 - q * var_x
        discriminant = coefficient_b**2 - 4.0 * coefficient_a * coefficient_c
        scale = max(
            abs(coefficient_a), abs(coefficient_b), abs(coefficient_c), abs(discriminant), 1.0
        )
        tolerance = 128.0 * np.finfo(float).eps * scale

        interval_kind: str
        intervals: tuple[ConfidenceInterval, ...]
        if abs(coefficient_a) <= tolerance:
            if abs(coefficient_b) <= tolerance:
                if coefficient_c <= tolerance:
                    interval_kind = "all_real"
                    intervals = (ConfidenceInterval(None, None),)
                else:
                    raise ExpressionError("E_FIELLER_EMPTY_SET", "Fieller confidence set is empty")
            else:
                boundary = -coefficient_c / coefficient_b
                interval_kind = "half_line"
                if coefficient_b > 0.0:
                    intervals = (ConfidenceInterval(None, float(boundary)),)
                else:
                    intervals = (ConfidenceInterval(float(boundary), None),)
        elif discriminant < -tolerance:
            if coefficient_a < 0.0:
                interval_kind = "all_real"
                intervals = (ConfidenceInterval(None, None),)
            else:
                raise ExpressionError("E_FIELLER_EMPTY_SET", "Fieller confidence set is empty")
        else:
            root_discriminant = math.sqrt(max(0.0, discriminant))
            roots = sorted(
                (
                    (-coefficient_b - root_discriminant) / (2.0 * coefficient_a),
                    (-coefficient_b + root_discriminant) / (2.0 * coefficient_a),
                )
            )
            if coefficient_a > 0.0:
                interval_kind = "bounded"
                intervals = (ConfidenceInterval(float(roots[0]), float(roots[1])),)
            else:
                interval_kind = "disjoint"
                intervals = (
                    ConfidenceInterval(None, float(roots[0])),
                    ConfidenceInterval(float(roots[1]), None),
                )

        estimate = None if mean_y == 0.0 else mean_x / mean_y
        denominator_se = math.sqrt(max(0.0, var_y))
        denominator_t = math.inf if denominator_se == 0.0 else abs(mean_y) / denominator_se
        return InferenceResult(
            status="PASS",
            method="FIELLER",
            estimate=estimate,
            standard_error=None,
            confidence_level=level,
            interval_kind=interval_kind,
            intervals=intervals,
            rejection_code=None,
            message=f"Fieller confidence set completed ({interval_kind})",
            operands=tuple(operands),
            diagnostics={
                "n_batches": int(n_batches),
                "degrees_of_freedom": int(n_batches - 1),
                "critical_value": critical,
                "covariance_estimator": "hac_bartlett",
                "hac_bandwidth": int(bartlett_bandwidth(n_batches)),
                "numerator_mean": mean_x,
                "denominator_mean": mean_y,
                "numerator_variance_of_mean": var_x,
                "denominator_variance_of_mean": var_y,
                "covariance_of_means": cov_xy,
                "denominator_t_statistic": denominator_t,
                "quadratic_a": coefficient_a,
                "quadratic_b": coefficient_b,
                "quadratic_c": coefficient_c,
                "discriminant": discriminant,
            },
        )
    except ExpressionError as exc:
        operands: tuple[str, ...] = ()
        try:
            operands = tuple(minimal_operands(expression))
        except Exception:
            pass
        return refuse_inference(exc.code, str(exc), method="FIELLER", operands=operands)
