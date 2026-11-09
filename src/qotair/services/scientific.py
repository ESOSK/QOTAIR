"""Query compilation and historical single-history analysis."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .._single_history import compiler, envelope as production, expression, router


def _thresholds():
    path = Path(router.__file__).with_name("thresholds.json")
    return router.RouterThresholds.from_mapping(json.loads(path.read_text(encoding="utf-8")))


def compile_query(query: Mapping[str, Any], available_operands: Sequence[str] | None = None) -> Any:
    return compiler.compile_query(query, available_operands=available_operands)


def query_operands(query: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(expression.minimal_operands(query["expression"]))


# Historical product guard, separate from statistical inference.
PAPER_ENGINEERING_ABS_ESTIMATE_CAP = 10.0


def _refuse_from_exception(exc: BaseException) -> dict[str, Any]:
    code = str(getattr(exc, "code", None) or "E_SELECTED_METHOD_FAILED")
    return {
        "status": "REFUSE",
        "method": "REFUSE",
        "reason_code": code,
        "estimate": None,
        "standard_error": None,
        "intervals": [],
        "engineering_finite_release": False,
        "message": str(exc),
    }


def apply_product_magnitude_guard(record: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse engineering-finite product release when |estimate| exceeds the product cap.

    Hashed ``route_production_ratio`` is not called and is not modified.
    Statistical status/reason are preserved; only engineering_finite_release is cleared.
    """

    rec = dict(record)
    est = rec.get("estimate")
    abs_est = None if est is None else abs(float(est))
    blocked = bool(rec.get("engineering_finite_release")) and abs_est is not None and abs_est > PAPER_ENGINEERING_ABS_ESTIMATE_CAP
    rec["product_magnitude_guard"] = {
        "cap": PAPER_ENGINEERING_ABS_ESTIMATE_CAP,
        "blocked": blocked,
        "abs_estimate": abs_est,
        "note": "Historical product guard.",
    }
    if not blocked:
        return rec
    rec["statistical_engineering_finite_release"] = rec.get("engineering_finite_release")
    rec["engineering_finite_release"] = False
    rec["product_reason_code"] = "R_PRODUCT_MAGNITUDE_CAP"
    rec["message"] = (
        f"|estimate|={abs_est:.6g} exceeds product magnitude cap "
        f"{PAPER_ENGINEERING_ABS_ESTIMATE_CAP:g}; not an engineering-finite product release"
    )
    return rec


def paper_engineering_envelope(
    ledger: np.ndarray,
    score_names: Sequence[str],
    *,
    numerator: str,
    denominator: str,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Paper I_env = hull(conservative HAC-delta ∪ block jackknife). Not the product router.

    Frozen production may raise on an exact-zero jackknife denominator. That is
    REFUSE, not a crash, and thresholds are not retuned.
    """

    expression = {
        "op": "divide",
        "args": [
            {"op": "score", "name": numerator},
            {"op": "score", "name": denominator},
        ],
    }
    try:
        return json_safe(
            apply_product_magnitude_guard(
                production.infer_production_expression(
                    np.asarray(ledger, dtype=float),
                    list(score_names),
                    expression,
                    thresholds=_thresholds(),
                    confidence_level=float(confidence_level),
                )
            )
        )
    except Exception as exc:
        if getattr(exc, "code", None) in {"E_ZERO_DENOMINATOR", "E_NONFINITE", "E_NONFINITE_INPUT"}:
            return _refuse_from_exception(exc)
        raise


def infer_query(
    query: Mapping[str, Any],
    ledger: np.ndarray,
    score_names: Sequence[str],
    *,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Route one query through the frozen ratio or production expression path."""

    class_name = str(query.get("expression_class", ""))
    if class_name == "ratio":
        expression = query.get("expression", {})
        args = expression.get("args", []) if isinstance(expression, Mapping) else []
        if expression.get("op") == "divide" and len(args) == 2 and all(
            isinstance(item, Mapping) and item.get("op") == "score" for item in args
        ):
            return router.route_ratio_inference(
                np.asarray(ledger, dtype=float),
                list(score_names),
                numerator=str(args[0]["name"]),
                denominator=str(args[1]["name"]),
                thresholds=_thresholds(),
                confidence_level=float(confidence_level),
            )
    try:
        return production.infer_production_expression(
            np.asarray(ledger, dtype=float),
            list(score_names),
            query["expression"],
            thresholds=_thresholds(),
            confidence_level=float(confidence_level),
        )
    except Exception as exc:
        if getattr(exc, "code", None) in {"E_ZERO_DENOMINATOR", "E_NONFINITE", "E_NONFINITE_INPUT"}:
            return _refuse_from_exception(exc)
        raise


def json_safe(value: Any) -> Any:
    """Convert NumPy scalars and frozen interval objects to JSON-safe values."""

    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return json_safe(value.as_dict())
    return value
