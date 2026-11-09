from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "p1-expression-v0"
ALLOWED_OPS = {"score", "constant", "add", "subtract", "multiply", "divide", "linear_combination"}


class ExpressionError(ValueError):
    """Expression failure with a stable machine-readable rejection code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class InferenceResult:
    status: str
    estimate: float | None
    standard_error: float | None
    rejection_code: str | None
    operands: tuple[str, ...]
    gradient: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "estimate": self.estimate,
            "standard_error": self.standard_error,
            "rejection_code": self.rejection_code,
            "operands": list(self.operands),
            "gradient": list(self.gradient),
        }


def load_expression(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_expression(data)
    return data


def validate_expression(node: Any) -> None:
    if not isinstance(node, dict):
        raise ExpressionError("E_SCHEMA", "Each expression node must be an object")
    op = node.get("op")
    if op not in ALLOWED_OPS:
        raise ExpressionError("E_SCHEMA", f"Unsupported operation: {op!r}")
    if op == "score":
        if not isinstance(node.get("name"), str) or not node["name"].strip():
            raise ExpressionError("E_SCHEMA", "score.name must be a non-empty string")
        return
    if op == "constant":
        value = node.get("value")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ExpressionError("E_SCHEMA", "constant.value must be finite")
        return
    if op == "linear_combination":
        terms = node.get("terms")
        if not isinstance(terms, list) or not terms:
            raise ExpressionError("E_SCHEMA", "linear_combination.terms must be non-empty")
        for term in terms:
            if not isinstance(term, dict) or set(term) != {"coefficient", "operand"}:
                raise ExpressionError("E_SCHEMA", "Each linear term needs coefficient and operand")
            coefficient = term["coefficient"]
            if not isinstance(coefficient, (int, float)) or not math.isfinite(float(coefficient)):
                raise ExpressionError("E_SCHEMA", "Linear coefficient must be finite")
            validate_expression(term["operand"])
        return
    args = node.get("args")
    if not isinstance(args, list) or len(args) != 2:
        raise ExpressionError("E_SCHEMA", f"{op}.args must contain exactly two nodes")
    validate_expression(args[0])
    validate_expression(args[1])


def minimal_operands(node: Mapping[str, Any]) -> list[str]:
    validate_expression(dict(node))
    ordered: list[str] = []

    def visit(current: Mapping[str, Any]) -> None:
        op = current["op"]
        if op == "score":
            name = current["name"]
            if name not in ordered:
                ordered.append(name)
        elif op == "linear_combination":
            for term in current["terms"]:
                visit(term["operand"])
        elif op not in {"constant"}:
            for child in current["args"]:
                visit(child)

    visit(node)
    return ordered


def value_gradient(
    node: Mapping[str, Any], means: Mapping[str, float], operand_index: Mapping[str, int],
    *, denominator_abs_floor: float, denominator_relative_floor: float,
) -> tuple[float, np.ndarray]:
    op = node["op"]
    size = len(operand_index)
    if op == "score":
        name = node["name"]
        if name not in means:
            raise ExpressionError("E_UNKNOWN_OPERAND", f"Missing operand {name!r}")
        value = float(means[name])
        if not math.isfinite(value):
            raise ExpressionError("E_NONFINITE", f"Operand {name!r} is non-finite")
        gradient = np.zeros(size)
        gradient[operand_index[name]] = 1.0
        return value, gradient
    if op == "constant":
        return float(node["value"]), np.zeros(size)
    if op == "linear_combination":
        value = 0.0
        gradient = np.zeros(size)
        for term in node["terms"]:
            child_value, child_gradient = value_gradient(
                term["operand"], means, operand_index,
                denominator_abs_floor=denominator_abs_floor,
                denominator_relative_floor=denominator_relative_floor,
            )
            coefficient = float(term["coefficient"])
            value += coefficient * child_value
            gradient += coefficient * child_gradient
        return value, gradient

    left_value, left_gradient = value_gradient(
        node["args"][0], means, operand_index,
        denominator_abs_floor=denominator_abs_floor,
        denominator_relative_floor=denominator_relative_floor,
    )
    right_value, right_gradient = value_gradient(
        node["args"][1], means, operand_index,
        denominator_abs_floor=denominator_abs_floor,
        denominator_relative_floor=denominator_relative_floor,
    )
    if op == "add":
        return left_value + right_value, left_gradient + right_gradient
    if op == "subtract":
        return left_value - right_value, left_gradient - right_gradient
    if op == "multiply":
        return left_value * right_value, right_value * left_gradient + left_value * right_gradient
    if op == "divide":
        if right_value == 0.0:
            raise ExpressionError("E_ZERO_DENOMINATOR", "Division denominator is exactly zero")
        scale = max(abs(left_value), abs(right_value), 1.0)
        threshold = max(denominator_abs_floor, denominator_relative_floor * scale)
        if abs(right_value) <= threshold:
            raise ExpressionError(
                "E_UNSTABLE_DENOMINATOR",
                f"Denominator {right_value:.6g} is below stability threshold {threshold:.6g}",
            )
        value = left_value / right_value
        gradient = (right_value * left_gradient - left_value * right_gradient) / right_value**2
        return value, gradient
    raise AssertionError(op)


def _value_gradient(
    node: Mapping[str, Any], means: Mapping[str, float], operand_index: Mapping[str, int],
    *, denominator_abs_floor: float, denominator_relative_floor: float,
) -> tuple[float, np.ndarray]:
    """Compatibility alias for pre-Week-06 snapshots; new code uses ``value_gradient``."""

    return value_gradient(
        node,
        means,
        operand_index,
        denominator_abs_floor=denominator_abs_floor,
        denominator_relative_floor=denominator_relative_floor,
    )


def infer_delta(
    ledger: np.ndarray,
    score_names: list[str] | tuple[str, ...],
    expression: Mapping[str, Any],
    *,
    denominator_abs_floor: float = 1.0e-14,
    denominator_relative_floor: float = 1.0e-10,
) -> InferenceResult:
    try:
        validate_expression(dict(expression))
        operands = minimal_operands(expression)
        array = np.asarray(ledger, dtype=float)
        if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] != len(score_names):
            raise ExpressionError("E_SHAPE", "Ledger must be n-by-len(score_names), with n >= 2")
        if not np.isfinite(array).all():
            raise ExpressionError("E_NONFINITE", "Ledger contains a non-finite value")
        if len(set(score_names)) != len(score_names):
            raise ExpressionError("E_SCHEMA", "score_names must be unique")
        missing = [name for name in operands if name not in score_names]
        if missing:
            raise ExpressionError("E_UNKNOWN_OPERAND", f"Missing operands: {missing}")
        selected = np.column_stack([array[:, score_names.index(name)] for name in operands])
        means = {name: float(selected[:, i].mean()) for i, name in enumerate(operands)}
        operand_index = {name: i for i, name in enumerate(operands)}
        value, gradient = value_gradient(
            expression, means, operand_index,
            denominator_abs_floor=denominator_abs_floor,
            denominator_relative_floor=denominator_relative_floor,
        )
        from .covariance import covariance_of_mean_hac
        covariance_of_mean = np.atleast_2d(covariance_of_mean_hac(selected))
        variance = float(gradient @ covariance_of_mean @ gradient)
        if variance < -1.0e-18:
            raise ExpressionError("E_NEGATIVE_VARIANCE", f"Propagated variance is negative: {variance}")
        standard_error = math.sqrt(max(0.0, variance))
        if not math.isfinite(value) or not math.isfinite(standard_error):
            raise ExpressionError("E_NONFINITE", "Inference result is non-finite")
        return InferenceResult("PASS", value, standard_error, None, tuple(operands), tuple(gradient.tolist()))
    except ExpressionError as exc:
        operands: tuple[str, ...] = ()
        try:
            operands = tuple(minimal_operands(expression))
        except Exception:
            pass
        return InferenceResult("REJECT", None, None, exc.code, operands, ())


RATIO_TOTAL_OVER_FLUX = {
    "schema_version": SCHEMA_VERSION,
    "name": "total_over_flux",
    "expression": {
        "op": "divide",
        "args": [{"op": "score", "name": "total"}, {"op": "score", "name": "flux"}],
    },
}

LINEAR_REACTION_INDEX = {
    "schema_version": SCHEMA_VERSION,
    "name": "reaction_index",
    "expression": {
        "op": "linear_combination",
        "terms": [
            {"coefficient": 1.0, "operand": {"op": "score", "name": "absorption"}},
            {"coefficient": 0.5, "operand": {"op": "score", "name": "fission"}},
            {"coefficient": -0.25, "operand": {"op": "score", "name": "total"}},
        ],
    },
}
