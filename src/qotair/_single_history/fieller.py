"""Dimension-aware Fieller set classification.

The historical router calls infer_fieller for diagnostic sets. That classifier
compares a, b, c (quadratic in a common score scale) and D (quartic) against one
tolerance 128*eps*max(|a|,|b|,|c|,|D|,1). This module studentizes first and
uses coefficient-specific tolerances. It is a diagnostic overlay and a property
test; it does not retune engineering-release gates.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


EPS_MULT = 128.0
MACHINE_EPS = float(np.finfo(np.float64).eps)


class FiellerScaleError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _tol(value: float) -> float:
    return EPS_MULT * MACHINE_EPS * max(1.0, abs(float(value)))


def studentize_ratio_moments(
    mean_x: float,
    mean_y: float,
    var_x: float,
    var_y: float,
    cov_xy: float,
) -> dict[str, float]:
    if not all(math.isfinite(v) for v in (mean_x, mean_y, var_x, var_y, cov_xy)):
        raise FiellerScaleError("E_FIELLER_NONFINITE", "non-finite Fieller moments")
    if var_x < -1e-18 or var_y < -1e-18:
        raise FiellerScaleError("E_NEGATIVE_VARIANCE", "negative variance")
    se_x = math.sqrt(max(0.0, var_x))
    se_y = math.sqrt(max(0.0, var_y))
    if se_x == 0.0 or se_y == 0.0:
        raise FiellerScaleError(
            "E_FIELLER_DEGENERATE_VARIANCE",
            "cannot studentize a ratio when a score standard error is zero",
        )
    ux = mean_x / se_x
    uy = mean_y / se_y
    rho = cov_xy / (se_x * se_y)
    rho = float(min(1.0, max(-1.0, rho)))
    return {"ux": ux, "uy": uy, "rho": rho, "se_x": se_x, "se_y": se_y}


def dimensionless_coefficients(ux: float, uy: float, rho: float, q: float) -> dict[str, float]:
    """a,b,c,D after studentizing. a,b,c are O(q); D is O(q^2)."""
    a = uy * uy - q
    b = -2.0 * (ux * uy - q * rho)
    c = ux * ux - q
    disc = b * b - 4.0 * a * c
    return {"a": a, "b": b, "c": c, "D": disc, "q": q}


def classify_dimensionless(a: float, b: float, c: float, disc: float) -> dict[str, Any]:
    """Separate tolerances: a,b,c vs D. Does not claim quadratic homogeneity of D."""
    tol_a, tol_b, tol_c, tol_d = _tol(a), _tol(b), _tol(c), _tol(disc)
    if abs(a) <= tol_a:
        if abs(b) <= tol_b:
            if c <= tol_c:
                return {
                    "interval_kind": "all_real",
                    "reason_code": "R_FIELLER_ALL_REAL",
                    "engineering_finite": False,
                }
            return {
                "interval_kind": "empty",
                "reason_code": "E_FIELLER_EMPTY_SET",
                "engineering_finite": False,
            }
        boundary = -c / b
        kind = "half_line"
        return {
            "interval_kind": kind,
            "reason_code": "R_FIELLER_HALF_LINE",
            "engineering_finite": False,
            "boundary": float(boundary),
            "opens_right": b > 0.0,
        }
    if disc < -tol_d:
        if a < 0.0:
            return {
                "interval_kind": "all_real",
                "reason_code": "R_FIELLER_ALL_REAL",
                "engineering_finite": False,
            }
        return {
            "interval_kind": "empty",
            "reason_code": "E_FIELLER_EMPTY_SET",
            "engineering_finite": False,
        }
    root = math.sqrt(max(0.0, disc))
    left = (-b - root) / (2.0 * a)
    right = (-b + root) / (2.0 * a)
    lo, hi = (left, right) if left <= right else (right, left)
    if a > 0.0:
        return {
            "interval_kind": "bounded",
            "reason_code": "R_FIELLER_BOUNDED_DIAGNOSTIC",
            "engineering_finite": False,
            "roots": [float(lo), float(hi)],
        }
    return {
        "interval_kind": "disjoint",
        "reason_code": "R_FIELLER_DISJOINT",
        "engineering_finite": False,
        "roots": [float(lo), float(hi)],
    }


def classify_fieller_moments(
    mean_x: float,
    mean_y: float,
    var_x: float,
    var_y: float,
    cov_xy: float,
    *,
    q: float,
) -> dict[str, Any]:
    """Studentized Fieller classification. Bounded sets remain diagnostic, not engineering release."""
    try:
        st = studentize_ratio_moments(mean_x, mean_y, var_x, var_y, cov_xy)
    except FiellerScaleError as exc:
        return {
            "interval_kind": "numerical_failure",
            "reason_code": exc.code,
            "engineering_finite": False,
            "message": str(exc),
        }
    coeff = dimensionless_coefficients(st["ux"], st["uy"], st["rho"], q)
    classified = classify_dimensionless(coeff["a"], coeff["b"], coeff["c"], coeff["D"])
    return {**classified, "studentized": st, "coefficients": coeff}


def physical_intervals(classified: dict[str, Any]) -> list[dict[str, float | None]]:
    """Map dimensionless roots back to the original ratio units: R = R* * se_x / se_y."""
    kind = classified.get("interval_kind")
    st = classified.get("studentized") or {}
    se_x = float(st.get("se_x") or 1.0)
    se_y = float(st.get("se_y") or 1.0)
    scale = se_x / se_y if se_y else 1.0
    if kind == "bounded":
        lo, hi = classified["roots"]
        return [{"lower": lo * scale, "upper": hi * scale}]
    if kind == "disjoint":
        lo, hi = classified["roots"]
        return [{"lower": None, "upper": lo * scale}, {"lower": hi * scale, "upper": None}]
    if kind == "half_line":
        boundary = float(classified["boundary"]) * scale
        if classified.get("opens_right"):
            return [{"lower": None, "upper": boundary}]
        return [{"lower": boundary, "upper": None}]
    if kind in {"all_real"}:
        return [{"lower": None, "upper": None}]
    return []


def scale_moments(mean_x, mean_y, var_x, var_y, cov_xy, sx: float, sy: float):
    return (
        mean_x * sx,
        mean_y * sy,
        var_x * sx * sx,
        var_y * sy * sy,
        cov_xy * sx * sy,
    )
