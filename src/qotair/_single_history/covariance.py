from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from scipy.linalg import qr


MIN_ROWS = 32
STRUCTURAL_TOL = 1e-12
MIN_ESS = 8.0
MAX_WINDOW_SPREAD = 1.0
MAX_BLOCK_DIFFERENCE = 1.5
MAX_PSD_RELATIVE = 1e-8


class DependenceError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _canonical(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype="<f8"))


def _hash(values: np.ndarray) -> str:
    return hashlib.sha256(_canonical(values).tobytes(order="C")).hexdigest()


def _bandwidth(n: int) -> int:
    maximum = max(1, min(40, n // 4))
    return min(maximum, max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))))


def bartlett_bandwidth(n: int) -> int:
    """Public Bartlett lag: min(cap, max(1, floor(4*(n/100)^(2/9)))), cap=min(40, floor(n/4))."""
    if int(n) < 2:
        raise DependenceError("E_DEPENDENCE_TOO_SHORT", "at least two batches are required")
    return _bandwidth(int(n))


def covariance_of_mean_sample(values: np.ndarray) -> np.ndarray:
    """Historical iid estimator: sample covariance (ddof=1) divided by n. Audit only."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    n = array.shape[0]
    if n < 2 or not np.isfinite(array).all():
        raise DependenceError("E_DEPENDENCE_INPUT", "finite series with n>=2 required")
    return np.atleast_2d(np.cov(array, rowvar=False, ddof=1)) / n


def covariance_of_mean_hac(values: np.ndarray, *, bandwidth: int | None = None) -> np.ndarray:
    """Canonical covariance of the batch-mean vector: Bartlett long-run variance / n.

    Omega = Gamma0 + sum_{l=1}^L w_l (Gamma_l + Gamma_l^T), w_l = 1 - l/(L+1).
    This is the production estimator for DELTA, Fieller, T_den, and router SE.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[0] < 2 or not np.isfinite(array).all():
        raise DependenceError("E_DEPENDENCE_INPUT", "finite series with n>=2 required")
    n = array.shape[0]
    lag = bartlett_bandwidth(n) if bandwidth is None else max(1, int(bandwidth))
    lag = min(lag, max(1, n // 4 if n >= 4 else 1))
    centered = array - array.mean(axis=0, keepdims=True)
    omega = _lrcov(centered, lag, "hac_bartlett")
    return omega / n


def _windows(n: int, base: int) -> list[int]:
    maximum = max(1, min(40, n // 4))
    return sorted({max(1, min(maximum, int(round(base * factor)))) for factor in (0.5, 1.0, 1.5, 2.0)})


def _autocovariances(centered: np.ndarray, maximum_lag: int) -> list[np.ndarray]:
    n = centered.shape[0]
    return [centered[: n - lag].T @ centered[lag:] / n for lag in range(maximum_lag + 1)]


def _lrcov(centered: np.ndarray, bandwidth: int, kind: str) -> np.ndarray:
    gammas = _autocovariances(centered, bandwidth)
    value = gammas[0].copy()
    for lag in range(1, bandwidth + 1):
        weight = 1.0 if kind == "gamma_rectangular" else 1.0 - lag / (bandwidth + 1.0)
        value += weight * (gammas[lag] + gammas[lag].T)
    return (value + value.T) / 2.0


def _acf_and_ess(centered: np.ndarray) -> tuple[list[list[float]], list[float]]:
    n, d = centered.shape
    maximum_lag = min(20, n // 4)
    acfs: list[list[float]] = []
    ess: list[float] = []
    for column in range(d):
        x = centered[:, column]
        variance = float(x @ x / n)
        if not np.isfinite(variance) or variance <= np.finfo(float).tiny:
            raise DependenceError("E_DEPENDENCE_CONSTANT_DIRECTION", "constant or numerically null direction")
        rho = [1.0]
        for lag in range(1, maximum_lag + 1):
            rho.append(float((x[:-lag] @ x[lag:] / n) / variance))
        tau = 1.0
        lag = 1
        while lag <= maximum_lag:
            pair = rho[lag]
            if lag + 1 <= maximum_lag:
                pair += rho[lag + 1]
            if pair <= 0.0:
                break
            tau += 2.0 * pair
            lag += 2
        tau = max(1.0, tau)
        acfs.append(rho)
        ess.append(float(min(n, max(1.0, n / tau))))
    return acfs, ess


def _rank_reduction(centered: np.ndarray) -> tuple[np.ndarray, int, int, float, float, list[float]]:
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    if singular.size == 0 or singular[0] <= np.finfo(float).tiny:
        raise DependenceError("E_DEPENDENCE_CONSTANT_DIRECTION", "all centered directions are null")
    numerical_rank = int(np.sum(singular / singular[0] > STRUCTURAL_TOL))
    if numerical_rank == 0:
        raise DependenceError("E_DEPENDENCE_CONSTANT_DIRECTION", "all centered directions are null")
    _, r, _ = qr(centered, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(r))
    qr_rank = int(np.sum(diagonal / diagonal[0] > STRUCTURAL_TOL)) if diagonal.size else 0
    basis = vt[:numerical_rank].T
    reduced = centered @ basis
    reconstructed = reduced @ basis.T
    reconstruction_error = float(np.linalg.norm(centered - reconstructed) / np.linalg.norm(centered))
    singular_ratio = float(singular[-1] / singular[0])
    return reduced, numerical_rank, qr_rank, reconstruction_error, singular_ratio, singular.astype(float).tolist()


def _effective_rank(eigenvalues: np.ndarray) -> float:
    positive = np.clip(eigenvalues, 0.0, None)
    total = float(positive.sum())
    if total <= np.finfo(float).tiny:
        return 0.0
    probabilities = positive[positive > 0.0] / total
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def _psd_error(eigenvalues: np.ndarray, matrix: np.ndarray) -> tuple[float, float]:
    negative = np.minimum(eigenvalues, 0.0)
    error = float(np.linalg.norm(negative))
    relative = error / max(float(np.linalg.norm(matrix, ord="fro")), np.finfo(float).tiny)
    return error, float(relative)


def _window_spread(centered: np.ndarray, kind: str, base: int) -> float:
    reference = _lrcov(centered, base, kind)
    denominator = max(float(np.linalg.norm(reference, ord="fro")), np.finfo(float).tiny)
    return float(max(np.linalg.norm(_lrcov(centered, width, kind) - reference, ord="fro") / denominator for width in _windows(len(centered), base)))


def _block_difference(centered: np.ndarray, kind: str, base: int) -> float:
    full = _lrcov(centered, base, kind)
    denominator = max(float(np.linalg.norm(full, ord="fro")), np.finfo(float).tiny)
    blocks = np.array_split(centered, 2)
    differences = []
    for block in blocks:
        width = min(_bandwidth(len(block)), max(1, len(block) // 4))
        differences.append(float(np.linalg.norm(_lrcov(block, width, kind) - full, ord="fro") / denominator))
    return max(differences)


def _estimator(centered: np.ndarray, kind: str, base: int) -> dict[str, Any]:
    matrix = _lrcov(centered, base, kind)
    eigenvalues = np.linalg.eigvalsh(matrix)
    psd_error, psd_relative = _psd_error(eigenvalues, matrix)
    return {
        "name": kind,
        "bandwidth": base,
        "bandwidth_candidates": _windows(len(centered), base),
        "matrix": matrix.astype(float).tolist(),
        "diagonal": np.diag(matrix).astype(float).tolist(),
        "eigenvalues": eigenvalues.astype(float).tolist(),
        "effective_rank": _effective_rank(eigenvalues),
        "psd_error_fro": psd_error,
        "psd_error_relative": psd_relative,
        "window_relative_spread": _window_spread(centered, kind, base),
        "block_relative_difference": _block_difference(centered, kind, base)
    }


def analyze_dependence(values: np.ndarray, *, case_id: str) -> dict[str, Any]:
    array = _canonical(values)
    if array.ndim != 2 or array.shape[1] < 1 or not np.isfinite(array).all():
        raise DependenceError("E_DEPENDENCE_INPUT", "input must be a finite two-dimensional matrix")
    if array.shape[0] < MIN_ROWS:
        raise DependenceError("E_DEPENDENCE_TOO_SHORT", f"at least {MIN_ROWS} batches are required")
    centered = array - array.mean(axis=0, keepdims=True)
    reduced, numerical_rank, qr_rank, reconstruction_error, singular_ratio, singular_values = _rank_reduction(centered)
    structural = numerical_rank < array.shape[1] and qr_rank == numerical_rank and reconstruction_error <= STRUCTURAL_TOL
    acf, ess = _acf_and_ess(reduced)
    base = _bandwidth(len(array))
    estimators = [_estimator(reduced, name, base) for name in ("gamma_rectangular", "hac_bartlett")]
    hac = estimators[1]
    unstable = (
        min(ess) < MIN_ESS
        or hac["window_relative_spread"] > MAX_WINDOW_SPREAD
        or hac["block_relative_difference"] > MAX_BLOCK_DIFFERENCE
        or hac["psd_error_relative"] > MAX_PSD_RELATIVE
    )
    if unstable:
        decision = "REFUSE_UNSTABLE"
        reason = "E_DEPENDENCE_UNRESOLVED" if hac["psd_error_relative"] <= MAX_PSD_RELATIVE else "E_DEPENDENCE_NONPSD"
    elif structural:
        decision = "ACCEPT_STRUCTURAL_SUBSPACE"
        reason = "STRUCTURAL_REDUNDANCY_IDENTIFIED"
    else:
        decision = "ACCEPT_STABLE"
        reason = "STABLE_DEPENDENCE_DIAGNOSTICS"
    return {
        "schema_version": "p1-long-run-covariance-v1",
        "case_id": case_id,
        "n_batches": int(array.shape[0]),
        "n_variables": int(array.shape[1]),
        "analyzed_subspace_dimension": numerical_rank,
        "values_sha256": _hash(array),
        "numerical_rank": numerical_rank,
        "qr_rank": qr_rank,
        "structural_redundancy": structural,
        "structural_reconstruction_relative_error": reconstruction_error,
        "smallest_to_largest_singular_ratio": singular_ratio,
        "singular_values": singular_values,
        "acf": acf,
        "ess": ess,
        "min_ess": float(min(ess)),
        "max_abs_acf_lag1": float(max(abs(row[1]) for row in acf)),
        "decision_estimator": "hac_bartlett",
        "estimators": estimators,
        "decision": decision,
        "reason_code": reason,
        "regularization_used_for_decision": False
    }
