"""Confidence sets from independent run means."""

import math

import numpy as np
from scipy.stats import f, t


def invert(mean, cov, q, linear=False):
    n, d = map(float, mean)
    v = np.asarray(cov, float)
    if not np.isfinite(v).all() or not np.isfinite(mean).all():
        return {"valid": False, "kind": "invalid", "intervals": [], "reason": "E_NONFINITE_MOMENTS"}
    if linear:
        if v[0, 0] <= 0:
            return {"valid": False, "kind": "invalid", "intervals": [], "reason": "E_NO_VARIANCE_INFORMATION"}
        h = math.sqrt(q * v[0, 0])
        return {"valid": True, "kind": "bounded", "intervals": [[n - h, n + h]]}
    A = d * d - q * v[1, 1]
    B = -2 * n * d + 2 * q * v[0, 1]
    C = n * n - q * v[0, 0]
    discr = B * B - 4 * A * C
    if discr < 0:
        return (
            {"valid": True, "kind": "all_real", "intervals": [[None, None]]}
            if A < 0
            else {"valid": False, "kind": "invalid", "intervals": [], "reason": "E_EMPTY_NUMERICAL_SET"}
        )
    if A == 0:
        if B == 0:
            return (
                {"valid": True, "kind": "all_real", "intervals": [[None, None]]}
                if C <= 0
                else {"valid": False, "kind": "invalid", "intervals": [], "reason": "E_EMPTY_SET"}
            )
        root = -C / B
        return {"valid": True, "kind": "half_line", "intervals": [[None, root] if B > 0 else [root, None]]}
    # Stable quadratic roots; product recovers the root vulnerable to cancellation.
    rootdisc = math.sqrt(max(discr, 0.0))
    u = -0.5 * (B + math.copysign(rootdisc, B))
    roots = sorted([u / A, C / u]) if u != 0 else [0.0, 0.0]
    if A > 0:
        return {"valid": True, "kind": "bounded", "intervals": [roots]}
    return {"valid": True, "kind": "disjoint", "intervals": [[None, roots[0]], [roots[1], None]]}


def contains(s, truth):
    if truth is None or not s["valid"]:
        return None
    return any((a is None or a <= truth) and (b is None or truth <= b) for a, b in s["intervals"])


def infer_means(
    run_means, run_ids, *, alpha=0.01, linear=False, source_qualified=False, independent_runs=False, minimum_R=32
):
    a = np.asarray(run_means, float)
    R = len(a)
    refusal = []
    if a.ndim != 2 or a.shape[1] != 2 or R < 3 or not np.isfinite(a).all():
        return {"valid": False, "kind": "invalid", "intervals": [], "status": "REFUSE", "reason": "E_BAD_RUN_MEANS"}
    if len(run_ids) != R or len(set(run_ids)) != R:
        refusal.append("E_DUPLICATE_OR_MISSING_RUN_ID")
    if not independent_runs:
        refusal.append("E_INDEPENDENT_RUNS_UNVERIFIED")
    if not source_qualified:
        refusal.append("E_SOURCE_UNQUALIFIED")
    if R < minimum_R:
        refusal.append("E_REPLICATION_BUDGET_UNQUALIFIED")
    mu = a.mean(0)
    cov = np.cov(a, rowvar=False, ddof=1) / R
    diag = np.diag(cov)
    if diag[0] <= 0 or (not linear and diag[1] <= 0):
        return {
            "valid": False,
            "kind": "invalid",
            "intervals": [],
            "status": "REFUSE",
            "reason": "E_NO_VARIANCE_INFORMATION",
            "R": R,
            "mean": mu.tolist(),
            "covariance": cov.tolist(),
        }
    corr = np.array([[1.0, 0.0], [0.0, 1.0]]) if linear else cov / np.sqrt(diag[:, None] * diag[None, :])
    eig = np.linalg.eigvalsh(corr)
    if eig[0] <= 64 * np.finfo(float).eps:
        return {
            "valid": False,
            "kind": "invalid",
            "intervals": [],
            "status": "REFUSE",
            "reason": "E_UNRESOLVED_RANK",
            "R": R,
            "mean": mu.tolist(),
            "covariance": cov.tolist(),
            "standardized_eigenvalues": eig.tolist(),
        }
    q = float(t.ppf(1 - alpha / 2, R - 1) ** 2) if linear else float(2 * (R - 1) / (R - 2) * f.ppf(1 - alpha, 2, R - 2))
    s = invert(mu, cov, q, linear)
    status = "REFUSE" if refusal or not s["valid"] else ("RELEASE" if s["kind"] == "bounded" else "DIAGNOSTIC")
    return {
        **s,
        "status": status,
        "reason": ";".join(refusal) if refusal else (None if status == "RELEASE" else "UNBOUNDED_CONFIDENCE_SET"),
        "R": R,
        "mean": mu.tolist(),
        "covariance": cov.tolist(),
        "q": q,
        "alpha_joint_means": alpha,
        "standardized_eigenvalues": eig.tolist(),
        "engineering_interval": s["intervals"][0] if status == "RELEASE" else None,
    }


def infer_histories(histories, run_ids, **kwargs):
    a = np.asarray(histories, float)
    if a.ndim != 3 or a.shape[2] != 2 or not np.isfinite(a).all() or a.shape[1] < 1:
        return {
            "valid": False,
            "kind": "invalid",
            "intervals": [],
            "status": "REFUSE",
            "reason": "E_BAD_HISTORY_TENSOR",
        }
    return infer_means(a.mean(1), run_ids, **kwargs)
