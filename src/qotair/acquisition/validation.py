"""Pure validation helpers. No OpenMC. Decision rules frozen before hits."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def wilson(hits: int, n: int, z: float = 1.96) -> dict[str, Any]:
    if n <= 0:
        return {"low": None, "high": None, "width": None, "hits": hits, "n": n}
    p = hits / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    low = float(max(0.0, center - half))
    high = float(min(1.0, center + half))
    return {"low": low, "high": high, "width": high - low, "hits": int(hits), "n": int(n)}


def ratio_point(two: np.ndarray) -> float:
    arr = np.asarray(two, float)
    den = float(arr[:, 1].mean())
    if den == 0.0:
        return float("nan")
    return float(arr[:, 0].mean() / den)


def source_protocol_decision(paired: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Choose P4 vs P100 from target-response pairs. Does not use keff, Shannon, or hits.

    Inadequate if median |r4-r100| exceeds 25% of the P4 halfwidth, or 2% of |r100|.
    Mixed seeds (some fail, some pass) -> P100.
    """

    if not paired:
        return {"protocol": None, "reason": "no pairs", "inadequate": None}
    diffs, rels, hws = [], [], []
    per = []
    for item in paired:
        r4 = float(item["r4"])
        r100 = float(item["r100"])
        hw4 = float(item["hw4"])
        d = abs(r4 - r100)
        rel = d / max(abs(r100), 1e-30)
        fail = (d > 0.25 * hw4) or (rel > 0.02)
        diffs.append(d)
        rels.append(rel)
        hws.append(hw4)
        per.append({"seed": item.get("seed"), "abs_diff": d, "rel_diff": rel, "hw4": hw4, "inadequate": fail})
    med_d = float(np.median(diffs))
    med_hw = float(np.median(hws))
    med_rel = float(np.median(rels))
    width_fail = med_d > 0.25 * med_hw
    rel_fail = med_rel > 0.02
    n_fail = sum(1 for row in per if row["inadequate"])
    mixed = 0 < n_fail < len(per)
    inadequate = bool(width_fail or rel_fail or mixed)
    return {
        "protocol": "P100" if inadequate else "P4",
        "inadequate": inadequate,
        "width_fail": width_fail,
        "rel_fail": rel_fail,
        "mixed_seeds": mixed,
        "n_pairs": len(per),
        "n_inadequate_pairs": n_fail,
        "median_abs_diff": med_d,
        "median_rel_diff": med_rel,
        "median_hw4": med_hw,
        "threshold_abs_vs_halfwidth": 0.25 * med_hw,
        "threshold_rel": 0.02,
        "pairs": per,
        "used_keff": False,
        "used_shannon": False,
        "used_hits": False,
    }


def hit_at(lo: float | None, hi: float | None, point: float | None) -> bool | None:
    if lo is None or hi is None or point is None:
        return None
    return float(lo) <= float(point) <= float(hi)


def reference_sensitivity(interval: Mapping[str, Any] | None, oracle: float, sd: float) -> dict[str, Any]:
    if not interval:
        return {"flip": None, "reason": "no interval"}
    lo, hi = interval.get("lower"), interval.get("upper")
    center = hit_at(lo, hi, oracle)
    minus = hit_at(lo, hi, oracle - 2.0 * sd)
    plus = hit_at(lo, hi, oracle + 2.0 * sd)
    vals = [v for v in (center, minus, plus) if v is not None]
    return {
        "center": center,
        "minus_2sd": minus,
        "plus_2sd": plus,
        "flip": (len(set(vals)) > 1) if vals else None,
        "two_sd": 2.0 * sd,
    }


def independent_mesh_columns(nx, ny, nz, i, j, k, energy_bin, scores=("flux", "absorption")):
    """OpenMC last-filter-fastest, score-fastest. Not bind_region."""

    n_mesh = int(nx) * int(ny) * int(nz)
    filter_index = int(energy_bin) * n_mesh + int(i) + int(nx) * (int(j) + int(ny) * int(k))
    n_scores = len(scores)
    names = list(scores)
    return (
        filter_index * n_scores + names.index("absorption"),
        filter_index * n_scores + names.index("flux"),
    )
