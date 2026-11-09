"""Experimental evaluation of frozen production output. Never feeds the router."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .baseline import simple_denominator_gate
from .pipeline import replay_from_two_column


# Pre-specified before looking at confirmation hits.
QUALITY_RULES = {
    "engineering_object": "paper_envelope",
    "hit": "released envelope contains independent oracle point",
    "unable_if": "4 * oracle_sd > envelope_halfwidth of the no-diagnostic envelope, or oracle_sd is None",
    "error_release": "RELEASE and envelope misses oracle point (able_to_judge only)",
    "too_conservative": "not RELEASE, but no-diagnostic envelope would have hit the oracle",
    "justified_refuse": "not RELEASE, and no-diagnostic envelope would have missed the oracle",
    "simple_gated_envelope": "same paper envelope object; simple |mean|/se only gates RELEASE vs DIAGNOSTIC/REFUSE",
}


def _halfwidth(interval: Mapping[str, Any] | None) -> float | None:
    if not interval:
        return None
    lo, hi = interval.get("lower"), interval.get("upper")
    if lo is None or hi is None:
        return None
    return 0.5 * abs(float(hi) - float(lo))


def _contains(interval: Mapping[str, Any] | None, point: float | None) -> bool:
    if point is None or not interval:
        return False
    lo, hi = interval.get("lower"), interval.get("upper")
    if lo is None or hi is None:
        return False
    return float(lo) <= float(point) <= float(hi)


def production_infer(two: np.ndarray) -> dict[str, Any]:
    """Frozen product router + paper envelope. No reference in, no threshold change."""

    rec = replay_from_two_column(np.asarray(two, float), numerator="absorption", denominator="flux")
    env = rec.get("paper_envelope") or {}
    rec["engineering_release"] = rec.get("paper_envelope_release")
    rec["engineering_interval"] = (env.get("intervals") or [{}])[0]
    rec["envelope_halfwidth"] = _halfwidth(rec["engineering_interval"])
    rec["envelope_width"] = None if rec["envelope_halfwidth"] is None else 2.0 * rec["envelope_halfwidth"]
    rec["rule"] = "frozen_router"
    return rec


def no_diagnostic_envelope(two: np.ndarray) -> dict[str, Any]:
    """Always take a finite paper envelope as engineering output; skip T_den/Fieller gating."""

    rec = replay_from_two_column(np.asarray(two, float), numerator="absorption", denominator="flux")
    env = rec.get("paper_envelope") or {}
    finite = bool(env.get("engineering_finite_release") and env.get("status") == "PASS")
    rec["engineering_release"] = "RELEASE" if finite else "REFUSE"
    rec["engineering_interval"] = (env.get("intervals") or [{}])[0]
    rec["envelope_halfwidth"] = _halfwidth(rec["engineering_interval"])
    rec["envelope_width"] = None if rec["envelope_halfwidth"] is None else 2.0 * rec["envelope_halfwidth"]
    rec["rule"] = "no_diagnostic_envelope"
    rec["product_method"] = env.get("method")
    return rec


def simple_gated_envelope(two: np.ndarray) -> dict[str, Any]:
    """Same paper envelope as production; simple T_den only changes the release gate."""

    gate = simple_denominator_gate(np.asarray(two, float))
    rec = no_diagnostic_envelope(two)
    if gate["release_status"] != "RELEASE":
        rec["engineering_release"] = gate["release_status"]
    rec["rule"] = "simple_gated_envelope"
    rec["t_den"] = gate["t_den"]
    rec["simple_label"] = gate["release_status"]
    return rec


def simple_labels(two: np.ndarray) -> dict[str, Any]:
    gate = simple_denominator_gate(np.asarray(two, float))
    return {
        "rule": "simple_denominator_gate",
        "engineering_release": gate["release_status"],
        "engineering_interval": None,
        "envelope_halfwidth": None,
        "envelope_width": None,
        "t_den": gate["t_den"],
        "note": "labels only; use simple_gated_envelope for same-object quality comparison",
    }


def judge_task(
    output: Mapping[str, Any],
    *,
    oracle_point: float | None,
    oracle_sd: float | None,
    no_diag: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    interval = output.get("engineering_interval")
    half = (no_diag or output).get("envelope_halfwidth")
    if half is None:
        half = _halfwidth((no_diag or output).get("engineering_interval"))
    unable = oracle_point is None or oracle_sd is None
    if not unable and half is not None and 4.0 * float(oracle_sd) > float(half):
        unable = True
    if not unable and half is None and output.get("engineering_interval") is None:
        # label-only method: still judgeable for labels, not for hits
        unable_hits = True
    else:
        unable_hits = unable
    released = output.get("engineering_release") == "RELEASE"
    hit = False if unable_hits else _contains(interval, oracle_point)
    no_diag_hit = False if unable_hits or no_diag is None else _contains(no_diag.get("engineering_interval"), oracle_point)
    quality = "unable_to_judge"
    if not unable_hits:
        if released:
            quality = "good" if hit else "error_release"
        elif no_diag is not None:
            quality = "too_conservative" if no_diag_hit else "justified_nonrelease"
        else:
            quality = "nonrelease"
    return {
        "able_to_judge": not unable_hits,
        "released": released,
        "hit": None if unable_hits else hit,
        "quality": quality,
        "no_diag_hit": None if unable_hits else no_diag_hit,
        "envelope_halfwidth": output.get("envelope_halfwidth"),
        "envelope_width": output.get("envelope_width"),
    }


def summarize_evaluation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    judged = [r for r in rows if r.get("able_to_judge")]
    released = [r for r in judged if r.get("released")]
    error_rel = [r for r in released if r.get("quality") == "error_release"]
    too_cons = [r for r in judged if r.get("quality") == "too_conservative"]
    justified = [r for r in judged if r.get("quality") == "justified_nonrelease"]
    widths = [float(r["envelope_width"]) for r in released if r.get("envelope_width") is not None]
    halfwidths = [float(r["envelope_halfwidth"]) for r in released if r.get("envelope_halfwidth") is not None]
    return {
        "n": n,
        "n_judged": len(judged),
        "n_unable": n - len(judged),
        "n_released_judged": len(released),
        "release_rate_judged": (len(released) / len(judged)) if judged else None,
        "quality_fail_among_released": (len(error_rel) / len(released)) if released else None,
        "error_release_among_judged": (len(error_rel) / len(judged)) if judged else None,
        "too_conservative_among_judged": (len(too_cons) / len(judged)) if judged else None,
        "justified_nonrelease_among_judged": (len(justified) / len(judged)) if judged else None,
        "envelope_width_median_released": float(np.median(widths)) if widths else None,
        "envelope_halfwidth_median_released": float(np.median(halfwidths)) if halfwidths else None,
        "quality_rules": QUALITY_RULES,
    }
