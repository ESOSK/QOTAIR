"""Query-set column planning, ledger reuse, and retained-vs-dropped range."""
from __future__ import annotations

from typing import Any, Sequence

from .bind import RegionSpec, bind_region
from .identity import AcquisitionIdentity
from .provenance import RunSource, ledger_mix_forbidden
from .select import ledger_column_count, two_score_column_indices


def planned_columns(
    identity: AcquisitionIdentity,
    specs: Sequence[RegionSpec],
    *,
    numerator: str = "absorption",
    denominator: str = "flux",
    nuclide: str = "total",
) -> dict[str, Any]:
    """Dedupe shared operand columns across a declared query set."""

    queries: list[dict[str, Any]] = []
    seen: list[int] = []
    for spec in specs:
        bound = bind_region(identity, spec)
        pair = two_score_column_indices(identity, bound["selection"], numerator, denominator, nuclide)
        queries.append(
            {
                "name": spec.name,
                "selection": bound["selection"],
                "columns": list(pair),
                "object_path": bound["object_path"],
            }
        )
        for col in pair:
            if col not in seen:
                seen.append(int(col))
    n_identity = ledger_column_count(identity)
    n_unique = len(seen)
    return {
        "unique_columns": seen,
        "n_unique": n_unique,
        "n_identity": n_identity,
        "n_if_separate": 2 * len(specs),
        "queries": queries,
        "retention": {
            "kept_columns": seen,
            "n_kept": n_unique,
            "n_dropped": max(0, n_identity - n_unique),
            "dropped_are": "all identity columns not in the declared query set",
        },
    }


def ledger_reuse_decision(
    stored_identity: AcquisitionIdentity,
    new_identity: AcquisitionIdentity,
    stored_source: RunSource,
    new_source: RunSource,
) -> dict[str, Any]:
    """Same identity + same run source: slice another bin. Otherwise re-acquire."""

    if ledger_mix_forbidden(stored_source, new_source):
        return {
            "reuse": False,
            "must_reacquire": True,
            "reason_code": "E_RUN_SOURCE_MISMATCH",
            "message": "different run source; do not splice batches",
        }
    if stored_identity.sha256 != new_identity.sha256:
        return {
            "reuse": False,
            "must_reacquire": True,
            "reason_code": "E_IDENTITY_MISMATCH",
            "message": "tally identity changed; re-run transport",
        }
    return {
        "reuse": True,
        "must_reacquire": False,
        "reason_code": None,
        "message": "same identity and run source; additional declared bins may be sliced from the stored ledger",
    }


def retention_record(identity: AcquisitionIdentity, kept_columns: Sequence[int]) -> dict[str, Any]:
    n_identity = ledger_column_count(identity)
    kept = [int(c) for c in kept_columns]
    return {
        "n_identity_columns": n_identity,
        "kept_columns": kept,
        "n_kept": len(kept),
        "n_dropped": max(0, n_identity - len(kept)),
        "tally_definition_changed": False,
    }
