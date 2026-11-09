"""Same acquisition identity across depletion material states.

Step routing reuses frozen T_den / hull on each step's ΔSUM ledger. It is not
a new Table 4.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .identity import AcquisitionIdentity
from .route import select_and_route
from .select import first_bin_selection


def identity_sha256(identity: AcquisitionIdentity) -> str:
    return identity.sha256


def same_identity(left: AcquisitionIdentity, right: AcquisitionIdentity) -> bool:
    return left.sha256 == right.sha256


def route_steps(
    step_ledgers: Sequence[np.ndarray],
    identity: AcquisitionIdentity,
    selection: Mapping[str, int] | None = None,
    *,
    numerator: str = "absorption",
    denominator: str = "flux",
    nuclide: str = "total",
    confidence_level: float = 0.95,
) -> list[dict[str, Any]]:
    chosen = dict(selection) if selection is not None else first_bin_selection(identity)
    results: list[dict[str, Any]] = []
    for index, ledger in enumerate(step_ledgers):
        routed = select_and_route(
            np.asarray(ledger, dtype=float),
            identity,
            chosen,
            numerator=numerator,
            denominator=denominator,
            nuclide=nuclide,
            query_id=f"depletion_step_{index}_absorption_over_flux",
            confidence_level=confidence_level,
        )
        routed["step"] = index
        routed["table4"] = "unchanged; per-step route is not a new Table 4"
        results.append(routed)
    return results
