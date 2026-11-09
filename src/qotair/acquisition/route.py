"""Route a two-column selected-bin ledger through the frozen T_den / hull gate.

Thresholds are imported from the freeze snapshot. This module does not retune
them and does not write a new threshold table.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .identity import AcquisitionError, AcquisitionIdentity
from .select import (
    center_mesh_selection,
    first_bin_selection,
    mesh_ijk_selection,
    slice_two_energy_groups,
    slice_two_scores,
)


def absorption_over_flux_query(
    *,
    query_id: str = "selected_bin_absorption_over_flux",
    numerator: str = "absorption",
    denominator: str = "flux",
) -> dict[str, Any]:
    return {
        "schema_version": "p1-query-v1",
        "query_id": query_id,
        "name": "Selected-bin two-score ratio",
        "expression_class": "ratio",
        "output_unit": "1",
        "operand_units": {numerator: "1", denominator: "1"},
        "domain_constraints": [
            "denominator must be stable",
            "selected filter bin is a two-column ledger",
        ],
        "expression": {
            "op": "divide",
            "args": [
                {"op": "score", "name": numerator},
                {"op": "score", "name": denominator},
            ],
        },
    }


def route_selected_ratio(
    ledger_two_col: np.ndarray,
    *,
    numerator: str = "absorption",
    denominator: str = "flux",
    query_id: str = "selected_bin_absorption_over_flux",
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    values = np.asarray(ledger_two_col, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise AcquisitionError("E_LEDGER_SHAPE", "router expects a two-column ledger")
    from ..services.scientific import infer_query, json_safe

    query = absorption_over_flux_query(
        query_id=query_id, numerator=numerator, denominator=denominator
    )
    result = infer_query(
        query,
        values,
        [numerator, denominator],
        confidence_level=float(confidence_level),
    )
    return json_safe(result)


def select_and_route(
    ledger: np.ndarray,
    identity: AcquisitionIdentity,
    selection: Mapping[str, int] | None = None,
    *,
    numerator: str = "absorption",
    denominator: str = "flux",
    nuclide: str = "total",
    query_id: str = "selected_bin_absorption_over_flux",
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    chosen = dict(selection) if selection is not None else first_bin_selection(identity)
    two = slice_two_scores(ledger, identity, chosen, numerator, denominator, nuclide)
    inference = route_selected_ratio(
        two,
        numerator=numerator,
        denominator=denominator,
        query_id=query_id,
        confidence_level=confidence_level,
    )
    return {
        "identity_sha256": identity.sha256,
        "selection": chosen,
        "nuclide": nuclide,
        "numerator": numerator,
        "denominator": denominator,
        "ledger_shape": list(np.asarray(ledger).shape),
        "selected_shape": list(two.shape),
        "thresholds": "frozen T_den / hull; not retuned",
        "table4": "unchanged",
        "inference": inference,
    }


def select_mesh_ijk_and_route(
    ledger: np.ndarray,
    identity: AcquisitionIdentity,
    i: int,
    j: int,
    k: int = 0,
    *,
    energy_bin: int = 0,
    numerator: str = "absorption",
    denominator: str = "flux",
    nuclide: str = "total",
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    chosen = mesh_ijk_selection(identity, i, j, k, energy_bin=energy_bin)
    routed = select_and_route(
        ledger,
        identity,
        chosen,
        numerator=numerator,
        denominator=denominator,
        nuclide=nuclide,
        query_id=f"mesh_{i}_{j}_{k}_e{energy_bin}_{numerator}_over_{denominator}",
        confidence_level=confidence_level,
    )
    routed["mesh_ijk"] = [int(i), int(j), int(k)]
    routed["energy_bin"] = int(energy_bin)
    return routed


def select_center_mesh_and_route(
    ledger: np.ndarray,
    identity: AcquisitionIdentity,
    **kwargs: Any,
) -> dict[str, Any]:
    chosen = center_mesh_selection(identity, energy_bin=int(kwargs.pop("energy_bin", 0)))
    return select_and_route(ledger, identity, chosen, **kwargs)


def route_energy_group_ratio(
    ledger: np.ndarray,
    identity: AcquisitionIdentity,
    *,
    numerator_energy: int = 1,
    denominator_energy: int = 0,
    score: str = "flux",
    nuclide: str = "total",
    mesh_ijk: tuple[int, int, int] | None = None,
    query_id: str = "fast_over_thermal_flux",
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    two = slice_two_energy_groups(
        ledger,
        identity,
        numerator_energy=numerator_energy,
        denominator_energy=denominator_energy,
        score=score,
        nuclide=nuclide,
        mesh_ijk=mesh_ijk,
    )
    numerator = f"{score}_g{int(numerator_energy)}"
    denominator = f"{score}_g{int(denominator_energy)}"
    inference = route_selected_ratio(
        two,
        numerator=numerator,
        denominator=denominator,
        query_id=query_id,
        confidence_level=confidence_level,
    )
    return {
        "identity_sha256": identity.sha256,
        "kind": "same_score_two_energy_bins",
        "score": score,
        "nuclide": nuclide,
        "numerator_energy": int(numerator_energy),
        "denominator_energy": int(denominator_energy),
        "mesh_ijk": list(mesh_ijk) if mesh_ijk is not None else None,
        "selected_shape": list(two.shape),
        "thresholds": "frozen T_den / hull; not retuned",
        "table4": "unchanged",
        "nested_ratio": False,
        "inference": inference,
    }
