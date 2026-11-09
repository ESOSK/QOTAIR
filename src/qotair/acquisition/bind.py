"""Explicit region specification. Silent most-instances fallback is not the formal path."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .identity import AcquisitionError, AcquisitionIdentity
from .paths import MESH_INDEX_CONVENTION
from .select import first_bin_selection, mesh_ijk_selection, mesh_shape


ALLOWED_KINDS = frozenset({"mesh_ijk", "cell", "distribcell", "universe", "material"})


@dataclass(frozen=True)
class RegionSpec:
    """User-declared region. Heuristic pick_* helpers are opt-in only."""

    kind: str
    name: str
    mesh_ijk: tuple[int, int, int] | None = None
    cell_id: int | None = None
    instance_index: int | None = None
    universe_id: int | None = None
    material_id: int | None = None
    energy_bin: int = 0
    allow_heuristic: bool = False

    def validate(self) -> None:
        if self.kind not in ALLOWED_KINDS:
            raise AcquisitionError("E_REGION_UNSPECIFIED", f"unknown region kind {self.kind!r}")
        if self.kind == "mesh_ijk" and self.mesh_ijk is None:
            raise AcquisitionError("E_REGION_UNSPECIFIED", "mesh_ijk requires (i,j,k)")
        if self.kind == "cell" and self.cell_id is None:
            raise AcquisitionError("E_REGION_UNSPECIFIED", "cell region requires cell_id")
        if self.kind == "distribcell":
            if self.cell_id is None and not self.allow_heuristic:
                raise AcquisitionError(
                    "E_REGION_UNSPECIFIED",
                    "distribcell requires cell_id; most-instances fallback is disabled on the formal path",
                )
        if self.kind == "universe" and self.universe_id is None and not self.allow_heuristic:
            raise AcquisitionError(
                "E_REGION_UNSPECIFIED",
                "universe requires universe_id; non-root fallback is disabled on the formal path",
            )
        if self.kind == "material" and self.material_id is None and not self.allow_heuristic:
            raise AcquisitionError(
                "E_REGION_UNSPECIFIED",
                "material requires material_id; fissionable fallback is disabled on the formal path",
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "mesh_ijk": list(self.mesh_ijk) if self.mesh_ijk is not None else None,
            "cell_id": self.cell_id,
            "instance_index": self.instance_index,
            "universe_id": self.universe_id,
            "material_id": self.material_id,
            "energy_bin": int(self.energy_bin),
            "allow_heuristic": bool(self.allow_heuristic),
        }


def _filter_bins(identity: AcquisitionIdentity, filter_type: str) -> tuple[Any, ...]:
    for item in identity.filters:
        if item.filter_type == filter_type:
            return item.bins
    raise AcquisitionError("E_BIN_SELECTION", f"identity has no {filter_type}")


def bind_region(identity: AcquisitionIdentity, spec: RegionSpec) -> dict[str, Any]:
    """Map a declared region onto filter-bin selection. No silent object pick."""

    spec.validate()
    selection: dict[str, int]
    object_path: dict[str, Any]
    if spec.kind == "mesh_ijk":
        i, j, k = spec.mesh_ijk  # type: ignore[misc]
        selection = mesh_ijk_selection(identity, i, j, k, energy_bin=spec.energy_bin)
        nx, ny, nz = mesh_shape(identity)
        object_path = {
            "mesh_shape": [nx, ny, nz],
            "mesh_ijk": [int(i), int(j), int(k)],
            "linear_index": selection["MeshFilter"],
            **MESH_INDEX_CONVENTION,
        }
    elif spec.kind == "cell":
        bins = _filter_bins(identity, "CellFilter")
        cid = int(spec.cell_id)  # type: ignore[arg-type]
        if cid not in bins:
            raise AcquisitionError("E_REGION_AMBIGUOUS", f"cell {cid} is not on the CellFilter bins {list(bins)}")
        selection = first_bin_selection(identity)
        selection["CellFilter"] = list(bins).index(cid)
        object_path = {"cell_id": cid, "filter_bin": selection["CellFilter"]}
    elif spec.kind == "distribcell":
        bins = _filter_bins(identity, "DistribcellFilter")
        cid = int(spec.cell_id)  # type: ignore[arg-type]
        if int(bins[0]) != cid:
            raise AcquisitionError(
                "E_REGION_AMBIGUOUS",
                f"declared cell {cid} does not match DistribcellFilter cell {bins[0]}",
            )
        n = int(bins[1]) if len(bins) >= 2 else 1
        inst = 0 if spec.instance_index is None else int(spec.instance_index)
        if inst < 0 or inst >= n:
            raise AcquisitionError("E_BIN_SELECTION", f"instance {inst} outside [0, {n})")
        selection = first_bin_selection(identity)
        selection["DistribcellFilter"] = inst
        object_path = {"cell_id": cid, "instance_index": inst, "expected_instances": n}
    elif spec.kind == "universe":
        bins = _filter_bins(identity, "UniverseFilter")
        uid = int(spec.universe_id)  # type: ignore[arg-type]
        if uid not in bins:
            raise AcquisitionError("E_REGION_AMBIGUOUS", f"universe {uid} is not on the filter bins {list(bins)}")
        selection = first_bin_selection(identity)
        selection["UniverseFilter"] = list(bins).index(uid)
        object_path = {"universe_id": uid, "filter_bin": selection["UniverseFilter"]}
    else:
        bins = _filter_bins(identity, "MaterialFilter")
        mid = int(spec.material_id)  # type: ignore[arg-type]
        if mid not in bins:
            raise AcquisitionError("E_REGION_AMBIGUOUS", f"material {mid} is not on the filter bins {list(bins)}")
        selection = first_bin_selection(identity)
        selection["MaterialFilter"] = list(bins).index(mid)
        object_path = {"material_id": mid, "filter_bin": selection["MaterialFilter"]}
    if "EnergyFilter" in selection:
        selection["EnergyFilter"] = int(spec.energy_bin)
    return {
        "region": spec.as_dict(),
        "selection": selection,
        "object_path": object_path,
        "heuristic_used": False,
    }


def binding_record(
    identity: AcquisitionIdentity,
    spec: RegionSpec,
    *,
    numerator: str,
    denominator: str,
    nuclide: str = "total",
    estimator: str | None = None,
    units: str = "1",
    run_source: Mapping[str, Any] | None = None,
    inference_rule: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    bound = bind_region(identity, spec)
    return {
        "physical_object": {
            "name": spec.name,
            "kind": spec.kind,
            **bound["object_path"],
        },
        "query_meaning": {
            "numerator": numerator,
            "denominator": denominator,
            "nuclide": nuclide,
            "estimator": estimator or identity.estimator,
            "energy_bin": spec.energy_bin,
            "units": units,
        },
        "data_map": {
            "identity_sha256": identity.sha256,
            "filters": [item.as_dict() for item in identity.filters],
            "scores": list(identity.scores),
            "nuclides": list(identity.nuclides),
            "selection": bound["selection"],
        },
        "run_source": dict(run_source or {}),
        "inference_source": dict(inference_rule or {"thresholds": "frozen T_den / hull; not retuned"}),
        "heuristic_used": False,
    }
