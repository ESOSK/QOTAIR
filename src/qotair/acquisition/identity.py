"""Canonical filter identity for the acquisition extension."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Sequence


class AcquisitionError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise AcquisitionError("E_NONFINITE", f"{name} must be finite")
    return result


def _sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FilterSpec:
    filter_type: str
    bins: tuple[Any, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"filter_type": self.filter_type, "bins": list(self.bins)}

    def native_mesh_id(self) -> int | None:
        if self.filter_type == "MeshFilter" and len(self.bins) == 11 and self.bins[0] == "RegularMesh":
            return int(self.bins[10])
        return None

    def energy_function_table(self) -> str | None:
        if self.filter_type == "EnergyFunctionFilter" and len(self.bins) >= 4:
            return str(self.bins[3])
        return None

    def bin_count(self) -> int:
        if self.filter_type == "EnergyFilter":
            return len(self.bins) - 1
        if self.filter_type == "MeshFilter":
            if not self.bins or self.bins[0] != "RegularMesh" or len(self.bins) not in {10, 11}:
                raise AcquisitionError("E_MESH_BINS", "MeshFilter requires RegularMesh bins")
            return int(self.bins[1]) * int(self.bins[2]) * int(self.bins[3])
        if self.filter_type == "EnergyFunctionFilter":
            return 1
        if self.filter_type == "DistribcellFilter":
            if len(self.bins) >= 2:
                n = int(self.bins[1])
                if n < 1:
                    raise AcquisitionError("E_DISTRIBCELL_INSTANCES", "expected_instances must be >= 1")
                return n
            return 1
        return len(self.bins)

    @classmethod
    def energy(cls, edges_eV: Sequence[float]) -> "FilterSpec":
        edges = tuple(_finite(v, "energy") for v in edges_eV)
        if len(edges) < 2 or any(right <= left for left, right in zip(edges, edges[1:])):
            raise AcquisitionError("E_ENERGY_EDGES", "energy edges must increase")
        return cls("EnergyFilter", edges)

    @classmethod
    def cell(cls, cell_ids: Sequence[int]) -> "FilterSpec":
        bins = tuple(int(v) for v in cell_ids)
        if not bins or len(bins) != len(set(bins)):
            raise AcquisitionError("E_CELL_BINS", "cell ids must be unique")
        return cls("CellFilter", bins)

    @classmethod
    def particle(cls, particles: Sequence[str]) -> "FilterSpec":
        bins = tuple(str(v) for v in particles)
        allowed = {"neutron", "photon", "electron", "positron"}
        if not bins or any(p not in allowed for p in bins):
            raise AcquisitionError("E_PARTICLE_BINS", "unsupported particle")
        return cls("ParticleFilter", bins)

    @classmethod
    def universe(cls, universe_ids: Sequence[int]) -> "FilterSpec":
        bins = tuple(int(v) for v in universe_ids)
        if not bins or len(bins) != len(set(bins)):
            raise AcquisitionError("E_UNIVERSE_BINS", "universe ids must be unique")
        return cls("UniverseFilter", bins)

    @classmethod
    def distribcell(cls, cell_id: int, expected_instances: int | None = None) -> "FilterSpec":
        cid = int(cell_id)
        if expected_instances is None:
            return cls("DistribcellFilter", (cid,))
        n = int(expected_instances)
        if n < 1:
            raise AcquisitionError("E_DISTRIBCELL_INSTANCES", "expected_instances must be >= 1")
        return cls("DistribcellFilter", (cid, n))

    @classmethod
    def material(cls, material_ids: Sequence[int]) -> "FilterSpec":
        bins = tuple(int(v) for v in material_ids)
        if not bins or len(bins) != len(set(bins)):
            raise AcquisitionError("E_MATERIAL_BINS", "material ids must be unique")
        return cls("MaterialFilter", bins)

    @classmethod
    def mesh(
        cls,
        nx: int,
        ny: int,
        nz: int,
        lower_left: Sequence[float],
        upper_right: Sequence[float],
    ) -> "FilterSpec":
        nx_i, ny_i, nz_i = int(nx), int(ny), int(nz)
        if min(nx_i, ny_i, nz_i) < 1:
            raise AcquisitionError("E_MESH_BINS", "mesh dimensions must be >= 1")
        ll = tuple(_finite(v, "lower_left") for v in lower_left)
        ur = tuple(_finite(v, "upper_right") for v in upper_right)
        if len(ll) != 3 or len(ur) != 3 or any(ur[i] <= ll[i] for i in range(3)):
            raise AcquisitionError("E_MESH_BINS", "invalid RegularMesh bounds")
        return cls("MeshFilter", ("RegularMesh", nx_i, ny_i, nz_i, *ll, *ur))

    @classmethod
    def native_mesh(
        cls,
        mesh_id: int,
        nx: int,
        ny: int,
        nz: int,
        lower_left: Sequence[float],
        upper_right: Sequence[float],
    ) -> "FilterSpec":
        base = cls.mesh(nx, ny, nz, lower_left, upper_right)
        return cls("MeshFilter", base.bins + (int(mesh_id),))

    @classmethod
    def energy_function(
        cls,
        energies: Sequence[float],
        y: Sequence[float],
        table_name: str | None = None,
    ) -> "FilterSpec":
        e = tuple(_finite(v, "eff energy") for v in energies)
        yy = tuple(_finite(v, "eff y") for v in y)
        if len(e) != len(yy) or len(e) < 2 or any(right <= left for left, right in zip(e, e[1:])):
            raise AcquisitionError("E_ENERGY_FUNCTION", "EnergyFunctionFilter needs increasing energies")
        if table_name is None:
            return cls("EnergyFunctionFilter", ("tabulated", e, yy))
        return cls("EnergyFunctionFilter", ("tabulated", e, yy, str(table_name)))

    @classmethod
    def energy_function_from_table(cls, table_name: str) -> "FilterSpec":
        from .tables import get_table

        table = get_table(table_name)
        return cls.energy_function(table.energies_eV, table.y, table_name=table.name)


@dataclass(frozen=True)
class AcquisitionIdentity:
    filters: tuple[FilterSpec, ...]
    nuclides: tuple[str, ...]
    scores: tuple[str, ...]
    estimator: str = "tracklength"

    def __post_init__(self) -> None:
        types = [item.filter_type for item in self.filters]
        if len(types) != len(set(types)):
            raise AcquisitionError("E_DUPLICATE_FILTER_TYPE", "one filter of each type")
        if self.estimator not in {"tracklength", "collision", "analog"}:
            raise AcquisitionError("E_ESTIMATOR", self.estimator)
        if not self.nuclides or not self.scores:
            raise AcquisitionError("E_IDENTITY_VALUES", "nuclides and scores required")

    def as_dict(self) -> dict[str, Any]:
        return {
            "filters": [item.as_dict() for item in self.filters],
            "nuclides": list(self.nuclides),
            "scores": list(self.scores),
            "estimator": self.estimator,
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.as_dict())

    def filter_bin_count(self) -> int:
        total = 1
        for item in self.filters:
            total *= max(1, item.bin_count())
        return total
