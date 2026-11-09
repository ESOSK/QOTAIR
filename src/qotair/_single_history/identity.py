from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


IDENTITY_SCHEMA = "p1-tally-acquisition-identity-v1"
PLAN_SCHEMA = "p1-acquisition-collection-plan-v1"


class TallyIdentityError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_float(value: Any, field_name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise TallyIdentityError("E_IDENTITY_NONFINITE", f"{field_name} must be finite")
    return result


@dataclass(frozen=True)
class FilterIdentity:
    filter_type: str
    bins: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.filter_type or not isinstance(self.filter_type, str):
            raise TallyIdentityError("E_FILTER_TYPE", "filter_type must be a non-empty string")
        if not self.bins:
            raise TallyIdentityError("E_FILTER_BINS", "filter bins must be non-empty")

    @classmethod
    def energy(cls, edges_eV: Sequence[float]) -> "FilterIdentity":
        edges = tuple(_finite_float(value, "energy edge") for value in edges_eV)
        if len(edges) < 2 or any(right <= left for left, right in zip(edges, edges[1:])):
            raise TallyIdentityError("E_ENERGY_EDGES", "Energy edges must be strictly increasing")
        return cls("EnergyFilter", edges)

    @classmethod
    def cell(cls, cell_ids: Sequence[int]) -> "FilterIdentity":
        bins = tuple(int(value) for value in cell_ids)
        if not bins or len(bins) != len(set(bins)):
            raise TallyIdentityError("E_CELL_BINS", "CellFilter bins must be unique cell ids")
        return cls("CellFilter", bins)

    @classmethod
    def particle(cls, particles: Sequence[str]) -> "FilterIdentity":
        bins = tuple(str(value) for value in particles)
        allowed = {"neutron", "photon", "electron", "positron"}
        if not bins or len(bins) != len(set(bins)) or any(item not in allowed for item in bins):
            raise TallyIdentityError("E_PARTICLE_BINS", "ParticleFilter bins must be unique supported particles")
        return cls("ParticleFilter", bins)

    def bin_count(self) -> int:
        if self.filter_type == "EnergyFilter":
            return len(self.bins) - 1
        return len(self.bins)

    def as_dict(self) -> dict[str, Any]:
        return {"filter_type": self.filter_type, "bins": list(self.bins)}


@dataclass(frozen=True)
class TallyAcquisitionIdentity:
    filters: tuple[FilterIdentity, ...]
    nuclides: tuple[str, ...]
    scores: tuple[str, ...]
    estimator: str = "tracklength"
    derivative: tuple[tuple[str, Any], ...] = ()
    multiply_density: bool = True
    schema_version: str = IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != IDENTITY_SCHEMA:
            raise TallyIdentityError("E_IDENTITY_SCHEMA", "Unsupported acquisition identity schema")
        if len({item.filter_type for item in self.filters}) != len(self.filters):
            raise TallyIdentityError("E_DUPLICATE_FILTER_TYPE", "Only one filter of each type is allowed in v1")
        for field_name, values in (("nuclides", self.nuclides), ("scores", self.scores)):
            if not values or len(values) != len(set(values)) or not all(isinstance(v, str) and v for v in values):
                raise TallyIdentityError("E_IDENTITY_VALUES", f"{field_name} must contain unique non-empty strings")
        if self.estimator not in {"tracklength", "collision", "analog"}:
            raise TallyIdentityError("E_ESTIMATOR", f"Unsupported estimator: {self.estimator}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "filters": [item.as_dict() for item in self.filters],
            "nuclides": list(self.nuclides),
            "scores": list(self.scores),
            "estimator": self.estimator,
            "derivative": dict(self.derivative),
            "multiply_density": self.multiply_density,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.as_dict())


@dataclass(frozen=True)
class ProjectionTerm:
    filter_bin: int
    nuclide: str
    score: str
    coefficient: float = 1.0

    def __post_init__(self) -> None:
        if self.filter_bin < 0 or not self.nuclide or not self.score:
            raise TallyIdentityError("E_PROJECTION_TERM", "Projection term has an invalid bin, nuclide, or score")
        _finite_float(self.coefficient, "projection coefficient")

    def as_dict(self) -> dict[str, Any]:
        return {"filter_bin": self.filter_bin, "nuclide": self.nuclide, "score": self.score, "coefficient": self.coefficient}


@dataclass(frozen=True)
class ResultProjection:
    ledger_column: str
    acquisition_sha256: str
    terms: tuple[ProjectionTerm, ...]

    def __post_init__(self) -> None:
        if not self.ledger_column or len(self.acquisition_sha256) != 64 or not self.terms:
            raise TallyIdentityError("E_PROJECTION", "Projection requires a column, identity hash, and at least one term")

    def as_dict(self) -> dict[str, Any]:
        return {"ledger_column": self.ledger_column, "acquisition_sha256": self.acquisition_sha256, "terms": [item.as_dict() for item in self.terms]}


@dataclass(frozen=True)
class PostprocessSemantics:
    output_unit: str
    scale: float = 1.0
    offset: float = 0.0
    volume_cm3: float | None = None
    density_g_cm3: float | None = None
    source_rate_per_s: float | None = None
    metadata: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.output_unit:
            raise TallyIdentityError("E_POSTPROCESS_UNIT", "output_unit must be non-empty")
        for name in ("scale", "offset", "volume_cm3", "density_g_cm3", "source_rate_per_s"):
            value = getattr(self, name)
            if value is not None:
                _finite_float(value, name)

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_unit": self.output_unit,
            "scale": self.scale,
            "offset": self.offset,
            "volume_cm3": self.volume_cm3,
            "density_g_cm3": self.density_g_cm3,
            "source_rate_per_s": self.source_rate_per_s,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class QueryAcquisitionMapping:
    query_id: str
    projections: tuple[ResultProjection, ...]
    postprocess: PostprocessSemantics

    def as_dict(self) -> dict[str, Any]:
        return {"query_id": self.query_id, "projections": [item.as_dict() for item in self.projections], "postprocess": self.postprocess.as_dict()}


@dataclass(frozen=True)
class AcquisitionCollectionPlan:
    identities: tuple[TallyAcquisitionIdentity, ...]
    query_mappings: tuple[QueryAcquisitionMapping, ...]
    required_existing_tally_ids: tuple[int, ...] = ()
    schema_version: str = PLAN_SCHEMA
    canonical_plan_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        hashes = [identity.sha256 for identity in self.identities]
        if not hashes or len(hashes) != len(set(hashes)):
            raise TallyIdentityError("E_PLAN_IDENTITY_DUPLICATE", "Plan identities must be non-empty and deduplicated")
        known = set(hashes)
        for mapping in self.query_mappings:
            if not mapping.projections or any(item.acquisition_sha256 not in known for item in mapping.projections):
                raise TallyIdentityError("E_PLAN_MAPPING", f"Query {mapping.query_id} references an unknown acquisition identity")
        payload = self.as_dict(include_hash=False)
        object.__setattr__(self, "canonical_plan_sha256", _canonical_sha256(payload))

    def as_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "identities": [{"sha256": item.sha256, **item.as_dict()} for item in self.identities],
            "query_mappings": [item.as_dict() for item in self.query_mappings],
            "required_existing_tally_ids": list(self.required_existing_tally_ids),
        }
        if include_hash:
            payload["canonical_plan_sha256"] = self.canonical_plan_sha256
        return payload


def deduplicate_identities(identities: Iterable[TallyAcquisitionIdentity]) -> tuple[TallyAcquisitionIdentity, ...]:
    by_hash: dict[str, TallyAcquisitionIdentity] = {}
    for identity in identities:
        by_hash.setdefault(identity.sha256, identity)
    return tuple(by_hash[key] for key in sorted(by_hash))


def identity_bin_count(identity: TallyAcquisitionIdentity) -> int:
    counts = [item.bin_count() for item in identity.filters]
    total = 1
    for count in counts:
        total *= max(1, int(count))
    return total


def validate_projection(identity: TallyAcquisitionIdentity, projection: ResultProjection) -> None:
    if projection.acquisition_sha256 != identity.sha256:
        raise TallyIdentityError("E_PROJECTION_IDENTITY", "Projection hash does not match the supplied identity")
    bin_count = identity_bin_count(identity)
    for term in projection.terms:
        if term.filter_bin >= bin_count or term.nuclide not in identity.nuclides or term.score not in identity.scores:
            raise TallyIdentityError("E_PROJECTION_OUT_OF_RANGE", f"Projection term is not present in identity {identity.sha256}")


def standard_acquisition_plan(*, include_fission: bool) -> AcquisitionCollectionPlan:
    scores = ("flux", "total", "absorption", "fission") if include_fission else ("flux", "total", "absorption")
    identity = TallyAcquisitionIdentity(
        filters=(FilterIdentity.energy((0.0, 0.625, 20_000_000.0)),),
        nuclides=("total",),
        scores=scores,
        estimator="tracklength",
    )
    def projection(column: str, terms: Sequence[tuple[int, str, float]]) -> ResultProjection:
        result = ResultProjection(column, identity.sha256, tuple(ProjectionTerm(bin_index, "total", score, coefficient) for bin_index, score, coefficient in terms))
        validate_projection(identity, result)
        return result
    projections = {
        "total": projection("total", ((0, "total", 1.0), (1, "total", 1.0))),
        "flux": projection("flux", ((0, "flux", 1.0), (1, "flux", 1.0))),
        "absorption": projection("absorption", ((0, "absorption", 1.0), (1, "absorption", 1.0))),
        "fast_flux": projection("fast_flux", ((1, "flux", 1.0),)),
        "thermal_flux": projection("thermal_flux", ((0, "flux", 1.0),)),
    }
    mappings = (
        QueryAcquisitionMapping("Q_RATIO_TOTAL_OVER_FLUX", (projections["total"], projections["flux"]), PostprocessSemantics("dimensionless")),
        QueryAcquisitionMapping("Q_NORMALIZED_FAST_FLUX", (projections["fast_flux"], projections["thermal_flux"]), PostprocessSemantics("dimensionless")),
        QueryAcquisitionMapping("Q_MGXS_ABSORPTION_EQUIVALENT", (projections["absorption"], projections["flux"]), PostprocessSemantics("cm^-1")),
    )
    return AcquisitionCollectionPlan((identity,), mappings)
