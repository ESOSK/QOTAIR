"""Inspectable query contracts: meaning vs batch-evidence reuse."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Sequence

from .bind import RegionSpec, bind_region
from .identity import AcquisitionError, AcquisitionIdentity
from .provenance import RunSource, ledger_mix_forbidden
from .select import two_score_column_indices


def _sha(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def energy_edges_from_identity(identity: AcquisitionIdentity) -> tuple[float, ...] | None:
    for item in identity.filters:
        if item.filter_type == "EnergyFilter":
            return tuple(float(v) for v in item.bins)
    return None


@dataclass(frozen=True)
class QueryContract:
    """Declared query meaning. Name is a label and is not part of meaning.

    Energy meaning is the actual edges plus the bin index, not the index alone.
    Integer ledger columns are storage layout; score names select columns.
    Bound reuse also checks identity_sha256, which includes the identity's
    score-tuple order. A score-order change therefore requires rebind; it is
    not absorbed automatically.
    """

    name: str
    kind: str
    numerator: str
    denominator: str
    nuclide: str = "total"
    estimator: str = "tracklength"
    energy_bin: int = 0
    energy_edges_eV: tuple[float, ...] | None = None
    units: str = "1"
    cell_id: int | None = None
    instance_index: int | None = None
    mesh_ijk: tuple[int, int, int] | None = None
    universe_id: int | None = None
    material_id: int | None = None
    expected_instances: int | None = None
    identity_sha256: str | None = None
    scores: tuple[str, ...] | None = None

    @classmethod
    def bind(cls, identity: AcquisitionIdentity, **kwargs: Any) -> "QueryContract":
        kwargs.setdefault("energy_edges_eV", energy_edges_from_identity(identity))
        kwargs.setdefault("identity_sha256", identity.sha256)
        kwargs.setdefault("estimator", identity.estimator)
        kwargs.setdefault("scores", tuple(identity.scores))
        return cls(**kwargs)

    def meaning(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "nuclide": self.nuclide,
            "estimator": self.estimator,
            "energy_bin": int(self.energy_bin),
            "energy_edges_eV": list(self.energy_edges_eV) if self.energy_edges_eV is not None else None,
            "units": self.units,
            "cell_id": self.cell_id,
            "instance_index": self.instance_index,
            "mesh_ijk": list(self.mesh_ijk) if self.mesh_ijk is not None else None,
            "universe_id": self.universe_id,
            "material_id": self.material_id,
            "expected_instances": self.expected_instances,
            "identity_sha256": self.identity_sha256,
            "scores": list(self.scores) if self.scores is not None else None,
        }

    @property
    def meaning_sha256(self) -> str:
        return _sha(self.meaning())

    def region_spec(self) -> RegionSpec:
        return RegionSpec(
            kind=self.kind,
            name=self.name,
            mesh_ijk=self.mesh_ijk,
            cell_id=self.cell_id,
            instance_index=self.instance_index,
            universe_id=self.universe_id,
            material_id=self.material_id,
            energy_bin=self.energy_bin,
            allow_heuristic=False,
        )

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "meaning_sha256": self.meaning_sha256, **self.meaning()}


def can_reuse_query_definition(stored: QueryContract, new: QueryContract) -> dict[str, Any]:
    """Same physical query meaning, ignoring display name."""

    if stored.meaning_sha256 == new.meaning_sha256:
        return {"reuse": True, "reason_code": None, "message": "query meaning matches"}
    reasons = []
    s, n = stored.meaning(), new.meaning()
    if s.get("energy_bin") != n.get("energy_bin") or s.get("energy_edges_eV") != n.get("energy_edges_eV"):
        reasons.append("E_ENERGY_MISMATCH")
    if s.get("identity_sha256") and n.get("identity_sha256") and s.get("identity_sha256") != n.get("identity_sha256"):
        reasons.append("E_IDENTITY_MISMATCH")
    if s.get("instance_index") != n.get("instance_index") or s.get("cell_id") != n.get("cell_id"):
        reasons.append("E_INSTANCE_MISMATCH")
    if s.get("nuclide") != n.get("nuclide") or s.get("estimator") != n.get("estimator"):
        reasons.append("E_NORMALIZATION_MISMATCH")
    if s.get("numerator") != n.get("numerator") or s.get("denominator") != n.get("denominator"):
        reasons.append("E_QUERY_MEANING_CHANGED")
    if s.get("mesh_ijk") != n.get("mesh_ijk"):
        reasons.append("E_REGION_MISMATCH")
    return {
        "reuse": False,
        "reason_code": reasons[0] if reasons else "E_QUERY_MEANING_CHANGED",
        "reasons": reasons,
        "message": "query definition does not match; rebind or refuse stored evidence",
    }


def require_complete_contract(
    contract: QueryContract, identity: AcquisitionIdentity | None = None
) -> dict[str, Any]:
    """Incomplete contracts cannot be used at execute or reuse entries.

    identity_sha256 is always required. Energy edges are required when a live
    identity has an EnergyFilter. Unbound reuse (no identity hash and no edges)
    is also refused so a missing edge list is not treated as a match.
    """

    issues: list[str] = []
    if contract.identity_sha256 is None:
        issues.append("E_IDENTITY_UNSPECIFIED")
    if identity is not None:
        if energy_edges_from_identity(identity) is not None and contract.energy_edges_eV is None:
            issues.append("E_ENERGY_UNSPECIFIED")
    elif contract.energy_edges_eV is None and contract.identity_sha256 is None:
        # Unbound reuse: refuse rather than treat a missing edge list as a match.
        issues.append("E_ENERGY_UNSPECIFIED")
    return {
        "ok": not issues,
        "reason_code": None if not issues else issues[0],
        "issues": issues,
        "message": "incomplete contract refused" if issues else "complete",
    }


def can_reuse_batch_evidence(
    stored_source: RunSource,
    new_source: RunSource,
    stored_contract: QueryContract,
    new_contract: QueryContract,
    *,
    kept_meanings: Sequence[str],
) -> dict[str, Any]:
    """Reuse already-extracted paired columns. Separate from definition reuse."""

    for item, label in ((stored_contract, "stored"), (new_contract, "new")):
        complete = require_complete_contract(item)
        if not complete["ok"]:
            return {
                "reuse": False,
                "must_reacquire": True,
                "reason_code": complete["reason_code"],
                "message": f"{label} contract incomplete: {complete['issues']}",
            }
    if ledger_mix_forbidden(stored_source, new_source):
        return {
            "reuse": False,
            "must_reacquire": True,
            "reason_code": "E_RUN_SOURCE_MISMATCH",
            "message": "materials/geometry/seed/nuclear data do not match; do not splice batches",
        }
    definition = can_reuse_query_definition(stored_contract, new_contract)
    if definition["reuse"]:
        if new_contract.meaning_sha256 in kept_meanings:
            return {"reuse": True, "must_reacquire": False, "reason_code": None, "message": "paired columns already retained"}
        return {
            "reuse": False,
            "must_reacquire": True,
            "reason_code": "E_EVIDENCE_INSUFFICIENT",
            "message": "query meaning matches but paired columns were not retained",
        }
    if new_contract.meaning_sha256 in kept_meanings:
        return {
            "reuse": True,
            "must_reacquire": False,
            "reason_code": None,
            "message": "a different declared name maps onto retained paired columns",
        }
    return {
        "reuse": False,
        "must_reacquire": True,
        "reason_code": definition["reason_code"],
        "message": definition["message"],
    }


def contract_columns(identity: AcquisitionIdentity, contract: QueryContract) -> tuple[int, int]:
    bound = bind_region(identity, contract.region_spec())
    return two_score_column_indices(
        identity, bound["selection"], contract.numerator, contract.denominator, contract.nuclide
    )


def preflight_contract(identity: AcquisitionIdentity, contract: QueryContract, source: RunSource | None = None) -> dict[str, Any]:
    """Check query vs identity before transport. Does not run OpenMC."""

    issues: list[dict[str, str]] = []
    complete = require_complete_contract(contract, identity)
    if not complete["ok"]:
        for code in complete["issues"]:
            issues.append({"code": code, "message": complete["message"]})
    spec = contract.region_spec()
    try:
        spec.validate()
        bind_region(identity, spec)
        contract_columns(identity, contract)
    except AcquisitionError as exc:
        issues.append({"code": exc.code, "message": str(exc)})
    if contract.estimator != identity.estimator:
        issues.append({"code": "E_NORMALIZATION_MISMATCH", "message": "estimator differs from identity"})
    if contract.nuclide not in identity.nuclides:
        issues.append({"code": "E_NORMALIZATION_MISMATCH", "message": f"nuclide {contract.nuclide} not on identity"})
    if contract.numerator not in identity.scores or contract.denominator not in identity.scores:
        issues.append({"code": "E_QUERY_MEANING_CHANGED", "message": "score names not on identity"})
    id_edges = energy_edges_from_identity(identity)
    if id_edges is not None:
        if contract.energy_edges_eV is None:
            issues.append({"code": "E_ENERGY_UNSPECIFIED", "message": "contract must bind actual energy edges, not only energy_bin"})
        elif tuple(float(v) for v in contract.energy_edges_eV) != tuple(id_edges):
            issues.append({"code": "E_ENERGY_EDGES_MISMATCH", "message": "energy edges do not match the identity EnergyFilter"})
        n_bins = max(0, len(id_edges) - 1)
        if contract.energy_bin < 0 or contract.energy_bin >= n_bins:
            issues.append({"code": "E_ENERGY_MISMATCH", "message": "energy_bin outside identity edges"})
    if contract.identity_sha256 and contract.identity_sha256 != identity.sha256:
        issues.append({"code": "E_IDENTITY_MISMATCH", "message": "contract identity_sha256 does not match the live identity"})
    return {
        "ok": not issues,
        "issues": issues,
        "meaning_sha256": contract.meaning_sha256,
        "run_source_sha256": None if source is None else source.sha256,
        "would_transport": not issues,
    }
