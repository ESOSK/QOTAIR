"""Separate query identity, acquisition plan, run source, and inference rule."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .identity import AcquisitionIdentity


def _sha(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path | None) -> str:
    if not path:
        return ""
    target = Path(path)
    if not target.is_file():
        return ""
    digest = hashlib.sha256()
    with target.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RunSource:
    geometry: str = ""
    materials: str = ""
    nuclear_data: str = ""
    settings: str = ""
    seed: int | None = None
    software: str = "qotair-0.3.0"
    openmc: str = "0.15.3"
    geometry_sha256: str = ""
    materials_sha256: str = ""
    nuclear_data_sha256: str = ""
    settings_sha256: str = ""

    @classmethod
    def bind_files(
        cls,
        *,
        geometry: str | Path = "",
        materials: str | Path = "",
        nuclear_data: str | Path = "",
        settings: Mapping[str, Any] | str = "",
        seed: int | None = None,
        software: str = "qotair-0.3.0",
        openmc: str = "0.15.3",
    ) -> "RunSource":
        settings_text = settings if isinstance(settings, str) else json.dumps(settings, sort_keys=True)
        return cls(
            geometry=str(geometry or ""),
            materials=str(materials or ""),
            nuclear_data=str(nuclear_data or ""),
            settings=settings_text,
            seed=seed,
            software=software,
            openmc=openmc,
            geometry_sha256=file_sha256(geometry),
            materials_sha256=file_sha256(materials),
            nuclear_data_sha256=file_sha256(nuclear_data),
            settings_sha256=_sha(settings_text),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "geometry": self.geometry,
            "materials": self.materials,
            "nuclear_data": self.nuclear_data,
            "settings": self.settings,
            "seed": self.seed,
            "software": self.software,
            "openmc": self.openmc,
            "geometry_sha256": self.geometry_sha256,
            "materials_sha256": self.materials_sha256,
            "nuclear_data_sha256": self.nuclear_data_sha256,
            "settings_sha256": self.settings_sha256,
        }

    @property
    def sha256(self) -> str:
        return _sha(self.as_dict())


@dataclass(frozen=True)
class InferenceRule:
    router: str = "single-history-router"
    thresholds: str = "frozen T_den / hull; not retuned"
    block_count: int = 6
    table4: str = "unchanged"

    def as_dict(self) -> dict[str, Any]:
        return {
            "router": self.router,
            "thresholds": self.thresholds,
            "block_count": int(self.block_count),
            "table4": self.table4,
        }

    @property
    def sha256(self) -> str:
        return _sha(self.as_dict())


@dataclass
class QueryTask:
    """One (model, state, region, query, replicate) record."""

    task_id: str
    model: str
    state: str
    region_name: str
    query_id: str
    replicate: int = 0
    identity: AcquisitionIdentity | None = None
    run_source: RunSource = field(default_factory=RunSource)
    inference_rule: InferenceRule = field(default_factory=InferenceRule)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "model": self.model,
            "state": self.state,
            "region_name": self.region_name,
            "query_id": self.query_id,
            "replicate": int(self.replicate),
            "identity_sha256": None if self.identity is None else self.identity.sha256,
            "run_source_sha256": self.run_source.sha256,
            "inference_rule_sha256": self.inference_rule.sha256,
        }


def same_query_identity(left: AcquisitionIdentity, right: AcquisitionIdentity) -> bool:
    return left.sha256 == right.sha256


def ledger_mix_forbidden(left: RunSource, right: RunSource) -> bool:
    """Same query identity must not splice batches from different run sources."""

    return left.sha256 != right.sha256
