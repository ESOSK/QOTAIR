"""Bind a model's own RegularMesh rather than an overlay grid."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from xml.etree import ElementTree as ET

import numpy as np

from .identity import AcquisitionError, FilterSpec


def _as_ints(values: Any) -> list[int]:
    return [int(v) for v in np.asarray(values, dtype=float).reshape(-1)]


def _as_floats(values: Any) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=float).reshape(-1)]


def is_regular_mesh(mesh: Any) -> bool:
    return type(mesh).__name__ == "RegularMesh"


def iter_model_meshes(model: Any) -> Iterable[Any]:
    seen: set[int] = set()
    meshes = getattr(model, "meshes", None)
    if meshes is not None:
        try:
            items = list(meshes)
        except TypeError:
            try:
                items = [meshes[key] for key in meshes]
            except Exception:
                items = []
        for mesh in items:
            if isinstance(mesh, int):
                try:
                    mesh = meshes[mesh]
                except Exception:
                    continue
            mesh_id = getattr(mesh, "id", None)
            if mesh_id is None or int(mesh_id) in seen:
                continue
            seen.add(int(mesh_id))
            yield mesh
    for tally in getattr(model, "tallies", None) or []:
        for item in getattr(tally, "filters", None) or []:
            mesh = getattr(item, "mesh", None)
            mesh_id = getattr(mesh, "id", None) if mesh is not None else None
            if mesh is None or mesh_id is None or int(mesh_id) in seen:
                continue
            seen.add(int(mesh_id))
            yield mesh


def _floats(text: str | None) -> list[float]:
    if not text:
        return []
    return [float(part) for part in text.split()]


def _ints(text: str | None) -> list[int]:
    if not text:
        return []
    return [int(float(part)) for part in text.split()]


def parse_regular_mesh_records(path: Path) -> list[dict[str, Any]]:
    """Read RegularMesh entries from tallies.xml or model.xml. Skip cylindrical/unstructured."""

    tree = ET.parse(path)
    records: list[dict[str, Any]] = []
    for elem in tree.iter("mesh"):
        mesh_type = (elem.get("type") or "regular").strip().lower()
        if mesh_type not in {"", "regular"}:
            continue
        dimension = _ints(elem.findtext("dimension"))
        lower = _floats(elem.findtext("lower_left"))
        upper = _floats(elem.findtext("upper_right"))
        if not dimension or not lower or not upper:
            continue
        mesh_id = elem.get("id")
        if mesh_id is None:
            continue
        while len(dimension) < 3:
            dimension.append(1)
        if len(lower) == 1:
            lower = [lower[0], -0.5, -0.5]
            upper = [upper[0], 0.5, 0.5]
        elif len(lower) == 2:
            lower = [lower[0], lower[1], -0.5]
            upper = [upper[0], upper[1], 0.5]
        if len(lower) != 3 or len(upper) != 3:
            continue
        records.append(
            {
                "id": int(mesh_id),
                "nx": int(dimension[0]),
                "ny": int(dimension[1]),
                "nz": int(dimension[2]),
                "lower_left": tuple(lower),
                "upper_right": tuple(upper),
                "source": str(path),
            }
        )
    return records


def spec_from_mesh_record(record: Mapping[str, Any] | dict[str, Any]) -> FilterSpec:
    return FilterSpec.native_mesh(
        int(record["id"]),
        int(record["nx"]),
        int(record["ny"]),
        int(record["nz"]),
        record["lower_left"],
        record["upper_right"],
    )


def candidate_mesh_xml_paths(xml_dir: Path, model_dir: Path | None = None) -> list[Path]:
    ordered: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        if not path.is_file():
            return
        key = str(path.resolve())
        if key in seen:
            return
        seen.add(key)
        ordered.append(path)

    add(xml_dir / "tallies.xml")
    add(xml_dir / "model.xml")
    if model_dir is not None:
        add(model_dir / "results" / "tallies.xml")
        add(model_dir / "results" / "model.xml")
        for pattern in ("tallies.xml", "model.xml"):
            for path in sorted(model_dir.rglob(pattern)):
                add(path)
    return ordered


def _record_bins(record: Mapping[str, Any]) -> int:
    return max(1, int(record["nx"])) * max(1, int(record["ny"])) * max(1, int(record["nz"]))


def records_from_xml_paths(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Prefer RegularMesh on the loaded XML directory; otherwise take the richest file."""

    ranked_paths = list(paths)
    local = ranked_paths[:2]
    rest = ranked_paths[2:]

    def _from(path: Path) -> list[dict[str, Any]]:
        try:
            records = parse_regular_mesh_records(path)
        except Exception:
            return []
        return sorted(records, key=lambda item: (-_record_bins(item), int(item["id"])))

    for path in local:
        records = _from(path)
        if records:
            return records
    best: list[dict[str, Any]] = []
    best_bins = -1
    for path in rest:
        records = _from(path)
        if not records:
            continue
        bins = _record_bins(records[0])
        if bins > best_bins:
            best = records
            best_bins = bins
    return best


def regular_mesh_from_record(record: Mapping[str, Any] | dict[str, Any]) -> Any:
    import openmc

    mesh = openmc.RegularMesh()
    mesh.id = int(record["id"])
    mesh.dimension = [int(record["nx"]), int(record["ny"]), int(record["nz"])]
    mesh.lower_left = [float(v) for v in record["lower_left"]]
    mesh.upper_right = [float(v) for v in record["upper_right"]]
    return mesh


def place_regular_mesh(model: Any, mesh: Any) -> None:
    meshes = getattr(model, "meshes", None)
    existing: list[Any] = []
    if meshes is not None:
        try:
            existing = list(meshes)
        except TypeError:
            try:
                existing = [meshes[key] for key in meshes]
            except Exception:
                existing = []
        cleaned: list[Any] = []
        for item in existing:
            if isinstance(item, int) and meshes is not None:
                try:
                    item = meshes[item]
                except Exception:
                    continue
            if getattr(item, "id", None) is not None:
                cleaned.append(item)
        existing = cleaned
    ids = {int(item.id) for item in existing}
    if int(mesh.id) in ids:
        return
    model.meshes = existing + [mesh]


def _mesh_bin_count(mesh: Any) -> int:
    dim = _as_ints(getattr(mesh, "dimension", [1]))
    total = 1
    for value in dim or [1]:
        total *= max(1, int(value))
    return total


def discover_regular_meshes(model: Any, xml_paths: Sequence[Path] | None = None) -> list[Any]:
    found = [mesh for mesh in iter_model_meshes(model) if is_regular_mesh(mesh)]
    if found:
        return sorted(found, key=lambda mesh: (-_mesh_bin_count(mesh), int(mesh.id)))
    if not xml_paths:
        return []
    placed: list[Any] = []
    for record in records_from_xml_paths(xml_paths):
        mesh = regular_mesh_from_record(record)
        place_regular_mesh(model, mesh)
        placed.append(mesh)
    return sorted(placed, key=lambda mesh: (-_mesh_bin_count(mesh), int(mesh.id)))


def find_regular_mesh(model: Any, mesh_id: int) -> Any:
    wanted = int(mesh_id)
    for mesh in discover_regular_meshes(model):
        if int(mesh.id) == wanted:
            return mesh
    raise AcquisitionError("E_NATIVE_MESH_MISSING", f"RegularMesh id={wanted} is not on the model")


def spec_from_regular_mesh(mesh: Any) -> FilterSpec:
    if not is_regular_mesh(mesh):
        raise AcquisitionError("E_MESH_NOT_REGULAR", type(mesh).__name__)
    dim = _as_ints(mesh.dimension)
    lower = _as_floats(mesh.lower_left)
    upper = _as_floats(mesh.upper_right)
    if len(dim) == 1:
        nx, ny, nz = dim[0], 1, 1
        if len(lower) == 1:
            lower = [lower[0], -0.5, -0.5]
            upper = [upper[0], 0.5, 0.5]
    elif len(dim) == 2:
        nx, ny, nz = dim[0], dim[1], 1
        if len(lower) == 2:
            lower = [lower[0], lower[1], -0.5]
            upper = [upper[0], upper[1], 0.5]
    elif len(dim) == 3:
        nx, ny, nz = dim
    else:
        raise AcquisitionError("E_MESH_BINS", f"RegularMesh dimension rank {len(dim)}")
    if len(lower) != 3 or len(upper) != 3:
        raise AcquisitionError("E_MESH_BINS", "RegularMesh bounds must have length 3 after padding")
    return FilterSpec.native_mesh(int(mesh.id), nx, ny, nz, lower, upper)
