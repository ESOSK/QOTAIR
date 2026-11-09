"""Pick physically named regions: fuel pin, assembly universe, fissionable material."""
from __future__ import annotations

from typing import Any

from .identity import AcquisitionError

FISSIONABLE = {
    "U233",
    "U235",
    "U238",
    "Pu239",
    "Pu240",
    "Pu241",
    "Th232",
    "Am241",
}


def canonical_nuclide(name: str) -> str:
    return str(name).split(".")[0]


def material_nuclide_names(material: Any) -> set[str]:
    names: set[str] = set()
    nuclides = getattr(material, "nuclides", None)
    if nuclides:
        for item in nuclides:
            names.add(canonical_nuclide(getattr(item, "name", item)))
    densities = getattr(material, "get_nuclide_densities", None)
    if callable(densities):
        try:
            for key in densities():
                names.add(canonical_nuclide(key))
        except Exception:
            pass
    return {item for item in names if item}


def is_fissionable_material(material: Any) -> bool:
    names = material_nuclide_names(material)
    return bool(names & FISSIONABLE)


def model_materials(model: Any) -> dict[int, Any]:
    found: dict[int, Any] = {}
    geometry = getattr(model, "geometry", None)
    if geometry is not None and hasattr(geometry, "get_all_materials"):
        try:
            for mid, material in geometry.get_all_materials().items():
                found[int(mid)] = material
        except Exception:
            pass
    for material in getattr(model, "materials", None) or []:
        mid = getattr(material, "id", None)
        if mid is not None:
            found[int(mid)] = material
    return found


def model_nuclide_names(model: Any) -> set[str]:
    names: set[str] = set()
    for material in model_materials(model).values():
        names.update(material_nuclide_names(material))
    return names


def require_nuclides_on_model(model: Any, nuclides: tuple[str, ...]) -> None:
    wanted = [name for name in nuclides if name != "total"]
    if not wanted:
        return
    known = model_nuclide_names(model)
    canonical = {canonical_nuclide(item) for item in known} | known
    missing = [name for name in wanted if name not in canonical and canonical_nuclide(name) not in canonical]
    if missing:
        raise AcquisitionError("E_NUCLIDE_MISSING", f"nuclides not in the model: {missing}")


def _cell_fill_material(cell: Any) -> Any | None:
    fill = getattr(cell, "fill", None)
    if fill is None:
        return None
    if type(fill).__name__ == "Material":
        return fill
    return None


def pick_distribcell_target(model: Any) -> dict[str, Any]:
    """Prefer a fissionable cell with more than one instance (a pin replica)."""

    geometry = getattr(model, "geometry", None)
    if geometry is None:
        raise AcquisitionError("E_DISTRIBCELL_CELL", "model has no geometry")
    if hasattr(geometry, "determine_paths"):
        geometry.determine_paths()
    cells = geometry.get_all_cells()
    fissionable: list[tuple[int, int, str]] = []
    others: list[tuple[int, int]] = []
    for cid, cell in cells.items():
        n = int(getattr(cell, "num_instances", 0) or 0)
        if n < 1:
            continue
        material = _cell_fill_material(cell)
        if material is not None and is_fissionable_material(material):
            names = sorted(material_nuclide_names(material) & FISSIONABLE)
            fissionable.append((n, int(cid), ",".join(names)))
        else:
            others.append((n, int(cid)))
    if fissionable:
        fissionable.sort(reverse=True)
        n, cid, nuclides = fissionable[0]
        kind = "fissionable"
    else:
        if not others:
            raise AcquisitionError("E_NO_DISTRIBCELL", "no cell with instances")
        others.sort(reverse=True)
        n, cid = others[0]
        nuclides = ""
        kind = "most_instances"
    return {
        "cell_id": cid,
        "instances": n,
        "kind": kind,
        "fissionable_nuclides": nuclides,
        "distributed": n > 1,
    }


def pick_universe_target(model: Any) -> dict[str, Any]:
    """Prefer a non-root universe used as a lattice element (assembly-like)."""

    geometry = model.geometry
    universes = geometry.get_all_universes()
    if not universes:
        raise AcquisitionError("E_NO_UNIVERSE", "model has no universes")
    root = getattr(geometry, "root_universe", None)
    root_id = getattr(root, "id", None)
    lattice_fills: set[int] = set()
    for cell in geometry.get_all_cells().values():
        fill = getattr(cell, "fill", None)
        name = type(fill).__name__ if fill is not None else ""
        if "Lattice" not in name:
            continue
        universes_attr = getattr(fill, "universes", None)
        if universes_attr is None:
            continue
        try:
            flat = np_array_ids(universes_attr)
        except Exception:
            flat = []
        lattice_fills.update(flat)
    if lattice_fills:
        uid = sorted(i for i in lattice_fills if i != root_id) or sorted(lattice_fills)
        chosen = int(uid[0])
        kind = "lattice_fill"
    else:
        ids = sorted(int(i) for i in universes)
        chosen = next((i for i in ids if i != root_id), ids[0])
        kind = "non_root" if chosen != root_id else "root"
    return {"universe_id": chosen, "kind": kind, "root_id": root_id}


def np_array_ids(values: Any) -> list[int]:
    ids: list[int] = []

    def walk(item: Any) -> None:
        if item is None:
            return
        uid = getattr(item, "id", None)
        if uid is not None and type(item).__name__ == "Universe":
            ids.append(int(uid))
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                walk(child)
            return
        try:
            import numpy as np

            array = np.asarray(item, dtype=object)
            for child in array.reshape(-1):
                walk(child)
        except Exception:
            pass

    walk(values)
    return ids


def pick_fissionable_material(model: Any) -> dict[str, Any]:
    materials = model_materials(model)
    ranked: list[tuple[int, int, tuple[str, ...]]] = []
    for mid, material in materials.items():
        names = tuple(sorted(material_nuclide_names(material) & FISSIONABLE))
        if names:
            ranked.append((len(names), int(mid), names))
    if not ranked:
        raise AcquisitionError("E_NO_FISSIONABLE_MATERIAL", "no fissionable material on the model")
    ranked.sort(reverse=True)
    _, mid, names = ranked[0]
    preferred = "U235" if "U235" in names else names[0]
    return {"material_id": mid, "fissionable_nuclides": list(names), "nuclide": preferred}
