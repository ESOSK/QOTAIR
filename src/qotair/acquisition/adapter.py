"""Insert an acquisition identity onto an isolated OpenMC model copy."""
from __future__ import annotations

import copy
from typing import Any

from .identity import AcquisitionError, AcquisitionIdentity, FilterSpec
from .native_mesh import find_regular_mesh, is_regular_mesh
from .tables import require_registered_table
from .targets import require_nuclides_on_model


def require_distribcell_instances(*, actual: int, expected: int, cell_id: int) -> None:
    actual_n = int(actual)
    expected_n = int(expected)
    if actual_n < 1:
        raise AcquisitionError(
            "E_DISTRIBCELL_INSTANCES",
            f"cell {int(cell_id)} has no OpenMC instances",
        )
    if actual_n != expected_n:
        raise AcquisitionError(
            "E_DISTRIBCELL_INSTANCE_MISMATCH",
            f"cell {int(cell_id)}: OpenMC instances={actual_n}, expected={expected_n}",
        )


def distribcell_instance_count(model: Any, cell_id: int) -> int:
    geometry = getattr(model, "geometry", None)
    if geometry is None:
        raise AcquisitionError("E_DISTRIBCELL_CELL", "model has no geometry")
    if hasattr(geometry, "determine_paths"):
        geometry.determine_paths()
    cells = geometry.get_all_cells()
    wanted = int(cell_id)
    cell = cells.get(wanted)
    if cell is None:
        raise AcquisitionError("E_DISTRIBCELL_CELL", f"cell {wanted} is not in the geometry")
    n = getattr(cell, "num_instances", None)
    if n is None:
        raise AcquisitionError("E_DISTRIBCELL_INSTANCES", f"cell {wanted} has no OpenMC instance count")
    return int(n)


def _material_ids(model: Any) -> set[int]:
    found: set[int] = set()
    geometry = getattr(model, "geometry", None)
    if geometry is not None and hasattr(geometry, "get_all_materials"):
        try:
            materials = geometry.get_all_materials()
            found.update(int(mid) for mid in materials)
        except Exception:
            pass
    for material in getattr(model, "materials", None) or []:
        mid = getattr(material, "id", None)
        if mid is not None:
            found.add(int(mid))
    return found


def to_openmc_filter(spec: FilterSpec, model: Any | None = None) -> Any:
    import openmc

    if spec.filter_type == "EnergyFilter":
        return openmc.EnergyFilter(spec.bins)
    if spec.filter_type == "CellFilter":
        return openmc.CellFilter(list(spec.bins))
    if spec.filter_type == "ParticleFilter":
        return openmc.ParticleFilter(list(spec.bins))
    if spec.filter_type == "UniverseFilter":
        return openmc.UniverseFilter(list(spec.bins))
    if spec.filter_type == "MaterialFilter":
        return openmc.MaterialFilter(list(spec.bins))
    if spec.filter_type == "DistribcellFilter":
        return openmc.DistribcellFilter(int(spec.bins[0]))
    if spec.filter_type == "MeshFilter":
        bins = spec.bins
        if not bins or bins[0] != "RegularMesh" or len(bins) not in {10, 11}:
            raise AcquisitionError("E_FILTER_UNSUPPORTED", "MeshFilter supports RegularMesh only")
        _, nx, ny, nz, x0, y0, z0, x1, y1, z1 = bins[:10]
        native_id = spec.native_mesh_id()
        if native_id is not None:
            if model is None:
                raise AcquisitionError("E_NATIVE_MESH_MISSING", "native RegularMesh bind needs the model")
            mesh = find_regular_mesh(model, native_id)
            if not is_regular_mesh(mesh):
                raise AcquisitionError("E_MESH_NOT_REGULAR", type(mesh).__name__)
            return openmc.MeshFilter(mesh)
        mesh = openmc.RegularMesh()
        mesh.id = 192000 + int(nx) * 100 + int(ny) * 10 + int(nz)
        mesh.dimension = [int(nx), int(ny), int(nz)]
        mesh.lower_left = [float(x0), float(y0), float(z0)]
        mesh.upper_right = [float(x1), float(y1), float(z1)]
        return openmc.MeshFilter(mesh)
    if spec.filter_type == "EnergyFunctionFilter":
        kind, energies, y = spec.bins[0], spec.bins[1], spec.bins[2]
        if kind != "tabulated":
            raise AcquisitionError("E_FILTER_UNSUPPORTED", "EnergyFunctionFilter tabulated only")
        return openmc.EnergyFunctionFilter(list(energies), list(y))
    raise AcquisitionError("E_FILTER_UNSUPPORTED", spec.filter_type)


def _preflight(identity: AcquisitionIdentity, model: Any) -> None:
    named = [name for name in identity.nuclides if name != "total"]
    if named and any(score == "flux" for score in identity.scores):
        raise AcquisitionError(
            "E_NUCLIDE_FLUX",
            "OpenMC cannot tally flux for an individual nuclide; use two reaction scores",
        )
    require_nuclides_on_model(model, identity.nuclides)
    for spec in identity.filters:
        if spec.filter_type == "EnergyFunctionFilter":
            require_registered_table(spec.energy_function_table())
        if spec.filter_type == "DistribcellFilter":
            if len(spec.bins) < 2:
                raise AcquisitionError(
                    "E_DISTRIBCELL_INSTANCES",
                    "DistribcellFilter requires expected_instances; bind is not an instance count",
                )
            actual = distribcell_instance_count(model, int(spec.bins[0]))
            require_distribcell_instances(
                actual=actual,
                expected=int(spec.bins[1]),
                cell_id=int(spec.bins[0]),
            )
        if spec.filter_type == "MaterialFilter":
            known = _material_ids(model)
            missing = [mid for mid in spec.bins if int(mid) not in known]
            if missing:
                raise AcquisitionError("E_MATERIAL_MISSING", f"materials not on the model: {missing}")
        if spec.filter_type == "MeshFilter" and spec.native_mesh_id() is not None:
            find_regular_mesh(model, spec.native_mesh_id())


def attach_identity(model: Any, identity: AcquisitionIdentity, *, tally_id: int = 191000) -> tuple[Any, int]:
    import openmc

    isolated = copy.deepcopy(model)
    _preflight(identity, isolated)
    known = {int(t.id) for t in (isolated.tallies or [])}
    next_id = int(tally_id)
    while next_id in known:
        next_id += 1
    tally = openmc.Tally(tally_id=next_id, name=f"qotair_{identity.sha256[:12]}")
    filters = []
    filter_id = 191500
    for item in identity.filters:
        filt = to_openmc_filter(item, model=isolated)
        try:
            filt.id = filter_id
        except Exception:
            pass
        filter_id += 1
        filters.append(filt)
    tally.filters = filters
    tally.nuclides = list(identity.nuclides)
    tally.scores = list(identity.scores)
    tally.estimator = identity.estimator
    existing = list(isolated.tallies or [])
    isolated.tallies = openmc.Tallies(existing + [tally])
    return isolated, next_id
