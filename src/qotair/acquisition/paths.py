"""Inspectable nested-universe / lattice paths and mesh index convention."""
from __future__ import annotations

from typing import Any

from .identity import AcquisitionError, AcquisitionIdentity


MESH_INDEX_CONVENTION = {
    "index_base": 0,
    "origin": "lower_left",
    "fastest_index": "i",
    "linear_index_formula": "i + nx * (j + ny * k)",
    "axial": "k increases toward upper_right z",
}


def mesh_bounds_from_identity(identity: AcquisitionIdentity) -> dict[str, Any]:
    for item in identity.filters:
        if item.filter_type == "MeshFilter" and item.bins and item.bins[0] == "RegularMesh":
            nx, ny, nz = int(item.bins[1]), int(item.bins[2]), int(item.bins[3])
            return {
                **MESH_INDEX_CONVENTION,
                "mesh_shape": [nx, ny, nz],
                "lower_left": [float(item.bins[4]), float(item.bins[5]), float(item.bins[6])],
                "upper_right": [float(item.bins[7]), float(item.bins[8]), float(item.bins[9])],
                "native_mesh_id": item.native_mesh_id(),
            }
    raise AcquisitionError("E_BIN_SELECTION", "identity has no RegularMesh")


def distribcell_instance_path(model: Any, cell_id: int, instance_index: int) -> str:
    """OpenMC instance path after determine_paths. Not a most-instances pick."""

    geometry = getattr(model, "geometry", None)
    if geometry is None:
        raise AcquisitionError("E_DISTRIBCELL_CELL", "model has no geometry")
    if hasattr(geometry, "determine_paths"):
        geometry.determine_paths()
    cells = geometry.get_all_cells()
    cell = cells.get(int(cell_id))
    if cell is None:
        raise AcquisitionError("E_DISTRIBCELL_CELL", f"cell {int(cell_id)} is not in the geometry")
    paths = list(getattr(cell, "paths", None) or [])
    inst = int(instance_index)
    if inst < 0 or inst >= len(paths):
        raise AcquisitionError(
            "E_BIN_SELECTION",
            f"instance {inst} has no path among {len(paths)} OpenMC paths for cell {int(cell_id)}",
        )
    return str(paths[inst])
