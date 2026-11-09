"""Unique-instance CellFilter helpers for full-core models.

The frozen compiler already emits CellFilter. This module only finds cells
that occur once so a full-core assembly can use that existing path.
"""
from __future__ import annotations

from typing import Any

from .identity import AcquisitionError, FilterSpec


def _cell_map(model: Any) -> dict[int, Any]:
    geometry = getattr(model, "geometry", None)
    if geometry is None:
        raise AcquisitionError("E_NO_UNIQUE_CELL", "model has no geometry")
    if hasattr(geometry, "determine_paths"):
        geometry.determine_paths()
    cells = geometry.get_all_cells()
    return {int(cid): cell for cid, cell in cells.items()}


def unique_cell_ids(model: Any) -> tuple[int, ...]:
    found: list[int] = []
    for cell_id, cell in sorted(_cell_map(model).items()):
        n = int(getattr(cell, "num_instances", 0) or 0)
        if n == 1:
            found.append(cell_id)
    return tuple(found)


def unique_cell_spec(model: Any, cell_id: int | None = None) -> FilterSpec:
    ids = unique_cell_ids(model)
    if not ids:
        raise AcquisitionError("E_NO_UNIQUE_CELL", "no cell with exactly one instance")
    if cell_id is None:
        return FilterSpec.cell((ids[0],))
    wanted = int(cell_id)
    if wanted not in ids:
        raise AcquisitionError("E_CELL_NOT_UNIQUE", f"cell {wanted} does not have exactly one instance")
    return FilterSpec.cell((wanted,))
