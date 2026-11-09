"""Whether a declared region still maps after a material / geometry state change."""
from __future__ import annotations

from typing import Any

from .adapter import distribcell_instance_count
from .bind import RegionSpec
from .identity import AcquisitionError
from .paths import distribcell_instance_path


def _universe_ids(model: Any) -> set[int]:
    geometry = getattr(model, "geometry", None)
    if geometry is None or not hasattr(geometry, "get_all_universes"):
        return set()
    try:
        return {int(uid) for uid in geometry.get_all_universes()}
    except Exception:
        return set()


def _material_ids(model: Any) -> set[int]:
    found: set[int] = set()
    geometry = getattr(model, "geometry", None)
    if geometry is not None and hasattr(geometry, "get_all_materials"):
        try:
            found.update(int(mid) for mid in geometry.get_all_materials())
        except Exception:
            pass
    for material in getattr(model, "materials", None) or []:
        mid = getattr(material, "id", None)
        if mid is not None:
            found.add(int(mid))
    return found


def _mesh_bounds(model: Any) -> dict[str, Any] | None:
    from .native_mesh import discover_regular_meshes, spec_from_regular_mesh

    meshes = discover_regular_meshes(model)
    if not meshes:
        return None
    spec = spec_from_regular_mesh(meshes[0])
    bins = spec.bins
    return {
        "id": spec.native_mesh_id(),
        "shape": [int(bins[1]), int(bins[2]), int(bins[3])],
        "lower_left": [float(bins[4]), float(bins[5]), float(bins[6])],
        "upper_right": [float(bins[7]), float(bins[8]), float(bins[9])],
    }


def snapshot_correspondence(kind: str, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Model-free reject/update cases for tests and declared-range counterexamples."""

    rec = {"kind": kind, "status": "still_valid"}
    if kind == "distribcell":
        if left.get("instances") != right.get("instances"):
            return {**rec, "status": "must_update", "reason": "instance count changed"}
        if left.get("path") != right.get("path"):
            return {**rec, "status": "must_update", "reason": "instance path changed"}
        return rec
    if kind == "cell":
        if left.get("cell_id") not in (right.get("cell_ids") or [right.get("cell_id")]):
            return {**rec, "status": "cannot_continue", "reason": "cell missing on one state"}
        return rec
    if kind == "mesh_ijk":
        if left.get("bounds") is None or right.get("bounds") is None:
            return {**rec, "status": "cannot_continue", "reason": "RegularMesh bounds missing on one state"}
        if left.get("bounds") != right.get("bounds"):
            return {**rec, "status": "must_update", "reason": "mesh physical bounds changed"}
        return rec
    if kind == "universe":
        if left.get("universe_id") not in (right.get("universe_ids") or []):
            return {**rec, "status": "cannot_continue", "reason": "universe missing on one state"}
        return rec
    if kind == "material":
        if left.get("material_id") not in (right.get("material_ids") or []):
            return {**rec, "status": "cannot_continue", "reason": "material missing on one state"}
        return rec
    return {**rec, "status": "cannot_continue", "reason": "unhandled region kind"}


def region_correspondence(left_model: Any, right_model: Any, spec: RegionSpec) -> dict[str, Any]:
    """still_valid / must_update / cannot_continue. Does not splice ledgers."""

    spec.validate()
    rec: dict[str, Any] = {"kind": spec.kind, "name": spec.name, "status": "still_valid"}
    try:
        if spec.kind == "distribcell":
            cid = int(spec.cell_id)  # type: ignore[arg-type]
            inst = 0 if spec.instance_index is None else int(spec.instance_index)
            n_left = distribcell_instance_count(left_model, cid)
            n_right = distribcell_instance_count(right_model, cid)
            rec.update({"cell_id": cid, "instance_index": inst, "instances_left": n_left, "instances_right": n_right})
            if n_left != n_right:
                rec["status"] = "must_update"
                rec["reason"] = "instance count changed"
                return rec
            rec["path_left"] = distribcell_instance_path(left_model, cid, inst)
            rec["path_right"] = distribcell_instance_path(right_model, cid, inst)
            if rec["path_left"] != rec["path_right"]:
                rec["status"] = "must_update"
                rec["reason"] = "instance path changed"
                return rec
            rec["geometry_path"] = rec["path_left"]
            return rec
        if spec.kind == "cell":
            cid = int(spec.cell_id)  # type: ignore[arg-type]
            left_cells = left_model.geometry.get_all_cells()
            right_cells = right_model.geometry.get_all_cells()
            rec["cell_id"] = cid
            if cid not in left_cells or cid not in right_cells:
                rec["status"] = "cannot_continue"
                rec["reason"] = "cell missing on one state"
                return rec
            return rec
        if spec.kind == "mesh_ijk":
            rec["mesh_ijk"] = list(spec.mesh_ijk) if spec.mesh_ijk is not None else None
            left_b = _mesh_bounds(left_model)
            right_b = _mesh_bounds(right_model)
            rec["bounds_left"] = left_b
            rec["bounds_right"] = right_b
            if left_b is None or right_b is None:
                rec["status"] = "cannot_continue"
                rec["reason"] = "RegularMesh bounds missing on one state"
                return rec
            if left_b != right_b:
                rec["status"] = "must_update"
                rec["reason"] = "mesh physical bounds changed"
                return rec
            i, j, k = spec.mesh_ijk  # type: ignore[misc]
            nx, ny, nz = left_b["shape"]
            if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
                rec["status"] = "cannot_continue"
                rec["reason"] = "mesh ijk outside bounds"
            return rec
        if spec.kind == "universe":
            uid = int(spec.universe_id)  # type: ignore[arg-type]
            rec["universe_id"] = uid
            if uid not in _universe_ids(left_model) or uid not in _universe_ids(right_model):
                rec["status"] = "cannot_continue"
                rec["reason"] = "universe missing on one state"
            return rec
        if spec.kind == "material":
            mid = int(spec.material_id)  # type: ignore[arg-type]
            rec["material_id"] = mid
            if mid not in _material_ids(left_model) or mid not in _material_ids(right_model):
                rec["status"] = "cannot_continue"
                rec["reason"] = "material missing on one state"
            return rec
        rec["status"] = "cannot_continue"
        rec["reason"] = "unhandled region kind"
        return rec
    except AcquisitionError as exc:
        return {"kind": spec.kind, "name": spec.name, "status": "cannot_continue", "reason_code": exc.code, "message": str(exc)}
