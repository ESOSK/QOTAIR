"""Slice a multi-bin ΔSUM ledger down to two score columns for one filter bin."""
from __future__ import annotations

from typing import Mapping

import numpy as np

from .identity import AcquisitionError, AcquisitionIdentity


def filter_bin_strides(identity: AcquisitionIdentity) -> tuple[tuple[str, int, int], ...]:
    """Last filter on the identity changes fastest (OpenMC row-major bins)."""

    counts = [max(1, item.bin_count()) for item in identity.filters]
    n = len(counts)
    strides = [1] * n
    for i in range(n - 2, -1, -1):
        strides[i] = strides[i + 1] * counts[i + 1]
    return tuple(
        (item.filter_type, counts[i], strides[i]) for i, item in enumerate(identity.filters)
    )


def first_bin_selection(identity: AcquisitionIdentity) -> dict[str, int]:
    return {item.filter_type: 0 for item in identity.filters}


def mesh_shape(identity: AcquisitionIdentity) -> tuple[int, int, int]:
    for item in identity.filters:
        if item.filter_type == "MeshFilter" and item.bins and item.bins[0] == "RegularMesh":
            return int(item.bins[1]), int(item.bins[2]), int(item.bins[3])
    raise AcquisitionError("E_BIN_SELECTION", "identity has no RegularMesh")


def mesh_linear_index(nx: int, ny: int, nz: int, i: int, j: int, k: int = 0) -> int:
    nx_i, ny_i, nz_i = int(nx), int(ny), int(nz)
    if not (0 <= i < nx_i and 0 <= j < ny_i and 0 <= k < nz_i):
        raise AcquisitionError("E_BIN_SELECTION", f"mesh ijk ({i},{j},{k}) outside {nx_i}x{ny_i}x{nz_i}")
    return int(i) + nx_i * (int(j) + ny_i * int(k))


def mesh_center_ijk(nx: int, ny: int, nz: int) -> tuple[int, int, int]:
    return int(nx) // 2, int(ny) // 2, int(nz) // 2


def mesh_ijk_selection(
    identity: AcquisitionIdentity,
    i: int,
    j: int,
    k: int = 0,
    *,
    energy_bin: int = 0,
) -> dict[str, int]:
    nx, ny, nz = mesh_shape(identity)
    chosen = first_bin_selection(identity)
    chosen["MeshFilter"] = mesh_linear_index(nx, ny, nz, i, j, k)
    if "EnergyFilter" in chosen:
        n_energy = next(item.bin_count() for item in identity.filters if item.filter_type == "EnergyFilter")
        if energy_bin < 0 or energy_bin >= n_energy:
            raise AcquisitionError("E_BIN_SELECTION", f"energy bin {energy_bin} outside [0, {n_energy})")
        chosen["EnergyFilter"] = int(energy_bin)
    return chosen


def center_mesh_selection(identity: AcquisitionIdentity, *, energy_bin: int = 0) -> dict[str, int]:
    nx, ny, nz = mesh_shape(identity)
    i, j, k = mesh_center_ijk(nx, ny, nz)
    return mesh_ijk_selection(identity, i, j, k, energy_bin=energy_bin)


def selected_filter_index(identity: AcquisitionIdentity, selection: Mapping[str, int]) -> int:
    wanted = {item.filter_type for item in identity.filters}
    got = set(selection)
    if wanted != got:
        raise AcquisitionError(
            "E_BIN_SELECTION",
            f"selection keys {sorted(got)} must match filters {sorted(wanted)}",
        )
    index = 0
    for filter_type, n_bins, stride in filter_bin_strides(identity):
        bin_index = int(selection[filter_type])
        if bin_index < 0 or bin_index >= n_bins:
            raise AcquisitionError(
                "E_BIN_SELECTION",
                f"{filter_type} bin {bin_index} is outside [0, {n_bins})",
            )
        index += bin_index * stride
    return index


def ledger_column_count(identity: AcquisitionIdentity) -> int:
    return identity.filter_bin_count() * len(identity.nuclides) * len(identity.scores)


def two_score_column_indices(
    identity: AcquisitionIdentity,
    selection: Mapping[str, int],
    numerator: str,
    denominator: str,
    nuclide: str = "total",
) -> tuple[int, int]:
    try:
        nuclide_index = list(identity.nuclides).index(nuclide)
        num_index = list(identity.scores).index(numerator)
        den_index = list(identity.scores).index(denominator)
    except ValueError as exc:
        raise AcquisitionError("E_BIN_SELECTION", "nuclide or score is not on the identity") from exc
    filter_index = selected_filter_index(identity, selection)
    base = (filter_index * len(identity.nuclides) + nuclide_index) * len(identity.scores)
    return base + num_index, base + den_index


def slice_two_scores(
    ledger: np.ndarray,
    identity: AcquisitionIdentity,
    selection: Mapping[str, int],
    numerator: str,
    denominator: str,
    nuclide: str = "total",
) -> np.ndarray:
    """Return shape (n_batches, 2) as [numerator, denominator] for one bin."""

    values = np.asarray(ledger, dtype=float)
    expected = ledger_column_count(identity)
    if values.ndim != 2 or values.shape[1] != expected:
        raise AcquisitionError(
            "E_LEDGER_SHAPE",
            f"ledger shape {values.shape} does not match identity columns {expected}",
        )
    try:
        nuclide_index = list(identity.nuclides).index(nuclide)
    except ValueError as exc:
        raise AcquisitionError("E_BIN_SELECTION", f"nuclide {nuclide!r} is not on the identity") from exc
    try:
        num_index = list(identity.scores).index(numerator)
        den_index = list(identity.scores).index(denominator)
    except ValueError as exc:
        raise AcquisitionError("E_BIN_SELECTION", "numerator or denominator is not an identity score") from exc
    filter_index = selected_filter_index(identity, selection)
    n_nuclides = len(identity.nuclides)
    n_scores = len(identity.scores)
    base = (filter_index * n_nuclides + nuclide_index) * n_scores
    two = np.column_stack([values[:, base + num_index], values[:, base + den_index]])
    if not np.isfinite(two).all():
        raise AcquisitionError("E_NONFINITE", "selected columns contain non-finite values")
    return two


def slice_score_at_selection(
    ledger: np.ndarray,
    identity: AcquisitionIdentity,
    selection: Mapping[str, int],
    score: str,
    nuclide: str = "total",
) -> np.ndarray:
    values = np.asarray(ledger, dtype=float)
    expected = ledger_column_count(identity)
    if values.ndim != 2 or values.shape[1] != expected:
        raise AcquisitionError(
            "E_LEDGER_SHAPE",
            f"ledger shape {values.shape} does not match identity columns {expected}",
        )
    try:
        nuclide_index = list(identity.nuclides).index(nuclide)
        score_index = list(identity.scores).index(score)
    except ValueError as exc:
        raise AcquisitionError("E_BIN_SELECTION", "nuclide or score is not on the identity") from exc
    filter_index = selected_filter_index(identity, selection)
    column = (filter_index * len(identity.nuclides) + nuclide_index) * len(identity.scores) + score_index
    series = values[:, column].astype(float, copy=True)
    if not np.isfinite(series).all():
        raise AcquisitionError("E_NONFINITE", "selected column contains non-finite values")
    return series


def slice_two_energy_groups(
    ledger: np.ndarray,
    identity: AcquisitionIdentity,
    *,
    numerator_energy: int,
    denominator_energy: int,
    score: str = "flux",
    nuclide: str = "total",
    mesh_ijk: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Two columns of the same score at two EnergyFilter bins. Not a nested ratio."""

    base = first_bin_selection(identity)
    if "EnergyFilter" not in base:
        raise AcquisitionError("E_BIN_SELECTION", "fast/thermal slice needs EnergyFilter")
    if mesh_ijk is not None:
        base = mesh_ijk_selection(identity, *mesh_ijk, energy_bin=0)
    num_sel = dict(base)
    den_sel = dict(base)
    num_sel["EnergyFilter"] = int(numerator_energy)
    den_sel["EnergyFilter"] = int(denominator_energy)
    left = slice_score_at_selection(ledger, identity, num_sel, score, nuclide)
    right = slice_score_at_selection(ledger, identity, den_sel, score, nuclide)
    return np.column_stack([left, right])
