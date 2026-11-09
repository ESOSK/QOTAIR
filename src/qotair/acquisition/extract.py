"""Active-batch Delta-SUM extract for one tally id."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np


def peak_rss_bytes() -> int | None:
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB; macOS reports bytes.
        return int(rss) * 1024 if rss < 1 << 32 else int(rss)
    except Exception:
        return None


def extract_tally_ledger(case_dir: Path, *, tally_id: int, threads: int = 1) -> np.ndarray:
    import openmc.lib

    os.environ["OMP_NUM_THREADS"] = str(threads)
    old = Path.cwd()
    rows: list[np.ndarray] = []
    previous = None
    try:
        os.chdir(case_dir)
        with openmc.lib.run_in_memory(args=["-s", str(threads)], output=False):
            openmc.lib.simulation_init()
            try:
                for _ in openmc.lib.iter_batches():
                    tally = openmc.lib.tallies[tally_id]
                    realizations = int(tally.num_realizations)
                    if realizations <= 0:
                        continue
                    cumulative = np.asarray(tally.mean, dtype=float).reshape(-1) * realizations
                    batch = cumulative.copy() if previous is None else cumulative - previous
                    rows.append(batch)
                    previous = cumulative.copy()
            finally:
                openmc.lib.simulation_finalize()
    finally:
        os.chdir(old)
    if not rows:
        return np.zeros((0, 0), dtype=float)
    return np.vstack(rows)


def extract_run(
    case_dir: Path,
    *,
    selections: dict[int, list[int] | tuple[int, ...]],
    threads: int = 1,
    record_keff: bool = False,
    n_active_hint: int | None = None,
) -> dict[str, Any]:
    """One in-memory transport; copy declared columns for each tally id.

    OpenMC still scores the full tally. This path copies only declared columns
    into a preallocated (n_active, n_kept) array when n_active_hint is given.
    """

    import openmc.lib

    if not selections:
        raise ValueError("selections must be non-empty")
    wanted = {int(tid): [int(i) for i in cols] for tid, cols in selections.items()}
    for tid, cols in wanted.items():
        if not cols:
            raise ValueError(f"tally {tid} has empty column list")
    os.environ["OMP_NUM_THREADS"] = str(threads)
    old = Path.cwd()
    hint = int(n_active_hint) if n_active_hint else 0
    allocated = {
        tid: np.zeros((hint, len(cols)), dtype=float) if hint > 0 else None for tid, cols in wanted.items()
    }
    rows: dict[int, list[np.ndarray]] = {tid: [] for tid in wanted}
    previous: dict[int, np.ndarray | None] = {tid: None for tid in wanted}
    keff_inactive: list[float] = []
    keff_active: list[float] = []
    entropy: list[float] = []
    mean_width: list[int] = []
    rss_after = None
    row_i = 0
    extract_copy_s = 0.0
    loop_s = 0.0
    try:
        os.chdir(case_dir)
        with openmc.lib.run_in_memory(args=["-s", str(threads)], output=False):
            openmc.lib.simulation_init()
            try:
                t_loop = time.perf_counter()
                for _ in openmc.lib.iter_batches():
                    k_val = None
                    if record_keff:
                        try:
                            k = openmc.lib.keff()
                            k_val = float(k[0]) if hasattr(k, "__getitem__") else float(k)
                        except Exception:
                            k_val = None
                        try:
                            ent = openmc.lib.entropy()
                            entropy.append(float(ent) if np.ndim(ent) == 0 else float(np.ravel(ent)[-1]))
                        except Exception:
                            pass
                    active = False
                    for tid, cols in wanted.items():
                        tally = openmc.lib.tallies[tid]
                        realizations = int(tally.num_realizations)
                        if realizations <= 0:
                            continue
                        active = True
                        t_copy = time.perf_counter()
                        cumulative = np.asarray(tally.mean, dtype=float).reshape(-1) * realizations
                        mean_width.append(int(cumulative.size))
                        selected = np.asarray(cumulative[cols], dtype=float)
                        batch = selected.copy() if previous[tid] is None else selected - previous[tid]
                        if allocated[tid] is not None and row_i < allocated[tid].shape[0]:
                            allocated[tid][row_i] = batch
                        else:
                            rows[tid].append(batch)
                        previous[tid] = selected.copy()
                        extract_copy_s += time.perf_counter() - t_copy
                    if active:
                        row_i += 1
                        if k_val is not None:
                            keff_active.append(k_val)
                    elif k_val is not None:
                        keff_inactive.append(k_val)
                loop_s = time.perf_counter() - t_loop
            finally:
                openmc.lib.simulation_finalize()
        rss_after = peak_rss_bytes()
    finally:
        os.chdir(old)
    ledgers: dict[int, np.ndarray] = {}
    n_active = int(row_i)
    for tid, cols in wanted.items():
        if allocated[tid] is not None and n_active <= allocated[tid].shape[0] and not rows[tid]:
            ledgers[tid] = allocated[tid][:n_active]
        elif rows[tid]:
            ledgers[tid] = np.vstack(rows[tid])
        elif allocated[tid] is not None:
            ledgers[tid] = allocated[tid][:n_active]
        else:
            ledgers[tid] = np.zeros((0, len(cols)), dtype=float)
    n_active = next((int(arr.shape[0]) for arr in ledgers.values()), 0)
    return {
        "ledgers": ledgers,
        "n_active": n_active,
        "keff_active": keff_active if record_keff else None,
        "keff_inactive": keff_inactive if record_keff else None,
        "entropy": entropy or None,
        "openmc_mean_width": max(mean_width) if mean_width else None,
        "peak_rss_bytes": rss_after,
        "extract_copy_s": extract_copy_s,
        "openmc_loop_s": loop_s,
        "copied_columns": {str(tid): cols for tid, cols in wanted.items()},
    }


def extract_selected_columns(
    case_dir: Path,
    *,
    tally_id: int,
    columns: tuple[int, int] | list[int],
    threads: int = 1,
) -> np.ndarray:
    """Copy only declared ledger columns each batch. Same ΔSUM contract as full extract."""

    result = extract_run(case_dir, selections={int(tally_id): list(columns)}, threads=threads)
    return result["ledgers"][int(tally_id)]


def shrink_settings(model: Any, *, particles: int = 200, batches: int = 3, inactive: int = 1) -> None:
    model.settings.particles = particles
    model.settings.batches = batches
    model.settings.inactive = inactive
    if hasattr(model.settings, "generations_per_batch"):
        model.settings.generations_per_batch = 1
