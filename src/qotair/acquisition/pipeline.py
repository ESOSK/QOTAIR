"""One declared query: prerun → bind → slice or extract → frozen route → classified record."""
from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

from .bind import RegionSpec, bind_region, binding_record
from .contract import QueryContract, preflight_contract
from .identity import AcquisitionError, AcquisitionIdentity
from .prerun import prerun_check, raise_if_blocking
from .provenance import InferenceRule, QueryTask, RunSource
from .route import route_selected_ratio
from .select import slice_two_scores, two_score_column_indices
from .status import classify_inference, execute_from_code


def _task_id(task: QueryTask | None, spec: RegionSpec, query_id: str) -> str:
    if task is not None:
        return task.task_id
    return f"{spec.name}:{query_id}"


def evaluate_declared_query(
    ledger: np.ndarray,
    identity: AcquisitionIdentity,
    spec: RegionSpec,
    *,
    numerator: str = "absorption",
    denominator: str = "flux",
    nuclide: str = "total",
    query_id: str = "declared_ratio",
    n_active_batches: int | None = None,
    block_count: int = 6,
    max_columns: int = 20000,
    run_source: RunSource | None = None,
    inference_rule: InferenceRule | None = None,
    task: QueryTask | None = None,
    confidence_level: float = 0.95,
    skip_prerun: bool = False,
    contract: QueryContract | None = None,
) -> dict[str, Any]:
    """Evaluate one user-declared region query on an already-extracted ledger.

    Does not retune T_den. Fieller is classified as DIAGNOSTIC.
    Integer ledger columns are a storage layout; score names and energy edges
    are the query meaning. Layout remapping uses score names, not a new query.
    """

    source = run_source or (task.run_source if task is not None else RunSource())
    rule = inference_rule or (task.inference_rule if task is not None else InferenceRule(block_count=block_count))
    n_active = int(n_active_batches if n_active_batches is not None else np.asarray(ledger).shape[0])
    started = time.perf_counter()
    try:
        bound_contract = contract or QueryContract.bind(
            identity,
            name=spec.name,
            kind=spec.kind,
            numerator=numerator,
            denominator=denominator,
            nuclide=nuclide,
            energy_bin=spec.energy_bin,
            cell_id=spec.cell_id,
            instance_index=spec.instance_index,
            mesh_ijk=spec.mesh_ijk,
            universe_id=spec.universe_id,
            material_id=spec.material_id,
        )
        pf = preflight_contract(identity, bound_contract, source)
        if not pf["ok"]:
            first = pf["issues"][0]
            raise AcquisitionError(first["code"], first["message"])
        if not skip_prerun:
            report = prerun_check(
                identity,
                n_active_batches=n_active,
                region=spec,
                block_count=rule.block_count,
                max_columns=max_columns,
            )
            raise_if_blocking(report)
        else:
            report = prerun_check(
                identity,
                n_active_batches=n_active,
                region=spec,
                block_count=rule.block_count,
                max_columns=max_columns,
            )
            spec.validate()
        bound = bind_region(identity, spec)
        two = slice_two_scores(ledger, identity, bound["selection"], numerator, denominator, nuclide)
        columns = two_score_column_indices(identity, bound["selection"], numerator, denominator, nuclide)
        inference = route_selected_ratio(
            two,
            numerator=numerator,
            denominator=denominator,
            query_id=query_id,
            confidence_level=confidence_level,
        )
        classified = classify_inference(inference)
        from ..services.scientific import paper_engineering_envelope

        envelope = paper_engineering_envelope(
            two,
            [numerator, denominator],
            numerator=numerator,
            denominator=denominator,
            confidence_level=confidence_level,
        )
        if envelope.get("status") == "REFUSE":
            envelope_release = "REFUSE"
        elif classified["product_method"] == "FIELLER":
            envelope_release = "DIAGNOSTIC"
        elif (envelope.get("product_magnitude_guard") or {}).get("blocked"):
            envelope_release = "REFUSE"
        elif envelope.get("engineering_finite_release") and envelope.get("status") == "PASS":
            envelope_release = "RELEASE"
        else:
            envelope_release = classified["release_status"]
        bind_doc = binding_record(
            identity,
            spec,
            numerator=numerator,
            denominator=denominator,
            nuclide=nuclide,
            run_source=source.as_dict(),
            inference_rule=rule.as_dict(),
        )
        bind_doc["data_map"]["columns"] = list(columns)
        return {
            "task_id": _task_id(task, spec, query_id),
            "execute_status": classified["execute_status"],
            "release_status": classified["release_status"],
            "statistical_object": classified["statistical_object"],
            "product_method": classified["product_method"],
            "product_status": classified["product_status"],
            "reason_code": classified["reason_code"],
            "binding": bind_doc,
            "prerun": report,
            "inference": inference,
            "paper_envelope": envelope,
            "paper_envelope_release": envelope_release,
            "two_column": two.tolist(),
            "selected_shape": list(two.shape),
            "column_indices": list(columns),
            "elapsed_s": time.perf_counter() - started,
            "thresholds": rule.thresholds,
            "table4": rule.table4,
        }
    except AcquisitionError as exc:
        execute = execute_from_code(exc.code)
        return {
            "task_id": _task_id(task, spec, query_id),
            "execute_status": execute,
            "release_status": "REFUSE",
            "statistical_object": {"kind": None, "method": None, "interval_kind": None, "intervals": []},
            "product_method": "REFUSE",
            "product_status": "REFUSE",
            "reason_code": exc.code,
            "message": str(exc),
            "binding": spec.as_dict(),
            "elapsed_s": time.perf_counter() - started,
            "thresholds": rule.thresholds,
            "table4": rule.table4,
        }


def replay_from_two_column(
    two: np.ndarray,
    *,
    numerator: str = "absorption",
    denominator: str = "flux",
    query_id: str = "replay",
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Rebuild frozen inference from a stored two-column ledger (no OpenMC)."""

    inference = route_selected_ratio(
        np.asarray(two, dtype=float),
        numerator=numerator,
        denominator=denominator,
        query_id=query_id,
        confidence_level=confidence_level,
    )
    classified = classify_inference(inference)
    from ..services.scientific import paper_engineering_envelope

    envelope = paper_engineering_envelope(
        np.asarray(two, dtype=float),
        [numerator, denominator],
        numerator=numerator,
        denominator=denominator,
        confidence_level=confidence_level,
    )
    if envelope.get("status") == "REFUSE":
        envelope_release = "REFUSE"
    elif classified["product_method"] == "FIELLER":
        envelope_release = "DIAGNOSTIC"
    elif (envelope.get("product_magnitude_guard") or {}).get("blocked"):
        envelope_release = "REFUSE"
    elif envelope.get("engineering_finite_release") and envelope.get("status") == "PASS":
        envelope_release = "RELEASE"
    else:
        envelope_release = classified["release_status"]
    return {
        "inference": inference,
        "paper_envelope": envelope,
        "paper_envelope_release": envelope_release,
        **classified,
    }


def summarize_records(records: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = {
        "n": len(records),
        "execute_success": 0,
        "execute_config_error": 0,
        "execute_resource_limit": 0,
        "execute_unsupported": 0,
        "release_RELEASE": 0,
        "release_DIAGNOSTIC": 0,
        "release_REFUSE": 0,
        "fieller_counted_as_release": 0,
    }
    for item in records:
        execute = str(item.get("execute_status") or "")
        release = str(item.get("release_status") or "")
        key = f"execute_{execute}"
        if key in counts:
            counts[key] += 1
        rkey = f"release_{release}"
        if rkey in counts:
            counts[rkey] += 1
        if item.get("product_method") == "FIELLER" and release == "RELEASE":
            counts["fieller_counted_as_release"] += 1
    return counts
