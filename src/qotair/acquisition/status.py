"""Split execute status, paper release status, and statistical object.

Product router methods stay frozen: DELTA / JACKKNIFE / FIELLER / REFUSE.
Paper release mapping (not a retune):

- FIELLER with a finite set -> DIAGNOSTIC (never an engineering RELEASE)
- DELTA or JACKKNIFE with status PASS -> RELEASE
- REFUSE / failed selected method -> REFUSE

Execute status is independent of that mapping.
"""
from __future__ import annotations

from typing import Any, Mapping

CONFIG_CODES = {
    "E_BLOCK_PARTITION",
    "E_ROUTER_INSUFFICIENT_BATCHES",
    "E_REGION_UNSPECIFIED",
    "E_REGION_AMBIGUOUS",
    "E_BIN_SELECTION",
    "E_LEDGER_SHAPE",
    "E_IDENTITY_VALUES",
    "E_CELL_BINS",
    "E_UNIVERSE_BINS",
    "E_MATERIAL_BINS",
    "E_DISTRIBCELL_INSTANCES",
    "E_DISTRIBCELL_INSTANCE_MISMATCH",
    "E_DISTRIBCELL_CELL",
    "E_PRERUN",
    "E_NAMESPACE_CONFLICT",
    "E_REQUIRED_TALLY_MISSING",
    "E_PROJECTION_OUT_OF_RANGE",
    "E_UNKNOWN_OPERAND",
    "E_INVALID_CONFIDENCE_LEVEL",
    "E_SHAPE",
}
RESOURCE_CODES = {
    "E_LEDGER_TOO_WIDE",
    "E_RESOURCE_LIMIT",
}
UNSUPPORTED_CODES = {
    "E_FILTER_UNSUPPORTED",
    "E_NESTED_RATIO",
    "E_QUERY_SCHEMA",
    "E_NUCLIDE_FLUX",
    "E_MESH_BINS",
}


def execute_from_code(code: str | None) -> str:
    if not code:
        return "success"
    if code in CONFIG_CODES or str(code).startswith("E_REGION"):
        return "config_error"
    if code in RESOURCE_CODES:
        return "resource_limit"
    if code in UNSUPPORTED_CODES:
        return "unsupported"
    if str(code).startswith("E_"):
        return "config_error"
    return "fail"


def paper_release_status(inference: Mapping[str, Any] | None, *, execute: str = "success") -> str:
    """Map frozen product inference onto paper RELEASE / DIAGNOSTIC / REFUSE."""

    if execute != "success" or inference is None:
        return "REFUSE"
    status = str(inference.get("status") or "")
    method = str(inference.get("method") or "")
    if status == "REFUSE" or method == "REFUSE":
        return "REFUSE"
    if method == "FIELLER":
        return "DIAGNOSTIC"
    if method in {"DELTA", "JACKKNIFE"} and status == "PASS":
        return "RELEASE"
    if status == "PASS":
        return "RELEASE"
    return "REFUSE"


def statistical_object(inference: Mapping[str, Any] | None, release: str) -> dict[str, Any]:
    if inference is None:
        return {"kind": None, "interval_kind": None, "method": None}
    method = inference.get("method")
    interval_kind = inference.get("interval_kind")
    if release == "DIAGNOSTIC":
        kind = "fieller_set"
    elif release == "RELEASE":
        kind = "product_interval"
    else:
        kind = None
    return {
        "kind": kind,
        "method": method,
        "interval_kind": interval_kind,
        "intervals": inference.get("intervals") or [],
        "note": (
            "Product router returns one selected method. "
            "FIELLER is DIAGNOSTIC. DELTA/JACKKNIFE PASS is a product_interval, "
            "not the paper hull(HAC ∪ jackknife). See paper_envelope."
        ),
    }


def classify_inference(inference: Mapping[str, Any]) -> dict[str, Any]:
    status = str(inference.get("status") or "")
    method = str(inference.get("method") or "")
    code = inference.get("reason_code") or inference.get("rejection_code")
    if status == "REFUSE" or method == "REFUSE":
        execute = execute_from_code(str(code) if code else "E_REFUSE")
        if str(code) in {
            "E_DENOMINATOR_UNIDENTIFIED",
            "E_BATCH_INSTABILITY",
            "E_ZERO_DENOMINATOR",
            "E_NONFINITE",
            "E_NONFINITE_INPUT",
            "E_JACKKNIFE_UNSTABLE_REPLICATE",
            "E_SELECTED_METHOD_FAILED",
        }:
            execute = "success"
        release = "REFUSE"
    else:
        execute = "success"
        release = paper_release_status(inference, execute=execute)
    return {
        "execute_status": execute,
        "release_status": release,
        "statistical_object": statistical_object(inference, release),
        "product_status": status,
        "product_method": method,
        "reason_code": code,
    }
