"""Transport-before checks: block partition, width, region uniqueness, support."""
from __future__ import annotations

from typing import Any

from .bind import RegionSpec
from .identity import AcquisitionError, AcquisitionIdentity
from .select import ledger_column_count


def jackknife_partition_ok(n_active: int, block_count: int = 6) -> bool:
    n = int(n_active)
    k = int(block_count)
    return n >= 2 * k and k > 0 and n % k == 0


def prerun_check(
    identity: AcquisitionIdentity,
    *,
    n_active_batches: int,
    region: RegionSpec | None = None,
    block_count: int = 6,
    max_columns: int = 20000,
    nested_ratio: bool = False,
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    if nested_ratio:
        issues.append({"class": "unsupported", "code": "E_NESTED_RATIO", "message": "nested ratios are out of scope"})
    if region is not None:
        try:
            region.validate()
        except AcquisitionError as exc:
            issues.append({"class": "config_error", "code": exc.code, "message": str(exc)})
    n_active = int(n_active_batches)
    k = int(block_count)
    if not jackknife_partition_ok(n_active, k):
        issues.append(
            {
                "class": "config_error",
                "code": "E_BLOCK_PARTITION",
                "message": (
                    f"active batches {n_active} is incompatible with jackknife k={k} "
                    f"(need n>=2k and n divisible by k; e.g. 60)"
                ),
            }
        )
    columns = ledger_column_count(identity)
    if columns > int(max_columns):
        issues.append(
            {
                "class": "resource_limit",
                "code": "E_LEDGER_TOO_WIDE",
                "message": f"identity has {columns} columns > max_columns {max_columns}",
            }
        )
    return {
        "ok": not issues,
        "issues": issues,
        "n_active_batches": n_active,
        "block_count": k,
        "n_columns": columns,
        "max_columns": int(max_columns),
        "would_transport": not any(item["class"] in {"config_error", "unsupported"} for item in issues),
    }


def raise_if_blocking(report: dict[str, Any]) -> None:
    blocking = [item for item in report["issues"] if item["class"] in {"config_error", "unsupported"}]
    if not blocking:
        return
    first = blocking[0]
    raise AcquisitionError(first["code"], first["message"])
