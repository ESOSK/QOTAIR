"""Legal acquisition plans for one query. Meaning is invariant; only extract/storage change."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence


LEGAL_PLAN_IDS = ("full_then_slice", "selected_copy")

# Recorded after the share-vs-split screen. Not an optimizer.
RECORDED_SIMPLE_RULES = (
    "selected_copy if n_kept < n_identity else full_then_slice",
    "one_transport_per_identity_for_the_declared_query_set",
)


@dataclass(frozen=True)
class AcquisitionPlan:
    plan_id: str
    description: str
    copy_all_identity_columns: bool
    tally_definition_changed: bool
    meaning_unchanged: bool
    retains_paired_batches: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def legal_plans_same_tally() -> list[AcquisitionPlan]:
    """Plans that keep the OpenMC tally, estimator, object, and pairing unchanged."""

    return [
        AcquisitionPlan(
            plan_id="full_then_slice",
            description="Copy every identity column each batch, then slice the declared pair",
            copy_all_identity_columns=True,
            tally_definition_changed=False,
            meaning_unchanged=True,
            retains_paired_batches=True,
        ),
        AcquisitionPlan(
            plan_id="selected_copy",
            description="Copy only the declared paired columns each batch",
            copy_all_identity_columns=False,
            tally_definition_changed=False,
            meaning_unchanged=True,
            retains_paired_batches=True,
        ),
    ]


def select_plan(
    measured: Sequence[dict[str, Any]],
    *,
    objective: str = "min_storage",
    rss_limit_bytes: int | None = None,
) -> dict[str, Any]:
    """Choose among measured legal plans. Does not invent a scalar 'total cost'."""

    feasible = [dict(item) for item in measured if item.get("legal")]
    rejected = []
    if rss_limit_bytes is not None:
        kept = []
        for item in feasible:
            rss = item.get("process_peak_rss_bytes")
            if rss is None:
                rejected.append(
                    {
                        "plan_id": item.get("plan_id"),
                        "reason": "rss_unknown_cannot_claim_limit",
                    }
                )
                continue
            if int(rss) <= int(rss_limit_bytes):
                kept.append(item)
            else:
                rejected.append({"plan_id": item.get("plan_id"), "reason": "rss_limit"})
        feasible = kept
    if not feasible:
        return {"selected": None, "objective": objective, "feasible": [], "rejected": rejected, "reason": "no feasible plan"}
    if objective == "min_storage":
        chosen = min(feasible, key=lambda item: int(item.get("file_bytes") or 1 << 62))
    elif objective == "min_end_to_end_s":
        chosen = min(feasible, key=lambda item: float(item.get("end_to_end_s") or 1e300))
    else:
        raise ValueError(objective)
    return {
        "selected": chosen["plan_id"],
        "objective": objective,
        "feasible": [item["plan_id"] for item in feasible],
        "rejected": rejected,
        "selected_record": chosen,
        "simple_rule": RECORDED_SIMPLE_RULES[0],
        "matches_simple_rule": chosen["plan_id"] == "selected_copy",
        "recorded_simple_rules": list(RECORDED_SIMPLE_RULES),
    }
