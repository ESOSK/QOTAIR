"""Engineering qualification of a frozen statistical result. No per-sample truth or reference is used."""

import hashlib
import json


def query_key(query, score_names):
    return hashlib.sha256(
        json.dumps(
            {"names": score_names, "numerator": query["numerator"], "denominator": query.get("denominator")},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def qualify(statistical, profile_id, run_manifest, registry, *, model_key, query_signature):
    """Keep the complete mathematical object; unknown acquisition profiles never become engineering releases."""
    result = dict(statistical)
    result["statistical_candidate_status"] = statistical["status"]
    result["engineering_qualification"] = "UNQUALIFIED"
    profile = registry.get("profiles", {}).get(profile_id)
    reason = None
    if profile is None:
        reason = "E_ACQUISITION_PROFILE_UNQUALIFIED"
    elif profile["model_key"] != model_key or profile["query_key"] != query_signature:
        reason = "E_MODEL_OR_QUERY_PROFILE_MISMATCH"
    elif len(run_manifest) != profile["R"]:
        reason = "E_PROFILE_REPLICATION_BUDGET_MISMATCH"
    else:
        for row in run_manifest:
            for field in ["batches", "particles", "generations_per_batch", "normalization"]:
                if field in profile and row.get(field) != profile[field]:
                    reason = "E_PROFILE_ACQUISITION_MISMATCH"
            if row.get("source_protocol_sha256") != profile["source_protocol_sha256"]:
                reason = "E_SOURCE_PROFILE_MISMATCH"
    if reason:
        if result["status"] != "REFUSE":
            result["status"] = "DIAGNOSTIC"
        result["engineering_interval"] = None
        result["profile_reason"] = reason
    else:
        result["engineering_qualification"] = "QUALIFIED_DECLARED_POPULATION_AND_BUDGET"
        result["validation_profile_id"] = profile_id
        result["validation_evidence"] = profile["evidence"]
    return result
