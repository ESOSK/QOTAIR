"""Check run metadata before constructing a confidence set."""

from .regions import infer_histories


def infer_bundle(histories, manifest, **kwargs):
    fields = [
        "model_sha256",
        "identity_sha256",
        "normalization",
        "particles",
        "batches",
        "generations_per_batch",
        "source_protocol_sha256",
        "reset_protocol_sha256",
    ]

    def no(reason):
        return {
            "valid": False,
            "kind": "invalid",
            "intervals": [],
            "status": "REFUSE",
            "reason": reason,
            "engineering_interval": None,
        }

    if not isinstance(manifest, list) or len(manifest) != len(histories):
        return no("E_RUN_MANIFEST_LENGTH")
    if not manifest:
        return no("E_EMPTY_RUN_BUNDLE")
    for row in manifest:
        if any(k not in row for k in fields + ["run_id", "seed", "independent_stream", "source_qualified"]):
            return no("E_RUN_MANIFEST_FIELDS")
        if not row["independent_stream"] or not row["source_qualified"]:
            return no("E_SOURCE_OR_INDEPENDENCE_UNQUALIFIED")
        if row["batches"] != histories.shape[1]:
            return no("E_BATCH_COUNT_MISMATCH")
    if len({r["run_id"] for r in manifest}) != len(manifest) or len({r["seed"] for r in manifest}) != len(manifest):
        return no("E_DUPLICATE_RUN_OR_SEED")
    if any(len({str(r[k]) for r in manifest}) != 1 for k in fields):
        return no("E_HETEROGENEOUS_RUN_PROTOCOL")
    r = infer_histories(
        histories, [x["run_id"] for x in manifest], source_qualified=True, independent_runs=True, **kwargs
    )
    r["manifest_bound"] = True
    r["inference_scope"] = (
        "Statistical result conditional on the registered independent-run/source assumptions; physical reference qualification remains separate."
    )
    return r
