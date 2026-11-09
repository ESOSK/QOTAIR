"""Read and analyze a bundle of independent tally histories."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .bundle import infer_bundle
from .qualification import qualify, query_key


def analyze_runs(data_path, manifest_path):
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest = json.loads(manifest_bytes)
    names = manifest["score_names"]
    query = manifest["query"]
    with np.load(data_path, allow_pickle=False) as archive:
        histories = archive["histories"]
    if histories.ndim != 3 or histories.shape[2] != len(names) or len(set(names)) != len(names):
        raise ValueError("The histories do not match the manifest's operand columns.")

    weights = np.zeros((2, len(names)))
    for row, form in enumerate((query["numerator"], query.get("denominator") or {})):
        for name, value in form.items():
            if name not in names:
                raise ValueError(f"Unknown operand: {name}")
            if not np.isfinite(float(value)):
                raise ValueError(f"Nonfinite weight for operand: {name}")
            weights[row, names.index(name)] = float(value)

    projected = np.einsum("rbp,kp->rbk", histories, weights)
    mathematical = infer_bundle(projected, manifest["runs"], linear=query.get("denominator") is None)
    directory = Path(__file__).parent
    registry_bytes = (directory / "profiles.json").read_bytes()
    registry = json.loads(registry_bytes)
    method_hash = hashlib.sha256((directory / "regions.py").read_bytes()).hexdigest()
    profile_id = manifest.get("validation_profile_id")
    profile = registry["profiles"].get(profile_id)
    if registry["candidate_sha256"] != method_hash or (profile and profile["candidate_sha256"] != method_hash):
        raise ValueError("The profile registry does not match this inference method.")

    result = qualify(
        mathematical,
        profile_id,
        manifest["runs"],
        registry,
        model_key=manifest.get("model_key"),
        query_signature=query_key(query, names),
    )
    result.setdefault("engineering_interval", None)
    with Path(data_path).open("rb") as data:
        digest = hashlib.sha256()
        for block in iter(lambda: data.read(1024 * 1024), b""):
            digest.update(block)
    result.update(
        query_name=query["name"],
        projection=weights.tolist(),
        score_names=names,
        query_sha256=hashlib.sha256(json.dumps(query, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        input_sha256=digest.hexdigest(),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        qualification_registry_sha256=hashlib.sha256(registry_bytes).hexdigest(),
    )
    return result
