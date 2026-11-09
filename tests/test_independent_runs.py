import json

import numpy as np

from qotair.cli import main


def test_finite_set_does_not_release_unregistered_runs(tmp_path):
    data, manifest, result = [tmp_path / name for name in ("runs.npz", "manifest.json", "result.json")]
    rng = np.random.default_rng(42)
    np.savez(data, histories=rng.normal(loc=[0.2, 1.0], scale=0.05, size=(32, 12, 2)))
    common = dict(
        independent_stream=True, source_qualified=True, model_sha256="model", identity_sha256="identity",
        normalization="per source particle", particles=1000, batches=12, generations_per_batch=1,
        source_protocol_sha256="source", reset_protocol_sha256="reset",
    )
    spec = dict(
        score_names=["absorption", "flux"],
        query=dict(name="ratio", numerator={"absorption": 1}, denominator={"flux": 1}),
        runs=[dict(common, run_id=i, seed=i + 1) for i in range(32)],
    )
    args = ["infer-runs", "--data", str(data), "--manifest", str(manifest), "--out", str(result)]
    manifest.write_text(json.dumps(spec), encoding="utf-8")
    assert main(args) == 0
    report = json.loads(result.read_text(encoding="utf-8"))
    assert report["kind"] == "bounded"
    assert report["status"] == "DIAGNOSTIC"
    assert report["engineering_interval"] is None
    assert main(args) == 2

    spec["runs"][1]["seed"] = spec["runs"][0]["seed"]
    manifest.write_text(json.dumps(spec), encoding="utf-8")
    result.unlink()
    assert main(args) != 0
    report = json.loads(result.read_text(encoding="utf-8"))
    assert report["status"] == "REFUSE"
    assert report["reason"] == "E_DUPLICATE_RUN_OR_SEED"
    assert report["engineering_interval"] is None
