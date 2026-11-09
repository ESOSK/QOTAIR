from __future__ import annotations

import json

from qotair.reproducibility import (
    BINARY_HASH_POLICY,
    CANONICAL_TEXT_HASH_POLICY,
    build_archive_manifest,
    sha256_file,
    verify_archive_manifest,
    write_json,
    write_sidecar,
)


def test_hash_policy_keeps_exact_bytes_separate_from_text_canonicalisation(tmp_path):
    lf = tmp_path / "lf.csv"
    crlf = tmp_path / "crlf.csv"
    lf.write_bytes(b"a,b\n1,2\n")
    crlf.write_bytes(b"a,b\r\n1,2\r\n")

    assert sha256_file(lf, policy=BINARY_HASH_POLICY) != sha256_file(crlf, policy=BINARY_HASH_POLICY)
    assert sha256_file(lf, policy=CANONICAL_TEXT_HASH_POLICY) == sha256_file(
        crlf,
        policy=CANONICAL_TEXT_HASH_POLICY,
    )


def test_archive_manifest_detects_tampering_and_ignores_declared_outputs(tmp_path):
    archive = tmp_path / "snapshot"
    (archive / "registry").mkdir(parents=True)
    (archive / "results").mkdir()
    (archive / "registry" / "input.json").write_bytes(b'{"value": 1}\n')
    (archive / "results" / "derived-v7-reproduce-report.json").write_bytes(b"generated\n")
    manifest_path = archive / "manifest.json"
    manifest = build_archive_manifest(archive, output=manifest_path)
    assert manifest["hash_policy"]["name"] == BINARY_HASH_POLICY
    assert verify_archive_manifest(archive)["status"] == "PASS"

    (archive / "registry" / "input.json").write_bytes(b'{"value": 2}\n')
    report = verify_archive_manifest(archive)
    assert report["status"] == "FAIL"
    assert "registry/input.json" in report["mismatched"]


def test_manifest_sidecar_is_detached_and_canonical(tmp_path):
    manifest = tmp_path / "manifest.json"
    digest = write_json(manifest, {"schema_version": "test", "value": 1})
    sidecar = write_sidecar(manifest, digest)
    assert sidecar.name == "manifest.sha256"
    assert sidecar.read_text(encoding="utf-8").endswith("manifest.json\n")
    assert json.loads(manifest.read_text(encoding="utf-8"))["value"] == 1
