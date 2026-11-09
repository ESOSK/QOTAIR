"""Persist two-column ΔSUM ledgers so intervals can be rebuilt without OpenMC."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def write_task_bundle(root: Path, task_id: str, two: np.ndarray, record: Mapping[str, Any]) -> Path:
    target = Path(root) / task_id
    target.mkdir(parents=True, exist_ok=True)
    values = np.asarray(two, dtype=float)
    header = "numerator,denominator"
    np.savetxt(target / "batch_operand_ledger.csv", values, delimiter=",", header=header, comments="")
    np.save(target / "batch_operand_ledger.npy", values)
    payload = {k: v for k, v in dict(record).items() if k != "two_column"}
    payload["ledger_file"] = "batch_operand_ledger.csv"
    payload["n_active"] = int(values.shape[0])
    payload["n_columns"] = int(values.shape[1]) if values.ndim == 2 else 0
    (target / "record.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return target


def load_two_column(bundle: Path) -> np.ndarray:
    csv = Path(bundle) / "batch_operand_ledger.csv"
    npy = Path(bundle) / "batch_operand_ledger.npy"
    if npy.is_file():
        return np.load(npy)
    return np.loadtxt(csv, delimiter=",")


def load_record(bundle: Path) -> dict[str, Any]:
    return json.loads((Path(bundle) / "record.json").read_text(encoding="utf-8"))
