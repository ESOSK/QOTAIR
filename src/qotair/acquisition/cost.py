"""Layered acquisition cost. Do not subtract unmeasured layers."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class LayeredCost:
    path: str
    end_to_end_s: float
    tracking_and_scoring_s: float | None
    tracking_s: float | None
    tally_scoring_s: float | None
    extract_s: float | None
    write_s: float | None
    replay_s: float | None
    copied_array_bytes: int
    file_bytes: int
    process_peak_rss_bytes: int | None
    openmc_mean_width: int | None
    n_copied_columns: int
    n_identity_columns: int
    tally_definition_changed: bool
    note: str
    extract_copy_s: float | None = None
    openmc_loop_s: float | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["copied_bytes"] = payload.get("copied_array_bytes")
        payload["field_meanings"] = {
            "copied_array_bytes": "nbytes of the array retained in Python after extract",
            "file_bytes": "bytes written to the ledger file",
            "extract_copy_s": "time spent copying declared columns from tally.mean",
            "openmc_loop_s": "iter_batches wall time including transport, scoring, and copies",
            "tracking_and_scoring_s": "openmc_loop_s minus extract_copy_s; tracking vs scoring not split",
            "end_to_end_s": "extract plus write plus replay when those were included",
        }
        return payload


def cost_note_inseparable() -> str:
    return (
        "OpenMC in-memory run does not expose tracking vs tally scoring separately; "
        "those layers are reported as tracking_and_scoring_s. "
        "Do not subtract end-to-end minus extract to invent a scoring gain."
    )
