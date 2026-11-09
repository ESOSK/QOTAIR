"""Simple denominator |mean|/se gate. Not Table 4 and not a retune."""
from __future__ import annotations

from typing import Any

import numpy as np


def simple_denominator_gate(
    two: np.ndarray,
    *,
    t_refuse: float = 1.0,
    t_diagnostic: float = 3.0,
) -> dict[str, Any]:
    values = np.asarray(two, dtype=float)
    den = values[:, 1]
    n = int(den.shape[0])
    mean = float(den.mean()) if n else float("nan")
    se = float(den.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    t_den = float(abs(mean) / se) if se and se > 0 else 0.0
    if t_den < t_refuse:
        release = "REFUSE"
    elif t_den < t_diagnostic:
        release = "DIAGNOSTIC"
    else:
        release = "RELEASE"
    return {
        "t_den": t_den,
        "mean_denominator": mean,
        "se_denominator": se,
        "n_active": n,
        "release_status": release,
        "note": "simple |mean|/se on the denominator only; not the frozen threshold table",
    }
