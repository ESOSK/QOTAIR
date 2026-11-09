"""Read-only environment and support-matrix checks."""
from __future__ import annotations

import importlib.util
import os
import platform
import sys
from typing import Any

from ..paths import ARCHIVE_ROOT, required_archive_paths


def _module_version(name: str) -> str | None:
    if importlib.util.find_spec(name) is None:
        return None
    try:
        module = __import__(name)
    except Exception:
        return "present_but_unimportable"
    return str(getattr(module, "__version__", "present"))


def environment_report() -> dict[str, Any]:
    required = {str(path.relative_to(ARCHIVE_ROOT)): path.exists() for path in required_archive_paths(ARCHIVE_ROOT)}
    cross_sections = os.environ.get("OPENMC_CROSS_SECTIONS") or os.environ.get("CROSS_SECTIONS")
    return {
        "package": "qotair",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "archive_root": str(ARCHIVE_ROOT),
        "archive_root_source": "QOTAIR_ARCHIVE_ROOT" if os.environ.get("QOTAIR_ARCHIVE_ROOT") else "source_checkout_default",
        "archive_ready": all(required.values()),
        "required_paths": required,
        "dependencies": {
            "openmc": _module_version("openmc"),
            "numpy": _module_version("numpy"),
            "scipy": _module_version("scipy"),
            "yaml": _module_version("yaml"),
            "textual": _module_version("textual"),
        },
        "nuclear_data": {
            "locator": cross_sections,
            "configured": bool(cross_sections),
            "bundled": False,
        },
        "support_matrix": {
            "analysis_replay": "supported",
            "csv_ledger_replay": "supported",
            "textual_tui": "enabled" if _module_version("textual") else "fallback_available",
            "openmc_transport": "explicit_model_adapter_required",
            "mpi": "not_verified",
            "restart": "not_verified",
            "mesh_filter": "extension",
            "native_regular_mesh": "extension",
            "universe_filter": "extension",
            "distribcell_filter": "extension",
            "distribcell_instance_check": "extension",
            "material_filter": "extension",
            "energy_function_filter": "extension",
            "energy_function_tables": "registered_only",
            "unique_cell_filter": "extension",
            "selected_bin_route": "extension",
            "depletion_same_identity": "extension",
            "depletion_step_route": "extension",
            "generations_per_batch_gt_1": "not_verified",
        },
        "accepted_environment_is_read_only": True,
    }
