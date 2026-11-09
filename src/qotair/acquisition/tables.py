"""Registered EnergyFunctionFilter tables.

Product attach accepts only names in this registry. Missing names and
length mismatches fail closed. There is no default ICRP table and no
dose-verified claim.
"""
from __future__ import annotations

from dataclasses import dataclass

from .identity import AcquisitionError


@dataclass(frozen=True)
class EnergyFunctionTable:
    name: str
    energies_eV: tuple[float, ...]
    y: tuple[float, ...]
    purpose: str


def _table(
    name: str,
    energies_eV: tuple[float, ...],
    y: tuple[float, ...],
    purpose: str,
) -> EnergyFunctionTable:
    if len(energies_eV) != len(y):
        raise AcquisitionError("E_ENERGY_FUNCTION_TABLE", f"{name}: energies and y length differ")
    if len(energies_eV) < 2:
        raise AcquisitionError("E_ENERGY_FUNCTION_TABLE", f"{name}: need at least two energy points")
    if any(right <= left for left, right in zip(energies_eV, energies_eV[1:])):
        raise AcquisitionError("E_ENERGY_FUNCTION_TABLE", f"{name}: energies must increase")
    return EnergyFunctionTable(name=name, energies_eV=energies_eV, y=y, purpose=purpose)


# y = 1 is a weighted-flux probe. It is not an ICRP conversion and is not a dose.
TABLES: dict[str, EnergyFunctionTable] = {
    "unity": _table(
        "unity",
        (1e-5, 0.625, 2.0e7),
        (1.0, 1.0, 1.0),
        "weighted-flux probe with y=1; not ICRP; not a verified dose",
    ),
}


def get_table(name: str) -> EnergyFunctionTable:
    key = str(name)
    if key not in TABLES:
        raise AcquisitionError("E_ENERGY_FUNCTION_TABLE", f"unregistered EnergyFunction table {key!r}")
    table = TABLES[key]
    if len(table.energies_eV) != len(table.y):
        raise AcquisitionError("E_ENERGY_FUNCTION_TABLE", f"{key}: energies and y length differ")
    return table


def require_registered_table(table_name: str | None) -> EnergyFunctionTable:
    if not table_name:
        raise AcquisitionError(
            "E_ENERGY_FUNCTION_TABLE",
            "EnergyFunctionFilter requires a registered table name",
        )
    return get_table(table_name)


def install_table(
    name: str,
    energies_eV: tuple[float, ...] | list[float],
    y: tuple[float, ...] | list[float],
    purpose: str,
) -> EnergyFunctionTable:
    """Register one tabulated (E, y) pair. ICRP/dose names are rejected."""

    key = str(name).strip()
    lowered = key.lower()
    if not key:
        raise AcquisitionError("E_ENERGY_FUNCTION_TABLE", "table name required")
    if "icrp" in lowered or "dose" in lowered:
        raise AcquisitionError(
            "E_ENERGY_FUNCTION_TABLE",
            "ICRP/dose tables are not accepted; this path is not a verified dose",
        )
    table = _table(key, tuple(float(v) for v in energies_eV), tuple(float(v) for v in y), str(purpose))
    TABLES[key] = table
    return table
