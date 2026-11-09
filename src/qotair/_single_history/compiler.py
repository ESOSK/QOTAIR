from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .expression import minimal_operands, validate_expression
from .identity import (
    AcquisitionCollectionPlan,
    QueryAcquisitionMapping,
)


QUERY_SCHEMA_VERSION = "p1-query-v1"
EXPRESSION_CLASSES = (
    "ratio",
    "linear_combination",
    "normalized_quantity",
    "mgxs_equivalent",
)


class QueryCompilationError(ValueError):
    """Query compilation failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class CollectionPlan:
    query_id: str
    expression_class: str
    schema_version: str
    canonical_query_sha256: str
    operands: tuple[str, ...]
    excluded_available_operands: tuple[str, ...]
    output_unit: str
    operand_units: tuple[tuple[str, str], ...]
    contains_division: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "expression_class": self.expression_class,
            "schema_version": self.schema_version,
            "canonical_query_sha256": self.canonical_query_sha256,
            "operands": list(self.operands),
            "excluded_available_operands": list(self.excluded_available_operands),
            "output_unit": self.output_unit,
            "operand_units": dict(self.operand_units),
            "contains_division": self.contains_division,
        }


@dataclass(frozen=True)
class AcquisitionAwareQueryPlan:
    legacy_plan: CollectionPlan
    acquisition_identities: tuple[dict[str, Any], ...]
    acquisition_mapping: QueryAcquisitionMapping
    acquisition_plan_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.legacy_plan.as_dict(),
            "acquisition_plan_sha256": self.acquisition_plan_sha256,
            "acquisition_identities": list(self.acquisition_identities),
            "query_acquisition_mapping": self.acquisition_mapping.as_dict(),
        }


def _require_text(mapping: Mapping[str, Any], field: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value.strip():
        raise QueryCompilationError("E_QUERY_SCHEMA", f"{field} must be a non-empty string")
    return value.strip()


def _normalize_node(node: Mapping[str, Any]) -> dict[str, Any]:
    op = node["op"]
    if op == "score":
        return {"op": "score", "name": node["name"]}
    if op == "constant":
        return {"op": "constant", "value": float(node["value"])}
    if op == "linear_combination":
        return {
            "op": "linear_combination",
            "terms": [
                {
                    "coefficient": float(term["coefficient"]),
                    "operand": _normalize_node(term["operand"]),
                }
                for term in node["terms"]
            ],
        }
    return {"op": op, "args": [_normalize_node(child) for child in node["args"]]}


def _contains_division(node: Mapping[str, Any]) -> bool:
    if node["op"] == "divide":
        return True
    if node["op"] == "linear_combination":
        return any(_contains_division(term["operand"]) for term in node["terms"])
    return any(_contains_division(child) for child in node.get("args", ()))


def validate_query(query: Mapping[str, Any]) -> None:
    required_fields = {
        "schema_version",
        "query_id",
        "name",
        "expression_class",
        "output_unit",
        "operand_units",
        "domain_constraints",
        "expression",
    }
    if set(query) != required_fields:
        missing = sorted(required_fields - set(query))
        extra = sorted(set(query) - required_fields)
        raise QueryCompilationError(
            "E_QUERY_SCHEMA", f"Query fields differ from v1 contract; missing={missing}, extra={extra}"
        )
    if query.get("schema_version") != QUERY_SCHEMA_VERSION:
        raise QueryCompilationError("E_QUERY_SCHEMA_VERSION", "Unsupported query schema version")
    _require_text(query, "query_id")
    _require_text(query, "name")
    expression_class = _require_text(query, "expression_class")
    if expression_class not in EXPRESSION_CLASSES:
        raise QueryCompilationError("E_EXPRESSION_CLASS", f"Unsupported class: {expression_class}")
    _require_text(query, "output_unit")
    constraints = query.get("domain_constraints")
    if not isinstance(constraints, list) or not constraints or not all(
        isinstance(item, str) and item.strip() for item in constraints
    ):
        raise QueryCompilationError("E_QUERY_SCHEMA", "domain_constraints must be a non-empty string list")
    units = query.get("operand_units")
    if not isinstance(units, dict) or not units:
        raise QueryCompilationError("E_QUERY_SCHEMA", "operand_units must be a non-empty object")
    if not all(
        isinstance(name, str) and name.strip() and isinstance(unit, str) and unit.strip()
        for name, unit in units.items()
    ):
        raise QueryCompilationError("E_QUERY_SCHEMA", "operand_units keys and values must be non-empty strings")
    expression = query.get("expression")
    try:
        validate_expression(expression)
    except Exception as exc:
        code = getattr(exc, "code", "E_EXPRESSION_SCHEMA")
        raise QueryCompilationError(code, str(exc)) from exc
    operands = minimal_operands(expression)
    unknown = sorted(set(operands) - set(units))
    unused_units = sorted(set(units) - set(operands))
    if unknown:
        raise QueryCompilationError("E_UNKNOWN_SCORE", f"Operands lack a unit/known-score entry: {unknown}")
    if unused_units:
        raise QueryCompilationError("E_UNUSED_UNIT_DECLARATION", f"Unused operand unit entries: {unused_units}")


def canonical_query_payload(query: Mapping[str, Any]) -> dict[str, Any]:
    validate_query(query)
    return {
        "schema_version": query["schema_version"],
        "query_id": query["query_id"],
        "name": query["name"],
        "expression_class": query["expression_class"],
        "output_unit": query["output_unit"],
        "operand_units": {name: query["operand_units"][name] for name in sorted(query["operand_units"])},
        "domain_constraints": list(query["domain_constraints"]),
        "expression": _normalize_node(query["expression"]),
    }


def canonical_query_sha256(query: Mapping[str, Any]) -> str:
    payload = canonical_query_payload(query)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def check_plan_inputs(plan: CollectionPlan, available_operands: Iterable[str]) -> tuple[str, ...]:
    available = tuple(available_operands)
    if len(set(available)) != len(available) or not all(
        isinstance(name, str) and name for name in available
    ):
        raise QueryCompilationError(
            "E_AVAILABLE_OPERANDS", "Available operands must be unique non-empty strings"
        )
    missing = [name for name in plan.operands if name not in available]
    if missing:
        raise QueryCompilationError("E_PLAN_MISSING_OPERAND", f"Required operands missing: {missing}")
    return tuple(name for name in available if name not in plan.operands)


def compile_query(
    query: Mapping[str, Any], *, available_operands: Iterable[str] | None = None
) -> CollectionPlan:
    validate_query(query)
    operands = tuple(minimal_operands(query["expression"]))
    initial = CollectionPlan(
        query_id=query["query_id"],
        expression_class=query["expression_class"],
        schema_version=query["schema_version"],
        canonical_query_sha256=canonical_query_sha256(query),
        operands=operands,
        excluded_available_operands=(),
        output_unit=query["output_unit"],
        operand_units=tuple((name, query["operand_units"][name]) for name in operands),
        contains_division=_contains_division(query["expression"]),
    )
    available = operands if available_operands is None else tuple(available_operands)
    excluded = check_plan_inputs(initial, available)
    return CollectionPlan(
        query_id=initial.query_id,
        expression_class=initial.expression_class,
        schema_version=initial.schema_version,
        canonical_query_sha256=initial.canonical_query_sha256,
        operands=initial.operands,
        excluded_available_operands=excluded,
        output_unit=initial.output_unit,
        operand_units=initial.operand_units,
        contains_division=initial.contains_division,
    )


def compile_query_with_acquisition(
    query: Mapping[str, Any],
    acquisition_plan: AcquisitionCollectionPlan,
    *,
    available_operands: Iterable[str] | None = None,
) -> AcquisitionAwareQueryPlan:
    """Bind the legacy expression plan to audited projection and acquisition layers."""
    legacy = compile_query(query, available_operands=available_operands)
    matches = [item for item in acquisition_plan.query_mappings if item.query_id == legacy.query_id]
    if len(matches) != 1:
        raise QueryCompilationError("E_ACQUISITION_MAPPING", f"Expected one acquisition mapping for {legacy.query_id}")
    mapping = matches[0]
    projected = {projection.ledger_column for projection in mapping.projections}
    missing = sorted(set(legacy.operands) - projected)
    if missing:
        raise QueryCompilationError("E_ACQUISITION_OPERAND", f"Acquisition mapping does not produce operands: {missing}")
    used_hashes = {projection.acquisition_sha256 for projection in mapping.projections}
    identities = tuple(
        {"sha256": identity.sha256, **identity.as_dict()}
        for identity in acquisition_plan.identities
        if identity.sha256 in used_hashes
    )
    return AcquisitionAwareQueryPlan(legacy, identities, mapping, acquisition_plan.canonical_plan_sha256)
