"""Small data contracts shared by the public API and its front ends."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ProjectValidationError(ValueError):
    """A project or query does not satisfy the public configuration contract."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class QuerySpec:
    """A user query before it is compiled by the frozen scientific core."""

    payload: dict[str, Any]

    @property
    def query_id(self) -> str:
        return str(self.payload.get("query_id", ""))

    @property
    def name(self) -> str:
        return str(self.payload.get("name", self.query_id))

    @property
    def expression_class(self) -> str:
        return str(self.payload.get("expression_class", ""))

    @property
    def expression(self) -> dict[str, Any]:
        value = self.payload.get("expression")
        return dict(value) if isinstance(value, Mapping) else {}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "QuerySpec":
        if not isinstance(value, Mapping):
            raise ProjectValidationError("E_QUERY_INPUT", "Each query must be a mapping")
        payload = dict(value)
        if "schema_version" not in payload:
            payload["schema_version"] = "p1-query-v1"
        required = ("query_id", "name", "expression_class", "output_unit", "operand_units", "domain_constraints", "expression")
        missing = [field for field in required if field not in payload]
        if missing:
            raise ProjectValidationError("E_QUERY_SCHEMA", f"Query is missing fields: {missing}")
        return cls(payload)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.payload)


@dataclass(frozen=True)
class ProjectConfig:
    """A project.yaml document with a resolved source path."""

    path: Path
    payload: dict[str, Any]

    @property
    def project_id(self) -> str:
        return str(self.payload.get("project_id") or self.path.stem)

    @property
    def mode(self) -> str:
        return str(self.payload.get("mode", "ledger_replay"))

    @property
    def model(self) -> dict[str, Any]:
        value = self.payload.get("model", {})
        return dict(value) if isinstance(value, Mapping) else {}

    @property
    def ledger(self) -> dict[str, Any]:
        value = self.payload.get("ledger", {})
        return dict(value) if isinstance(value, Mapping) else {}

    @property
    def run(self) -> dict[str, Any]:
        value = self.payload.get("run", {})
        return dict(value) if isinstance(value, Mapping) else {}

    @property
    def baselines(self) -> tuple[str, ...]:
        value = self.payload.get("baselines", ("collect_all", "native_merge", "manual_joint", "compiled_plan"))
        if isinstance(value, (list, tuple)):
            return tuple(str(item) for item in value)
        return ("compiled_plan",)

    def queries(self) -> tuple[QuerySpec, ...]:
        value = self.payload.get("queries")
        if value is None and self.payload.get("query") is not None:
            value = [self.payload["query"]]
        if isinstance(value, Mapping):
            value = [value]
        if not isinstance(value, list) or not value:
            raise ProjectValidationError("E_QUERY_INPUT", "Project must contain at least one query")
        resolved = []
        for item in value:
            if isinstance(item, (str, Path)):
                from ..storage.projects import read_mapping

                item = read_mapping(self.resolve_path(item))
            resolved.append(QuerySpec.from_mapping(item))
        return tuple(resolved)

    def resolve_path(self, value: str | Path) -> Path:
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate
        # Relative project references are intentionally anchored to the project
        # file. Falling back to the caller's working directory makes a copied
        # project appear valid only when invoked from one particular checkout.
        return (self.path.parent / candidate).resolve()

    def ledger_path(self) -> Path | None:
        value = self.ledger.get("path")
        if not value:
            return None
        return self.resolve_path(str(value))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.payload)


@dataclass(frozen=True)
class RunResult:
    """Stable summary returned by a run and rendered by all interfaces."""

    run_id: str
    status: str
    path: Path
    message: str
    result_path: Path | None = None
    plan_path: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "path": str(self.path),
            "message": self.message,
            "result_path": str(self.result_path) if self.result_path else None,
            "plan_path": str(self.plan_path) if self.plan_path else None,
        }
