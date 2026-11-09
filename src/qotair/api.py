"""Stable public Python API for QOTAIR.

The API is the programmatic contract behind the command line interface.  It
only composes the application services and the preserved scientific adapters;
it does not copy or alter the frozen statistical formulas.

The supported execution boundary is deliberately explicit: QOTAIR can compile
queries, build plans, replay CSV BatchLedgers, route inference or refusal, and
verify artifacts.  OpenMC transport remains an explicit unsupported adapter
boundary until a separately verified model adapter is available.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeAlias, cast

from .domain.models import ProjectConfig, QuerySpec, RunResult
from .runtime import doctor as _doctor
from .runtime import replay_report as _replay_report
from .runtime import run_tests as _run_tests
from .runtime import test_report as _test_report
from .paths import ARCHIVE_ROOT
from .reproducibility import verify_archive_manifest as _verify_archive_manifest
from .reproducibility import verify_bundle_manifest as _verify_bundle_manifest
from .services.execution import history as _history
from .services.execution import replay_ledger as _replay_ledger
from .services.execution import run_project as _run_project
from .services.inspection import inspect_project as _inspect_project
from .services.inspection import inspect_query as _inspect_query
from .services.inspection import verify_run as _verify_run
from .services.planning import build_plan as _build_plan
from .services.planning import write_plan as _write_plan
from .services.scientific import compile_query as _compile_query
from .services.scientific import infer_query as _infer_query
from .services.scientific import json_safe as _json_safe
from .services.scientific import query_operands as _query_operands
from .storage.projects import ProjectLayout
from .storage.projects import create_project as _create_project
from .storage.projects import list_runs as _list_runs
from .storage.projects import load_project as _load_project
from .storage.projects import read_mapping as _read_mapping
from .storage.projects import save_query as _save_query
from .storage.projects import sha256_mapping as _sha256_mapping


API_VERSION = "qotair-api-v1"
ProjectRef: TypeAlias = str | Path | ProjectConfig
QueryRef: TypeAlias = str | Path | Mapping[str, Any]
ProgressCallback = Callable[[str], None]


class APIError(RuntimeError):
    """A stable public API error with a machine-readable reason code."""

    def __init__(self, message: str, code: str = "E_API"):
        super().__init__(message)
        self.code = code


def _error_code(exc: BaseException, fallback: str) -> str:
    code = getattr(exc, "code", None)
    if code:
        return str(code)
    if isinstance(exc, FileNotFoundError):
        return "E_FILE_NOT_FOUND"
    if isinstance(exc, FileExistsError):
        return "E_EXISTS"
    if isinstance(exc, PermissionError):
        return "E_PERMISSION"
    return fallback


def _as_api_error(exc: BaseException, fallback: str) -> APIError:
    if isinstance(exc, APIError):
        return exc
    return APIError(str(exc) or exc.__class__.__name__, _error_code(exc, fallback))


def _project(value: ProjectRef) -> ProjectConfig:
    if isinstance(value, ProjectConfig):
        return value
    try:
        return cast(ProjectConfig, _load_project(value))
    except Exception as exc:
        raise _as_api_error(exc, "E_PROJECT_INPUT") from exc


def _query_payload(value: QueryRef) -> dict[str, Any]:
    try:
        payload: Mapping[str, Any]
        if isinstance(value, Mapping):
            payload = dict(value)
        else:
            path = Path(value).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = cast(Mapping[str, Any], _read_mapping(path))
        if "query" in payload and isinstance(payload["query"], Mapping):
            payload = dict(payload["query"])
        return cast(dict[str, Any], QuerySpec.from_mapping(payload).as_dict())
    except Exception as exc:
        raise _as_api_error(exc, "E_QUERY_INPUT") from exc


def _compiled_mapping(value: Any) -> dict[str, Any]:
    raw = value.as_dict() if hasattr(value, "as_dict") and callable(value.as_dict) else value
    safe = _json_safe(raw)
    if not isinstance(safe, Mapping):
        raise APIError("The frozen compiler returned a non-mapping contract.", "E_API_CONTRACT")
    return dict(safe)


def doctor() -> dict[str, Any]:
    """Return a read-only environment, archive, and support report."""

    return dict(_doctor())


def check_environment() -> dict[str, Any]:
    """Alias for :func:`doctor` with an API-oriented name."""

    return doctor()


def verify_archive(
    archive_root: str | Path | None = None,
    *,
    manifest: str | Path | None = None,
    strict_extra: bool = False,
) -> dict[str, Any]:
    """Verify an archive's exact-byte SHA-256 manifest without running replay."""

    root = Path(archive_root).expanduser().resolve() if archive_root is not None else ARCHIVE_ROOT
    try:
        return dict(_verify_archive_manifest(root, manifest=manifest, strict_extra=strict_extra))
    except Exception as exc:
        raise _as_api_error(exc, "E_ARCHIVE_INTEGRITY") from exc


def verify_bundle(
    bundle_root: str | Path,
    *,
    manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Verify software, archive, and lock-file bindings in a reviewer bundle."""

    try:
        return dict(_verify_bundle_manifest(Path(bundle_root), manifest=manifest))
    except Exception as exc:
        raise _as_api_error(exc, "E_ARCHIVE_INTEGRITY") from exc


def compile_query(
    query: QueryRef,
    *,
    available_operands: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Compile a query document and return its JSON-safe frozen contract.

    ``query`` may be a YAML/JSON path or an in-memory mapping.  When no
    available operand list is supplied, the query's own minimal operand set is
    used, matching the inspection and planning services.
    """

    payload = _query_payload(query)
    try:
        operands = _query_operands(payload)
    except Exception as exc:
        raise _as_api_error(exc, "E_QUERY_OPERANDS") from exc
    candidates = tuple(str(item) for item in (available_operands or operands))
    try:
        compiled = _compile_query(payload, available_operands=candidates)
    except Exception as exc:
        raise _as_api_error(exc, "E_QUERY_COMPILER") from exc
    try:
        return _compiled_mapping(compiled)
    except Exception as exc:
        raise _as_api_error(exc, "E_API_CONTRACT") from exc


def query_operands(query: QueryRef) -> tuple[str, ...]:
    """Return the inclusion-minimal primitive operands for a query."""

    payload = _query_payload(query)
    try:
        return tuple(_query_operands(payload))
    except Exception as exc:
        raise _as_api_error(exc, "E_QUERY_OPERANDS") from exc


def inspect_query(
    query: QueryRef,
    *,
    available_operands: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Validate a query and expose its hash, operands, and compiled contract."""

    if not isinstance(query, Mapping):
        try:
            return dict(_inspect_query(query, available_operands=available_operands))
        except Exception as exc:
            raise _as_api_error(exc, "E_QUERY_INSPECTION") from exc
    payload = _query_payload(query)
    try:
        operands = _query_operands(payload)
    except Exception as exc:
        raise _as_api_error(exc, "E_QUERY_OPERANDS") from exc
    candidates = tuple(str(item) for item in (available_operands or operands))
    return {
        "status": "PASS",
        "reason_code": None,
        "schema_version": "qotair-query-inspection-v1",
        "path": None,
        "query_id": payload["query_id"],
        "name": payload["name"],
        "expression_class": payload["expression_class"],
        "query_sha256": _sha256_mapping(payload),
        "requested_operands": list(operands),
        "available_operands": list(candidates),
        "compiled": compile_query(payload, available_operands=candidates),
    }


def build_plan(project: ProjectRef, *, write: bool = False) -> dict[str, Any]:
    """Build a deterministic collection plan, optionally writing its artifact.

    The returned plan is the same contract used by the CLI.  ``write=True``
    adds the resulting ``plan_path`` without changing the deterministic plan
    hash.
    """

    config = _project(project)
    try:
        plan = dict(_build_plan(config))
    except Exception as exc:
        raise _as_api_error(exc, "E_PROJECT_PLAN") from exc
    if write:
        try:
            plan["plan_path"] = str(_write_plan(config.path))
        except Exception as exc:
            raise _as_api_error(exc, "E_PROJECT_PLAN_WRITE") from exc
    return plan


def inspect_project(project: ProjectRef) -> dict[str, Any]:
    """Validate a project and compile its plan without writing files."""

    config = _project(project)
    try:
        return dict(_inspect_project(config.path))
    except Exception as exc:
        raise _as_api_error(exc, "E_PROJECT_INSPECTION") from exc


def run_project(
    project: ProjectRef,
    *,
    progress: ProgressCallback | None = None,
) -> RunResult:
    """Execute one supported project and return its committed run summary."""

    config = _project(project)
    try:
        return _run_project(config.path, progress=progress)
    except Exception as exc:
        raise _as_api_error(exc, "E_PROJECT_EXECUTION") from exc


def replay_ledger(
    ledger: str | Path,
    query: str | Path,
    *,
    batch_column: str | None = None,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Replay a CSV BatchLedger through the frozen inference router."""

    try:
        return dict(
            _replay_ledger(
                ledger,
                query,
                batch_column=batch_column,
                confidence_level=confidence_level,
            )
        )
    except Exception as exc:
        raise _as_api_error(exc, "E_LEDGER_INPUT") from exc


def infer_or_refuse(
    query: QueryRef,
    ledger: Any,
    score_names: Sequence[str],
    *,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Route an in-memory ledger to frozen inference or a reason-coded refusal."""

    payload = _query_payload(query)
    try:
        result = _infer_query(
            payload,
            ledger,
            tuple(str(item) for item in score_names),
            confidence_level=confidence_level,
        )
    except Exception as exc:
        raise _as_api_error(exc, "E_INFERENCE_INPUT") from exc
    try:
        safe = _json_safe(result)
    except Exception as exc:
        raise _as_api_error(exc, "E_API_CONTRACT") from exc
    if not isinstance(safe, Mapping):
        raise APIError("The inference router returned a non-mapping result.", "E_API_CONTRACT")
    return dict(safe)


def verify_run(project: ProjectRef, run_id: str = "latest") -> dict[str, Any]:
    """Recompute a run's SHA-256 manifest and report integrity status."""

    config = _project(project)
    try:
        return dict(_verify_run(config.path, run_id))
    except Exception as exc:
        raise _as_api_error(exc, "E_PROJECT_INPUT") from exc


def history(project: ProjectRef) -> list[dict[str, Any]]:
    """Return committed run records, newest first."""

    config = _project(project)
    try:
        return [dict(item) for item in _history(config.path)]
    except Exception as exc:
        raise _as_api_error(exc, "E_PROJECT_INPUT") from exc


def _safe_run_id(run_id: str) -> str:
    if run_id == "latest":
        return run_id
    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise APIError("run_id must name one run directory.", "E_RUN_NOT_FOUND")
    return run_id


def _manifest_for(project: ProjectConfig, run_id: str) -> tuple[str, Path]:
    selected = _safe_run_id(run_id)
    if selected == "latest":
        records = _list_runs(project.path)
        if not records:
            raise APIError("No committed runs exist for this project.", "E_RUN_NOT_FOUND")
        selected = str(records[0].get("run_id", ""))
    run_dir = ProjectLayout(project).run / selected
    manifest = run_dir / "run_manifest.json"
    if not manifest.is_file():
        raise APIError(f"Run manifest not found: {manifest}", "E_RUN_NOT_FOUND")
    return selected, manifest


def show_run(
    project: ProjectRef,
    run_id: str = "latest",
    *,
    result: bool = False,
    report: bool = False,
) -> dict[str, Any]:
    """Return a run manifest, result payload, or report-path record."""

    config = _project(project)
    try:
        selected, manifest_path = _manifest_for(config, run_id)
    except Exception as exc:
        raise _as_api_error(exc, "E_RUN_NOT_FOUND") from exc
    run_dir = manifest_path.parent
    if report:
        report_path = run_dir / "result" / "report.html"
        return {
            "status": "PASS",
            "run_id": selected,
            "report_path": str(report_path),
            "exists": report_path.is_file(),
        }
    if result:
        result_path = run_dir / "result" / "inference.json"
        if not result_path.is_file():
            raise APIError(f"Result file not found: {result_path}", "E_RESULT_NOT_FOUND")
        try:
            return dict(_read_mapping(result_path))
        except Exception as exc:
            raise _as_api_error(exc, "E_RESULT_INPUT") from exc
    try:
        payload = dict(_read_mapping(manifest_path))
    except Exception as exc:
        raise _as_api_error(exc, "E_RUN_MANIFEST") from exc
    payload.setdefault("status", "PASS")
    payload["path"] = str(run_dir)
    return payload


def create_project(path: str | Path) -> Path:
    """Create a portable project template without overwriting an existing file."""

    try:
        return cast(Path, _create_project(path))
    except Exception as exc:
        raise _as_api_error(exc, "E_PROJECT_CREATE") from exc


def load_project(path: str | Path) -> ProjectConfig:
    """Load and validate a project document for programmatic workflows."""

    return _project(path)


def save_query(
    project: str | Path,
    query: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Persist one validated query below a project and attach its reference."""

    try:
        return cast(tuple[Path, Path], _save_query(project, query))
    except Exception as exc:
        raise _as_api_error(exc, "E_QUERY_SAVE") from exc


def reproduce() -> dict[str, Any]:
    """Run the preserved analysis replay in its isolated compatibility context."""

    return dict(_replay_report())


def test_preserved() -> dict[str, Any]:
    """Run the preserved freeze-snapshot test suite and return structured output."""

    return dict(_test_report())


def run_preserved_tests() -> int:
    """Run the preserved freeze-snapshot tests with their native terminal output."""

    return int(_run_tests())


__all__ = [
    "API_VERSION",
    "APIError",
    "build_plan",
    "check_environment",
    "compile_query",
    "create_project",
    "doctor",
    "history",
    "infer_or_refuse",
    "inspect_project",
    "inspect_query",
    "load_project",
    "query_operands",
    "replay_ledger",
    "reproduce",
    "run_preserved_tests",
    "run_project",
    "save_query",
    "show_run",
    "test_preserved",
    "verify_archive",
    "verify_bundle",
    "verify_run",
]
