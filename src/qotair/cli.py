"""Canonical CLI for QOTAIR.

The command line is the primary product interface. It is deliberately
dependency-light, scriptable, and shared with the optional TUI through the
same application services.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from typing import Any, Mapping, Sequence, TextIO

from . import __version__
from .api import (
    build_plan,
    create_project,
    doctor,
    history,
    inspect_project,
    inspect_query,
    replay_ledger,
    reproduce,
    run_preserved_tests,
    run_project,
    show_run,
    test_preserved,
    verify_archive,
    verify_bundle,
    verify_run,
)


class CLIInputError(ValueError):
    """A user-facing input error with a stable exit reason."""

    def __init__(self, message: str, code: str = "E_INPUT"):
        super().__init__(message)
        self.code = code


def _add_json(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")


def build_parser() -> argparse.ArgumentParser:
    description = (
        "Plan derived-response acquisition, replay batch ledgers, run frozen "
        "inference, and return reason-coded refusal."
    )
    parser = argparse.ArgumentParser(prog="qotair", description=description)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--debug", action="store_true", help="show a traceback for unexpected errors")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    doctor_parser = subparsers.add_parser("doctor", help="inspect dependencies, archive, and support boundaries")
    _add_json(doctor_parser)

    test_parser = subparsers.add_parser("test", help="run the preserved freeze-snapshot test suite")
    _add_json(test_parser)

    reproduce_parser = subparsers.add_parser("reproduce", help="rebuild the preserved analysis replay")
    _add_json(reproduce_parser)

    archive_parser = subparsers.add_parser(
        "verify-archive",
        help="verify an archive's exact-byte SHA-256 manifest without running replay",
    )
    archive_parser.add_argument("--archive-root", help="extracted reproduction snapshot directory")
    archive_parser.add_argument("--manifest", help="archive manifest.json path")
    archive_parser.add_argument("--strict-extra", action="store_true", help="also reject unlisted files")
    _add_json(archive_parser)

    bundle_parser = subparsers.add_parser(
        "verify-bundle",
        help="verify software, archive, and dependency-lock bindings in a reviewer bundle",
    )
    bundle_parser.add_argument("bundle", help="reviewer bundle directory")
    bundle_parser.add_argument("--manifest", help="outer reproduction_manifest.json path")
    _add_json(bundle_parser)

    new_parser = subparsers.add_parser("new", help="create a project.yaml template")
    new_parser.add_argument("path", nargs="?", default="project.yaml", help="target project.yaml path")
    _add_json(new_parser)

    validate_parser = subparsers.add_parser("validate", help="validate a project without writing a plan")
    validate_parser.add_argument("project", help="project.yaml path")
    _add_json(validate_parser)

    query_parser = subparsers.add_parser("query", help="inspect and validate query documents")
    query_subparsers = query_parser.add_subparsers(dest="query_command", required=True, metavar="ACTION")
    for action, help_text in (
        ("check", "compile a query and show its minimal operands"),
        ("operands", "show only the minimal primitive operands"),
    ):
        action_parser = query_subparsers.add_parser(action, help=help_text)
        action_parser.add_argument("query", help="query YAML or JSON path")
        action_parser.add_argument(
            "--available",
            help="comma-separated available operand names for compiler validation",
        )
        _add_json(action_parser)

    plan_parser = subparsers.add_parser("plan", help="validate and preview a collection plan")
    plan_parser.add_argument("project", help="project.yaml path")
    plan_parser.add_argument("--no-write", action="store_true", help="preview without writing plan/collection_plan.json")
    _add_json(plan_parser)

    run_parser = subparsers.add_parser("run", help="run a supported project and publish an atomic run record")
    run_parser.add_argument("project", help="project.yaml path")
    run_parser.add_argument(
        "--progress",
        choices=("auto", "always", "never"),
        default="auto",
        help="progress messages on stderr (default: auto)",
    )
    _add_json(run_parser)

    replay_parser = subparsers.add_parser("replay", help="replay a ledger, or the frozen archive when no ledger is given")
    replay_parser.add_argument("ledger", nargs="?", help="CSV ledger path")
    replay_parser.add_argument("--query", help="query YAML or JSON path for a ledger replay")
    replay_parser.add_argument("--batch-column", help="explicit batch identifier column")
    replay_parser.add_argument("--confidence-level", type=float, default=0.95, help="confidence level for direct replay")
    _add_json(replay_parser)

    runs_parser = subparsers.add_parser("infer-runs", help="analyze independent tally histories")
    runs_parser.add_argument("--data", required=True, help="NPZ file containing histories")
    runs_parser.add_argument("--manifest", required=True, help="JSON run and query manifest")
    runs_parser.add_argument("--out", required=True, help="new JSON result file")
    _add_json(runs_parser)

    history_parser = subparsers.add_parser("history", help="list committed project runs")
    history_parser.add_argument("project", help="project.yaml path")
    history_parser.add_argument("--status", choices=("PASS", "REFUSE", "UNSUPPORTED", "FAIL"), help="filter by run status")
    history_parser.add_argument("--limit", type=int, help="return at most N newest runs")
    _add_json(history_parser)

    show_parser = subparsers.add_parser("show", help="show a run manifest or result")
    show_parser.add_argument("project", help="project.yaml path")
    show_parser.add_argument("run_id", nargs="?", default="latest", help="run identifier, or latest")
    show_parser.add_argument("--result", action="store_true", help="show result/inference.json instead of the manifest")
    show_parser.add_argument("--report", action="store_true", help="print the HTML report path")
    _add_json(show_parser)

    verify_parser = subparsers.add_parser("verify", help="verify a run's SHA-256 integrity manifest")
    verify_parser.add_argument("project", help="project.yaml path")
    verify_parser.add_argument("run_id", nargs="?", default="latest", help="run identifier, or latest")
    _add_json(verify_parser)

    tui_parser = subparsers.add_parser("tui", help="optional terminal interface; CLI remains the primary interface")
    tui_parser.add_argument("--project", help="project.yaml to open on startup")
    tui_parser.add_argument("--headless", action="store_true", help="print the optional interface contract and exit")

    return parser


def _json_dump(value: Any, stream: TextIO | None = None) -> None:
    if stream is None:
        stream = sys.stdout
    json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=False, default=str)
    stream.write("\n")


def _status(payload: Mapping[str, Any]) -> str:
    return str(payload.get("status", "UNKNOWN")).upper()


def _status_line(status: str, title: str) -> str:
    return f"{status:<12} {title}"


def _print_inference(inference: Mapping[str, Any], *, stream: TextIO = sys.stdout) -> None:
    if not inference:
        return
    for key, label in (
        ("estimate", "Estimate"),
        ("standard_error", "Standard error"),
        ("method", "Method"),
        ("reason_code", "Reason code"),
    ):
        if key in inference and inference[key] is not None:
            print(f"{label}: {inference[key]}", file=stream)
    interval = inference.get("interval", inference.get("confidence_interval"))
    if interval is not None:
        print(f"Interval: {interval}", file=stream)
    if inference.get("message"):
        print(f"Message: {inference['message']}", file=stream)


def _render_doctor(payload: Mapping[str, Any], stream: TextIO = sys.stdout) -> None:
    print("QOTAIR doctor", file=stream)
    print(_status_line(_status(payload), "Environment check"), file=stream)
    print(f"Python: {payload.get('python')}", file=stream)
    print(f"Platform: {payload.get('platform')}", file=stream)
    print(f"Archive: {'ready' if payload.get('archive_ready') else 'incomplete'}", file=stream)
    dependencies = payload.get("dependencies", {})
    if isinstance(dependencies, Mapping):
        print("Dependencies: " + ", ".join(f"{key}={value}" for key, value in dependencies.items()), file=stream)
    nuclear = payload.get("nuclear_data", {})
    if isinstance(nuclear, Mapping):
        print(f"Nuclear data: {nuclear.get('locator') or 'not configured'}", file=stream)
    support = payload.get("support_matrix", {})
    if isinstance(support, Mapping):
        print("Support:", file=stream)
        for key, value in support.items():
            print(f"  {key}: {value}", file=stream)
    if payload.get("reason_code"):
        print(f"Reason code: {payload['reason_code']}", file=stream)


def _render_query(payload: Mapping[str, Any], *, operands_only: bool = False, stream: TextIO = sys.stdout) -> None:
    print(_status_line(_status(payload), f"Query {payload.get('query_id', '')}"), file=stream)
    if payload.get("path"):
        print(f"Path: {payload['path']}", file=stream)
    if payload.get("name"):
        print(f"Name: {payload['name']}", file=stream)
    operands = payload.get("requested_operands", payload.get("operands", []))
    print("Minimal operands: " + ", ".join(str(item) for item in operands), file=stream)
    if operands_only:
        return
    for key in ("expression_class", "query_sha256"):
        if payload.get(key) is not None:
            print(f"{key.replace('_', ' ').title()}: {payload[key]}", file=stream)
    compiled = payload.get("compiled")
    if compiled is not None:
        print("Compiled contract:", file=stream)
        print(json.dumps(compiled, indent=2, ensure_ascii=False, default=str), file=stream)


def _render_project(payload: Mapping[str, Any], stream: TextIO = sys.stdout) -> None:
    print(_status_line(_status(payload), "Project validation"), file=stream)
    for key in ("project", "project_id", "mode", "plan_sha256", "config_sha256"):
        if payload.get(key) is not None:
            print(f"{key.replace('_', ' ').title()}: {payload[key]}", file=stream)
    ledger = payload.get("ledger", {})
    if isinstance(ledger, Mapping):
        print(f"Ledger: {ledger.get('path') or 'not configured'} ({'present' if ledger.get('exists') else 'missing'})", file=stream)
    print("Required operands: " + ", ".join(str(item) for item in payload.get("required_operands", [])), file=stream)
    excluded = payload.get("excluded_available_operands", [])
    print("Excluded available operands: " + (", ".join(str(item) for item in excluded) or "none"), file=stream)
    if payload.get("reason_code"):
        print(f"Reason code: {payload['reason_code']}", file=stream)


def _render_plan(payload: Mapping[str, Any], stream: TextIO = sys.stdout) -> None:
    print(_status_line(_status(payload), "Collection plan"), file=stream)
    print(f"Project: {payload.get('project_id')}", file=stream)
    if payload.get("plan_path"):
        print(f"Plan path: {payload['plan_path']}", file=stream)
    print("Required operands: " + ", ".join(str(item) for item in payload.get("required_operands", [])), file=stream)
    excluded = payload.get("excluded_available_operands", [])
    print("Excluded available operands: " + (", ".join(str(item) for item in excluded) or "none"), file=stream)
    if payload.get("plan_sha256"):
        print(f"Plan SHA256: {payload['plan_sha256']}", file=stream)
    if payload.get("support"):
        support = payload["support"]
        print("Boundaries:", file=stream)
        for key, value in support.items():
            print(f"  {key}: {value}", file=stream)


def _render_run(payload: Mapping[str, Any], stream: TextIO = sys.stdout) -> None:
    print(_status_line(_status(payload), "Run"), file=stream)
    for key in ("run_id", "path", "plan_path", "result_path", "message"):
        if payload.get(key) is not None:
            print(f"{key.replace('_', ' ').title()}: {payload[key]}", file=stream)
    result = payload.get("result")
    if not isinstance(result, Mapping) and ("queries" in payload or "inference" in payload):
        result = payload
    if isinstance(result, Mapping):
        print(f"Result status: {_status(result)}", file=stream)
        queries = result.get("queries", [])
        if isinstance(queries, list):
            for item in queries:
                if not isinstance(item, Mapping):
                    continue
                print(f"Query: {item.get('query_id', '')} ({item.get('name', '')})", file=stream)
                inference = item.get("inference")
                if isinstance(inference, Mapping):
                    _print_inference(inference, stream=stream)
        if isinstance(result.get("inference"), Mapping):
            _print_inference(result["inference"], stream=stream)
        if result.get("limitations"):
            print("Limitations:", file=stream)
            for limitation in result["limitations"]:
                print(f"  - {limitation}", file=stream)


def _render_replay(payload: Mapping[str, Any], stream: TextIO = sys.stdout) -> None:
    print(_status_line(_status(payload), "Ledger replay"), file=stream)
    for key in ("query_id", "schema_version"):
        if payload.get(key) is not None:
            print(f"{key.replace('_', ' ').title()}: {payload[key]}", file=stream)
    ledger = payload.get("ledger", {})
    if isinstance(ledger, Mapping):
        print(f"Ledger: {ledger.get('path')}", file=stream)
        print(f"Batches: {ledger.get('n_batches')}", file=stream)
    inference = payload.get("inference")
    if isinstance(inference, Mapping):
        _print_inference(inference, stream=stream)


def _render_history(records: Sequence[Mapping[str, Any]], stream: TextIO = sys.stdout) -> None:
    if not records:
        print("No runs found.", file=stream)
        return
    print(f"{'STATUS':<12} {'RUN ID':<30} {'MODE':<18} PROJECT", file=stream)
    print(f"{'-' * 12} {'-' * 30} {'-' * 18} {'-' * 20}", file=stream)
    for record in records:
        print(
            f"{str(record.get('status', 'UNKNOWN')):<12} "
            f"{str(record.get('run_id', '')):<30} "
            f"{str(record.get('mode', '')):<18} "
            f"{record.get('project_id', '')}",
            file=stream,
        )


def _render_show(payload: Mapping[str, Any], stream: TextIO = sys.stdout) -> None:
    print(_status_line(_status(payload), "Run manifest"), file=stream)
    for key in ("run_id", "project_id", "status", "mode", "started_at", "finished_at", "path", "plan_sha256", "result_path", "report_path"):
        if payload.get(key) is not None:
            print(f"{key.replace('_', ' ').title()}: {payload[key]}", file=stream)


def _render_verify(payload: Mapping[str, Any], stream: TextIO = sys.stdout) -> None:
    print(_status_line(_status(payload), "Integrity verification"), file=stream)
    for key in ("project", "run_id", "run_path", "manifest_status", "files_checked"):
        if payload.get(key) is not None:
            print(f"{key.replace('_', ' ').title()}: {payload[key]}", file=stream)
    for key in ("missing", "mismatched", "extra", "malformed"):
        values = payload.get(key, [])
        if values:
            print(f"{key.title()}: {', '.join(str(item) for item in values)}", file=stream)


def _render_reproduce(payload: Mapping[str, Any], stream: TextIO = sys.stdout) -> None:
    print(_status_line(_status(payload), "Preserved analysis replay"), file=stream)
    print(f"Kind: {payload.get('kind', 'analysis_replay_not_full_transport')}", file=stream)
    if payload.get("report"):
        print(f"Report: {payload['report']}", file=stream)
    if payload.get("returncode") is not None:
        print(f"Return code: {payload['returncode']}", file=stream)
    if payload.get("missing"):
        print("Missing: " + ", ".join(str(item) for item in payload["missing"]), file=stream)


def _render_archive_verify(payload: Mapping[str, Any], stream: TextIO = sys.stdout) -> None:
    print(_status_line(_status(payload), "Archive integrity"), file=stream)
    for key in ("archive_root", "manifest", "hash_policy", "files_checked"):
        if payload.get(key) is not None:
            print(f"{key.replace('_', ' ').title()}: {payload[key]}", file=stream)
    for key in ("missing", "mismatched", "extra", "malformed"):
        values = payload.get(key, [])
        if values:
            print(f"{key.title()}: {', '.join(str(item) for item in values)}", file=stream)
    sidecar = payload.get("manifest_sidecar")
    if isinstance(sidecar, Mapping):
        print(f"Manifest sidecar: {'verified' if sidecar.get('verified') else sidecar.get('error')}", file=stream)


def _render_bundle_verify(payload: Mapping[str, Any], stream: TextIO = sys.stdout) -> None:
    print(_status_line(_status(payload), "Reviewer bundle integrity"), file=stream)
    for key in ("bundle_root", "manifest"):
        if payload.get(key) is not None:
            print(f"{key.replace('_', ' ').title()}: {payload[key]}", file=stream)
    for key in ("missing", "mismatched"):
        values = payload.get(key, [])
        if values:
            print(f"{key.title()}: {', '.join(str(item) for item in values)}", file=stream)


def _render_generic(payload: Mapping[str, Any], stream: TextIO = sys.stdout) -> None:
    print(_status_line(_status(payload), str(payload.get("message", "QOTAIR command"))), file=stream)
    for key, value in payload.items():
        if key not in {"status", "message"}:
            print(f"{key.replace('_', ' ').title()}: {value}", file=stream)


def _reason_code(exc: BaseException, fallback: str = "E_RUNTIME") -> str:
    code = getattr(exc, "code", None)
    if code:
        return str(code)
    if isinstance(exc, FileNotFoundError):
        return "E_FILE_NOT_FOUND"
    if isinstance(exc, FileExistsError):
        return "E_EXISTS"
    if isinstance(exc, PermissionError):
        return "E_PERMISSION"
    if isinstance(exc, ValueError):
        return "E_INPUT"
    return fallback


def _exit_code(status: str, reason_code: str | None = None) -> int:
    if reason_code == "E_INTEGRITY":
        return 6
    if reason_code in {"E_ARCHIVE_INTEGRITY", "E_MANIFEST_SCHEMA", "E_MANIFEST_POLICY", "E_MANIFEST_MALFORMED"}:
        return 6
    if reason_code and (
        reason_code.startswith("E_QUERY_")
        or reason_code.startswith("E_PROJECT_")
        or reason_code in {"E_SCHEMA", "E_PLAN", "E_COMMAND"}
    ):
        return 2
    if reason_code in {
        "E_INPUT",
        "E_FILE_NOT_FOUND",
        "E_EXISTS",
        "E_PERMISSION",
        "E_LIMIT",
        "E_QUERY_REQUIRED",
        "E_QUERY_WITHOUT_LEDGER",
        "E_RUN_NOT_FOUND",
        "E_RESULT_NOT_FOUND",
        "E_PROJECT_EXISTS",
    }:
        return 2
    if reason_code in {"E_ARCHIVE_INCOMPLETE", "E_DEPENDENCY", "E_TRANSPORT_UNSUPPORTED", "E_MANIFEST_MISSING"}:
        return 3
    return {"PASS": 0, "REFUSE": 4, "UNSUPPORTED": 3, "FAIL": 5}.get(status.upper(), 5)


def _emit(payload: Any, *, json_mode: bool, renderer: Any = _render_generic) -> None:
    if json_mode:
        _json_dump(payload)
    else:
        renderer(payload)


def _available(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _doctor_payload() -> dict[str, Any]:
    report = dict(doctor())
    missing_paths = [name for name, present in report.get("required_paths", {}).items() if not present]
    missing_dependencies = [
        name for name in ("numpy", "scipy", "yaml")
        if not report.get("dependencies", {}).get(name)
    ]
    if missing_paths:
        report.update({"status": "UNSUPPORTED", "reason_code": "E_ARCHIVE_INCOMPLETE", "missing": missing_paths})
    elif missing_dependencies:
        report.update({"status": "UNSUPPORTED", "reason_code": "E_DEPENDENCY", "missing": missing_dependencies})
    else:
        report.update({"status": "PASS", "reason_code": None})
    return report


def _dispatch(args: argparse.Namespace) -> int:
    command = args.command
    json_mode = bool(getattr(args, "json", False))

    if command is None:
        build_parser().print_help()
        return 0
    if command == "doctor":
        payload = _doctor_payload()
        _emit(payload, json_mode=json_mode, renderer=_render_doctor)
        return _exit_code(_status(payload), payload.get("reason_code"))
    if command == "test":
        if json_mode:
            payload = test_preserved()
            _emit(payload, json_mode=True)
            return _exit_code(_status(payload), payload.get("reason_code"))
        return run_preserved_tests()
    if command == "reproduce":
        payload = reproduce()
        _emit(payload, json_mode=json_mode, renderer=_render_reproduce)
        return _exit_code(_status(payload), payload.get("reason_code"))
    if command == "verify-archive":
        payload = verify_archive(
            args.archive_root,
            manifest=args.manifest,
            strict_extra=args.strict_extra,
        )
        _emit(payload, json_mode=json_mode, renderer=_render_archive_verify)
        return _exit_code(_status(payload), payload.get("reason_code"))
    if command == "verify-bundle":
        payload = verify_bundle(args.bundle, manifest=args.manifest)
        _emit(payload, json_mode=json_mode, renderer=_render_bundle_verify)
        return _exit_code(_status(payload), payload.get("reason_code"))
    if command == "new":
        target = create_project(args.path)
        payload = {"status": "PASS", "project": str(target), "message": "Project template created"}
        _emit(payload, json_mode=json_mode)
        return 0
    if command == "validate":
        payload = inspect_project(args.project)
        _emit(payload, json_mode=json_mode, renderer=_render_project)
        return _exit_code(_status(payload), payload.get("reason_code"))
    if command == "query":
        available = _available(args.available)
        payload = inspect_query(args.query, available_operands=available)
        if args.query_command == "operands":
            payload = {
                "status": payload["status"],
                "query_id": payload["query_id"],
                "path": payload["path"],
                "operands": payload["requested_operands"],
                "query_sha256": payload["query_sha256"],
            }
            _emit(payload, json_mode=json_mode, renderer=lambda value: _render_query(value, operands_only=True))
        else:
            _emit(payload, json_mode=json_mode, renderer=_render_query)
        return _exit_code(_status(payload), payload.get("reason_code"))
    if command == "plan":
        plan = build_plan(args.project, write=not args.no_write)
        payload = {"status": "PASS", **plan}
        _emit(payload, json_mode=json_mode, renderer=_render_plan)
        return 0
    if command == "run":
        progress = None
        if args.progress == "always" or (args.progress == "auto" and sys.stderr.isatty()):
            progress = lambda message: print(f"[qotair] {message}", file=sys.stderr, flush=True)
        summary = run_project(args.project, progress=progress).as_dict()
        if summary.get("run_id"):
            summary["result"] = show_run(args.project, summary["run_id"], result=True)
        _emit(summary, json_mode=json_mode, renderer=_render_run)
        return _exit_code(_status(summary))
    if command == "replay":
        if not args.ledger:
            if args.query:
                raise CLIInputError("--query requires a ledger path.", "E_QUERY_WITHOUT_LEDGER")
            payload = reproduce()
            _emit(payload, json_mode=json_mode, renderer=_render_reproduce)
            return _exit_code(_status(payload), payload.get("reason_code"))
        if not args.query:
            raise CLIInputError("--query is required when replaying a ledger.", "E_QUERY_REQUIRED")
        payload = replay_ledger(
            args.ledger,
            args.query,
            batch_column=args.batch_column,
            confidence_level=args.confidence_level,
        )
        _emit(payload, json_mode=json_mode, renderer=_render_replay)
        return _exit_code(_status(payload))
    if command == "infer-runs":
        from pathlib import Path

        from .inference.replay import analyze_runs

        result = analyze_runs(args.data, args.manifest)
        with Path(args.out).open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        if json_mode:
            _json_dump(result)
        else:
            print(f"{result['status']}: {args.out}")
        return _exit_code("PASS" if result["status"] in {"RELEASE", "DIAGNOSTIC"} else result["status"])
    if command == "history":
        records = history(args.project)
        if args.status:
            records = [record for record in records if str(record.get("status", "")).upper() == args.status]
        if args.limit is not None:
            if args.limit < 0:
                raise CLIInputError("--limit must be non-negative.", "E_LIMIT")
            records = records[: args.limit]
        _emit(records, json_mode=json_mode, renderer=_render_history)
        return 0
    if command == "show":
        payload = show_run(args.project, args.run_id, result=args.result, report=args.report)
        if args.report:
            _emit(payload, json_mode=json_mode)
            return 0 if payload["exists"] else 2
        if args.result:
            _emit(payload, json_mode=json_mode, renderer=_render_run)
            return _exit_code(_status(payload))
        _emit(payload, json_mode=json_mode, renderer=_render_show)
        return _exit_code(_status(payload))
    if command == "verify":
        payload = verify_run(args.project, args.run_id)
        _emit(payload, json_mode=json_mode, renderer=_render_verify)
        return _exit_code(_status(payload), payload.get("reason_code"))
    if command == "tui":
        from .tui.app import launch_tui

        return launch_tui(args.project, headless=args.headless)
    raise CLIInputError(f"Unknown command: {command}", "E_COMMAND")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except BrokenPipeError:
        return 0
    except Exception as exc:
        payload = {
            "status": "FAIL",
            "reason_code": _reason_code(exc),
            "message": str(exc) or exc.__class__.__name__,
        }
        if getattr(args, "json", False):
            _json_dump(payload)
        else:
            print(f"ERROR [{payload['reason_code']}] {payload['message']}", file=sys.stderr)
        if getattr(args, "debug", False):
            traceback.print_exc(file=sys.stderr)
        return _exit_code(payload["status"], payload["reason_code"])


if __name__ == "__main__":
    raise SystemExit(main())
