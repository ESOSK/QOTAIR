"""Dependency-free line-oriented fallback for WSL and minimal SSH sessions."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, TextIO

from ..api import build_plan, check_environment, create_project, history, load_project, run_project, save_query


def _show(output: TextIO, value: object) -> None:
    if isinstance(value, str):
        print(value, file=output)
    else:
        print(json.dumps(value, indent=2, ensure_ascii=False, default=str), file=output)


def run_fallback(
    project_path: str | None = None,
    *,
    input_fn: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
) -> int:
    """Run a small menu that remains usable when Textual is not installed."""

    current = Path(project_path).expanduser().resolve() if project_path else None
    print("QOTAIR | OpenMC Derived Response Workbench", file=output)
    print("Textual is unavailable; using the compatible terminal interface.", file=output)
    while True:
        print("\n[1] New project  [2] Environment  [3] Query builder / plan", file=output)
        print("[4] Run or replay  [5] History  [6] Exit", file=output)
        if current:
            print(f"Project: {current}", file=output)
        try:
            choice = input_fn("Select an action: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.", file=output)
            return 0
        try:
            if choice in {"1", "new", "n"}:
                target = input_fn("Project path [project.yaml]: ").strip() or "project.yaml"
                current = create_project(target)
                _show(output, {"status": "PASS", "project": str(current)})
            elif choice in {"2", "environment", "e"}:
                _show(output, check_environment())
            elif choice in {"3", "plan", "p"}:
                if current is None:
                    current = Path(input_fn("Project path: ").strip()).expanduser().resolve()
                project = load_project(current)
                action = input_fn("Query action [plan/review/save]: ").strip().lower() or "plan"
                if action in {"review", "r"}:
                    _show(output, {"status": "PASS", "project": str(project.path), "queries": [query.as_dict() for query in project.queries()]})
                elif action in {"save", "s"}:
                    query_id = input_fn("Query id [absorption_over_flux]: ").strip() or "absorption_over_flux"
                    name = input_fn("Query name [Absorption over flux]: ").strip() or "Absorption over flux"
                    numerator = input_fn("Numerator [absorption]: ").strip() or "absorption"
                    denominator = input_fn("Denominator [flux]: ").strip() or "flux"
                    if numerator == denominator:
                        raise ValueError("Numerator and denominator must be distinct")
                    payload = {
                        "schema_version": "p1-query-v1",
                        "query_id": query_id,
                        "name": name,
                        "expression_class": "ratio",
                        "output_unit": "1",
                        "operand_units": {numerator: "1", denominator: "1"},
                        "domain_constraints": ["denominator must be stable"],
                        "expression": {"op": "divide", "args": [{"op": "score", "name": numerator}, {"op": "score", "name": denominator}]},
                    }
                    project_path, query_path = save_query(current, payload)
                    _show(output, {"status": "PASS", "project": str(project_path), "query": str(query_path)})
                else:
                    plan = build_plan(project, write=True)
                    _show(output, {"status": "PASS", "plan_path": plan.get("plan_path"), "plan_sha256": plan["plan_sha256"], "required_operands": plan["required_operands"]})
            elif choice in {"4", "run", "r"}:
                if current is None:
                    current = Path(input_fn("Project path: ").strip()).expanduser().resolve()
                result = run_project(current)
                _show(output, result.as_dict())
            elif choice in {"5", "history", "h"}:
                if current is None:
                    current = Path(input_fn("Project path: ").strip()).expanduser().resolve()
                _show(output, history(current))
            elif choice in {"6", "exit", "quit", "q"}:
                print("Exiting.", file=output)
                return 0
            else:
                print("Unknown action. Choose 1-6.", file=output)
        except Exception as exc:
            _show(output, {"status": "FAIL", "reason_code": getattr(exc, "code", "E_TUI"), "message": str(exc)})
