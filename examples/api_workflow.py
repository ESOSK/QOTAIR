"""Read-only public API example for QOTAIR."""
from __future__ import annotations

from pathlib import Path

from qotair.api import build_plan, compile_query, inspect_project


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    project = ROOT / "examples" / "projects" / "pin-replay.yaml"
    query = ROOT / "examples" / "projects" / "query-ratio.yaml"

    compiled = compile_query(query)
    plan = build_plan(project)
    validation = inspect_project(project)

    print("QOTAIR public API")
    print("Query operands:", ", ".join(compiled.get("operands", [])))
    print("Plan hash:", plan["plan_sha256"])
    print("Project status:", validation["status"])


if __name__ == "__main__":
    main()
