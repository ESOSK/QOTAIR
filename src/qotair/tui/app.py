"""Textual application with a dependency-free fallback and headless contract."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

from ..api import (
    build_plan,
    check_environment,
    create_project,
    history,
    load_project,
    replay_ledger,
    run_project,
    save_query,
    show_run,
)
from .fallback import run_fallback


def headless_contract(project_path: str | None = None) -> dict[str, Any]:
    """Return the deterministic interface contract used by CI and non-TTY sessions."""
    return {
        "application": "QOTAIR | OpenMC Derived Response Workbench",
        "mode": "headless",
        "project": str(Path(project_path).expanduser().resolve()) if project_path else None,
        "screens": ["Home", "Environment", "Model", "Query Builder", "Plan Preview", "Run Monitor", "Result", "History", "Settings"],
        "actions": ["new", "doctor", "model", "query", "plan", "run", "replay", "history", "show", "settings", "exit"],
        "execution_boundary": "CLI, TUI, and future GUI call the same public Python API",
        "worker_boundary": "OpenMC transport and analysis replay run outside the UI thread",
        "support": check_environment()["support_matrix"],
    }


try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical
    from textual.screen import Screen
    from textual.widgets import Button, Footer, Header, Input, RichLog, Static
    from textual.worker import Worker
    try:  # Textual moved the decorator to the package root in newer releases.
        from textual.worker import work
    except ImportError:  # pragma: no cover - depends on the installed Textual generation
        from textual import work
except ImportError:  # pragma: no cover - used when the optional extra is absent
    App = None  # type: ignore[assignment]


if App is not None:

    def _write_json(log: Any, value: Any) -> None:
        log.write(json.dumps(value, indent=2, ensure_ascii=False, default=str))


    class HomeScreen(Screen):
        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            yield Vertical(
                Static("QOTAIR\nOpenMC Derived Response Workbench", id="home-title"),
                Static("Choose a workflow. Scientific logic stays in the application services.", id="home-subtitle"),
                Horizontal(Button("New project", id="new"), Button("Environment", id="environment"), Button("Model", id="model"), Button("Query builder", id="query")),
                Horizontal(Button("Plan preview", id="plan"), Button("Run / replay", id="run"), Button("History", id="history"), Button("Settings", id="settings")),
                Button("Exit", id="exit", variant="error"),
            )
            yield Footer()

        def on_button_pressed(self, event: Button.Pressed) -> None:
            app = self.app
            if event.button.id == "new":
                app.push_screen(NewProjectScreen())
            elif event.button.id == "environment":
                app.push_screen(EnvironmentScreen())
            elif event.button.id == "model":
                app.push_screen(ModelScreen(app.project_path))
            elif event.button.id == "query":
                app.push_screen(QueryScreen(app.project_path))
            elif event.button.id == "plan":
                app.push_screen(PlanScreen(app.project_path))
            elif event.button.id == "run":
                app.push_screen(RunScreen(app.project_path))
            elif event.button.id == "history":
                app.push_screen(HistoryScreen(app.project_path))
            elif event.button.id == "settings":
                app.push_screen(SettingsScreen())
            elif event.button.id == "exit":
                app.exit()


    class EnvironmentScreen(Screen):
        def compose(self) -> ComposeResult:
            yield Header()
            yield Static("Environment and support matrix", id="screen-title")
            yield RichLog(id="environment-log", highlight=True, markup=False)
            yield Button("Back", id="back")
            yield Footer()

        def on_mount(self) -> None:
            _write_json(self.query_one("#environment-log", RichLog), check_environment())

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "back":
                self.app.pop_screen()


    class NewProjectScreen(Screen):
        def compose(self) -> ComposeResult:
            yield Header()
            yield Static("Create a project", id="screen-title")
            yield Input(value="project.yaml", placeholder="Path to project.yaml", id="project-path")
            yield Horizontal(Button("Create", id="create", variant="primary"), Button("Back", id="back"))
            yield RichLog(id="new-log", markup=False)
            yield Footer()

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "back":
                self.app.pop_screen()
                return
            target = self.query_one("#project-path", Input).value.strip() or "project.yaml"
            try:
                path = create_project(target)
                self.app.project_path = str(path)
                self.query_one("#new-log", RichLog).write(f"Created {path}")
            except Exception as exc:
                self.query_one("#new-log", RichLog).write(f"ERROR: {exc}")


    class ModelScreen(Screen):
        def __init__(self, project_path: str | None = None, **kwargs: Any):
            super().__init__(**kwargs)
            self.initial_project = project_path or "project.yaml"

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static("Model and source", id="screen-title")
            yield Input(value=self.initial_project, placeholder="Path to project.yaml", id="project-path")
            yield Horizontal(Button("Inspect", id="inspect", variant="primary"), Button("Back", id="back"))
            yield RichLog(id="model-log", markup=False)
            yield Footer()

        def on_mount(self) -> None:
            self._inspect()

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "back":
                self.app.pop_screen()
            elif event.button.id == "inspect":
                self._inspect()

        def _inspect(self) -> None:
            log = self.query_one("#model-log", RichLog)
            try:
                project = load_project(self.query_one("#project-path", Input).value.strip())
                _write_json(log, {"project": str(project.path), "project_id": project.project_id, "mode": project.mode, "model": project.model, "ledger": project.ledger})
            except Exception as exc:
                log.write(f"ERROR: {exc}")


    class QueryScreen(Screen):
        def __init__(self, project_path: str | None = None, **kwargs: Any):
            super().__init__(**kwargs)
            self.initial_project = project_path or "project.yaml"

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static("Query builder", id="screen-title")
            yield Input(value=self.initial_project, placeholder="Path to project.yaml", id="project-path")
            yield Horizontal(Input(value="absorption_over_flux", placeholder="Query identifier", id="query-id"), Input(value="Absorption over flux", placeholder="Query name", id="query-name"))
            yield Horizontal(Input(value="absorption", placeholder="Numerator operand", id="numerator"), Input(value="flux", placeholder="Denominator operand", id="denominator"))
            yield Input(value="1", placeholder="Output unit", id="output-unit")
            yield Horizontal(Button("Review queries", id="review"), Button("Save ratio query", id="save", variant="primary"), Button("Back", id="back"))
            yield RichLog(id="query-log", markup=False)
            yield Footer()

        def on_mount(self) -> None:
            self._review()

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "back":
                self.app.pop_screen()
            elif event.button.id == "review":
                self._review()
            elif event.button.id == "save":
                self._save_ratio()

        def _save_ratio(self) -> None:
            log = self.query_one("#query-log", RichLog)
            try:
                numerator = self.query_one("#numerator", Input).value.strip()
                denominator = self.query_one("#denominator", Input).value.strip()
                if not numerator or not denominator or numerator == denominator:
                    raise ValueError("Numerator and denominator must be distinct non-empty operands")
                payload = {
                    "schema_version": "p1-query-v1",
                    "query_id": self.query_one("#query-id", Input).value.strip(),
                    "name": self.query_one("#query-name", Input).value.strip(),
                    "expression_class": "ratio",
                    "output_unit": self.query_one("#output-unit", Input).value.strip() or "1",
                    "operand_units": {
                        numerator: "1",
                        denominator: "1",
                    },
                    "domain_constraints": ["denominator must be stable"],
                    "expression": {
                        "op": "divide",
                        "args": [
                            {"op": "score", "name": numerator},
                            {"op": "score", "name": denominator},
                        ],
                    },
                }
                project_path, query_path = save_query(self.query_one("#project-path", Input).value.strip(), payload)
                self.app.project_path = str(project_path)
                log.write(f"Saved {query_path} and attached it to {project_path}")
                self._review()
            except Exception as exc:
                log.write(f"ERROR: {exc}")

        def _review(self) -> None:
            log = self.query_one("#query-log", RichLog)
            try:
                project = load_project(self.query_one("#project-path", Input).value.strip())
                _write_json(log, {"project": str(project.path), "queries": [query.as_dict() for query in project.queries()], "note": "Edit project.yaml or an external query document, then build a plan."})
            except Exception as exc:
                log.write(f"ERROR: {exc}")


    class PlanScreen(Screen):
        def __init__(self, project_path: str | None = None, **kwargs: Any):
            super().__init__(**kwargs)
            self.initial_project = project_path or "project.yaml"

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static("Query and collection plan", id="screen-title")
            yield Input(value=self.initial_project, placeholder="Path to project.yaml", id="project-path")
            yield Horizontal(Button("Build plan", id="build", variant="primary"), Button("Back", id="back"))
            yield RichLog(id="plan-log", markup=False)
            yield Footer()

        def on_mount(self) -> None:
            self._build()

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "back":
                self.app.pop_screen()
            elif event.button.id == "build":
                self._build()

        def _build(self) -> None:
            log = self.query_one("#plan-log", RichLog)
            try:
                project = load_project(self.query_one("#project-path", Input).value.strip())
                plan = build_plan(project, write=True)
                self.app.project_path = str(project.path)
                _write_json(log, plan)
            except Exception as exc:
                log.write(f"ERROR: {exc}")


    class RunScreen(Screen):
        def __init__(self, project_path: str | None = None, **kwargs: Any):
            super().__init__(**kwargs)
            self.initial_project = project_path or "project.yaml"
            self.active_worker: Worker[Any] | None = None

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static("Run monitor", id="screen-title")
            yield Input(value=self.initial_project, placeholder="Path to project.yaml", id="project-path")
            yield Horizontal(Input(placeholder="CSV ledger for direct replay (optional)", id="ledger-path"), Input(placeholder="Query YAML/JSON for direct replay (optional)", id="query-path"))
            yield Horizontal(Button("Start project", id="start", variant="primary"), Button("Replay ledger", id="replay"), Button("Cancel", id="cancel"), Button("Back", id="back"))
            yield RichLog(id="run-log", markup=False)
            yield Footer()

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "back":
                self.app.pop_screen()
                return
            if event.button.id == "cancel":
                if self.active_worker is not None:
                    self.active_worker.cancel()
                    self.query_one("#run-log", RichLog).write("Cancellation requested.")
                return
            if event.button.id == "start":
                project = self.query_one("#project-path", Input).value.strip()
                self.query_one("#run-log", RichLog).write(f"Starting {project}")
                self.active_worker = self.run_in_worker("project", project, "", "")
            elif event.button.id == "replay":
                ledger = self.query_one("#ledger-path", Input).value.strip()
                query = self.query_one("#query-path", Input).value.strip()
                self.query_one("#run-log", RichLog).write(f"Replaying {ledger}")
                self.active_worker = self.run_in_worker("replay", "", ledger, query)

        @work(thread=True)
        def run_in_worker(self, mode: str, project: str, ledger: str, query: str) -> None:
            try:
                if mode == "project":
                    summary = run_project(project).as_dict()
                    if summary.get("run_id"):
                        summary["result"] = show_run(project, summary["run_id"], result=True)
                else:
                    summary = replay_ledger(ledger, query)
                self.app.call_from_thread(self._show_result, summary)
            except Exception as exc:
                self.app.call_from_thread(self._show_result, {"status": "FAIL", "reason_code": "E_TUI_RUN", "message": str(exc)})

        def _show_result(self, result: dict[str, Any]) -> None:
            self.query_one("#run-log", RichLog).write(json.dumps(result, indent=2, ensure_ascii=False, default=str))
            self.app.push_screen(ResultScreen(result))


    class ResultScreen(Screen):
        def __init__(self, result: Mapping[str, Any], **kwargs: Any):
            super().__init__(**kwargs)
            self.result = dict(result)

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static(f"Result: {self.result.get('status', 'UNKNOWN')}", id="screen-title")
            yield RichLog(id="result-log", markup=False)
            yield Button("Back", id="back")
            yield Footer()

        def on_mount(self) -> None:
            _write_json(self.query_one("#result-log", RichLog), self.result)

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "back":
                self.app.pop_screen()


    class HistoryScreen(Screen):
        def __init__(self, project_path: str | None = None, **kwargs: Any):
            super().__init__(**kwargs)
            self.initial_project = project_path or "project.yaml"

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static("Run history", id="screen-title")
            yield Input(value=self.initial_project, placeholder="Path to project.yaml", id="project-path")
            yield Horizontal(Button("Refresh", id="refresh", variant="primary"), Button("Back", id="back"))
            yield RichLog(id="history-log", markup=False)
            yield Footer()

        def on_mount(self) -> None:
            self._refresh()

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "back":
                self.app.pop_screen()
            elif event.button.id == "refresh":
                self._refresh()

        def _refresh(self) -> None:
            log = self.query_one("#history-log", RichLog)
            try:
                _write_json(log, history(self.query_one("#project-path", Input).value.strip()))
            except Exception as exc:
                log.write(f"ERROR: {exc}")


    class SettingsScreen(Screen):
        def compose(self) -> ComposeResult:
            yield Header()
            yield Static("Settings and boundaries", id="screen-title")
            yield RichLog(id="settings-log", markup=False)
            yield Button("Back", id="back")
            yield Footer()

        def on_mount(self) -> None:
            self.query_one("#settings-log", RichLog).write("Accepted environment is read-only.\nMesh/Universe/Distribcell/Material/EnergyFunction (registered tables only) are acquisition extensions. Selected-bin ratios reuse frozen T_den/hull; frozen router thresholds are unchanged. MPI, restart, and generations_per_batch > 1 remain unverified.\nNo unmeasured runtime, memory, or statepoint gain is shown as a result. EnergyFunction is not an ICRP dose claim.")

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "back":
                self.app.pop_screen()


    class QotairApp(App[None]):
        CSS = """
        Screen { align: center middle; }
        Vertical { width: 94%; height: 92%; padding: 1 2; }
        Horizontal { height: auto; margin: 1 0; }
        Horizontal Input { width: 1fr; }
        Button { margin: 0 1; }
        #home-title, #screen-title { text-style: bold; color: $accent; margin: 1 0; }
        #home-subtitle { color: $text-muted; margin-bottom: 1; }
        RichLog { height: 1fr; border: round $panel; }
        Input { margin: 1 0; }
        """

        def __init__(self, project_path: str | None = None, **kwargs: Any):
            super().__init__(**kwargs)
            self.project_path = project_path

        def on_mount(self) -> None:
            self.push_screen(HomeScreen())


def launch_tui(project_path: str | None = None, *, headless: bool = False) -> int:
    """Start Textual, fall back to a line UI, or print a CI contract."""
    if headless or not sys.stdin.isatty():
        print(json.dumps(headless_contract(project_path), indent=2, ensure_ascii=False, default=str))
        return 0
    if App is None:
        return run_fallback(project_path)
    QotairApp(project_path).run()
    return 0
