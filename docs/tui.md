# QOTAIR optional terminal interface

The CLI is the canonical interface. This document describes the optional TUI;
users who work through SSH, batch systems, or CI can use the CLI alone.

## Screens

When the optional Textual dependency is installed, the application exposes the
following screens:

1. Home: choose a project workflow.
2. Environment: inspect versions, nuclear-data configuration, permissions, and
   the support matrix.
3. Model: inspect the project mode, model declaration, and ledger source.
4. Query Builder: review declared queries or save a ratio query to `query/` and
   attach it to the project atomically.
5. Plan Preview: compile the minimal acquisition plan and show its hashes.
6. Run Monitor: start a supported run in a worker and stream the final status.
7. Result: inspect the JSON result and local report path.
8. History: list immutable run manifests and open a selected run.
9. Settings: read-only execution and evidence boundaries.

The Run Monitor accepts either a project path or a CSV ledger plus query path.
Project runs and direct replays use the same public API calls; a completed run
opens the Result screen with its status, refusal reason when applicable, and
artifact paths.

## Interaction boundary

Widgets collect input and display state. They do not implement statistical
formulas, denominator thresholds, plan rules, OpenMC path discovery, or evidence
hashing. The CLI, TUI, and future GUI call the same public Python API, so the same input
has one configuration hash, plan hash, and inference contract.

## Long-running work

Textual workers keep project execution outside the UI thread. OpenMC transport,
when a verified adapter is added, must run as an isolated child process. The
worker may read logs and state events, but it must not execute scientific formulas
inside a widget. The line-oriented fallback follows the same service boundary.

## Testing

The headless contract is deterministic and is used by CI. With the optional
Textual dependency installed, `textual pilot` tests can exercise buttons and
navigation. Without it, the fallback exposes the same core actions: create a
project, inspect the environment, review or save a query, build a plan, run or
replay, inspect history, and exit.
