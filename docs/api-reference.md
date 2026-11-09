# QOTAIR Python API

QOTAIR uses an API-first architecture with a CLI-first product surface. The
public Python API is the programmatic contract; the `qotair` command invokes
the same functions for reproducibility, batch execution, and CI.

## Import

```python
from qotair.api import API_VERSION, build_plan, compile_query, run_project
```

The package root re-exports the same public functions:

```python
from qotair import build_plan, compile_query, run_project
```

The current contract version is available as `qotair.api.API_VERSION`.

## Query compilation

```python
from qotair.api import compile_query, inspect_query, query_operands

query_path = "examples/projects/query-ratio.yaml"
compiled = compile_query(query_path)
inspection = inspect_query(query_path)
operands = query_operands(query_path)
```

`compile_query` accepts a YAML/JSON path or an in-memory mapping and returns a
JSON-safe frozen compiler contract. It does not write files or run transport.

## Planning

```python
from qotair.api import build_plan, inspect_project

project_path = "examples/projects/pin-replay.yaml"
preview = build_plan(project_path)
written = build_plan(project_path, write=True)
validation = inspect_project(project_path)
```

The default is read-only. `write=True` creates the project's
`plan/collection_plan.json` and adds `plan_path` to the returned mapping.

## Execution and inference

```python
from qotair.api import infer_or_refuse, replay_ledger, run_project

run = run_project(project_path)
direct = replay_ledger(
    "examples/data/batch_operand_ledger.csv",
    query_path,
)
```

For an in-memory ledger, call `infer_or_refuse(query, ledger, score_names)`.
The returned result is either a valid inference or a reason-coded `REFUSE`;
the API does not convert an invalid diagnostic into a finite interval.

`run_project` returns a `RunResult` with the committed run identifier and
artifact paths. A progress callback is optional:

```python
run = run_project(project_path, progress=lambda message: print(message))
```

## Results and integrity

```python
from qotair.api import history, show_run, verify_run

records = history(project_path)
manifest = show_run(project_path)
result = show_run(project_path, result=True)
report = show_run(project_path, report=True)
integrity = verify_run(project_path)
```

`verify_run` recomputes the run's SHA-256 manifest. A scientific `REFUSE` is a
valid analysis result; an `E_INTEGRITY` result indicates artifact corruption or
incomplete publication.

Archive and reviewer-bundle checks are separate from project-run integrity:

```python
from qotair.api import verify_archive, verify_bundle

archive_check = verify_archive("archive/reproduction_snapshot")
bundle_check = verify_bundle("QOTAIR-Reviewer-Bundle")
```

These checks use exact-byte SHA-256 for identity and do not normalise CRLF/LF
unless a caller explicitly requests a separate canonical text comparison.

## Environment and preserved evidence

```python
from qotair.api import check_environment, reproduce, test_preserved

environment = check_environment()
replay = reproduce()
tests = test_preserved()
```

The preserved replay reports `analysis_replay_not_full_transport`. These
functions do not claim transport acceleration, statepoint reduction, runtime
speedup, or memory reduction.

## Error and result contract

Boundary failures in the public facade raise `qotair.api.APIError`. The
exception always carries a machine-readable `.code`, for example
`E_QUERY_SCHEMA`, `E_LEDGER_INPUT`, `E_FILE_NOT_FOUND`, `E_RUN_NOT_FOUND`, or
`E_API_CONTRACT`. Callers can display the message while branching on the code
without parsing English text. Unexpected internal failures are not silently
converted into scientific results.

Scientific refusals and unsupported execution boundaries are not exceptions:
they are structured mappings with `status` equal to `REFUSE` or `UNSUPPORTED`
and a reason code such as `E_DENOMINATOR_UNIDENTIFIED`,
`E_TRANSPORT_UNSUPPORTED`, or `E_LEDGER_MISSING`.

## Stability boundary

Only names exported in `qotair.api.__all__` are public. Modules under
`qotair.services`, `qotair.storage`, and `qotair.runtime` remain implementation
layers and may change without being treated as the public API. The frozen
freeze archive is read through explicit adapters and is never rewritten by
these functions.
