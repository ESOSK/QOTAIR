# QOTAIR user guide

## Product model

The Python API is the stable programmatic contract and the CLI is the stable
user and reproducibility front end. Project files declare the model, ledger,
queries, baselines, and run settings; the API validates and executes them; run
directories retain the evidence needed to inspect and reproduce the decision.

The optional TUI calls the same API but is not required for any supported
workflow.

## Python API

```python
from qotair.api import build_plan, compile_query, run_project, verify_run

query = compile_query("query.yaml")
plan = build_plan("project.yaml")
run = run_project("project.yaml")
integrity = verify_run("project.yaml", run.run_id)
```

The API accepts project/query paths and, where documented, in-memory mappings.
The CLI invokes these same functions and adds process-level output and exit
codes. See [api-reference.md](api-reference.md) for the complete contract.

## Project contract

Every project is a directory containing `project.yaml`. QOTAIR creates these
directories as needed:

```text
project.yaml
query/                  # optional external query documents
plan/                   # collection_plan.json
run/                    # immutable run directories and index.jsonl
data/                   # user-owned or imported ledger data
result/                 # optional project-level exports
integrity/              # optional project-level manifests
```

The project schema is `qotair-project-v1`. Supported modes are
`ledger_replay`, `analysis`, `archive_replay`, and `transport`. The first three
have explicit boundaries; `transport` is currently unsupported until a verified
model adapter exists.

## Query contract

Queries use the frozen `p1-query-v1` expression schema. Each query declares an
identifier, expression class, units, domain constraints, and expression tree.
The API sends this document to the audited compiler rather than reimplementing
statistical formulas. The CLI is only one client of that API.

```bash
qotair query check query.yaml
qotair query operands query.yaml
```

## Plan contract

`qotair plan` compiles a deterministic collection plan. The plan records:

- required primitive operands;
- available ledger columns and excluded columns;
- query, configuration, and plan hashes;
- declared comparison baselines;
- supported and unsupported execution boundaries.

Operand or bin counts are not reported as transport, statepoint, runtime, or
memory savings. Those claims require a named baseline and direct measurement.

## Execution contract

`qotair run project.yaml` creates a new run directory and never overwrites an
earlier run. The result contains the inference status, estimates and intervals
when available, diagnostics, limitations, source hashes, and a reproducible
artifact path.

`qotair replay ledger.csv --query query.yaml` is a direct shortcut for an
existing CSV ledger. It uses the same inference service but deliberately does
not create a project run record.

## Inspecting evidence

```bash
qotair history project.yaml
qotair show project.yaml latest
qotair show project.yaml latest --result
qotair show project.yaml latest --report
qotair verify project.yaml latest
```

`verify` recomputes the SHA-256 integrity manifest and distinguishes file
corruption from a scientific refusal.

## Automation

Use JSON on stdout for machine consumption. Progress messages from `run` go to
stderr, so this pattern remains parseable:

```bash
qotair run project.yaml --json --progress never > run-summary.json
```

The exit code is also stable:

```text
0  PASS
2  user input, schema, or missing-file error
3  unsupported environment or feature
4  scientific refusal with valid execution
5  program or runtime failure
6  integrity verification failure
130 user interruption
```

## Scientific and reproducibility boundaries

The product layer preserves the frozen inference router and production
expression path. It does not silently turn static configuration reduction into
transport, runtime, memory, statepoint, or physical-decision claims. The
preserved archive remains a separate evidence boundary and is not rewritten by
ordinary CLI commands.
