# QOTAIR CLI reference

The CLI is the canonical user and reproducibility interface. Every command can
run without a TTY and the commands that return data accept `--json` for scripts
and CI. Programmatic callers should use [the Python API](api-reference.md),
which is the implementation contract behind these commands.

## Global options

```text
qotair --version
qotair --debug COMMAND ...
qotair --help
```

`--debug` adds a traceback to an otherwise concise error. It does not change the
scientific result or the stored run artifacts.

## Environment and evidence

### `doctor`

Read-only environment, dependency, nuclear-data, archive, and support check.

```bash
qotair doctor
qotair doctor --json
```

The command does not start transport. A missing archive or required scientific
dependency returns `UNSUPPORTED`.

### `test`

Runs the preserved freeze-snapshot test suite in its original import context.

```bash
qotair test
qotair test --json
```

### `reproduce`

Runs the preserved analysis replay and returns a structured report. This is not
a full OpenMC transport campaign.

```bash
qotair reproduce
qotair reproduce --json
```

`qotair replay` with no ledger is an alias for the same operation.

### `verify-archive`

Verifies the extracted reproduction snapshot against its exact-byte SHA-256
manifest. It never starts OpenMC transport or rewrites the archive.

```bash
qotair verify-archive --archive-root archive/reproduction_snapshot
qotair verify-archive --archive-root archive/reproduction_snapshot --strict-extra --json
```

The default identity policy is binary SHA-256. CRLF/LF normalisation is not
applied implicitly.

### `verify-bundle`

Verifies the outer reviewer-bundle binding, including the archive tree hash,
QOTAIR wheel hash, and dependency-lock hash.

```bash
qotair verify-bundle reviewer-bundle --json
```

## Projects and queries

### `new`

Creates a project template and refuses to overwrite an existing file.

```bash
qotair new project.yaml
qotair new project.yaml --json
```

The template expects a user-owned ledger at `data/batch_operand_ledger.csv`;
it does not silently point a copied project at the preserved research archive.

### `validate`

Loads a project, validates all query documents, compiles the plan in memory, and
does not write `plan/collection_plan.json`.

```bash
qotair validate project.yaml
qotair validate project.yaml --json
```

### `query check`

Validates and compiles one YAML or JSON query document.

```bash
qotair query check query.yaml
qotair query check query.yaml --available absorption,flux,total --json
```

The output includes the query hash, expression class, minimal primitive
operands, and the frozen compiler contract.

### `query operands`

Prints only the minimal primitive operands needed by one query.

```bash
qotair query operands query.yaml
qotair query operands query.yaml --json
```

## Planning and execution

### `plan`

Compiles and, by default, writes `plan/collection_plan.json`.

```bash
qotair plan project.yaml
qotair plan project.yaml --json
qotair plan project.yaml --no-write
```

The plan records required operands, available and excluded ledger columns,
configuration and plan hashes, baselines, and support boundaries.

### `run`

Runs a project. Each invocation creates a new run identifier and atomically
publishes its result, report, manifest, and integrity file.

```bash
qotair run project.yaml
qotair run project.yaml --json
qotair run project.yaml --progress always
```

Progress is written to stderr, so JSON on stdout remains parseable. Use
`--progress never` for completely quiet batch execution.

### `replay`

Replays an existing CSV ledger with a query document. It does not create a
project run directory.

```bash
qotair replay ledger.csv --query query.yaml
qotair replay ledger.csv --query query.yaml --confidence-level 0.95 --json
qotair replay ledger.csv --query query.yaml --batch-column active_batch
```

Without a ledger, `qotair replay` runs the preserved analysis replay.

## Run records

### `history`

Lists committed runs, newest first.

```bash
qotair history project.yaml
qotair history project.yaml --status REFUSE --limit 10 --json
```

### `show`

Shows the latest run by default, or a specific run identifier.

```bash
qotair show project.yaml
qotair show project.yaml RUN_ID
qotair show project.yaml latest --result
qotair show project.yaml latest --report
qotair show project.yaml latest --json
```

### `verify`

Recomputes every SHA-256 entry in a run's integrity manifest.

```bash
qotair verify project.yaml
qotair verify project.yaml RUN_ID --json
```

An integrity failure returns exit code 6. It is distinct from a scientific
`REFUSE` and from a program error.

## Status and exit codes

```text
0  PASS
2  user input, schema, or missing-file error
3  unsupported environment or feature
4  scientific refusal with a valid execution
5  program or runtime failure
6  integrity verification failure
130 user interruption
```

The JSON form uses a top-level `status` field. Scientific refusals are valid
results and are never relabelled as crashes.

## Run artifact contract

```text
project.yaml
plan/collection_plan.json
run/<RUN_ID>/
├── run_manifest.json
├── result/inference.json
├── result/report.html
└── integrity/MANIFEST.sha256
```

Failed, unsupported, and refused executions remain visible as run records after
their workspace is committed. An interrupted process may leave only a hidden
staging directory, which can be removed safely before retrying. The archive
evidence itself is not rewritten by the product layer.

## `qotair infer-runs`

Construct a confidence set from independent tally histories. Supply an NPZ
bundle with `--data`, its JSON manifest with `--manifest` and a new JSON
output path with `--out`. See [Reproduction](reproducibility.md) for the
input format and reporting statuses.
