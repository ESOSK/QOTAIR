# Development guide

## API-first architecture and CLI-first surface

The public Python API is the programmatic boundary and the command line is the
primary user and reproducibility surface. Keep the following layers
independent:

- `domain/`: stable project, query, and run contracts;
- `api.py`: public functions, typed input conventions, and API error contract;
- `services/`: environment, planning, inspection, execution, and scientific
  adapters;
- `storage/`: project paths, atomic runs, history, and integrity manifests;
- `cli.py`: argument parsing, output rendering, and exit-code mapping over the
  public API;
- `tui/`: optional front end over the public API.

Only names exported in `qotair.api.__all__` are public. Do not reimplement
formulas, denominator thresholds, query compilation, path discovery, or hash
rules inside the API, command, or widget.

When adding a public function, document its accepted path/mapping inputs,
return shape, failure behavior, and evidence boundary. Add a compatibility
test before using it from the CLI.

## Adding a command

1. Define a stable input and output contract.
2. Put behavior in a service with an explicit failure code.
3. Add or extend the public API facade and document its contract.
4. Make the CLI call the public API; add human-readable output and a `--json`
   representation.
5. Map PASS, REFUSE, UNSUPPORTED, FAIL, and integrity failure to stable exit
   codes.
6. Add an API compatibility, example, or fault-injection test.
7. Update `docs/api-reference.md`, `docs/cli-reference.md`, and
   `docs/user-guide.md`.

## Local checks

```bash
PYTHONPATH=src python -m pytest -q tests
./qotair --version
./qotair doctor
./qotair validate examples/projects/pin-replay.yaml
./qotair plan examples/projects/pin-replay.yaml --json
./qotair run examples/projects/pin-replay.yaml --json --progress never
./qotair verify examples/projects/pin-replay.yaml latest
./qotair test
./qotair reproduce
```

When the optional development extras are available, also run the scoped type
check and strict documentation build:

```bash
python -m mypy
PYTHONPATH=src python -m mkdocs build --strict
```

The type check intentionally covers the public facade (`src/qotair/api.py`),
not the preserved research archive or every dynamic compatibility adapter.
The documentation build renders the symbol-level API page from the same
`qotair.api.__all__` contract used by callers.

The accepted OpenMC 0.15.3 environment remains read-only. Install optional
Textual or other development dependencies in a separate environment.

## Evidence preservation

Keep the historical archive read-only unless repairing a replay defect. Do not
create product directories named after research weeks, dates, or manuscript
versions. Preserve negative results, hashes, and explicit unsupported claims.
