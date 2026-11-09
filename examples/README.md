# Examples

The examples are small, source-only API and CLI workflows. Generated plans and
run directories are created only when a command is executed.

## Included projects

- `projects/pin-replay.yaml` validates, plans, and replays
  `examples/data/batch_operand_ledger.csv`.
- `projects/query-ratio.yaml` is a standalone query document for direct replay.
- `projects/archive-reproduction.yaml` invokes the preserved analysis replay
  boundary and reports that it is not full transport.
- `projects/mesh-ratio.yaml` documents a selected-bin absorption/flux query
  for use with `qotair.acquisition` (native RegularMesh preferred).

Run from the package root:

```bash
./qotair validate examples/projects/pin-replay.yaml
./qotair plan examples/projects/pin-replay.yaml
./qotair run examples/projects/pin-replay.yaml --json --progress never
./qotair verify examples/projects/pin-replay.yaml latest
```

The Python API example is read-only:

```bash
PYTHONPATH=src python examples/api_workflow.py
```

The examples call the same public API used by the CLI and optional TUI.
