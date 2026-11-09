# QOTAIR quick start

QOTAIR is designed to be used from the command line. A TTY is optional.

## 1. Check the environment

```bash
cd /path/to/QOTAIR
python -m qotair doctor
# POSIX source checkout: ./qotair doctor
```

The check is read-only. It reports Python, scientific dependencies, archive
readiness, nuclear-data configuration, and the explicit support matrix.

## 2. Validate a project and query

```bash
./qotair validate examples/projects/pin-replay.yaml
./qotair query check examples/projects/query-ratio.yaml
./qotair query operands examples/projects/query-ratio.yaml
```

Validation compiles the query without writing a plan.

## 3. Preview the acquisition plan

```bash
./qotair plan examples/projects/pin-replay.yaml
./qotair plan examples/projects/pin-replay.yaml --no-write
```

The plan shows minimal operands, excluded available columns, hashes, baselines,
and support limits. It does not start OpenMC transport.

## 4. Run and verify

```bash
./qotair run examples/projects/pin-replay.yaml
./qotair history examples/projects/pin-replay.yaml
./qotair show examples/projects/pin-replay.yaml latest --result
./qotair verify examples/projects/pin-replay.yaml latest
```

Each run receives a new identifier and is committed atomically below the
project's `run/` directory. JSON, HTML, and a SHA-256 manifest are retained.

For automation, use JSON on stdout and progress on stderr:

```bash
./qotair run examples/projects/pin-replay.yaml --json --progress never
```

## 5. Replay a ledger directly

```bash
./qotair replay \
  examples/data/batch_operand_ledger.csv \
  --query examples/projects/query-ratio.yaml
```

This uses the historical single-history method but does not create a project run
record.

## 6. Reproduce the preserved analysis snapshot

```bash
./qotair reproduce
```

The result is an analysis replay and does not rerun the full OpenMC transport
campaigns.

See [cli-reference.md](cli-reference.md) for all commands, options, status
codes, and artifact contracts.
