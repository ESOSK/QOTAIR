# QOTAIR

QOTAIR compiles derived-response queries for OpenMC tallies and analyzes
retained tally histories. The compiler identifies the operands required by
each query. The same histories can be reused for queries over those operands.

## Installation

```sh
git clone --branch v0.3.0 https://github.com/ESOSK/QOTAIR.git
cd QOTAIR
python -m pip install .
```

QOTAIR requires Python 3.10 or later. Offline analysis uses NumPy and SciPy.
Collecting new histories also requires OpenMC.

## Quick start

From the repository root, compile the example query and replay the included
CSV ledger:

```sh
qotair query check examples/projects/query-ratio.yaml
qotair replay examples/data/batch_operand_ledger.csv --query examples/projects/query-ratio.yaml
```

This example uses the historical single-history method.

## Analyze independent runs

```sh
qotair infer-runs --data histories.npz --manifest manifest.json --out result.json
```

The NPZ contains the tally values indexed by run, active batch and operand.
The manifest defines the query and records the acquisition and source
conditions. The output reports the complete confidence set and a separate
reporting status. An unregistered acquisition profile yields no engineering
interval.

See the [reproduction guide](docs/reproducibility.md) for the input format,
the [CLI reference](docs/cli-reference.md) for command options, and the
[Python API](docs/api-reference.md) for use from Python.

## License

QOTAIR is available under the [MIT License](LICENSE).
