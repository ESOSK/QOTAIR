# Architecture

The query compiler resolves each expression to primitive tally operands.
The acquisition adapter binds those operands to OpenMC scores and filters.
The ledger keeps active-batch increments and the corresponding operand
identities for reuse.

`qotair replay` evaluates the historical single-history methods on a CSV
ledger. `qotair infer-runs` evaluates the independent-run method on an NPZ
bundle. It constructs a confidence set from independent run means, then
checks the reporting conditions separately.

The package installs as `qotair`. See the [Python API](api-reference.md),
[acquisition guide](acquisition.md) and [reproduction guide](reproducibility.md).
