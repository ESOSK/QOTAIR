# Reproduction

Use the software release and data record cited with the article. The data
record supplies the input files, retained histories and result-specific
commands. The source tree includes a small single-history example.

## Independent histories

```sh
qotair infer-runs --data histories.npz --manifest manifest.json --out result.json
```

The NPZ file contains `histories`, a numeric array with axes for runs,
active batches and tally operands. `score_names` in the JSON manifest is
the ordered list of operand names. `query` has a `name`, a `numerator`
mapping operand names to weights, and a `denominator` mapping or `null`
for a linear response.

`runs` has one entry per run. Each entry supplies `run_id`, `seed`,
`independent_stream`, `source_qualified`, `model_sha256`, `identity_sha256`,
`normalization`, `particles`, `batches`, `generations_per_batch`,
`source_protocol_sha256` and `reset_protocol_sha256`. Repeated seeds,
incompatible settings and unqualified source conditions cause refusal.
Only the matching `model_key` and `validation_profile_id` for a registered
query and acquisition protocol can yield an engineering interval. A finite
mathematical set can still be reported with status `DIAGNOSTIC`.

## Historical analysis

The single-history `replay` command can read a CSV ledger directly. The
`verify-archive` and `reproduce` commands use an extracted historical data
archive supplied through `QOTAIR_ARCHIVE_ROOT`.
