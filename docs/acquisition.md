# Acquisition extensions

The query compiler handles cell, energy and particle filters. The adapter
adds the filters listed below. Selected-bin replay uses the historical
single-history method.

## What is in product

- `RegularMesh` / `MeshFilter`, including the model's own RegularMesh
- `UniverseFilter`
- `DistribcellFilter` with an OpenMC instance-count check
- `MaterialFilter` (two scores still divide)
- `EnergyFunctionFilter` from a **registered table only**
- unique-instance `CellFilter` (the compiler's existing cell path)
- selected-bin absorption/flux columns routed through the frozen gate
- the same identity hash on depleted materials, then per-step ΔSUM routing

```python
from qotair.acquisition import (
    AcquisitionIdentity,
    FilterSpec,
    attach_identity,
    first_bin_selection,
    select_and_route,
    spec_from_regular_mesh,
    discover_regular_meshes,
)

meshes = discover_regular_meshes(model)
identity = AcquisitionIdentity(
    (
        FilterSpec.energy((0.0, 0.625, 2.0e7)),
        spec_from_regular_mesh(meshes[0]),
    ),
    ("total",),
    ("flux", "absorption"),
)
model, tally_id = attach_identity(model, identity)
# After extract of the ΔSUM ledger:
routed = select_and_route(ledger, identity, first_bin_selection(identity))
```

A selected mesh bin, universe, or distribcell instance is sliced to two
columns and passed to `qotair.services.scientific.infer_query`. Short or
ill-conditioned ledgers return `REFUSE`. That is the three-state gate, not a
new table.

Use `mesh_ijk_selection` / `select_mesh_ijk_and_route` for a RegularMesh
\((i,j,k)\) (center of a 17×17 assembly mesh is \((8,8,0)\)), not linear bin 0.
`route_energy_group_ratio` divides the same score at two EnergyFilter bins
(fast/thermal flux). That is still two columns, not a nested ratio.
`pick_distribcell_target` prefers a fissionable pin cell whose OpenMC instance
count is then checked. Named nuclides missing from the model fail closed.

## Distribcell instances

Binding a `DistribcellFilter` is not an instance count. The identity must
carry `expected_instances`. After `geometry.determine_paths()`, OpenMC's
`cell.num_instances` must match. A mismatch raises
`E_DISTRIBCELL_INSTANCE_MISMATCH` and does not proceed.

## Native RegularMesh

Prefer the RegularMesh already on the model (power mesh, tally mesh). If the
loaded geometry XML does not carry it, `discover_regular_meshes` reads
`tallies.xml` / `model.xml`. Overlay `2×2×1` remains a constructor for models
that have no RegularMesh. Cylindrical, spherical, and unstructured meshes are
rejected.

## EnergyFunctionFilter

Only names in `qotair.acquisition.tables.TABLES` attach. The shipped table is
`unity` (\(y = 1\)): a weighted-flux probe. Missing names or length mismatches
fail closed. There is no default ICRP table and no “dose verified” claim.

## Query contract and reuse

A `QueryContract` records query meaning. Energy meaning is the actual edges
plus the bin index, not the index alone. Integer ledger columns are storage
layout; score names select columns. `evaluate_declared_query` binds a complete
contract and calls `preflight_contract` before routing. Incomplete contracts
(`identity_sha256` missing, or energy edges missing when the identity has an
EnergyFilter) are refused at execute and at `can_reuse_batch_evidence`.

Bound-contract reuse also checks `identity_sha256`. A score-order change on
the identity therefore requires rebind; it is not absorbed automatically.

Material, geometry, seed, or nuclear-data byte changes set
`E_RUN_SOURCE_MISMATCH`. That is isolation, not a depletion-physics claim.

## Legal plans and recorded simple rules

Same-tally legal plans are `full_then_slice` and `selected_copy`. They do not
change the object, estimator, normalization, or paired batches.
`select_plan` chooses among **measured** records. If `rss_limit_bytes` is set
and a plan's RSS is unknown, the plan is rejected as
`rss_unknown_cannot_claim_limit`.

Recorded simple rules (`RECORDED_SIMPLE_RULES`):

- `selected_copy if n_kept < n_identity else full_then_slice`
- `one_transport_per_identity_for_the_declared_query_set`

These are fixed schemes, not an optimizer and not a cost model. On the
measured mesh task the storage ratio is about 510; three isolated repeats did
not show a stable wall-time or RSS advantage. Against a competent workflow
that already selects columns and shares one transport, QOTAIR does not add a
resource increment; it adds bind/reuse checks.

Paper-envelope evaluation maps an exact-zero jackknife denominator
(`E_ZERO_DENOMINATOR`) to REFUSE. It does not retune \(T_{den}\).

For the documented queries (UAM cell, BEAVRS mesh center, BEAVRS pin,
NUR axial bottom), independent target-response prep chose 100 inactive
batches rather than 4. That decision did not use \(k_\mathrm{eff}\) or confirmation
hit rate. Pin oracles were raised to 8×20000 so all 30 confirms were
judgeable; the 40000 fallback was not used.

## Depletion steps

Identity hash does not depend on material composition. Each depletion step
extracts its own ΔSUM ledger, then routes that ledger. The per-step decisions
are not a new Table 4.
