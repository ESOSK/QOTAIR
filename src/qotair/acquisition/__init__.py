"""OpenMC acquisition extensions (Mesh, Universe, Distribcell, Material, EnergyFunction).

Frozen query Cell/Energy compilation is unchanged. These types are an
additive adapter surface. Selected-bin ratios reuse the frozen T_den / hull
router; they do not retune thresholds or replace frozen thresholds.
"""

from .adapter import attach_identity, distribcell_instance_count, require_distribcell_instances
from .cells import unique_cell_ids, unique_cell_spec
from .depletion import identity_sha256, route_steps, same_identity
from .extract import extract_run, extract_selected_columns, extract_tally_ledger, peak_rss_bytes
from .identity import AcquisitionError, AcquisitionIdentity, FilterSpec
from .native_mesh import candidate_mesh_xml_paths, discover_regular_meshes, spec_from_regular_mesh
from .route import route_energy_group_ratio, select_and_route, select_mesh_ijk_and_route
from .select import (
    center_mesh_selection,
    first_bin_selection,
    mesh_ijk_selection,
    mesh_linear_index,
    slice_two_energy_groups,
    slice_two_scores,
)
from .tables import get_table, install_table
from .targets import pick_distribcell_target, pick_fissionable_material, pick_universe_target
from .bind import RegionSpec, bind_region, binding_record
from .pipeline import evaluate_declared_query, replay_from_two_column, summarize_records
from .prerun import jackknife_partition_ok, prerun_check
from .provenance import InferenceRule, QueryTask, RunSource, file_sha256, ledger_mix_forbidden, same_query_identity
from .archive import load_record, load_two_column, write_task_bundle
from .status import classify_inference, paper_release_status
from .paths import MESH_INDEX_CONVENTION, distribcell_instance_path, mesh_bounds_from_identity
from .correspondence import region_correspondence, snapshot_correspondence
from .plan import ledger_reuse_decision, planned_columns, retention_record
from .baseline import simple_denominator_gate
from .contract import (
    QueryContract,
    can_reuse_batch_evidence,
    can_reuse_query_definition,
    preflight_contract,
    require_complete_contract,
)
from .cost import LayeredCost
from .diagnose import (
    QUALITY_RULES,
    judge_task,
    no_diagnostic_envelope,
    production_infer,
    simple_gated_envelope,
    simple_labels,
    summarize_evaluation,
)
from .planner import RECORDED_SIMPLE_RULES, legal_plans_same_tally, select_plan
from .validation import (
    independent_mesh_columns,
    ratio_point,
    reference_sensitivity,
    source_protocol_decision,
    wilson,
)

__all__ = [
    "AcquisitionError",
    "AcquisitionIdentity",
    "FilterSpec",
    "attach_identity",
    "candidate_mesh_xml_paths",
    "discover_regular_meshes",
    "distribcell_instance_count",
    "extract_tally_ledger",
    "extract_run",
    "peak_rss_bytes",
    "extract_selected_columns",
    "center_mesh_selection",
    "first_bin_selection",
    "get_table",
    "identity_sha256",
    "install_table",
    "mesh_ijk_selection",
    "mesh_linear_index",
    "pick_distribcell_target",
    "pick_fissionable_material",
    "pick_universe_target",
    "require_distribcell_instances",
    "route_energy_group_ratio",
    "route_steps",
    "same_identity",
    "select_and_route",
    "select_mesh_ijk_and_route",
    "slice_two_energy_groups",
    "slice_two_scores",
    "spec_from_regular_mesh",
    "unique_cell_ids",
    "unique_cell_spec",
    "RegionSpec",
    "bind_region",
    "binding_record",
    "evaluate_declared_query",
    "replay_from_two_column",
    "summarize_records",
    "jackknife_partition_ok",
    "prerun_check",
    "InferenceRule",
    "QueryTask",
    "RunSource",
    "file_sha256",
    "write_task_bundle",
    "load_two_column",
    "load_record",
    "ledger_mix_forbidden",
    "same_query_identity",
    "classify_inference",
    "paper_release_status",
    "MESH_INDEX_CONVENTION",
    "distribcell_instance_path",
    "mesh_bounds_from_identity",
    "region_correspondence",
    "snapshot_correspondence",
    "ledger_reuse_decision",
    "planned_columns",
    "retention_record",
    "simple_denominator_gate",
    "QueryContract",
    "can_reuse_query_definition",
    "can_reuse_batch_evidence",
    "preflight_contract",
    "require_complete_contract",
    "LayeredCost",
    "QUALITY_RULES",
    "judge_task",
    "no_diagnostic_envelope",
    "production_infer",
    "simple_labels",
    "simple_gated_envelope",
    "summarize_evaluation",
    "legal_plans_same_tally",
    "select_plan",
    "RECORDED_SIMPLE_RULES",
    "wilson",
    "ratio_point",
    "source_protocol_decision",
    "reference_sensitivity",
    "independent_mesh_columns",
]
