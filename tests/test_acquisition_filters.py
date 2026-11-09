from __future__ import annotations

import numpy as np
import pytest

from qotair.acquisition import (
    AcquisitionIdentity,
    FilterSpec,
    first_bin_selection,
    install_table,
    mesh_ijk_selection,
    mesh_linear_index,
    require_distribcell_instances,
    route_energy_group_ratio,
    route_steps,
    same_identity,
    select_and_route,
    select_mesh_ijk_and_route,
    slice_two_energy_groups,
    slice_two_scores,
)
from qotair.acquisition.identity import AcquisitionError
from qotair.acquisition.select import mesh_center_ijk
from qotair.acquisition.tables import TABLES, get_table


def test_mesh_bin_count_scales() -> None:
    a = FilterSpec.mesh(2, 2, 1, (-1, -1, -1), (1, 1, 1))
    b = FilterSpec.mesh(4, 4, 1, (-1, -1, -1), (1, 1, 1))
    c = FilterSpec.mesh(8, 8, 1, (-1, -1, -1), (1, 1, 1))
    assert a.bin_count() == 4
    assert b.bin_count() == 16
    assert c.bin_count() == 64
    native = FilterSpec.native_mesh(17, 2, 2, 1, (-1, -1, -1), (1, 1, 1))
    assert native.bin_count() == 4
    assert native.native_mesh_id() == 17
    assert a.native_mesh_id() is None
    overlay = AcquisitionIdentity((a,), ("total",), ("flux", "absorption"))
    bound = AcquisitionIdentity((native,), ("total",), ("flux", "absorption"))
    assert overlay.sha256 != bound.sha256


def test_universe_and_distribcell() -> None:
    u = FilterSpec.universe((3, 7))
    d = FilterSpec.distribcell(12)
    assert u.bin_count() == 2
    assert d.bin_count() == 1
    counted = FilterSpec.distribcell(12, expected_instances=157)
    assert counted.bin_count() == 157
    ident = AcquisitionIdentity((u,), ("total",), ("flux",))
    assert len(ident.sha256) == 64


def test_distribcell_instance_mismatch_is_fail_closed() -> None:
    require_distribcell_instances(actual=157, expected=157, cell_id=12)
    with pytest.raises(AcquisitionError) as mismatch:
        require_distribcell_instances(actual=8, expected=7, cell_id=12)
    assert mismatch.value.code == "E_DISTRIBCELL_INSTANCE_MISMATCH"
    with pytest.raises(AcquisitionError) as empty:
        require_distribcell_instances(actual=0, expected=1, cell_id=12)
    assert empty.value.code == "E_DISTRIBCELL_INSTANCES"
    with pytest.raises(AcquisitionError) as bad:
        FilterSpec.distribcell(12, expected_instances=0)
    assert bad.value.code == "E_DISTRIBCELL_INSTANCES"


def test_energy_function_identity() -> None:
    spec = FilterSpec.energy_function((1e-5, 0.625, 2e7), (1.0, 1.0, 1.0))
    assert spec.filter_type == "EnergyFunctionFilter"
    assert spec.bin_count() == 1
    assert spec.energy_function_table() is None


def test_energy_function_tables_are_registered_only() -> None:
    unity = FilterSpec.energy_function_from_table("unity")
    assert unity.energy_function_table() == "unity"
    assert unity.bin_count() == 1
    table = get_table("unity")
    assert table.purpose.startswith("weighted-flux probe")
    assert "not ICRP" in table.purpose
    with pytest.raises(AcquisitionError) as missing:
        FilterSpec.energy_function_from_table("icrp-116")
    assert missing.value.code == "E_ENERGY_FUNCTION_TABLE"
    with pytest.raises(AcquisitionError) as empty:
        FilterSpec.energy_function_from_table("")
    assert empty.value.code == "E_ENERGY_FUNCTION_TABLE"


def test_xml_regular_mesh_skips_cylindrical(tmp_path) -> None:
    from qotair.acquisition.native_mesh import parse_regular_mesh_records, spec_from_mesh_record

    xml = tmp_path / "tallies.xml"
    xml.write_text(
        """<?xml version="1.0"?>
<tallies>
  <mesh id="2">
    <dimension>1 1 1</dimension>
    <lower_left>-1 -1 -1</lower_left>
    <upper_right>1 1 1</upper_right>
  </mesh>
  <mesh id="1">
    <dimension>17 17 1</dimension>
    <lower_left>-182.78 -182.78 36.748</lower_left>
    <upper_right>182.78 182.78 402.508</upper_right>
  </mesh>
  <mesh id="3" type="cylindrical">
    <dimension>5 8 4</dimension>
    <origin>0 0 0</origin>
  </mesh>
</tallies>
""",
        encoding="utf-8",
    )
    from qotair.acquisition.native_mesh import records_from_xml_paths

    records = parse_regular_mesh_records(xml)
    assert {item["id"] for item in records} == {1, 2}
    ranked = records_from_xml_paths([xml])
    assert ranked[0]["id"] == 1
    spec = spec_from_mesh_record(ranked[0])
    assert spec.native_mesh_id() == 1
    assert spec.bin_count() == 289


def test_material_filter_identity() -> None:
    spec = FilterSpec.material((3, 8))
    assert spec.filter_type == "MaterialFilter"
    assert spec.bin_count() == 2
    with pytest.raises(AcquisitionError):
        FilterSpec.material((3, 3))


def test_depletion_same_identity() -> None:
    mesh = FilterSpec.mesh(2, 2, 1, (-1, -1, -1), (1, 1, 1))
    energy = FilterSpec.energy((0.0, 0.625, 2e7))
    left = AcquisitionIdentity((energy, mesh), ("total",), ("flux", "absorption"))
    right = AcquisitionIdentity((energy, mesh), ("total",), ("flux", "absorption"))
    assert same_identity(left, right)


def test_duplicate_filter_type_rejected() -> None:
    e1 = FilterSpec.energy((0.0, 1.0, 2.0))
    e2 = FilterSpec.energy((0.0, 0.5, 2.0))
    with pytest.raises(AcquisitionError):
        AcquisitionIdentity((e1, e2), ("total",), ("flux",))


def _mesh_energy_identity() -> AcquisitionIdentity:
    return AcquisitionIdentity(
        (
            FilterSpec.energy((0.0, 0.625, 2e7)),
            FilterSpec.mesh(2, 2, 1, (-1, -1, -1), (1, 1, 1)),
        ),
        ("total",),
        ("flux", "absorption"),
    )


def test_selected_mesh_bin_is_two_columns() -> None:
    identity = _mesh_energy_identity()
    n_batch = 12
    n_filter = 2 * 4
    ledger = np.zeros((n_batch, n_filter * 2))
    # scores = (flux, absorption); last filter (mesh) changes fastest
    energy_bin, mesh_bin = 1, 3
    filter_index = energy_bin * 4 + mesh_bin
    flux = 4.0 + 0.01 * np.arange(n_batch)
    absorption = 1.0 + 0.002 * np.arange(n_batch)
    ledger[:, filter_index * 2] = flux
    ledger[:, filter_index * 2 + 1] = absorption
    two = slice_two_scores(
        ledger,
        identity,
        {"EnergyFilter": energy_bin, "MeshFilter": mesh_bin},
        "absorption",
        "flux",
    )
    assert two.shape == (n_batch, 2)
    np.testing.assert_allclose(two[:, 0], absorption)
    np.testing.assert_allclose(two[:, 1], flux)
    with pytest.raises(AcquisitionError) as missing:
        slice_two_scores(ledger, identity, {"MeshFilter": 0}, "absorption", "flux")
    assert missing.value.code == "E_BIN_SELECTION"


def test_selected_bin_routes_through_frozen_thresholds() -> None:
    identity = _mesh_energy_identity()
    n_batch = 60
    n_cols = identity.filter_bin_count() * 2
    batches = np.arange(1, n_batch + 1, dtype=float)
    ledger = np.zeros((n_batch, n_cols))
    for filter_index in range(identity.filter_bin_count()):
        ledger[:, filter_index * 2] = 4.0 + 0.001 * batches
        ledger[:, filter_index * 2 + 1] = 1.0 + 0.0002 * batches
    routed = select_and_route(
        ledger,
        identity,
        {"EnergyFilter": 0, "MeshFilter": 2},
        numerator="absorption",
        denominator="flux",
    )
    assert routed["selected_shape"] == [60, 2]
    assert routed["table4"] == "unchanged"
    status = routed["inference"]["status"]
    assert status in {"PASS", "REFUSE"}
    from qotair.services.scientific import _thresholds

    thresholds = _thresholds()
    assert thresholds.denominator_t_refuse_below == 1.0
    assert thresholds.denominator_t_fieller_below == 3.0
    assert thresholds.minimum_batches == 12


def test_short_selected_ledger_is_refused() -> None:
    identity = _mesh_energy_identity()
    ledger = np.ones((2, identity.filter_bin_count() * 2))
    ledger[:, 0::2] = 4.0
    routed = select_and_route(ledger, identity, first_bin_selection(identity))
    assert routed["inference"]["status"] == "REFUSE"


def test_depletion_steps_reuse_same_identity_and_router() -> None:
    identity = _mesh_energy_identity()
    n_batch = 60
    n_cols = identity.filter_bin_count() * 2
    batches = np.arange(1, n_batch + 1, dtype=float)
    step0 = np.zeros((n_batch, n_cols))
    step1 = np.zeros((n_batch, n_cols))
    for filter_index in range(identity.filter_bin_count()):
        step0[:, filter_index * 2] = 4.0 + 0.001 * batches
        step0[:, filter_index * 2 + 1] = 1.0 + 0.0002 * batches
        step1[:, filter_index * 2] = 3.5 + 0.001 * batches
        step1[:, filter_index * 2 + 1] = 0.9 + 0.0002 * batches
    rows = route_steps([step0, step1], identity, first_bin_selection(identity))
    assert len(rows) == 2
    assert rows[0]["identity_sha256"] == rows[1]["identity_sha256"] == identity.sha256
    assert rows[0]["step"] == 0 and rows[1]["step"] == 1
    assert "not a new Table 4" in rows[0]["table4"]
    assert rows[0]["inference"]["status"] in {"PASS", "REFUSE"}
    assert rows[1]["inference"]["status"] in {"PASS", "REFUSE"}


def test_mesh_ijk_and_center_selection() -> None:
    assert mesh_linear_index(2, 2, 1, 1, 1, 0) == 3
    assert mesh_center_ijk(17, 17, 1) == (8, 8, 0)
    identity = _mesh_energy_identity()
    chosen = mesh_ijk_selection(identity, 1, 1, 0, energy_bin=1)
    assert chosen == {"EnergyFilter": 1, "MeshFilter": 3}


def test_fast_over_thermal_is_two_columns_not_nested() -> None:
    identity = _mesh_energy_identity()
    n_batch = 60
    ledger = np.zeros((n_batch, identity.filter_bin_count() * 2))
    batches = np.arange(1, n_batch + 1, dtype=float)
    # mesh (1,1,0) = 3; thermal energy 0, fast energy 1; score flux at even columns
    thermal = 5.0 + 0.001 * batches
    fast = 2.0 + 0.0004 * batches
    ledger[:, (0 * 4 + 3) * 2] = thermal
    ledger[:, (1 * 4 + 3) * 2] = fast
    two = slice_two_energy_groups(
        ledger,
        identity,
        numerator_energy=1,
        denominator_energy=0,
        score="flux",
        mesh_ijk=(1, 1, 0),
    )
    np.testing.assert_allclose(two[:, 0], fast)
    np.testing.assert_allclose(two[:, 1], thermal)
    routed = route_energy_group_ratio(
        ledger,
        identity,
        numerator_energy=1,
        denominator_energy=0,
        mesh_ijk=(1, 1, 0),
    )
    assert routed["nested_ratio"] is False
    assert routed["table4"] == "unchanged"
    assert routed["inference"]["status"] in {"PASS", "REFUSE"}
    ijk = select_mesh_ijk_and_route(ledger, identity, 1, 1, 0, energy_bin=0)
    assert ijk["mesh_ijk"] == [1, 1, 0]
    assert ijk["inference"]["status"] in {"PASS", "REFUSE"}


def test_named_nuclide_cannot_use_flux_score() -> None:
    from qotair.acquisition.adapter import _preflight

    identity = AcquisitionIdentity(
        (FilterSpec.material((1,)),),
        ("U235",),
        ("fission", "flux"),
    )
    class Dummy:
        geometry = None
        materials = []

    with pytest.raises(AcquisitionError) as flux:
        _preflight(identity, Dummy())
    assert flux.value.code == "E_NUCLIDE_FLUX"


def test_install_table_rejects_icrp_and_length_mismatch() -> None:
    with pytest.raises(AcquisitionError) as icrp:
        install_table("icrp-116", (1e-5, 2e7), (1.0, 1.0), "no")
    assert icrp.value.code == "E_ENERGY_FUNCTION_TABLE"
    with pytest.raises(AcquisitionError) as dose:
        install_table("neutron-dose", (1e-5, 2e7), (1.0, 1.0), "no")
    assert dose.value.code == "E_ENERGY_FUNCTION_TABLE"
    with pytest.raises(AcquisitionError) as length:
        install_table("probe", (1e-5, 0.625, 2e7), (1.0, 1.0), "len")
    assert length.value.code == "E_ENERGY_FUNCTION_TABLE"
    table = install_table("probe_unity", (1e-5, 2e7), (1.0, 1.0), "test probe")
    assert table.name == "probe_unity"
    TABLES.pop("probe_unity", None)


def test_openmc_filters_optional() -> None:
    openmc = pytest.importorskip("openmc")
    from qotair.acquisition.adapter import to_openmc_filter

    mesh = to_openmc_filter(FilterSpec.mesh(2, 2, 1, (-1, -1, -1), (1, 1, 1)))
    assert type(mesh).__name__ == "MeshFilter"
    univ = to_openmc_filter(FilterSpec.universe((1,)))
    assert type(univ).__name__ == "UniverseFilter"
    dist = to_openmc_filter(FilterSpec.distribcell(1, expected_instances=4))
    assert type(dist).__name__ == "DistribcellFilter"
    eff = to_openmc_filter(FilterSpec.energy_function_from_table("unity"))
    assert type(eff).__name__ == "EnergyFunctionFilter"
    mat = to_openmc_filter(FilterSpec.material((1,)))
    assert type(mat).__name__ == "MaterialFilter"
    _ = openmc
