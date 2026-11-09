from qotair.paths import ARCHIVE_ROOT, required_archive_paths
from qotair.runtime import doctor


def test_archive_boundary_is_explicit() -> None:
    assert ARCHIVE_ROOT.name == "reproduction_snapshot"
    required = required_archive_paths()
    assert all(path.parent == ARCHIVE_ROOT for path in required)


def test_doctor_reports_archive_state_without_assuming_bundled_evidence() -> None:
    report = doctor()
    expected = all(path.exists() for path in required_archive_paths())
    assert report["archive_ready"] is expected
    assert report["archive_manifest"]["status"] in {"PASS", "UNSUPPORTED", "FAIL"}
