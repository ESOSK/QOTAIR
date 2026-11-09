"""QOTAIR public package facade.

The stable Python API is exposed from :mod:`qotair.api`; the command line
interface is a reproducible front end over the same functions.
"""

__version__ = "0.3.0"

from .api import (
    API_VERSION,
    APIError,
    build_plan,
    check_environment,
    compile_query,
    create_project,
    doctor,
    history,
    infer_or_refuse,
    inspect_project,
    inspect_query,
    load_project,
    query_operands,
    replay_ledger,
    reproduce,
    run_preserved_tests,
    run_project,
    save_query,
    show_run,
    test_preserved,
    verify_archive,
    verify_bundle,
    verify_run,
)

__all__ = [
    "__version__",
    "API_VERSION",
    "APIError",
    "build_plan",
    "check_environment",
    "compile_query",
    "create_project",
    "doctor",
    "history",
    "infer_or_refuse",
    "inspect_project",
    "inspect_query",
    "load_project",
    "query_operands",
    "replay_ledger",
    "reproduce",
    "run_preserved_tests",
    "run_project",
    "save_query",
    "show_run",
    "test_preserved",
    "verify_archive",
    "verify_bundle",
    "verify_run",
]
