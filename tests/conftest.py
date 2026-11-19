"""Shared fixtures + a results recorder.

Every test carries ``@pytest.mark.case(ID, component, condition, expected)`` and calls ``actual("...")`` with a short
description of what it observed.  At the end of the session the recorder writes
``outputs/validation/test_results.csv`` in the exact layout of the template's *Test Cases* sheet
(Test ID | Component | Input / Condition | Expected Result | Actual Result | Pass / Fail), so the workbook evidence
is produced by really executing the tests - not typed by hand.
"""
from __future__ import annotations

import sys
import warnings

import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from cvi import config as C  # noqa: E402
from cvi.utils import PIPELINE_PACKAGES, environment_issues  # noqa: E402

_missing = [i for i in environment_issues(only=PIPELINE_PACKAGES) if "not installed" in i]
if _missing:     # a bare `pytest` often belongs to another Python (e.g. Anaconda): say so instead of failing on an import deep in a test module
    pytest.exit(f"This Python ({sys.executable}) is missing packages the tests need:\n  - " + "\n  - ".join(_missing)
                + "\nUse the project's environment:   ./.venv/bin/python -m pytest\nor install the pinned packages:  python -m pip install -r requirements.txt", returncode=2)

RESULTS: list[dict] = []


def pytest_configure(config):
    config.addinivalue_line("markers", "case(id, component, condition, expected): test-case metadata for the workbook")


@pytest.fixture
def actual(request):
    box = {"text": ""}
    request.node._actual = box

    def setter(text) -> None:
        box["text"] = str(text)
    return setter


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    m = item.get_closest_marker("case")
    if m is None:
        return
    if rep.when == "call" or (rep.when == "setup" and not rep.passed):
        tid, comp, cond, exp = m.args
        act = getattr(item, "_actual", {"text": ""})["text"]
        if not rep.passed:
            act = (act + " | " if act else "") + (str(call.excinfo.value)[:160].replace("\n", " ") if call.excinfo else "setup failed / skipped")
        RESULTS.append({"Test ID": tid, "Component": comp, "Input / Condition": cond, "Expected Result": exp, "Actual Result": act,
                        "Pass / Fail": "Pass" if rep.passed else "Fail"})


def pytest_sessionfinish(session, exitstatus):
    if not RESULTS:
        return
    C.VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(RESULTS).drop_duplicates("Test ID", keep="last").sort_values("Test ID")
    df.to_csv(C.VALIDATION_DIR / "test_results.csv", index=False)


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="session", autouse=True)
def _artifacts_present():
    need = [C.BRONZE_FILE, C.SILVER_FILE, C.WAREHOUSE_DB, C.ANALYTICS_FILE, C.METRIC_DIR / "sentiment.json", C.METRIC_DIR / "decision.json"]
    missing = [p.name for p in need if not p.exists()]
    if missing:
        pytest.exit(f"Pipeline artifacts missing ({missing}). Run `python run_pipeline.py` first.", returncode=2)


@pytest.fixture(scope="session")
def bronze() -> pd.DataFrame:
    return pd.read_parquet(C.BRONZE_FILE)


@pytest.fixture(scope="session")
def raw_excel() -> pd.DataFrame:
    return pd.read_excel(C.RAW_XLSX, sheet_name=C.RAW_SHEET)


@pytest.fixture(scope="session")
def silver() -> pd.DataFrame:
    return pd.read_parquet(C.SILVER_FILE)


@pytest.fixture(scope="session")
def analytics() -> pd.DataFrame:
    return pd.read_parquet(C.ANALYTICS_FILE)


@pytest.fixture
def bronze_sample(bronze) -> pd.DataFrame:
    """60-row copy of the Bronze layer for fault injection (fresh copy per test)."""
    return bronze.head(60).copy().reset_index(drop=True)
