"""The dashboard's payloads must stay true to the artefacts they summarise.

The endpoints are tested directly rather than over HTTP: the value is in the
payload builders, and starting a server in a test buys a socket and nothing
else. The one thing worth asserting about the server is the guard on the live
endpoint, which is checked by reading it rather than by calling it -- the whole
point of that guard is that a test must not be able to trip it into loading a
model onto a GPU an evaluation run is using.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DASH_PATH = ROOT / "scripts" / "dashboard.py"


def load_dashboard():
    spec = importlib.util.spec_from_file_location("agentic_ir_dashboard", DASH_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["agentic_ir_dashboard"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dash():
    if not DASH_PATH.exists():
        pytest.skip("scripts/dashboard.py is absent")
    return load_dashboard()


@pytest.fixture(scope="module")
def a_complete_run(dash):
    """A run with all 250 questions, or a skip if none is on this machine."""
    for row in dash.status_payload(None)["runs"]:
        if row["complete"] and row["run_id"]:
            return row["run_id"]
    pytest.skip("no complete run on this machine")


def test_the_status_grid_shows_cells_nobody_has_started(dash):
    """A configuration with no run must appear as an empty row, not vanish.

    This is the entire purpose of a progress view: a dashboard that lists only
    the runs that exist cannot show the ones that do not, and reads as finished
    when it is not.
    """
    payload = dash.status_payload(None)
    configs = dash.expected_configs()
    if not configs:
        pytest.skip("the harness exposes no configuration list")
    shown = {(r["config"], r["dataset"]) for r in payload["runs"]}
    for config in configs:
        for dataset in ("hotpotqa", "twowiki"):
            assert (config, dataset) in shown, f"{config}/{dataset} is missing a cell"


def test_progress_is_reported_against_the_evaluation_slice(dash):
    payload = dash.status_payload(None)
    for row in payload["runs"]:
        assert row["target"] == dash.TARGET_N
        assert row["complete"] == (row["n"] >= dash.TARGET_N)
        assert row["n"] >= 0


def test_question_rows_carry_the_scores_beside_the_trace(dash, a_complete_run):
    rows = dash.question_rows(a_complete_run)
    assert len(rows) >= dash.TARGET_N
    scored = [r for r in rows if r["em"] is not None]
    assert scored, "no question in a complete run carries a score"
    for row in scored[:20]:
        assert row["em"] in (0.0, 1.0), f"EM is not binary: {row['em']}"
        assert 0.0 <= row["f1"] <= 1.0


def test_a_question_detail_holds_the_whole_record(dash, a_complete_run):
    qid = dash.question_rows(a_complete_run)[0]["qid"]
    detail = dash.question_detail(a_complete_run, qid)
    record = detail["record"]
    for field in ("question", "final_answer", "transitions", "evidence", "metrics"):
        assert field in record, f"the detail view would render {field} as missing"


def test_tables_parse_into_a_grid_with_no_latex_left(dash):
    tables = dash.table_payload()
    if not tables:
        pytest.skip("no generated tables yet")
    for table in tables:
        assert table["header"], f"{table['label']} parsed to no header"
        assert table["rows"], f"{table['label']} parsed to no rows"
        for row in table["rows"]:
            for cell in row:
                assert "\\" not in cell, (
                    f"{table['label']} leaks LaTeX into the page: {cell!r}"
                )


def test_the_live_endpoint_is_off_unless_asked_for(dash):
    """Answering a question loads models onto the GPU. Default must be off."""
    assert dash.LIVE_ENABLED is False, (
        "importing the dashboard enabled live answering, which would let a "
        "browser tab compete with an evaluation run for VRAM"
    )
    source = DASH_PATH.read_text(encoding="utf-8")
    assert "if not LIVE_ENABLED:" in source, "the POST guard is gone"
    assert "_ASK_LOCK" in source, "concurrent live questions are no longer serialised"
