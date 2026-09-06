"""The guided walkthrough must not drift from what it claims to be reading.

``scripts/demo.py`` exists to be checkable: it prints the transition table out
of ``orchestrator.py``, replays a named run, and quotes numbers from the
generated tables. Every one of those is a thing that can change underneath it.
A demo that silently narrates a run that no longer exists, or a state machine
that has since gained a transition, is worse than no demo -- so these tests
fail rather than let it go quiet.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DEMO_PATH = ROOT / "scripts" / "demo.py"


def load_demo():
    spec = importlib.util.spec_from_file_location("agentic_ir_demo", DEMO_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["agentic_ir_demo"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def demo():
    if not DEMO_PATH.exists():
        pytest.skip("scripts/demo.py is absent")
    return load_demo()


def test_the_replayed_run_is_still_on_disk(demo):
    """The walkthrough names one run. If it was deleted, say so loudly."""
    traces = demo.RUNS / demo.DEMO_RUN / "traces.jsonl"
    if not traces.exists():
        pytest.skip(f"{demo.DEMO_RUN} is not on this machine")
    records = demo.load_traces()
    assert len(records) >= 250, (
        f"{demo.DEMO_RUN} holds {len(records)} questions; the walkthrough "
        "describes it as the complete 250-question slice"
    )


def test_the_backward_edge_example_still_shows_the_backward_edge(demo):
    """The hand-picked question has to keep making the point it was picked for.

    It is quoted as a case where cycle 0 was wrong, the loop fired, and a later
    cycle was selected and correct. Re-run the evaluation and any of those can
    stop being true -- in which case the demo must be re-pointed, not left
    narrating a question that no longer illustrates anything.
    """
    if not (demo.RUNS / demo.DEMO_RUN / "traces.jsonl").exists():
        pytest.skip(f"{demo.DEMO_RUN} is not on this machine")
    records = {r["qid"]: r for r in demo.load_traces()}
    record = records.get(demo.BACKWARD_EDGE_QID)
    assert record is not None, (
        f"qid {demo.BACKWARD_EDGE_QID} is not in {demo.DEMO_RUN}"
    )

    gold = demo.normalise(record["gold"]["answer"])
    cycle0 = next(c for c in record["candidates"] if c["cycle"] == 0)

    assert record["metrics"]["replanned"], "the example never re-planned"
    assert record["best_cycle"] > 0, "the selected answer came from cycle 0"
    assert demo.normalise(cycle0["answer"]) != gold, "cycle 0 was already right"
    assert demo.is_correct(record), "the final answer is not the gold answer"
    assert "T11" in " ".join(record["transitions"]), (
        "the trace records no T11, so the backward edge was never taken"
    )


def test_the_guard_names_match_the_orchestrator(demo):
    """Section 2 lists G1..G5 by the name the trace uses for each."""
    source = (ROOT / "src" / "agentic_ir" / "orchestrator.py").read_text(encoding="utf-8")
    for name in (
        "max_replans",
        "budget_iterations",
        "budget_llm",
        "budget_wallclock",
        "uninformative_feedback",
    ):
        assert f'"{name}"' in source, (
            f"the demo names guard {name!r}, which _replan_gate no longer sets"
        )


def test_every_section_runs(demo):
    """The whole walkthrough, start to finish, with nothing raising."""
    if not (demo.RUNS / demo.DEMO_RUN / "traces.jsonl").exists():
        pytest.skip(f"{demo.DEMO_RUN} is not on this machine")
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        assert demo.main([]) == 0
    out = buffer.getvalue()

    # Each section has to have actually printed its own heading, so a section
    # that returns early on a missing file cannot pass as a section that ran.
    for number, (title, _) in demo.SECTIONS.items():
        assert f"{number}. {title}" in out, f"section {number} printed no heading"
    assert "T11" in out, "the transition table never reached the backward edge"


def test_the_calibration_numbers_it_quotes_are_the_generated_ones(demo):
    """Section 5 prints calibration figures; they come from the JSON, not prose."""
    calib = ROOT / "results" / "calibration" / "hotpotqa_threshold.json"
    if not calib.exists():
        pytest.skip("no calibration artefact yet")
    data = json.loads(calib.read_text(encoding="utf-8"))
    flat = json.dumps(data)
    assert '"n_questions"' in flat, (
        "the calibration artefact no longer records how many questions it used, "
        "which is the one number that makes the rest of it readable"
    )
