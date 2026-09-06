"""The confidence diagnostic has to be right about direction, not only value.

Its whole content is a claim about whether a signal separates two classes and
whether a trigger picks the right questions. Both are quantities whose *sign*
carries the meaning, and a sign error would read as a finding rather than as a
bug -- so the tests here fix the direction with constructed data before they
check anything measured.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_ir.eval.confidence import (
    auc,
    diagnose,
    expected_calibration_error,
    load_pairs,
    render_table,
    threshold_gap,
)

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"


def test_auc_is_one_when_confidence_orders_perfectly():
    pairs = [(0.9, 1.0), (0.8, 1.0), (0.2, 0.0), (0.1, 0.0)]
    assert auc(pairs) == 1.0


def test_auc_is_zero_when_confidence_is_exactly_backwards():
    pairs = [(0.1, 1.0), (0.2, 1.0), (0.8, 0.0), (0.9, 0.0)]
    assert auc(pairs) == 0.0


def test_auc_is_a_half_when_every_score_ties():
    """Ties are half a win each; dropping them would report perfect ranking."""
    pairs = [(0.5, 1.0), (0.5, 0.0), (0.5, 1.0), (0.5, 0.0)]
    assert auc(pairs) == 0.5


def test_auc_is_none_rather_than_zero_when_a_class_is_empty():
    """A run where everything was correct has no AUC. It does not have AUC 0."""
    assert auc([(0.9, 1.0), (0.8, 1.0)]) is None


def test_ece_is_zero_for_a_perfectly_calibrated_signal():
    pairs = [(0.0, 0.0)] * 10 + [(1.0, 1.0)] * 10
    assert expected_calibration_error(pairs) == pytest.approx(0.0, abs=1e-9)


def test_ece_grows_with_overconfidence():
    honest = [(0.5, 1.0)] * 5 + [(0.5, 0.0)] * 5
    boastful = [(0.95, 1.0)] * 5 + [(0.95, 0.0)] * 5
    assert expected_calibration_error(boastful) > expected_calibration_error(honest)


def test_the_trigger_gap_is_negative_when_the_trigger_is_useful():
    """The sign is the finding.

    Below the threshold is what gets sent back. A trigger that works sends back
    the wrong answers, so accuracy below must be LOWER than accuracy above and
    the gap must be negative. The generated caption says exactly this, and a
    sign flip here would invert the report's conclusion while still producing a
    plausible-looking number.
    """
    useful = [(0.2, 0.0), (0.3, 0.0), (0.8, 1.0), (0.9, 1.0)]
    assert threshold_gap(useful, 0.55) == pytest.approx(-1.0)

    backwards = [(0.2, 1.0), (0.3, 1.0), (0.8, 0.0), (0.9, 0.0)]
    assert threshold_gap(backwards, 0.55) == pytest.approx(+1.0)

    blind = [(0.2, 1.0), (0.3, 0.0), (0.8, 1.0), (0.9, 0.0)]
    assert threshold_gap(blind, 0.55) == pytest.approx(0.0)


def test_the_caption_states_the_useful_direction():
    """Prose and arithmetic must agree about which sign is good."""
    rows = [
        diagnose(d, "hotpotqa", resamples=50)
        for d in [RUNS / "agentic_full_hotpotqa_20260904T221533Z"]
        if d.exists()
    ]
    if not rows or rows[0] is None:
        pytest.skip("no run to render")
    body = render_table([r for r in rows if r is not None])
    assert "negative" in body, "the caption no longer names the useful direction"
    assert "diagnostic, not a calibration" in body, (
        "the caption must keep saying it selects no parameter, or a reader "
        "will take an evaluation-slice number for a tuned threshold"
    )


def _a_run() -> Path | None:
    for name in sorted(RUNS.glob("agentic_full_*")):
        if (name / "traces.jsonl").exists() and (name / "scores.csv").exists():
            return name
    return None


def test_pairs_come_from_both_the_trace_and_the_scores():
    run = _a_run()
    if run is None:
        pytest.skip("no scored run on this machine")
    pairs = load_pairs(run)
    assert pairs, "no question carried both a confidence and a score"
    for confidence, correct in pairs:
        assert 0.0 <= confidence <= 1.0
        assert correct in (0.0, 1.0), "EM must be binary; F1 would blur the question"


def test_the_measured_result_is_a_null_and_is_labelled_one():
    """The finding this module exists to report, asserted as a finding.

    If a future change makes the confidence signal actually discriminate, this
    test fails and the prose in ch5 that calls it a null has to be rewritten --
    which is the point. A silent flip from null to positive would leave the
    report claiming the opposite of its own table.
    """
    run = RUNS / "agentic_full_twowiki_20260905T184526Z"
    if not (run / "scores.csv").exists():
        pytest.skip("the 2WikiMultihopQA full run is not on this machine")
    row = diagnose(run, "twowiki", resamples=400)
    assert row is not None
    assert not row.discriminates, (
        "the 2WikiMultihopQA confidence signal now beats chance; ch5 says it "
        "does not, and explains the verifier ablation's null with it"
    )
    assert row.notes, "a null with no interval is not a finding"
