"""Tests for the verifier threshold calibration, ``eval/calibrate.py``.

The failure modes worth guarding here are all failures of *honesty* rather than
of arithmetic, because the arithmetic is what produces the honest-looking number:

* a threshold recommended from 50 questions with no interval attached -- which
  reads as evidence and is not;
* an ``accept_em`` of ``0.000`` at a threshold that accepts nothing, which draws
  a curve falling to the floor off an empty set;
* a re-plan column presented as a measurement rather than as the counterfactual
  read of first-cycle confidences that it is;
* a threshold read off a confidence signal that carries no information at all,
  which the Brier skill is there to expose;
* calibrating on a run that quietly used the evaluation 250 -- the leakage the
  whole module exists to prevent.

Every fixture is hand-constructed so the expected answer is computable on paper:
the separable case has a known plateau, and the ECE case has ten points whose
bins, gaps and Brier score are worked out in the test body.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import orjson
import pytest

from agentic_ir.config import load_config
from agentic_ir.eval import calibrate as calib
from agentic_ir.types import GoldAnswer

DATASET = "hotpotqa"


# ---------------------------------------------------------------------------
# Fixtures: traces and points built by hand
# ---------------------------------------------------------------------------

def make_record(
    qid: str,
    confidence: float,
    answer: str,
    *,
    first_confidence: float | None = None,
    sufficient: bool = True,
    verdict: str = "accept",
    replanned: bool = False,
) -> dict[str, Any]:
    """A trace record carrying exactly the fields the sweep reads.

    Shaped after the schema in ``docs/architecture.md`` section 6: a top-level
    ``confidence`` for the selected answer and one ``VerificationResult`` per
    cycle, each wrapping its own ``candidate``.
    """
    first = first_confidence if first_confidence is not None else confidence
    return {
        "schema_version": "1.0",
        "qid": qid,
        "dataset": DATASET,
        "final_answer": answer,
        "confidence": confidence,
        "verdict": verdict,
        "metrics": {"replanned": replanned, "replans": int(replanned)},
        "verifications": [
            {
                "verdict": verdict,
                "confidence": first,
                "candidate": {"answer": answer, "sufficient": sufficient},
            }
        ],
    }


def make_gold(qid: str, answer: str) -> GoldAnswer:
    return GoldAnswer(
        qid=qid, question=f"question {qid}", answer=answer, dataset=DATASET
    )


def points_from(pairs: list[tuple[float, bool]]) -> list[calib.QuestionPoint]:
    """Build points from ``(confidence, correct)`` pairs via the real loader.

    Going through :func:`load_points` rather than constructing ``QuestionPoint``
    directly keeps the tests honest about the trace format: a schema change that
    broke the reader would otherwise pass every one of them.
    """
    records = []
    golds = {}
    for i, (confidence, correct) in enumerate(pairs):
        qid = f"q{i:03d}"
        gold_answer = "right"
        records.append(
            make_record(qid, confidence, gold_answer if correct else "wrong")
        )
        golds[qid] = make_gold(qid, gold_answer)
    return calib.load_points(records, golds)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_load_points_scores_against_the_slice_not_the_trace():
    """The trace's own gold block was written by the run under analysis.

    A run that recorded the wrong gold would otherwise mark its own homework.
    """
    record = make_record("q1", 0.9, "Arthur's Magazine")
    record["gold"] = {"answer": "something else entirely"}
    points = calib.load_points([record], {"q1": make_gold("q1", "Arthur's Magazine")})
    assert len(points) == 1
    assert points[0].em == 1.0


def test_load_points_drops_questions_outside_the_calibration_slice():
    """A qid from another split must not be scored here -- that is the leakage."""
    records = [make_record("in", 0.9, "yes"), make_record("out", 0.9, "yes")]
    points = calib.load_points(records, {"in": make_gold("in", "yes")})
    assert [p.qid for p in points] == ["in"]


def test_load_points_is_sorted_by_qid():
    """The bootstrap indexes into this list; run completion order must not leak in."""
    records = [make_record(q, 0.5, "a") for q in ("q9", "q1", "q5")]
    golds = {q: make_gold(q, "a") for q in ("q9", "q1", "q5")}
    assert [p.qid for p in calib.load_points(records, golds)] == ["q1", "q5", "q9"]


def test_load_points_reads_the_first_cycle_confidence_separately():
    """The threshold gated cycle 0, not the answer that survived selection."""
    record = make_record("q1", 0.90, "a", first_confidence=0.31, replanned=True)
    p = calib.load_points([record], {"q1": make_gold("q1", "a")})[0]
    assert p.confidence == pytest.approx(0.90)
    assert p.first_confidence == pytest.approx(0.31)
    assert p.replanned is True


def test_load_points_flags_a_forced_revise():
    """3.5 step 6 re-plans an insufficient candidate at every threshold."""
    record = make_record("q1", 0.9, "a", sufficient=False)
    assert calib.load_points([record], {"q1": make_gold("q1", "a")})[0].forced_revise


def test_load_points_survives_a_missing_or_broken_confidence():
    """A torn record must not take the whole calibration down with it."""
    record = make_record("q1", 0.5, "a")
    record["confidence"] = None
    record["verifications"] = []
    p = calib.load_points([record], {"q1": make_gold("q1", "a")})[0]
    assert p.confidence == 0.0 and p.first_confidence == 0.0


# ---------------------------------------------------------------------------
# The confusion matrix, worked out by hand
# ---------------------------------------------------------------------------

def test_confusion_matrix_at_a_known_threshold():
    # Six questions. At threshold 0.55: accepted = {0.90 correct, 0.60 correct,
    # 0.70 incorrect} -> tp 2, fp 1; rejected = {0.50 correct, 0.40 incorrect,
    # 0.10 incorrect} -> fn 1, tn 2.
    points = points_from([
        (0.90, True), (0.60, True), (0.70, False),
        (0.50, True), (0.40, False), (0.10, False),
    ])
    row = calib.sweep(points)[15]           # 0.40 + 15*0.01
    assert row.threshold == pytest.approx(0.55)
    assert (row.tp, row.fp, row.fn, row.tn) == (2, 1, 1, 2)
    assert row.precision == pytest.approx(2 / 3)
    assert row.recall == pytest.approx(2 / 3)
    assert row.specificity == pytest.approx(2 / 3)
    assert row.youden_j == pytest.approx(1 / 3)
    assert row.f1 == pytest.approx(2 / 3)
    assert row.decision_accuracy == pytest.approx(4 / 6)


def test_accept_is_inclusive_of_the_threshold():
    """``conf >= threshold`` is the Verifier's own rule (3.5 step 6)."""
    points = points_from([(0.55, True)])
    assert calib.sweep(points)[15].n_accept == 1


def test_sweep_covers_the_documented_grid():
    rows = calib.sweep(points_from([(0.5, True)]))
    assert len(rows) == 36
    assert rows[0].threshold == pytest.approx(0.40)
    assert rows[-1].threshold == pytest.approx(0.75)


def test_accept_accuracy_of_an_empty_set_is_none_not_zero():
    """0.000 would read as "the accepted answers were all wrong"."""
    points = points_from([(0.30, True), (0.35, False)])
    row = calib.sweep(points)[-1]           # 0.75 accepts nothing
    assert row.n_accept == 0
    assert row.accept_em is None and row.accept_f1 is None
    assert row.reject_em is not None


def test_replan_rate_is_monotone_and_counts_forced_revisions_everywhere():
    """A candidate the synthesiser called insufficient re-plans at every bar."""
    records = [
        make_record("q1", 0.9, "a", first_confidence=0.90),
        make_record("q2", 0.5, "a", first_confidence=0.50),
        make_record("q3", 0.9, "a", first_confidence=0.99, sufficient=False),
    ]
    golds = {q: make_gold(q, "a") for q in ("q1", "q2", "q3")}
    rows = calib.sweep(calib.load_points(records, golds))
    rates = [r.replan_rate for r in rows]
    assert rates == sorted(rates), "raising the bar cannot re-plan fewer questions"
    assert rows[0].replan_rate == pytest.approx(1 / 3)   # 0.40: only forced q3
    assert rows[-1].replan_rate == pytest.approx(2 / 3)  # 0.75: q3 and q2, not q1


# ---------------------------------------------------------------------------
# Selection: a case whose optimum is computable on paper
# ---------------------------------------------------------------------------

def test_perfectly_separable_case_selects_the_middle_of_the_plateau():
    # Incorrect answers top out at 0.50, correct ones start at 0.60. Every
    # threshold in 0.51..0.60 separates them perfectly (J = 1), which is ten
    # tied grid points; the middle one, index 5, is 0.56.
    points = points_from([
        (0.10, False), (0.30, False), (0.50, False),
        (0.60, True), (0.80, True), (0.95, True),
    ])
    rows = calib.sweep(points)
    rec = calib.recommend(points, rows, samples=200, seed=42)
    assert rec.youden_j == pytest.approx(1.0)
    assert rec.threshold == pytest.approx(0.56)
    assert 0.51 <= rec.threshold <= 0.60


def test_argmax_plateau_takes_the_middle_not_the_first_index():
    grid = (0.40, 0.41, 0.42, 0.43, 0.44)
    threshold, best = calib._argmax_plateau([0.1, 0.9, 0.9, 0.9, 0.2], grid)
    assert best == pytest.approx(0.9)
    assert threshold == pytest.approx(0.42)


def test_a_separable_case_puts_the_true_boundary_inside_the_interval():
    points = points_from(
        [(0.20 + 0.01 * i, False) for i in range(20)]
        + [(0.65 + 0.01 * i, True) for i in range(20)]
    )
    rec = calib.recommend(points, calib.sweep(points), samples=300, seed=42)
    assert rec.ci_low <= rec.threshold <= rec.ci_high
    assert rec.ci_width < calib.MAX_RECOMMENDATION_CI_WIDTH


def test_youden_and_f1_optima_are_both_reported():
    """A disagreement between the criteria has to be visible, not resolved silently."""
    points = points_from([(0.9, True)] * 8 + [(0.45, True)] * 8 + [(0.5, False)] * 2)
    rec = calib.recommend(points, calib.sweep(points), samples=100, seed=42)
    assert calib.SWEEP[0] <= rec.f1_optimal <= calib.SWEEP[-1]
    assert 0.0 <= rec.f1_optimal_score <= 1.0
    assert any("F1-optimal point" in line for line in rec.reasoning)


# ---------------------------------------------------------------------------
# Calibration quality, worked out by hand
# ---------------------------------------------------------------------------

def test_ece_mce_and_brier_on_a_hand_computed_set():
    # Ten questions in two bins.
    #   bin 8 [0.8, 0.9): five at conf 0.85, four correct -> acc 0.80, gap 0.05
    #   bin 2 [0.2, 0.3): five at conf 0.25, one correct  -> acc 0.20, gap 0.05
    # ECE = 0.5*0.05 + 0.5*0.05 = 0.05; MCE = 0.05.
    # Brier = (4*0.15^2 + 0.85^2 + 0.75^2 + 4*0.25^2) / 10 = 1.625/10 = 0.1625.
    # base rate 0.5 -> baseline 0.25 -> skill 1 - 0.65 = 0.35.
    points = points_from(
        [(0.85, True)] * 4 + [(0.85, False)]
        + [(0.25, True)] + [(0.25, False)] * 4
    )
    q = calib.calibration_quality(points)
    assert q.n == 10
    assert q.ece == pytest.approx(0.05)
    assert q.mce == pytest.approx(0.05)
    assert q.brier == pytest.approx(0.1625)
    assert q.base_rate == pytest.approx(0.5)
    assert q.brier_baseline == pytest.approx(0.25)
    assert q.brier_skill == pytest.approx(0.35)
    assert q.well_calibrated is True
    assert sum(b.n for b in q.bins) == 10
    assert [b.index for b in q.bins if b.n] == [2, 8]


def test_reliability_bins_are_ten_equal_width_and_the_top_one_holds_exactly_one():
    """A confidence of 1.0 belongs in the top bin, not in an eleventh one."""
    q = calib.calibration_quality(points_from([(1.0, True), (0.0, False)]))
    assert len(q.bins) == calib.BINS
    assert q.bins[-1].n == 1 and q.bins[0].n == 1
    assert q.bins[-1].upper == pytest.approx(1.0)


def test_empty_bins_report_none_rather_than_zero_accuracy():
    q = calib.calibration_quality(points_from([(0.85, True)]))
    empty = [b for b in q.bins if b.n == 0]
    assert empty and all(b.accuracy is None and b.mean_confidence is None for b in empty)


def test_an_anticorrelated_signal_is_reported_as_carrying_no_information(cfg):
    """The failure that would make any threshold meaningless has to be named.

    An AUC of 0 -- every wrong answer ranked above every right one -- is the
    only kind of evidence that licenses this claim. A negative Brier skill is
    not: see ``test_an_overconfident_but_discriminative_signal...``.
    """
    points = points_from([(0.9, False)] * 10 + [(0.1, True)] * 10)
    q = calib.calibration_quality(points)
    assert q.auc == pytest.approx(0.0)
    assert q.discriminative is False
    assert q.brier_skill < 0.0
    report = calib.calibrate(points, dataset=DATASET, cfg=cfg, samples=100)
    assert any("no usable information" in n for n in report.notes)


def test_calibration_quality_of_nothing_is_zeroed_but_shaped():
    q = calib.calibration_quality([])
    assert q.n == 0 and len(q.bins) == calib.BINS
    assert all(b.n == 0 for b in q.bins)
    assert q.auc is None and q.brier_skill is None


# ---------------------------------------------------------------------------
# Discrimination: AUC, worked out by hand, and kept apart from calibration
# ---------------------------------------------------------------------------

def test_auc_is_the_hand_counted_pairwise_win_rate():
    # Correct at {0.90, 0.60}; incorrect at {0.80, 0.60, 0.40}. Six pairs:
    #   0.90 beats 0.80, 0.60, 0.40           -> 3.0
    #   0.60 loses to 0.80, ties 0.60, beats 0.40 -> 0 + 0.5 + 1 = 1.5
    # AUC = 4.5 / 6 = 0.75.
    points = points_from([
        (0.90, True), (0.60, True),
        (0.80, False), (0.60, False), (0.40, False),
    ])
    assert calib.auc_score(points) == pytest.approx(0.75)
    assert calib.calibration_quality(points).auc == pytest.approx(0.75)


def test_auc_counts_a_tie_as_half_not_as_a_win():
    """One correct and one incorrect answer at the same confidence is a coin flip."""
    assert calib.auc_score(points_from([(0.5, True), (0.5, False)])) == pytest.approx(0.5)


def test_auc_of_a_perfectly_separable_signal_is_one():
    points = points_from([(0.9, True), (0.8, True), (0.4, False), (0.1, False)])
    assert calib.auc_score(points) == pytest.approx(1.0)


def test_auc_is_none_when_one_outcome_class_is_empty():
    """An AUC over an empty class is undefined; 0.5 would read as a measured coin flip."""
    assert calib.auc_score(points_from([(0.9, True), (0.8, True)])) is None
    assert calib.auc_score(points_from([(0.9, False)])) is None


def test_a_degenerate_base_rate_reports_no_brier_skill_rather_than_zero(cfg):
    """Base rate 1.0 makes the constant predictor perfect and the ratio undefined."""
    points = points_from([(0.9, True), (0.8, True), (0.7, True)])
    q = calib.calibration_quality(points)
    assert q.base_rate == pytest.approx(1.0)
    assert q.brier_baseline == pytest.approx(0.0)
    assert q.brier_skill is None
    assert q.auc is None
    report = calib.calibrate(points, dataset=DATASET, cfg=cfg, samples=50)
    assert any("neither" in n and "Brier skill nor AUC is defined" in n
               for n in report.notes)
    assert any("could not be measured at all" in n for n in report.notes)


def test_an_overconfident_but_discriminative_signal_is_not_called_uninformative(cfg):
    """The regression this module exists not to commit.

    Four correct answers at confidence 0.99 and four wrong ones at 0.95. The
    ordering is perfect -- AUC 1.0, and a threshold anywhere in (0.95, 0.99]
    separates them exactly -- but the *scale* is nonsense: every confidence sits
    in the top bin against a base rate of 0.5, so

        ECE   = |0.5 - 0.97|                                  = 0.470
        Brier = (4*0.01^2 + 4*0.95^2) / 8 = 3.6104 / 8        = 0.4513
        skill = 1 - 0.4513 / 0.25                             = -0.805

    Reading the negative Brier skill as "carries no usable information" -- which
    an earlier version of this module did -- would throw away a threshold that
    demonstrably works. Calibration and discrimination are different failures.
    """
    points = points_from([(0.99, True)] * 4 + [(0.95, False)] * 4)
    q = calib.calibration_quality(points)
    assert q.ece == pytest.approx(0.47)
    assert q.brier == pytest.approx(0.4513)
    assert q.brier_skill == pytest.approx(-0.8052)
    assert q.well_calibrated is False        # the scale is wrong ...
    assert q.auc == pytest.approx(1.0)       # ... and the ordering is perfect
    assert q.discriminative is True

    report = calib.calibrate(points, dataset=DATASET, cfg=cfg, samples=100)
    joined = " ".join(report.notes)
    assert "no usable information" not in joined
    assert "NOT well calibrated" in joined
    assert "AUC = 1.000" in joined
    assert "doing real work" in joined


def test_the_latex_caption_separates_calibration_from_discrimination(tmp_path, cfg):
    points = points_from([(0.99, True)] * 4 + [(0.95, False)] * 4)
    report = calib.calibrate(points, dataset=DATASET, cfg=cfg, samples=100)
    paths = calib.write_artefacts(
        report, cfg=cfg, json_path=tmp_path / "j.json", tex_path=tmp_path / "t.tex"
    )
    text = paths["tex"].read_text(encoding="utf-8")
    assert "Discrimination is a separate question from calibration" in text
    assert "AUC $=$ 1.000" in text
    assert "not} well calibrated" in text
    assert "----" not in text, "the MISSING marker must render as -- , not as ----"


# ---------------------------------------------------------------------------
# Fifty questions is small: the recommendation must say so
# ---------------------------------------------------------------------------

def test_fifty_questions_is_reported_as_under_powered_and_keeps_the_configured_value(cfg):
    points = points_from(
        [(0.40 + 0.01 * i, i % 3 == 0) for i in range(50)]
    )
    report = calib.calibrate(points, dataset=DATASET, cfg=cfg, samples=200)
    rec = report.recommendation
    assert rec is not None
    assert rec.n == 50
    assert rec.sufficient is False
    assert rec.keep_configured is True
    assert any("UNDER-POWERED" in line for line in rec.reasoning)
    assert any("keep confidence_threshold" in line for line in rec.reasoning)
    # "keep" at this n is forced by MIN_RECOMMENDATION_N, so it must not be
    # readable as the data having endorsed the configured value.
    assert any(
        "not a positive" in line and "endorsement" in line for line in rec.reasoning
    )


def test_a_partial_run_is_reported_as_partial_not_as_a_smaller_study(cfg):
    """17 questions reads as a small sample; 17 of 50 reads as an unfinished run."""
    points = points_from([(0.3, False), (0.8, True)] * 6)   # 12 of a 50-question slice
    report = calib.calibrate(points, dataset=DATASET, cfg=cfg, n_slice=50, samples=100)
    assert report.n_questions == 12 and report.n_slice == 50
    assert report.complete is False
    assert report.coverage == pytest.approx(0.24)
    assert report.notes and report.notes[0].startswith("PARTIAL RUN:")
    assert "12 of the 50" in report.notes[0]


def test_a_complete_run_is_not_flagged_as_partial(cfg):
    points = points_from([(0.3, False), (0.8, True)] * 25)
    report = calib.calibrate(points, dataset=DATASET, cfg=cfg, n_slice=50, samples=100)
    assert report.complete is True
    assert report.coverage == pytest.approx(1.0)
    assert not any(n.startswith("PARTIAL RUN") for n in report.notes)


def test_an_unknown_slice_size_claims_neither_completeness_nor_a_coverage(cfg):
    report = calib.calibrate(
        points_from([(0.3, False), (0.8, True)]), dataset=DATASET, cfg=cfg, samples=50
    )
    assert report.n_slice == 0
    assert report.coverage is None and report.complete is False
    assert not any(n.startswith("PARTIAL RUN") for n in report.notes)


def test_a_partial_run_says_so_in_the_json_and_the_caption(tmp_path, cfg):
    """The table travels on its own, so the caveat has to travel in the caption."""
    points = points_from([(0.3, False), (0.8, True)] * 6)
    report = calib.calibrate(points, dataset=DATASET, cfg=cfg, n_slice=50, samples=100)
    paths = calib.write_artefacts(
        report, cfg=cfg, json_path=tmp_path / "p.json", tex_path=tmp_path / "p.tex"
    )
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["n_questions"] == 12
    assert payload["n_slice"] == 50
    assert payload["complete"] is False
    assert payload["coverage"] == pytest.approx(0.24)
    assert "12 of 50 questions scored so far" in paths["tex"].read_text(encoding="utf-8")


def test_run_records_the_slice_size_so_coverage_is_computable(tmp_path, cfg):
    """``run()`` must pass the slice size through, or nothing above can fire."""
    golds = _load_calib_golds(cfg)
    if not golds:
        pytest.skip("calibration slice has not been sampled yet")
    runs = tmp_path / "runs"
    runs.mkdir()
    ordered = sorted({g.qid: g for g in golds})[:5]
    _write_run(
        runs, f"agentic_full_{DATASET}_20260101T000000Z", split="calib",
        records=[make_record(qid, 0.5, "a") for qid in ordered],
    )
    report, _ = calib.run(
        DATASET, cfg=cfg, runs_dir=runs, samples=50,
        json_path=tmp_path / "r.json", tex_path=tmp_path / "r.tex",
    )
    assert report.n_slice == len(golds)
    assert report.n_questions == 5
    assert report.complete is False


def test_the_recommendation_always_carries_an_interval(cfg):
    report = calib.calibrate(
        points_from([(0.3, False), (0.8, True)] * 25), dataset=DATASET,
        cfg=cfg, samples=100,
    )
    rec = report.recommendation
    assert rec is not None
    assert rec.ci_low <= rec.ci_high
    assert rec.ci_width == pytest.approx(rec.ci_high - rec.ci_low)
    assert any("95% interval" in line for line in rec.reasoning)


def test_no_data_yields_no_confident_number(cfg):
    report = calib.calibrate([], dataset=DATASET, cfg=cfg, samples=50)
    assert report.n_questions == 0
    assert any("no recommendation is possible" in n for n in report.notes)
    assert report.recommendation is not None
    assert report.recommendation.keep_configured is True


def test_every_recommendation_states_what_the_sweep_cannot_establish(cfg):
    report = calib.calibrate(
        points_from([(0.3, False), (0.8, True)]), dataset=DATASET, cfg=cfg, samples=50
    )
    joined = " ".join(report.recommendation.reasoning)
    assert "post-hoc" in joined
    assert "does NOT measure" in joined
    assert any("counterfactual" in lim for lim in calib.LIMITATIONS)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_the_whole_report_is_deterministic_under_the_same_seed(cfg):
    points = points_from([(0.2 + 0.013 * i, i % 2 == 0) for i in range(40)])
    a = calib.calibrate(points, dataset=DATASET, cfg=cfg, samples=200, seed=42)
    b = calib.calibrate(points, dataset=DATASET, cfg=cfg, samples=200, seed=42)
    assert json.dumps(a.to_dict()) == json.dumps(b.to_dict())


def test_the_default_seed_is_the_project_seed(cfg):
    report = calib.calibrate(
        points_from([(0.5, True), (0.6, False)]), dataset=DATASET, cfg=cfg, samples=50
    )
    assert report.recommendation.seed == int(cfg.get("project.seed", 42)) == 42


# ---------------------------------------------------------------------------
# Run discovery: the leakage guard
# ---------------------------------------------------------------------------

def _write_run(root: Path, name: str, *, split: str, records: list[dict[str, Any]]):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "meta.json").write_bytes(
        orjson.dumps({"run_id": name, "split": split, "config_name": "agentic_full",
                      "dataset": DATASET})
    )
    with (directory / "traces.jsonl").open("wb") as fh:
        for record in records:
            fh.write(orjson.dumps(record) + b"\n")
    return directory


def test_discovery_ignores_a_run_on_the_evaluation_split(tmp_path, cfg):
    """Calibrating on the evaluation 250 is the leakage risk 4 exists to stop."""
    _write_run(tmp_path, "agentic_full_hotpotqa_20260101T000000Z",
               split="eval", records=[make_record("q1", 0.9, "a")])
    assert calib.discover_calibration_run(DATASET, cfg=cfg, root=tmp_path) is None


def test_discovery_picks_the_newest_calibration_run(tmp_path, cfg):
    _write_run(tmp_path, "agentic_full_hotpotqa_20260101T000000Z",
               split="calib", records=[make_record("q1", 0.9, "a")])
    newest = _write_run(tmp_path, "agentic_full_hotpotqa_20260202T000000Z",
                        split="calib", records=[make_record("q1", 0.9, "a")])
    assert calib.discover_calibration_run(DATASET, cfg=cfg, root=tmp_path) == newest


def test_discovery_skips_an_empty_run_directory(tmp_path, cfg):
    """A run that died before its first question must not shadow a complete one."""
    good = _write_run(tmp_path, "agentic_full_hotpotqa_20260101T000000Z",
                      split="calib", records=[make_record("q1", 0.9, "a")])
    _write_run(tmp_path, "agentic_full_hotpotqa_20260303T000000Z",
               split="calib", records=[])
    assert calib.discover_calibration_run(DATASET, cfg=cfg, root=tmp_path) == good


def test_discovery_skips_a_run_with_no_meta(tmp_path, cfg):
    """An unreadable meta.json cannot vouch for the split it consumed."""
    directory = tmp_path / "agentic_full_hotpotqa_20260101T000000Z"
    directory.mkdir()
    (directory / "traces.jsonl").write_bytes(orjson.dumps(make_record("q1", 0.9, "a")))
    assert calib.discover_calibration_run(DATASET, cfg=cfg, root=tmp_path) is None


# ---------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------

@pytest.fixture()
def written(tmp_path, cfg):
    points = points_from([(0.20 + 0.012 * i, i % 3 != 0) for i in range(50)])
    report = calib.calibrate(
        points, dataset=DATASET, run_id="agentic_full_hotpotqa_TEST",
        config_name="agentic_full", cfg=cfg, samples=200,
    )
    paths = calib.write_artefacts(
        report, cfg=cfg,
        json_path=tmp_path / "hotpotqa_threshold.json",
        tex_path=tmp_path / "calibration.tex",
    )
    return report, paths


def test_json_summary_carries_the_table_the_recommendation_and_the_limits(written):
    _, paths = written
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["dataset"] == DATASET
    assert payload["n_questions"] == 50
    assert len(payload["sweep"]) == 36
    assert payload["sweep"][0]["threshold"] == 0.40
    assert payload["sweep"][-1]["threshold"] == 0.75
    assert payload["recommendation"]["ci_low"] <= payload["recommendation"]["ci_high"]
    assert payload["recommendation"]["reasoning"]
    assert payload["calibration_quality"]["n_bins"] == calib.BINS
    assert len(payload["calibration_quality"]["reliability"]) == calib.BINS
    assert payload["limitations"], "the caveats must ship with the numbers"
    assert any("post-hoc" in lim.lower() for lim in payload["limitations"])
    assert payload["correctness_label"]["kind"] == "em"


def test_json_summary_is_utf8_and_reparses(written):
    _, paths = written
    assert json.loads(paths["json"].read_text(encoding="utf-8"))


def test_latex_fragment_is_a_booktabs_table_with_caption_and_label(written):
    _, paths = written
    text = paths["tex"].read_text(encoding="utf-8")
    assert r"\begin{table}" in text and r"\end{table}" in text
    assert r"\toprule" in text and r"\midrule" in text and r"\bottomrule" in text
    assert r"\caption{" in text
    assert r"\label{tab:calibration}" in text
    assert r"\label{tab:calibration-reliability}" in text


def test_latex_fragment_escapes_every_underscore(written):
    """An unescaped underscore is a hard LaTeX error in a file nobody hand-edits."""
    _, paths = written
    text = paths["tex"].read_text(encoding="utf-8")
    body = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("%")
    )
    stripped = body.replace(r"\_", "")
    assert "_" not in stripped.replace("$^{", "").replace("_{\\mathrm{acc}}", "")


def test_latex_caption_states_the_post_hoc_limitation(written):
    """The qualification has to travel with the table, not sit in a footnote."""
    _, paths = written
    text = paths["tex"].read_text(encoding="utf-8")
    assert "post-hoc" in text
    assert "does not predict the re-plan rate" in text
    assert "counterfactual" in text


def test_latex_marks_the_configured_and_selected_thresholds(written):
    report, paths = written
    text = paths["tex"].read_text(encoding="utf-8")
    assert r"$^{\dagger}$" in text          # configured
    assert r"$^{\ast}$" in text             # Youden-optimal
    assert f"{report.configured:.2f}" in text


def test_latex_is_byte_identical_on_regeneration(written, tmp_path, cfg):
    """No timestamp in the fragment: a diff must always mean a number moved."""
    report, paths = written
    first = paths["tex"].read_bytes()
    again = calib.write_artefacts(
        report, cfg=cfg,
        json_path=tmp_path / "again.json", tex_path=tmp_path / "again.tex",
    )
    assert again["tex"].read_bytes() == first


def test_an_empty_report_still_writes_both_files(tmp_path, cfg):
    """A chapter's \\input has to resolve even before the calibration run lands."""
    report = calib.calibrate([], dataset=DATASET, cfg=cfg, samples=50)
    paths = calib.write_artefacts(
        report, cfg=cfg,
        json_path=tmp_path / "empty.json", tex_path=tmp_path / "empty.tex",
    )
    assert paths["json"].exists() and paths["tex"].exists()
    text = paths["tex"].read_text(encoding="utf-8")
    assert r"\label{tab:calibration}" in text
    assert calib.MISSING in text
    assert "0.000" not in text, "an unmeasured cell must never render as a zero"


# ---------------------------------------------------------------------------
# End to end, through the CLI
# ---------------------------------------------------------------------------

def test_cli_runs_end_to_end_on_a_synthetic_calibration_run(tmp_path, capsys, cfg):
    runs = tmp_path / "runs"
    runs.mkdir()
    golds = {g.qid: g for g in _load_calib_golds(cfg)}
    qids = sorted(golds)[:12]
    if not qids:
        pytest.skip("calibration slice has not been sampled yet")
    records = [
        make_record(
            qid,
            0.30 + 0.05 * i,
            golds[qid].answer if i % 2 else "definitely not the answer",
        )
        for i, qid in enumerate(qids)
    ]
    _write_run(runs, f"agentic_full_{DATASET}_20260101T000000Z",
               split="calib", records=records)

    exit_code = calib.main([
        "--dataset", DATASET, "--runs", str(runs), "--bootstrap", "100",
        "--json", str(tmp_path / "out.json"), "--tex", str(tmp_path / "out.tex"),
    ])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "calibration run" in out and "threshold" in out
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert payload["n_questions"] == len(qids)
    assert payload["split"] == "calib"


def _load_calib_golds(cfg):
    from agentic_ir.indexing.corpus import load_eval_set

    try:
        return load_eval_set(DATASET, split="calib", cfg=cfg)
    except (FileNotFoundError, KeyError, ValueError):
        return ()


def test_run_without_a_calibration_run_says_so_rather_than_inventing_one(tmp_path, cfg):
    report, paths = calib.run(
        DATASET, cfg=cfg, runs_dir=tmp_path, samples=50,
        json_path=tmp_path / "none.json", tex_path=tmp_path / "none.tex",
    )
    assert report.n_questions == 0
    assert any("was found" in note for note in report.notes)
    assert paths["json"].exists() and paths["tex"].exists()
