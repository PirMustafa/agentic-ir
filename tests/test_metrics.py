"""Metric tests with hand-computed expected values.

Every number asserted here was worked out by hand from the definition, not
read back from the implementation. That is the point: a metric bug produces
plausible numbers, so agreeing with itself proves nothing.
"""

from __future__ import annotations

import math

import pytest

from agentic_ir.eval.metrics import (
    answer_f1,
    exact_match,
    joint_scores,
    normalize_answer,
    rankings_to_run,
    retrieval_metrics_per_query,
    score_question,
    summarise_agent_metrics,
    supporting_fact_scores,
)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("The Beatles", "beatles"),
    ("a Book,  the End.", "book end"),
    ("Arthur's Magazine", "arthurs magazine"),
    ("  YES  ", "yes"),
    ("1844-1846", "18441846"),          # punctuation stripped, digits kept
    ("Theodore", "theodore"),           # 'the' inside a word must survive
    ("", ""),
])
def test_normalize_answer(raw, expected):
    assert normalize_answer(raw) == expected


def test_article_removal_is_word_bounded():
    # "Theodore" starts with "the" - a naive replace would yield "odore"
    assert normalize_answer("Theodore Roosevelt") == "theodore roosevelt"


# ---------------------------------------------------------------------------
# Exact match / F1
# ---------------------------------------------------------------------------

def test_exact_match_ignores_articles_and_case():
    assert exact_match("The Beatles", "beatles") == 1.0
    assert exact_match("Beatles", "The Rolling Stones") == 0.0


def test_answer_f1_partial_overlap_hand_computed():
    # pred tokens: {arthurs, magazine}          -> 2 tokens
    # gold tokens: {arthurs, magazine, monthly} -> 3 tokens
    # common = 2  ->  P = 2/2 = 1.0,  R = 2/3
    # F1 = 2 * 1.0 * (2/3) / (1.0 + 2/3) = (4/3) / (5/3) = 0.8
    f1, p, r = answer_f1("Arthur's Magazine", "Arthur's Magazine Monthly")
    assert p == pytest.approx(1.0)
    assert r == pytest.approx(2 / 3)
    assert f1 == pytest.approx(0.8)


def test_answer_f1_no_overlap_is_zero():
    assert answer_f1("Paris", "London") == (0.0, 0.0, 0.0)


def test_answer_f1_empty_prediction_is_zero():
    assert answer_f1("", "London") == (0.0, 0.0, 0.0)


def test_yesno_mismatch_scores_zero_not_partial():
    """The official short-circuit. Without it, bag-of-words F1 can award
    partial credit for answering 'no' where the gold answer is 'yes'."""
    assert answer_f1("no", "yes") == (0.0, 0.0, 0.0)
    assert answer_f1("yes", "yes")[0] == pytest.approx(1.0)
    # a yes/no gold against a free-text prediction also scores zero
    assert answer_f1("yes it was", "yes") == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Supporting facts
# ---------------------------------------------------------------------------

def test_supporting_facts_perfect():
    gold = [("A", 0), ("B", 1)]
    em, p, r, f1 = supporting_fact_scores(gold, gold)
    assert (em, p, r, f1) == (1.0, 1.0, 1.0, 1.0)


def test_supporting_facts_hand_computed():
    # pred = {(A,0), (A,1), (B,0)}   gold = {(A,0), (B,0), (C,0)}
    # tp = 2 ((A,0),(B,0)),  fp = 1 ((A,1)),  fn = 1 ((C,0))
    # P = 2/3, R = 2/3, F1 = 2/3, EM = 0 (fp and fn both non-zero)
    pred = [("A", 0), ("A", 1), ("B", 0)]
    gold = [("A", 0), ("B", 0), ("C", 0)]
    em, p, r, f1 = supporting_fact_scores(pred, gold)
    assert em == 0.0
    assert p == pytest.approx(2 / 3)
    assert r == pytest.approx(2 / 3)
    assert f1 == pytest.approx(2 / 3)


def test_supporting_facts_em_requires_exact_set():
    # a superset is not exact match, even though recall is perfect
    em, p, r, _ = supporting_fact_scores([("A", 0), ("B", 0)], [("A", 0)])
    assert em == 0.0
    assert r == pytest.approx(1.0)
    assert p == pytest.approx(0.5)


def test_supporting_facts_empty_prediction():
    em, p, r, f1 = supporting_fact_scores([], [("A", 0)])
    assert (em, p, r, f1) == (0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Joint
# ---------------------------------------------------------------------------

def test_joint_hand_computed():
    # ans P=1.0 R=0.5 ; sp P=0.5 R=1.0
    # joint_P = 1.0*0.5 = 0.5 ; joint_R = 0.5*1.0 = 0.5
    # joint_F1 = 2*0.5*0.5/(0.5+0.5) = 0.5
    j_em, j_f1 = joint_scores(1.0, 1.0, 0.5, 0.0, 0.5, 1.0)
    assert j_em == 0.0          # ans_em * sp_em = 1.0 * 0.0
    assert j_f1 == pytest.approx(0.5)


def test_score_question_end_to_end():
    s = score_question(
        prediction="Arthur's Magazine",
        gold_answer="Arthur's Magazine",
        predicted_facts=[("Arthur's Magazine", 0)],
        gold_facts=[("Arthur's Magazine", 0)],
    )
    assert s.em == 1.0 and s.f1 == pytest.approx(1.0)
    assert s.sp_em == 1.0 and s.sp_f1 == pytest.approx(1.0)
    assert s.joint_em == 1.0 and s.joint_f1 == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Retrieval metrics -- hand-computed against the trec_eval definitions
# ---------------------------------------------------------------------------

def test_rankings_to_run_preserves_order_strictly():
    run = rankings_to_run({"q1": ["a", "b", "c"]})
    assert run["q1"]["a"] > run["q1"]["b"] > run["q1"]["c"]


def test_retrieval_metrics_hand_computed():
    pytest.importorskip("pytrec_eval")
    # q1: one relevant doc at rank 1 (0-indexed 0)
    #     recall@2 = 1/1 = 1.0 ; MRR = 1/1 = 1.0
    #     nDCG@10: DCG = 1/log2(1+1) = 1.0 ; IDCG = 1.0 -> 1.0
    # q2: one relevant doc at rank 3
    #     recall@2 = 0.0 ; recall@5 = 1.0 ; MRR = 1/3
    #     nDCG@10: DCG = 1/log2(3+1) = 0.5 ; IDCG = 1.0 -> 0.5
    rankings = {
        "q1": ["d1", "d2", "d3", "d4"],
        "q2": ["d9", "d8", "d7", "d6"],
    }
    qrels = {"q1": ["d1"], "q2": ["d7"]}
    per_q = retrieval_metrics_per_query(rankings, qrels, k_values=(2, 5, 10))

    assert per_q["q1"]["recall@2"] == pytest.approx(1.0)
    assert per_q["q1"]["mrr"] == pytest.approx(1.0)
    assert per_q["q1"]["ndcg@10"] == pytest.approx(1.0)

    assert per_q["q2"]["recall@2"] == pytest.approx(0.0)
    assert per_q["q2"]["recall@5"] == pytest.approx(1.0)
    assert per_q["q2"]["mrr"] == pytest.approx(1 / 3)
    assert per_q["q2"]["ndcg@10"] == pytest.approx(1 / math.log2(4))  # == 0.5


def test_unjudged_questions_are_excluded_not_zeroed():
    """A question with no qrels must not silently count as a zero."""
    pytest.importorskip("pytrec_eval")
    per_q = retrieval_metrics_per_query(
        {"q1": ["d1"], "q_unjudged": ["d5"]}, {"q1": ["d1"]}
    )
    assert set(per_q) == {"q1"}


# ---------------------------------------------------------------------------
# Agent metrics
# ---------------------------------------------------------------------------

def test_replan_rate_is_fraction_of_questions_not_mean_count():
    records = [
        {"replans": 0, "llm_calls": 4},
        {"replans": 2, "llm_calls": 12},
        {"replans": 0, "llm_calls": 5},
        {"replans": 1, "llm_calls": 9},
    ]
    m = summarise_agent_metrics(records)
    assert m.replan_rate == pytest.approx(0.5)      # 2 of 4 questions re-planned
    assert m.replans == pytest.approx(0.75)         # mean count = 3/4
    assert m.llm_calls == pytest.approx(7.5)


def test_citation_grounding_excludes_unanswered():
    records = [
        {"citation_grounding": 1.0, "answered": True},
        {"citation_grounding": 0.5, "answered": True},
        {"citation_grounding": 0.0, "answered": False},   # must not drag the mean
    ]
    m = summarise_agent_metrics(records)
    assert m.citation_grounding == pytest.approx(0.75)


def test_empty_records_do_not_divide_by_zero():
    m = summarise_agent_metrics([])
    assert m.n_questions == 0 and m.llm_calls == 0.0
