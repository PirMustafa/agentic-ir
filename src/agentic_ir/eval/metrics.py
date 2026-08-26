"""Evaluation metrics.

Three families, per ``docs/architecture.md`` section 6: retrieval quality,
answer quality, and agent-specific cost.

Two decisions here are about comparability rather than taste:

* Answer and supporting-fact metrics reproduce the **official HotpotQA
  evaluation script** exactly, including its yes/no short-circuit. A subtly
  different normaliser still produces plausible numbers, and those numbers
  would not be comparable to any published result.
* Retrieval metrics go through ``pytrec_eval``, which wraps the same C++ core
  as ``trec_eval`` -- the tool that produced the BEIR numbers we cite. A
  hand-rolled nDCG would quietly break that comparison.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

# ---------------------------------------------------------------------------
# Answer normalisation -- official HotpotQA / SQuAD definition
# ---------------------------------------------------------------------------

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = set(string.punctuation)
_YESNO = {"yes", "no", "noanswer"}


def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation, strip articles, collapse whitespace.

    Order matters and matches the reference implementation: lower, then
    de-punctuate, then de-article, then fix whitespace. Removing articles
    before punctuation would leave "the-book" intact.
    """
    text = s.lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def answer_f1(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    """Token-level F1, precision, recall.

    The yes/no short-circuit is part of the official script and easy to omit:
    a bag-of-words F1 would award partial credit for answering "no" to a "yes"
    question whenever the tokens happen to overlap. These are categorical
    answers, so a mismatch scores zero.
    """
    pred = normalize_answer(prediction)
    gold = normalize_answer(ground_truth)
    zero = (0.0, 0.0, 0.0)

    if pred in _YESNO and pred != gold:
        return zero
    if gold in _YESNO and pred != gold:
        return zero

    pred_tokens = pred.split()
    gold_tokens = gold.split()
    if not pred_tokens or not gold_tokens:
        return zero

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return zero

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


# ---------------------------------------------------------------------------
# Supporting facts -- set metrics over (title, sent_id) pairs
# ---------------------------------------------------------------------------

SupportingFact = tuple[str, int]


def supporting_fact_scores(
    predicted: Iterable[SupportingFact],
    gold: Iterable[SupportingFact],
) -> tuple[float, float, float, float]:
    """Return ``(em, precision, recall, f1)`` over supporting-fact pairs.

    EM is 1 only when the predicted set equals the gold set exactly -- no
    false positives and no false negatives.
    """
    pred_set = {(str(t), int(i)) for t, i in predicted}
    gold_set = {(str(t), int(i)) for t, i in gold}

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    em = 1.0 if fp == 0 and fn == 0 else 0.0
    return em, precision, recall, f1


def joint_scores(
    ans_em: float, ans_prec: float, ans_recall: float,
    sp_em: float, sp_prec: float, sp_recall: float,
) -> tuple[float, float]:
    """Joint EM and F1, per the HotpotQA paper's product formulation."""
    joint_prec = ans_prec * sp_prec
    joint_recall = ans_recall * sp_recall
    if joint_prec + joint_recall > 0:
        joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
    else:
        joint_f1 = 0.0
    return ans_em * sp_em, joint_f1


@dataclass(frozen=True, slots=True)
class AnswerScores:
    em: float
    f1: float
    precision: float
    recall: float
    sp_em: float
    sp_f1: float
    sp_precision: float
    sp_recall: float
    joint_em: float
    joint_f1: float


def score_question(
    prediction: str,
    gold_answer: str,
    predicted_facts: Iterable[SupportingFact] = (),
    gold_facts: Iterable[SupportingFact] = (),
) -> AnswerScores:
    """Full answer + supporting-fact scoring for one question."""
    em = exact_match(prediction, gold_answer)
    f1, prec, rec = answer_f1(prediction, gold_answer)
    sp_em, sp_prec, sp_rec, sp_f1 = supporting_fact_scores(predicted_facts, gold_facts)
    j_em, j_f1 = joint_scores(em, prec, rec, sp_em, sp_prec, sp_rec)
    return AnswerScores(
        em=em, f1=f1, precision=prec, recall=rec,
        sp_em=sp_em, sp_f1=sp_f1, sp_precision=sp_prec, sp_recall=sp_rec,
        joint_em=j_em, joint_f1=j_f1,
    )


# ---------------------------------------------------------------------------
# Retrieval metrics -- pytrec_eval adapter
# ---------------------------------------------------------------------------

DEFAULT_K_VALUES: tuple[int, ...] = (2, 5, 10)


def rankings_to_run(
    rankings: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, float]]:
    """Convert ``{qid: [doc_id, ...]}`` into pytrec_eval's run format.

    Scores are synthesised as descending rank positions. trec_eval sorts by
    score, so the only thing that must survive is the ORDER; using
    ``len - rank`` keeps ties impossible and the order exact.
    """
    run: dict[str, dict[str, float]] = {}
    for qid, docs in rankings.items():
        n = len(docs)
        run[qid] = {doc_id: float(n - i) for i, doc_id in enumerate(docs)}
    return run


def qrels_to_dict(
    qrels: Mapping[str, Iterable[str]],
) -> dict[str, dict[str, int]]:
    """Convert ``{qid: [relevant_doc_id, ...]}`` into pytrec_eval qrels."""
    return {qid: {doc_id: 1 for doc_id in docs} for qid, docs in qrels.items()}


def retrieval_metrics(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Iterable[str]],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    ndcg_at: int = 10,
) -> dict[str, float]:
    """Recall@k, nDCG@k and MRR via pytrec_eval.

    Returns corpus-level means. Per-question values are available from
    :func:`retrieval_metrics_per_query` when bootstrapping.
    """
    per_q = retrieval_metrics_per_query(rankings, qrels, k_values, ndcg_at)
    if not per_q:
        return {}
    keys = next(iter(per_q.values())).keys()
    return {k: sum(q[k] for q in per_q.values()) / len(per_q) for k in keys}


def retrieval_metrics_per_query(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Iterable[str]],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    ndcg_at: int = 10,
) -> dict[str, dict[str, float]]:
    """Per-question retrieval metrics, keyed by qid.

    Bootstrap resampling needs per-question values, so this is the primitive
    and the corpus-level function is the aggregate over it.
    """
    import pytrec_eval  # lazy: heavy C++ extension

    # Only score questions that have judgments; an unjudged question would
    # otherwise silently count as a zero and drag every mean down.
    common = [q for q in rankings if q in qrels]
    if not common:
        return {}

    run = rankings_to_run({q: rankings[q] for q in common})
    qrel = qrels_to_dict({q: qrels[q] for q in common})

    recall_arg = "recall." + ",".join(str(k) for k in k_values)
    ndcg_arg = f"ndcg_cut.{ndcg_at}"
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {recall_arg, ndcg_arg, "recip_rank"})
    raw = evaluator.evaluate(run)

    out: dict[str, dict[str, float]] = {}
    for qid, scores in raw.items():
        row = {f"recall@{k}": scores.get(f"recall_{k}", 0.0) for k in k_values}
        row[f"ndcg@{ndcg_at}"] = scores.get(f"ndcg_cut_{ndcg_at}", 0.0)
        row["mrr"] = scores.get("recip_rank", 0.0)
        out[qid] = row
    return out


# ---------------------------------------------------------------------------
# Agent-specific metrics
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AgentMetrics:
    """Corpus-level agent cost, aggregated across questions.

    Reported beside quality, never instead of it: a system that gains three F1
    points for twenty times the compute has not obviously won.
    """

    n_questions: int = 0
    llm_calls: float = 0.0
    llm_calls_saved: float = 0.0
    llm_cache_hits: float = 0.0
    tool_calls: float = 0.0
    parse_failures: float = 0.0
    latency_s: float = 0.0
    plan_depth: float = 0.0
    n_subqueries: float = 0.0
    cycles: float = 0.0
    replans: float = 0.0
    replan_rate: float = 0.0
    citation_grounding: float = 0.0
    degraded_steps: float = 0.0
    budget_exhausted_rate: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        d = {
            "n_questions": self.n_questions,
            "llm_calls": self.llm_calls,
            "llm_calls_saved": self.llm_calls_saved,
            "llm_cache_hits": self.llm_cache_hits,
            "tool_calls": self.tool_calls,
            "parse_failures": self.parse_failures,
            "latency_s": self.latency_s,
            "plan_depth": self.plan_depth,
            "n_subqueries": self.n_subqueries,
            "cycles": self.cycles,
            "replans": self.replans,
            "replan_rate": self.replan_rate,
            "citation_grounding": self.citation_grounding,
            "degraded_steps": self.degraded_steps,
            "budget_exhausted_rate": self.budget_exhausted_rate,
        }
        d.update(self.extra)
        return d


_MEAN_FIELDS = (
    "llm_calls", "llm_calls_saved", "llm_cache_hits", "tool_calls",
    "parse_failures", "latency_s", "plan_depth", "n_subqueries",
    "cycles", "replans", "degraded_steps",
)


def summarise_agent_metrics(records: Sequence[Mapping[str, Any]]) -> AgentMetrics:
    """Aggregate the ``metrics`` block of per-question trace records.

    Two definitions worth stating, because they are easy to conflate:

    * ``replan_rate`` is the FRACTION OF QUESTIONS that triggered at least one
      re-plan -- not the mean number of re-plans. Both are reported;
      ``replans`` carries the mean count.
    * ``citation_grounding`` is averaged only over questions that produced a
      non-empty answer. Including empty answers as zeros would conflate "cited
      badly" with "did not answer", which are different failures.
    """
    n = len(records)
    m = AgentMetrics(n_questions=n)
    if n == 0:
        return m

    for fld in _MEAN_FIELDS:
        setattr(m, fld, sum(float(r.get(fld, 0) or 0) for r in records) / n)

    m.replan_rate = sum(1 for r in records if float(r.get("replans", 0) or 0) > 0) / n
    m.budget_exhausted_rate = sum(1 for r in records if r.get("budget_exhausted")) / n

    grounded = [
        float(r["citation_grounding"])
        for r in records
        if r.get("citation_grounding") is not None and r.get("answered", True)
    ]
    m.citation_grounding = sum(grounded) / len(grounded) if grounded else 0.0
    return m


__all__ = [
    "normalize_answer", "exact_match", "answer_f1",
    "supporting_fact_scores", "joint_scores", "score_question", "AnswerScores",
    "rankings_to_run", "qrels_to_dict",
    "retrieval_metrics", "retrieval_metrics_per_query", "DEFAULT_K_VALUES",
    "AgentMetrics", "summarise_agent_metrics",
]
