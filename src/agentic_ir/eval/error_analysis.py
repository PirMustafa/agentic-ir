"""Post-hoc failure taxonomy for Chapter 4's qualitative analysis.

Specification: the error-taxonomy table in ``docs/architecture.md`` section 6.
Eight labels, assigned **deterministically, first match wins**, from the trace
record and the gold supporting facts. Nothing here runs the system; every
signal is read back out of ``traces.jsonl``, which is what makes the analysis
reproducible from an archived run and re-runnable after a rule changes.

The label order is not a ranking of severity. It is a cascade from "the system
never had a chance" to "the system had everything it needed and still got it
wrong", so that the first rule that fires is the earliest point in the pipeline
where the failure was already inevitable.

Two things are deliberately reported *outside* the cascade:

* ``verifier_false_reject`` -- a discarded cycle's candidate was right and the
  selected one is not. It sits last in the specified order, where it is nearly
  unreachable: if a discarded candidate was correct then the evidence was
  sufficient, so ``synthesis_error`` fires first and hides it. Since this is
  the count that makes the Verifier->Planner loop falsifiable, it is *also*
  computed as an independent flag on every diagnosis and reported separately.
  The ordered label is unchanged; the loop simply stops being invisible.
* ``verifier_false_accept`` -- same treatment, for symmetry, so the pair can be
  read as the Verifier's confusion matrix rather than as two cascade buckets
  of unequal reachability.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..indexing.corpus import make_doc_id
from ..types import GoldAnswer
from .metrics import normalize_answer, score_question

__all__ = [
    "BUDGET_TERMINATORS",
    "ERROR_LABELS",
    "Diagnosis",
    "ErrorSummary",
    "analyse",
    "bridge_entities",
    "classify",
    "is_correct",
    "summarise",
]

#: The taxonomy, in the order section 6 specifies. First match wins.
ERROR_LABELS: tuple[str, ...] = (
    "parse_failure",
    "budget_exhausted",
    "retrieval_miss",
    "decomposition_error",
    "bridge_link_failure",
    "synthesis_error",
    "verifier_false_accept",
    "verifier_false_reject",
)

#: ``terminated_by`` values that mean a budget stopped the machine.
#: ``budget_iterations`` is in the specification's rule but is not currently
#: emitted by the orchestrator (its iteration cap surfaces as the outer
#: ``machine_stalled`` stop); it is kept so the rule matches the spec verbatim
#: and starts working the moment the orchestrator emits it.
BUDGET_TERMINATORS: frozenset[str] = frozenset(
    {"budget_llm", "budget_wallclock", "budget_iterations"}
)

#: Gold question types that imply more than one hop. HotpotQA uses
#: ``bridge``/``comparison``; 2Wiki adds ``compositional`` and ``inference``.
_MULTIHOP_TYPES = frozenset(
    {"bridge", "comparison", "compositional", "inference", "bridge_comparison"}
)

_WORD_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Diagnosis:
    """One question's verdict, its label, and the evidence for that label.

    ``signals`` carries every intermediate boolean the cascade consulted. A
    label without its signals is an assertion; with them it is a claim a reader
    can check against the same trace.
    """

    qid: str
    dataset: str
    config_name: str
    correct: bool
    em: float
    f1: float
    label: str | None
    detail: str
    false_accept: bool = False
    false_reject: bool = False
    signals: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        """Flat row for ``pandas`` / CSV."""
        return {
            "qid": self.qid,
            "dataset": self.dataset,
            "config_name": self.config_name,
            "correct": self.correct,
            "em": self.em,
            "f1": self.f1,
            "label": self.label or "",
            "detail": self.detail,
            "verifier_false_accept": self.false_accept,
            "verifier_false_reject": self.false_reject,
        }


@dataclass(slots=True)
class ErrorSummary:
    """Corpus-level failure profile for one configuration."""

    n_questions: int = 0
    n_correct: int = 0
    n_errors: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    examples: dict[str, list[str]] = field(default_factory=dict)
    false_accepts: int = 0
    false_rejects: int = 0
    recoverable: int = 0          # a correct answer existed in SOME cycle
    unlabelled: int = 0           # wrong, but no rule fired

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_questions if self.n_questions else 0.0

    def rate(self, label: str) -> float:
        """Share of ALL questions carrying ``label`` -- not share of errors.

        Rates over errors move when accuracy moves, which makes two systems'
        error profiles incomparable. Over questions they do not.
        """
        return self.counts.get(label, 0) / self.n_questions if self.n_questions else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_questions": self.n_questions,
            "n_correct": self.n_correct,
            "n_errors": self.n_errors,
            "accuracy": self.accuracy,
            "counts": dict(self.counts),
            "rates": {label: self.rate(label) for label in ERROR_LABELS},
            "verifier_false_accepts": self.false_accepts,
            "verifier_false_rejects": self.false_rejects,
            "recoverable": self.recoverable,
            "unlabelled": self.unlabelled,
        }


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------

def _gold_of(gold: GoldAnswer | Mapping[str, Any]) -> tuple[str, tuple[tuple[str, int], ...], str | None]:
    """``(answer, supporting_facts, qtype)`` from a dataclass or a trace dict.

    The trace carries whatever object the harness put on ``QuestionState.gold``,
    serialised. A ``GoldAnswer`` becomes a dict with ``qtype``; the section-6
    example shows a hand-built dict with ``type``. Both are accepted, because a
    taxonomy that silently treats every question as single-hop when the key is
    spelled differently would quietly empty two of its own buckets.
    """
    if isinstance(gold, GoldAnswer):
        return gold.answer, tuple(gold.supporting_facts), gold.qtype
    answer = str(gold.get("answer") or "")
    facts = tuple(
        (str(t), int(s)) for t, s in (gold.get("supporting_facts") or ()) if t is not None
    )
    qtype = gold.get("qtype") or gold.get("type")
    return answer, facts, (str(qtype) if qtype else None)


def is_correct(prediction: str, gold_answer: str, *, f1_threshold: float = 1.0) -> tuple[bool, float, float]:
    """``(correct, em, f1)``.

    The default threshold is 1.0, so "correct" means an exact match or a
    perfect token-level F1 and nothing else. A looser default would be a tuning
    knob on every number in the error table, set by the person writing the
    analysis -- which is exactly the kind of quiet choice this project's
    reproducibility rules exist to prevent. It is a parameter so a sensitivity
    check can be run and reported, not so it can be nudged.
    """
    scores = score_question(prediction, gold_answer)
    correct = scores.em >= 1.0 or scores.f1 >= f1_threshold
    return correct, scores.em, scores.f1


def _retrieved_docs(record: Mapping[str, Any]) -> set[str]:
    """Every doc id in any sub-query's retrieved list."""
    docs: set[str] = set()
    for block in (record.get("retrieved") or {}).values():
        for passage in block.get("passages") or []:
            doc_id = passage.get("doc_id")
            if doc_id:
                docs.add(str(doc_id))
    return docs


def _per_subquery_docs(record: Mapping[str, Any], *, top_k: int = 10) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for sq_id, block in (record.get("retrieved") or {}).items():
        out[str(sq_id)] = {
            str(p["doc_id"])
            for p in (block.get("passages") or [])[:top_k]
            if p.get("doc_id")
        }
    return out


def _evidence_facts(record: Mapping[str, Any]) -> set[tuple[str, int]]:
    facts: set[tuple[str, int]] = set()
    for item in record.get("evidence") or []:
        title, sent_id = item.get("title"), item.get("sent_id")
        if title is not None and sent_id is not None:
            facts.add((str(title), int(sent_id)))
    return facts


def bridge_entities(record: Mapping[str, Any]) -> set[str]:
    """Every intermediate entity the system committed to.

    Two sources, because the bridge can be found either way: the KG
    navigator's ``bridge_entity`` per sub-query, and the answer extracted at a
    node the orchestrator marked as a bridge (``steps[*].output_summary`` with
    ``bridge: true``). ``QuestionState.bridge_entities`` itself is not in the
    trace schema, so it is reconstructed from what is.
    """
    found: set[str] = set()
    for block in (record.get("kg") or {}).values():
        entity = block.get("bridge_entity")
        if entity:
            found.add(str(entity))
    for step in record.get("steps") or []:
        summary = step.get("output_summary") or {}
        if summary.get("bridge") and summary.get("answer"):
            found.add(str(summary["answer"]))
    return {e for e in (s.strip() for s in found) if e}


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_WORD_RE.findall(normalize_answer(text)))


def _matches_any_title(entity: str, titles: Iterable[str]) -> bool:
    """Whether ``entity`` names one of ``titles``.

    Normalised equality first, then containment either way: an extracted bridge
    is routinely "Gian Carlo Menotti" against a title of "Gian Carlo Menotti
    (composer)", and calling that a link failure would fabricate errors.
    """
    ent = normalize_answer(entity)
    if not ent:
        return False
    ent_tokens = _tokens(entity)
    for title in titles:
        gold = normalize_answer(title)
        if not gold:
            continue
        if ent == gold or ent in gold or gold in ent:
            return True
        gold_tokens = _tokens(title)
        if ent_tokens and gold_tokens and (ent_tokens <= gold_tokens or gold_tokens <= ent_tokens):
            return True
    return False


def _selected_plan(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The plan of the SELECTED cycle (section 6: metrics describe ``best_cycle``).

    ``trace.py`` summarises ``plan_depth``/``n_subqueries`` from the LAST plan,
    which differs from the selected one whenever a re-plan was discarded at
    FINALIZE. The taxonomy judges the decomposition the answer actually came
    from, so it re-derives it from ``plans`` rather than reading the summary.
    """
    plans = record.get("plans") or []
    if not plans:
        return None
    best_cycle = record.get("best_cycle")
    if best_cycle is not None:
        for plan in plans:
            if plan.get("revision") == best_cycle:
                return plan
    return plans[-1]


# ---------------------------------------------------------------------------
# The cascade
# ---------------------------------------------------------------------------

def classify(
    record: Mapping[str, Any],
    gold: GoldAnswer | Mapping[str, Any] | None = None,
    *,
    corpus_titles: frozenset[str] | set[str] | None = None,
    f1_threshold: float = 1.0,
    top_k: int = 10,
) -> Diagnosis:
    """Diagnose one trace record. Deterministic; first matching rule wins.

    ``corpus_titles`` makes the ``decomposition_error`` rule's "gold
    supporting facts exist in the corpus" clause checkable. Without it that
    clause is assumed true, which is the permissive direction: a gold fact
    genuinely missing from the corpus would be blamed on the decomposition.
    The caller should pass ``frozenset(Corpus.load(dataset).titles())``.
    """
    qid = str(record.get("qid") or "")
    dataset = str(record.get("dataset") or "")
    config_name = str(record.get("config_name") or "")
    gold = gold if gold is not None else (record.get("gold") or {})
    gold_answer, gold_facts, qtype = _gold_of(gold)

    prediction = str(record.get("final_answer") or "")
    correct, em, f1 = is_correct(prediction, gold_answer, f1_threshold=f1_threshold)

    metrics = record.get("metrics") or {}
    terminated_by = str(record.get("terminated_by") or "")
    verdict = record.get("verdict")

    gold_titles = [t for t, _ in gold_facts]
    gold_docs = {make_doc_id(dataset, t) for t in gold_titles} if dataset else set()
    retrieved = _retrieved_docs(record)
    per_sq = _per_subquery_docs(record, top_k=top_k)
    covered_any_sq = {doc for docs in per_sq.values() for doc in docs}

    plan = _selected_plan(record)
    n_subqueries = len(plan.get("subqueries") or ()) if plan else int(metrics.get("n_subqueries", 0) or 0)

    multihop = len(set(gold_titles)) >= 2 or (qtype or "").lower() in _MULTIHOP_TYPES
    two_hop = (qtype or "").lower() in {"bridge", "compositional", "inference"} or (
        qtype is None and len(set(gold_titles)) == 2
    )
    gold_in_corpus = (
        all(t in corpus_titles for t in set(gold_titles)) if corpus_titles is not None else True
    )

    bridges = bridge_entities(record)
    bridge_ok = any(_matches_any_title(entity, gold_titles) for entity in bridges)

    evidence_facts = _evidence_facts(record)
    all_facts_in_evidence = bool(gold_facts) and set(gold_facts) <= evidence_facts

    # -- the two verifier flags, computed independently of the cascade -----
    selected_cycle = record.get("best_cycle")
    discarded_correct: list[int] = []
    for candidate in record.get("candidates") or []:
        cycle = candidate.get("cycle")
        if cycle == selected_cycle:
            continue
        ok, _, _ = is_correct(
            str(candidate.get("answer") or ""), gold_answer, f1_threshold=f1_threshold
        )
        if ok:
            discarded_correct.append(int(cycle) if cycle is not None else -1)
    false_reject = bool(discarded_correct) and not correct
    false_accept = verdict == "accept" and not correct

    signals: dict[str, Any] = {
        "parse_failures": int(metrics.get("parse_failures", 0) or 0),
        "terminated_by": terminated_by,
        "verdict": verdict,
        "n_gold_docs": len(gold_docs),
        "n_gold_docs_retrieved": len(gold_docs & retrieved),
        "n_gold_docs_in_topk": len(gold_docs & covered_any_sq),
        "n_subqueries": n_subqueries,
        "multihop_gold": multihop,
        "two_hop_gold": two_hop,
        "gold_in_corpus": gold_in_corpus,
        "bridge_entities": sorted(bridges),
        "bridge_matches_gold": bridge_ok,
        "all_gold_facts_in_evidence": all_facts_in_evidence,
        "discarded_correct_cycles": discarded_correct,
        "n_candidates": len(record.get("candidates") or ()),
        "harness_error": terminated_by == "harness_error",
    }

    if correct:
        return Diagnosis(
            qid=qid, dataset=dataset, config_name=config_name, correct=True,
            em=em, f1=f1, label=None, detail="correct",
            false_accept=False, false_reject=False, signals=signals,
        )

    label, detail = _first_match(
        signals=signals,
        gold_docs=gold_docs,
        retrieved=retrieved,
        covered_any_sq=covered_any_sq,
        gold_facts=gold_facts,
        false_accept=false_accept,
        false_reject=false_reject,
    )
    return Diagnosis(
        qid=qid, dataset=dataset, config_name=config_name, correct=False,
        em=em, f1=f1, label=label, detail=detail,
        false_accept=false_accept, false_reject=false_reject, signals=signals,
    )


def _first_match(
    *,
    signals: Mapping[str, Any],
    gold_docs: set[str],
    retrieved: set[str],
    covered_any_sq: set[str],
    gold_facts: Sequence[tuple[str, int]],
    false_accept: bool,
    false_reject: bool,
) -> tuple[str | None, str]:
    """The section-6 cascade, in order, over an already-wrong answer."""
    if signals["parse_failures"] > 0:
        return "parse_failure", f"{signals['parse_failures']} unparseable structured call(s)"

    if signals["terminated_by"] in BUDGET_TERMINATORS:
        return "budget_exhausted", f"terminated_by={signals['terminated_by']}"

    if gold_docs and not (gold_docs & retrieved):
        return (
            "retrieval_miss",
            f"none of {len(gold_docs)} gold document(s) appear in any retrieved list",
        )

    # Partial coverage on a multi-hop gold with a decomposition too shallow to
    # reach the second hop. Section 6's clauses read as one rule; "no sub-query's
    # top-10 contains them" is taken as "not ALL of them", since the total miss
    # is already claimed above by retrieval_miss.
    missing_in_topk = gold_docs - covered_any_sq
    if (
        missing_in_topk
        and signals["multihop_gold"]
        and signals["n_subqueries"] < 2
        and signals["gold_in_corpus"]
    ):
        return (
            "decomposition_error",
            f"{len(missing_in_topk)} gold document(s) outside every top-k with a "
            f"{signals['n_subqueries']}-node plan on a multi-hop question",
        )

    if (
        signals["two_hop_gold"]
        and (gold_docs & retrieved)
        and not signals["bridge_matches_gold"]
    ):
        found = ", ".join(signals["bridge_entities"]) or "(none)"
        return "bridge_link_failure", f"hop-1 retrieved, bridge entity {found}"

    if signals["all_gold_facts_in_evidence"]:
        return (
            "synthesis_error",
            f"all {len(gold_facts)} gold supporting fact(s) were in the evidence pool",
        )

    if false_accept:
        return "verifier_false_accept", "verdict=accept on a wrong answer"

    if false_reject:
        cycles = ", ".join(str(c) for c in signals["discarded_correct_cycles"])
        return "verifier_false_reject", f"correct candidate discarded from cycle(s) {cycles}"

    return None, "wrong, but no taxonomy rule fired"


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------

def analyse(
    records: Iterable[Mapping[str, Any]],
    golds: Mapping[str, GoldAnswer] | None = None,
    *,
    corpus_titles: frozenset[str] | set[str] | None = None,
    f1_threshold: float = 1.0,
    top_k: int = 10,
) -> list[Diagnosis]:
    """Classify a whole run, in trace order.

    ``golds`` is optional: the trace carries the gold block. Passing the qrels
    is preferred anyway, because the trace's copy was written by the run under
    analysis and the qrels were not.
    """
    out: list[Diagnosis] = []
    for record in records:
        qid = str(record.get("qid") or "")
        gold = (golds or {}).get(qid) if golds else None
        out.append(
            classify(
                record, gold,
                corpus_titles=corpus_titles,
                f1_threshold=f1_threshold,
                top_k=top_k,
            )
        )
    return out


def summarise(diagnoses: Sequence[Diagnosis], *, n_examples: int = 3) -> ErrorSummary:
    """Aggregate diagnoses into the profile ``tables.py`` renders."""
    summary = ErrorSummary(n_questions=len(diagnoses))
    summary.counts = dict.fromkeys(ERROR_LABELS, 0)
    summary.examples = {label: [] for label in ERROR_LABELS}
    for d in diagnoses:
        if d.correct:
            summary.n_correct += 1
            continue
        summary.n_errors += 1
        if d.label is None:
            summary.unlabelled += 1
        else:
            summary.counts[d.label] = summary.counts.get(d.label, 0) + 1
            if len(summary.examples[d.label]) < n_examples:
                summary.examples[d.label].append(d.qid)
        summary.false_accepts += int(d.false_accept)
        summary.false_rejects += int(d.false_reject)
        summary.recoverable += int(bool(d.signals.get("discarded_correct_cycles")))
    return summary
