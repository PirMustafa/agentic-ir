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
# Trace-derived recomputation of the per-question ``metrics`` block
# ---------------------------------------------------------------------------
#
# Three fields the orchestrator writes into ``metrics`` are re-derived here
# from the step-level trace rather than trusted, because the value the agent
# wrote does not match the definition the report gives for it. Doing it at
# table time keeps the completed runs comparable: nothing an agent does
# changes, only what is counted.
#
# ``llm_calls_saved``. ``Budget.note_saved`` is called by four agents, and two
# of the four credit a call that the configuration in question cannot make.
# The definition enforced here: a call is *saved* only when a deterministic
# rule produced an output that, absent the rule, THIS configuration would
# have obtained by an LLM call. Whether that LLM path is reachable is read
# from the run's own config snapshot (``meta.json``), so the same trace can
# be recounted under a different configuration without touching the agents.
#
# ``plan_depth``. ``QuestionState.metrics`` reports the depth of the *latest*
# plan; ``docs/architecture.md`` section 6 defines the metric on the
# *selected* plan (``best_cycle``). They diverge whenever a re-plan was
# executed and the first cycle's answer won anyway.
#
# ``cycles`` / ``replans``. ``cycles`` is the number of plans that executed
# and ``replans`` the number of re-plan directives the gate issued; a re-plan
# the planner produced and T2b discarded as a near-duplicate is counted by
# the second and not the first. Both are kept, and the discarded count is
# made explicit so a reader can see which one a rate is built on.

#: Per-question keys of the saved-call decomposition. The sum of the *counted*
#: ones is ``llm_calls_saved``; every one of them is reported so a reader who
#: prefers a different definition can apply it from the same table.
SAVED_SOURCES: tuple[str, ...] = (
    "saved_routing",            # retriever: heuristic table / planner hint
    "saved_kg_alias",           # KG navigator: alias match found seeds
    "saved_verifier_gate",      # verifier: confidence outside the uncertainty band
    "saved_verifier_pointless", # verifier: skip for a reason that is not a gate
    "saved_extraction",         # extractor: rung 1-3 answered before rung 4 (LLM)
    "saved_planner_template",   # planner: comparison template instead of a decompose call
    "saved_planner_ablation",   # NoPlanner (agentic_no_planner): identity plan
)


def _snapshot_get(config: Any, dotted: str, default: Any = None) -> Any:
    """Dotted lookup that works on a ``Config`` and on the ``meta.json`` snapshot."""
    if config is None:
        return default
    getter = getattr(config, "get", None)
    if getter is not None and not isinstance(config, Mapping):
        try:
            return getter(dotted, default)
        except Exception:  # noqa: BLE001 - a snapshot lookup must not fail a table
            return default
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def reachable_llm_paths(config: Any, *, config_name: str | None = None) -> dict[str, bool]:
    """Which of the deterministic rules gate an LLM call this configuration can make.

    * ``router``: ``RetrievalAgent._select`` only renders the routing prompt when
      ``agents.retriever.heuristic_shortcut`` is false. With the shortcut on
      the LLM router is unreachable for every sub-query, so a routing rule
      replaces nothing.
    * ``kg_linker``: ``KGNavigator._link`` tries the alias table first and
      only falls through to ``_link_with_llm`` when ``entity_linker == "llm"``
      or ``llm_link_on_empty`` is set. Under ``alias_match`` alone, the LLM
      linker is a call that does not exist.
    * ``verifier_llm``: the adjudication call exists when the verifier is
      enabled and its method is not ``nli`` (``method_nli_only`` is a skip
      reason, not a gate).
    * ``planner_llm``: the decompose call exists unless the planner has been
      replaced by the ``agentic_no_planner`` identity floor, in which case the
      configuration has no LLM planner to save a call against.
    * ``extractor_llm``: the rung-4 extraction call is unconditional in every
      agentic configuration -- ``Orchestrator._extract_answer`` calls the
      extractor agent whenever rungs 1-3 produced nothing.
    """
    shortcut = bool(_snapshot_get(config, "agents.retriever.heuristic_shortcut", True))
    linker = str(_snapshot_get(config, "agents.kg.entity_linker", "alias_match"))
    link_on_empty = bool(_snapshot_get(config, "agents.kg.llm_link_on_empty", False))
    verifier_on = bool(_snapshot_get(config, "agents.verifier.enabled", True))
    method = str(_snapshot_get(config, "agents.verifier.method", "nli_plus_llm"))
    return {
        "router": not shortcut,
        "kg_linker": linker == "llm" or link_on_empty,
        "verifier_llm": verifier_on and method != "nli",
        "planner_llm": config_name != "agentic_no_planner",
        "extractor_llm": True,
    }


def decompose_saved_calls(record: Mapping[str, Any]) -> dict[str, int]:
    """Every ``note_saved`` credit in one trace record, attributed to its rule.

    Read off the per-step ``output_summary`` / ``input_summary`` fields the
    agents already write, which is what makes the recount possible without a
    rerun: the retriever records ``selector``, the navigator ``linked_by``,
    the verifier ``adjudication_skipped``, the planner ``origin`` and
    ``ablation``, and the extractor its ``rung``. The extractor never calls
    ``note_saved`` at all, so its rungs are the one source that was
    *under*-counted rather than over-counted.
    """
    out = dict.fromkeys(SAVED_SOURCES, 0)
    for step in record.get("steps") or ():
        agent = step.get("agent")
        output = step.get("output_summary") or {}
        inputs = step.get("input_summary") or {}
        if agent == "retriever":
            # Mirrors ``_select``: a save is noted on every path that does not
            # render the routing prompt, which the trace records as a selector
            # other than the LLM's own two outcomes.
            if output.get("selector") not in ("llm", "fallback", None):
                out["saved_routing"] += 1
        elif agent == "kg":
            if output.get("linked_by") == "alias_match":
                out["saved_kg_alias"] += 1
        elif agent == "verifier":
            skipped = inputs.get("adjudication_skipped")
            if skipped == "outside_uncertainty_band":
                out["saved_verifier_gate"] += 1
            elif skipped is not None and skipped != "budget_exhausted":
                # empty_answer, synthesizer_insufficient, method_nli_only: the
                # agent credits these too, and none of them is a call the
                # rule replaced -- a call that would have bought nothing.
                out["saved_verifier_pointless"] += 1
        elif agent == "extractor":
            rung = output.get("rung")
            if rung in (1, 2, 3):
                out["saved_extraction"] += 1
        elif agent == "planner":
            if output.get("ablation") == "no_planner":
                out["saved_planner_ablation"] += 1
            elif output.get("origin") == "template_shortcut":
                out["saved_planner_template"] += 1
    return out


def count_saved_calls(
    decomposition: Mapping[str, int], reachable: Mapping[str, bool]
) -> int:
    """The honest ``llm_calls_saved``: only rules that gate a reachable LLM call.

    ``saved_verifier_pointless`` is never counted: an adjudication of an empty
    or already-insufficient answer is not a call the rule replaced, it is a
    call nobody would make. ``saved_planner_ablation`` is never counted
    either: an ablation that removes the planner has no LLM planner to save
    a call against, and the saving it is entitled to is already visible in
    its lower ``llm_calls``.
    """
    total = 0
    if reachable.get("router"):
        total += int(decomposition.get("saved_routing", 0))
    if reachable.get("kg_linker"):
        total += int(decomposition.get("saved_kg_alias", 0))
    if reachable.get("verifier_llm"):
        total += int(decomposition.get("saved_verifier_gate", 0))
    if reachable.get("extractor_llm", True):
        total += int(decomposition.get("saved_extraction", 0))
    if reachable.get("planner_llm", True):
        total += int(decomposition.get("saved_planner_template", 0))
    return total


def selected_plan_depth(record: Mapping[str, Any]) -> int | None:
    """Depth of the plan whose cycle produced the selected answer.

    ``plans`` in the trace are the executed plans in revision order and
    ``best_cycle`` is the cycle FINALIZE picked (argmax confidence over all
    cycles). A T2b-discarded re-plan is never appended, so the list is indexed
    by revision. Falls back to the latest plan when the record carries no
    ``best_cycle``, which is the definition the orchestrator wrote.
    """
    plans = record.get("plans") or []
    if not plans:
        metrics = record.get("metrics") or {}
        value = metrics.get("plan_depth")
        return int(value) if value is not None else None
    best = record.get("best_cycle")
    if best is None:
        best = (record.get("metrics") or {}).get("best_cycle")
    chosen = None
    if best is not None:
        for plan in plans:
            if plan.get("revision") == best:
                chosen = plan
                break
    if chosen is None:
        chosen = plans[-1]
    depth = chosen.get("depth")
    return int(depth) if depth is not None else None


def derive_question_metrics(
    record: Mapping[str, Any],
    *,
    config: Any = None,
    config_name: str | None = None,
) -> dict[str, Any]:
    """The ``metrics`` block of one record, with the recomputed fields on top.

    Everything the orchestrator wrote is kept; the recomputed value replaces
    the headline key and the original is retained under ``*_recorded`` /
    ``*_latest`` so the two definitions can be printed side by side.
    """
    metrics = dict(record.get("metrics") or {})
    name = config_name or record.get("config_name")
    decomposition = decompose_saved_calls(record)
    reachable = reachable_llm_paths(config, config_name=name)

    metrics["llm_calls_saved_recorded"] = metrics.get("llm_calls_saved", 0) or 0
    metrics.update(decomposition)
    metrics["llm_calls_saved"] = count_saved_calls(decomposition, reachable)

    if "plan_depth" in metrics:
        metrics["plan_depth_latest"] = metrics.get("plan_depth")
    depth = selected_plan_depth(record)
    if depth is not None:
        metrics["plan_depth"] = depth

    cycles = metrics.get("cycles")
    replans = metrics.get("replans")
    if cycles is not None and replans is not None:
        executed = max(0, int(cycles) - 1)
        metrics["replans_executed"] = executed
        metrics["replans_discarded"] = max(0, int(replans) - executed)
    return metrics


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
    # Recomputed-at-table-time companions (see ``derive_question_metrics``).
    llm_calls_saved_recorded: float = 0.0
    saved_routing: float = 0.0
    saved_kg_alias: float = 0.0
    saved_verifier_gate: float = 0.0
    saved_verifier_pointless: float = 0.0
    saved_extraction: float = 0.0
    saved_planner_template: float = 0.0
    saved_planner_ablation: float = 0.0
    plan_depth_latest: float = 0.0
    replans_executed: float = 0.0
    replans_discarded: float = 0.0
    replan_executed_rate: float = 0.0
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
            "llm_calls_saved_recorded": self.llm_calls_saved_recorded,
            "saved_routing": self.saved_routing,
            "saved_kg_alias": self.saved_kg_alias,
            "saved_verifier_gate": self.saved_verifier_gate,
            "saved_verifier_pointless": self.saved_verifier_pointless,
            "saved_extraction": self.saved_extraction,
            "saved_planner_template": self.saved_planner_template,
            "saved_planner_ablation": self.saved_planner_ablation,
            "plan_depth_latest": self.plan_depth_latest,
            "replans_executed": self.replans_executed,
            "replans_discarded": self.replans_discarded,
            "replan_executed_rate": self.replan_executed_rate,
        }
        d.update(self.extra)
        return d


_MEAN_FIELDS = (
    "llm_calls", "llm_calls_saved", "llm_cache_hits", "tool_calls",
    "parse_failures", "latency_s", "plan_depth", "n_subqueries",
    "cycles", "replans", "degraded_steps",
    "llm_calls_saved_recorded", *SAVED_SOURCES, "plan_depth_latest",
    "replans_executed", "replans_discarded",
)


def _replans_executed(record: Mapping[str, Any]) -> float:
    """Re-plans that produced a cycle: the derived field, else ``cycles - 1``.

    A raw ``metrics`` block that has not been through
    :func:`derive_question_metrics` still carries ``cycles``; a block with
    neither falls back to ``replans`` so an older trace is not read as zero.
    """
    value = record.get("replans_executed")
    if value is not None:
        return float(value)
    cycles = record.get("cycles")
    if cycles is not None:
        return max(0.0, float(cycles) - 1.0)
    return float(record.get("replans", 0) or 0)


def summarise_agent_metrics(records: Sequence[Mapping[str, Any]]) -> AgentMetrics:
    """Aggregate the ``metrics`` block of per-question trace records.

    Two definitions worth stating, because they are easy to conflate:

    * ``replan_rate`` is the FRACTION OF QUESTIONS that triggered at least one
      re-plan -- not the mean number of re-plans. Both are reported;
      ``replans`` carries the mean count. ``replan_executed_rate`` is the
      fraction on which a re-planned cycle actually ran: a re-plan T2b
      discarded as a near-duplicate is triggered but never executed.
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
    m.replan_executed_rate = sum(1 for r in records if _replans_executed(r) > 0) / n
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
    "SAVED_SOURCES", "reachable_llm_paths", "decompose_saved_calls",
    "count_saved_calls", "selected_plan_depth", "derive_question_metrics",
]
