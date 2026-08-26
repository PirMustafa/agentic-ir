"""Data contracts crossing agent boundaries.

Every object an agent produces or consumes is defined here. Agents import this
module; agents never import one another. Frozen dataclasses are artefacts --
they are traced verbatim -- while mutable ones are working memory.

Specification: ``docs/architecture.md`` section 1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Enumerations
#
# Literal rather than Enum: these serialise straight to JSON with no encoder,
# which matters because every one of them lands in the trace.
# ---------------------------------------------------------------------------

ToolName = Literal[
    "bm25_search", "dense_search", "hybrid_search", "rerank",
    "kg_link", "kg_neighbors", "kg_path",
]
Intent = Literal["lookup", "attribute", "comparison", "bridge", "temporal", "yesno"]
AnswerType = Literal["entity", "date", "number", "yesno", "string"]
Strategy = Literal["single_hop", "bridge", "comparison", "attribute", "bridge_comparison"]
Provenance = Literal["bm25", "dense", "hybrid", "rerank", "kg"]
Verdict = Literal["accept", "revise", "abstain"]
Selector = Literal["planner_hint", "heuristic", "llm", "fallback"]
Origin = Literal["llm", "llm_repaired", "fallback_rule", "template_shortcut"]
Source = Literal["hotpotqa", "twowiki"]
ReplanReason = Literal[
    "low_confidence", "missing_evidence", "contradiction",
    "no_citations", "empty_retrieval", "synthesizer_insufficient",
]

SUBQUERY_ID_RE = re.compile(r"^q[1-9][0-9]*$")
PLACEHOLDER_RE = re.compile(r"\{\{(q[1-9][0-9]*)\.(answer|entity|title)\}\}")


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SubQuery:
    """One node in the plan DAG."""

    id: str
    text: str
    depends_on: tuple[str, ...] = ()
    hop: int = 1
    intent: Intent = "lookup"
    entities: tuple[str, ...] = ()
    answer_type: AnswerType = "string"
    tool_hint: ToolName | None = None
    rewrites: tuple[str, ...] = ()
    # A node that restates the question and depends on every other node. The
    # answer comes from composing its dependencies, not from a passage, so
    # retrieval for it is wasted work. Measured: the model emits one reliably.
    is_combiner: bool = False

    def is_template(self) -> bool:
        return "{{" in self.text

    def placeholders(self) -> tuple[tuple[str, str], ...]:
        """``((subquery_id, field), ...)`` for every placeholder in ``text``."""
        return tuple((m.group(1), m.group(2)) for m in PLACEHOLDER_RE.finditer(self.text))


@dataclass(frozen=True, slots=True)
class Plan:
    """An immutable sub-query DAG.

    A re-plan produces a NEW Plan appended to ``QuestionState.plans``; nothing
    is mutated in place, which is what makes the trace a complete audit record.
    """

    question: str
    subqueries: tuple[SubQuery, ...]
    strategy: Strategy = "single_hop"
    # The model labels this wrongly ~7 times in 8 (measured), so it is derived
    # from the DAG shape. `strategy_llm` keeps what the model claimed, so the
    # disagreement rate is reportable rather than silently discarded.
    strategy_llm: Strategy | None = None
    revision: int = 0
    origin: Origin = "llm"
    depth: int = 1
    repairs: tuple[str, ...] = ()
    directive_id: str | None = None
    prompt_id: str | None = None
    raw_llm_output: str | None = None

    def by_id(self, sq_id: str) -> SubQuery:
        for sq in self.subqueries:
            if sq.id == sq_id:
                return sq
        raise KeyError(f"no sub-query {sq_id!r} in plan")

    def topo_order(self) -> tuple[tuple[SubQuery, ...], ...]:
        """Level-synchronous batches, each sorted by numeric id.

        Assumes the DAG is already acyclic -- the Planner's validator repairs
        cycles before a Plan is constructed. Any node still unplaceable after
        the levels are exhausted is appended as a final level rather than
        dropped, so execution never silently loses a sub-query.
        """
        remaining = {sq.id: sq for sq in self.subqueries}
        placed: set[str] = set()
        levels: list[tuple[SubQuery, ...]] = []
        while remaining:
            ready = [
                sq for sq in remaining.values()
                if all(dep in placed or dep not in remaining for dep in sq.depends_on)
            ]
            if not ready:  # defensive: unreachable for a repaired plan
                ready = list(remaining.values())
            ready.sort(key=lambda s: int(s.id[1:]))
            levels.append(tuple(ready))
            for sq in ready:
                placed.add(sq.id)
                del remaining[sq.id]
        return tuple(levels)


# ---------------------------------------------------------------------------
# Corpus and retrieval
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Passage:
    """One corpus paragraph.

    ``sentences`` is not optional. HotpotQA supporting facts are (title,
    sent_id) pairs, so sp_em/sp_f1 are uncomputable without sentence-level
    identity. The split is made once at corpus-build time and persisted --
    re-splitting at query time would drift the ids away from the qrels.
    """

    doc_id: str
    title: str
    text: str
    sentences: tuple[str, ...]
    source: Source
    meta: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScoredPassage:
    passage: Passage
    score: float
    rank: int
    provenance: Provenance
    component_scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolSelection:
    tool: ToolName
    selector: Selector
    rule_id: str | None = None
    rerank_applied: bool = False
    reason: str = ""
    features: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    subquery_id: str
    query_text: str
    selection: ToolSelection
    passages: tuple[ScoredPassage, ...]
    queries_issued: tuple[str, ...] = ()
    n_candidates: int = 0
    latency_s: float = 0.0
    degraded: bool = False
    error: str | None = None


# ---------------------------------------------------------------------------
# Knowledge graph
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Entity:
    entity_id: str
    name: str
    aliases: tuple[str, ...] = ()
    doc_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Triple:
    subject: str
    relation: str
    object: str
    doc_id: str | None = None
    sent_id: int | None = None
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class KGPath:
    nodes: tuple[str, ...]
    edges: tuple[Triple, ...]
    hops: int
    score: float
    bridge_entity: str | None = None


@dataclass(frozen=True, slots=True)
class KGResult:
    subquery_id: str
    seeds: tuple[Entity, ...] = ()
    linked_by: Literal["alias_match", "llm", "retrieval_titles", "none"] = "none"
    paths: tuple[KGPath, ...] = ()
    bridge_entity: str | None = None
    neighbors: tuple[Entity, ...] = ()
    evidence: tuple["Evidence", ...] = ()
    latency_s: float = 0.0
    degraded: bool = False
    error: str | None = None


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Evidence:
    """A citable unit: ONE sentence, or a verbalised triple.

    Sentence granularity is deliberate -- it matches the supporting-fact ground
    truth, keeps NLI premises short enough for DeBERTa to score well, and keeps
    the synthesiser prompt inside num_ctx with room to spare.
    """

    evidence_id: str
    kind: Literal["passage", "kg_triple"]
    text: str
    score: float
    subquery_ids: tuple[str, ...] = ()
    provenance: Provenance = "hybrid"
    doc_id: str | None = None
    title: str | None = None
    sent_id: int | None = None
    triple: Triple | None = None


# ---------------------------------------------------------------------------
# Synthesis and verification
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AnswerCandidate:
    answer: str
    answer_sentence: str
    citations: tuple[str, ...] = ()
    cycle: int = 0
    origin: Origin = "llm"
    sufficient: bool = True
    confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class Claim:
    claim_id: str
    text: str
    cited_evidence_ids: tuple[str, ...]
    nli_label: Literal["entailment", "neutral", "contradiction"]
    nli_score: float
    best_premise_id: str | None = None
    supported: bool = False


@dataclass(frozen=True, slots=True)
class VerificationResult:
    verdict: Verdict
    candidate: AnswerCandidate
    confidence: float
    claims: tuple[Claim, ...] = ()
    nli_support: float = 0.0
    citation_grounding: float = 0.0
    retrieval_agreement: float = 0.0
    llm_support: float | None = None
    hallucinated_citations: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()
    missing_information: tuple[str, ...] = ()
    failed_subquery_ids: tuple[str, ...] = ()
    suggested_subqueries: tuple[str, ...] = ()
    reason: ReplanReason | None = None
    method: Literal["nli_plus_llm", "nli", "llm", "heuristic"] = "nli_plus_llm"
    degraded: bool = False


@dataclass(frozen=True, slots=True)
class ReplanDirective:
    """The feedback edge, made concrete. The only thing that flows backwards.

    ``banned_subquery_texts`` is the anti-loop mechanism and the load-bearing
    field: without it, an 8B model told "that was insufficient, try again"
    re-emits the same decomposition and burns every remaining re-plan.
    """

    directive_id: str
    revision: int
    reason: ReplanReason
    confidence: float
    missing_information: tuple[str, ...] = ()
    failed_subquery_ids: tuple[str, ...] = ()
    suggested_subqueries: tuple[str, ...] = ()
    covered_entities: tuple[str, ...] = ()
    seen_doc_ids: tuple[str, ...] = ()
    banned_subquery_texts: tuple[str, ...] = ()
    previous_plan_summary: str = ""

    def is_informative(self) -> bool:
        """Guard G5: an empty directive gives the Planner nothing to act on."""
        return bool(
            self.missing_information or self.failed_subquery_ids or self.suggested_subqueries
        )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class GoldAnswer:
    """Ground truth for one question.

    Carried on QuestionState for the trace, but NO agent may read it. A test
    greps ``agents/`` for ``state.gold`` -- cheap insurance against leakage
    that would invalidate every number in the report.
    """

    qid: str
    question: str
    answer: str
    dataset: Source
    supporting_facts: tuple[tuple[str, int], ...] = ()
    evidence_triples: tuple[tuple[str, str, str], ...] = ()
    level: str | None = None
    qtype: str | None = None
    context_titles: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LLMCallTrace:
    call_id: str
    agent: str
    prompt_id: str
    prompt_sha1: str
    model: str
    latency_s: float
    parse_ok: bool
    purpose: str = ""
    prompt_chars: int = 0
    completion_chars: int = 0
    think_chars: int = 0
    retries: int = 0
    cache_hit: bool = False
    truncated: bool = False
    raw_output: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ToolCallTrace:
    call_id: str
    agent: str
    tool: ToolName
    query: str
    n_results: int
    latency_s: float
    ok: bool = True
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AgentTrace:
    agent: str
    state: str
    step: int
    cycle: int
    started_at: float
    latency_s: float
    subquery_id: str | None = None
    llm_calls: tuple[LLMCallTrace, ...] = ()
    tool_calls: tuple[ToolCallTrace, ...] = ()
    input_summary: dict[str, Any] = field(default_factory=dict)
    output_summary: dict[str, Any] = field(default_factory=dict)
    degraded: bool = False
    fallback_reason: str | None = None


__all__ = [
    "AnswerType", "Intent", "Origin", "Provenance", "ReplanReason", "Selector",
    "Source", "Strategy", "ToolName", "Verdict",
    "SUBQUERY_ID_RE", "PLACEHOLDER_RE",
    "SubQuery", "Plan",
    "Passage", "ScoredPassage", "ToolSelection", "RetrievalResult",
    "Entity", "Triple", "KGPath", "KGResult",
    "Evidence", "AnswerCandidate", "Claim", "VerificationResult", "ReplanDirective",
    "GoldAnswer", "LLMCallTrace", "ToolCallTrace", "AgentTrace",
]
