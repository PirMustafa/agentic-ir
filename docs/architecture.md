# Architecture and Interface Specification

**Project:** Agentic AI for Information Retrieval · Student ID D03000104
**Status:** written as the M0 contract, and since reconciled against the implementation. Where the built system diverged from the design the divergence is recorded here rather than quietly edited away — §3.0c (the response cache that was never built), §1.10 (the per-cycle state defect), §7 (the `llm/` package that became one module) and §10 (which risks materialised). `config/config.yaml` remains the source of truth for values, with the ten exceptions in §8.1 that live in code.
**Scope:** what a coder needs to implement M1–M5 without guessing.

---

## 0. Design axioms

These are load-bearing. Every later decision follows from them.

1. **The types module is the spine.** `src/agentic_ir/types.py` defines every object crossing an agent boundary. Agents import types; agents never import each other.
2. **No agent may raise.** Every `Agent.run()` wraps its body in `try/except Exception` and returns a deterministic fallback. The orchestrator wraps every agent call. The eval harness wraps every question. Three independent layers, because a 250-question run cannot die at question 194.
3. **Graceful degradation has a floor, and the floor is a working baseline.** If the Planner LLM fails completely, the plan degenerates to a single sub-query equal to the question, routed to `hybrid_search`+rerank. That is exactly the `hybrid_rerank` baseline. The agentic system can therefore never score *below* its strongest non-agentic baseline for reasons of infrastructure failure — only for reasons of judgement. This is worth a sentence in Chapter 4.
4. **Budget is checked, never thrown.** `Budget.try_spend_llm()` returns `bool`. Agents branch to their fallback when it returns `False`. No exception-driven control flow for budget.
5. **Determinism by construction.** Every sort has a tiebreaker. Every set is converted to a sorted tuple before it leaves a function. Every prompt is a versioned file whose SHA-1 is logged.
6. **Local only.** No module may import `requests`, `httpx`, or `openai` except the Ollama client, which talks to `http://localhost:11434` and nothing else. Enforced by `tests/test_guards.py`, which greps the source tree — `test_no_network_libraries_outside_the_llm_client` and `test_no_hosted_llm_provider_anywhere`. (The design named a separate `tests/test_no_network.py`; all five source-grepping guards ended up in one file instead.)

---

## 1. Data contracts

All of this lives in `src/agentic_ir/types.py`. Dataclasses, not TypedDicts: we want `__post_init__` validation, `dataclasses.replace()` for the immutable-update pattern, and `asdict()` for tracing. Immutable objects (plan/result artefacts) are `frozen=True, slots=True`; mutable working memory (`QuestionState`, `Budget`) is a plain dataclass.

```python
# src/agentic_ir/types.py
"""Data contracts crossing agent boundaries.

Every object an agent produces or consumes is defined here. Agents import
this module; agents never import one another. Frozen dataclasses are
artefacts (they are traced verbatim); mutable ones are working memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# --- Enumerations (Literal, not Enum: they serialise to JSON unchanged) ---

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
ReplanReason = Literal[
    "low_confidence", "missing_evidence", "contradiction",
    "no_citations", "empty_retrieval", "synthesizer_insufficient",
]
```

### 1.1 `SubQuery` — a node in the plan DAG

```python
@dataclass(frozen=True, slots=True)
class SubQuery:
    id: str                                   # ^q[1-9][0-9]*$ ; unique within a Plan
    text: str                                 # may contain {{qN.answer}} / {{qN.entity}} / {{qN.title}}
    depends_on: tuple[str, ...] = ()          # ids of earlier SubQueries; must be acyclic
    hop: int = 1                              # 1 + longest path length to a root; set by validator
    intent: Intent = "lookup"
    entities: tuple[str, ...] = ()            # surface forms the planner believes are entities
    answer_type: AnswerType = "string"
    tool_hint: ToolName | None = None         # planner may pre-select; retriever rule R1 honours it
    rewrites: tuple[str, ...] = ()            # <= agents.retriever.max_rewrites expansion variants

    def is_template(self) -> bool: ...        # True iff text contains "{{"
    def placeholders(self) -> tuple[tuple[str, str], ...]: ...   # ((qid, field), ...)
```

`rewrites` is the query-expansion channel required by `agents.planner.allow_query_rewrite`. It rides along inside the *same* planner JSON response — expansion therefore costs **zero additional LLM calls**. Say so in Chapter 4; it is a real efficiency argument.

### 1.2 `Plan` — the DAG

```python
@dataclass(frozen=True, slots=True)
class Plan:
    question: str
    subqueries: tuple[SubQuery, ...]
    strategy: Strategy = "single_hop"
    revision: int = 0                         # 0 = initial; 1..max_replans = re-plans
    origin: Origin = "llm"
    depth: int = 1                            # longest path in the DAG == agent metric plan_depth
    repairs: tuple[str, ...] = ()             # e.g. ("cycle_broken:q3->q1", "dropped_dep:q2->q9")
    directive_id: str | None = None           # ReplanDirective that produced this revision
    prompt_id: str | None = None              # e.g. "planner.decompose.v1"
    raw_llm_output: str | None = None         # truncated to trace.raw_output_chars

    def topo_order(self) -> tuple[tuple[SubQuery, ...], ...]: ...  # level-synchronous batches
    def by_id(self, sq_id: str) -> SubQuery: ...
```

`Plan` is immutable. A re-plan produces a *new* `Plan` appended to `QuestionState.plans`; nothing is mutated in place. This is what makes the trace a complete audit record.

### 1.3 `Passage` and `ScoredPassage`

```python
@dataclass(frozen=True, slots=True)
class Passage:
    doc_id: str                               # "hotpotqa:Arthur%27s_Magazine" — stable, URL-safe
    title: str
    text: str                                 # full paragraph
    sentences: tuple[str, ...]                # supporting-fact granularity; index == sent_id
    source: Literal["hotpotqa", "twowiki"]
    meta: dict[str, str] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class ScoredPassage:
    passage: Passage
    score: float                              # score of the tool that produced the final ranking
    rank: int                                 # 0-based, post-fusion, post-rerank
    provenance: Provenance
    component_scores: dict[str, float] = field(default_factory=dict)
    # e.g. {"bm25": 14.2, "dense": 0.81, "rrf": 0.0317, "ce": 5.44}
```

`sentences` is non-negotiable: HotpotQA supporting facts are `(title, sent_id)` pairs, and `sp_em` / `sp_f1` are uncomputable without sentence-level identity. Split once at corpus build time with a fixed, deterministic splitter and persist the split — never re-split at query time, or sentence ids will drift between the index and the qrels.

### 1.4 `RetrievalResult`

```python
@dataclass(frozen=True, slots=True)
class ToolSelection:
    tool: ToolName
    selector: Selector                        # planner_hint | heuristic | llm | fallback
    rule_id: str | None                       # "R3_oov" when selector == "heuristic"
    rerank_applied: bool
    reason: str = ""                          # one line, for qualitative error analysis
    features: dict[str, float] = field(default_factory=dict)  # the routing features, logged

@dataclass(frozen=True, slots=True)
class RetrievalResult:
    subquery_id: str
    query_text: str                           # the RESOLVED text actually sent to the index
    queries_issued: tuple[str, ...]           # query_text + rewrites actually used
    selection: ToolSelection
    passages: tuple[ScoredPassage, ...]       # len <= retrieval.top_k
    n_candidates: int                         # pool size before truncation
    latency_s: float
    degraded: bool = False
    error: str | None = None
```

### 1.5 `Evidence` — the citable unit

```python
@dataclass(frozen=True, slots=True)
class Evidence:
    evidence_id: str                          # "e1".."eN"; assigned in AGGREGATE, deterministic
    kind: Literal["passage", "kg_triple"]
    text: str                                 # ONE sentence, or a verbalised triple
    score: float
    subquery_ids: tuple[str, ...]             # which sub-queries surfaced it (dedup unions these)
    provenance: Provenance
    doc_id: str | None = None
    title: str | None = None
    sent_id: int | None = None                # index into Passage.sentences
    triple: "Triple | None" = None
```

Evidence is **sentence-granular**, not passage-granular. Three reasons: it matches the supporting-fact ground truth, it makes NLI premises the right length (DeBERTa NLI degrades badly on 200-word premises), and it keeps the synthesiser prompt inside `num_ctx: 8192` with room to spare (20 sentences ≈ 800 tokens).

### 1.6 Knowledge-graph types

```python
@dataclass(frozen=True, slots=True)
class Entity:
    entity_id: str                            # normalised canonical form of the page title
    name: str
    aliases: tuple[str, ...] = ()
    doc_ids: tuple[str, ...] = ()

@dataclass(frozen=True, slots=True)
class Triple:
    subject: str                              # entity_id
    relation: str                             # relation label (see 3.3 for how it is derived)
    object: str                               # entity_id
    doc_id: str | None = None                 # provenance: the passage asserting it
    sent_id: int | None = None
    weight: float = 1.0

@dataclass(frozen=True, slots=True)
class KGPath:
    nodes: tuple[str, ...]                    # entity_ids, len == hops + 1
    edges: tuple[Triple, ...]                 # len == hops
    hops: int
    score: float                              # product of edge weights, normalised
    bridge_entity: str | None = None          # meeting node for bidirectional search

@dataclass(frozen=True, slots=True)
class KGResult:
    subquery_id: str
    seeds: tuple[Entity, ...]
    linked_by: Literal["alias_match", "llm", "retrieval_titles", "none"]
    paths: tuple[KGPath, ...]
    bridge_entity: str | None
    neighbors: tuple[Entity, ...]             # len <= agents.kg.max_neighbors
    evidence: tuple[Evidence, ...]            # mention sentences supporting the traversed edges
    latency_s: float
    degraded: bool = False
    error: str | None = None
```

### 1.7 Synthesis and verification

```python
@dataclass(frozen=True, slots=True)
class AnswerCandidate:
    answer: str                               # short span, EM/F1-comparable
    answer_sentence: str                      # one declarative sentence == the NLI hypothesis
    citations: tuple[str, ...]                # evidence_ids
    cycle: int                                # which plan revision produced it
    origin: Origin
    sufficient: bool = True                   # synthesiser's own self-report
    confidence: float = 0.0                   # filled by the Verifier

@dataclass(frozen=True, slots=True)
class Claim:
    claim_id: str
    text: str                                 # the hypothesis actually sent to NLI
    cited_evidence_ids: tuple[str, ...]
    nli_label: Literal["entailment", "neutral", "contradiction"]
    nli_score: float                          # P(entailment) of the best premise
    best_premise_id: str | None
    supported: bool

@dataclass(frozen=True, slots=True)
class VerificationResult:
    verdict: Verdict
    candidate: AnswerCandidate
    confidence: float
    claims: tuple[Claim, ...]
    # --- component signals, all logged so the confidence blend is auditable ---
    nli_support: float                        # max P(entailment) over cited premises
    citation_grounding: float                 # fraction of claims with >=1 valid supporting citation
    retrieval_agreement: float                # fraction of sub-queries whose top title is cited
    llm_support: float | None                 # None when the adjudication call was skipped
    hallucinated_citations: tuple[str, ...]   # cited ids that do not exist
    contradictions: tuple[str, ...]
    # --- feedback channel to the Planner ---
    missing_information: tuple[str, ...]
    failed_subquery_ids: tuple[str, ...]
    suggested_subqueries: tuple[str, ...]
    reason: ReplanReason | None
    method: Literal["nli_plus_llm", "nli", "llm", "heuristic"]
    degraded: bool = False
```

### 1.8 `ReplanDirective` — the feedback edge, made concrete

This object *is* the contribution the README claims. It is the only thing that flows backwards.

```python
@dataclass(frozen=True, slots=True)
class ReplanDirective:
    directive_id: str                         # f"{qid}:r{revision}"
    revision: int                             # the revision the Planner is being asked to produce
    reason: ReplanReason
    confidence: float                         # what the Verifier scored the previous cycle
    missing_information: tuple[str, ...]      # natural language: what could not be established
    failed_subquery_ids: tuple[str, ...]      # sub-queries that returned nothing useful
    suggested_subqueries: tuple[str, ...]     # Verifier's proposals; the Planner may ignore them
    covered_entities: tuple[str, ...]         # already have evidence for these — do not re-ask
    seen_doc_ids: tuple[str, ...]             # novelty pressure
    banned_subquery_texts: tuple[str, ...]    # normalised texts already tried; anti-loop
    previous_plan_summary: str                # compact "q1: ... -> q2: ..." rendering
```

`banned_subquery_texts` is the anti-loop mechanism and the most important field here. Without it, a local 8B model faced with "that was insufficient, try again" reliably re-emits the same decomposition, burning both re-plans for nothing. See §2.4 for the duplicate test.

### 1.9 Tracing

```python
@dataclass(frozen=True, slots=True)
class LLMCallTrace:
    call_id: str
    agent: str
    prompt_id: str                            # "planner.decompose.v1"
    prompt_sha1: str                          # of the rendered prompt, for exact reproduction
    model: str
    purpose: str
    prompt_chars: int
    completion_chars: int
    think_chars: int                          # size of the stripped <think> block; see 3.0
    latency_s: float
    parse_ok: bool
    retries: int                              # 0..llm.max_format_retries
    cache_hit: bool
    truncated: bool                           # completion hit num_predict
    raw_output: str | None                    # truncated to trace.raw_output_chars
    error: str | None = None

@dataclass(frozen=True, slots=True)
class ToolCallTrace:
    call_id: str
    agent: str
    tool: ToolName
    query: str
    n_results: int
    latency_s: float
    ok: bool
    error: str | None = None

@dataclass(frozen=True, slots=True)
class AgentTrace:
    agent: str                                # planner | retriever | kg | synthesizer | verifier
    state: str                                # orchestrator state during which it ran
    step: int                                 # monotonic within the question
    cycle: int                                # plan revision index
    subquery_id: str | None                   # None for whole-question agents
    started_at: float                         # perf_counter offset from t0
    latency_s: float
    llm_calls: tuple[LLMCallTrace, ...] = ()
    tool_calls: tuple[ToolCallTrace, ...] = ()
    input_summary: dict[str, Any] = field(default_factory=dict)
    output_summary: dict[str, Any] = field(default_factory=dict)
    degraded: bool = False
    fallback_reason: str | None = None        # why the deterministic path was taken
```

### 1.10 Working memory

```python
@dataclass
class Budget:
    max_llm_calls: int
    max_iterations: int
    max_wall_clock_s: float
    max_replans: int
    reserve_llm_calls: int = 2                # only synthesizer+verifier may spend the last 2
    llm_calls: int = 0
    llm_cache_hits: int = 0
    llm_calls_saved: int = 0                  # heuristic shortcuts that avoided a call
    tool_calls: int = 0
    iterations: int = 0
    replans: int = 0
    t0: float = 0.0

    def elapsed(self) -> float: ...
    def remaining_llm(self, *, privileged: bool = False) -> int: ...
    def try_spend_llm(self, *, privileged: bool = False) -> bool: ...
    def note_saved(self, n: int = 1) -> None: ...
    def wallclock_exceeded(self) -> bool: ...

@dataclass
class QuestionState:
    qid: str
    question: str
    dataset: str
    config_name: str
    gold: dict[str, Any] | None = None        # never read by agents; carried for the trace only
    plans: list[Plan] = field(default_factory=list)
    results: dict[str, RetrievalResult] = field(default_factory=dict)
    kg_results: dict[str, KGResult] = field(default_factory=dict)
    answers: dict[str, str] = field(default_factory=dict)      # subquery_id -> extracted span
    bridge_entities: dict[str, str] = field(default_factory=dict)
    evidence: dict[str, Evidence] = field(default_factory=dict)
    candidates: list[AnswerCandidate] = field(default_factory=list)
    verifications: list[VerificationResult] = field(default_factory=list)
    traces: list[AgentTrace] = field(default_factory=list)
    budget: Budget = ...
    state: str = "INIT"
    terminated_by: str | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def plan(self) -> Plan | None: ...        # latest
    def banned_texts(self) -> tuple[str, ...]: ...
```

**Known defect (observed in the HotpotQA `agentic_full` run, not fixed).** `results`,
`kg_results`, `answers` and `bridge_entities` are keyed by `subquery_id`, and every plan
revision numbers its nodes from `q1`. A re-plan's `q1` therefore **overwrites** cycle 0's
`q1`, while any id the new plan does not reuse **survives stale** from the earlier cycle.
In `agentic_full_hotpotqa_20260904T221533Z`, 118 of 250 questions re-planned; **all 118**
had at least one cycle-0 id reused by a later plan, so cycle 0's results for that id were
overwritten, and **92** end with at least one entry in `retrieved` whose key the final
plan does not use — a result carried over from a superseded cycle. (An earlier draft said
115 for the first figure. It reproduces to 118 under every definition tried; 92 reproduces
exactly under the definition just stated, which is why that one is spelled out.) Two
consequences: (i) the retrieval metrics for re-planned questions are computed over a
last-cycle-plus-stale pool, not over everything the question retrieved; (ii) the trace
is **not** the complete audit record §1.3 promises — `plans` is immutable and complete,
but the per-sub-query state the plans produced is not, so cycle 0 of a re-planned
question cannot be reconstructed. **Fix:** namespace the four maps by cycle
(`dict[tuple[int, str], ...]`, or one `CycleState` per revision appended to a list) and
have `AGGREGATE` pool over all cycles while `EXECUTE` resolves placeholders against the
current one only. The fix changes what `AGGREGATE` sees and therefore the numbers; it
belongs to a re-run, not to a table regeneration. Reported in the Chapter 5 limitations.

`gold` is carried on the state but **no agent may read it**. Enforced by `tests/test_guards.py::test_agents_never_read_gold`, which greps `agents/` for `state.gold`. Cheap insurance against accidental leakage that would invalidate every number in the report.

---

## 2. Orchestrator state machine

`src/agentic_ir/orchestrator.py`. Explicit state machine, not implicit control flow — the professor wants the mechanism visible, and an explicit machine is directly drawable as a TikZ figure in Chapter 4.

**Nine states and fifteen transitions**, and both are data rather than prose: `STATES` is a 9-tuple and `TRANSITIONS` is a 15-entry `dict[str, tuple[from, to, guard]]`, both exported from `orchestrator.py`, so §2.2 below is checkable against the module rather than trusted. The count is fifteen and not fourteen because `T2b` is a second edge out of `PLAN` and does not renumber the `T1..T14` sequence; the module docstring says "fourteen" and is undercounting it. `_MAX_MACHINE_STEPS = 400` is an outer stop against a mis-wired handler, not a budget.

### 2.1 Diagram

```
                                   ┌────────┐
                                   │  INIT  │
                                   └───┬────┘
                                       │ T1
                                       ▼
              ┌───────────────────►┌────────┐  Planner: 1 LLM call (0 on fallback)
              │        T11         │  PLAN  │  emits Plan(revision=n)
              │  re-plan, carrying └───┬────┘
              │  ReplanDirective       │                    T2b  revision>0 AND
              │                        │ T2                 plan ≈ a previous plan
              │                        ▼                         │
              │                  ┌───────────┐                   │
              │        T3 ┌─────►│  EXECUTE  │  DAG driver, level-synchronous
              │  next ready└─────┤           │  per node: RESOLVE → ROUTE → RETRIEVE
              │       node       └─────┬─────┘             → KG → EXTRACT
              │                        │ T4  all nodes terminal | budget | wall clock
              │                        ▼
              │                  ┌───────────┐
              │                  │ AGGREGATE │  dedup, rank, assign e1..eN  (0 LLM calls)
              │                  └─────┬─────┘
              │                        │ T5
              │                        ▼
              │                 ┌──────────────┐
              │                 │  SYNTHESIZE  │  1 LLM call (privileged/reserved)
              │                 └───┬──────┬───┘
              │              T6     │      │ T7  verifier disabled (ablation)
              │                     ▼      │                    │
              │                ┌────────┐  │                    │
              │                │ VERIFY │  │  NLI: 0 calls      │
              │                └─┬───┬──┘  │  adjudication: <=1 │
              │        T8 accept │   │ T9 revise                │
              │       T10 abstain│   ▼                          │
              │                  │ ┌──────────────┐             │
              │  G1..G5 all pass │ │ REPLAN_GATE  │             │
              └──────────────────┴─┤              │             │
                                   └──────┬───────┘             │
                                          │ T12 any guard fails │
                                          ▼                     ▼
     ANY ── T14 unhandled exception ──►┌──────────┐◄────────────┘◄──── T2b
                                       │ FINALIZE │  argmax-confidence over ALL cycles
                                       └────┬─────┘
                                            │ T13  emit JSONL trace record
                                            ▼
                                         ┌──────┐
                                         │ DONE │
                                         └──────┘
```

### 2.2 Transition table

| # | From | Guard / event | To | Actions |
|---|---|---|---|---|
| T1 | INIT | always | PLAN | `budget.t0 = perf_counter()`; build `Budget` from config |
| T2 | PLAN | plan non-empty and validated (always true — fallback guarantees ≥1 node) | EXECUTE | `budget.iterations += 1`; append `Plan` to `state.plans` |
| T2b | PLAN | `revision > 0` and new plan is a near-duplicate (§2.4) of any prior plan | FINALIZE | discard the plan; `terminated_by = "degenerate_replan"` |
| T3 | EXECUTE | ready set non-empty and not `budget.wallclock_exceeded()` | EXECUTE | run the next DAG node's micro-pipeline (§4) |
| T4 | EXECUTE | all nodes terminal, or wall clock exceeded, or `remaining_llm(privileged=False) == 0` | AGGREGATE | mark unrun nodes `skipped` |
| T5 | AGGREGATE | always (empty evidence is allowed; synthesiser degrades) | SYNTHESIZE | assign `e1..eN` |
| T6 | SYNTHESIZE | `agents.verifier.enabled` | VERIFY | append `AnswerCandidate` |
| T7 | SYNTHESIZE | not `agents.verifier.enabled` (`agentic_no_verifier`) | FINALIZE | `terminated_by = "verifier_disabled"` |
| T8 | VERIFY | `verdict == "accept"` | FINALIZE | `terminated_by = "verified"` |
| T9 | VERIFY | `verdict == "revise"` | REPLAN_GATE | build `ReplanDirective` |
| T10 | VERIFY | `verdict == "abstain"` | FINALIZE | `terminated_by = "abstained"` |
| T11 | REPLAN_GATE | **G1 ∧ G2 ∧ G3 ∧ G4 ∧ G5** | PLAN | `budget.replans += 1`; pass the directive |
| T12 | REPLAN_GATE | any guard fails | FINALIZE | `terminated_by = <id of first failing guard>` |
| T13 | FINALIZE | always | DONE | select best candidate; write the trace line |
| T14 | any | unhandled exception escapes both agent- and orchestrator-level handlers | FINALIZE | record in `state.errors`; `terminated_by = "error"` |

### 2.3 Re-plan guards

Evaluated in order; the first failure names the termination reason.

| id | Guard | Config source | `terminated_by` on failure |
|---|---|---|---|
| G1 | `state.budget.replans < max_replans` (2) | `agents.planner.max_replans` | `max_replans` |
| G2 | `state.budget.iterations < max_iterations` (6) | `orchestrator.max_iterations` | `budget_iterations` |
| G3 | `budget.remaining_llm(privileged=True) >= 3` (1 planner + 1 synth + 1 verify) | `orchestrator.max_llm_calls_per_question` | `budget_llm` |
| G4 | `budget.elapsed() < 0.6 * max_wall_clock_s` (180 s of 300) | `orchestrator.max_wall_clock_s` | `budget_wallclock` |
| G5 | directive is informative: `missing_information` or `failed_subquery_ids` or `suggested_subqueries` non-empty | — | `uninformative_feedback` |

G4 uses a flat 0.6 fraction rather than a predicted cycle cost: predicting is more accurate but not reproducible across machines, and reproducibility wins. One line of justification for the report.

**Iteration is defined as one PLAN→…→VERIFY macro-cycle.** With `max_replans = 2` the normal ceiling is 3 iterations; `max_iterations = 6` is an outer guard covering pathological paths, and is expected never to bind. Report it as such rather than pretending it is tuned.

### 2.4 Near-duplicate plan test (T2b)

```
normalise(t) = lowercase, strip punctuation, strip {{...}} placeholders,
               drop a fixed 25-word stopword list, tokenise, return a set

plans P and Q are near-duplicates iff
    |P.subqueries| == |Q.subqueries|
    and there is a bijection m: P -> Q with
        jaccard(normalise(p.text), normalise(m(p).text)) >= 0.85 for all p
```

Bijection by greedy max-Jaccard matching (plans have ≤5 nodes; exhaustive is fine). The same `normalise` builds `banned_subquery_texts`.

### 2.5 Answer selection at FINALIZE

`orchestrator.answer_selection: best_confidence`. Choose `argmax(candidate.confidence)` over **all** cycles; ties broken by lower cycle index (prefer the earlier, cheaper answer). Never blindly take the last cycle: a re-plan can make things worse, and letting it silently overwrite a better first answer would make the feedback loop look harmful in the results table for the wrong reason. Log `best_cycle` in the trace so "did re-planning actually help?" is directly answerable — that is a Chapter 4 table on its own.

### 2.6 LLM budget accounting

Worst case for `agentic_full`, per question:

| Consumer | Calls per cycle | Cycles | Worst case |
|---|---|---|---|
| Planner (decompose / re-plan) | 1 | 3 | 3 |
| Retriever tool selection | 0 (heuristic, §5) | 3 | **0** |
| Sub-query answer extraction | ≤2 (non-leaf nodes only, §4.4) | 3 | 6 |
| KG entity linking | 0 (`alias_match`) | 3 | **0** |
| Synthesizer | 1 | 3 | 3 |
| Verifier adjudication | ≤1 (uncertainty band only, §3.5) | 3 | 3 |
| | | **Total** | **15 ≤ 20** ✓ |

The cap has ~25% headroom, so it should almost never bind — which is the point: it is a safety net, not a tuning knob. Expected typical cost is 4–7 calls.

**Reservation.** `reserve_llm_calls: 2`. Non-privileged callers (planner, extraction, KG) see `remaining_llm() = max_llm_calls - llm_calls - 2`; privileged callers (synthesizer, verifier) see the true remainder. This guarantees an answer is always produced, which the naive cap does not.

---

## 3. Per-agent contracts

Common protocol, `src/agentic_ir/agents/base.py`:

```python
class Agent(Protocol):
    name: str
    def run(self, state: QuestionState, **kwargs) -> Any: ...   # never raises
```

Each agent uses the `state.step(...)` context manager, which times the call, catches `Exception`, records an `AgentTrace`, sets `degraded=True`, and returns control to the agent's fallback branch.

### 3.0 The LLM client — three things that must be right

Implemented in `src/agentic_ir/llm.py` (see the deviation note in §7).

**(a) Suppress qwen3 thinking.** `qwen3:8b` is a hybrid reasoning model that emits `<think>…</think>` by default. Measured: with `think=True` and `num_predict=128`, the model spent the **entire** budget on a 503-character reasoning block and produced no answer at all. Thinking does not merely add latency — it starves the response. Belt and braces, all three:

1. pass `think=False` to `ollama.chat` (ignored by older clients, hence the rest),
2. append `/no_think` to the system prompt,
3. strip `<think>.*?</think>` (DOTALL) from the response before parsing, and record `think_chars` in the trace so the report can state how often thinking leaked through.

**(b) JSON extraction ladder.** Call with a JSON schema in `format`, then:

| Rung | Action |
|---|---|
| 1 | `json.loads(raw)` |
| 2 | strip `<think>` blocks and ```` ```json ```` fences, retry rung 1 |
| 3 | string-aware balanced-brace scan for the first complete `{...}` / `[...]`, retry rung 1 |
| 4 | remove trailing commas before `}` / `]`, retry rung 1 |
| 5 | re-prompt with the repair template (`repair.json.v1`: the schema + the bad output + "return only valid JSON"), counts against `llm.max_format_retries` (2) |
| 6 | give up → `parse_ok = False`, increment `parse_failures`, agent takes its deterministic fallback |

No `eval`, no `ast.literal_eval`, no regex-based JSON "fixing" beyond rung 4. Schema validation is hand-written per agent (no pydantic — not in `requirements.txt`).

**(c) Response cache — designed, never built.** The original design called for `data/cache/llm_cache.sqlite` keyed on `sha256(model | sorted(options) | prompt_id | rendered_prompt)`, to make a completed run exactly reproducible and to turn eval-harness re-runs from hours into seconds. **It does not exist.** There is no sqlite cache anywhere in `src/`; `cache_hit` is `False` on all 1065 calls of the headline run, and `run_eval.py` reads `llm.cache` only to record cache coldness in `meta.json`. The config key and its `path` are kept so `cache_cold` keeps its meaning, and `config/config.yaml` says so at the declaration.

Two consequences, both of which the report depends on and Chapter 4 states:

* Every latency figure in the report is a **cold** measurement. That is what the report needs — a cached latency would measure the cache, not the system — so the absence costs nothing here.
* Re-runs are **not** free, and a completed run is not bit-reproducible by replay. See risk 7.

`llm_calls` is therefore a count of real calls, and `llm_cache_hits` is structurally zero rather than merely low.

---

### 3.1 Planner

| | |
|---|---|
| **Input** | `question: str`, `directive: ReplanDirective \| None` |
| **Output** | `Plan` |
| **Prompts** | `planner.decompose.v1` (revision 0), `planner.replan.v1` (revision ≥ 1) |
| **LLM calls** | exactly 1, or 0 when budget is exhausted / template shortcut fires |

**Output schema** (both prompts):

```json
{
  "strategy": "bridge",
  "subqueries": [
    {"id": "q1", "text": "Who directed the film Titanic?",
     "depends_on": [], "intent": "lookup", "entities": ["Titanic"],
     "answer_type": "entity", "tool_hint": null,
     "rewrites": ["director of Titanic film"]},
    {"id": "q2", "text": "When was {{q1.answer}} born?",
     "depends_on": ["q1"], "intent": "attribute", "entities": [],
     "answer_type": "date", "tool_hint": null, "rewrites": []}
  ]
}
```

**Validation** (`_validate_plan`, all repairs recorded in `Plan.repairs`):

| Violation | Repair |
|---|---|
| `id` not matching `^q[1-9][0-9]*$` | renumber positionally |
| duplicate ids | renumber positionally |
| `> max_subqueries` (5) | truncate to the first 5, drop now-dangling deps |
| `depends_on` names an unknown id | drop that edge (`dropped_dep:...`) |
| cycle | drop the back edge whose target id is numerically higher (`cycle_broken:...`); Kahn re-run |
| placeholder `{{qK.f}}` with `qK ∉ depends_on` | add the edge if `qK` exists, else strip the placeholder |
| `text` empty or `< 3` chars | drop the node |
| `rewrites` longer than `max_rewrites` (2) | truncate |
| all nodes dropped | fall through to the fallback ladder |

**`strategy` is DERIVED, not trusted.** Measured on 8 representative questions
(qwen3:8b, `temperature 0`, schema-constrained): the model produced structurally
**correct** decompositions 8/8 — right bridge shape, right `{{qN.answer}}`
placeholders — but labelled `strategy` as `single_hop` on 7 of 8, including
obvious bridge and comparison cases. It is good at building the graph and bad at
naming it. Since `strategy` gates the deterministic comparison shortcut (§5.4)
and fallback F1, trusting the label would silently disable both.

Infer it from the validated DAG instead, and overwrite whatever the model said:

| DAG shape | Derived strategy |
|---|---|
| 1 node | `single_hop` |
| ≥2 roots, plus one node depending on all of them | `comparison` |
| a chain with ≥1 dependent node | `bridge` |
| ≥2 roots and a dependent chain of depth ≥2 | `bridge_comparison` |
| dependent node whose intent is `attribute` | `attribute` |

Log both `strategy_llm` and `strategy_derived` in the trace: their disagreement
rate is a cheap, honest datapoint for Chapter 4 on where an 8B model's
self-reporting can and cannot be relied upon.

**Combiner nodes.** The model reliably emits a final node that restates the
original question and depends on all the others (`q3: "Which magazine was
started first, ..." deps=[q1,q2]`). Retrieving for it is wasted work — the
answer comes from composing q1 and q2, not from a passage. Detect it (depends on
every other node **and** Jaccard ≥ 0.6 with the original question), mark
`is_combiner=True`, skip RETRIEVE and KG for it, and let the Synthesizer handle
the composition. On a 3-node comparison plan this removes a third of all
retrieval work.

**Fallback ladder** (deterministic, no LLM):

- **F1 — comparison template.** Question matches one of a small fixed regex set (`which .* (first|earlier|older|newer|later)`, `(are|were|is|was) (both|the same)`, `who was born (first|earlier)`, `X or Y`) **and** two entity spans are extractable (capitalised token runs, or the operands of `" or "` / `", "`). Emit `strategy="comparison"` with two independent look-up nodes, one per entity, `answer_type` inherited from the comparison type. This covers a large fraction of both datasets and produces a *correct* plan, not merely a safe one.
- **F2 — iterative bridge.** ≥2 distinct capitalised entity runs and no comparison marker. Emit `q1 = question` (`answer_type="entity"`) and `q2 = "{{q1.title}} " + question`, `depends_on=["q1"]`, `strategy="bridge"`. This is self-ask-lite: it preserves multi-hop behaviour when the LLM is unavailable.
- **F3 — identity.** Single node, `text = question`, `strategy="single_hop"`, `tool_hint="hybrid_search"`. Always succeeds. This is the floor from axiom 3.

Set `Plan.origin = "fallback_rule"` and `AgentTrace.fallback_reason ∈ {parse_failure, budget_exhausted, validation_empty}`.

**`planner.replan.v1` prompt content** — this is where precision matters most:

```
Original question: {question}
Previous plan:     {directive.previous_plan_summary}
What we already established: {directive.covered_entities}
What the verifier could NOT establish: {directive.missing_information}
Sub-queries that returned nothing useful: {directive.failed_subquery_ids}
Confidence of the previous answer: {directive.confidence:.2f}

Do NOT repeat any of these sub-queries (they were already tried):
{directive.banned_subquery_texts}

Produce a NEW plan that targets the missing information.
At least one sub-query must be materially different from every banned one.
```

Passing retrieved *titles* rather than retrieved *text* keeps this prompt at roughly 300–500 tokens instead of several thousand. At this hardware's speed, prompt length is latency; be miserly.

---

### 3.2 Retrieval Agent

| | |
|---|---|
| **Input** | `SubQuery` (placeholders already resolved), `QuestionState` |
| **Output** | `RetrievalResult` |
| **Prompts** | `retriever.select_tool.v1` — **issued only when `agents.retriever.heuristic_shortcut` is `false`** |
| **LLM calls** | **0** in the shipped config. See §5. |

**Multi-query retrieval.** Issue `[subquery.text] + subquery.rewrites[:max_rewrites]` against the selected tool and fuse with RRF (`rrf_k: 60`), then apply the rerank gate. Free expansion — the rewrites came with the plan.

**Output schema** (LLM path only):

```json
{"tool": "hybrid_search", "rerank": true, "reason": "entity-sparse natural language question"}
```

**Fallback:** rule R6 — `hybrid_search` + rerank, `selector="fallback"`. Unconditionally safe.

**Tool registry** (`tools/registry.py`) — a plain dict of `ToolSpec(name, fn, arg_schema, description)`. The `description` strings are what get injected into `retriever.select_tool.v1`, so the LLM-facing tool documentation and the executable registry can never drift apart. That single-source property is worth a paragraph in Chapter 4 on tool-invocation management.

---

### 3.3 KG Navigator

| | |
|---|---|
| **Input** | `SubQuery`, its `RetrievalResult`, the loaded `networkx` graph |
| **Output** | `KGResult` |
| **Prompts** | `kg.link_entities.v1` — only when `agents.kg.entity_linker == "llm"`, or alias matching returns zero seeds **and** non-privileged budget remains |
| **LLM calls** | **0** in the shipped config (`entity_linker: alias_match`) |

**Graph construction is offline** (`scripts/build_kg.py`), from the retrieval corpus only:

- **Nodes** = passage titles, normalised (NFKD, lowercase, strip diacritics, strip parenthetical disambiguators, collapse whitespace). Alias table = surface title + no-parenthetical variant + comma-inverted variant + any redirect strings the dataset provides.
- **Edges** `A → B` when an alias of `B` occurs in the text of passage `A`. Stored with `doc_id`, `sent_id`, and the containing sentence — so **every edge is independently citable evidence**, which is what makes KG results scoreable rather than merely decorative.
- **Relation label** = the shortest verb-anchored window between the two mentions in that sentence, truncated to 6 tokens; `"mentions"` if no verb is found. No parser dependency, no LLM.
- **Weight** = `1 / (1 + log(1 + out_degree(A)))`, so hub pages contribute weak edges. Deterministic.

**Traversal.**

- 1 seed → BFS to `max_hops` (2), keeping the top `max_neighbors` (25) per node by `(weight desc, entity_id asc)`.
- ≥2 seeds → **bidirectional BFS**; the meeting node is `bridge_entity`. This is the single highest-value operation in the whole KG agent, because on bridge questions the bridge entity is usually a page title and the KG finds it in milliseconds where the LLM would take seconds.
- Comparison questions (2 seeds, no path expected) → return each seed's own attribute-bearing sentences as evidence; do not force a path.

**Fallback ladder:** alias match → (zero seeds) top-3 retrieved passage titles as seeds, `linked_by="retrieval_titles"` → (graph unavailable / exception) empty `KGResult` with `degraded=True`. An empty `KGResult` is a legal, non-fatal outcome; the pipeline continues on passage evidence alone.

---

### 3.4 Synthesizer

| | |
|---|---|
| **Input** | `question`, ranked `Evidence` list (≤ `verifier.max_evidence`, default 20) |
| **Output** | `AnswerCandidate` |
| **Prompts** | `synth.answer.v1` |
| **LLM calls** | 1, privileged |

**Output schema:**

```json
{"answer": "Arthur's Magazine",
 "answer_sentence": "Arthur's Magazine was started before First for Women.",
 "citations": ["e3", "e7"],
 "sufficient": true}
```

`answer_sentence` costs nothing extra and gives the Verifier a clean NLI hypothesis — far better than the usual hack of concatenating question and answer. `sufficient: false` is an early, cheap re-plan trigger (`reason = "synthesizer_insufficient"`) that skips the adjudication call entirely.

**Fallback ladder (extractive, deterministic):**

1. `answer_type == "yesno"` → comparison rule over the two operand evidence sets (see §5.4).
2. `answer_type in {date, number}` → highest-scoring evidence sentence containing a matching regex (`\b(1[0-9]{3}|20[0-9]{2})\b` for years; a date/number pattern otherwise); answer = the matched span.
3. `answer_type == "entity"` → title of the top-ranked evidence's passage.
4. otherwise → the top-ranked evidence sentence, truncated to 20 tokens.
5. no evidence at all → `answer = ""`, `sufficient = False`, `degraded = True`.

Citations are set to the evidence actually used, so citation grounding stays meaningful even on the fallback path.

---

### 3.5 Verifier

| | |
|---|---|
| **Input** | `AnswerCandidate`, `Evidence` map, `QuestionState` |
| **Output** | `VerificationResult` |
| **Prompts** | `verifier.adjudicate.v1` — **only inside the uncertainty band** |
| **LLM calls** | 0 or 1, privileged |

**Pipeline:**

1. **Citation resolution.** Cited ids not present in the evidence map → `hallucinated_citations`; they contribute 0 support. (This alone catches a real and common local-model failure.)
2. **NLI.** `cross-encoder/nli-deberta-v3-base`, premise = each cited evidence sentence, hypothesis = `answer_sentence`. `nli_support = max P(entailment)`. Additionally score the top-5 *uncited* evidence sentences; any with `P(contradiction) > 0.7` are recorded in `contradictions`.
3. **Signals.**
   - `citation_grounding` = |claims with ≥1 non-hallucinated, non-contradicted supporting citation| / |claims|. The Synthesizer emits one `answer_sentence`, so the implementation builds exactly one claim (`c1`) and this signal is **binary in practice**: 1.0 when at least one resolved citation survives the contradiction screen, 0.0 otherwise. Read the reported grounding rate as "the share of questions with at least one surviving citation", not as a per-claim average — nothing in the pipeline currently produces a second claim.
   - `retrieval_agreement` = fraction of executed sub-queries whose top-1 passage title appears among cited evidence titles
4. **Confidence blend** (weights in config, `agents.verifier.weights`):

   ```
   conf = 0.45*nli_support + 0.25*citation_grounding + 0.15*retrieval_agreement + 0.15*llm_support
   ```

   When the LLM term is skipped, drop it and renormalise the remaining weights to sum to 1.
5. **Adjudication gate.** Compute `conf` with the LLM term omitted, then walk five rungs in order; the first that fires skips the call and is recorded as `adjudication_skipped`:

   | Rung | Skip reason | Why it is cheaper than the band |
   |---|---|---|
   | 1 | `method_nli_only` | `verifier.method: nli` never adjudicates |
   | 2 | `empty_answer` | nothing to adjudicate |
   | 3 | `synthesizer_insufficient` | `sufficient: false` is already a re-plan trigger (§3.4); a call would buy nothing |
   | 4 | `outside_uncertainty_band` | `\|conf − 0.55\| > 0.15`, i.e. `conf ∉ [0.40, 0.70]` — a confident accept or a confident reject |
   | 5 | `budget_exhausted` | no privileged calls left |

   Only rungs 1–4 count as a *saved* call (`budget.note_saved()`). Rung 5 is a call the budget refused, not one the gate declined, and conflating the two would overstate the efficiency claim in the results table — which is why the code distinguishes them.
6. **Verdict.** `conf ≥ 0.55` → `accept`. `conf < 0.55` and re-plans remain → `revise`. `conf < 0.55` and none remain → `abstain` (the best candidate is still returned; abstention is a label, not a refusal — EM/F1 must still be computable).

**Adjudication schema:**

```json
{"supported": false, "confidence": 0.35,
 "missing_information": ["the founding year of First for Women"],
 "contradictions": [],
 "verdict": "revise"}
```

**Fallback ladder:** NLI model fails to load or errors → `method = "heuristic"`, `nli_support` replaced by token-containment overlap between `answer` and the cited evidence text (`|answer ∩ evidence| / |answer|`), same threshold. Adjudication unparseable → `llm_support = None`, renormalise, proceed. The Verifier never blocks; a Verifier that crashes would take the whole contribution of the project with it.

---

## 4. DAG execution semantics

`src/agentic_ir/orchestrator.py::_execute_plan`, plus a per-node micro-pipeline.

### 4.1 Ordering rule

**Level-synchronous Kahn topological order; within a level, ascending numeric sub-query id; strictly sequential execution.**

```
level 0 = {nodes with no unmet dependencies}, sorted by int(id[1:])
level k = {nodes whose deps are all in levels < k}, sorted by int(id[1:])
```

No threads and no async. Two reasons: (a) a single 8 GB GPU is already the bottleneck — concurrent cross-encoder and Ollama work causes VRAM thrash, not speedup; (b) sequential execution makes the trace a total order, which makes qualitative error analysis in Chapter 4 actually readable. `Plan.depth` = number of levels = the `plan_depth` agent metric.

### 4.2 Per-node micro-pipeline

```
RESOLVE ──► ROUTE ──► RETRIEVE ──► KG ──► EXTRACT
placeholders  §5 rules   tool +      bounded   answer
              (0 LLM)    rewrites    traversal ladder §4.4
                         + RRF       (0 LLM)
                         + rerank gate
```

Any stage may fail; failure sets `RetrievalResult.degraded = True` and execution continues to the next node. A node is *terminal* when it has a `RetrievalResult` (possibly degraded) or was skipped.

### 4.3 Placeholder resolution

**Syntax:** `{{<subquery_id>.<field>}}`, `field ∈ {answer, entity, title}`. Regex: `\{\{(q[1-9][0-9]*)\.(answer|entity|title)\}\}`.

**Value sources, in order:**

| field | source | secondary |
|---|---|---|
| `answer` | `state.answers[qid]` | `state.bridge_entities[qid]` |
| `entity` | `state.bridge_entities[qid]` | `state.answers[qid]` |
| `title` | `state.results[qid].passages[0].passage.title` | `state.answers[qid]` |

**Resolution fallback ladder** (each step recorded in `ToolSelection.reason` and `AgentTrace.fallback_reason`):

1. Primary source non-empty → substitute.
2. Secondary source non-empty → substitute, `degraded = True`.
3. Dependency's top-1 passage title non-empty → substitute, `degraded = True`.
4. Nothing available → **delete the placeholder token and collapse the surrounding whitespace**, leaving the residual natural-language query (`"When was  born?"` → `"When was born?"`). Ugly but retrievable; `degraded = True`, `fallback_reason = "unresolved_placeholder"`.

Never substitute an empty string without collapsing whitespace, and never abort the node — a node that produces *some* passages still contributes evidence, and the Verifier is the component entitled to judge whether that evidence is enough.

**Post-resolution** the node is re-routed from scratch (§5 features are computed on the *resolved* text). A hop-2 query is lexically very different after its bridge entity is filled in, and routing on the template would be wrong.

### 4.4 Answer-extraction ladder (§4.2 EXTRACT)

Run **only if the node has at least one dependent node** — extracting an answer nobody consumes is pure waste, and leaf answers are the Synthesizer's job. This single rule removes roughly half of all candidate extraction calls.

1. `KGResult.bridge_entity` present → use it. (0 LLM)
2. `answer_type == "entity"` → top-1 reranked passage **title**. On HotpotQA the bridge entity is a page title with high frequency, so this is both cheap and accurate. (0 LLM)
3. `answer_type in {date, number}` → regex over the top-3 passages' sentences; first match. (0 LLM)
4. Otherwise, and only if `budget.try_spend_llm()` succeeds → `extract.span.v1` with `num_predict: 48`, schema `{"answer": "...", "found": true}`. (1 LLM)
5. Budget exhausted or parse failure → top-1 passage title, `degraded = True`. (0 LLM)

Store the result in `state.answers[qid]`; store rungs 1–2 results additionally in `state.bridge_entities[qid]`.

---

## 5. Where the LLM is deliberately not used

Measured throughput on this machine is **52 tok/s** warm (`qwen3:8b` Q4_K_M, RTX 5060, 100% GPU, no offload), with a mean schema-constrained planner call at **3.75 s**. Every avoided call is still 2–5 s of real wall clock, and over 250 questions × multiple configurations that compounds into hours. These are the rules.

### 5.1 Tool routing (`agents.retriever.heuristic_shortcut: true`)

Features, computed on the **resolved** sub-query text, all `O(len(query))`, no model of any kind:

| Feature | Definition |
|---|---|
| `n_tokens` | whitespace tokens after stopword removal |
| `has_quoted` | contains a `"…"` span |
| `entity_runs` | count of maximal runs of capitalised tokens (excluding sentence-initial) |
| `entity_ratio` | tokens inside entity runs ÷ `n_tokens` |
| `oov_rate` | content terms absent from the BM25 lexicon ÷ content terms |
| `min_df` | minimum document frequency among content terms (`∞` if all OOV) |
| `is_dependent` | `len(subquery.depends_on) > 0` |

**Decision table — first match wins, every row bypasses the LLM:**

| Rule | Condition | Tool | Rationale (one line, for the report) |
|---|---|---|---|
| **R1** | `tool_hint` is set and is a valid `ToolName` | the hint | The planner already reasoned about this; re-asking pays twice for one decision |
| **R2** | `oov_rate > 0.34` | `dense_search` | ≥1/3 of content terms are absent from the lexicon; BM25 is structurally incapable |
| **R3** | `has_quoted` or (`entity_ratio ≥ 0.5` and `n_tokens ≤ 8`) | `bm25_search` | Short, proper-noun-dense queries are exact-match problems; BM25 dominates and is ~50× cheaper |
| **R4** | `min_df ≤ 3` and `entity_runs ≥ 1` | `bm25_search` | A very rare term is a near-unique key; lexical match is decisive |
| **R5** | `is_dependent` | `hybrid_search` | A bridge-filled query mixes one precise entity with paraphrased relation text; neither channel alone is sufficient |
| **R6** | `n_tokens ≥ 12` or `entity_runs == 0` | `hybrid_search` | Long, entity-sparse natural language — semantic matching carries it, lexical breaks ties |
| **R7** | default | `hybrid_search` | Safe, and empirically the best single choice |

R7 makes the table total, so **with `heuristic_shortcut: true` the Retrieval Agent issues zero LLM calls, ever.** `llm_calls_saved` increments once per routed sub-query.

**The honest comparison.** Setting `heuristic_shortcut: false` routes every sub-query with `retriever.select_tool.v1` instead. Run this on a 50-question dev slice and report three numbers: agreement rate between rule and LLM, ΔnDCG@10, and Δ wall clock. That is a genuinely interesting empirical result — "the rule agrees with the LLM N% of the time and costs 0 s" is a much stronger claim than an unmeasured assertion, and it directly serves the brief's demand for agent-specific measures. Optionally log the LLM's *shadow* decision alongside the rule on that slice, so agreement can be computed without a second full run.

### 5.2 Rerank gate (also LLM-free, and it saves GPU rather than tokens)

Apply the cross-encoder when **all** hold:

- `retrieval.rerank.enabled`
- candidate pool ≥ 10
- `rrf_margin = (s₁ − s₂) / s₁ < rerank_margin_gate` (0.15)

If the fused top-1 is already decisively ahead, reranking is 50 cross-encoder forward passes that cannot change the answer. Log `rerank_skipped` — it appears in the tool-call budget table.

### 5.3 Other LLM-free components

| Component | Mechanism |
|---|---|
| Entity linking | Alias-trie longest match (`entity_linker: alias_match`) |
| Graph traversal | Bidirectional BFS |
| Answer extraction, rungs 1–3 | KG bridge entity / passage title / regex |
| Entailment checking | DeBERTa NLI cross-encoder |
| Verdict outside `[0.40, 0.70]` | Arithmetic on the confidence blend |
| Query expansion | Rides inside the planner call (`SubQuery.rewrites`) |
| Evidence dedup and ranking | Deterministic sort |

### 5.4 Deterministic comparison shortcut

When `strategy == "comparison"` and both operand evidence sets yield a parseable year/date/number for the compared attribute, the Synthesizer's fallback answers by arithmetic — no generation. Covers a large share of both benchmarks' comparison types with perfect precision and zero latency.

### 5.5 Optional planner template shortcut (default **off**)

`agents.planner.template_shortcut: false`. When enabled, fallback rule F1's comparison template fires *before* the LLM, producing `Plan(origin="template_shortcut")` at zero cost. Kept **off** in the headline `agentic_full` configuration so the planning claim stays honest, and reported as a separate efficiency ablation. Do not quietly turn this on to make the latency table look better.

---

> **Warning about the running example.** This document uses *"Which magazine was
> started first, Arthur's Magazine or First for Women?"* throughout, including in
> the trace record below. **Neither title exists in the processed corpus** --
> verified against all 66,581 HotpotQA passages. It is a train-split question,
> and the corpus is built from the validation split.
>
> The consequence is a trap: any run of that question retrieves the wrong
> passages and looks like a retrieval or routing failure, when in fact the gold
> evidence is simply not there. The index verification probe in
> `data/indexes/hotpotqa_index_stats.json` uses it and returns "Woman's Era",
> which means nothing. Do not cite that probe as evidence of anything.
>
> Use an example from the frozen eval slice instead. Chapter 3 uses
> `5a7793275542992a6e59df04` -- "Which country is the firm that owns Babycham
> located?" (gold: Babycham s0 -> Accolade Wines s0), both present, and whose
> hop-2 gold sentence shares zero content terms with the question. Chapter 1
> uses `5a79be0e5542994f819ef084` (Taylor Swift / *Red*), also verified present.

## 6. Trace schema

One JSONL record per question, appended to `results/runs/{run_id}/traces.jsonl`. Written with `orjson`. `run_id = f"{config_name}_{dataset}_{utc_timestamp}"`.

Alongside it: `meta.json` (full config snapshot, library versions, `cache_cold`, model digest, host GPU, seed) and `metrics.csv` (one row per question, flat, for pandas).

```jsonc
{
  "schema_version": "1.0",
  "run_id": "agentic_full_hotpotqa_20260826T1830Z",
  "config_name": "agentic_full",
  "dataset": "hotpotqa",
  "qid": "5a8b57f25542995d1e6f1371",
  "seed": 42,
  "model": "qwen3:8b",

  "question": "Which magazine was started first, Arthur's Magazine or First for Women?",
  "gold": {"answer": "Arthur's Magazine",
           "supporting_facts": [["Arthur's Magazine", 0], ["First for Women", 0]],
           "level": "medium", "type": "comparison"},

  "final_answer": "Arthur's Magazine",
  "answer_sentence": "Arthur's Magazine was started before First for Women.",
  "citations": ["e3", "e7"],
  "confidence": 0.78,
  "verdict": "accept",
  "terminated_by": "verified",
  "best_cycle": 1,

  "metrics": {
    "llm_calls": 6, "llm_calls_saved": 3, "llm_cache_hits": 0,
    "tool_calls": 11, "parse_failures": 0,
    "latency_s": 34.7, "llm_latency_s": 24.1,
    "retrieval_latency_s": 6.2, "nli_latency_s": 1.9,
    "plan_depth": 1, "n_subqueries": 2, "cycles": 2,
    "replans": 1, "replanned": true,
    "citation_grounding": 1.0, "nli_support": 0.91,
    "hallucinated_citations": 0,
    "degraded_steps": 0, "budget_exhausted": false,
    "rerank_skipped": 1
  },

  "plans": [ { /* Plan, asdict, revision 0 */ }, { /* revision 1 */ } ],
  "directives": [ { /* ReplanDirective, asdict */ } ],

  "steps": [
    {"agent": "planner", "state": "PLAN", "step": 0, "cycle": 0,
     "subquery_id": null, "started_at": 0.00, "latency_s": 4.9,
     "llm_calls": [{"call_id": "c0", "prompt_id": "planner.decompose.v1",
                    "prompt_sha1": "9f1c…", "latency_s": 4.9, "parse_ok": true,
                    "retries": 0, "cache_hit": false, "think_chars": 0,
                    "truncated": false, "raw_output": "{\"strategy\":…"}],
     "tool_calls": [],
     "input_summary": {"question_chars": 71},
     "output_summary": {"strategy": "comparison", "n_subqueries": 2, "depth": 1},
     "degraded": false, "fallback_reason": null}
    /* … one entry per agent invocation, in execution order … */
  ],

  "retrieved": {
    "q1": {"query_text": "When was Arthur's Magazine started?",
           "tool": "bm25_search", "selector": "heuristic", "rule_id": "R3_entity_dense",
           "rerank_applied": true, "features": {"n_tokens": 5, "entity_ratio": 0.6,
                                                "oov_rate": 0.0, "min_df": 2},
           "passages": [{"doc_id": "hotpotqa:Arthur%27s_Magazine", "rank": 0,
                         "score": 8.41, "title": "Arthur's Magazine",
                         "component_scores": {"bm25": 14.2, "ce": 8.41}}]}
  },

  "kg": {"q1": {"seeds": ["arthurs magazine"], "linked_by": "alias_match",
                "bridge_entity": null, "n_paths": 0, "n_neighbors": 12}},

  "evidence": [{"evidence_id": "e3", "kind": "passage",
                "doc_id": "hotpotqa:Arthur%27s_Magazine", "title": "Arthur's Magazine",
                "sent_id": 0, "score": 8.41, "provenance": "rerank",
                "subquery_ids": ["q1"], "text": "Arthur's Magazine (1844-1846) was…"}],

  "verifications": [ { /* VerificationResult, asdict, one per cycle */ } ],

  "errors": []
}
```

**Metric definitions, so the report and the code agree:**

- `plan_depth` — number of DAG levels of the **selected** plan (`best_cycle`). Corpus-level: mean.
- `replan_rate` — corpus-level: `mean(replanned)`, the fraction of questions that triggered ≥1 re-plan. Also report `mean(replans)`.
- `citation_grounding` — per question, from `VerificationResult`. Corpus-level: mean over questions that produced a non-empty answer.
- `llm_calls` — **logical** calls, cache hits included. `llm_cache_hits` reported separately; latency tables come only from cold-cache runs.
- `tool_calls` — every index/graph invocation, including each rewrite variant and each rerank pass.

**Error taxonomy** for the Chapter 4 qualitative analysis, assigned post hoc by `eval/error_analysis.py` using gold supporting facts. Deterministic, first match wins:

| Label | Rule |
|---|---|
| `parse_failure` | `metrics.parse_failures > 0` and answer is wrong |
| `budget_exhausted` | `terminated_by ∈ {budget_llm, budget_wallclock, budget_iterations}` |
| `retrieval_miss` | no gold supporting fact appears in any `retrieved` passage set |
| `decomposition_error` | gold supporting facts exist in the corpus, but no sub-query's top-10 contains them, and the plan has <2 nodes on a multi-hop gold |
| `bridge_link_failure` | 2-hop gold, hop-1 evidence retrieved, `bridge_entities` empty or wrong |
| `synthesis_error` | all gold supporting facts present in `evidence`, answer still wrong |
| `verifier_false_accept` | `verdict == "accept"` and answer wrong |
| `verifier_false_reject` | a discarded cycle's candidate was correct and the selected one is not |

That last row is the one that makes the feedback loop falsifiable. Report it.

---

## 7. Module layout

```
src/agentic_ir/
  types.py                   ALL dataclasses (§1). No imports from siblings.
  config.py                  dotted-path reader over config.yaml; frozen at load
  state.py                   QuestionState, Budget, step() context manager
  trace.py                   JSONL writer, meta.json, metrics.csv
  llm.py                     Ollama client, think-stripping, JSON ladder, CallLedger
  prompts/                   seven templates; see the note on the eighth
    planner.decompose.v1.txt   planner.replan.v1.txt
    extract.span.v1.txt        kg.link_entities.v1.txt
    synth.answer.v1.txt        verifier.adjudicate.v1.txt
    repair.json.v1.txt
  agents/
    base.py  planner.py  retriever.py  kg_navigator.py  synthesizer.py  verifier.py
  tools/
    registry.py
  indexing/
    corpus.py  bm25_index.py  dense_index.py  hybrid.py  rerank.py
  kg/
    build.py  graph.py  entity_link.py  traverse.py
  baselines/
    base.py  bm25_only.py  dense_only.py  hybrid_rerank.py  naive_rag.py  self_ask.py
  eval/
    metrics.py  bootstrap.py  run_eval.py  error_analysis.py  tables.py  calibrate.py
  orchestrator.py
  cli.py
scripts/
  download_data.py  build_corpus.py  build_indexes.py  build_kg.py  sample_eval_set.py
  demo.py  dashboard.py  dashboard.html
```

> **Three deviations from the original design**, recorded rather than edited away.
>
> **1. `llm/` became `llm.py`.** The spec first proposed a package
> (`ollama_client.py`, `json_parse.py`, `registry.py`). The implementation is a single
> module providing the same surface — `OllamaClient`, `CallLedger`, the extraction
> ladder, and `get_client()`. It is implemented and tested. Prompt templates live in
> `src/agentic_ir/prompts/` accordingly.
>
> **2. `tools/search_tools.py` and `tools/kg_tools.py` do not exist.** `registry.py`
> holds the `ToolSpec` table and the callables together. Nothing was lost — the
> single-source property §3.2 relies on (registry descriptions are the LLM-facing
> tool documentation) is a property of `registry.py`, not of the split.
>
> **3. `retriever.select_tool.v1.txt` was never externalised.** Seven of the eight
> prompt ids are files; this one is an inline template in `retriever.py`
> (`_SELECT_TOOL_INSTRUCTIONS`), and `_render_select_prompt` reads the file only *if*
> it exists, falling back to the inline string. That keeps the §5.1 honest-comparison
> ablation runnable either way, but it is a real gap in §9's versioning discipline:
> an inline prompt can be edited without a `.v1 → .v2` bump, and the shipped
> configuration never issues this call anyway (`heuristic_shortcut: true`), so a
> silent edit would surface only in the ablation. Externalise it before running that
> comparison for the report.

**Import discipline:** `types.py` imports nothing from the package. `agents/*` import `types`, `llm`, `tools`, `state` — never each other, never `orchestrator`. `orchestrator` imports everything. This keeps every agent unit-testable with a stub LLM.

---

## 8. Configuration

Every key this section once listed as "to be added" is now present in `config/config.yaml`, with the values §2.6, §3.5, §5.2 and §5.5 assume: `llm.think: false`, `llm.cache` (declared, unimplemented — §3.0c), `agents.planner.template_shortcut: false`, `agents.retriever.{multi_query, max_rewrites, rerank_margin_gate}`, `agents.verifier.{max_evidence, uncertainty_band, weights}`, `orchestrator.{reserve_llm_calls, answer_selection}` and the `trace` block. Nothing here is outstanding.

### 8.1 Keys read from code but not declared in the file

Ten keys are read with an in-code default that `config/config.yaml` never states. They are not bugs — the defaults are the shipped behaviour — but they are invisible to anyone reading the config to find out what ran, and the report cannot cite a value that is not written down. Declare them, or accept that these numbers live in the source:

| Key | In-code default | Where |
|---|---|---|
| `agents.verifier.evidence_passages` | `3` | `orchestrator.py` — passages per sub-query contributing sentences to the evidence pool |
| `agents.verifier.entail_threshold` | `0.5` | `verifier.py` — `Claim.supported` |
| `agents.verifier.contradiction_threshold` | `0.7` | `verifier.py` — the contradiction screen of §3.5 step 2 |
| `agents.retriever.rerank_min_pool` | `10` | `indexing/rerank.py` — the pool floor of §5.2 |
| `agents.kg.max_seeds` | `4` | `kg_navigator.py` |
| `agents.kg.llm_link_on_empty` | `False` | `kg_navigator.py` — the §3.3 escape to `kg.link_entities.v1` |
| `retrieval.rerank.batch_size` | `32` | `indexing/rerank.py` |
| `retrieval.sparse.index_title` | `True` | `indexing/bm25_index.py` |
| `retrieval.sparse.method_variant` | `"lucene"` | `indexing/bm25_index.py` |
| `retrieval.sparse.stopwords` | `"english_plus"` | `indexing/bm25_index.py` |

Two of these are load-bearing enough to be worth a sentence in Chapter 3 wherever they are quoted: `evidence_passages: 3` bounds the evidence pool as tightly as `max_evidence: 20` does, and `entail_threshold: 0.5` decides what "supported" means in the NLI column.

### 8.2 Two config values the file's own comments contradict

`retrieval.rerank.model` is `cross-encoder/ms-marco-MiniLM-L-6-v2`; the canonical id is `cross-encoder/ms-marco-MiniLM-L6-v2` and the old one resolves only through a 307 redirect (`docs/environment-validation.md` §7). `indexing/rerank.py` already defaults to the canonical spelling, so the config is the one place still pinning the stale name. `retrieval.dense.batch_size` is still `256`, which environment validation §9 recommended lowering to `128`; the index builds completed, so this is a latent risk rather than a live one.

---

## 9. Reproducibility

- Seed `random`, `numpy` and `torch` through one function, `eval/run_eval.py::seed_everything`, which both entry points call — never a second copy.
- `PYTHONHASHSEED` is fixed at interpreter start-up, so assigning it inside a running process does nothing. `cli.py::ensure_hash_seed` therefore **re-executes the interpreter once** with `PYTHONHASHSEED=42` set (a child process on Windows, `execve` on POSIX), guarded against looping, before any package import that might sample. Every import in `cli.py` is deferred into the command that needs it for the same reason.
- **The two entry points are not equally strong, and this is worth knowing before quoting a number.** `python -m agentic_ir.cli eval …` goes through the re-exec and guarantees the hash seed. `python -m agentic_ir.eval.run_eval …` does not: `seed_everything` seeds the RNGs and then *records* whatever `PYTHONHASHSEED` happened to be in the environment (`meta.json → seeding.pythonhashseed`, `null` if unset). Prefer the `cli` form, and check that field before treating two runs as comparable.
- `llm.options.seed: 42`, `temperature: 0.0`.
- Every sort has an explicit tiebreaker: `(-score, doc_id)`, `(-weight, entity_id)`, `int(id[1:])`.
- The 250-question stratified eval sample is generated **once** by `scripts/sample_eval_set.py` with seed 42 and materialised to `data/processed/{dataset}_eval_250.jsonl`; its SHA-256 is recorded in `meta.json`. Never re-sample at run time.
- Prompt templates are versioned files; changing one requires bumping `.v1` → `.v2`. The SHA-1 in the trace makes a stale-prompt run detectable after the fact.
- `meta.json` captures library versions, GPU name, Ollama model digest, and `cache_cold`.
- The eval harness **checkpoints**: it appends to `traces.jsonl` and on restart skips qids already present (`--resume`). If question 194 does die, the run resumes at 194 rather than at 1.
- **What a clean clone does and does not carry.** `results/runs/` is gitignored, so the per-question traces that `eval/tables.py` reads are not in the repository; the generated `results/tables/*.tex` are, and so is the report that `\input{}`s them. Regenerating a table therefore means re-running the configuration behind it, not just re-running `tables`. `data/processed/` is gitignored too, but its contents are reproducible: `scripts/sample_eval_set.py` redraws the frozen 250- and 50-question slices deterministically from `project.seed` and prints their SHA-256, and `meta.json` records the SHA-256 of the slice each run actually used — so a redrawn slice can be *checked* against the one the report used rather than assumed to match.

---

## 10. Risks

1. **KG ground-truth leakage (highest severity).** 2WikiMultihopQA ships gold Wikidata evidence triples. Building the KG from those triples would let the KG Navigator read the answer key, and every KG-attributed gain in the report would be an artefact. **Mitigation, non-negotiable:** the KG is built from corpus passage text only (§3.3); gold triples are loaded exclusively by `eval/metrics.py` for scoring path quality. Enforced by `tests/test_guards.py::test_kg_build_never_opens_gold_evidence`, which asserts the builder never opens a file whose name contains `evidence` or `triple`.

2. **Total evaluation wall clock.** 9 configurations × 2 datasets × 250 questions. Estimated at ≈22–25 s per agentic question after live measurement of the LLM path; the measured medians came in above that, at **28.3 s on HotpotQA and 35.4 s on 2WikiMultihopQA**, because the estimate under-counted CPU reranking (§5.2 and the `retrieval.rerank` comment in `config.yaml`) rather than generation. One agentic config on one dataset is therefore ≈2 h, and the grid ≈14–18 h. Note the one-time cost this hides: the first call after a model load spends ~9.5 s on prompt evaluation alone (CUDA warmup), which is why `OLLAMA_KEEP_ALIVE` is not optional. **What was actually done:** the planned 150-question subsample for the ablations was not needed — every ablation that has run, ran on the full 250, so every delta in the ablation tables is a paired comparison on an identical question set. The schedule was absorbed by running the grid over several days rather than by shrinking it, because the response cache that was supposed to make re-runs free does not exist (§3.0c). At the time of writing 16 of the 18 cells are complete; `agentic_no_planner` and `agentic_no_kg` on 2WikiMultihopQA are not, and `tables.py` renders them as `--` rather than omitting the row.

3. **8 GB VRAM contention.** `qwen3:8b` (Q4 ≈ 5.2 GB) + 8192-token KV cache (≈0.6–1 GB) + bge-small (0.13 GB) + MiniLM cross-encoder (0.09 GB) + DeBERTa-v3-base NLI (≈0.7 GB) is right at the edge, and an OOM mid-run is exactly the question-194 failure we are trying to prevent. **Mitigation:** run all three HF encoders on **CPU at query time** — they process ~5 short queries and ~50 rerank pairs per question, which is a fraction of a second on CPU — and reserve the GPU entirely for Ollama. Use the GPU only for the offline corpus embedding build. Set `OLLAMA_KEEP_ALIVE=30m` so the model is not reloaded between questions.

4. **`confidence_threshold: 0.55` was a guess.** It controls the re-plan rate, which is a headline agent metric; tuning it on the eval set would be leakage. **Done, with a caveat that matters.** `src/agentic_ir/eval/calibrate.py` sweeps 0.40–0.75 on the disjoint 50-question calibration slice (`data/processed/{dataset}_calib_50.jsonl`) and writes `results/tables/calibration.tex` and `results/calibration/hotpotqa_threshold.json`. The value stayed at 0.55: the Youden-optimal point is 0.54 with a bootstrap interval of [0.45, 0.64], too wide on 50 questions to justify a change. The caveat is that the sweep is **post-hoc** — every confidence in it was produced by a single run at 0.55, and the threshold is not a read-out filter but a decision that changes the plan, the retrieval and therefore the answer. A genuine sweep needs one full run per threshold. The module's docstring and the table caption both say so; do not upgrade the claim. The sweep also reports ECE 0.332 / AUC 0.633, i.e. the confidence discriminates but is not calibrated as a probability, so the threshold is a ranking cut-point and nothing more. The blend weights were set a priori and were **not** calibrated; say that rather than implying otherwise. Only HotpotQA has been calibrated.

5. **Two re-plans may be too few to show an effect.** *(Materialised, but not in the way predicted — and this is now the report's headline finding.)* The worry was that the loop would rarely fire. It fires constantly: `replan_rate` is 0.472 on HotpotQA and 0.392 on 2WikiMultihopQA. What failed to replicate is the *effect*. Removing the verifier costs ΔF1 −0.037 [−0.065, −0.011], *p* = 0.008 on HotpotQA and ΔF1 −0.001 [−0.025, +0.025], *p* = 1.000 on 2WikiMultihopQA, where the loop ran on 98 of 250 questions and a later cycle was selected on 64 of them and still moved the aggregate by one thousandth of an F1 point. So the defensible statement is the conditional one — *the backward edge helps on HotpotQA and not on 2WikiMultihopQA* — and Chapter 4 states it that way. The mitigation still stands and is what makes that claim checkable: `best_cycle` and per-cycle verifications are in the trace schema, so conditional effectiveness is computable rather than asserted. **Do not let any document restate this as a general property of the architecture.**

6. **Local-model JSON reliability.** *(Did not materialise.)* The concern was that an 8B model at `temperature 0` would emit malformed JSON on a nontrivial minority of structured calls, and that if the Planner's rate were high the "agentic" behaviour would be partly the fallback rules. Measured over the headline run `agentic_full_hotpotqa_20260904T221533Z`: **1 parse failure in 1065 LLM calls** (0.09%), against a tolerance of 5%. The five-rung ladder and two repair retries hold. The fallback rules are therefore a safety net rather than a co-author of the results, and the report can say so with a number. `parse_failures` is still carried per question in `metrics.csv`, so the check is repeatable on any later run.

7. **Seeded ≠ bit-reproducible, and nothing rescues it.** Ollama with a fixed seed is not guaranteed bit-identical across GPU batching or driver versions. The planned mitigation was the response cache, which would have made a *completed* run exactly replayable — and it does not exist (§3.0c). What remains is provenance, not reproduction: `meta.json` records the model digest, the config snapshot, the git commit, the platform and the GPU, and `traces.jsonl` records every call, so a divergent re-run can be **diagnosed** but not prevented. The deterministic parts of the system — retrieval, fusion, routing, evidence ranking, the state machine — are reproducible by construction (§9); the generative parts are not. State exactly that in Chapter 4.
