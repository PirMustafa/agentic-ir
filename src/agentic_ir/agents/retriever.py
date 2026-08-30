"""Retrieval Agent: route, issue, fuse, gate (architecture sections 3.2 and 5.1).

With ``agents.retriever.heuristic_shortcut: true`` -- the shipped configuration
-- this agent issues **zero LLM calls, ever**. The seven-rule decision table is
total (R7 is the default), every feature is ``O(len(query))``, and each routed
sub-query increments ``llm_calls_saved``. At 52 tok/s a saved call is 2-5 s of
wall clock, and over 250 questions x 9 configurations that is the difference
between a schedule and an excuse.

Setting ``heuristic_shortcut: false`` routes with the LLM instead, so the
report can state the rule/LLM agreement rate rather than assert it. The routing
prompt is assembled from the tool registry's own descriptions, so the
documentation the model reads is the documentation of the code that runs.

Query expansion rides on ``SubQuery.rewrites``, which arrived inside the
planner call: multi-query retrieval here costs zero additional LLM calls.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..config import Config
from ..indexing.hybrid import fuse_results
from ..state import QuestionState, StepRecorder
from ..tools.registry import ToolRegistry
from ..types import (
    RetrievalResult,
    ScoredPassage,
    SubQuery,
    ToolCallTrace,
    ToolName,
    ToolSelection,
)
from .base import BaseAgent, content_tokens, entity_runs, word_tokens

__all__ = [
    "RoutingFeatures",
    "RetrievalAgent",
    "SpanExtractor",
    "SELECT_TOOL_SCHEMA",
    "route",
]

#: ``min_df`` is infinite when every content term is out of vocabulary. JSON has
#: no infinity, so the trace carries this sentinel instead.
INF_DF = 1e9

_QUOTED = re.compile(r"[\"“‘']([^\"”’']{2,})[\"”’']")

SELECT_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool": {"type": "string", "enum": ["bm25_search", "dense_search", "hybrid_search"]},
        "rerank": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["tool", "rerank"],
}

_SELECT_TOOL_INSTRUCTIONS = (
    "Choose the single best retrieval tool for the sub-query below.\n\n"
    "Tools:\n$tools\n\n"
    "Sub-query: $query\n"
    "Depends on an earlier sub-query: $is_dependent\n\n"
    'Answer with JSON only: {"tool": "hybrid_search", "rerank": true, "reason": "one line"}'
)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RoutingFeatures:
    """The section 5.1 feature vector, computed on the RESOLVED query text.

    Resolved, not templated: a hop-2 query is lexically a different question
    once its bridge entity is filled in, and routing on the template would
    measure the placeholder rather than the query.
    """

    n_tokens: int
    has_quoted: bool
    entity_runs: int
    entity_ratio: float
    oov_rate: float
    min_df: float
    is_dependent: bool

    def as_dict(self) -> dict[str, float]:
        return {
            "n_tokens": float(self.n_tokens),
            "has_quoted": float(self.has_quoted),
            "entity_runs": float(self.entity_runs),
            "entity_ratio": round(self.entity_ratio, 4),
            "oov_rate": round(self.oov_rate, 4),
            "min_df": INF_DF if math.isinf(self.min_df) else float(self.min_df),
            "is_dependent": float(self.is_dependent),
        }


def extract_features(
    text: str,
    *,
    is_dependent: bool = False,
    bm25: Any = None,
) -> RoutingFeatures:
    """Compute the routing features. Lexicon statistics degrade to neutral.

    A missing BM25 index yields ``oov_rate=0`` and ``min_df=inf``: R2 and R4
    then cannot fire, and the query falls through to a hybrid default rather
    than to a decision made on numbers nobody computed.
    """
    runs = entity_runs(text)
    content = content_tokens(text)
    n_tokens = len(content)
    entity_tokens = sum(len(word_tokens(run)) for run in runs)
    oov_rate = 0.0
    min_df = math.inf
    if bm25 is not None:
        try:
            oov_rate = float(bm25.oov_rate(text))
            min_df = float(bm25.min_df(text))
        except Exception:  # noqa: BLE001 - a feature probe never fails a question
            oov_rate, min_df = 0.0, math.inf
    return RoutingFeatures(
        n_tokens=n_tokens,
        has_quoted=bool(_QUOTED.search(text)),
        entity_runs=len(runs),
        entity_ratio=(entity_tokens / n_tokens) if n_tokens else 0.0,
        oov_rate=oov_rate,
        min_df=min_df,
        is_dependent=is_dependent,
    )


# ---------------------------------------------------------------------------
# The decision table (section 5.1)
# ---------------------------------------------------------------------------

def route(
    features: RoutingFeatures,
    *,
    tool_hint: ToolName | None = None,
    available: Sequence[str] = ("bm25_search", "dense_search", "hybrid_search"),
) -> tuple[ToolName, str, str, str]:
    """First match wins. Returns ``(tool, rule_id, selector, reason)``.

    Every row bypasses the LLM, and R7 makes the table total -- which is the
    property that lets the shipped configuration claim zero routing calls
    rather than "usually zero".
    """
    if tool_hint in ("bm25_search", "dense_search", "hybrid_search") and tool_hint in available:
        return tool_hint, "R1_planner_hint", "planner_hint", (
            "the planner already reasoned about this; re-asking pays twice for one decision"
        )
    if features.oov_rate > 0.34:
        return _pick("dense_search", available), "R2_oov", "heuristic", (
            f"oov_rate={features.oov_rate:.2f} > 0.34: BM25 is structurally incapable"
        )
    if features.has_quoted or (features.entity_ratio >= 0.5 and features.n_tokens <= 8):
        return _pick("bm25_search", available), "R3_entity_dense", "heuristic", (
            "short, proper-noun-dense or quoted: an exact-match problem"
        )
    if features.min_df <= 3 and features.entity_runs >= 1:
        return _pick("bm25_search", available), "R4_rare_term", "heuristic", (
            f"min_df={features.min_df:.0f} <= 3: a near-unique key, lexical match is decisive"
        )
    if features.is_dependent:
        return _pick("hybrid_search", available), "R5_dependent", "heuristic", (
            "bridge-filled query: one precise entity plus paraphrased relation text"
        )
    if features.n_tokens >= 12 or features.entity_runs == 0:
        return _pick("hybrid_search", available), "R6_long_or_entity_sparse", "heuristic", (
            "long or entity-sparse natural language: semantics carries, lexical breaks ties"
        )
    return _pick("hybrid_search", available), "R7_default", "heuristic", (
        "no rule fired; hybrid is the best single choice"
    )


def _pick(preferred: ToolName, available: Sequence[str]) -> ToolName:
    """The routed tool, or the closest channel this deployment actually has."""
    if preferred in available:
        return preferred
    for candidate in ("hybrid_search", "bm25_search", "dense_search"):
        if candidate in available:
            return candidate  # type: ignore[return-value]
    return preferred


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

class RetrievalAgent(BaseAgent):
    """Routes one sub-query, issues it (plus its rewrites), fuses, and gates."""

    name = "retriever"

    def __init__(
        self,
        registry: ToolRegistry,
        cfg: Config | None = None,
        *,
        bm25: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(cfg, **kwargs)
        self.registry = registry
        self.bm25 = bm25
        self.heuristic_shortcut = bool(self.cfg.get("agents.retriever.heuristic_shortcut", True))
        self.multi_query = bool(self.cfg.get("agents.retriever.multi_query", True))
        self.max_rewrites = int(self.cfg.get("agents.retriever.max_rewrites", 2))
        self.top_k = int(self.cfg.get("retrieval.top_k", 10))
        self.candidate_k = int(self.cfg.get("retrieval.rerank.top_n", 50))
        self.rrf_k = int(self.cfg.get("retrieval.hybrid.rrf_k", 60))
        self.rerank_enabled = bool(self.cfg.get("retrieval.rerank.enabled", True))

    # -- public API --------------------------------------------------------
    def run(
        self,
        state: QuestionState,
        subquery: SubQuery,
        *,
        query_text: str | None = None,
        degraded: bool = False,
        degraded_reason: str = "",
    ) -> RetrievalResult:
        """Retrieve for one sub-query. Never raises; may return zero passages.

        ``degraded``/``degraded_reason`` carry the placeholder-resolution
        verdict in from the orchestrator, so one flag on the result answers
        "was this query even well formed?".
        """
        text = query_text if query_text is not None else subquery.text
        result: RetrievalResult | None = None
        with state.step(
            self.name,
            subquery_id=subquery.id,
            query_chars=len(text),
            resolved=query_text is not None,
        ) as rec:
            result = self._retrieve(state, rec, subquery, text, degraded, degraded_reason)
        if result is None:  # step() swallowed something unexpected
            result = _empty_result(subquery, text, "agent_error")
            state.results.setdefault(subquery.id, result)
        return result

    # -- internals ---------------------------------------------------------
    def _retrieve(
        self,
        state: QuestionState,
        rec: StepRecorder,
        subquery: SubQuery,
        text: str,
        degraded: bool,
        degraded_reason: str,
    ) -> RetrievalResult:
        started = time.perf_counter()
        features = extract_features(
            text, is_dependent=bool(subquery.depends_on), bm25=self.bm25
        )
        available = self.registry.search_names()
        if not available:
            rec.degrade("no_search_tool")
            return _empty_result(subquery, text, "no_search_tool", features=features)

        selection = self._select(state, rec, subquery, text, features, available)
        queries = self._queries(text, subquery)

        pool, tool_traces, error = self._issue(state, selection.tool, queries)
        if error and selection.tool != "hybrid_search" and "hybrid_search" in available:
            # Section 3.2's fallback: hybrid+rerank is unconditionally safe.
            selection = ToolSelection(
                tool="hybrid_search", selector="fallback", rule_id="R6_fallback",
                rerank_applied=False, reason=f"{selection.tool} failed: {error}",
                features=selection.features,
            )
            pool, retry_traces, error = self._issue(state, "hybrid_search", queries)
            tool_traces.extend(retry_traces)
        for trace in tool_traces:
            rec.note_tool(trace)

        n_candidates = len(pool)
        pool, reranked, rerank_skipped = self._rerank(state, rec, text, pool)
        features_out = dict(selection.features)
        features_out["rerank_skipped"] = float(rerank_skipped)
        features_out["n_queries"] = float(len(queries))
        selection = ToolSelection(
            tool=selection.tool,
            selector=selection.selector,
            rule_id=selection.rule_id,
            rerank_applied=reranked,
            reason=_join_reasons(selection.reason, degraded_reason),
            features=features_out,
        )

        passages = tuple(
            ScoredPassage(
                passage=sp.passage,
                score=sp.score,
                rank=rank,
                provenance=sp.provenance,
                component_scores=dict(sp.component_scores),
            )
            for rank, sp in enumerate(pool[: self.top_k])
        )
        result = RetrievalResult(
            subquery_id=subquery.id,
            query_text=text,
            selection=selection,
            passages=passages,
            queries_issued=tuple(queries),
            n_candidates=n_candidates,
            latency_s=round(time.perf_counter() - started, 4),
            degraded=bool(degraded or error or not passages),
            error=error,
        )
        if result.degraded and not rec.degraded:
            rec.degrade(degraded_reason or error or "empty_retrieval")
        rec.output_summary = {
            "tool": selection.tool,
            "rule_id": selection.rule_id,
            "selector": selection.selector,
            "n_passages": len(passages),
            "n_candidates": n_candidates,
            "rerank_applied": reranked,
            "top_title": passages[0].passage.title if passages else None,
        }
        state.results[subquery.id] = result
        return result

    def _select(
        self,
        state: QuestionState,
        rec: StepRecorder,
        subquery: SubQuery,
        text: str,
        features: RoutingFeatures,
        available: Sequence[str],
    ) -> ToolSelection:
        """Heuristic table, or -- when configured -- one LLM call per sub-query."""
        tool, rule_id, selector, reason = route(
            features, tool_hint=subquery.tool_hint, available=available
        )
        if self.heuristic_shortcut or selector == "planner_hint":
            state.budget.note_saved()
            return ToolSelection(
                tool=tool, selector=selector, rule_id=rule_id, rerank_applied=False,
                reason=reason, features=features.as_dict(),
            )

        prompt = _render_select_prompt(self.registry, text, features)
        call = self.call_json_prompt(
            state, rec, prompt_id="retriever.select_tool.v1", prompt=prompt,
            schema=SELECT_TOOL_SCHEMA, purpose="select_tool", privileged=False, repair=False,
            num_predict=64,
        )
        chosen = str((call.parsed or {}).get("tool") or "")
        if call.ok and chosen in available:
            return ToolSelection(
                tool=chosen,  # type: ignore[arg-type]
                selector="llm",
                rule_id=None,
                rerank_applied=False,
                reason=str((call.parsed or {}).get("reason") or "")[:200],
                features=features.as_dict(),
            )
        return ToolSelection(
            tool=tool, selector="fallback", rule_id=rule_id, rerank_applied=False,
            reason=f"llm routing unusable ({call.reason or chosen or 'empty'}); {reason}",
            features=features.as_dict(),
        )

    def _queries(self, text: str, subquery: SubQuery) -> list[str]:
        """The query plus its rewrites -- expansion that cost no extra call."""
        queries = [text]
        if self.multi_query and self.max_rewrites > 0:
            for rewrite in subquery.rewrites[: self.max_rewrites]:
                cleaned = rewrite.strip()
                if cleaned and cleaned.lower() != text.lower() and cleaned not in queries:
                    queries.append(cleaned)
        return queries

    def _issue(
        self,
        state: QuestionState,
        tool: ToolName,
        queries: Sequence[str],
    ) -> tuple[list[ScoredPassage], list[ToolCallTrace], str | None]:
        """Run one tool over every query variant and fuse the results by RRF."""
        results: list[list[ScoredPassage]] = []
        traces: list[ToolCallTrace] = []
        error: str | None = None
        for index, query in enumerate(queries):
            started = time.perf_counter()
            try:
                hits = list(self.registry.call(tool, query, self.candidate_k))
                ok, message = True, None
            except Exception as exc:  # noqa: BLE001 - axiom 2
                hits, ok, message = [], False, f"{type(exc).__name__}: {exc}"
                error = error or message
            state.budget.note_tool()
            traces.append(
                ToolCallTrace(
                    call_id=f"t{index}", agent=self.name, tool=tool, query=query,
                    n_results=len(hits), latency_s=round(time.perf_counter() - started, 4),
                    ok=ok, error=message,
                )
            )
            if hits:
                results.append(hits)
        if not results:
            return [], traces, error
        if len(results) == 1:
            return results[0], traces, error
        return fuse_results(results, rrf_k=self.rrf_k), traces, error

    def _rerank(
        self,
        state: QuestionState,
        rec: StepRecorder,
        query: str,
        pool: Sequence[ScoredPassage],
    ) -> tuple[list[ScoredPassage], bool, bool]:
        """Apply the gated cross-encoder. Returns ``(pool, ran, skipped)``.

        A gate that closes is not a failure: 50 forward passes that cannot
        change the answer are GPU time spent on nothing, and ``rerank_skipped``
        is a reported number in the tool-call budget table.
        """
        if not pool or not self.rerank_enabled or "rerank" not in self.registry:
            # No reranker configured is not a gate decision, so it is not a skip:
            # `rerank_skipped` must keep meaning "the margin was already decisive".
            return list(pool), False, False
        started = time.perf_counter()
        try:
            outcome = self.registry.call("rerank", query, list(pool))
            passages = list(getattr(outcome, "passages", pool))
            ran = bool(getattr(outcome, "ran", False))
        except Exception as exc:  # noqa: BLE001 - reranking is an optimisation
            rec.note_tool(
                ToolCallTrace(
                    call_id="t_rerank", agent=self.name, tool="rerank", query=query,
                    n_results=len(pool), latency_s=round(time.perf_counter() - started, 4),
                    ok=False, error=f"{type(exc).__name__}: {exc}",
                )
            )
            return list(pool), False, True
        if ran:
            state.budget.note_tool()
            rec.note_tool(
                ToolCallTrace(
                    call_id="t_rerank", agent=self.name, tool="rerank", query=query,
                    n_results=len(passages), latency_s=round(time.perf_counter() - started, 4),
                    ok=True,
                )
            )
        return passages, ran, not ran


def _join_reasons(*parts: str) -> str:
    return "; ".join(p for p in parts if p)


def _render_select_prompt(registry: ToolRegistry, query: str, features: RoutingFeatures) -> str:
    """Build the routing prompt from the registry's own tool documentation.

    Uses ``retriever.select_tool.v1`` when that file exists and the inline
    template otherwise, so the honest-comparison ablation runs whether or not
    the prompt has been externalised yet.
    """
    from string import Template

    from .base import PROMPT_DIR

    path = PROMPT_DIR / "retriever.select_tool.v1.txt"
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            template = fh.read()
    else:
        template = _SELECT_TOOL_INSTRUCTIONS
    return Template(template).safe_substitute(
        tools=registry.describe(registry.search_names()),
        query=query,
        is_dependent="yes" if features.is_dependent else "no",
    )


def _empty_result(
    subquery: SubQuery,
    text: str,
    error: str,
    *,
    features: RoutingFeatures | None = None,
) -> RetrievalResult:
    """A legal, non-fatal zero-passage result. The pipeline continues."""
    return RetrievalResult(
        subquery_id=subquery.id,
        query_text=text,
        selection=ToolSelection(
            tool="hybrid_search", selector="fallback", rule_id=None, rerank_applied=False,
            reason=error, features=features.as_dict() if features else {},
        ),
        passages=(),
        queries_issued=(text,),
        n_candidates=0,
        degraded=True,
        error=error,
    )


# ---------------------------------------------------------------------------
# Answer extraction (section 4.4, rung 4)
# ---------------------------------------------------------------------------

EXTRACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "found": {"type": "boolean"}},
    "required": ["answer", "found"],
}


class SpanExtractor(BaseAgent):
    """Rung 4 of the extraction ladder: one small, non-privileged LLM call.

    Rungs 1-3 are deterministic and live in the orchestrator, which is what
    knows about KG results and passage titles. Only the residual case reaches
    here, and section 4.4's "non-leaf nodes only" rule means it is roughly half
    of what a naive implementation would spend.
    """

    name = "extractor"

    def run(
        self,
        state: QuestionState,
        rec: StepRecorder,
        *,
        question: str,
        passages: Sequence[ScoredPassage],
        answer_type: str = "string",
        max_passages: int = 3,
    ) -> str:
        """The extracted span, or ``""``. Called inside an open step."""
        if not passages:
            return ""
        rendered = "\n".join(
            f"[{i + 1}] {sp.passage.title}: {sp.passage.text[:600]}"
            for i, sp in enumerate(passages[:max_passages])
        )
        call = self.call_json(
            state,
            rec,
            prompt_id="extract.span.v1",
            variables={"question": question, "answer_type": answer_type, "passages": rendered},
            schema=EXTRACT_SCHEMA,
            purpose="extract_span",
            privileged=False,
            repair=False,
            num_predict=48,
        )
        parsed: Mapping[str, Any] = call.parsed or {}
        if not call.ok or not parsed.get("found"):
            return ""
        return str(parsed.get("answer") or "").strip()[:200]
