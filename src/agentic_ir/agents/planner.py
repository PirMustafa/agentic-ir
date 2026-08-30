"""Planner: question in, validated sub-query DAG out (architecture section 3.1).

Exactly one LLM call, schema-constrained. Everything after it is deterministic:
the repair table, the derived ``strategy``, combiner detection, and a
three-rung fallback ladder whose floor (F3) is an identity plan routed to
hybrid+rerank -- i.e. exactly the ``hybrid_rerank`` baseline. That floor is
axiom 3: the agentic system cannot score below its strongest non-agentic
baseline because of infrastructure failure, only because of judgement.

Two measured facts shape this module:

* the model builds the graph well and names it badly -- structurally correct
  decompositions 8/8, ``strategy`` labelled ``single_hop`` on 7 of 8 including
  obvious bridge and comparison cases. So ``strategy`` is **derived from the
  DAG** and the model's label is kept only as ``strategy_llm``, where its
  disagreement rate is reportable rather than silently load-bearing;
* it reliably emits a final node that restates the question and depends on all
  the others. Retrieving for that node is wasted work, so it is marked
  ``is_combiner`` and the orchestrator skips retrieval for it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ..config import Config
from ..state import QuestionState, jaccard, normalise_text
from ..types import (
    SUBQUERY_ID_RE,
    AnswerType,
    Intent,
    Plan,
    ReplanDirective,
    Strategy,
    SubQuery,
    ToolName,
)
from .base import (
    BaseAgent,
    collapse_whitespace,
    entity_runs,
    placeholders_of,
    strip_placeholder,
)

__all__ = [
    "PLAN_SCHEMA",
    "Planner",
    "assign_hops",
    "comparison_template",
    "derive_strategy",
    "fallback_plan",
    "mark_combiners",
    "validate_subqueries",
]

_INTENTS: tuple[Intent, ...] = ("lookup", "attribute", "comparison", "bridge", "temporal", "yesno")
_ANSWER_TYPES: tuple[AnswerType, ...] = ("entity", "date", "number", "yesno", "string")
_TOOL_HINTS: tuple[ToolName, ...] = ("bm25_search", "dense_search", "hybrid_search")

#: Jaccard against the original question above which a node that depends on
#: every other node is a combiner rather than a real retrieval target.
COMBINER_JACCARD = 0.6

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "strategy": {
            "type": "string",
            "enum": ["single_hop", "bridge", "comparison", "attribute", "bridge_comparison"],
        },
        "subqueries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                    "intent": {"type": "string", "enum": list(_INTENTS)},
                    "entities": {"type": "array", "items": {"type": "string"}},
                    "answer_type": {"type": "string", "enum": list(_ANSWER_TYPES)},
                    "tool_hint": {"type": "string", "enum": [*_TOOL_HINTS, ""]},
                    "rewrites": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "text", "depends_on"],
            },
        },
    },
    "required": ["strategy", "subqueries"],
}

# F1's markers. Deliberately small and fixed: a template that fires on the
# wrong question produces a confidently wrong plan, which is worse than the
# identity floor.
_COMPARISON_PATTERNS = (
    re.compile(r"\bwhich\b.+\b(first|earlier|older|newer|later|longer|shorter|bigger|smaller)\b", re.I),
    re.compile(r"\b(are|were|is|was|do|does|did)\b.+\b(both|the same)\b", re.I),
    re.compile(r"\bwho\b.+\bborn\b.+\b(first|earlier|later)\b", re.I),
    re.compile(r"\b(who|which|what)\b.+\b\w+\s+or\s+\w+", re.I),
)
_TEMPORAL_MARKER = re.compile(
    r"\b(first|earlier|later|older|newer|born|founded|started|released|died)\b", re.I
)
_YESNO_MARKER = re.compile(r"\b(both|the same)\b", re.I)


# ---------------------------------------------------------------------------
# Validation (architecture 3.1, the repair table)
# ---------------------------------------------------------------------------

def _as_list(value: Any) -> list[str]:
    """Coerce a JSON field into a list of strings; a bare string is one item."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()]


def _dedup(items: Iterable[str]) -> list[str]:
    """Order-preserving dedup. Determinism by construction (axiom 5)."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _node_number(sq_id: str) -> int:
    digits = sq_id[1:]
    return int(digits) if digits.isdigit() else 10**6


def validate_subqueries(
    raw: Sequence[Mapping[str, Any]],
    *,
    max_subqueries: int = 5,
    max_rewrites: int = 2,
) -> tuple[list[SubQuery], list[str]]:
    """Apply the repair table, returning ``(subqueries, repairs)``.

    Every repair is recorded rather than silently applied: the fraction of
    plans that needed one is a direct measurement of how far an 8B model is
    from usable structured output, which Chapter 4 has to report honestly.
    """
    repairs: list[str] = []
    nodes: list[dict[str, Any]] = []

    # -- coerce, and drop nodes with no usable text ------------------------
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            repairs.append(f"dropped_malformed:{index}")
            continue
        text = collapse_whitespace(str(item.get("text") or ""))
        original_id = str(item.get("id") or "").strip()
        if len(text) < 3:
            repairs.append(f"dropped_empty:{original_id or index}")
            continue
        nodes.append(
            {
                "old_id": original_id or f"#{index}",
                "text": text,
                "depends_on": _dedup(_as_list(item.get("depends_on"))),
                "intent": item.get("intent"),
                "entities": _as_list(item.get("entities"))[:8],
                "answer_type": item.get("answer_type"),
                "tool_hint": item.get("tool_hint"),
                "rewrites": _as_list(item.get("rewrites")),
            }
        )

    if not nodes:
        return [], repairs

    # -- cap the plan size, before anything expensive ----------------------
    if len(nodes) > max_subqueries:
        repairs.append(f"truncated_subqueries:{len(nodes) - max_subqueries}")
        nodes = nodes[:max_subqueries]

    # -- ids: renumber positionally when any is malformed or duplicated ----
    old_ids = [n["old_id"] for n in nodes]
    valid = all(SUBQUERY_ID_RE.match(i) for i in old_ids) and len(set(old_ids)) == len(old_ids)
    id_map: dict[str, str] = {}
    if valid:
        for node in nodes:
            node["id"] = node["old_id"]
            id_map[node["old_id"]] = node["old_id"]
    else:
        for position, node in enumerate(nodes, start=1):
            new_id = f"q{position}"
            node["id"] = new_id
            # First occurrence wins, so a duplicated id resolves deterministically.
            id_map.setdefault(node["old_id"], new_id)
            if node["old_id"] != new_id:
                repairs.append(f"renumbered:{node['old_id']}->{new_id}")

    known = {node["id"] for node in nodes}

    # -- placeholders written against the model's own ids ------------------
    # PLACEHOLDER_RE only recognises canonical ``qN`` ids, so a model that
    # numbered its nodes "one"/"two" would leave a literal "{{one.answer}}" in
    # the query text -- invisible to both the resolver and the stripper, and
    # sent verbatim to the index. Canonicalise here, while the mapping exists.
    for node in nodes:
        node["text"] = _canonical_placeholders(node, id_map, known, repairs)

    # -- dependencies: remap, then drop the ones that name nothing ---------
    for node in nodes:
        kept: list[str] = []
        for dep in node["depends_on"]:
            target = id_map.get(dep, dep)
            if target == node["id"]:
                repairs.append(f"dropped_dep:{node['id']}->{dep}")
                continue
            if target not in known:
                repairs.append(f"dropped_dep:{node['id']}->{dep}")
                continue
            kept.append(target)
        node["depends_on"] = _dedup(kept)

    # -- placeholders imply dependencies -----------------------------------
    for node in nodes:
        for qid, field in placeholders_of(node["text"]):
            target = id_map.get(qid, qid)
            if target == node["id"] or target not in known:
                node["text"] = strip_placeholder(node["text"], qid, field)
                repairs.append(f"stripped_placeholder:{node['id']}:{qid}.{field}")
            elif target not in node["depends_on"]:
                node["depends_on"].append(target)
                repairs.append(f"added_dep:{node['id']}->{target}")

    _break_cycles(nodes, repairs)

    # -- a placeholder whose edge did not survive is dead weight -----------
    for node in nodes:
        for qid, field in placeholders_of(node["text"]):
            if qid not in node["depends_on"]:
                node["text"] = strip_placeholder(node["text"], qid, field)
                repairs.append(f"stripped_placeholder:{node['id']}:{qid}.{field}")

    # -- field-level repairs -----------------------------------------------
    out: list[SubQuery] = []
    for node in nodes:
        intent = node["intent"] if node["intent"] in _INTENTS else "lookup"
        if node["intent"] not in (None, "", intent):
            repairs.append(f"bad_intent:{node['id']}:{node['intent']}")
        answer_type = node["answer_type"] if node["answer_type"] in _ANSWER_TYPES else "string"
        if node["answer_type"] not in (None, "", answer_type):
            repairs.append(f"bad_answer_type:{node['id']}:{node['answer_type']}")
        hint = node["tool_hint"] if node["tool_hint"] in _TOOL_HINTS else None
        if node["tool_hint"] not in (None, "", hint):
            repairs.append(f"bad_tool_hint:{node['id']}:{node['tool_hint']}")
        rewrites = [r for r in node["rewrites"] if len(r) >= 3]
        if len(rewrites) > max_rewrites:
            repairs.append(f"truncated_rewrites:{node['id']}")
            rewrites = rewrites[:max_rewrites]
        if len(node["text"]) < 3:  # a placeholder strip can empty a node
            repairs.append(f"dropped_empty:{node['id']}")
            continue
        out.append(
            SubQuery(
                id=node["id"],
                text=node["text"],
                depends_on=tuple(node["depends_on"]),
                intent=intent,
                entities=tuple(node["entities"]),
                answer_type=answer_type,
                tool_hint=hint,
                rewrites=tuple(rewrites),
            )
        )

    # Dropping a node above can orphan an edge into it.
    survivors = {sq.id for sq in out}
    cleaned: list[SubQuery] = []
    for sq in out:
        deps = tuple(d for d in sq.depends_on if d in survivors)
        if deps != sq.depends_on:
            for dep in sq.depends_on:
                if dep not in survivors:
                    repairs.append(f"dropped_dep:{sq.id}->{dep}")
            sq = SubQuery(**{**_as_kwargs(sq), "depends_on": deps})
        cleaned.append(sq)
    return assign_hops(cleaned), repairs


_ANY_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_\-]+)\s*\.\s*(answer|entity|title)\s*\}\}")


def _canonical_placeholders(
    node: Mapping[str, Any],
    id_map: Mapping[str, str],
    known: set[str],
    repairs: list[str],
) -> str:
    """Rewrite ``{{X.field}}`` to the canonical id, or delete it."""

    def rewrite(match: re.Match[str]) -> str:
        raw, field = match.group(1), match.group(2)
        target = id_map.get(raw, raw if raw in known else "")
        if not target or target == node["id"] or target not in known:
            repairs.append(f"stripped_placeholder:{node['id']}:{raw}.{field}")
            return " "
        if target != raw:
            repairs.append(f"remapped_placeholder:{node['id']}:{raw}->{target}")
        return f"{{{{{target}.{field}}}}}"

    return collapse_whitespace(_ANY_PLACEHOLDER.sub(rewrite, node["text"]))


def _as_kwargs(sq: SubQuery) -> dict[str, Any]:
    """``SubQuery`` fields as a dict; ``slots=True`` rules out ``__dict__``."""
    return {
        "id": sq.id,
        "text": sq.text,
        "depends_on": sq.depends_on,
        "hop": sq.hop,
        "intent": sq.intent,
        "entities": sq.entities,
        "answer_type": sq.answer_type,
        "tool_hint": sq.tool_hint,
        "rewrites": sq.rewrites,
        "is_combiner": sq.is_combiner,
    }


def _break_cycles(nodes: list[dict[str, Any]], repairs: list[str]) -> None:
    """Kahn, then drop one back edge and re-run, until the graph is acyclic.

    The back edge is the one whose *target* id is numerically higher: the model
    writes plans top-down, so an edge pointing forwards is the mistake, and
    dropping it preserves the intended reading order. One edge at a time, so a
    single bad edge cannot flatten an otherwise good plan.
    """
    by_id = {node["id"]: node for node in nodes}
    while True:
        placed: set[str] = set()
        remaining = set(by_id)
        progressed = True
        while progressed and remaining:
            progressed = False
            for node_id in sorted(remaining, key=_node_number):
                deps = by_id[node_id]["depends_on"]
                if all(d in placed for d in deps):
                    placed.add(node_id)
                    remaining.discard(node_id)
                    progressed = True
        if not remaining:
            return
        candidates = [
            (node_id, dep)
            for node_id in sorted(remaining, key=_node_number)
            for dep in by_id[node_id]["depends_on"]
            if dep in remaining and _node_number(dep) > _node_number(node_id)
        ]
        if not candidates:  # a cycle with no forward edge: cut the lowest node free
            node_id = min(remaining, key=_node_number)
            for dep in list(by_id[node_id]["depends_on"]):
                if dep in remaining:
                    by_id[node_id]["depends_on"].remove(dep)
                    repairs.append(f"cycle_broken:{node_id}->{dep}")
            continue
        node_id, dep = min(candidates, key=lambda e: (_node_number(e[0]), _node_number(e[1])))
        by_id[node_id]["depends_on"].remove(dep)
        repairs.append(f"cycle_broken:{node_id}->{dep}")


def assign_hops(subqueries: Sequence[SubQuery]) -> list[SubQuery]:
    """Set ``hop`` = 1 + longest path to a root. Assumes an acyclic DAG."""
    by_id = {sq.id: sq for sq in subqueries}
    hops: dict[str, int] = {}

    def hop_of(sq_id: str, seen: frozenset[str] = frozenset()) -> int:
        if sq_id in hops:
            return hops[sq_id]
        if sq_id in seen or sq_id not in by_id:  # defensive: repaired plans are acyclic
            return 1
        deps = [d for d in by_id[sq_id].depends_on if d in by_id]
        value = 1 if not deps else 1 + max(hop_of(d, seen | {sq_id}) for d in deps)
        hops[sq_id] = value
        return value

    return [
        SubQuery(**{**_as_kwargs(sq), "hop": hop_of(sq.id)})
        for sq in sorted(subqueries, key=lambda s: _node_number(s.id))
    ]


# ---------------------------------------------------------------------------
# Derived strategy and combiner detection
# ---------------------------------------------------------------------------

def derive_strategy(subqueries: Sequence[SubQuery]) -> Strategy:
    """Infer ``strategy`` from the DAG shape, overwriting whatever the model said.

    First match wins. ``bridge`` is tested before ``attribute`` because the
    spec's own worked example is a two-node chain whose dependent node has
    ``intent="attribute"`` and is labelled ``bridge`` -- what makes it a bridge
    is that a value flows along the edge, which is exactly what a placeholder
    is. An ``attribute`` plan is therefore the residual case: a dependent node
    that asks for a property without consuming its dependency's answer.
    """
    if not subqueries:
        return "single_hop"
    if len(subqueries) == 1:
        return "single_hop"
    roots = [sq for sq in subqueries if not sq.depends_on]
    dependents = [sq for sq in subqueries if sq.depends_on]
    root_ids = {sq.id for sq in roots}
    depth = max(sq.hop for sq in subqueries)

    if len(roots) >= 2 and depth >= 3:
        return "bridge_comparison"
    if len(roots) >= 2 and any(root_ids <= set(sq.depends_on) for sq in dependents):
        return "comparison"
    if len(roots) >= 2 and not dependents:
        return "comparison"
    if any(sq.is_template() for sq in dependents):
        return "bridge"
    if any(sq.intent == "attribute" for sq in dependents):
        return "attribute"
    if dependents:
        return "bridge"
    return "single_hop"


def mark_combiners(subqueries: Sequence[SubQuery], question: str) -> list[SubQuery]:
    """Flag nodes that restate the question and depend on every other node.

    Such a node is answered by composing its dependencies, not by a passage, so
    retrieving for it is wasted work -- a third of all retrieval on a three-node
    comparison plan.
    """
    if len(subqueries) < 2:
        return list(subqueries)
    question_terms = normalise_text(question)
    ids = {sq.id for sq in subqueries}
    out: list[SubQuery] = []
    for sq in subqueries:
        others = ids - {sq.id}
        is_combiner = bool(others) and others <= set(sq.depends_on)
        if is_combiner:
            is_combiner = jaccard(normalise_text(sq.text), question_terms) >= COMBINER_JACCARD
        out.append(SubQuery(**{**_as_kwargs(sq), "is_combiner": is_combiner}))
    return out


def _finish(
    question: str,
    subqueries: Sequence[SubQuery],
    *,
    revision: int,
    origin: str,
    repairs: Sequence[str] = (),
    strategy_llm: Strategy | None = None,
    directive_id: str | None = None,
    prompt_id: str | None = None,
    raw_llm_output: str | None = None,
) -> Plan:
    """Assemble the immutable Plan: hops, derived strategy, combiners, depth."""
    nodes = mark_combiners(assign_hops(subqueries), question)
    depth = max((sq.hop for sq in nodes), default=1)
    return Plan(
        question=question,
        subqueries=tuple(nodes),
        strategy=derive_strategy(nodes),
        strategy_llm=strategy_llm,
        revision=revision,
        origin=origin,  # type: ignore[arg-type]
        depth=depth,
        repairs=tuple(repairs),
        directive_id=directive_id,
        prompt_id=prompt_id,
        raw_llm_output=raw_llm_output,
    )


# ---------------------------------------------------------------------------
# Fallback ladder (deterministic, zero LLM calls)
# ---------------------------------------------------------------------------

def _split_operands(question: str) -> tuple[str, str] | None:
    """The two things being compared, from ``" or "`` or a capitalised pair."""
    tail = question.rstrip("?").strip()
    segment = tail.rsplit(",", 1)[-1] if "," in tail else tail
    parts = re.split(r"\s+or\s+", segment, flags=re.I)
    if len(parts) == 2:
        left, right = (collapse_whitespace(p.strip(" ,")) for p in parts)
        # "which magazine was started first" splits nothing useful; require
        # both sides to look like names rather than sentence fragments.
        if left and right and any(c.isupper() for c in left) and any(c.isupper() for c in right):
            return left, right
    runs = entity_runs(question)
    if len(runs) >= 2:
        return runs[0], runs[1]
    return None


def _comparison_stem(question: str, operands: tuple[str, str]) -> str:
    """The question with the operand pair removed -- the shared attribute ask."""
    stem = question
    for operand in operands:
        stem = re.sub(rf"\s*{re.escape(operand)}\s*", " ", stem)
    stem = re.sub(r"\s+or\s+", " ", stem, flags=re.I)
    return collapse_whitespace(stem.strip(" ,?"))


def comparison_template(question: str) -> list[SubQuery] | None:
    """F1: one independent look-up per compared entity, or None.

    Deliberately narrow. A template that fires on the wrong question emits a
    confidently wrong plan, which is worse than the identity floor.
    """
    if not any(pattern.search(question) for pattern in _COMPARISON_PATTERNS):
        return None
    operands = _split_operands(question)
    if operands is None:
        return None
    if _YESNO_MARKER.search(question):
        answer_type: AnswerType = "string"
    elif _TEMPORAL_MARKER.search(question):
        answer_type = "date"
    else:
        answer_type = "string"
    stem = _comparison_stem(question, operands)
    return [
        SubQuery(
            id=f"q{i}",
            text=collapse_whitespace(f"{operand} {stem}"),
            intent="comparison",
            entities=(operand,),
            answer_type=answer_type,
        )
        for i, operand in enumerate(operands, start=1)
    ]


def iterative_bridge(question: str) -> list[SubQuery] | None:
    """F2: self-ask-lite. Two hops when the LLM is unavailable but entities exist."""
    runs = entity_runs(question)
    if len(runs) < 2 or _YESNO_MARKER.search(question):
        return None
    return [
        SubQuery(id="q1", text=question, intent="lookup", entities=tuple(runs[:3]),
                 answer_type="entity"),
        SubQuery(id="q2", text=f"{{{{q1.title}}}} {question}", depends_on=("q1",),
                 intent="bridge", answer_type="string"),
    ]


def identity_plan(question: str) -> list[SubQuery]:
    """F3: the floor. One node, routed to hybrid+rerank -- the strongest baseline."""
    return [
        SubQuery(
            id="q1",
            text=question,
            intent="lookup",
            entities=tuple(entity_runs(question)[:3]),
            answer_type="string",
            tool_hint="hybrid_search",
        )
    ]


def fallback_plan(
    question: str,
    *,
    revision: int = 0,
    reason: str = "fallback",
    directive_id: str | None = None,
) -> Plan:
    """Walk F1 -> F2 -> F3 and return the first that fires. Never fails."""
    for rung, builder in (("F1", comparison_template), ("F2", iterative_bridge)):
        nodes = builder(question)
        if nodes:
            return _finish(
                question, nodes, revision=revision, origin="fallback_rule",
                repairs=(f"fallback:{rung}:{reason}",), directive_id=directive_id,
            )
    return _finish(
        question, identity_plan(question), revision=revision, origin="fallback_rule",
        repairs=(f"fallback:F3:{reason}",), directive_id=directive_id,
    )


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

class Planner(BaseAgent):
    """One LLM call, then determinism all the way down."""

    name = "planner"

    def __init__(self, cfg: Config | None = None, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        self.max_subqueries = int(self.cfg.get("agents.planner.max_subqueries", 5))
        self.max_rewrites = int(self.cfg.get("agents.retriever.max_rewrites", 2))
        self.template_shortcut = bool(self.cfg.get("agents.planner.template_shortcut", False))
        if not bool(self.cfg.get("agents.planner.allow_query_rewrite", True)):
            self.max_rewrites = 0

    # -- public API --------------------------------------------------------
    def run(self, state: QuestionState, *, directive: ReplanDirective | None = None) -> Plan:
        """Produce a Plan. Never raises; never returns an empty DAG."""
        revision = directive.revision if directive is not None else 0
        plan = self._plan(state, directive, revision)
        if plan is None:  # only reachable if step() swallowed something unexpected
            plan = fallback_plan(
                state.question, revision=revision, reason="agent_error",
                directive_id=directive.directive_id if directive else None,
            )
        return plan

    # -- internals ---------------------------------------------------------
    def _plan(
        self,
        state: QuestionState,
        directive: ReplanDirective | None,
        revision: int,
    ) -> Plan | None:
        plan: Plan | None = None
        directive_id = directive.directive_id if directive else None
        with state.step(
            self.name,
            question_chars=len(state.question),
            revision=revision,
            replan=directive is not None,
        ) as rec:
            if self.template_shortcut and directive is None:
                nodes = comparison_template(state.question)
                if nodes:
                    state.budget.note_saved()
                    plan = _finish(
                        state.question, nodes, revision=revision,
                        origin="template_shortcut", repairs=("template_shortcut:F1",),
                    )

            if plan is None:
                prompt_id = "planner.replan.v1" if directive else "planner.decompose.v1"
                call = self.call_json(
                    state,
                    rec,
                    prompt_id=prompt_id,
                    variables=self._variables(state.question, directive),
                    schema=PLAN_SCHEMA,
                    purpose="replan" if directive else "decompose",
                    privileged=False,
                )
                if call.ok and call.parsed is not None:
                    nodes, repairs = validate_subqueries(
                        _subquery_payload(call.parsed),
                        max_subqueries=self.max_subqueries,
                        max_rewrites=self.max_rewrites,
                    )
                    if nodes:
                        plan = _finish(
                            state.question, nodes, revision=revision,
                            origin="llm_repaired" if repairs else "llm",
                            repairs=repairs,
                            strategy_llm=_claimed_strategy(call.parsed),
                            directive_id=directive_id, prompt_id=prompt_id,
                            raw_llm_output=call.raw,
                        )
                    else:
                        rec.degrade("validation_empty")
                else:
                    rec.degrade(call.reason or "parse_failure")

            if plan is None:
                plan = fallback_plan(
                    state.question, revision=revision,
                    reason=rec.fallback_reason or "unknown", directive_id=directive_id,
                )

            rec.output_summary = {
                "strategy": plan.strategy,
                "strategy_llm": plan.strategy_llm,
                "strategy_agrees": plan.strategy_llm == plan.strategy,
                "n_subqueries": len(plan.subqueries),
                "depth": plan.depth,
                "origin": plan.origin,
                "repairs": list(plan.repairs),
                "combiners": [sq.id for sq in plan.subqueries if sq.is_combiner],
            }
        return plan

    def _variables(self, question: str, directive: ReplanDirective | None) -> dict[str, Any]:
        """Render-time variables. Titles, not passage text: prompt length is latency."""
        base: dict[str, Any] = {
            "question": question,
            "max_subqueries": self.max_subqueries,
            "max_rewrites": self.max_rewrites,
        }
        if directive is None:
            return base
        base.update(
            {
                "previous_plan_summary": directive.previous_plan_summary or "(none)",
                "covered_entities": _bullets(directive.covered_entities) or "(nothing yet)",
                "missing_information": _bullets(directive.missing_information) or "(unspecified)",
                "failed_subquery_ids": ", ".join(directive.failed_subquery_ids) or "(none)",
                "suggested_subqueries": _bullets(directive.suggested_subqueries) or "(none)",
                "banned_subquery_texts": _bullets(directive.banned_subquery_texts) or "(none)",
                "confidence": f"{directive.confidence:.2f}",
            }
        )
        return base


def _bullets(items: Sequence[str], limit: int = 8) -> str:
    return "\n".join(f"- {item}" for item in items[:limit])


def _subquery_payload(parsed: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Find the node list wherever the model put it.

    ``llm._coerce_json`` wraps a top-level array as ``{"items": [...]}``, and
    the model occasionally names the key ``plan`` or ``steps``. Accepting all of
    them costs four lines and saves a fallback.
    """
    for key in ("subqueries", "sub_queries", "items", "plan", "steps"):
        value = parsed.get(key)
        if isinstance(value, (list, tuple)):
            return [v for v in value if isinstance(v, Mapping)]
    return []


def _claimed_strategy(parsed: Mapping[str, Any]) -> Strategy | None:
    value = str(parsed.get("strategy") or "").strip().lower()
    valid: tuple[Strategy, ...] = (
        "single_hop", "bridge", "comparison", "attribute", "bridge_comparison",
    )
    return value if value in valid else None  # type: ignore[return-value]
