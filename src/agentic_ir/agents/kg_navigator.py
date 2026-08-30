"""KG Navigator: seed entities, traverse, and return citable graph evidence.

Contract: ``docs/architecture.md`` 3.3. Input is one ``SubQuery`` plus its
``RetrievalResult``; output is a ``KGResult``; and in the shipped configuration
(``agents.kg.entity_linker: alias_match``) this agent issues **zero LLM calls**.
An empty ``KGResult`` is a legal, non-fatal outcome -- the pipeline continues on
passage evidence alone -- so nothing here may raise.

The three traversal modes, and why they differ
----------------------------------------------
*One seed* -> bounded BFS to ``max_hops``, keeping ``max_neighbors`` per node.
This is the common HotpotQA bridge case: the question names the first hop, and
the second hop is a page the first one mentions. Measured on
``hotpotqa_calib_50``, one of the two gold supporting titles is within the top
25 neighbours of the other for 64% of questions -- the graph already holds the
hop.

*Two or more seeds* -> bidirectional BFS. The meeting node is the bridge
entity. This is the highest-value operation in the agent: the bridge is usually
itself a page title, so a search from both ends finds it in milliseconds, where
asking the model would cost seconds and could be wrong.

*Comparison* -> no path is expected, and forcing one would manufacture a
spurious relationship between two entities whose only real connection is that a
question mentioned both. Each seed instead returns its own attribute-bearing
sentences, which is what a comparison actually needs: two dates, two numbers,
and an arithmetic decision made downstream by the Synthesizer (5.4).

Evidence identity
-----------------
Every edge carries the sentence that licensed it, so each traversal step
becomes a real, independently citable :class:`~agentic_ir.types.Evidence` -- an
NLI premise the Verifier can score, not a decoration. Ids assigned here are
provisional (``kg:q1:03``); AGGREGATE renumbers the whole pool to ``e1..eN``.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..config import Config, load_config
from ..kg.entity_link import normalise_entity
from ..kg.graph import KnowledgeGraph
from ..kg.traverse import bfs_neighborhood, bridge_paths
from ..state import QuestionState
from ..types import Entity, Evidence, KGPath, KGResult, SubQuery, ToolCallTrace
from .base import BaseAgent

__all__ = ["KGNavigator", "PROMPT_ID", "intermediate_entity", "kg_evidence_id"]

PROMPT_ID = "kg.link_entities.v1"

#: Provisional evidence-id prefix. AGGREGATE assigns the final ``e1..eN``; a
#: distinguishable prefix makes a renumbering bug obvious in a trace instead of
#: silently colliding with passage evidence.
_EVIDENCE_PREFIX = "kg"

#: A sentence "bears an attribute" if it states a year, a date or a number.
#: Comparison questions in both benchmarks compare exactly these.
_ATTRIBUTE_RE = re.compile(
    r"\b(?:1[0-9]{3}|20[0-9]{2})\b"
    r"|\b\d{1,3}(?:[.,]\d+)+\b"
    r"|\b\d+\s*(?:km|mi|m|ft|kg|lb|years?|months?|days?)\b"
    r"|\b(?:January|February|March|April|May|June|July|August|September|October"
    r"|November|December)\b",
    re.IGNORECASE,
)

_LINK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
    },
    "required": ["entities"],
}


def kg_evidence_id(subquery_id: str, index: int) -> str:
    """Provisional, deterministic id for one piece of graph evidence."""
    return f"{_EVIDENCE_PREFIX}:{subquery_id}:{index:02d}"


def intermediate_entity(path: KGPath, seeds: Sequence[str]) -> str | None:
    """The first node on ``path`` that is not itself a seed.

    ``KGPath.bridge_entity`` is the *meeting node* of the bidirectional search,
    which is the literal type contract but is a seed itself whenever the two
    seeds are directly adjacent. What downstream consumers actually want -- the
    value to substitute into ``{{q1.entity}}``, or to take as the hop-1 answer
    (architecture 4.4 rung 1) -- is a node strictly between the seeds.
    """
    seen = set(seeds)
    if path.bridge_entity and path.bridge_entity not in seen:
        return path.bridge_entity
    for node in path.nodes[1:-1]:
        if node not in seen:
            return node
    return None


@lru_cache(maxsize=4)
def _load_graph(dataset: str, config_path: str | None) -> KnowledgeGraph:
    """Process-wide graph cache: a 250-question run must not reload it 250 times."""
    return KnowledgeGraph.load(dataset, cfg=load_config(config_path))


class KGNavigator(BaseAgent):
    """Bounded, LLM-free traversal of the passage-mention graph."""

    name = "kg"

    def __init__(
        self,
        graph: KnowledgeGraph | None = None,
        *,
        cfg: Config | None = None,
        dataset: str | None = None,
        corpus: Any | None = None,
        client: Any | None = None,
        prompt_dir: Path | None = None,
    ) -> None:
        super().__init__(cfg, client=client, prompt_dir=prompt_dir)
        self.graph = graph
        self.dataset = dataset or (graph.dataset if graph is not None else None)
        self.corpus = corpus
        self.max_hops = int(self.cfg.get("agents.kg.max_hops", 2))
        self.max_neighbors = int(self.cfg.get("agents.kg.max_neighbors", 25))
        self.entity_linker = str(self.cfg.get("agents.kg.entity_linker", "alias_match"))
        self.max_seeds = int(self.cfg.get("agents.kg.max_seeds", 4))
        # Not present in config.yaml, and deliberately defaulted OFF. Section
        # 3.3 can be read as licensing an LLM rescue when alias matching finds
        # nothing, but 2.6 budgets the KG agent at zero calls and 3.3's own
        # header says zero in the shipped config. Zero wins; the knob exists so
        # the other reading is one config override away.
        self.llm_link_on_empty = bool(self.cfg.get("agents.kg.llm_link_on_empty", False))

    # -- construction ------------------------------------------------------

    @classmethod
    def load(
        cls,
        dataset: str,
        *,
        cfg: Config | None = None,
        corpus: Any | None = None,
        config_path: str | None = None,
    ) -> KGNavigator:
        """Load a dataset's persisted graph (cached) and wrap it."""
        return cls(
            _load_graph(dataset, config_path),
            cfg=cfg or load_config(config_path),
            dataset=dataset,
            corpus=corpus,
        )

    def _ensure_graph(self) -> KnowledgeGraph | None:
        if self.graph is None and self.dataset:
            self.graph = _load_graph(self.dataset, None)
        return self.graph

    # -- entry point -------------------------------------------------------

    def run(
        self,
        state: QuestionState,
        subquery: SubQuery,
        retrieval: Any | None = None,
    ) -> KGResult:
        """Traverse for one sub-query. Never raises; may return an empty result."""
        started = time.perf_counter()
        result: KGResult | None = None
        with state.step(
            self.name, subquery_id=subquery.id, text=subquery.text[:200]
        ) as rec:
            graph = self._ensure_graph()
            if graph is None or len(graph) == 0:
                rec.degrade("kg_unavailable")
            else:
                seeds, linked_by = self._seed(state, subquery, retrieval, graph, rec)
                if not seeds:
                    rec.degrade("no_seeds")
                    result = KGResult(
                        subquery_id=subquery.id,
                        linked_by="none",
                        latency_s=round(time.perf_counter() - started, 4),
                        degraded=True,
                    )
                else:
                    result = self._traverse(
                        state, subquery, graph, seeds, linked_by, rec, started
                    )
        if result is None:
            result = KGResult(
                subquery_id=subquery.id,
                linked_by="none",
                latency_s=round(time.perf_counter() - started, 4),
                degraded=True,
                error=rec.fallback_reason,
            )
        state.kg_results[subquery.id] = result
        return result

    # -- seeding -----------------------------------------------------------

    def _seed(
        self,
        state: QuestionState,
        subquery: SubQuery,
        retrieval: Any | None,
        graph: KnowledgeGraph,
        rec: Any,
    ) -> tuple[tuple[Entity, ...], str]:
        """The fallback ladder of 3.3: alias match -> (LLM) -> retrieved titles.

        Sub-query ``entities`` are tried before the free text: the Planner
        already believes those spans are entities, and honouring that is free.
        """
        seeds: list[Entity] = []
        seen: set[str] = set()

        def _push(entity: Entity | None) -> None:
            if entity is not None and entity.entity_id not in seen:
                seen.add(entity.entity_id)
                seeds.append(entity)

        for surface in subquery.entities:
            for entity in graph.link(surface, limit=2):
                _push(entity)
        for entity in graph.link(subquery.text, limit=self.max_seeds):
            _push(entity)
        if not seeds:
            # A question genuinely about a frequently-mentioned page deserves a
            # hub seed rather than none.
            for entity in graph.link(subquery.text, limit=self.max_seeds, skip_hubs=False):
                _push(entity)
        state.budget.note_tool()
        rec.note_tool(
            ToolCallTrace(
                call_id=f"{subquery.id}:kg_link",
                agent=self.name,
                tool="kg_link",
                query=subquery.text[:200],
                n_results=len(seeds),
                latency_s=0.0,
            )
        )
        if seeds:
            state.budget.note_saved()  # the LLM linker was not needed
            return tuple(seeds[: self.max_seeds]), "alias_match"

        if self.entity_linker == "llm" or self.llm_link_on_empty:
            llm_seeds = self._link_with_llm(state, subquery, graph, rec)
            if llm_seeds:
                return llm_seeds, "llm"

        titles = self._retrieved_titles(state, subquery, retrieval)
        for title in titles[:3]:
            _push(graph.entity_for_title(title))
        if seeds:
            rec.degrade("retrieval_titles")
            return tuple(seeds[: self.max_seeds]), "retrieval_titles"
        return (), "none"

    @staticmethod
    def _retrieved_titles(
        state: QuestionState, subquery: SubQuery, retrieval: Any | None
    ) -> tuple[str, ...]:
        """Top retrieved titles for this sub-query, however they reach us."""
        result = retrieval if retrieval is not None else state.results.get(subquery.id)
        if result is None:
            return ()
        return tuple(sp.passage.title for sp in getattr(result, "passages", ())[:3])

    def _link_with_llm(
        self,
        state: QuestionState,
        subquery: SubQuery,
        graph: KnowledgeGraph,
        rec: Any,
    ) -> tuple[Entity, ...]:
        """``kg.link_entities.v1``. Off in the shipped config; see ``__init__``.

        Non-privileged: the synthesiser and verifier reserve must survive an
        entity-linking miss, because an answer with weak evidence beats no
        answer at all.
        """
        call = self.call_json(
            state,
            rec,
            prompt_id=PROMPT_ID,
            variables={
                "question": subquery.text,
                "hints": ", ".join(subquery.entities) or "(none)",
            },
            schema=_LINK_SCHEMA,
            purpose="entity_link",
            privileged=False,
            num_predict=96,
        )
        if call.failed:
            rec.degrade(f"llm_link_failed:{call.reason}")
            return ()
        names = (call.parsed or {}).get("entities") or []
        out: list[Entity] = []
        seen: set[str] = set()
        for name in names:
            entity = graph.entity_for_title(str(name)) or next(
                iter(graph.link(str(name), limit=1, skip_hubs=False)), None
            )
            if entity is not None and entity.entity_id not in seen:
                seen.add(entity.entity_id)
                out.append(entity)
        return tuple(out[: self.max_seeds])

    # -- traversal ---------------------------------------------------------

    def _is_comparison(self, state: QuestionState, subquery: SubQuery) -> bool:
        """Comparison sub-queries expect two attribute look-ups, not a path."""
        if subquery.intent == "comparison" or subquery.answer_type == "yesno":
            return True
        plan = state.plan
        return bool(plan and plan.strategy == "comparison")

    def _traverse(
        self,
        state: QuestionState,
        subquery: SubQuery,
        graph: KnowledgeGraph,
        seeds: tuple[Entity, ...],
        linked_by: str,
        rec: Any,
        started: float,
    ) -> KGResult:
        seed_ids = tuple(e.entity_id for e in seeds)
        paths: tuple[KGPath, ...] = ()
        bridge: str | None = None
        neighbours: list[Entity] = []
        triples = []

        if len(seeds) >= 2 and self._is_comparison(state, subquery):
            tool = "kg_neighbors"
            evidence = self._comparison_evidence(graph, subquery, seeds)
            for seed in seeds:
                for nid in graph.neighbors(seed.entity_id, limit=self.max_neighbors):
                    entity = graph.entity(nid)
                    if entity is not None:
                        neighbours.append(entity)
        elif len(seeds) >= 2:
            tool = "kg_path"
            paths = bridge_paths(
                graph,
                seed_ids,
                max_hops=self.max_hops,
                max_neighbors=self.max_neighbors,
                max_paths=3,
            )
            if paths:
                bridge_id = intermediate_entity(paths[0], seed_ids)
                bridge = graph.name_of(bridge_id) if bridge_id else None
                for path in paths:
                    triples.extend(path.edges)
                    for node in path.nodes:
                        entity = graph.entity(node)
                        if entity is not None:
                            neighbours.append(entity)
                evidence = self._edge_evidence(graph, subquery, triples)
            else:
                # Two seeds that do not connect within the hop budget still
                # have neighbourhoods worth citing; falling back to them keeps
                # the sub-query from contributing nothing at all.
                rec.degrade("no_path_between_seeds")
                evidence = self._comparison_evidence(graph, subquery, seeds)
        else:
            tool = "kg_neighbors"
            hood = bfs_neighborhood(
                graph,
                seed_ids[0],
                max_hops=self.max_hops,
                max_neighbors=self.max_neighbors,
            )
            triples = list(hood.triples[: self.max_neighbors])
            for nid in hood.nodes[: self.max_neighbors]:
                entity = graph.entity(nid)
                if entity is not None:
                    neighbours.append(entity)
            evidence = self._edge_evidence(graph, subquery, triples)

        state.budget.note_tool()
        rec.note_tool(
            ToolCallTrace(
                call_id=f"{subquery.id}:{tool}",
                agent=self.name,
                tool=tool,  # type: ignore[arg-type]
                query=" | ".join(seed_ids),
                n_results=len(evidence),
                latency_s=round(time.perf_counter() - started, 4),
            )
        )
        deduped: list[Entity] = []
        seen_nodes: set[str] = set()
        for entity in neighbours:
            if entity.entity_id not in seen_nodes and entity.entity_id not in seed_ids:
                seen_nodes.add(entity.entity_id)
                deduped.append(entity)
        rec.output_summary = {
            "seeds": list(seed_ids),
            "linked_by": linked_by,
            "n_paths": len(paths),
            "bridge_entity": bridge,
            "n_neighbors": len(deduped[: self.max_neighbors]),
            "n_evidence": len(evidence),
        }
        if bridge:
            state.bridge_entities[subquery.id] = bridge
        return KGResult(
            subquery_id=subquery.id,
            seeds=seeds,
            linked_by=linked_by,  # type: ignore[arg-type]
            paths=paths,
            bridge_entity=bridge,
            neighbors=tuple(deduped[: self.max_neighbors]),
            evidence=evidence,
            latency_s=round(time.perf_counter() - started, 4),
            degraded=rec.degraded,
        )

    # -- evidence ----------------------------------------------------------

    def _edge_evidence(
        self, graph: KnowledgeGraph, subquery: SubQuery, triples: Sequence[Any]
    ) -> tuple[Evidence, ...]:
        """One :class:`Evidence` per traversed edge, carrying its sentence."""
        out: list[Evidence] = []
        seen: set[tuple[str, str]] = set()
        for triple in triples:
            key = (triple.subject, triple.object)
            if key in seen:
                continue
            seen.add(key)
            sentence = graph.edge_sentence(triple.subject, triple.object)
            if not sentence:
                continue
            out.append(
                Evidence(
                    evidence_id=kg_evidence_id(subquery.id, len(out) + 1),
                    kind="kg_triple",
                    text=sentence,
                    score=round(float(triple.weight), 6),
                    subquery_ids=(subquery.id,),
                    provenance="kg",
                    doc_id=triple.doc_id,
                    title=graph.name_of(triple.subject),
                    sent_id=triple.sent_id,
                    triple=triple,
                )
            )
            if len(out) >= self.max_neighbors:
                break
        return tuple(out)

    def _comparison_evidence(
        self,
        graph: KnowledgeGraph,
        subquery: SubQuery,
        seeds: tuple[Entity, ...],
    ) -> tuple[Evidence, ...]:
        """Each seed's own attribute-bearing sentences. No path is forced.

        Prefers the seed's article sentences when a corpus is available (that
        is where "founded in 1844" actually lives) and falls back to the
        sentences on its outgoing edges, so the graph alone still answers.
        """
        out: list[Evidence] = []
        per_seed = max(1, self.max_neighbors // max(1, len(seeds)))
        for seed in seeds:
            found = 0
            for doc_id, sent_id, sentence in self._seed_sentences(graph, seed):
                if not _ATTRIBUTE_RE.search(sentence):
                    continue
                out.append(
                    Evidence(
                        evidence_id=kg_evidence_id(subquery.id, len(out) + 1),
                        kind="kg_triple",
                        text=sentence,
                        score=round(1.0 - 0.01 * found, 6),
                        subquery_ids=(subquery.id,),
                        provenance="kg",
                        doc_id=doc_id,
                        title=seed.name,
                        sent_id=sent_id,
                    )
                )
                found += 1
                if found >= per_seed:
                    break
        return tuple(out)

    def _seed_sentences(
        self, graph: KnowledgeGraph, seed: Entity
    ) -> list[tuple[str | None, int | None, str]]:
        """``(doc_id, sent_id, sentence)`` for one seed, article text first."""
        rows: list[tuple[str | None, int | None, str]] = []
        if self.corpus is not None:
            for doc_id in seed.doc_ids:
                try:
                    passage = self.corpus.get(doc_id)
                except KeyError:
                    continue
                rows.extend(
                    (doc_id, i, s) for i, s in enumerate(passage.sentences) if s
                )
        rows.extend(graph.attribute_sentences(seed.entity_id))
        for triple in graph.out_triples(seed.entity_id, limit=self.max_neighbors):
            sentence = graph.edge_sentence(triple.subject, triple.object)
            if sentence:
                rows.append((triple.doc_id, triple.sent_id, sentence))
        seen: set[str] = set()
        unique: list[tuple[str | None, int | None, str]] = []
        for row in rows:
            if row[2] not in seen:
                seen.add(row[2])
                unique.append(row)
        return unique


def _normalise(title: str) -> str:
    """Re-exported for callers that hold a raw title, not an entity id."""
    return normalise_entity(title)
