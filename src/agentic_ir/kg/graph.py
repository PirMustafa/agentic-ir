"""The knowledge graph object: a ``networkx.DiGraph`` with a typed façade.

The graph is a *mention* graph over corpus page titles. An edge ``A -> B``
means "an alias of page B occurs in the text of page A", and every edge carries
the ``doc_id``, ``sent_id`` and the verbatim containing sentence that licensed
it. That last field is the whole point: it is what turns a traversal result
into :class:`~agentic_ir.types.Evidence` the Verifier can run NLI over, so KG
contributions are *scoreable* rather than decorative.

Why a façade rather than passing the raw ``DiGraph`` around
-----------------------------------------------------------
Three things have to be true everywhere and are easy to get wrong once:

1. **Every sort has a tiebreaker** (design axiom 5). Neighbour ordering is
   ``(-weight, entity_id)``, never dictionary order, or two runs of the same
   configuration would produce different evidence.
2. **The alias table must come from the same normalisation as the build.**
   Rebuilding it here from the persisted node aliases guarantees that, and
   costs one pass over the nodes at load time.
3. **Traversal is over the undirected view.** Mention direction is an artefact
   of which page happened to be written first; "Titanic mentions Cameron" and
   "Cameron mentions Titanic" are the same fact for path finding. The stored
   triple keeps its true direction, so the evidence never lies about who
   asserted what.

Persistence is two gzipped JSONL files plus a small JSON metadata file, written
through binary handles (orjson emits UTF-8 bytes, which also sidesteps the
cp1252 default Windows applies to text handles).
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import orjson

from ..config import Config, Paths, load_config
from ..types import Entity, Triple
from .entity_link import AliasTable, normalise_entity

__all__ = [
    "EDGE_SUFFIX",
    "META_SUFFIX",
    "NODE_SUFFIX",
    "KnowledgeGraph",
    "kg_paths",
]

NODE_SUFFIX = "_kg_nodes.jsonl.gz"
EDGE_SUFFIX = "_kg_edges.jsonl.gz"
META_SUFFIX = "_kg_meta.json"


@dataclass(frozen=True, slots=True)
class _KGFiles:
    nodes: Path
    edges: Path
    meta: Path


def kg_paths(dataset: str, cfg: Config | None = None, directory: Path | None = None) -> _KGFiles:
    """Where a dataset's graph lives, by default under ``paths.indexes``."""
    root = directory or Paths.from_config(cfg or load_config()).indexes
    return _KGFiles(
        nodes=root / f"{dataset}{NODE_SUFFIX}",
        edges=root / f"{dataset}{EDGE_SUFFIX}",
        meta=root / f"{dataset}{META_SUFFIX}",
    )


def _write_gzip_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as fh:
        for record in records:
            fh.write(orjson.dumps(record))
            fh.write(b"\n")
            written += 1
    return written


def _read_gzip_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as raw, gzip.GzipFile(fileobj=raw, mode="rb") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield orjson.loads(line)


class KnowledgeGraph:
    """A loaded mention graph plus its alias index."""

    __slots__ = ("_alias_table", "dataset", "g", "meta")

    def __init__(
        self,
        graph: nx.DiGraph | None = None,
        *,
        dataset: str | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        self.g: nx.DiGraph = graph if graph is not None else nx.DiGraph()
        self.dataset = dataset
        self.meta: dict[str, Any] = dict(meta or {})
        self._alias_table: AliasTable | None = None

    # -- construction ------------------------------------------------------

    def add_entity(
        self,
        entity_id: str,
        *,
        name: str,
        aliases: Sequence[str] = (),
        doc_ids: Sequence[str] = (),
        attribute_sentences: Sequence[tuple[str, int, str]] = (),
        mention_df: int = 0,
    ) -> None:
        """Create or extend a node.

        Titles that normalise to the same id are *merged*, which is a real
        consequence of stripping parenthetical disambiguators. The first name
        seen (the corpus is iterated in ``doc_id`` order) wins, so the merge is
        deterministic rather than dependent on iteration order.
        """
        node = self.g.nodes.get(entity_id)
        if node is None:
            self.g.add_node(
                entity_id,
                name=name,
                aliases=tuple(dict.fromkeys(aliases)),
                doc_ids=tuple(dict.fromkeys(doc_ids)),
                attribute_sentences=tuple(dict.fromkeys(attribute_sentences)),
                mention_df=mention_df,
            )
            self._alias_table = None
            return
        node["aliases"] = tuple(dict.fromkeys((*node["aliases"], *aliases)))
        node["doc_ids"] = tuple(dict.fromkeys((*node["doc_ids"], *doc_ids)))
        node["attribute_sentences"] = tuple(
            dict.fromkeys((*node.get("attribute_sentences", ()), *attribute_sentences))
        )
        node["mention_df"] = max(int(node.get("mention_df", 0)), mention_df)
        self._alias_table = None

    def add_mention_edge(
        self,
        subject: str,
        obj: str,
        *,
        relation: str,
        doc_id: str,
        sent_id: int,
        sentence: str,
        weight: float = 1.0,
    ) -> None:
        """Record that page ``subject`` mentions page ``obj`` in one sentence.

        Only the first (lowest ``sent_id``) supporting sentence is kept per
        ordered pair. A second sentence would not make the edge more true and
        would multiply the on-disk size of the graph by the mention count.
        """
        existing = self.g.edges.get((subject, obj))
        if existing is not None and existing["sent_id"] <= sent_id:
            return
        self.g.add_edge(
            subject,
            obj,
            relation=relation,
            doc_id=doc_id,
            sent_id=sent_id,
            sentence=sentence,
            weight=weight,
        )

    def apply_hub_weights(self) -> None:
        """Set ``weight = 1 / (1 + log(1 + out_degree(A)))`` on every edge.

        Architecture 3.3. A page that mentions three hundred other pages is a
        list or a hub, and each of its edges says correspondingly little; a
        page that mentions two says a lot about both. Computed after all edges
        exist, because out-degree is only final then.
        """
        from math import log

        for subject in self.g.nodes:
            out_degree = self.g.out_degree(subject)
            weight = 1.0 / (1.0 + log(1.0 + out_degree)) if out_degree else 1.0
            for _, _, data in self.g.out_edges(subject, data=True):
                data["weight"] = weight

    # -- container protocol ------------------------------------------------

    def __len__(self) -> int:
        return self.g.number_of_nodes()

    def __contains__(self, entity_id: object) -> bool:
        return entity_id in self.g

    def __repr__(self) -> str:
        return (
            f"KnowledgeGraph(dataset={self.dataset!r}, "
            f"nodes={self.g.number_of_nodes()}, edges={self.g.number_of_edges()})"
        )

    @property
    def num_edges(self) -> int:
        return self.g.number_of_edges()

    # -- lookup ------------------------------------------------------------

    def entity(self, entity_id: str) -> Entity | None:
        """The typed node, or ``None``. Data-driven miss, so no exception."""
        node = self.g.nodes.get(entity_id)
        if node is None:
            return None
        return Entity(
            entity_id=entity_id,
            name=str(node.get("name") or entity_id),
            aliases=tuple(node.get("aliases") or ()),
            doc_ids=tuple(node.get("doc_ids") or ()),
        )

    def name_of(self, entity_id: str) -> str:
        """The surface title of a node, falling back to the id itself.

        Callers that feed a value back into a *query* -- placeholder
        resolution, answer extraction -- want this, not the folded id.
        """
        node = self.g.nodes.get(entity_id)
        return str(node.get("name") or entity_id) if node else entity_id

    def triple(self, subject: str, obj: str) -> Triple | None:
        """The stored triple for one ordered pair, or ``None``."""
        data = self.g.edges.get((subject, obj))
        if data is None:
            return None
        return Triple(
            subject=subject,
            relation=str(data.get("relation") or "mentions"),
            object=obj,
            doc_id=data.get("doc_id"),
            sent_id=data.get("sent_id"),
            weight=float(data.get("weight", 1.0)),
        )

    def edge_sentence(self, subject: str, obj: str) -> str:
        """The verbatim sentence that licensed an edge, or ``""``."""
        data = self.g.edges.get((subject, obj))
        return str(data.get("sentence") or "") if data else ""

    def any_triple(self, u: str, v: str) -> tuple[Triple | None, bool]:
        """The triple joining ``u`` and ``v`` in either direction.

        Returns ``(triple, reversed_)``; ``reversed_`` is True when the stored
        assertion runs ``v -> u``. Path finding is undirected, but the evidence
        must report the direction actually written down.
        """
        forward = self.triple(u, v)
        if forward is not None:
            return forward, False
        backward = self.triple(v, u)
        if backward is not None:
            return backward, True
        return None, False

    # -- neighbourhoods ----------------------------------------------------

    def out_triples(self, entity_id: str, limit: int | None = None) -> tuple[Triple, ...]:
        """Outgoing edges, sorted ``(-weight, object)``."""
        if entity_id not in self.g:
            return ()
        triples = [
            Triple(
                subject=entity_id,
                relation=str(data.get("relation") or "mentions"),
                object=obj,
                doc_id=data.get("doc_id"),
                sent_id=data.get("sent_id"),
                weight=float(data.get("weight", 1.0)),
            )
            for _, obj, data in self.g.out_edges(entity_id, data=True)
        ]
        triples.sort(key=lambda t: (-t.weight, t.object))
        return tuple(triples[:limit]) if limit is not None else tuple(triples)

    def in_triples(self, entity_id: str, limit: int | None = None) -> tuple[Triple, ...]:
        """Incoming edges, sorted ``(-weight, subject)``."""
        if entity_id not in self.g:
            return ()
        triples = [
            Triple(
                subject=subj,
                relation=str(data.get("relation") or "mentions"),
                object=entity_id,
                doc_id=data.get("doc_id"),
                sent_id=data.get("sent_id"),
                weight=float(data.get("weight", 1.0)),
            )
            for subj, _, data in self.g.in_edges(entity_id, data=True)
        ]
        triples.sort(key=lambda t: (-t.weight, t.subject))
        return tuple(triples[:limit]) if limit is not None else tuple(triples)

    def neighbors(self, entity_id: str, limit: int | None = None) -> tuple[str, ...]:
        """Undirected neighbours, ``(-weight, entity_id)`` ordered, deduplicated.

        The cap is what keeps a hub page from turning a two-hop BFS into a scan
        of the corpus. Because the order is total, truncation is reproducible.
        """
        if entity_id not in self.g:
            return ()
        best: dict[str, float] = {}
        for _, obj, data in self.g.out_edges(entity_id, data=True):
            if obj == entity_id:
                continue
            w = float(data.get("weight", 1.0))
            if w > best.get(obj, -1.0):
                best[obj] = w
        for subj, _, data in self.g.in_edges(entity_id, data=True):
            if subj == entity_id:
                continue
            w = float(data.get("weight", 1.0))
            if w > best.get(subj, -1.0):
                best[subj] = w
        ordered = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
        if limit is not None:
            ordered = ordered[:limit]
        return tuple(k for k, _ in ordered)

    def degree(self, entity_id: str) -> int:
        return self.g.degree(entity_id) if entity_id in self.g else 0

    def mention_df(self, entity_id: str) -> int:
        """How many corpus passages mentioned this entity, *before* pruning.

        Recorded at build time because pruning destroys the evidence for it,
        and the navigator needs it: seeding a traversal on a page mentioned by
        six thousand passages produces a neighbourhood, not an answer.
        """
        node = self.g.nodes.get(entity_id)
        return int(node.get("mention_df", 0)) if node else 0

    def attribute_sentences(self, entity_id: str) -> tuple[tuple[str, int, str], ...]:
        """Dated or numeric sentences from an entity's own corpus pages."""
        node = self.g.nodes.get(entity_id)
        if node is None:
            return ()
        return tuple(tuple(row) for row in node.get("attribute_sentences", ()))

    @property
    def mention_df_cap(self) -> int:
        """The hub cap this graph was built with; ``0`` when unpruned."""
        stats = self.meta.get("stats") or {}
        return int(stats.get("mention_df_cap") or self.meta.get("mention_df_cap") or 0)

    def is_hub(self, entity_id: str) -> bool:
        """True when this entity was pruned as a hub. Never raises on a miss."""
        cap = self.mention_df_cap
        return bool(cap) and self.mention_df(entity_id) >= cap

    # -- linking -----------------------------------------------------------

    def alias_table(self) -> AliasTable:
        """The alias trie, rebuilt from node aliases on first use.

        Rebuilt rather than persisted so that it can never disagree with the
        normalisation this build of the code performs.
        """
        if self._alias_table is None:
            self._alias_table = AliasTable.from_entities(
                {"entity_id": nid, "aliases": data.get("aliases") or ()}
                for nid, data in self.g.nodes(data=True)
            )
        return self._alias_table

    def link(
        self,
        text: str,
        *,
        limit: int | None = None,
        skip_hubs: bool = True,
    ) -> tuple[Entity, ...]:
        """Entities mentioned in ``text``, longest alias match, in text order.

        Ordering is by (descending alias length, first occurrence), so that the
        most specific mention seeds the traversal first.

        ``skip_hubs`` drops entities the build pruned as hubs: a question
        containing the word "one" should not seed the traversal on the
        Wikipedia page *One*. Callers should retry with ``skip_hubs=False``
        when this yields nothing, because a question genuinely *about* a
        frequently-mentioned page is better served by a hub seed than by none.
        """
        matches = self.alias_table().find_all(text)
        ranked = sorted(
            
                (-(m.token_end - m.token_start), m.token_start, entity_id)
                for m in matches
                for entity_id in m.entity_ids
            
        )
        out: list[Entity] = []
        seen: set[str] = set()
        for _, _, entity_id in ranked:
            if entity_id in seen or entity_id not in self.g:
                continue
            if skip_hubs and self.is_hub(entity_id):
                continue
            seen.add(entity_id)
            entity = self.entity(entity_id)
            if entity is not None:
                out.append(entity)
            if limit is not None and len(out) >= limit:
                break
        return tuple(out)

    def entity_for_title(self, title: str) -> Entity | None:
        """Resolve a raw page title straight to its node, bypassing the trie."""
        return self.entity(normalise_entity(title))

    # -- persistence -------------------------------------------------------

    def save(self, *, dataset: str | None = None, directory: Path | None = None,
             cfg: Config | None = None, meta: Mapping[str, Any] | None = None) -> _KGFiles:
        """Write nodes, edges and metadata. Returns the paths written."""
        name = dataset or self.dataset
        if not name:
            raise ValueError("KnowledgeGraph.save needs a dataset name")
        files = kg_paths(name, cfg, directory)
        _write_gzip_jsonl(
            files.nodes,
            (
                {
                    "entity_id": nid,
                    "name": data.get("name") or nid,
                    "aliases": list(data.get("aliases") or ()),
                    "doc_ids": list(data.get("doc_ids") or ()),
                    "attribute_sentences": [
                        {"doc_id": doc_id, "sent_id": sent_id, "sentence": sentence}
                        for doc_id, sent_id, sentence in data.get("attribute_sentences") or ()
                    ],
                    "mention_df": int(data.get("mention_df", 0)),
                }
                for nid, data in sorted(self.g.nodes(data=True))
            ),
        )
        _write_gzip_jsonl(
            files.edges,
            (
                {
                    "subject": u,
                    "relation": data.get("relation") or "mentions",
                    "object": v,
                    "doc_id": data.get("doc_id"),
                    "sent_id": data.get("sent_id"),
                    "sentence": data.get("sentence") or "",
                    "weight": round(float(data.get("weight", 1.0)), 6),
                }
                for u, v, data in sorted(self.g.edges(data=True), key=lambda e: (e[0], e[1]))
            ),
        )
        payload = {**self.meta, **dict(meta or {}), "dataset": name,
                   "nodes": self.g.number_of_nodes(), "edges": self.g.number_of_edges()}
        files.meta.parent.mkdir(parents=True, exist_ok=True)
        with files.meta.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, sort_keys=True)
        self.meta = payload
        return files

    @classmethod
    def load(
        cls,
        dataset: str,
        *,
        cfg: Config | None = None,
        directory: Path | None = None,
    ) -> KnowledgeGraph:
        """Load a persisted graph.

        Raises ``FileNotFoundError`` with an actionable message: a missing
        graph is a setup error, and the KG Navigator's own fallback ladder
        turns the exception into a degraded (but non-fatal) ``KGResult``.
        """
        files = kg_paths(dataset, cfg, directory)
        if not files.nodes.exists() or not files.edges.exists():
            raise FileNotFoundError(
                f"{files.nodes} / {files.edges} not found -- run "
                f"`python scripts/build_kg.py --dataset {dataset}` first"
            )
        graph = nx.DiGraph()
        for record in _read_gzip_jsonl(files.nodes):
            graph.add_node(
                record["entity_id"],
                name=record.get("name") or record["entity_id"],
                aliases=tuple(record.get("aliases") or ()),
                doc_ids=tuple(record.get("doc_ids") or ()),
                attribute_sentences=tuple(
                    (
                        str(row.get("doc_id") or ""),
                        int(row.get("sent_id", 0)),
                        str(row.get("sentence") or ""),
                    )
                    for row in record.get("attribute_sentences") or ()
                ),
                mention_df=int(record.get("mention_df", 0)),
            )
        for record in _read_gzip_jsonl(files.edges):
            graph.add_edge(
                record["subject"],
                record["object"],
                relation=record.get("relation") or "mentions",
                doc_id=record.get("doc_id"),
                sent_id=record.get("sent_id"),
                sentence=record.get("sentence") or "",
                weight=float(record.get("weight", 1.0)),
            )
        meta: dict[str, Any] = {}
        if files.meta.exists():
            with files.meta.open("r", encoding="utf-8") as fh:
                meta = json.load(fh)
        return cls(graph, dataset=dataset, meta=meta)

    # -- reporting ---------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Graph statistics for ``meta.json`` and Chapter 3 of the report."""
        n = self.g.number_of_nodes()
        m = self.g.number_of_edges()
        out_degrees = [d for _, d in self.g.out_degree()]
        in_degrees = [d for _, d in self.g.in_degree()]
        isolated = sum(1 for d in self.g.degree() if d[1] == 0)
        top_in = sorted(self.g.in_degree(), key=lambda kv: (-kv[1], kv[0]))[:10]
        relations = {}
        for _, _, data in self.g.edges(data=True):
            rel = data.get("relation") or "mentions"
            relations[rel] = relations.get(rel, 0) + 1
        top_relations = sorted(relations.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
        return {
            "nodes": n,
            "edges": m,
            "density": round(m / (n * (n - 1)), 9) if n > 1 else 0.0,
            "isolated_nodes": isolated,
            "connected_nodes": n - isolated,
            "mean_out_degree": round(sum(out_degrees) / n, 4) if n else 0.0,
            "max_out_degree": max(out_degrees) if out_degrees else 0,
            "max_in_degree": max(in_degrees) if in_degrees else 0,
            "distinct_relations": len(relations),
            "labelled_relation_edges": m - relations.get("mentions", 0),
            "labelled_relation_fraction": round((m - relations.get("mentions", 0)) / m, 4)
            if m
            else 0.0,
            "top_in_degree": [{"entity_id": k, "in_degree": v} for k, v in top_in],
            "top_relations": [{"relation": k, "count": v} for k, v in top_relations],
        }
