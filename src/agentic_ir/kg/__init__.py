"""The knowledge-graph layer: build it offline, traverse it at query time.

The graph is a *mention* graph over corpus page titles, built by
:mod:`agentic_ir.kg.build` from passage text and nothing else. Gold Wikidata
assertions shipped with 2WikiMultihopQA are the property of ``eval/metrics.py``
and are never read here -- see ``docs/architecture.md`` risk 1, and the
structural guard in ``tests/test_guards.py``.

Four modules, in dependency order:

``entity_link``
    Normalisation, alias variants, and the longest-match alias trie. Shared by
    the builder and the navigator so their notions of identity cannot drift.
``graph``
    :class:`KnowledgeGraph` -- a ``networkx.DiGraph`` with typed accessors,
    deterministic neighbour ordering, and gzipped JSONL persistence.
``build``
    The two-pass offline construction, plus the verb-anchored relation
    labeller.
``traverse``
    Bounded BFS and the bidirectional bridge search.
"""

from __future__ import annotations

from .build import build_alias_table, build_entities, build_graph, relation_label
from .entity_link import AliasMatch, AliasTable, alias_variants, fold, normalise_entity, tokenize
from .graph import KnowledgeGraph, kg_paths
from .traverse import Neighborhood, bfs_neighborhood, bidirectional_bfs, bridge_paths, path_score

__all__ = [
    "AliasMatch",
    "AliasTable",
    "KnowledgeGraph",
    "Neighborhood",
    "alias_variants",
    "bfs_neighborhood",
    "bidirectional_bfs",
    "bridge_paths",
    "build_alias_table",
    "build_entities",
    "build_graph",
    "fold",
    "kg_paths",
    "normalise_entity",
    "path_score",
    "relation_label",
    "tokenize",
]
