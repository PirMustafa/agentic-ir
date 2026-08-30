"""Graph traversal: bounded BFS, and the bidirectional search for a bridge.

``docs/architecture.md`` 3.3 calls bidirectional BFS "the single highest-value
operation in the whole KG agent", and the reason is worth restating because it
shapes every choice here. On a bridge question -- *"When was the director of
Titanic born?"* -- the entity that joins the two hops is almost always a page
title. The graph therefore already contains the bridge; finding it is a search
over two frontiers that meet in the middle, which completes in milliseconds,
where asking an 8B model to guess it costs several seconds and can be wrong.

Three decisions, and why
------------------------
**Search is undirected, evidence is directed.** An edge's direction records
which page's author happened to write the mention down; for path finding
"Titanic mentions Cameron" and "Cameron mentions Titanic" are the same fact.
So the frontier expands over ``neighbors()`` (both directions), while every
:class:`~agentic_ir.types.Triple` attached to the resulting path keeps the
direction actually stored, so the citation never misreports who asserted what.

**Every expansion is capped and ordered.** A node is expanded to at most
``max_neighbors`` neighbours in ``(-weight, entity_id)`` order. Without the cap
a two-hop BFS from a hub is a scan of the corpus; without the total order the
truncation would differ between runs of the same configuration.

**Path score is the geometric mean of edge weights.** "Product of edge weights,
normalised" (3.3) has to be normalised by length or a three-hop path can never
outrank a one-hop path however strong its edges. The geometric mean
``prod(w) ** (1/hops)`` is the length-invariant form, and it keeps the score in
the same ``(0, 1]`` range as a single weight.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ..types import KGPath, Triple
from .graph import KnowledgeGraph

__all__ = [
    "Neighborhood",
    "bfs_neighborhood",
    "bidirectional_bfs",
    "bridge_paths",
    "path_from_nodes",
    "path_score",
]

#: Hard ceiling on nodes touched by one traversal. The per-node cap already
#: bounds the branching factor; this bounds the pathological case where a dense
#: cluster is entirely below the cap. Never expected to bind on these corpora.
MAX_VISITED = 20_000


@dataclass(frozen=True, slots=True)
class Neighborhood:
    """The result of a single-seed BFS."""

    seed: str
    nodes_by_hop: tuple[tuple[str, ...], ...]   # index 0 == 1 hop out
    triples: tuple[Triple, ...]                 # the edges actually traversed

    @property
    def nodes(self) -> tuple[str, ...]:
        """Every reached node, nearest hop first, order-preserving."""
        out: list[str] = []
        seen: set[str] = set()
        for level in self.nodes_by_hop:
            for node in level:
                if node not in seen:
                    seen.add(node)
                    out.append(node)
        return tuple(out)


def path_score(edges: tuple[Triple, ...]) -> float:
    """Geometric mean of the edge weights; ``0.0`` for an empty path."""
    if not edges:
        return 0.0
    product = 1.0
    for edge in edges:
        product *= max(edge.weight, 1e-12)
    return float(product ** (1.0 / len(edges)))


def path_from_nodes(
    graph: KnowledgeGraph,
    nodes: tuple[str, ...],
    *,
    bridge_entity: str | None = None,
) -> KGPath | None:
    """Materialise a node sequence into a scored :class:`KGPath`.

    Returns ``None`` if any consecutive pair is not joined by a stored edge in
    either direction -- which would mean the caller invented a path.
    """
    if len(nodes) < 2:
        return None
    edges: list[Triple] = []
    for u, v in zip(nodes, nodes[1:], strict=False):
        triple, _reversed = graph.any_triple(u, v)
        if triple is None:
            return None
        edges.append(triple)
    return KGPath(
        nodes=nodes,
        edges=tuple(edges),
        hops=len(edges),
        score=round(path_score(tuple(edges)), 6),
        bridge_entity=bridge_entity,
    )


# ---------------------------------------------------------------------------
# Single seed
# ---------------------------------------------------------------------------

def bfs_neighborhood(
    graph: KnowledgeGraph,
    seed: str,
    *,
    max_hops: int = 2,
    max_neighbors: int = 25,
) -> Neighborhood:
    """Bounded BFS from one seed (architecture 3.3, one-seed branch).

    Keeps the top ``max_neighbors`` per node by ``(weight desc, entity_id
    asc)``. Nodes already reached at a shorter distance are not re-reported, so
    ``nodes_by_hop`` is a partition and the hop count of a node is its true
    shortest distance under the capped expansion.
    """
    if seed not in graph:
        return Neighborhood(seed=seed, nodes_by_hop=(), triples=())
    visited = {seed}
    frontier = [seed]
    levels: list[tuple[str, ...]] = []
    triples: list[Triple] = []
    for _hop in range(max_hops):
        nxt: list[str] = []
        for node in frontier:
            for neighbour in graph.neighbors(node, limit=max_neighbors):
                triple, _reversed = graph.any_triple(node, neighbour)
                if triple is not None:
                    triples.append(triple)
                if neighbour in visited or len(visited) >= MAX_VISITED:
                    continue
                visited.add(neighbour)
                nxt.append(neighbour)
        if not nxt:
            break
        levels.append(tuple(nxt))
        frontier = nxt
    # Deterministic, and the strongest evidence first.
    triples.sort(key=lambda t: (-t.weight, t.subject, t.object))
    deduped: list[Triple] = []
    seen_pairs: set[tuple[str, str]] = set()
    for triple in triples:
        key = (triple.subject, triple.object)
        if key not in seen_pairs:
            seen_pairs.add(key)
            deduped.append(triple)
    return Neighborhood(seed=seed, nodes_by_hop=tuple(levels), triples=tuple(deduped))


# ---------------------------------------------------------------------------
# Two seeds: the bridge search
# ---------------------------------------------------------------------------

def bidirectional_bfs(
    graph: KnowledgeGraph,
    source: str,
    target: str,
    *,
    max_hops: int = 2,
    max_neighbors: int = 25,
) -> KGPath | None:
    """Shortest connecting path, searched from both ends at once.

    ``max_hops`` bounds *each* side, so the longest path returned has
    ``2 * max_hops`` edges. Frontiers alternate, smaller first, and the search
    stops at the first meeting node -- which, level-synchronous BFS being what
    it is, lies on a shortest path.

    ``KGPath.bridge_entity`` is set to that meeting node, per the type
    contract. Callers wanting something to *substitute into a query* should
    prefer an intermediate node of ``KGPath.nodes`` (see
    :func:`agentic_ir.agents.kg_navigator.intermediate_entity`), because when
    the two seeds are directly adjacent the meeting node is a seed itself.
    """
    if source not in graph or target not in graph or source == target:
        return None

    parents_f: dict[str, str | None] = {source: None}
    parents_b: dict[str, str | None] = {target: None}
    frontier_f: deque[str] = deque([source])
    frontier_b: deque[str] = deque([target])
    depth_f = depth_b = 0

    def _expand(
        frontier: deque[str],
        parents: dict[str, str | None],
        other: dict[str, str | None],
    ) -> str | None:
        """Expand one whole level; return a meeting node if one is reached."""
        meeting: str | None = None
        for _ in range(len(frontier)):
            node = frontier.popleft()
            for neighbour in graph.neighbors(node, limit=max_neighbors):
                if neighbour in parents:
                    continue
                parents[neighbour] = node
                frontier.append(neighbour)
                if neighbour in other and (meeting is None or neighbour < meeting):
                    meeting = neighbour
            if len(parents) >= MAX_VISITED:
                break
        return meeting

    if target in graph.neighbors(source, limit=max_neighbors) or source in graph.neighbors(
        target, limit=max_neighbors
    ):
        return path_from_nodes(graph, (source, target), bridge_entity=target)

    while (depth_f < max_hops or depth_b < max_hops) and (frontier_f or frontier_b):
        forward = len(frontier_f) <= len(frontier_b)
        if forward and depth_f >= max_hops:
            forward = False
        elif not forward and depth_b >= max_hops:
            forward = True
        if forward:
            if depth_f >= max_hops or not frontier_f:
                break
            depth_f += 1
            meeting = _expand(frontier_f, parents_f, parents_b)
        else:
            if depth_b >= max_hops or not frontier_b:
                break
            depth_b += 1
            meeting = _expand(frontier_b, parents_b, parents_f)
        if meeting is not None:
            nodes = _join(meeting, parents_f, parents_b)
            return path_from_nodes(graph, nodes, bridge_entity=meeting)
    return None


def _join(
    meeting: str,
    parents_f: dict[str, str | None],
    parents_b: dict[str, str | None],
) -> tuple[str, ...]:
    """Stitch the two half-paths into one source-to-target node sequence."""
    left: list[str] = []
    node: str | None = meeting
    while node is not None:
        left.append(node)
        node = parents_f.get(node)
    left.reverse()
    right: list[str] = []
    node = parents_b.get(meeting)
    while node is not None:
        right.append(node)
        node = parents_b.get(node)
    return tuple(left + right)


def bridge_paths(
    graph: KnowledgeGraph,
    seeds: tuple[str, ...],
    *,
    max_hops: int = 2,
    max_neighbors: int = 25,
    max_paths: int = 3,
) -> tuple[KGPath, ...]:
    """Connect every ordered seed pair, best paths first.

    Ordered by ``(-score, hops, nodes)`` so the ranking is total and a re-run
    returns the same paths in the same order.
    """
    found: dict[tuple[str, ...], KGPath] = {}
    for i, source in enumerate(seeds):
        for target in seeds[i + 1:]:
            path = bidirectional_bfs(
                graph, source, target, max_hops=max_hops, max_neighbors=max_neighbors
            )
            if path is not None and path.nodes not in found:
                found[path.nodes] = path
    ranked = sorted(found.values(), key=lambda p: (-p.score, p.hops, p.nodes))
    return tuple(ranked[:max_paths])
