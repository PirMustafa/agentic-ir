"""Offline construction of the mention graph, from corpus passage text ONLY.

Read this first
---------------
2WikiMultihopQA ships gold Wikidata assertions in its qrels. Building the graph
from those would let the KG Navigator read the answer key, and every
KG-attributed gain in Chapter 4 would be an artefact -- while the scores merely
looked good. This module therefore has exactly one input: the passage corpus
(``title``, ``text``, ``sentences``) as written by ``scripts/build_corpus.py``.
It never opens the qrels, and ``tests/test_guards.py`` enforces that
structurally. Ground truth belongs to ``eval/metrics.py`` and nowhere else.
See ``docs/architecture.md`` risk 1.

What gets built
---------------
Per architecture 3.3:

* **Nodes** are normalised page titles (:func:`~agentic_ir.kg.entity_link.normalise_entity`).
* **Edges** ``A -> B`` exist when an alias of ``B`` occurs in the text of ``A``.
  Each edge stores ``doc_id``, ``sent_id`` and the verbatim containing
  sentence, so a traversal result is independently citable and therefore
  scoreable rather than decorative.
* **Relation labels** are the shortest verb-anchored window between the two
  mentions, truncated to six tokens, and ``"mentions"`` when no verb is found.
  No parser, no model: a dependency parse over 270k sentences would cost more
  than the whole rest of the pipeline and the label is only ever read by a
  human doing error analysis.
* **Weights** are ``1 / (1 + log(1 + out_degree(A)))``, so a page that mentions
  three hundred others contributes weak edges and a page that mentions two
  contributes strong ones.

Hub pruning: one refinement beyond 3.3, and the measurement behind it
--------------------------------------------------------------------
The specification's weight discounts hub *sources* -- a page that mentions
three hundred others -- but nothing discounts hub *targets*. Measured on the
real HotpotQA corpus, that gap is severe: the 549 nodes with in-degree above
100 carry **73% of all 417,881 edges**, and they are almost entirely common
English words that happen to be Wikipedia page titles -- ``one`` (in-degree
9,426), ``after`` (7,986), ``the first`` (7,469), ``two``, ``album``, ``time``,
``state``, the month names. Left in, they are what a bidirectional search meets
in the middle, and the bridge entity it reports would be noise.

So an entity mentioned by more than ``mention_df_ratio`` of the corpus has its
*incoming* mention edges deleted. Its outgoing edges and the node itself
survive, so it remains linkable and traversable from its own article.

The threshold was calibrated on ``hotpotqa_calib_50`` -- the slice that exists
precisely so that thresholds are not tuned on the evaluation 250. Metric: for
each question whose two gold supporting titles both resolve to nodes, is one
within the top-25 neighbours of the other?

===========  ==========================  ================
cap          gold pair linked            edges retained
===========  ==========================  ================
none         32/50 (64%)                 417,881 (100%)
500          32/50 (64%)                 202,382 (48%)
133          32/50 (64%)                 123,759 (30%)
100          32/50 (64%)                 111,845 (27%)
50           30/50 (60%)                 87,788 (21%)
===========  ==========================  ================

Recall is flat while seven edges in ten disappear: the pruned edges are pure
noise. ``mention_df_ratio = 0.002`` (a cap of 133 for HotpotQA, 110 for 2Wiki)
sits inside the flat region with margin. Pruning is reported in the graph
metadata, never silent.

Determinism
-----------
Passages are consumed in corpus order (the builder sorts by ``doc_id``), the
first supporting sentence wins per ordered pair, and every truncation is
applied to an already totally-ordered sequence. Two runs produce byte-identical
output files.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable, Sequence
from typing import Any

from ..types import Passage
from .entity_link import (
    STOP_ALIASES,
    AliasTable,
    Token,
    alias_variants,
    normalise_entity,
    tokenize,
)
from .graph import KnowledgeGraph

__all__ = [
    "DEFAULT_MAX_EDGES_PER_PASSAGE",
    "DEFAULT_MENTION_DF_RATIO",
    "MAX_RELATION_TOKENS",
    "MIN_MENTION_DF_CAP",
    "build_alias_table",
    "build_entities",
    "build_graph",
    "mention_df_cap",
    "prune_hub_mentions",
    "relation_label",
]

#: Relation labels are truncated to this many tokens (architecture 3.3).
MAX_RELATION_TOKENS = 6

#: Tokens of left context scanned for a verb when the source page is not itself
#: named in the sentence ("He was born in Warsaw" -- "He" is not a title).
MAX_LEFT_CONTEXT_TOKENS = 12

#: Cap on distinct outgoing edges per source passage. Reached only by list and
#: index pages, whose edges the hub weight would discount to near-nothing
#: anyway; the cap keeps a handful of pathological pages from doubling the
#: graph. Counted and reported, never silent.
DEFAULT_MAX_EDGES_PER_PASSAGE = 64

#: An entity mentioned by more than this fraction of the corpus loses its
#: incoming mention edges. Calibrated on ``hotpotqa_calib_50``; see the module
#: docstring for the sweep.
DEFAULT_MENTION_DF_RATIO = 0.002

#: Floor on the derived cap, so that a small corpus (a unit test, a ``--limit``
#: smoke run) is not pruned into an empty graph by a ratio meant for 66k pages.
MIN_MENTION_DF_CAP = 50

# The navigator needs a seed article's own dated or numeric claims for
# comparison questions. Keeping only these short snippets avoids duplicating
# the corpus inside the graph.
_ATTRIBUTE_SENTENCE_RE = re.compile(r"\d")

#: Verb forms common in encyclopedic prose. A lexicon plus two suffix rules is
#: a deliberate approximation: the label is a human-readable hint for error
#: analysis, not a typed relation, so precision matters more than recall and a
#: parser dependency is not justified.
_VERB_LEXICON = frozenset(
    """
    is are was were be been being am has have had do does did
    become became becomes include includes included contain contains containing
    star starred stars starring direct directed directs directing
    write wrote written writes producing produce produced produces
    release released releases publish published publishes
    found founded founds establish established establishes
    bear born die died dies live lived lives work worked works
    play played plays win won wins receive received receives
    name named names call called calls know known knows
    marry married marries locate located situate situated
    serve served serves join joined joins lead led leads
    create created creates develop developed develops
    appear appeared appears record recorded records perform performed performs
    own owned owns base based bases use used uses compose composed composes
    design designed designs broadcast air aired airs premiere premiered
    open opened opens close closed closes attend attended attends
    graduate graduated study studied teach taught succeed succeeded
    replace replaced replaces feature featured features
    make made makes take took taken give gave given
    """.split()  # noqa: SIM905 -- a lexicon is reviewable as prose, not as a list literal
)

#: Nouns and adjectives the ``-ed`` / ``-ing`` suffix rule would otherwise
#: mislabel. Short, because the rule already requires five characters.
_NOT_VERBS = frozenset(
    """
    during building king ring thing string spring wing swing evening morning
    something nothing everything anything ceiling sibling holding
    united limited hundred sacred hatred kindred ahmed mixed red
    """.split()  # noqa: SIM905
)

_VERB_SUFFIXES = ("ed", "ing")


def _is_verb(token: str) -> bool:
    """Heuristic verb test. Lexicon first, then a guarded suffix rule."""
    if token in _VERB_LEXICON:
        return True
    if token in _NOT_VERBS or token in STOP_ALIASES or len(token) < 5:
        return False
    return token.endswith(_VERB_SUFFIXES)


# ---------------------------------------------------------------------------
# Relation labelling
# ---------------------------------------------------------------------------

def relation_label(
    tokens: Sequence[Token],
    object_span: tuple[int, int],
    subject_spans: Sequence[tuple[int, int]] = (),
) -> str:
    """Label the edge asserted by one mention.

    ``object_span`` is the token range of the mention of ``B``;
    ``subject_spans`` are the token ranges where ``A`` itself is named in the
    same sentence (often none -- encyclopedic prose says "He was born in" after
    the first sentence). The window is the text between the nearest subject
    mention and the object mention, or the left context when the subject is not
    named. The label starts at the first verb in that window and runs at most
    :data:`MAX_RELATION_TOKENS` tokens.

    Returns ``"mentions"`` when the window holds no verb -- which is the honest
    label for "these two pages co-occur and we cannot say why".
    """
    ob_start, ob_end = object_span
    window: Sequence[Token] = ()
    best_gap: int | None = None
    for sb_start, sb_end in subject_spans:
        if sb_end <= ob_start:
            gap = ob_start - sb_end
            candidate: Sequence[Token] = tokens[sb_end:ob_start]
        elif ob_end <= sb_start:
            gap = sb_start - ob_end
            candidate = tokens[ob_end:sb_start]
        else:  # overlapping spans: the same tokens matched both entities
            continue
        if best_gap is None or gap < best_gap:
            best_gap, window = gap, candidate
    if best_gap is None:
        window = tokens[max(0, ob_start - MAX_LEFT_CONTEXT_TOKENS):ob_start]

    for i, token in enumerate(window):
        if _is_verb(token.norm):
            return " ".join(t.norm for t in window[i:i + MAX_RELATION_TOKENS])
    return "mentions"


# ---------------------------------------------------------------------------
# Pass 1: nodes and aliases
# ---------------------------------------------------------------------------

def build_entities(
    passages: Iterable[Passage],
    *,
    progress: bool = False,
    total: int | None = None,
) -> tuple[KnowledgeGraph, dict[str, Any]]:
    """Create one node per normalised title and collect its surface aliases."""
    graph = KnowledgeGraph()
    stats = {"passages": 0, "merged_titles": 0}
    seen_titles: dict[str, str] = {}
    stream: Iterable[Passage] = passages
    if progress:
        from tqdm import tqdm

        stream = tqdm(passages, total=total, desc="kg nodes", unit="doc")
    for passage in stream:
        stats["passages"] += 1
        entity_id = normalise_entity(passage.title)
        if entity_id in seen_titles and seen_titles[entity_id] != passage.title:
            stats["merged_titles"] += 1
        seen_titles.setdefault(entity_id, passage.title)
        graph.add_entity(
            entity_id,
            name=seen_titles[entity_id],
            aliases=alias_variants(passage.title),
            doc_ids=(passage.doc_id,),
            attribute_sentences=tuple(
                (passage.doc_id, sent_id, sentence)
                for sent_id, sentence in enumerate(passage.sentences)
                if _ATTRIBUTE_SENTENCE_RE.search(sentence)
            ),
        )
    stats["nodes"] = len(graph)
    return graph, stats


def build_alias_table(graph: KnowledgeGraph) -> tuple[AliasTable, dict[str, int]]:
    """Register every node alias in the trie, counting rejections."""
    table = AliasTable()
    accepted = rejected = 0
    for entity_id, data in graph.g.nodes(data=True):
        for surface in data.get("aliases") or ():
            if table.add(surface, entity_id):
                accepted += 1
            else:
                rejected += 1
    return table, {
        "aliases_accepted": accepted,
        "aliases_rejected": rejected,
        "distinct_alias_keys": len(table),
        "ambiguous_alias_keys": sum(
            1 for key in table.aliases() if len(table.lookup(key)) > 1
        ),
    }


# ---------------------------------------------------------------------------
# Hub pruning
# ---------------------------------------------------------------------------

def mention_df_cap(n_passages: int, ratio: float = DEFAULT_MENTION_DF_RATIO) -> int:
    """Absolute mention-frequency cap for a corpus of ``n_passages``.

    Expressed as a ratio so the same setting means the same thing on both
    datasets, with a floor so that small corpora survive it intact.
    """
    if ratio <= 0:
        return 0  # 0 disables pruning; see prune_hub_mentions
    return max(MIN_MENTION_DF_CAP, int(ratio * n_passages))


def prune_hub_mentions(graph: KnowledgeGraph, *, cap: int) -> dict[str, Any]:
    """Delete incoming mention edges of entities mentioned at least ``cap`` times.

    In-degree here *is* the mention document frequency: edges are deduplicated
    per ordered pair, so one in-edge means one source passage. ``cap <= 0``
    disables pruning entirely, which is what the no-pruning row of the
    calibration table was produced with.

    The node survives with its outgoing edges, so a hub is still linkable from
    a question and still traversable from its own article -- what is removed is
    the claim that every page mentioning "one" is meaningfully related to it.
    """
    if cap <= 0:
        return {"mention_df_cap": 0, "hub_nodes_pruned": 0, "edges_pruned": 0,
                "pruned_hubs": []}
    hubs = sorted(
        ((nid, deg) for nid, deg in graph.g.in_degree() if deg >= cap),
        key=lambda kv: (-kv[1], kv[0]),
    )
    removed = 0
    for nid, _deg in hubs:
        edges = list(graph.g.in_edges(nid))
        removed += len(edges)
        graph.g.remove_edges_from(edges)
    return {
        "mention_df_cap": cap,
        "hub_nodes_pruned": len(hubs),
        "edges_pruned": removed,
        "pruned_hubs": [{"entity_id": nid, "mention_df": deg} for nid, deg in hubs[:20]],
    }


# ---------------------------------------------------------------------------
# Pass 2: edges
# ---------------------------------------------------------------------------

def build_graph(
    passages: Sequence[Passage] | Iterable[Passage],
    *,
    dataset: str | None = None,
    max_edges_per_passage: int = DEFAULT_MAX_EDGES_PER_PASSAGE,
    mention_df_ratio: float = DEFAULT_MENTION_DF_RATIO,
    progress: bool = False,
) -> tuple[KnowledgeGraph, dict[str, Any]]:
    """Build the whole graph from passage text. Returns ``(graph, stats)``.

    Two passes: nodes and aliases first, because an edge can only be drawn once
    every possible target is registered in the trie; then a scan of every
    sentence, matching aliases and drawing an edge per distinct target. Hub
    mention edges are pruned next, and only then are out-degree weights
    applied -- pruning changes out-degree, and a weight computed before it
    would describe a graph that no longer exists.
    """
    started = time.perf_counter()
    materialised: Sequence[Passage] = (
        passages if isinstance(passages, Sequence) else list(passages)
    )
    total = len(materialised)

    graph, node_stats = build_entities(materialised, progress=progress, total=total)
    table, alias_stats = build_alias_table(graph)

    stream: Iterable[Passage] = materialised
    if progress:
        from tqdm import tqdm

        stream = tqdm(materialised, total=total, desc="kg edges", unit="doc")

    sentences_scanned = 0
    self_mentions = 0
    capped_passages = 0
    dropped_over_cap = 0

    for passage in stream:
        subject = normalise_entity(passage.title)
        targets: set[str] = set()
        capped = False
        for sent_id, sentence in enumerate(passage.sentences):
            sentences_scanned += 1
            if not sentence:
                continue
            tokens = tokenize(sentence)
            if not tokens:
                continue
            matches = table.match_tokens(tokens)
            if not matches:
                continue
            subject_spans = [
                (m.token_start, m.token_end) for m in matches if subject in m.entity_ids
            ]
            for match in matches:
                for target in match.entity_ids:
                    if target == subject:
                        self_mentions += 1
                        continue
                    if target not in targets:
                        if len(targets) >= max_edges_per_passage:
                            dropped_over_cap += 1
                            capped = True
                            continue
                        targets.add(target)
                    graph.add_mention_edge(
                        subject,
                        target,
                        relation=relation_label(
                            tokens, (match.token_start, match.token_end), subject_spans
                        ),
                        doc_id=passage.doc_id,
                        sent_id=sent_id,
                        sentence=sentence,
                    )
        if capped:
            capped_passages += 1

    edges_before_pruning = graph.num_edges
    # Mention frequency must be read before the first deletion, so that a node
    # whose in-edges are removed still reports how often it was actually
    # mentioned -- the navigator uses it to avoid seeding on a hub.
    mention_df = {nid: deg for nid, deg in graph.g.in_degree() if deg}
    prune_stats = prune_hub_mentions(
        graph, cap=mention_df_cap(total, mention_df_ratio)
    )
    for nid, deg in mention_df.items():
        graph.g.nodes[nid]["mention_df"] = deg
    graph.apply_hub_weights()
    graph.dataset = dataset

    stats: dict[str, Any] = {
        **node_stats,
        **alias_stats,
        **prune_stats,
        "mention_df_ratio": mention_df_ratio,
        "edges_before_pruning": edges_before_pruning,
        "sentences_scanned": sentences_scanned,
        "self_mentions_skipped": self_mentions,
        "passages_hitting_edge_cap": capped_passages,
        "mentions_dropped_over_cap": dropped_over_cap,
        "max_edges_per_passage": max_edges_per_passage,
        "build_seconds": round(time.perf_counter() - started, 2),
        **graph.stats(),
    }
    return graph, stats
