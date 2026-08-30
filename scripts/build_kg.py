"""Build the passage-mention knowledge graph.

    python scripts/build_kg.py --dataset all

Reads ``data/processed/{dataset}_corpus.jsonl`` and writes, per dataset, into
``data/indexes/``:

* ``{dataset}_kg_nodes.jsonl.gz``   one node per line: id, name, aliases, doc_ids
* ``{dataset}_kg_edges.jsonl.gz``   one edge per line, each carrying the
  ``doc_id``, ``sent_id`` and verbatim sentence that licensed it
* ``{dataset}_kg_meta.json``        the statistics quoted in Chapter 3

The input is the corpus and only the corpus
-------------------------------------------
2WikiMultihopQA ships gold Wikidata assertions alongside its questions. If the
graph were built from those, the KG Navigator would be reading the answer key
and every KG-attributed gain in the report would be an artefact. This script
opens ``{dataset}_corpus.jsonl`` and nothing else; the qrels are the scorer's
input, never the builder's. See ``docs/architecture.md`` risk 1, enforced by
``tests/test_guards.py``.

Runtime is a few minutes per dataset on one core -- it is a linear scan of
every sentence with a trie lookup per token position. There is nothing to
parallelise that would survive the determinism requirement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agentic_ir.config import Paths, load_config  # noqa: E402
from agentic_ir.indexing.corpus import DATASETS, Corpus, corpus_path  # noqa: E402
from agentic_ir.kg.build import (  # noqa: E402
    DEFAULT_MAX_EDGES_PER_PASSAGE,
    DEFAULT_MENTION_DF_RATIO,
    build_graph,
)
from agentic_ir.kg.graph import kg_paths  # noqa: E402

#: Statistics printed as the headline summary, in report order.
_HEADLINE = (
    "passages", "nodes", "edges", "merged_titles", "distinct_alias_keys",
    "ambiguous_alias_keys", "aliases_rejected", "sentences_scanned",
    "edges_before_pruning", "mention_df_cap", "hub_nodes_pruned", "edges_pruned",
    "mean_out_degree", "max_out_degree", "max_in_degree", "isolated_nodes",
    "density", "distinct_relations", "labelled_relation_fraction",
    "passages_hitting_edge_cap", "build_seconds",
)


def rule(title: str = "") -> None:
    """A visual separator, so a multi-dataset run stays readable."""
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)) if title else "=" * 66)


def build_one(
    dataset: str,
    *,
    limit: int | None,
    max_edges_per_passage: int,
    mention_df_ratio: float,
    out_dir: Path | None,
    progress: bool,
) -> dict[str, Any]:
    """Build, persist and summarise one dataset's graph."""
    cfg = load_config()
    source = corpus_path(dataset, cfg)  # type: ignore[arg-type]
    rule(f"{dataset}: {source.name}")
    corpus = Corpus.load(dataset, cfg=cfg, progress=progress)  # type: ignore[arg-type]
    passages = list(corpus)
    if limit is not None:
        passages = passages[:limit]
        print(f"  --limit {limit}: building over a {len(passages)}-passage slice")
    print(f"  passages: {len(passages)}")

    graph, stats = build_graph(
        passages,
        dataset=dataset,
        max_edges_per_passage=max_edges_per_passage,
        mention_df_ratio=mention_df_ratio,
        progress=progress,
    )
    files = graph.save(
        dataset=dataset,
        directory=out_dir,
        cfg=cfg,
        meta={
            "source_corpus": str(source),
            "source_passages": len(passages),
            "limited": limit is not None,
            "stats": stats,
        },
    )
    for key in _HEADLINE:
        if key in stats:
            print(f"  {key:<30} {stats[key]}")
    print("  pruned as hubs (mention_df before pruning):")
    for row in stats.get("pruned_hubs", [])[:5]:
        print(f"    {row['mention_df']:>7}  {row['entity_id']}")
    print("  top in-degree after pruning:")
    for row in stats.get("top_in_degree", [])[:5]:
        print(f"    {row['in_degree']:>7}  {row['entity_id']}")
    print("  most frequent relation labels:")
    for row in stats.get("top_relations", [])[:8]:
        print(f"    {row['count']:>7}  {row['relation']}")
    print(f"  wrote {files.nodes.name}, {files.edges.name}, {files.meta.name}")
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset", default="all", choices=[*DATASETS, "all"],
        help="dataset to build; 'all' builds every configured dataset",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="build over the first N passages only (smoke tests; marks meta as limited)",
    )
    parser.add_argument(
        "--max-edges-per-passage", type=int, default=DEFAULT_MAX_EDGES_PER_PASSAGE,
        help="cap on distinct outgoing edges per source passage",
    )
    parser.add_argument(
        "--mention-df-ratio", type=float, default=DEFAULT_MENTION_DF_RATIO,
        help="prune incoming edges of entities mentioned by more than this "
             "fraction of the corpus; 0 disables pruning",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="override the output directory (default: paths.indexes)",
    )
    parser.add_argument("--no-progress", action="store_true", help="suppress tqdm bars")
    parser.add_argument(
        "--stats-only", action="store_true",
        help="print the stats of an already-built graph and exit",
    )
    args = parser.parse_args(argv)

    cfg = load_config()
    Paths.from_config(cfg)  # ensure data/indexes exists
    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]

    if args.stats_only:
        for dataset in datasets:
            files = kg_paths(dataset, cfg, args.out_dir)
            rule(dataset)
            if not files.meta.exists():
                print(f"  no graph at {files.meta}")
                continue
            with files.meta.open("r", encoding="utf-8") as fh:
                print(json.dumps(json.load(fh), indent=2)[:4000])
        return 0

    for dataset in datasets:
        build_one(
            dataset,
            limit=args.limit,
            max_edges_per_passage=args.max_edges_per_passage,
            mention_df_ratio=args.mention_df_ratio,
            out_dir=args.out_dir,
            progress=not args.no_progress,
        )
    rule("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
