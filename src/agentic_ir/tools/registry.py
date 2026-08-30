"""The executable tool registry (architecture section 3.2).

One dict of :class:`ToolSpec`. The ``description`` strings are what get
injected into ``retriever.select_tool.v1``, so the LLM-facing documentation and
the executable tools cannot drift apart -- there is exactly one place where a
tool is described, and it is the same place it is defined.

Nothing here decides *which* tool to use; that is the Retrieval Agent's
decision table (section 5.1). This module only says what exists, how to call
it, and what it returns.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ..config import Config, load_config
from ..types import ScoredPassage, ToolName

__all__ = [
    "SEARCH_TOOLS",
    "SearchBackend",
    "ToolRegistry",
    "ToolSpec",
    "build_registry",
]

ToolKind = Literal["search", "rerank", "kg"]

#: The three retrieval channels rule R1..R7 may select between.
SEARCH_TOOLS: tuple[ToolName, ...] = ("bm25_search", "dense_search", "hybrid_search")


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolSpec:
    """One callable tool, its JSON argument schema, and its documentation."""

    name: ToolName
    fn: Callable[..., Any]
    arg_schema: dict[str, Any]
    description: str
    kind: ToolKind = "search"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


_QUERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "top_k": {"type": "integer", "minimum": 1},
    },
    "required": ["query"],
}

_DESCRIPTIONS: dict[str, str] = {
    "bm25_search": (
        "Exact lexical match (BM25). Best for short, proper-noun-dense queries, "
        "quoted spans, rare terms, and page titles. Cannot match paraphrase."
    ),
    "dense_search": (
        "Semantic match (bge-small embeddings, cosine). Best when the query uses "
        "words the corpus does not, or when terms are out of vocabulary. Weak on "
        "rare exact identifiers such as dates and numbers."
    ),
    "hybrid_search": (
        "BM25 and dense fused by reciprocal rank fusion. The safe default: it "
        "keeps lexical precision while tolerating paraphrase. Use when a query "
        "mixes one precise entity with paraphrased relation text."
    ),
    "rerank": (
        "Cross-encoder reranking of an existing candidate pool. Applied after a "
        "search tool, and only when the fused top-1 is not already decisively "
        "ahead."
    ),
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """A name-to-:class:`ToolSpec` mapping with a stable iteration order."""

    def __init__(self, specs: Sequence[ToolSpec] = ()) -> None:
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: ToolSpec) -> ToolRegistry:
        self._specs[spec.name] = spec
        return self

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __iter__(self) -> Iterator[ToolSpec]:
        return iter(self._specs[name] for name in sorted(self._specs))

    def __len__(self) -> int:
        return len(self._specs)

    def __repr__(self) -> str:
        return f"ToolRegistry({', '.join(sorted(self._specs)) or 'empty'})"

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def search_names(self) -> tuple[ToolName, ...]:
        """Only the tools rule R1..R7 is allowed to route to, in table order."""
        return tuple(n for n in SEARCH_TOOLS if n in self._specs)

    def has_search(self) -> bool:
        return bool(self.search_names())

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        spec = self._specs.get(name)
        if spec is None:
            raise KeyError(f"no tool {name!r} registered; have {self.names()}")
        return spec.fn(*args, **kwargs)

    def describe(self, names: Sequence[str] | None = None) -> str:
        """The tool documentation block injected into the routing prompt.

        Rendered from the same objects that execute, which is the single-source
        property section 3.2 asks for.
        """
        wanted = list(names) if names is not None else list(self.names())
        return "\n".join(
            f"- {n}: {self._specs[n].description}" for n in wanted if n in self._specs
        )

    def to_schemas(self) -> list[dict[str, Any]]:
        """Ollama-style tool declarations, for the tool-calling path."""
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.arg_schema,
                },
            }
            for spec in self
        ]


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

@dataclass
class SearchBackend:
    """The index objects the search tools close over.

    Every field is optional so that a registry can be built over whatever is
    actually available: an ablation with no dense channel, a unit test with a
    stub, or a full hybrid index. :meth:`registry` simply omits the tools it
    cannot serve, and the Retrieval Agent's rules fall through to what is left.
    """

    corpus: Any = None
    bm25: Any = None
    dense: Any = None
    hybrid: Any = None
    reranker: Any = None
    top_k: int = 10
    candidate_k: int = 50
    extras: dict[str, ToolSpec] = field(default_factory=dict)

    @classmethod
    def from_indexes(
        cls,
        *,
        corpus: Any = None,
        bm25: Any = None,
        dense: Any = None,
        hybrid: Any = None,
        reranker: Any = None,
        cfg: Config | None = None,
    ) -> SearchBackend:
        """Build from already-loaded indexes, taking depths from the config."""
        cfg = cfg or load_config()
        if hybrid is not None:
            bm25 = bm25 if bm25 is not None else getattr(hybrid, "bm25", None)
            dense = dense if dense is not None else getattr(hybrid, "dense", None)
        return cls(
            corpus=corpus,
            bm25=bm25,
            dense=dense,
            hybrid=hybrid,
            reranker=reranker,
            top_k=int(cfg.get("retrieval.top_k", 10)),
            candidate_k=int(cfg.get("retrieval.rerank.top_n", 50)),
        )

    # -- the tools ---------------------------------------------------------
    def _bm25_search(self, query: str, top_k: int | None = None) -> list[ScoredPassage]:
        return list(self.bm25.search(query, top_k or self.candidate_k))

    def _dense_search(self, query: str, top_k: int | None = None) -> list[ScoredPassage]:
        return list(self.dense.search(query, top_k or self.candidate_k))

    def _hybrid_search(self, query: str, top_k: int | None = None) -> list[ScoredPassage]:
        return list(
            self.hybrid.search(query, top_k or self.candidate_k, candidate_k=self.candidate_k)
        )

    def _rerank(
        self,
        query: str,
        passages: Sequence[ScoredPassage],
        *,
        top_n: int | None = None,
        force: bool = False,
    ) -> Any:
        return self.reranker.rerank(query, list(passages), top_n=top_n, force=force)

    def registry(self) -> ToolRegistry:
        """A registry over exactly the channels this backend can serve."""
        specs: list[ToolSpec] = []
        if self.bm25 is not None:
            specs.append(_spec("bm25_search", self._bm25_search))
        if self.dense is not None:
            specs.append(_spec("dense_search", self._dense_search))
        if self.hybrid is not None:
            specs.append(_spec("hybrid_search", self._hybrid_search))
        if self.reranker is not None:
            specs.append(
                ToolSpec(
                    name="rerank",
                    fn=self._rerank,
                    arg_schema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_n": {"type": "integer", "minimum": 1},
                        },
                        "required": ["query"],
                    },
                    description=_DESCRIPTIONS["rerank"],
                    kind="rerank",
                )
            )
        specs.extend(self.extras.values())
        return ToolRegistry(specs)


def _spec(name: ToolName, fn: Callable[..., Any]) -> ToolSpec:
    return ToolSpec(
        name=name,
        fn=fn,
        arg_schema=dict(_QUERY_SCHEMA),
        description=_DESCRIPTIONS[name],
        kind="search",
    )


def build_registry(
    *,
    corpus: Any = None,
    bm25: Any = None,
    dense: Any = None,
    hybrid: Any = None,
    reranker: Any = None,
    cfg: Config | None = None,
    extra_tools: Mapping[str, ToolSpec] | None = None,
) -> ToolRegistry:
    """Convenience wrapper: indexes in, ready-to-call registry out.

    ``extra_tools`` is the seam for the KG tools, which live in a package this
    milestone does not own; the orchestrator injects them when they exist.
    """
    backend = SearchBackend.from_indexes(
        corpus=corpus, bm25=bm25, dense=dense, hybrid=hybrid, reranker=reranker, cfg=cfg
    )
    if extra_tools:
        backend.extras.update(extra_tools)
    return backend.registry()
