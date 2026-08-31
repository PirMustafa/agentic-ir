"""The evaluation harness: one configuration, one dataset, one run directory.

Specification: ``docs/architecture.md`` sections 6 (artefacts) and 9
(reproducibility). This module produces every number Chapter 4 reports, so the
three properties it is built around are all about auditability rather than
speed.

**It checkpoints.** ``traces.jsonl`` is appended per question and a restart
skips the qids already in it. One agentic configuration on one dataset is
roughly 1.7 hours (architecture risk 2); a crash at question 194 that forced a
restart at question 1 would cost a day of the schedule, so resume is a
requirement and not a convenience.

**It never dies on one question.** ``Orchestrator.run`` already promises not to
raise, but index loads, a dropped Ollama connection and a torn write do not go
through it. Every question is wrapped, the failure is recorded *in the trace*
with ``terminated_by = "harness_error"``, and the run continues. A failed
question still gets a record, so it is checkpointed rather than retried
forever; ``--retry-failed`` is the deliberate opt-in to try it again.

**It records what makes latency comparable.** ``meta.json`` carries the config
snapshot, library versions, GPU, Ollama model digest, seed, the SHA-256 of the
frozen eval slice, and ``cache_cold``. Latency figures in the report are only
valid from a cold-cache run, and a per-leg record of cache state is what lets a
reader check that claim instead of trusting it.

Scoring lives here too (:func:`score_records`), but the scores are written to
``scores.csv`` as a *derived* artefact: they are recomputed from the trace by
``eval/tables.py`` rather than trusted, so fixing a metric never means
re-running the model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import os
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT, Config, load_config
from ..indexing.corpus import Corpus, eval_set_path, load_eval_set, make_doc_id
from ..state import QuestionState
from ..trace import TraceWriter, iter_records, make_run_id
from ..types import GoldAnswer
from .metrics import retrieval_metrics_per_query, score_question

__all__ = [
    "ABLATION_OVERRIDES",
    "AGENTIC_CONFIGS",
    "BASELINE_CONFIGS",
    "BASELINE_MODULES",
    "CONFIGURATIONS",
    "Pipeline",
    "RunResult",
    "RunSpec",
    "build_system",
    "config_for",
    "configurations",
    "gold_doc_ids",
    "latest_run_dir",
    "load_baseline",
    "load_pipeline",
    "main",
    "predicted_supporting_facts",
    "question_ranking",
    "run_eval",
    "score_records",
    "seed_everything",
    "write_scores_csv",
]

#: The nine systems of ``evaluation.configurations``. Kept as a literal so this
#: module has a meaning before the config is loaded (argparse choices, tests).
CONFIGURATIONS: tuple[str, ...] = (
    "bm25_only", "dense_only", "hybrid_rerank", "naive_rag", "self_ask",
    "agentic_full", "agentic_no_planner", "agentic_no_kg", "agentic_no_verifier",
)

#: Configurations driven by :class:`~agentic_ir.orchestrator.Orchestrator`.
AGENTIC_CONFIGS: tuple[str, ...] = (
    "agentic_full", "agentic_no_planner", "agentic_no_kg", "agentic_no_verifier",
)

#: Configurations owned by ``baselines/`` -- a sibling milestone. Imported
#: lazily, by name, so this module runs before they land.
BASELINE_CONFIGS: tuple[str, ...] = (
    "bm25_only", "dense_only", "hybrid_rerank", "naive_rag", "self_ask",
)

#: Dotted-path config overrides that define each ablation. ``agentic_no_planner``
#: has no config switch -- the planner is replaced by :class:`NoPlanner` in
#: :func:`build_system` -- so it appears here only for the record.
ABLATION_OVERRIDES: dict[str, dict[str, Any]] = {
    "agentic_full": {},
    "agentic_no_planner": {"agents.planner.template_shortcut": False},
    "agentic_no_kg": {"agents.kg.enabled": False},
    "agentic_no_verifier": {"agents.verifier.enabled": False},
}

#: ``config_name -> (module, candidate factory names)``. The factory names are a
#: list rather than one name for the same reason the orchestrator's optional
#: agents are: this milestone does not own ``baselines/`` and must not dictate
#: its spelling. See :func:`load_baseline` for the full seam contract.
BASELINE_MODULES: dict[str, tuple[str, tuple[str, ...]]] = {
    "bm25_only": (".baselines.bm25_only", ("BM25Only", "BM25OnlyBaseline", "Baseline", "build")),
    "dense_only": (".baselines.dense_only", ("DenseOnly", "DenseOnlyBaseline", "Baseline", "build")),
    "hybrid_rerank": (
        ".baselines.hybrid_rerank",
        ("HybridRerank", "HybridRerankBaseline", "Baseline", "build"),
    ),
    "naive_rag": (".baselines.naive_rag", ("NaiveRAG", "NaiveRag", "NaiveRAGBaseline", "Baseline", "build")),
    "self_ask": (".baselines.self_ask", ("SelfAsk", "SelfAskBaseline", "Baseline", "build")),
}

#: Retrieval cut-offs reported in Chapter 4 (``evaluation.retrieval_metrics``).
K_VALUES: tuple[int, ...] = (2, 5, 10)


def configurations(cfg: Config | None = None) -> tuple[str, ...]:
    """The configured system list, falling back to :data:`CONFIGURATIONS`."""
    cfg = cfg or load_config()
    configured = tuple(str(c) for c in cfg.get("evaluation.configurations", ()) or ())
    return configured or CONFIGURATIONS


# ---------------------------------------------------------------------------
# Determinism (architecture section 9)
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> dict[str, Any]:
    """Seed ``random``, ``numpy`` and ``torch``; report what was reachable.

    Returns the record rather than logging it, because ``meta.json`` should be
    able to say *which* RNGs were actually seeded. A run on a machine without
    torch is still reproducible for everything that does not use torch, and the
    honest way to express that is to write it down.
    """
    record: dict[str, Any] = {"seed": seed, "seeded": ["random"], "pythonhashseed": os.environ.get("PYTHONHASHSEED")}
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
        record["seeded"].append("numpy")
    except Exception:  # noqa: BLE001 - a probe must never break a run
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)  # cuBLAS GEMM has no det. path here
        record["seeded"].append("torch")
    except Exception:  # noqa: BLE001
        pass
    return record


# ---------------------------------------------------------------------------
# Run specification and result
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RunSpec:
    """Everything that identifies one evaluation run."""

    config_name: str
    dataset: str
    limit: int | None = None
    split: str = "eval"
    size: int | None = None
    resume: str | None = None          # run_id, or "latest"
    retry_failed: bool = False
    seed: int = 42
    warm_cache: bool = False           # assert a WARM cache; never silently assumed
    run_id: str | None = None
    root: Path | None = None

    def overrides(self) -> dict[str, Any]:
        return dict(ABLATION_OVERRIDES.get(self.config_name, {}))


@dataclass(slots=True)
class RunResult:
    """What one call to :func:`run_eval` produced."""

    run_id: str
    run_dir: Path
    config_name: str
    dataset: str
    n_questions: int = 0        # in the slice, after --limit
    n_skipped: int = 0          # already checkpointed
    n_completed: int = 0        # produced a record this leg
    n_failed: int = 0           # harness_error records this leg
    elapsed_s: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        parts = [
            f"run_id={self.run_id}",
            f"questions={self.n_questions}",
            f"completed={self.n_completed}",
            f"skipped={self.n_skipped}",
            f"failed={self.n_failed}",
            f"elapsed={self.elapsed_s:.1f}s",
        ]
        for key in ("em", "f1", "sp_f1", "recall@10", "ndcg@10"):
            if key in self.scores:
                parts.append(f"{key}={self.scores[key]:.4f}")
        return "  ".join(parts)


# ---------------------------------------------------------------------------
# The system under test
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Pipeline:
    """The loaded indexes one dataset needs, shared by every configuration.

    Loading is the expensive part of a run's start-up (a 66k-passage corpus,
    a FAISS index and two cross-encoders), and every configuration wants the
    same objects, so it happens once and is passed around.
    """

    dataset: str
    corpus: Any = None
    bm25: Any = None
    dense: Any = None
    hybrid: Any = None
    reranker: Any = None
    registry: Any = None
    kg: Any = None
    notes: list[str] = field(default_factory=list)


def load_pipeline(
    dataset: str,
    cfg: Config | None = None,
    *,
    with_kg: bool = True,
    with_rerank: bool = True,
) -> Pipeline:
    """Load corpus, hybrid index, reranker and knowledge graph for ``dataset``.

    Each component is optional at the seam: a missing dense index degrades the
    registry to BM25 rather than aborting, and the Retrieval Agent's routing
    rules fall through to what is left. That is what lets `hotpotqa` be
    evaluated while `twowiki`'s indexes are still building.
    """
    cfg = cfg or load_config()
    from ..indexing.hybrid import HybridIndex
    from ..tools.registry import build_registry

    pipeline = Pipeline(dataset=dataset)
    indexes = Path(cfg.resolve_path("paths.indexes"))

    pipeline.corpus = Corpus.load(dataset, cfg=cfg)  # a missing corpus IS fatal
    try:
        pipeline.hybrid = HybridIndex.load(indexes / dataset / "hybrid", corpus=pipeline.corpus)
        pipeline.bm25 = getattr(pipeline.hybrid, "bm25", None)
        pipeline.dense = getattr(pipeline.hybrid, "dense", None)
    except Exception as exc:  # noqa: BLE001 - degrade, and say so in meta.json
        pipeline.notes.append(f"hybrid index unavailable: {type(exc).__name__}: {exc}")

    if with_rerank and bool(cfg.get("retrieval.rerank.enabled", True)):
        try:
            from ..indexing.rerank import CrossEncoderReranker

            pipeline.reranker = CrossEncoderReranker.from_config(cfg)
        except Exception as exc:  # noqa: BLE001
            pipeline.notes.append(f"reranker unavailable: {type(exc).__name__}: {exc}")

    pipeline.registry = build_registry(
        corpus=pipeline.corpus,
        bm25=pipeline.bm25,
        dense=pipeline.dense,
        hybrid=pipeline.hybrid,
        reranker=pipeline.reranker,
        cfg=cfg,
    )

    if with_kg and bool(cfg.get("agents.kg.enabled", True)):
        try:
            from ..agents.kg_navigator import KGNavigator

            pipeline.kg = KGNavigator.load(dataset, cfg=cfg, corpus=pipeline.corpus)
        except Exception as exc:  # noqa: BLE001
            pipeline.notes.append(f"kg unavailable: {type(exc).__name__}: {exc}")
    return pipeline


class NoPlanner:
    """The ``agentic_no_planner`` ablation: no decomposition, ever.

    Not the Planner's own fallback ladder -- that ladder still *decomposes*
    (F1 splits a comparison into two nodes), so using it would ablate the LLM
    call while leaving the multi-hop structure in place and the ablation would
    measure the wrong thing. This is the identity floor: one node carrying the
    whole question, routed to hybrid+rerank, which is exactly the strongest
    baseline plus synthesis and verification.
    """

    name = "planner"

    def __init__(self, cfg: Config | None = None, **_: Any) -> None:
        self.cfg = cfg or load_config()

    def run(self, state: QuestionState, *, directive: Any = None) -> Any:
        from ..agents.planner import identity_plan
        from ..types import Plan

        revision = getattr(directive, "revision", 0) if directive is not None else 0
        with state.step(
            self.name,
            question_chars=len(state.question),
            revision=revision,
            replan=directive is not None,
        ) as rec:
            state.budget.note_saved()
            nodes = tuple(identity_plan(state.question))
            plan = Plan(
                question=state.question,
                subqueries=nodes,
                strategy="single_hop",
                revision=revision,
                origin="fallback_rule",
                depth=1,
                repairs=("ablation:no_planner",),
                directive_id=getattr(directive, "directive_id", None),
            )
            rec.output_summary = {
                "strategy": plan.strategy,
                "n_subqueries": len(nodes),
                "depth": 1,
                "origin": plan.origin,
                "ablation": "no_planner",
            }
        return plan


def config_for(config_name: str, cfg: Config | None = None) -> Config:
    """The configuration an ablation runs under.

    Returns a derived :class:`Config`; the process-wide cached instance is
    never mutated, so running four ablations in one process cannot leave the
    third one describing the second one's settings.
    """
    cfg = cfg or load_config()
    overrides = ABLATION_OVERRIDES.get(config_name)
    return cfg.with_overrides(overrides) if overrides else cfg


def load_baseline(
    config_name: str,
    *,
    cfg: Config,
    dataset: str,
    pipeline: Pipeline,
    client: Any = None,
) -> Any:
    """Construct a baseline system, or return ``None`` if it does not exist yet.

    ``baselines/`` belongs to a sibling milestone, so the seam is published here
    rather than assumed. A baseline module must expose a class or factory named
    in :data:`BASELINE_MODULES`, constructible from any subset of the keywords
    ``cfg, dataset, config_name, registry, corpus, bm25, dense, hybrid,
    reranker, client`` (the unaccepted ones are dropped by signature
    inspection), and the resulting object must expose::

        run(qid: str, question: str, *, gold=None, state=None) -> QuestionState

    -- the same shape as :meth:`Orchestrator.run`, so the harness, the trace
    writer and the scorer treat every system identically.

    Absence returns ``None`` instead of raising: this module has to run, and
    be testable, before ``baselines/`` lands.
    """
    import importlib

    entry = BASELINE_MODULES.get(config_name)
    if entry is None:
        return None
    module_name, candidates = entry
    try:
        module = importlib.import_module(module_name, package="agentic_ir")
    except Exception:  # noqa: BLE001 - not written yet, or broken; both are "absent"
        return None

    factory = next((getattr(module, n) for n in candidates if hasattr(module, n)), None)
    if factory is None:
        return None

    kwargs: dict[str, Any] = {
        "cfg": cfg,
        "config": cfg,
        "dataset": dataset,
        "config_name": config_name,
        "registry": pipeline.registry,
        "corpus": pipeline.corpus,
        "bm25": pipeline.bm25,
        "dense": pipeline.dense,
        "hybrid": pipeline.hybrid,
        "reranker": pipeline.reranker,
        "client": client,
    }
    try:
        return factory(**_acceptable_kwargs(factory, kwargs))
    except Exception:  # noqa: BLE001
        try:
            return factory(cfg)
        except Exception:  # noqa: BLE001
            return None


def _acceptable_kwargs(fn: Any, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Drop keywords the callee does not declare (unless it takes ``**kwargs``)."""
    target = fn.__init__ if inspect.isclass(fn) else fn
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return dict(kwargs)
    params = signature.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def build_system(
    config_name: str,
    dataset: str,
    *,
    cfg: Config,
    pipeline: Pipeline,
    client: Any = None,
) -> Any:
    """The system under test: an orchestrator, or a baseline.

    The KG navigator, synthesizer and verifier are injected explicitly rather
    than left to the orchestrator's lazy import. The lazy path constructs a
    collaborator by trying signatures, and ``KGNavigator(cfg, client=...)``
    happens to be accepted with ``cfg`` bound to its first positional
    parameter, ``graph`` -- an object that is silently the wrong type. Passing
    the loaded navigator removes the guess.
    """
    from ..orchestrator import Orchestrator

    if config_name in BASELINE_CONFIGS:
        baseline = load_baseline(
            config_name, cfg=cfg, dataset=dataset, pipeline=pipeline, client=client
        )
        if baseline is None:
            raise SystemExit(
                f"configuration {config_name!r} needs src/agentic_ir/baselines/"
                f"{config_name}.py, which is not present (or exposes no known "
                f"factory: {', '.join(BASELINE_MODULES[config_name][1])}). "
                "It is owned by a sibling milestone; run an agentic "
                "configuration, or wait for it to land."
            )
        return baseline

    kg = pipeline.kg if bool(cfg.get("agents.kg.enabled", True)) else None
    synthesizer = verifier = None
    try:
        from ..agents.synthesizer import Synthesizer

        synthesizer = Synthesizer(cfg, client=client)
    except Exception:  # noqa: BLE001 - orchestrator degrades to extractive answers
        synthesizer = None
    if bool(cfg.get("agents.verifier.enabled", True)):
        try:
            from ..agents.verifier import Verifier

            verifier = Verifier(cfg, client=client)
        except Exception:  # noqa: BLE001
            verifier = None

    planner = NoPlanner(cfg) if config_name == "agentic_no_planner" else None
    return Orchestrator(
        pipeline.registry,
        cfg=cfg,
        config_name=config_name,
        dataset=dataset,
        planner=planner,
        kg=kg,
        synthesizer=synthesizer,
        verifier=verifier,
        client=client,
        bm25=pipeline.bm25,
    )


# ---------------------------------------------------------------------------
# Provenance probes for meta.json (architecture section 9)
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str | None:
    """SHA-256 of a file, or ``None``. Never raises: a probe is not a run."""
    try:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except Exception:  # noqa: BLE001
        return None


def model_digest(client: Any, model: str) -> str | None:
    """The Ollama blob digest of ``model``, or ``None``.

    Section 9 records this because a fixed seed does not make Ollama
    bit-reproducible; the digest is what lets a reader tell "same numbers, same
    weights" from "same numbers, different weights". ``ping()`` returns names
    only, so this reaches through to the raw listing -- inside a try, because
    a provenance probe must never be able to fail a run.
    """
    if client is None:
        return None
    try:
        listing = client._backend().list()  # noqa: SLF001 - no public accessor
        for entry in getattr(listing, "models", None) or listing.get("models", []):
            name = getattr(entry, "model", None) or getattr(entry, "name", None)
            if name is None and isinstance(entry, Mapping):
                name = entry.get("model") or entry.get("name")
            if name != model:
                continue
            digest = getattr(entry, "digest", None)
            if digest is None and isinstance(entry, Mapping):
                digest = entry.get("digest")
            return str(digest) if digest else None
    except Exception:  # noqa: BLE001
        return None
    return None


def cache_state(cfg: Config, *, forced_warm: bool = False) -> dict[str, Any]:
    """Whether the LLM response cache is cold, and the evidence for saying so.

    Latency in Chapter 4 is only meaningful from a cold-cache run, so this
    records the file, its size and the verdict rather than a bare boolean the
    reader would have to take on faith. ``forced_warm`` is the explicit
    ``--warm-cache`` assertion: warmth is declared, never inferred as an
    excuse.
    """
    enabled = bool(cfg.get("llm.cache.enabled", False))
    raw = cfg.get("llm.cache.path", "") or ""
    path = Path(raw)
    if raw and not path.is_absolute():
        path = PROJECT_ROOT / path
    exists = bool(raw) and path.exists()
    size = path.stat().st_size if exists else 0
    cold = not forced_warm and not (enabled and exists and size > 0)
    return {
        "cache_cold": cold,
        "cache_enabled": enabled,
        "cache_path": str(path) if raw else None,
        "cache_exists": exists,
        "cache_bytes": size,
        "cache_declared_warm": forced_warm,
    }


def prompt_digests() -> dict[str, str]:
    """SHA-1 per prompt template, so a stale-prompt run is detectable (section 9)."""
    from ..agents.base import PROMPT_DIR, prompt_sha1

    out: dict[str, str] = {}
    for path in sorted(PROMPT_DIR.glob("*.txt")):
        try:
            out[path.stem] = prompt_sha1(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
    return out


def git_commit() -> str | None:
    """The commit that produced a run, or ``None`` outside a checkout."""
    import subprocess  # noqa: PLC0415 - local: only this probe needs it

    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Run directory resolution
# ---------------------------------------------------------------------------

def runs_root(cfg: Config | None = None) -> Path:
    cfg = cfg or load_config()
    raw = Path(str(cfg.get("trace.dir", "results/runs")))
    return raw if raw.is_absolute() else PROJECT_ROOT / raw


def latest_run_dir(config_name: str, dataset: str, cfg: Config | None = None) -> Path | None:
    """Most recent ``{config}_{dataset}_*`` run directory, or ``None``.

    Run ids embed a UTC timestamp in ``%Y%m%dT%H%M%SZ``, which sorts
    lexicographically, so "latest" needs no filesystem timestamps -- and
    therefore survives a copied or restored directory tree.
    """
    root = runs_root(cfg)
    if not root.is_dir():
        return None
    matches = sorted(
        (p for p in root.iterdir() if p.is_dir() and p.name.startswith(f"{config_name}_{dataset}_")),
        key=lambda p: p.name,
    )
    return matches[-1] if matches else None


def _resolve_run_id(spec: RunSpec, cfg: Config) -> tuple[str, bool]:
    """``(run_id, resuming)``."""
    if spec.run_id:
        return spec.run_id, (runs_root(cfg) / spec.run_id).exists()
    if spec.resume:
        if spec.resume != "latest":
            return spec.resume, True
        found = latest_run_dir(spec.config_name, spec.dataset, cfg)
        if found is None:
            return make_run_id(spec.config_name, spec.dataset), False
        return found.name, True
    return make_run_id(spec.config_name, spec.dataset), False


def _completed_qids(path: Path, *, retry_failed: bool) -> set[str]:
    """qids that must be skipped on restart.

    With ``retry_failed`` a record that produced no answer, or that the harness
    itself failed to run, is not treated as done -- a dropped Ollama connection
    should not permanently poison one question's row in the results table.
    """
    done: set[str] = set()
    for record in iter_records(path):
        qid = record.get("qid")
        if not qid:
            continue
        if retry_failed:
            failed = record.get("terminated_by") == "harness_error" or not record.get("final_answer")
            if failed:
                continue
        done.add(str(qid))
    return done


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def run_eval(
    spec: RunSpec,
    *,
    cfg: Config | None = None,
    pipeline: Pipeline | None = None,
    system: Any = None,
    client: Any = None,
    progress: bool = True,
    on_question: Any = None,
) -> RunResult:
    """Evaluate one configuration on one dataset. Returns a :class:`RunResult`.

    ``system`` and ``pipeline`` are injectable so tests can drive the whole
    harness -- checkpointing, error handling, artefacts -- with a stub system
    and no indexes, no GPU and no Ollama.
    """
    base_cfg = cfg or load_config()
    run_cfg = config_for(spec.config_name, base_cfg)
    seeding = seed_everything(spec.seed)

    golds = tuple(load_eval_set(spec.dataset, split=spec.split, size=spec.size, cfg=run_cfg))
    if spec.limit is not None:
        golds = golds[: max(0, spec.limit)]

    run_id, resuming = _resolve_run_id(spec, run_cfg)
    root = Path(spec.root) if spec.root is not None else runs_root(run_cfg)
    model = str(run_cfg.get("llm.default_model", ""))
    writer = TraceWriter.create(
        config_name=spec.config_name,
        dataset=spec.dataset,
        cfg=run_cfg,
        run_id=run_id,
        model=model,
        root=root,
    )

    done = _completed_qids(writer.traces_path, retry_failed=spec.retry_failed)
    pending = [g for g in golds if g.qid not in done]

    if client is None and spec.config_name in AGENTIC_CONFIGS:
        from ..llm import get_client

        client = get_client()

    if system is None:
        if pipeline is None:
            pipeline = load_pipeline(
                spec.dataset,
                run_cfg,
                with_kg=bool(run_cfg.get("agents.kg.enabled", True)),
            )
        system = build_system(
            spec.config_name, spec.dataset, cfg=run_cfg, pipeline=pipeline, client=client
        )

    _write_meta(
        writer,
        spec=spec,
        cfg=run_cfg,
        client=client,
        seeding=seeding,
        pipeline=pipeline,
        n_pending=len(pending),
        n_total=len(golds),
        resuming=resuming,
    )

    result = RunResult(
        run_id=writer.run_id,
        run_dir=writer.run_dir,
        config_name=spec.config_name,
        dataset=spec.dataset,
        n_questions=len(golds),
        n_skipped=len(golds) - len(pending),
    )

    started = time.perf_counter()
    iterator: Iterable[GoldAnswer] = pending
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(pending, desc=f"{spec.config_name}/{spec.dataset}", unit="q")
        except Exception:  # noqa: BLE001
            iterator = pending

    for gold in iterator:
        record = _run_one(system, writer, gold, spec=spec, config_name=spec.config_name)
        if record.get("terminated_by") == "harness_error":
            result.n_failed += 1
        result.n_completed += 1
        if on_question is not None:
            on_question(record)
    result.elapsed_s = time.perf_counter() - started

    records = writer.read_records()
    scores = score_records(records, {g.qid: g for g in golds}, cfg=run_cfg)
    write_scores_csv(writer.run_dir / "scores.csv", scores)
    result.scores = _mean_scores(scores)
    return result


def _run_one(
    system: Any,
    writer: TraceWriter,
    gold: GoldAnswer,
    *,
    spec: RunSpec,
    config_name: str,
) -> dict[str, Any]:
    """One question, from state to appended trace record. Never raises.

    The state is built here, not inside the system, so that a system which
    fails catastrophically still has a state to serialise -- which is what
    keeps the failed question's row in ``metrics.csv`` column-aligned with
    every other row instead of truncating the header for the whole run.
    """
    state = QuestionState(
        qid=gold.qid,
        question=gold.question,
        dataset=spec.dataset,
        config_name=config_name,
        gold=gold,
    )
    transitions: Sequence[str] = ()
    try:
        returned = system.run(gold.qid, gold.question, gold=gold, state=state)
        if isinstance(returned, QuestionState):
            state = returned
        transitions = tuple(getattr(system, "transitions", ()) or ())
    except TypeError:
        # A baseline may not accept `state=`; the orchestrator's own contract
        # does, so this is the narrow compatibility path, not the normal one.
        try:
            returned = system.run(gold.qid, gold.question, gold=gold)
            if isinstance(returned, QuestionState):
                state = returned
            transitions = tuple(getattr(system, "transitions", ()) or ())
        except Exception as exc:  # noqa: BLE001
            _mark_harness_error(state, exc)
    except Exception as exc:  # noqa: BLE001 - axiom 2, at the harness layer
        _mark_harness_error(state, exc)

    try:
        return writer.write_question(state, transitions=transitions)
    except Exception as exc:  # noqa: BLE001 - a torn write must not kill the run
        fallback = QuestionState(
            qid=gold.qid, question=gold.question, dataset=spec.dataset,
            config_name=config_name, gold=gold,
        )
        _mark_harness_error(fallback, exc, stage="trace_write")
        return writer.write_question(fallback, transitions=())


def _mark_harness_error(state: QuestionState, exc: BaseException, *, stage: str = "run") -> None:
    state.errors.append(f"harness[{stage}]: {type(exc).__name__}: {exc}")
    state.terminated_by = "harness_error"
    state.state = "DONE"


def _write_meta(
    writer: TraceWriter,
    *,
    spec: RunSpec,
    cfg: Config,
    client: Any,
    seeding: Mapping[str, Any],
    pipeline: Pipeline | None,
    n_pending: int,
    n_total: int,
    resuming: bool,
) -> None:
    """Write (or extend) ``meta.json``.

    On resume the original ``created_utc`` and the first leg's cache verdict
    are preserved and a new leg is appended. A resumed run's latency figures
    span two process lifetimes and possibly two cache states, and ``legs`` is
    what makes that visible instead of averaging it away.
    """
    import orjson

    previous: dict[str, Any] = {}
    if writer.meta_path.exists():
        try:
            previous = orjson.loads(writer.meta_path.read_bytes())
        except Exception:  # noqa: BLE001
            previous = {}

    cache = cache_state(cfg, forced_warm=spec.warm_cache)
    eval_path = eval_set_path(spec.dataset, split=spec.split, size=spec.size, cfg=cfg)
    now = datetime.now(UTC).isoformat(timespec="seconds")

    legs: list[dict[str, Any]] = list(previous.get("legs") or [])
    legs.append(
        {
            "started_utc": now,
            "resumed": bool(resuming and legs),
            "n_already_done": n_total - n_pending,
            "n_pending": n_pending,
            **cache,
        }
    )

    extra: dict[str, Any] = {
        "config_name": spec.config_name,
        "dataset": spec.dataset,
        "split": spec.split,
        "limit": spec.limit,
        "n_questions": n_total,
        "overrides": spec.overrides(),
        "eval_set": {
            "path": str(eval_path),
            "sha256": sha256_file(eval_path),
            "n_questions": n_total,
        },
        "seeding": dict(seeding),
        "prompt_sha1": prompt_digests(),
        "git_commit": git_commit(),
        "pipeline_notes": list(pipeline.notes) if pipeline is not None else [],
        "legs": legs,
        # The whole-run verdict: cold only if EVERY leg was cold.
        "cache_cold": all(bool(leg.get("cache_cold")) for leg in legs),
        "cache": cache,
    }
    if previous.get("created_utc"):
        extra["created_utc"] = previous["created_utc"]

    writer.write_meta(
        cfg,
        cache_cold=extra["cache_cold"],
        model_digest=model_digest(client, str(cfg.get("llm.default_model", ""))),
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Scoring (derived; recomputable from traces.jsonl at any time)
# ---------------------------------------------------------------------------

def _numeric_id(sq_id: str) -> tuple[int, str]:
    digits = sq_id[1:]
    return (int(digits), sq_id) if digits.isdigit() else (10**6, sq_id)


def question_ranking(record: Mapping[str, Any], *, rrf_k: int = 60) -> list[str]:
    """One document ranking per question, fused across the plan's sub-queries.

    A multi-hop system issues several queries, so "the ranking" is not directly
    observed. Reciprocal rank fusion over the per-sub-query lists is the
    honest aggregate: it is exactly how the orchestrator pools evidence, it
    needs no score calibration between BM25 and cosine, and the ``-score,
    doc_id`` tiebreak makes it deterministic.
    """
    scores: dict[str, float] = {}
    retrieved = record.get("retrieved") or {}
    for sq_id in sorted(retrieved, key=_numeric_id):
        block = retrieved.get(sq_id) or {}
        for position, passage in enumerate(block.get("passages") or []):
            doc_id = passage.get("doc_id")
            if not doc_id:
                continue
            rank = int(passage.get("rank", position))
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    return [doc for doc, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def gold_doc_ids(gold: GoldAnswer | Mapping[str, Any], dataset: str) -> list[str]:
    """The gold supporting facts' document ids, deduplicated, in first-seen order."""
    facts = (
        gold.supporting_facts
        if isinstance(gold, GoldAnswer)
        else (gold.get("supporting_facts") or ())
    )
    out: list[str] = []
    for title, _sent_id in facts:
        doc_id = make_doc_id(dataset, str(title))
        if doc_id not in out:
            out.append(doc_id)
    return out


def predicted_supporting_facts(record: Mapping[str, Any]) -> list[tuple[str, int]]:
    """The ``(title, sent_id)`` pairs the system stands behind.

    Cited evidence when there are citations, the whole evidence pool otherwise.
    The fallback is what keeps the comparison against a retrieval-only baseline
    meaningful -- a baseline that never emits citations would otherwise score a
    structural zero on sp_em and sp_f1 -- and reporting sp_precision beside
    sp_f1 is what keeps the fallback honest, since an uncited pool of twenty
    sentences pays for its recall in precision.
    """
    evidence = record.get("evidence") or []
    cited = set(record.get("citations") or ())
    pool = [e for e in evidence if e.get("evidence_id") in cited] if cited else list(evidence)
    facts: list[tuple[str, int]] = []
    for item in pool:
        title, sent_id = item.get("title"), item.get("sent_id")
        if title is None or sent_id is None:
            continue
        pair = (str(title), int(sent_id))
        if pair not in facts:
            facts.append(pair)
    return facts


def score_records(
    records: Sequence[Mapping[str, Any]],
    golds: Mapping[str, GoldAnswer],
    *,
    cfg: Config | None = None,
    k_values: Sequence[int] = K_VALUES,
) -> dict[str, dict[str, float]]:
    """Per-question answer, supporting-fact and retrieval scores, keyed by qid.

    Derived purely from the trace and the qrels, so a metric fix is a rerun of
    this function rather than a rerun of the model.
    """
    cfg = cfg or load_config()
    rrf_k = int(cfg.get("retrieval.hybrid.rrf_k", 60))

    out: dict[str, dict[str, float]] = {}
    rankings: dict[str, list[str]] = {}
    qrels: dict[str, list[str]] = {}

    for record in records:
        qid = str(record.get("qid") or "")
        gold = golds.get(qid)
        if gold is None:
            continue
        dataset = str(record.get("dataset") or gold.dataset)
        scores = score_question(
            str(record.get("final_answer") or ""),
            gold.answer,
            predicted_supporting_facts(record),
            gold.supporting_facts,
        )
        row = {
            "em": scores.em, "f1": scores.f1,
            "precision": scores.precision, "recall": scores.recall,
            "sp_em": scores.sp_em, "sp_f1": scores.sp_f1,
            "sp_precision": scores.sp_precision, "sp_recall": scores.sp_recall,
            "joint_em": scores.joint_em, "joint_f1": scores.joint_f1,
        }
        out[qid] = row
        ranking = question_ranking(record, rrf_k=rrf_k)
        gold_docs = gold_doc_ids(gold, dataset)
        if ranking and gold_docs:
            rankings[qid] = ranking
            qrels[qid] = gold_docs

    if rankings:
        per_query = retrieval_metrics_per_query(rankings, qrels, k_values=k_values)
        for qid, row in per_query.items():
            out.setdefault(qid, {}).update(row)
    # A question whose system retrieved nothing still scores zero, not blank:
    # dropping it would silently evaluate the survivors and inflate recall.
    metric_names = [f"recall@{k}" for k in k_values] + ["ndcg@10", "mrr"]
    for row in out.values():
        for name in metric_names:
            row.setdefault(name, 0.0)
    return out


def _mean_scores(scores: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    if not scores:
        return {}
    keys: list[str] = []
    for row in scores.values():
        for key in row:
            if key not in keys:
                keys.append(key)
    n = len(scores)
    return {k: sum(float(r.get(k, 0.0)) for r in scores.values()) / n for k in keys}


def write_scores_csv(path: Path, scores: Mapping[str, Mapping[str, float]]) -> Path:
    """Per-question scores as CSV. Derived, and safe to delete and regenerate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = ["qid"]
    for row in scores.values():
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for qid in sorted(scores):
            writer.writerow({"qid": qid, **{k: scores[qid].get(k, "") for k in fields[1:]}})
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agentic_ir.eval.run_eval",
        description="Run one configuration on one dataset, with checkpointing.",
    )
    parser.add_argument("--dataset", default="hotpotqa", choices=("hotpotqa", "twowiki"))
    parser.add_argument("--config", dest="config_name", default="agentic_full",
                        choices=CONFIGURATIONS, help="one of evaluation.configurations")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N questions of the slice")
    parser.add_argument("--split", default="eval", choices=("eval", "calib"))
    parser.add_argument("--size", type=int, default=None,
                        help="slice size, e.g. 250; defaults to the configured value")
    parser.add_argument("--resume", nargs="?", const="latest", default=None, metavar="RUN_ID",
                        help="resume a run: the latest matching one, or a named run_id")
    parser.add_argument("--run-id", default=None,
                        help="write into this exact run id (creates or extends it)")
    parser.add_argument("--retry-failed", action="store_true",
                        help="on resume, re-run questions that failed or produced no answer")
    parser.add_argument("--warm-cache", action="store_true",
                        help="declare the LLM cache warm; latency tables must not use this run")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config()
    spec = RunSpec(
        config_name=args.config_name,
        dataset=args.dataset,
        limit=args.limit,
        split=args.split,
        size=args.size,
        resume=args.resume,
        retry_failed=args.retry_failed,
        seed=args.seed if args.seed is not None else int(cfg.get("project.seed", 42)),
        warm_cache=args.warm_cache,
        run_id=args.run_id,
    )
    result = run_eval(spec, cfg=cfg, progress=not args.no_progress)
    print(result.summary())
    print(f"artefacts: {result.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
