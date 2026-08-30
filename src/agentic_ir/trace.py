"""Run artefacts: ``traces.jsonl``, ``meta.json``, ``metrics.csv``.

Specification: ``docs/architecture.md`` section 6. One JSONL record per
question, appended as the question finishes rather than buffered, because the
eval harness checkpoints on this file: if question 194 dies, the run resumes at
194 rather than at 1.

Everything is normalised to plain JSON types before it is written. Frozen
dataclasses carry tuples, and a tuple is not a JSON value -- normalising here
rather than at every call site is what keeps the trace schema stable while the
dataclasses evolve.
"""

from __future__ import annotations

import csv
import platform
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

import orjson

from .config import PROJECT_ROOT, Config, load_config
from .state import QuestionState

__all__ = [
    "SCHEMA_VERSION",
    "TraceWriter",
    "build_trace_record",
    "iter_records",
    "make_run_id",
    "metrics_row",
]

SCHEMA_VERSION = "1.0"

#: Libraries whose versions decide whether two runs are comparable.
_VERSIONED = (
    "ollama", "bm25s", "faiss-cpu", "faiss", "sentence-transformers",
    "torch", "transformers", "numpy", "networkx", "orjson", "PyYAML", "pandas",
)


# ---------------------------------------------------------------------------
# JSON normalisation
# ---------------------------------------------------------------------------

def _jsonable(value: Any) -> Any:
    """Recursively convert to types ``orjson`` accepts.

    Tuples and MappingProxy views come from the frozen dataclasses and from the
    deep-frozen config; neither is a JSON value, and both appear all over the
    objects this module is handed.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, (Mapping, MappingProxyType)):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return [_jsonable(v) for v in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _truncate(text: str | None, limit: int) -> str | None:
    if text is None:
        return None
    return text if len(text) <= limit else text[:limit] + "..."


def _numeric_id(sq_id: str) -> tuple[int, str]:
    """Sort ``q10`` after ``q9``; fall back to lexical for anything odd."""
    digits = sq_id[1:]
    return (int(digits), sq_id) if digits.isdigit() else (10**6, sq_id)


# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------

def make_run_id(config_name: str, dataset: str, *, now: datetime | None = None) -> str:
    """``{config}_{dataset}_{utc}`` -- sortable, and unique per second."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"{config_name}_{dataset}_{stamp}"


def _plan_summary(state: QuestionState) -> dict[str, Any]:
    plan = state.plan
    if plan is None:
        return {"plan_depth": 0, "n_subqueries": 0}
    return {"plan_depth": plan.depth, "n_subqueries": len(plan.subqueries)}


def _retrieved_block(state: QuestionState, *, top_k: int) -> dict[str, Any]:
    """Compact per-sub-query retrieval view.

    Passage *text* is deliberately omitted: error analysis needs doc ids,
    titles and scores, and keeping full paragraphs would multiply the trace
    size by roughly forty for no analytical gain.
    """
    block: dict[str, Any] = {}
    for sq_id in sorted(state.results, key=_numeric_id):
        result = state.results[sq_id]
        sel = result.selection
        block[sq_id] = {
            "query_text": result.query_text,
            "queries_issued": list(result.queries_issued),
            "tool": sel.tool,
            "selector": sel.selector,
            "rule_id": sel.rule_id,
            "rerank_applied": sel.rerank_applied,
            "reason": sel.reason,
            "features": {k: _jsonable(v) for k, v in sel.features.items()},
            "n_candidates": result.n_candidates,
            "latency_s": round(result.latency_s, 4),
            "degraded": result.degraded,
            "error": result.error,
            "passages": [
                {
                    "doc_id": sp.passage.doc_id,
                    "title": sp.passage.title,
                    "rank": sp.rank,
                    "score": round(float(sp.score), 6),
                    "provenance": sp.provenance,
                    "component_scores": {
                        k: round(float(v), 6) for k, v in sp.component_scores.items()
                    },
                }
                for sp in result.passages[:top_k]
            ],
        }
    return block


def _kg_block(state: QuestionState) -> dict[str, Any]:
    block: dict[str, Any] = {}
    for sq_id in sorted(state.kg_results, key=_numeric_id):
        kg = state.kg_results[sq_id]
        block[sq_id] = {
            "seeds": [e.entity_id for e in kg.seeds],
            "linked_by": kg.linked_by,
            "bridge_entity": kg.bridge_entity,
            "n_paths": len(kg.paths),
            "n_neighbors": len(kg.neighbors),
            "n_evidence": len(kg.evidence),
            "latency_s": round(kg.latency_s, 4),
            "degraded": kg.degraded,
            "error": kg.error,
        }
    return block


def _extra_metrics(state: QuestionState) -> dict[str, Any]:
    """The section-6 metrics ``QuestionState.metrics()`` cannot know about.

    ``nli_latency_s`` is an approximation: the Verifier's step latency minus the
    adjudication call it may have made. There is no finer signal without the
    Verifier co-operating, and inventing one would flatter the "NLI is cheap"
    claim rather than test it.
    """
    retrieval_latency = sum(r.latency_s for r in state.results.values())
    kg_latency = sum(k.latency_s for k in state.kg_results.values())
    nli_latency = 0.0
    for trace in state.traces:
        if trace.agent == "verifier":
            nli_latency += max(0.0, trace.latency_s - sum(c.latency_s for c in trace.llm_calls))
    rerank_skipped = sum(
        1
        for r in state.results.values()
        if float(r.selection.features.get("rerank_skipped", 0.0)) > 0.0
    )
    return {
        "retrieval_latency_s": round(retrieval_latency, 4),
        "kg_latency_s": round(kg_latency, 4),
        "nli_latency_s": round(nli_latency, 4),
        "rerank_skipped": rerank_skipped,
        "n_evidence": len(state.evidence),
        "n_results": len(state.results),
    }


def build_trace_record(
    state: QuestionState,
    *,
    run_id: str,
    seed: int = 42,
    model: str = "",
    raw_output_chars: int = 2000,
    top_k: int = 10,
    transitions: Sequence[str] = (),
) -> dict[str, Any]:
    """Assemble one question's JSONL record (architecture section 6)."""
    best = state.best_candidate()
    ver = state.verification
    plans: list[dict[str, Any]] = []
    for plan in state.plans:
        payload = _jsonable(plan)
        payload["raw_llm_output"] = _truncate(plan.raw_llm_output, raw_output_chars)
        plans.append(payload)

    steps: list[dict[str, Any]] = []
    for trace in state.traces:
        payload = _jsonable(trace)
        for call in payload.get("llm_calls", []):
            call["raw_output"] = _truncate(call.get("raw_output"), raw_output_chars)
        steps.append(payload)

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "config_name": state.config_name,
        "dataset": state.dataset,
        "qid": state.qid,
        "seed": seed,
        "model": model,
        "question": state.question,
        "gold": _jsonable(state.gold),
        "final_answer": best.answer if best else "",
        "answer_sentence": best.answer_sentence if best else "",
        "citations": list(best.citations) if best else [],
        "confidence": round(float(best.confidence), 6) if best else 0.0,
        "verdict": ver.verdict if ver else None,
        "terminated_by": state.terminated_by,
        "best_cycle": best.cycle if best else None,
        "metrics": {**state.metrics(), **_plan_summary(state), **_extra_metrics(state)},
        "transitions": list(transitions),
        "plans": plans,
        "directives": [_jsonable(d) for d in state.directives],
        "steps": steps,
        "retrieved": _retrieved_block(state, top_k=top_k),
        "kg": _kg_block(state),
        "evidence": [
            _jsonable(state.evidence[eid])
            for eid in sorted(state.evidence, key=_numeric_id)
        ],
        "candidates": [_jsonable(c) for c in state.candidates],
        "verifications": [_jsonable(v) for v in state.verifications],
        "errors": list(state.errors),
    }


def metrics_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one trace record into a ``metrics.csv`` row for pandas."""
    row: dict[str, Any] = {
        "qid": record.get("qid"),
        "run_id": record.get("run_id"),
        "config_name": record.get("config_name"),
        "dataset": record.get("dataset"),
        "final_answer": record.get("final_answer"),
        "confidence": record.get("confidence"),
        "verdict": record.get("verdict"),
        "terminated_by": record.get("terminated_by"),
        "best_cycle": record.get("best_cycle"),
        "n_citations": len(record.get("citations") or ()),
        "n_errors": len(record.get("errors") or ()),
    }
    for key, value in (record.get("metrics") or {}).items():
        row[key] = len(value) if isinstance(value, (list, tuple, dict)) else value
    return row


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class TraceWriter:
    """Owns one run directory: ``traces.jsonl``, ``meta.json``, ``metrics.csv``.

    Append-only and flushed per question. Buffering would be faster and would
    also lose a whole run to one crash, which is the failure mode section 9
    exists to prevent.
    """

    def __init__(
        self,
        run_dir: str | Path,
        *,
        run_id: str | None = None,
        raw_output_chars: int = 2000,
        top_k: int = 10,
        seed: int = 42,
        model: str = "",
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or self.run_dir.name
        self.raw_output_chars = int(raw_output_chars)
        self.top_k = int(top_k)
        self.seed = int(seed)
        self.model = model
        self._csv_fields: list[str] | None = None

    # -- construction ------------------------------------------------------
    @classmethod
    def create(
        cls,
        *,
        config_name: str,
        dataset: str,
        cfg: Config | None = None,
        run_id: str | None = None,
        model: str = "",
        root: str | Path | None = None,
    ) -> TraceWriter:
        """Open (or re-open) ``{trace.dir}/{run_id}``.

        Re-opening an existing ``run_id`` is the resume path: nothing is
        truncated, and :meth:`existing_qids` tells the harness what to skip.
        """
        cfg = cfg or load_config()
        base = Path(root) if root is not None else _resolve(cfg, "trace.dir", "results/runs")
        rid = run_id or make_run_id(config_name, dataset)
        return cls(
            base / rid,
            run_id=rid,
            raw_output_chars=int(cfg.get("trace.raw_output_chars", 2000)),
            top_k=int(cfg.get("retrieval.top_k", 10)),
            seed=int(cfg.get("project.seed", 42)),
            model=model or str(cfg.get("llm.default_model", "")),
        )

    # -- paths -------------------------------------------------------------
    @property
    def traces_path(self) -> Path:
        return self.run_dir / "traces.jsonl"

    @property
    def meta_path(self) -> Path:
        return self.run_dir / "meta.json"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.csv"

    # -- writing -----------------------------------------------------------
    def write_meta(
        self,
        cfg: Config | None = None,
        *,
        cache_cold: bool = True,
        model_digest: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Path:
        """Snapshot everything needed to say what produced this run."""
        cfg = cfg or load_config()
        meta: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "seed": self.seed,
            "model": self.model,
            "model_digest": model_digest,
            "cache_cold": bool(cache_cold),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "gpu": _gpu_name(),
            "versions": _library_versions(),
            "config_path": str(cfg.path),
            "config": _jsonable(cfg.raw),
        }
        if extra:
            meta.update(_jsonable(dict(extra)))
        self.meta_path.write_bytes(orjson.dumps(meta, option=orjson.OPT_INDENT_2))
        return self.meta_path

    def append(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Append one record to ``traces.jsonl`` and one row to ``metrics.csv``."""
        payload = _jsonable(dict(record))
        with self.traces_path.open("ab") as fh:
            fh.write(orjson.dumps(payload))
            fh.write(b"\n")
        self._append_metrics(metrics_row(payload))
        return payload

    def write_question(
        self,
        state: QuestionState,
        *,
        transitions: Sequence[str] = (),
        model: str | None = None,
    ) -> dict[str, Any]:
        """Build and append the record for one finished question."""
        record = build_trace_record(
            state,
            run_id=self.run_id,
            seed=self.seed,
            model=self.model if model is None else model,
            raw_output_chars=self.raw_output_chars,
            top_k=self.top_k,
            transitions=transitions,
        )
        return self.append(record)

    def _append_metrics(self, row: Mapping[str, Any]) -> None:
        """Append to ``metrics.csv``, writing the header on first use.

        The field list is frozen by the first row: a later question carrying an
        extra key is dropped from the CSV rather than corrupting the column
        alignment of everything already written. The JSONL keeps it either way.
        """
        if self._csv_fields is None:
            self._csv_fields = _existing_header(self.metrics_path) or list(row)
            if not self.metrics_path.exists():
                with self.metrics_path.open("w", encoding="utf-8", newline="") as fh:
                    csv.DictWriter(fh, fieldnames=self._csv_fields).writeheader()
        with self.metrics_path.open("a", encoding="utf-8", newline="") as fh:
            csv.DictWriter(fh, fieldnames=self._csv_fields).writerow(
                {k: row.get(k, "") for k in self._csv_fields}
            )

    # -- resume ------------------------------------------------------------
    def existing_qids(self) -> set[str]:
        """qids already in ``traces.jsonl``; the harness skips these on restart."""
        return {str(r["qid"]) for r in iter_records(self.traces_path) if "qid" in r}

    def read_records(self) -> list[dict[str, Any]]:
        """Every record written so far. For the tables in Chapter 4."""
        return list(iter_records(self.traces_path))

    def __repr__(self) -> str:
        return f"TraceWriter(run_id={self.run_id!r}, dir={self.run_dir!s})"


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream a ``traces.jsonl`` file.

    A torn final line -- the signature of a run killed mid-write -- is skipped
    rather than raised: the point of checkpointing is that a resumed run reads
    what survived.
    """
    target = Path(path)
    if not target.exists():
        return
    with target.open("rb") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = orjson.loads(stripped)
            except orjson.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


# ---------------------------------------------------------------------------
# Environment probes
# ---------------------------------------------------------------------------

def _resolve(cfg: Config, dotted: str, default: str) -> Path:
    raw = Path(str(cfg.get(dotted, default)))
    return raw if raw.is_absolute() else PROJECT_ROOT / raw


def _existing_header(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as fh:
        header = next(csv.reader(fh), None)
    return list(header) if header else None


def _library_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for name in _VERSIONED:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
    return out


def _gpu_name() -> str | None:
    """GPU model, or None. A probe must never break a run."""
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:  # noqa: BLE001
        return None
    return None
