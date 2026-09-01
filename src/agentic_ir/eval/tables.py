r"""LaTeX table generation: the single source of truth for every number in the report.

Specification: ``docs/architecture.md`` section 6 (metric definitions) and the
rule ``report/README.md`` states plainly -- *numbers must never be typed by hand
into the report*. The chapters ``\input{}`` the fragments this module writes into
``results/tables/``, so re-running the evaluation and re-running this module is
the whole update path for the PDF. There is no manual editing step in which the
report and the code can drift apart.

Four properties are what make that claim safe rather than merely stated.

**Scores are recomputed, never read back.** ``scores.csv`` is a derived artefact
of a run; this module ignores it and re-derives every score from
``traces.jsonl`` and the frozen eval slice via :func:`~.run_eval.score_records`.
Fixing a metric is then a re-run of *this* module, not of the model.

**A missing configuration renders as** ``--`` **, never as** ``0.0``. Most rows
have no data until late in the schedule, and a zero would read as a measured
failure rather than an unrun experiment -- the most expensive kind of typo a
results table can contain.

**Significance is marked, not implied.** A cell is bolded only when a paired
bootstrap against ``hybrid_rerank`` -- the strongest non-agentic baseline, the
honest one to beat -- puts the whole 95% interval of the difference on one side
of zero. A two-point gap on 250 questions is usually noise, and a table that
bolds it overstates its evidence.

**Output is deterministic.** No wall-clock timestamp is written into the
fragments, only the ``run_id``s they came from, so regenerating from the same
runs produces byte-identical files and a diff always means a number moved.

Emitted fragments, all carrying ``\caption`` and ``\label`` so the chapters can
``\ref`` them:

* ``main_results.tex``   -- ``tab:main-answer-{dataset}``, ``tab:main-retrieval-{dataset}``
* ``agent_metrics.tex``  -- ``tab:agent-metrics-{dataset}``
* ``ablations.tex``      -- ``tab:ablations-{dataset}``
* ``dataset_stats.tex``  -- ``tab:dataset-corpus``, ``tab:dataset-sample``
* ``error_analysis.tex`` -- ``tab:error-analysis-{dataset}``
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT, Config, load_config
from ..indexing.corpus import DATASETS, load_eval_set
from ..trace import iter_records
from ..types import GoldAnswer
from .bootstrap import BootstrapResult, ComparisonResult, bootstrap_mean, paired_bootstrap
from .error_analysis import ERROR_LABELS, ErrorSummary, analyse, summarise
from .metrics import AgentMetrics, summarise_agent_metrics
from .run_eval import AGENTIC_CONFIGS, configurations, runs_root, score_records

__all__ = [
    "ABLATION_REFERENCE",
    "MISSING",
    "REFERENCE_CONFIG",
    "TABLE_FILES",
    "RunData",
    "ablations_table",
    "agent_metrics_table",
    "dataset_stats_tables",
    "discover_runs",
    "error_analysis_table",
    "fmt",
    "fmt_ci",
    "fmt_count",
    "fmt_delta",
    "generate_all",
    "latex_escape",
    "load_golds",
    "load_run",
    "main",
    "main_results_tables",
    "mark",
    "table",
]

#: What an unrun configuration renders as. Never a zero: a zero is a measured
#: failure and this is the absence of a measurement.
MISSING = "--"

#: The system every headline comparison is made against. It is the strongest
#: non-agentic baseline, so beating it is the claim Chapter 4 has to earn;
#: comparing against a weaker one would flatter the agentic system for free.
REFERENCE_CONFIG = "hybrid_rerank"

#: The ablation reference: an ablation's delta is only meaningful against the
#: full system it removes a component from.
ABLATION_REFERENCE = "agentic_full"

#: The fragments this module owns, in the order the chapters use them.
TABLE_FILES: tuple[str, ...] = (
    "main_results.tex",
    "agent_metrics.tex",
    "ablations.tex",
    "dataset_stats.tex",
    "error_analysis.tex",
)

#: The metric that carries a bootstrap interval in each of the two main tables.
_CI_METRIC = "f1"
_RETRIEVAL_CI_METRIC = "ndcg@10"


# ---------------------------------------------------------------------------
# LaTeX primitives
# ---------------------------------------------------------------------------

#: Every character TeX would either eat or choke on. ``bm25_only`` and
#: ``agentic_no_kg`` are the ones that actually occur, and an unescaped
#: underscore is a hard compile error rather than a cosmetic defect: the report
#: would simply not build.
_LATEX_SPECIALS: dict[str, str] = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(value: object) -> str:
    """Escape every TeX special in ``value``.

    One pass over the characters rather than chained ``str.replace`` calls,
    because a chain re-escapes the backslashes it just introduced and turns
    ``bm25_only`` into ``bm25\\textbackslash{}_only``.
    """
    return "".join(_LATEX_SPECIALS.get(ch, ch) for ch in str(value))


def _tt(name: object) -> str:
    """A configuration or dataset name, escaped, in monospace."""
    return r"\texttt{" + latex_escape(name) + "}"


def fmt(value: float | int | None, digits: int = 3) -> str:
    """A number, or :data:`MISSING` when there is nothing to report."""
    if value is None:
        return MISSING
    try:
        number = float(value)
    except (TypeError, ValueError):
        return MISSING
    if number != number:  # NaN: an undefined mean is not a measurement either
        return MISSING
    return f"{number:.{digits}f}"


def fmt_int(value: int | float | None) -> str:
    """An integer count, or :data:`MISSING`."""
    if value is None:
        return MISSING
    try:
        return f"{int(round(float(value))):d}"
    except (TypeError, ValueError):
        return MISSING


def fmt_count(value: int | float | None) -> str:
    r"""A large count with TeX thin-space thousands separators (``66\,581``).

    TeX's own spacing rather than a comma, so the column stays aligned under any
    font and no separator can be misread as a decimal point.
    """
    if value is None:
        return MISSING
    try:
        return f"{int(round(float(value))):,}".replace(",", r"\,")
    except (TypeError, ValueError):
        return MISSING


def fmt_ci(result: BootstrapResult | None, digits: int = 3) -> str:
    """``[low, high]`` for a bootstrap interval, or :data:`MISSING`."""
    if result is None or result.n == 0:
        return MISSING
    return f"[{result.ci_low:.{digits}f}, {result.ci_high:.{digits}f}]"


def fmt_delta(comparison: ComparisonResult | None, digits: int = 3) -> str:
    """``+0.061 [+0.010, +0.112]`` for a paired difference, or :data:`MISSING`.

    The interval is always shown beside the point estimate. A bare delta invites
    the reader to treat it as established; the interval is what says how much of
    it survives resampling.
    """
    if comparison is None or comparison.n == 0:
        return MISSING
    return (
        f"{comparison.delta:+.{digits}f} "
        f"[{comparison.ci_low:+.{digits}f}, {comparison.ci_high:+.{digits}f}]"
    )


def fmt_p(comparison: ComparisonResult | None) -> str:
    """A two-sided bootstrap p-value, floored at the resolution the resamples give."""
    if comparison is None or comparison.n == 0:
        return MISSING
    return f"{comparison.p_value:.3f}" if comparison.p_value >= 0.001 else r"$<$0.001"


def mark(
    value: float | None,
    comparison: ComparisonResult | None,
    *,
    reference: bool = False,
    digits: int = 3,
) -> str:
    r"""Format a cell and mark it against the reference system.

    Bold means *significantly better than the reference*; ``$^{\downarrow}$``
    means significantly worse; the reference itself carries a dagger. Everything
    else is unmarked, which is the honest rendering of a difference the
    bootstrap cannot separate from zero. Bolding "the best number in the column"
    instead would make noise look like a finding.
    """
    text = fmt(value, digits)
    if text == MISSING:
        return text
    if reference:
        return text + r"$^{\dagger}$"
    if comparison is None or comparison.n == 0 or not comparison.significant:
        return text
    if comparison.delta > 0:
        return r"\textbf{" + text + "}"
    return text + r"$^{\downarrow}$"


def table(
    *,
    colspec: str,
    header: Sequence[str],
    groups: Sequence[tuple[str | None, Sequence[Sequence[str]]]],
    caption: str,
    label: str,
    size: str = r"\small",
) -> str:
    r"""One ``booktabs`` table environment, as a string.

    ``groups`` is ``[(title | None, rows)]``; a titled group gets an italic
    spanning row, which is how the baseline ladder and the agentic systems stay
    visually separate inside a nine-row table without a second table.

    The tabular is wrapped in the shrink-only ``\resizebox`` idiom
    (``\ifdim\width>\textwidth``). A generated table's width depends on data
    nobody has seen yet -- a confidence interval is three characters wider the
    day a bound goes negative -- so a fixed font size either overflows the text
    block silently or is set small enough to be unreadable forever. This shrinks
    only the tables that would overflow, and leaves the rest alone. It needs
    ``graphicx``, which ``report/main.tex`` already loads.
    """
    n_cols = len(header)
    lines: list[str] = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  {size}",
        r"  \setlength{\tabcolsep}{4pt}",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%",
        f"  \\begin{{tabular}}{{{colspec}}}",
        r"    \toprule",
        "    " + " & ".join(header) + r" \\",
        r"    \midrule",
    ]
    for index, (title, rows) in enumerate(groups):
        if title is not None:
            if index:
                lines.append(r"    \addlinespace")
            lines.append(
                f"    \\multicolumn{{{n_cols}}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\"
            )
        for row in rows:
            lines.append("    " + " & ".join(row) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def _header(*names: str) -> list[str]:
    """Escape a header row, leaving cells that are already TeX or math alone."""
    return [n if ("\\" in n or "$" in n) else latex_escape(n) for n in names]


# ---------------------------------------------------------------------------
# Run discovery and scoring
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RunData:
    """One configuration's run on one dataset, scored.

    ``scores`` is recomputed from the trace rather than read from
    ``scores.csv``: the CSV is a derived artefact of the run that produced it,
    and a metric fixed after that run would leave the CSV describing the old
    definition while looking perfectly current.
    """

    config_name: str
    dataset: str
    run_id: str
    run_dir: Path
    records: list[dict[str, Any]] = field(default_factory=list)
    scores: dict[str, dict[str, float]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_questions(self) -> int:
        return len(self.records)

    def metric(self, name: str) -> dict[str, float]:
        """``{qid: value}`` for one metric -- the shape both bootstraps want."""
        return {
            qid: float(row[name])
            for qid, row in self.scores.items()
            if row.get(name) is not None
        }

    def mean(self, name: str) -> float | None:
        """Corpus-level mean, or ``None`` when the metric was never computed."""
        values = list(self.metric(name).values())
        return sum(values) / len(values) if values else None

    def agent_metrics(self) -> AgentMetrics:
        """Aggregate of the per-question ``metrics`` blocks in the trace."""
        return summarise_agent_metrics([r.get("metrics") or {} for r in self.records])


def load_run(run_dir: Path, *, golds: Mapping[str, GoldAnswer], cfg: Config) -> RunData:
    """Read one run directory and score it. Never raises on a partial run."""
    import orjson

    records = list(iter_records(run_dir / "traces.jsonl"))
    meta: dict[str, Any] = {}
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = orjson.loads(meta_path.read_bytes())
        except Exception:  # noqa: BLE001 - a broken meta.json must not lose the scores
            meta = {}

    config_name = str(meta.get("config_name") or "")
    dataset = str(meta.get("dataset") or "")
    if records:
        config_name = config_name or str(records[0].get("config_name") or "")
        dataset = dataset or str(records[0].get("dataset") or "")

    scores = score_records(records, golds, cfg=cfg) if (records and golds) else {}
    return RunData(
        config_name=config_name,
        dataset=dataset,
        run_id=str(meta.get("run_id") or run_dir.name),
        run_dir=run_dir,
        records=records,
        scores=scores,
        meta=meta,
    )


def load_golds(dataset: str, *, cfg: Config) -> dict[str, GoldAnswer]:
    """The frozen eval slice, keyed by qid; empty if it has not been sampled yet.

    The slice is preferred over the trace's own copy of the gold block for the
    reason ``error_analysis.analyse`` gives: the trace's copy was written by the
    run under analysis, and the slice was not.
    """
    try:
        return {g.qid: g for g in load_eval_set(dataset, cfg=cfg)}
    except (FileNotFoundError, KeyError, ValueError):
        return {}


def discover_runs(
    datasets: Sequence[str] = DATASETS,
    *,
    cfg: Config | None = None,
    root: Path | None = None,
) -> dict[str, dict[str, RunData]]:
    """``{dataset: {config_name: RunData}}`` -- the latest run per pair.

    "Latest" is the lexicographically greatest ``run_id`` that actually holds
    records, which works because run ids embed a sortable UTC timestamp. The
    "holds records" clause matters: a directory created by a run that died
    before its first question would otherwise shadow a complete earlier one and
    silently empty a whole row of the results table.
    """
    cfg = cfg or load_config()
    base = Path(root) if root is not None else runs_root(cfg)
    out: dict[str, dict[str, RunData]] = {d: {} for d in datasets}
    if not base.is_dir():
        return out

    for dataset in datasets:
        golds = load_golds(dataset, cfg=cfg)
        for config_name in configurations(cfg):
            prefix = f"{config_name}_{dataset}_"
            candidates = sorted(
                (p for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix)),
                key=lambda p: p.name,
                reverse=True,
            )
            for candidate in candidates:
                run = load_run(candidate, golds=golds, cfg=cfg)
                if run.records:
                    out[dataset][config_name] = run
                    break
    return out


# ---------------------------------------------------------------------------
# Comparison plumbing
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _Comparisons:
    """Paired bootstraps of every configuration against one reference system."""

    reference: str
    available: bool
    per_metric: dict[str, dict[str, ComparisonResult]] = field(default_factory=dict)

    def get(self, config_name: str, metric: str) -> ComparisonResult | None:
        return self.per_metric.get(metric, {}).get(config_name)


def _compare_all(
    runs: Mapping[str, RunData],
    *,
    reference: str,
    metrics: Sequence[str],
    samples: int,
    seed: int,
) -> _Comparisons:
    """Paired-bootstrap every configuration against ``reference``.

    Only overlapping qids are compared -- :func:`paired_bootstrap` enforces that
    -- so a 250-question reference and a five-question smoke run are compared on
    the five they share and the interval widens accordingly, rather than
    claiming a precision the overlap does not support.
    """
    reference_run = runs.get(reference)
    comparisons = _Comparisons(reference=reference, available=reference_run is not None)
    if reference_run is None:
        return comparisons
    for metric in metrics:
        base = reference_run.metric(metric)
        if not base:
            continue
        row: dict[str, ComparisonResult] = {}
        for config_name, run in runs.items():
            if config_name == reference:
                continue
            other = run.metric(metric)
            if not other:
                continue
            row[config_name] = paired_bootstrap(
                base, other,
                name_a=reference, name_b=config_name,
                samples=samples, confidence=0.95, seed=seed,
            )
        comparisons.per_metric[metric] = row
    return comparisons


def _grouped(config_names: Sequence[str]) -> list[tuple[str | None, list[str]]]:
    """Split the configuration list into the baseline ladder and the agentic systems."""
    baselines = [c for c in config_names if c not in AGENTIC_CONFIGS]
    agentic = [c for c in config_names if c in AGENTIC_CONFIGS]
    groups: list[tuple[str | None, list[str]]] = []
    if baselines:
        groups.append(("Baselines (weakest first)", baselines))
    if agentic:
        groups.append(("Agentic system and ablations", agentic))
    return groups


def _significance_note(comparisons: _Comparisons) -> str:
    """The legend that keeps the marks readable -- and checkable."""
    if not comparisons.available:
        return (
            f"No {_tt(comparisons.reference)} run is available, so no significance marks "
            f"are shown. {_tt(MISSING)} marks a configuration that has not been run."
        )
    return (
        f"Reference system {_tt(comparisons.reference)} is marked $\\dagger$. "
        r"\textbf{Bold} = significantly better than the reference, "
        r"$^{\downarrow}$ = significantly worse "
        r"(paired bootstrap, 95\% CI of the paired difference excluding zero). "
        f"Unmarked differences are not significant. {_tt(MISSING)} marks a "
        "configuration that has not been run."
    )


# ---------------------------------------------------------------------------
# main_results.tex
# ---------------------------------------------------------------------------

def main_results_tables(
    runs: Mapping[str, RunData],
    dataset: str,
    *,
    cfg: Config,
    config_names: Sequence[str],
    samples: int,
    seed: int,
) -> str:
    """Answer quality and retrieval quality for every configuration.

    Two tables rather than one: nine systems against ten metrics with intervals
    does not fit an A4 text block, and shrinking one until it does produces a
    table nobody checks. Split by metric family, each half stays readable.
    """
    comparisons = _compare_all(
        runs,
        reference=REFERENCE_CONFIG,
        metrics=("em", "f1", "sp_f1", "recall@10", "ndcg@10"),
        samples=samples,
        seed=seed,
    )
    note = _significance_note(comparisons)

    def interval(run: RunData | None, metric: str) -> BootstrapResult | None:
        if run is None:
            return None
        values = list(run.metric(metric).values())
        return bootstrap_mean(values, samples=samples, seed=seed) if values else None

    answer_groups: list[tuple[str | None, Sequence[Sequence[str]]]] = []
    for title, names in _grouped(config_names):
        rows: list[list[str]] = []
        for name in names:
            run = runs.get(name)
            is_ref = name == REFERENCE_CONFIG
            rows.append([
                _tt(name),
                fmt_int(run.n_questions) if run is not None else MISSING,
                mark(run.mean("em") if run else None,
                     comparisons.get(name, "em"), reference=is_ref),
                mark(run.mean("f1") if run else None,
                     comparisons.get(name, "f1"), reference=is_ref),
                fmt_ci(interval(run, _CI_METRIC)),
                fmt(run.mean("sp_em") if run else None),
                mark(run.mean("sp_f1") if run else None,
                     comparisons.get(name, "sp_f1"), reference=is_ref),
                MISSING if is_ref else fmt_delta(comparisons.get(name, "f1")),
            ])
        answer_groups.append((title, rows))

    answer = table(
        colspec="lrrrcrrr",
        header=_header(
            "System", "$n$", "EM", "F1", r"F1 95\% CI", "SP-EM", "SP-F1",
            r"$\Delta$F1 vs.\ ref.\ [95\% CI]",
        ),
        groups=answer_groups,
        caption=(
            f"Answer quality on {_tt(dataset)}. Exact match and token-level F1 follow the "
            "official HotpotQA evaluation script, including its yes/no short circuit; "
            r"SP-EM and SP-F1 are set metrics over $(\textit{title}, \textit{sent\_id})$ "
            f"supporting-fact pairs. $n$ is the number of questions scored. {note}"
        ),
        label=f"tab:main-answer-{dataset}",
    )

    retrieval_groups: list[tuple[str | None, Sequence[Sequence[str]]]] = []
    for title, names in _grouped(config_names):
        rows = []
        for name in names:
            run = runs.get(name)
            is_ref = name == REFERENCE_CONFIG
            rows.append([
                _tt(name),
                fmt(run.mean("recall@2") if run else None),
                fmt(run.mean("recall@5") if run else None),
                mark(run.mean("recall@10") if run else None,
                     comparisons.get(name, "recall@10"), reference=is_ref),
                mark(run.mean("ndcg@10") if run else None,
                     comparisons.get(name, "ndcg@10"), reference=is_ref),
                fmt_ci(interval(run, _RETRIEVAL_CI_METRIC)),
                fmt(run.mean("mrr") if run else None),
            ])
        retrieval_groups.append((title, rows))

    retrieval = table(
        colspec="lrrrrcr",
        header=_header(
            "System", "R@2", "R@5", "R@10", "nDCG@10", r"nDCG@10 95\% CI", "MRR",
        ),
        groups=retrieval_groups,
        caption=(
            f"Retrieval quality on {_tt(dataset)}, computed with "
            r"\texttt{pytrec\_eval} over the reciprocal-rank fusion of every sub-query's "
            "ranking and judged against the gold supporting-fact documents. A multi-hop "
            "system issues several queries, so ``the ranking'' is not directly observed; "
            "fusing them is how the orchestrator itself pools evidence, and it needs no "
            f"score calibration between BM25 and cosine. {note}"
        ),
        label=f"tab:main-retrieval-{dataset}",
    )
    return answer + "\n\n" + retrieval


# ---------------------------------------------------------------------------
# agent_metrics.tex
# ---------------------------------------------------------------------------

def agent_metrics_table(
    runs: Mapping[str, RunData],
    dataset: str,
    *,
    config_names: Sequence[str],
) -> str:
    """Agent-specific cost, reported beside quality and never instead of it.

    A system that gains three F1 points for twenty times the compute has not
    obviously won, and this is the table that lets a reader say so. Runs whose
    ``meta.json`` reports a warm cache are called out in the caption, because a
    warm-cache latency is not comparable to a cold-cache one and the difference
    is invisible in the number itself.
    """
    groups: list[tuple[str | None, Sequence[Sequence[str]]]] = []
    warm: list[str] = []
    for title, names in _grouped(config_names):
        rows: list[list[str]] = []
        for name in names:
            run = runs.get(name)
            if run is None:
                rows.append([_tt(name)] + [MISSING] * 8)
                continue
            if not bool(run.meta.get("cache_cold", True)):
                warm.append(name)
            m = run.agent_metrics()
            rows.append([
                _tt(name),
                fmt(m.llm_calls, 2),
                fmt(m.llm_calls_saved, 2),
                fmt(m.tool_calls, 2),
                fmt(m.latency_s, 1),
                fmt(m.plan_depth, 2),
                fmt(m.n_subqueries, 2),
                fmt(m.replan_rate, 3),
                fmt(m.citation_grounding, 3),
            ])
        groups.append((title, rows))

    caveat = ""
    if warm:
        names = ", ".join(_tt(c) for c in sorted(set(warm)))
        caveat = (
            f" \\textbf{{Latency for {names} came from a warm LLM cache and is not "
            r"comparable}}; \texttt{meta.json} records the cache state of every leg."
        )
    return table(
        colspec="lrrrrrrrr",
        header=_header(
            "System", "LLM calls", "Saved", "Tool calls", "Latency (s)",
            "Plan depth", "Sub-queries", "Re-plan rate", "Cite grounding",
        ),
        groups=groups,
        caption=(
            f"Agent-specific cost on {_tt(dataset)}, per question. "
            r"\textit{LLM calls} counts logical calls, cache hits included; "
            r"\textit{Saved} counts calls a deterministic rule replaced; "
            r"\textit{Re-plan rate} is the fraction of questions that triggered at least "
            r"one re-plan, not the mean count; \textit{Cite grounding} is averaged only "
            "over questions that produced a non-empty answer, so it cannot be inflated by "
            "abstentions. Latency is only comparable between cold-cache runs."
            + caveat
            + f" {_tt(MISSING)} marks a configuration that has not been run."
        ),
        label=f"tab:agent-metrics-{dataset}",
    )


# ---------------------------------------------------------------------------
# ablations.tex
# ---------------------------------------------------------------------------

def ablations_table(
    runs: Mapping[str, RunData],
    dataset: str,
    *,
    samples: int,
    seed: int,
) -> str:
    """What each component is worth, measured against the full system.

    The reference here is ``agentic_full`` and not ``hybrid_rerank``: an
    ablation's number only means something as a difference from the system it
    removes a component from. A negative delta therefore means the component was
    helping, and it is reported whether or not it flatters the architecture --
    an honestly analysed negative ablation is worth more than a suppressed one.
    """
    comparisons = _compare_all(
        runs, reference=ABLATION_REFERENCE, metrics=("em", "f1"),
        samples=samples, seed=seed,
    )

    ordered: list[str] = []
    for name in (ABLATION_REFERENCE, *AGENTIC_CONFIGS, REFERENCE_CONFIG):
        if name not in ordered:
            ordered.append(name)

    rows: list[list[str]] = []
    for name in ordered:
        run = runs.get(name)
        is_ref = name == ABLATION_REFERENCE
        agent = run.agent_metrics() if run is not None else None
        label = _tt(name)
        if name == REFERENCE_CONFIG:
            label += r" \textit{(non-agentic floor)}"
        rows.append([
            label,
            fmt_int(run.n_questions) if run is not None else MISSING,
            mark(run.mean("em") if run else None,
                 comparisons.get(name, "em"), reference=is_ref),
            mark(run.mean("f1") if run else None,
                 comparisons.get(name, "f1"), reference=is_ref),
            MISSING if is_ref else fmt_delta(comparisons.get(name, "f1")),
            MISSING if is_ref else fmt_p(comparisons.get(name, "f1")),
            fmt(agent.llm_calls, 2) if agent else MISSING,
            fmt(agent.latency_s, 1) if agent else MISSING,
        ])

    if comparisons.available:
        note = (
            f"Deltas are measured against {_tt(ABLATION_REFERENCE)} ($\\dagger$), so a "
            r"\textit{negative} $\Delta$F1 means the removed component was helping. "
            r"\textbf{Bold} marks an ablation significantly \textit{better} than the full "
            r"system and $^{\downarrow}$ one significantly worse (paired bootstrap, "
            r"95\% CI of the difference excluding zero); unmarked differences are not "
            "significant."
        )
    else:
        note = (
            f"No {_tt(ABLATION_REFERENCE)} run is available, so no deltas or significance "
            "marks can be computed."
        )
    return table(
        colspec="lrrrrrrr",
        header=_header(
            "System", "$n$", "EM", "F1", r"$\Delta$F1 vs.\ full [95\% CI]", "$p$",
            "LLM calls", "Latency (s)",
        ),
        groups=[(None, rows)],
        caption=(
            f"Ablation study on {_tt(dataset)}. {note} "
            f"{_tt(REFERENCE_CONFIG)} is repeated from the main results as the "
            "non-agentic floor, since removing planning, synthesis and verification "
            f"reduces the pipeline to it. {_tt(MISSING)} marks a configuration that has "
            "not been run."
        ),
        label=f"tab:ablations-{dataset}",
    )


# ---------------------------------------------------------------------------
# dataset_stats.tex
# ---------------------------------------------------------------------------

def _corpus_stats(dataset: str, *, cfg: Config) -> dict[str, Any]:
    """``{dataset}_corpus_stats.json``, as written by ``scripts/build_corpus.py``."""
    import orjson

    path = cfg.resolve_path("paths.processed") / f"{dataset}_corpus_stats.json"
    if not path.exists():
        return {}
    try:
        return orjson.loads(path.read_bytes())
    except Exception:  # noqa: BLE001 - a missing statistic is a "--", not a crash
        return {}


def _eval_slice_strata(dataset: str, *, cfg: Config) -> dict[str, int]:
    """Question-type counts in the frozen eval slice."""
    counts: dict[str, int] = {}
    for gold in load_golds(dataset, cfg=cfg).values():
        key = str(gold.qtype or "unlabelled")
        counts[key] = counts.get(key, 0) + 1
    return counts


def dataset_stats_tables(datasets: Sequence[str], *, cfg: Config) -> str:
    """Corpus statistics and the composition of the evaluation slice.

    Datasets are columns rather than rows, because the comparison a reader
    wants is between the two benchmarks on the same statistic.
    """
    stats = {d: _corpus_stats(d, cfg=cfg) for d in datasets}

    def cell(dataset: str, *path: str, digits: int | None = None) -> str:
        node: Any = stats.get(dataset) or {}
        for key in path:
            if not isinstance(node, Mapping) or key not in node:
                return MISSING
            node = node[key]
        if node is None:
            return MISSING
        return fmt(node, digits) if digits is not None else fmt_count(node)

    def sample_size(dataset: str) -> str:
        return fmt_int(cfg.get(f"datasets.{dataset}.eval_sample", None))

    spec: list[tuple[str, Callable[[str], str]]] = [
        ("Questions in split", lambda d: cell(d, "questions")),
        ("Paragraphs read", lambda d: cell(d, "paragraphs_read")),
        ("Passages after de-duplication", lambda d: cell(d, "passages")),
        ("Identical paragraphs dropped", lambda d: cell(d, "dedupe", "repeat_identical")),
        ("Title collisions resolved",
         lambda d: cell(d, "dedupe", "collisions_same_title_different_text")),
        ("Sentences", lambda d: cell(d, "sentences", "total")),
        ("Mean sentences per passage",
         lambda d: cell(d, "sentences", "mean_per_passage", digits=2)),
        ("Mean passage length (chars)", lambda d: cell(d, "passage_chars", "mean", digits=1)),
        ("Mean sentence length (chars)", lambda d: cell(d, "sentences", "mean_chars", digits=1)),
        ("Questions with gold triples", lambda d: cell(d, "gold", "questions_with_triples")),
        ("Gold supporting facts", lambda d: cell(d, "validation", "gold_facts")),
        ("Supporting facts unresolvable", lambda d: cell(d, "validation", "unresolved")),
        ("Evaluation sample", sample_size),
    ]
    rows = [[latex_escape(name)] + [getter(d) for d in datasets] for name, getter in spec]
    corpus = table(
        colspec="l" + "r" * len(datasets),
        header=_header("Statistic", *[_tt(d) for d in datasets]),
        groups=[(None, rows)],
        caption=(
            "Corpus statistics after preprocessing, read from "
            r"\texttt{\{dataset\}\_corpus\_stats.json}. Passages are de-duplicated by "
            "content under a canonical title-derived id; sentence splits are the "
            "datasets' own and are persisted verbatim, because gold supporting facts "
            "index into them by position and re-splitting anywhere downstream would "
            "renumber the ground truth while every metric still looked plausible. "
            f"{_tt(MISSING)} marks a statistic the build did not record."
        ),
        label="tab:dataset-corpus",
    )

    sample_groups: list[tuple[str | None, Sequence[Sequence[str]]]] = []
    for dataset in datasets:
        split_strata = ((stats.get(dataset) or {}).get("strata") or {}).get("type") or {}
        sample_strata = _eval_slice_strata(dataset, cfg=cfg)
        split_total = sum(split_strata.values())
        sample_total = sum(sample_strata.values())
        rows = []
        for key in sorted(set(split_strata) | set(sample_strata)):
            in_split = split_strata.get(key)
            in_sample = sample_strata.get(key)
            rows.append([
                latex_escape(key),
                fmt_count(in_split),
                fmt(in_split / split_total, 3) if in_split and split_total else MISSING,
                fmt_count(in_sample),
                fmt(in_sample / sample_total, 3) if in_sample and sample_total else MISSING,
            ])
        if rows:
            rows.append([
                r"\textit{total}",
                fmt_count(split_total) if split_total else MISSING,
                fmt(1.0, 3) if split_total else MISSING,
                fmt_count(sample_total) if sample_total else MISSING,
                fmt(1.0, 3) if sample_total else MISSING,
            ])
        else:
            rows.append([MISSING] * 5)
        sample_groups.append((_tt(dataset), rows))

    sample = table(
        colspec="lrrrr",
        header=_header("Question type", "Split $n$", "Split share", "Sample $n$", "Sample share"),
        groups=sample_groups,
        caption=(
            "Composition of the frozen evaluation slice against the full split. The slice "
            "is stratified on question type and sampled once, with seed 42, by "
            r"\texttt{scripts/sample\_eval\_set.py}; its SHA-256 is recorded in every "
            r"run's \texttt{meta.json} and it is never re-sampled at run time. HotpotQA's "
            r"validation split is entirely \texttt{level=hard}, so difficulty gives one "
            "stratum and a silently unstratified sample; question type is the informative "
            "split there."
        ),
        label="tab:dataset-sample",
    )
    return corpus + "\n\n" + sample


# ---------------------------------------------------------------------------
# error_analysis.tex
# ---------------------------------------------------------------------------

def error_analysis_table(
    runs: Mapping[str, RunData],
    dataset: str,
    *,
    cfg: Config,
    config_names: Sequence[str],
    corpus_titles: frozenset[str] | None = None,
) -> str:
    """The failure profile of every run, from ``eval/error_analysis.py``.

    Configurations are columns and labels are rows -- the transpose of the
    natural reading order, and the only orientation in which nine systems fit
    the page. Each cell is ``count (share of ALL questions)``: a rate over
    errors moves whenever accuracy moves, which would make two systems' profiles
    incomparable exactly when the comparison matters.
    """
    golds = load_golds(dataset, cfg=cfg)
    present = [c for c in config_names if c in runs]
    summaries: dict[str, ErrorSummary] = {
        name: summarise(
            analyse(runs[name].records, golds or None, corpus_titles=corpus_titles)
        )
        for name in present
    }
    columns = present or [""]

    def counted(name: str, label: str) -> str:
        summary = summaries.get(name)
        if summary is None or not summary.n_questions:
            return MISSING
        return f"{summary.counts.get(label, 0)} ({summary.rate(label):.3f})"

    def summary_cell(name: str, getter: Callable[[ErrorSummary], str]) -> str:
        summary = summaries.get(name)
        return MISSING if summary is None else getter(summary)

    label_rows = [
        [_tt(label)] + [counted(name, label) for name in columns] for label in ERROR_LABELS
    ]
    total_spec: list[tuple[str, Callable[[ErrorSummary], str]]] = [
        ("Questions scored", lambda s: fmt_int(s.n_questions)),
        ("Correct", lambda s: fmt_int(s.n_correct)),
        ("Accuracy", lambda s: fmt(s.accuracy, 3)),
        ("Errors", lambda s: fmt_int(s.n_errors)),
        ("Errors with no rule firing", lambda s: fmt_int(s.unlabelled)),
        ("Verifier false accepts", lambda s: fmt_int(s.false_accepts)),
        ("Verifier false rejects", lambda s: fmt_int(s.false_rejects)),
        ("Recoverable (some cycle was right)", lambda s: fmt_int(s.recoverable)),
    ]
    total_rows = [
        [latex_escape(name)] + [summary_cell(c, getter) for c in columns]
        for name, getter in total_spec
    ]

    corpus_note = (
        "Corpus titles were loaded, so the "
        r"\texttt{decomposition\_error} rule's ``gold facts exist in the corpus'' clause "
        "is checked rather than assumed."
        if corpus_titles is not None
        else "Corpus titles were not loaded, so the "
        r"\texttt{decomposition\_error} rule assumes gold facts exist in the corpus -- "
        "the permissive direction, which can blame the decomposition for a genuine "
        "corpus gap."
    )
    return table(
        colspec="l" + "r" * len(columns),
        header=_header("Label", *[_tt(c) if c else MISSING for c in columns]),
        groups=[
            ("Failure labels: count (share of all questions)", label_rows),
            ("Totals", total_rows),
        ],
        caption=(
            f"Failure profile on {_tt(dataset)}. Labels are assigned post hoc by "
            r"\texttt{eval/error\_analysis.py} from the trace and the gold supporting "
            "facts -- deterministically, first match wins -- in a cascade from ``the "
            "system never had a chance'' to ``it had everything it needed and still got "
            r"it wrong''. \texttt{verifier\_false\_accept} and "
            r"\texttt{verifier\_false\_reject} are counted again in the totals, "
            "independently of the cascade: the ordered label makes a false reject nearly "
            "unreachable, and it is the count that makes the "
            r"Verifier$\rightarrow$Planner loop falsifiable. "
            f"{corpus_note} A configuration with no run has no column."
        ),
        label=f"tab:error-analysis-{dataset}",
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _preamble(title: str, runs: Mapping[str, Mapping[str, RunData]]) -> str:
    """A provenance header naming the runs the numbers came from.

    Deliberately carries no wall-clock timestamp: regenerating from unchanged
    runs must produce a byte-identical file, so that a diff under
    ``results/tables/`` always means a number actually moved.
    """
    lines = [
        f"% {title}",
        "% GENERATED by src/agentic_ir/eval/tables.py -- do not edit by hand.",
        "% Regenerate with: python -m agentic_ir.cli tables",
        "% Requires \\usepackage{booktabs} and \\usepackage{graphicx};",
        "% report/main.tex already loads both.",
    ]
    sources = [
        f"%   {dataset}/{config}: {run.run_id} ({run.n_questions} questions)"
        for dataset, by_config in sorted(runs.items())
        for config, run in sorted(by_config.items())
    ]
    if sources:
        lines.append("% Source runs:")
        lines.extend(sources)
    else:
        lines.append("% Source runs: none found -- every cell renders as --.")
    return "\n".join(lines) + "\n\n"


def generate_all(
    *,
    cfg: Config | None = None,
    datasets: Sequence[str] = DATASETS,
    out_dir: Path | None = None,
    runs_dir: Path | None = None,
    samples: int | None = None,
    seed: int | None = None,
    use_corpus_titles: bool = True,
) -> dict[str, Path]:
    """Write every fragment into ``out_dir``. Returns ``{filename: path}``.

    All five files are written even when there is not a single run: a chapter's
    ``\\input`` has to resolve, or the report stops building for a reason that
    has nothing to do with the results.
    """
    cfg = cfg or load_config()
    samples = int(cfg.get("evaluation.bootstrap_samples", 1000)) if samples is None else samples
    seed = int(cfg.get("project.seed", 42)) if seed is None else seed
    target = Path(out_dir) if out_dir is not None else cfg.resolve_path("paths.results") / "tables"
    target.mkdir(parents=True, exist_ok=True)

    datasets = list(datasets)
    config_names = list(configurations(cfg))
    runs = discover_runs(datasets, cfg=cfg, root=runs_dir)

    titles: dict[str, frozenset[str] | None] = dict.fromkeys(datasets)
    if use_corpus_titles:
        from ..indexing.corpus import Corpus

        for dataset in datasets:
            if not runs.get(dataset):
                continue  # loading 66k passages to analyse nothing is pure cost
            try:
                titles[dataset] = frozenset(Corpus.load(dataset, cfg=cfg).titles())
            except Exception:  # noqa: BLE001 - the taxonomy degrades, it does not fail
                titles[dataset] = None

    written: dict[str, Path] = {}

    def write(filename: str, title: str, body: str) -> None:
        path = target / filename
        path.write_text(_preamble(title, runs) + body + "\n", encoding="utf-8")
        written[filename] = path

    write(
        "main_results.tex",
        "Chapter 4 -- main results",
        "\n\n".join(
            main_results_tables(
                runs[d], d, cfg=cfg, config_names=config_names, samples=samples, seed=seed
            )
            for d in datasets
        ),
    )
    write(
        "agent_metrics.tex",
        "Chapter 4 -- agent-specific measures",
        "\n\n".join(
            agent_metrics_table(runs[d], d, config_names=config_names) for d in datasets
        ),
    )
    write(
        "ablations.tex",
        "Chapter 4 -- ablation study",
        "\n\n".join(ablations_table(runs[d], d, samples=samples, seed=seed) for d in datasets),
    )
    write(
        "dataset_stats.tex",
        "Chapter 2 -- data and preprocessing",
        dataset_stats_tables(datasets, cfg=cfg),
    )
    write(
        "error_analysis.tex",
        "Chapter 4 -- error analysis",
        "\n\n".join(
            error_analysis_table(
                runs[d], d, cfg=cfg, config_names=config_names, corpus_titles=titles[d]
            )
            for d in datasets
        ),
    )
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agentic_ir.eval.tables",
        description="Regenerate every LaTeX fragment in results/tables/ from results/runs/.",
    )
    parser.add_argument("--dataset", action="append", dest="datasets", choices=DATASETS,
                        help="restrict to one dataset (repeatable; default: both)")
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument("--runs", type=Path, default=None, help="run directory root")
    parser.add_argument("--bootstrap", type=int, default=None,
                        help="resamples (default: evaluation.bootstrap_samples)")
    parser.add_argument("--no-corpus-titles", action="store_true",
                        help="skip loading the corpus for the error taxonomy: faster, and "
                             "more permissive about decomposition_error")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    written = generate_all(
        cfg=load_config(),
        datasets=args.datasets or list(DATASETS),
        out_dir=args.out,
        runs_dir=args.runs,
        samples=args.bootstrap,
        use_corpus_titles=not args.no_corpus_titles,
    )
    for name in TABLE_FILES:
        path = written.get(name)
        if path is None:
            continue
        try:
            shown: Path | str = path.relative_to(PROJECT_ROOT)
        except ValueError:
            shown = path
        print(f"wrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
