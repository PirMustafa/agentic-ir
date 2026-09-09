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

**A metric a system never attempted renders as** ``--`` **as well.** The three
retrieval-only baselines emit a ranking and no answer at all; ``score_records``
scores their empty answer string and so produces ``em = f1 = 0.0``, which is
arithmetic on a quantity the system never produced. Printed, it asserts that
they tried and failed, and it hands every generative system a free
several-hundred-point margin over a floor that does not exist. Answer columns
are therefore blanked for a run whose trace contains no answer, and the caption
says explicitly that this ``--`` means *does not attempt answers* rather than
*not run*. Which of the two a row is showing is not guessable from the glyph,
so it is stated.

**Significance is marked, not implied, and only between comparable runs.** A
cell is bolded only when a paired bootstrap puts the whole 95% interval of the
difference on one side of zero. A two-point gap on 250 questions is usually
noise, and a table that bolds it overstates its evidence. Two further
conditions gate the comparison itself:

* *The reference has to be able to hold the metric.* ``hybrid_rerank`` is the
  strongest non-agentic baseline on retrieval and supporting facts and is the
  reference there, but it answers no questions, so it cannot be the reference
  for EM and F1. Those columns are referenced instead to the strongest
  non-agentic system that does answer, chosen from the runs on disk and named
  in the caption. Comparing answers against a system that never answered
  measures the answering, not the improvement.
* *Both runs have to cover the same evaluation slice.* A run still in flight
  holds a prefix of the questions, and a paired test between a prefix and the
  full slice is not a paired test. Such a row keeps its point estimates, is
  marked ``$^{\ddagger}$`` beside an ``n`` that shows how far it got, and has
  its deltas, p-values and significance marks withheld. The rule is a
  comparison of ``n`` against ``datasets.{dataset}.eval_sample``, so a growing
  run rejoins the comparison by itself the moment it finishes.

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
import csv
import statistics
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
from .metrics import (
    AgentMetrics,
    derive_question_metrics,
    reachable_llm_paths,
    summarise_agent_metrics,
)
from .run_eval import (
    AGENTIC_CONFIGS,
    config_for,
    configurations,
    resolved_citations,
    runs_root,
    score_records,
)

__all__ = [
    "ABLATION_REFERENCE",
    "ANSWER_METRICS",
    "MISSING",
    "PARTIAL_MARK",
    "REFERENCE_CONFIG",
    "TABLE_FILES",
    "RunData",
    "ablations_table",
    "agent_metrics_table",
    "dataset_stats_tables",
    "answer_reference",
    "discover_runs",
    "error_analysis_table",
    "eval_sample_size",
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

#: What a run that has not covered the whole evaluation slice is marked with.
#: It keeps its point estimates and loses its deltas: the numbers are real, the
#: comparison is not.
PARTIAL_MARK = r"$^{\ddagger}$"

#: What a supporting-fact score that is a harness artefact rather than a
#: measurement is marked with. See :meth:`RunData.citations_unresolved`.
ARTEFACT_MARK = r"$^{\ast}$"

#: The system the retrieval and supporting-fact comparisons are made against.
#: It is the strongest non-agentic baseline on those metrics, so beating it is
#: the claim Chapter 4 has to earn; comparing against a weaker one would
#: flatter the agentic system for free.
#:
#: It is deliberately *not* the reference for EM and F1: it is retrieval-only
#: and answers nothing, so a difference against it is not an improvement in
#: answering, it is the whole of the other system's score. See
#: :func:`answer_reference`.
REFERENCE_CONFIG = "hybrid_rerank"

#: Metrics that presuppose the system emitted an answer. A run that emitted
#: none has no value for these -- not a zero -- and takes no part in a paired
#: comparison on them, whether as the reference or as the other side.
ANSWER_METRICS: frozenset[str] = frozenset({"em", "f1"})

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


def _hours(seconds: float) -> str:
    """A duration in hours, for a number too large for seconds to convey."""
    return f"{seconds / 3600.0:.1f}\\,h"


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



# ---------------------------------------------------------------------------
# The improvement loop's tables
#
# Deliberately not part of :data:`TABLE_FILES` and deliberately unreachable
# from :func:`discover_runs`. Those two facts are the design: the report's
# nine-system grid is generated from ``results/runs``, and the improvement
# loop's runs live in ``results/runs_v2`` precisely so that no amount of
# regenerating the report's tables can pull them in. A reader who wants the
# loop's numbers in LaTeX asks for them by name, here.
# ---------------------------------------------------------------------------

#: The loop's runs, by (role, dataset). Keyed by role rather than by
#: configuration name because ``L1c_think_*`` records ``agentic_no_planner`` in
#: its own ``meta.json`` -- the thinking settings reached it through the config
#: file it ran under, which is the provenance gap
#: ``docs/improvement-loop.log.md`` records under Loop 5.
V2_RUNS: dict[tuple[str, str], str] = {
    ("kept", "hotpotqa"): "L1c_think_hotpotqa",
    ("kept", "twowiki"): "L1c_think_twowiki",
    ("no_think", "hotpotqa"): "L1c_base_hotpotqa",
    ("no_think", "twowiki"): "L1c_base_twowiki",
}

#: The three systems a reader will ask to see the kept configuration against.
#: Their runs are the report's own, read from ``results/runs``.
V2_REFERENCES: tuple[str, ...] = ("agentic_full", "agentic_no_planner", "self_ask")

#: Disjoint from :data:`TABLE_FILES` by construction, and a test says so.
V2_TABLE_FILES: tuple[str, ...] = ("v2_results.tex", "v2_cost.tex", "v2_failures.tex")

#: Its own banner, naming its own entry point, because ``cli tables`` will not
#: regenerate these and must not: it would have to look at ``results/runs_v2``.
V2_GENERATED_BY = "\n".join(
    (
        "% GENERATED by src/agentic_ir/eval/tables.py::write_v2_tables -- do not edit by hand.",
        "% Source runs: results/runs_v2 (the improvement loop). `cli tables` neither",
        "% writes nor reads these -- discover_runs cannot see that directory, which is",
        "% what keeps the report's nine-system grid exactly nine systems.",
        "% Requires \\usepackage{booktabs} and \\usepackage{graphicx}.",
        "",
        "",
    )
)


def _v2_scores(run_dir: Path, field: str = "em") -> dict[str, float]:
    """``{qid: score}`` straight from ``scores.csv``.

    Read rather than recomputed. These runs are evidence; a table that re-scores
    them is a table of what this code believes today, not of what was measured.
    """
    out: dict[str, float] = {}
    with (run_dir / "scores.csv").open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["qid"]] = float(row[field])
    return out


def _v2_call_stats(run_dir: Path) -> dict[str, Any]:
    """Latency, model calls, and the synthesis-failure census, from the trace.

    The census counts per *call*, inside ``traces[*].llm_calls[*]``. A
    question-level metric cannot see it -- the loop spent two rounds reading
    one and concluding the wrong thing -- so this walks the calls.
    """
    latencies: list[float] = []
    calls: list[float] = []
    synth_total = synth_failed = 0
    q_failed = q_retried = 0
    for record in iter_records(run_dir / "traces.jsonl"):
        metrics = record.get("metrics") or {}
        if metrics.get("latency_s") is not None:
            latencies.append(float(metrics["latency_s"]))
        if metrics.get("llm_calls") is not None:
            calls.append(float(metrics["llm_calls"]))
        failed = retried = False
        for trace in record.get("traces") or record.get("steps") or []:
            for call in trace.get("llm_calls") or []:
                if call.get("agent") != "synthesizer":
                    continue
                synth_total += 1
                if call.get("parse_ok") is False:
                    synth_failed += 1
                    failed = True
                elif (call.get("retries") or 0) > 0:
                    retried = True
        q_failed += int(failed)
        q_retried += int(retried)
    latencies.sort()
    n = len(latencies)
    return {
        "n": n,
        "median_s": statistics.median(latencies) if latencies else float("nan"),
        "p90_s": latencies[int(0.9 * n)] if n else float("nan"),
        "mean_calls": statistics.mean(calls) if calls else float("nan"),
        "synth_total": synth_total,
        "synth_failed": synth_failed,
        "q_failed": q_failed,
        "q_retried": q_retried,
    }


def write_v2_tables(
    *,
    v2_root: Path | None = None,
    runs_root_dir: Path | None = None,
    out_dir: Path | None = None,
    cfg: Config | None = None,
    samples: int = 2000,
) -> list[Path]:
    r"""Generate the improvement loop's LaTeX tables. Returns what it wrote.

    ``v2_root`` holds the loop's runs (default ``results/runs_v2``);
    ``runs_root_dir`` holds the report's, for the reference columns. Nothing
    here consults :func:`discover_runs`, so running this can never change a
    table that ``report/`` inputs.
    """
    cfg = cfg or load_config()
    results = Path(cfg.get("paths.results", "results"))
    v2_root = Path(v2_root) if v2_root is not None else results / "runs_v2"
    runs_root_dir = Path(runs_root_dir) if runs_root_dir is not None else results / "runs"
    out_dir = Path(out_dir) if out_dir is not None else results / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    def newest(prefix: str) -> Path | None:
        found = sorted(d for d in runs_root_dir.glob(prefix + "_*") if (d / "scores.csv").exists())
        return found[-1] if found else None

    written: list[Path] = []
    datasets = (("hotpotqa", "HotpotQA"), ("twowiki", "2WikiMultihopQA"))

    # -- what it scores ------------------------------------------------------
    groups: list[tuple[str | None, list[list[str]]]] = []
    for dataset, pretty in datasets:
        kept_dir = v2_root / V2_RUNS[("kept", dataset)]
        if not (kept_dir / "scores.csv").exists():
            continue
        kept_em = _v2_scores(kept_dir, "em")
        kept_f1 = _v2_scores(kept_dir, "f1")
        rows: list[list[str]] = [[
            r"\textbf{agentic\_v2} (kept)",
            fmt_int(len(kept_em)),
            r"\textbf{" + fmt(sum(kept_em.values()) / len(kept_em)) + "}",
            r"\textbf{" + fmt(sum(kept_f1.values()) / len(kept_f1)) + "}",
            "--", "--", "--",
        ]]
        for name in V2_REFERENCES:
            ref_dir = newest(name + "_" + dataset)
            if ref_dir is None:
                continue
            ref_em = _v2_scores(ref_dir, "em")
            ref_f1 = _v2_scores(ref_dir, "f1")
            delta = paired_bootstrap(ref_em, kept_em, name, "agentic_v2", samples=samples)
            shared = sorted(set(ref_em) & set(kept_em))
            gained = sum(1 for q in shared if kept_em[q] > ref_em[q])
            lost = sum(1 for q in shared if kept_em[q] < ref_em[q])
            rows.append([
                _tt(name),
                fmt_int(len(ref_em)),
                fmt(sum(ref_em.values()) / len(ref_em)),
                fmt(sum(ref_f1.values()) / len(ref_f1)),
                fmt_delta(delta),
                fmt_p(delta),
                str(gained) + r"\,/\," + str(lost),
            ])
        groups.append((pretty, rows))

    path = out_dir / "v2_results.tex"
    path.write_text(
        V2_GENERATED_BY
        + table(
            colspec="lrrrlrr",
            header=_header(
                "System", "$n$", "EM", "F1",
                r"$\Delta$EM vs. kept [95\% CI]", "$p$", r"gained\,/\,lost",
            ),
            groups=groups,
            caption=(
                "The improvement loop's kept configuration against the reported system, its "
                "own base, and the strongest answer baseline. Paired on question id over the "
                "same frozen 250-question slice. Five of the six intervals exclude zero; the "
                "exception is \\texttt{self\\_ask} on HotpotQA, where the kept configuration "
                "ties a single-model baseline that has no agents in it."
            ),
            label="tab:v2-results",
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(path)

    # -- what it costs -------------------------------------------------------
    cost_rows: list[list[str]] = []
    for dataset, pretty in datasets:
        kept_dir = v2_root / V2_RUNS[("kept", dataset)]
        if not (kept_dir / "traces.jsonl").exists():
            continue
        kept = _v2_call_stats(kept_dir)
        cost_rows.append([
            pretty,
            r"\textbf{agentic\_v2} (kept)",
            r"\textbf{" + f"{kept['median_s']:.1f}" + "}",
            f"{kept['p90_s']:.1f}",
            r"\textbf{" + f"{kept['mean_calls']:.2f}" + "}",
        ])
        ref_dir = newest("agentic_full_" + dataset)
        if ref_dir is not None:
            ref = _v2_call_stats(ref_dir)
            cost_rows.append([
                "", _tt("agentic_full"),
                f"{ref['median_s']:.1f}", f"{ref['p90_s']:.1f}", f"{ref['mean_calls']:.2f}",
            ])
    path = out_dir / "v2_cost.tex"
    path.write_text(
        V2_GENERATED_BY
        + table(
            colspec="llrrr",
            header=_header("Dataset", "System", "Median (s)", "p90 (s)", "LLM calls"),
            groups=[(None, cost_rows)],
            caption=(
                "Cost per question. The kept configuration answers with roughly a third of the "
                "model calls and a faster median, and pays for it in the tail: a synthesis call "
                "that reasons may run to the full 5{,}120-token completion budget."
            ),
            label="tab:v2-cost",
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(path)

    # -- what it costs that the scores do not show ---------------------------
    census_rows: list[list[str]] = []
    for role, label in (("no_think", "reasoning off"), ("kept", "reasoning on")):
        for dataset, pretty in datasets:
            run_dir = v2_root / V2_RUNS[(role, dataset)]
            if not (run_dir / "traces.jsonl").exists():
                continue
            st = _v2_call_stats(run_dir)
            n = st["n"] or 1
            census_rows.append([
                pretty + ", " + label,
                fmt_int(st["n"]),
                fmt_int(st["synth_total"]),
                f"{st['q_failed']} ({st['q_failed'] / n * 100:.1f}\\%)",
                f"{st['q_retried']} ({st['q_retried'] / n * 100:.1f}\\%)",
            ])
    path = out_dir / "v2_failures.tex"
    path.write_text(
        V2_GENERATED_BY
        + table(
            colspec="lrrrr",
            header=_header(
                "Condition", "$n$", "Synthesis calls",
                "Questions losing one", "Questions retrying",
            ),
            groups=[(None, census_rows)],
            caption=(
                "What reasoning on the synthesis call costs. The model spends its entire "
                "generation budget inside the thinking channel and never opens the content "
                "channel (\\texttt{done\\_reason} \\texttt{length} at the "
                "\\texttt{num\\_predict} cap), and the extractive fallback answers instead. "
                "Every one of these losses is already inside the exact-match figures of "
                "Table~\\ref{tab:v2-results}."
            ),
            label="tab:v2-failures",
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(path)
    return written


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
    #: The configuration the run executed under -- ``meta.json``'s snapshot
    #: when it has one, the project config with the ablation overrides
    #: otherwise. It decides which LLM calls were *reachable*, which is what
    #: the saved-call recount needs.
    config_snapshot: Any = None

    @property
    def n_questions(self) -> int:
        return len(self.records)

    @property
    def n_answered(self) -> int:
        """Questions for which the trace holds a non-empty ``final_answer``."""
        return sum(
            1 for record in self.records if str(record.get("final_answer") or "").strip()
        )

    @property
    def attempts_answers(self) -> bool:
        """Whether this system produces answers at all.

        Read off the trace rather than off a hard-coded list of retrieval-only
        configurations, so a baseline that gains or loses a generation stage
        is classified by what it did, not by what this module was told about it
        when it was written. ``run_eval._BaselineAdapter`` is explicit that a
        retrieval-only baseline leaves the answer unset precisely because "it
        never claimed one"; the scorer nevertheless scores the empty string,
        and this property is what stops that zero reaching the page.
        """
        return self.n_answered > 0

    @property
    def citations_unresolved(self) -> bool:
        r"""Whether every citation this run emitted names no evidence it holds.

        ``run_eval.predicted_supporting_facts`` intersects ``citations`` with
        the evidence pool by ``evidence_id``. When a system writes its
        citations in a different id namespace than its evidence -- baselines
        cite passage ids ``p1, p2`` while the adapter stores evidence as
        ``e1..en`` -- that intersection is empty for every question and SP-EM
        and SP-F1 come out as a structural ``0.000`` that describes the harness
        and not the system. It is a real zero in the arithmetic and a false one
        in the report, so it is flagged rather than blanked: blanking it would
        hide the defect, and printing it unflagged would attribute it to the
        system.
        """
        cited = 0
        for record in self.records:
            citations = set(record.get("citations") or ())
            if not citations:
                continue
            cited += 1
            if resolved_citations(record):
                return False
        return cited > 0

    @property
    def sp_protocol(self) -> str:
        """``"cited"`` or ``"pool"``: which supporting-fact protocol scored this run.

        ``predicted_supporting_facts`` scores the cited subset when a record's
        citations resolve to its evidence and the whole evidence pool
        otherwise. A run is ``cited`` when any of its records resolved; a run
        that never cited, or cited in an unusable namespace, is ``pool``. The
        two are different measurements -- roughly two sentences against
        seven -- and a paired test between them is not a test of the systems.
        """
        return "cited" if any(resolved_citations(r) for r in self.records) else "pool"

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

    def question_metrics(self) -> list[dict[str, Any]]:
        """The per-question ``metrics`` blocks, recomputed from the step trace.

        ``derive_question_metrics`` replaces ``llm_calls_saved`` with the
        count of calls a rule replaced *in this configuration*, ``plan_depth``
        with the selected plan's depth, and adds the decomposition and the
        recorded originals beside them. The trace itself is never rewritten.
        """
        return [
            derive_question_metrics(
                record, config=self.config_snapshot, config_name=self.config_name
            )
            for record in self.records
        ]

    def agent_metrics(self) -> AgentMetrics:
        """Aggregate of the recomputed per-question ``metrics`` blocks."""
        return summarise_agent_metrics(self.question_metrics())

    def reachable(self) -> dict[str, bool]:
        """Which deterministic rules gate an LLM call this run could have made."""
        return reachable_llm_paths(self.config_snapshot, config_name=self.config_name)

    def series(self, name: str) -> list[tuple[str, float]]:
        """``(qid, value)`` for one field of the recomputed per-question metrics.

        The per-question values, not the aggregate: a summary statistic that is
        robust to an outlier cannot be recovered from a mean that has already
        absorbed it.
        """
        out: list[tuple[str, float]] = []
        for record, metrics in zip(self.records, self.question_metrics()):
            value = metrics.get(name)
            if value is None:
                continue
            try:
                out.append((str(record.get("qid") or ""), float(value)))
            except (TypeError, ValueError):
                continue
        return out

    def median(self, name: str) -> float | None:
        """Median of one per-question agent metric, or ``None`` if never recorded."""
        values = sorted(v for _, v in self.series(name))
        if not values:
            return None
        middle = len(values) // 2
        if len(values) % 2:
            return values[middle]
        return (values[middle - 1] + values[middle]) / 2.0

    def worst(self, name: str) -> tuple[str, float] | None:
        """The ``(qid, value)`` with the largest value, or ``None``."""
        series = self.series(name)
        return max(series, key=lambda pair: pair[1]) if series else None


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
    snapshot: Any = meta.get("config")
    if not isinstance(snapshot, Mapping) or not snapshot:
        try:
            snapshot = config_for(config_name, cfg).raw
        except Exception:  # noqa: BLE001 - no snapshot means defaults, not a crash
            snapshot = None
    return RunData(
        config_name=config_name,
        dataset=dataset,
        run_id=str(meta.get("run_id") or run_dir.name),
        run_dir=run_dir,
        records=records,
        scores=scores,
        meta=meta,
        config_snapshot=snapshot,
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
# Comparability: sample size and whether the system answers at all
# ---------------------------------------------------------------------------

def eval_sample_size(dataset: str, *, cfg: Config) -> int | None:
    """The size of the frozen evaluation slice for ``dataset``, or ``None``.

    Read from ``datasets.{dataset}.eval_sample`` on every call rather than
    frozen into a constant, because it is the number the sample was drawn with
    and the number a run has to reach to be comparable. A run in flight is
    therefore judged against the design, not against whichever sibling run
    happens to sit beside it in the table.
    """
    value = cfg.get(f"datasets.{dataset}.eval_sample", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_partial(run: RunData | None, expected: int | None) -> bool:
    """Whether ``run`` covers less than the whole evaluation slice.

    Purely a comparison of counts, so a sweep that is still writing rows
    rejoins the comparison of its own accord the moment it reaches the slice
    size -- there is nothing to update by hand when it finishes.
    """
    return run is not None and expected is not None and run.n_questions < expected


def answer_reference(
    runs: Mapping[str, RunData],
    *,
    config_names: Sequence[str],
    expected: int | None,
) -> str | None:
    """The non-agentic system EM and F1 are compared against, or ``None``.

    :data:`REFERENCE_CONFIG` cannot serve: it is retrieval-only, so a
    ``$\\Delta$F1`` against it is just the other system's F1 with a plus sign
    in front, and every generative row would carry a large, significant,
    meaningless improvement. The honest reference for an answer metric is the
    strongest baseline that *answers*, so it is chosen here from the runs on
    disk -- highest mean F1 among complete, non-agentic, answering runs, ties
    broken by configuration order -- and named in the caption. Choosing it
    from the data rather than declaring it means the comparison cannot quietly
    become one against a system that was later found not to answer.
    """
    best: tuple[float, int, str] | None = None
    for index, name in enumerate(config_names):
        run = runs.get(name)
        if run is None or name in AGENTIC_CONFIGS:
            continue
        if not run.attempts_answers or _is_partial(run, expected):
            continue
        score = run.mean("f1")
        if score is None:
            continue
        candidate = (float(score), -index, name)
        if best is None or candidate > best:
            best = candidate
    return best[2] if best is not None else None


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
    reference: str | None,
    metrics: Sequence[str],
    samples: int,
    seed: int,
    expected: int | None = None,
) -> _Comparisons:
    """Paired-bootstrap every eligible configuration against ``reference``.

    Two eligibility rules, both of which suppress a comparison rather than
    weaken it, because the alternative in each case is a number that looks like
    evidence and is not:

    *Answer metrics need two answering systems.* A run that emitted no answer
    has ``em = f1 = 0.0`` for every question by construction, and pairing it
    against anything yields exactly the other system's score with a confidence
    interval that excludes zero. That is not a finding about the difference,
    it is the definition of the zero. Such a run is skipped for
    :data:`ANSWER_METRICS`, and if the *reference* is the one that does not
    answer, the whole metric is dropped and the column renders ``--``.

    *A paired test needs the same questions.* :func:`paired_bootstrap` would
    happily compare a 250-question reference with a 212-question prefix on the
    212 they share, but "the 212 that happened to finish first" is not the
    frozen slice the design specifies and the other rows were scored on, and a
    delta on it moves every time the sweep writes another line. An incomplete
    run is therefore excluded outright: its point estimates are still printed,
    its comparisons are not.
    """
    comparisons = _Comparisons(reference=reference or "", available=False)
    if reference is None:
        return comparisons
    reference_run = runs.get(reference)
    if reference_run is None or _is_partial(reference_run, expected):
        return comparisons
    comparisons.available = True
    for metric in metrics:
        if metric in ANSWER_METRICS and not reference_run.attempts_answers:
            continue
        base = reference_run.metric(metric)
        if not base:
            continue
        row: dict[str, ComparisonResult] = {}
        for config_name, run in runs.items():
            if config_name == reference or _is_partial(run, expected):
                continue
            if metric in ANSWER_METRICS and not run.attempts_answers:
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


def _significance_note(
    comparisons: _Comparisons, *, what: str = "", legend: bool = True
) -> str:
    """The legend that keeps the marks readable -- and checkable.

    ``legend=False`` names the reference without repeating the mark key, for
    the second and later reference in the same caption: a caption that states
    the same legend twice is a caption nobody finishes reading.
    """
    subject = f" for the {what}" if what else ""
    if not comparisons.available:
        return (
            f"No usable reference run is available{subject}, so no significance marks "
            "are shown there."
        )
    head = f"Reference system{subject}: {_tt(comparisons.reference)}, marked $\\dagger$."
    if not legend:
        return head
    return head + (
        r" \textbf{Bold} = significantly better than the reference, "
        r"$^{\downarrow}$ = significantly worse "
        r"(paired bootstrap, 95\% CI of the paired difference excluding zero). "
        "Unmarked differences are not significant."
    )


def _flagged(text: str, flag: str) -> str:
    r"""Append a footnote mark to a cell, unless the cell is :data:`MISSING`.

    Adjacent superscripts are merged into one group: ``$^{\downarrow}$$^{\ast}$``
    typesets as two separate scripts with a gap between them, which reads as
    two marks on two different things.
    """
    if not flag or text == MISSING:
        return text
    if text.endswith(r"}$") and flag.startswith(r"$^{") and flag.endswith(r"}$"):
        return text[:-2] + flag[3:-2] + r"}$"
    return text + flag


def _names(config_names: Sequence[str]) -> str:
    """``a``, ``a and b``, ``a, b and c`` -- a list a caption can read aloud."""
    marked = [_tt(name) for name in config_names]
    if len(marked) < 2:
        return "".join(marked)
    return ", ".join(marked[:-1]) + " and " + marked[-1]


def _unrun_note() -> str:
    return f"{_tt(MISSING)} across a whole row marks a configuration that has not been run."


def _no_answer_note(
    runs: Mapping[str, RunData], *, config_names: Sequence[str], columns: str
) -> str:
    """Say which rows are blank because the system does not answer, not because it did not run.

    Both states print the same glyph and mean opposite things, so which one a
    reader is looking at is stated rather than left to be inferred from the
    system's name.
    """
    silent = [n for n in config_names if (r := runs.get(n)) is not None and not r.attempts_answers]
    if not silent:
        return ""
    return (
        f" {_names(silent)} {'is' if len(silent) == 1 else 'are'} retrieval-only: "
        f"{'it emits' if len(silent) == 1 else 'they emit'} a ranking and no answer, so the "
        f"{columns} columns show {_tt(MISSING)} because there is no answer to score, "
        "not because the run is absent. Scoring the unattempted answer would report "
        f"{fmt(0.0)} and assert a failed attempt that never happened."
    )


def _partial_note(
    runs: Mapping[str, RunData], *, config_names: Sequence[str], expected: int | None
) -> str:
    """Say which rows are a prefix of the slice, and that they are not compared."""
    partial = [
        f"{_tt(n)} ($n={runs[n].n_questions}$)"
        for n in config_names
        if n in runs and _is_partial(runs[n], expected)
    ]
    if not partial:
        return ""
    return (
        f" \\textbf{{{PARTIAL_MARK} marks a preliminary row}}: {', '.join(partial)} "
        f"had not covered the full {expected}-question evaluation slice when this table "
        "was generated, so the point estimates are over the questions completed so far "
        "and every delta, $p$-value and significance mark against it is withheld -- a "
        "paired test between a prefix and the full slice is not a paired test. These "
        "numbers will move; they are not final results."
    )


def _artefact_note(runs: Mapping[str, RunData], *, config_names: Sequence[str]) -> str:
    """Flag supporting-fact scores that measure the harness rather than the system."""
    broken = [n for n in config_names if (r := runs.get(n)) is not None and r.citations_unresolved]
    if not broken:
        return ""
    return (
        f" \\textbf{{{ARTEFACT_MARK} marks a score measured differently, not a worse "
        f"system}}: every record of {_names(broken)} cites passage identifiers "
        r"(\texttt{p1}, \texttt{p2}) that name none of the evidence identifiers the "
        r"same record stores, because the adapter keys evidence \texttt{e1..en}. The "
        r"citation signal is therefore unusable and \texttt{predicted\_supporting\_facts} "
        "falls back to scoring the full evidence pool rather than the cited subset. "
        "That fallback favours recall and pays for it in precision, so these scores "
        "are not strictly comparable to a system whose citations resolve and which is "
        "judged on what it actually cited. Before the fallback was made conditional on "
        "citations resolving, the same mismatch produced exact zeros that read as a "
        r"categorical finding; see \texttt{docs/report-audit.md}."
    )


def _sp_protocol_note(runs: Mapping[str, RunData], *, config_names: Sequence[str]) -> str:
    """One sentence saying which rows were scored on what, and what that forbids.

    The cited-subset and whole-pool protocols are both legitimate and neither
    is the other: the first measures what a system stands behind, the second
    what it retrieved. What is not legitimate is a significance mark between
    them, so the sentence names the rows under each protocol, gives the mean
    size of the set each was scored on, and says that the marks are withheld
    across the boundary and kept on the one column that is the same protocol
    for every row.
    """
    cited = [n for n in config_names if (r := runs.get(n)) is not None and r.sp_protocol == "cited"]
    pooled = [n for n in config_names if (r := runs.get(n)) is not None and r.sp_protocol == "pool"]
    if not cited or not pooled:
        return ""

    def mean_size(names: Sequence[str]) -> str:
        values = [v for n in names for v in runs[n].metric("sp_n_predicted").values()]
        return fmt(sum(values) / len(values), 1) if values else MISSING

    return (
        r"\textbf{SP-EM, SP-P, SP-R and SP-F1 are not one protocol}: "
        f"{_names(cited)} {'is' if len(cited) == 1 else 'are'} scored on the sentences "
        f"{'it' if len(cited) == 1 else 'they'} actually cited (mean {mean_size(cited)} "
        f"sentences per question), whereas {_names(pooled)} "
        f"{'is' if len(pooled) == 1 else 'are'} scored on {'its' if len(pooled) == 1 else 'their'} "
        f"whole evidence pool (mean {mean_size(pooled)}) because "
        f"{'it emits' if len(pooled) == 1 else 'they emit'} no citation that resolves to "
        "that evidence, so a single SP-F1 is not comparable across the two groups and no "
        "significance mark is placed between rows of different protocol; "
        r"\textit{Pool SP-R} is the recall of the whole evidence pool for every row alike "
        "and is the one supporting-fact column compared like for like."
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
    expected = eval_sample_size(dataset, cfg=cfg)
    answer_ref = answer_reference(runs, config_names=config_names, expected=expected)
    answer_comparisons = _compare_all(
        runs,
        reference=answer_ref,
        metrics=("em", "f1"),
        samples=samples,
        seed=seed,
        expected=expected,
    )
    support_comparisons = _compare_all(
        runs,
        reference=REFERENCE_CONFIG,
        metrics=(
            "sp_f1", "sp_precision", "sp_recall", "sp_pool_recall",
            "recall@10", "ndcg@10",
        ),
        samples=samples,
        seed=seed,
        expected=expected,
    )
    answer_note = _significance_note(answer_comparisons, what="EM and F1 columns")
    support_note = _significance_note(
        support_comparisons, what="supporting-fact columns", legend=False
    )
    retrieval_note = _significance_note(support_comparisons, what="retrieval columns")
    partial_note = _partial_note(runs, config_names=config_names, expected=expected)
    artefact_note = _artefact_note(runs, config_names=config_names)
    protocol_note = _sp_protocol_note(runs, config_names=config_names)
    reference_run = runs.get(REFERENCE_CONFIG)

    def sp_mark(run: RunData | None, name: str, metric: str, *, like_for_like: bool) -> str:
        """A supporting-fact cell, marked only against a like-for-like reference.

        The bootstrap will happily pair a cited-subset score with a whole-pool
        score; the interval it returns is then a statement about the two
        protocols, not about the two systems. Marks are withheld whenever the
        row and the reference were scored under different protocols, and kept
        for the one column (``sp_pool_recall``) that is the same protocol for
        every row.
        """
        comparison = support_comparisons.get(name, metric)
        if (
            not like_for_like
            and run is not None
            and reference_run is not None
            and run.sp_protocol != reference_run.sp_protocol
        ):
            comparison = None
        return mark(
            run.mean(metric) if run else None, comparison, reference=(name == REFERENCE_CONFIG)
        )

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
            answers = run is not None and run.attempts_answers
            partial = _is_partial(run, expected)
            artefact = run is not None and run.citations_unresolved
            is_answer_ref = answers and name == answer_ref
            n_cell = MISSING if run is None else fmt_int(run.n_questions)
            if partial:
                n_cell = _flagged(n_cell, PARTIAL_MARK)
            flag = ARTEFACT_MARK if artefact else ""
            rows.append([
                _tt(name),
                n_cell,
                mark(run.mean("em") if answers else None,
                     answer_comparisons.get(name, "em"), reference=is_answer_ref),
                mark(run.mean("f1") if answers else None,
                     answer_comparisons.get(name, "f1"), reference=is_answer_ref),
                fmt_ci(interval(run, _CI_METRIC)) if answers else MISSING,
                _flagged(fmt(run.mean("sp_em") if run else None), flag),
                _flagged(sp_mark(run, name, "sp_precision", like_for_like=False), flag),
                _flagged(sp_mark(run, name, "sp_recall", like_for_like=False), flag),
                _flagged(sp_mark(run, name, "sp_f1", like_for_like=False), flag),
                sp_mark(run, name, "sp_pool_recall", like_for_like=True),
                MISSING if is_answer_ref else fmt_delta(answer_comparisons.get(name, "f1")),
            ])
        answer_groups.append((title, rows))

    answer = table(
        colspec="lrrrcrrrrrr",
        header=_header(
            "System", "$n$", "EM", "F1", r"F1 95\% CI", "SP-EM", "SP-P", "SP-R", "SP-F1",
            "Pool SP-R", r"$\Delta$F1 vs.\ ref.\ [95\% CI]",
        ),
        groups=answer_groups,
        caption=(
            f"Answer quality on {_tt(dataset)}. Exact match and token-level F1 follow the "
            "official HotpotQA evaluation script, including its yes/no short circuit; "
            r"SP-EM, SP-P, SP-R and SP-F1 are set metrics over "
            r"$(\textit{title}, \textit{sent\_id})$ supporting-fact pairs. "
            f"$n$ is the number of questions scored. "
            f"{answer_note} {support_note} {protocol_note} {_unrun_note()}"
            + _no_answer_note(runs, config_names=config_names, columns="EM, F1 and $\\Delta$F1")
            + partial_note
            + artefact_note
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
                _flagged(_tt(name), PARTIAL_MARK if _is_partial(run, expected) else ""),
                fmt(run.mean("recall@2") if run else None),
                fmt(run.mean("recall@5") if run else None),
                mark(run.mean("recall@10") if run else None,
                     support_comparisons.get(name, "recall@10"), reference=is_ref),
                mark(run.mean("ndcg@10") if run else None,
                     support_comparisons.get(name, "ndcg@10"), reference=is_ref),
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
            f"score calibration between BM25 and cosine. {retrieval_note} {_unrun_note()}"
            + partial_note
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
    cfg: Config,
    config_names: Sequence[str],
) -> str:
    """Agent-specific cost, reported beside quality and never instead of it.

    A system that gains three F1 points for twenty times the compute has not
    obviously won, and this is the table that lets a reader say so. Runs whose
    ``meta.json`` reports a warm cache are called out in the caption, because a
    warm-cache latency is not comparable to a cold-cache one and the difference
    is invisible in the number itself.

    **Latency is reported as a median beside its maximum, never as a bare
    mean.** Every other column here is a small bounded count -- the orchestrator
    caps LLM calls, sub-queries and cycles per question -- so one pathological
    question can move their means by at most a cap divided by $n$. Latency has
    no such cap in practice: one HotpotQA question hung inside a single blocking
    call for 18.3 hours against a 300-second budget, which alone lifts the
    ``agentic_full`` mean from a median of 28.3 s to 295.1 s. Printed as
    "latency per question" that mean is wrong by an order of magnitude in the
    direction that makes the agentic system look uncompetitive, and it is
    exactly the number a reader quotes. The median is what a question actually
    costs; the maximum column and the caption keep the hang visible, because
    trimming it away would hide a real defect rather than report it.
    """
    expected = eval_sample_size(dataset, cfg=cfg)
    budget = cfg.get("orchestrator.max_wall_clock_s", None)
    groups: list[tuple[str | None, Sequence[Sequence[str]]]] = []
    warm: list[str] = []
    hangs: list[str] = []
    saved_breakdown: list[str] = []
    depth_notes: list[str] = []
    discard_notes: list[str] = []
    for title, names in _grouped(config_names):
        rows: list[list[str]] = []
        for name in names:
            run = runs.get(name)
            if run is None:
                rows.append([_tt(name)] + [MISSING] * 10)
                continue
            if not bool(run.meta.get("cache_cold", True)):
                warm.append(name)
            m = run.agent_metrics()
            worst = run.worst("latency_s")
            if worst is not None and budget is not None and worst[1] > float(budget):
                hangs.append(
                    f"{_tt(name)} ({fmt(worst[1], 1)}\\,s on "
                    f"{_tt(worst[0])}, {_hours(worst[1])})"
                )
            if name in AGENTIC_CONFIGS:
                saved_breakdown.append(_saved_breakdown(name, run, m))
                if abs(m.plan_depth - m.plan_depth_latest) >= 0.0005:
                    depth_notes.append(
                        f"{_tt(name)} {fmt(m.plan_depth, 2)} selected against "
                        f"{fmt(m.plan_depth_latest, 2)} latest"
                    )
                discarded = sum(
                    1 for q in run.question_metrics() if float(q.get("replans_discarded", 0) or 0) > 0
                )
                if discarded:
                    discard_notes.append(
                        f"{_tt(name)} ({discarded} of {run.n_questions} questions, "
                        f"executed rate {fmt(m.replan_executed_rate, 3)})"
                    )
            rows.append([
                _flagged(_tt(name), PARTIAL_MARK if _is_partial(run, expected) else ""),
                fmt(m.llm_calls, 2),
                fmt(m.llm_calls_saved, 2),
                fmt(m.tool_calls, 2),
                fmt(run.median("latency_s"), 1),
                fmt(worst[1] if worst else None, 1),
                fmt(m.plan_depth, 2),
                fmt(m.n_subqueries, 2),
                fmt(m.replan_rate, 3),
                fmt(m.replan_executed_rate, 3),
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
    hang_note = ""
    if hangs:
        hang_note = (
            f" \\textbf{{One or more questions overran the "
            f"{fmt_int(budget)}\\,s \\texttt{{orchestrator.max\\_wall\\_clock\\_s}} "
            f"budget}}: {', '.join(hangs)}. The budget is evidently only tested between "
            "state transitions, so it cannot pre-empt a call that blocks inside one; "
            "this is a system defect and not only a reporting one, and it is why the "
            r"central column is a median. The arithmetic mean of \texttt{latency\_s} on "
            "such a run is that single question divided by $n$ and describes nothing "
            r"that happens per question. See \texttt{docs/report-audit.md}."
        )
    saved_note = ""
    if saved_breakdown:
        saved_note = (
            r" \textbf{\textit{Saved} is recomputed from the step trace under one "
            r"definition}: a call is saved only when a deterministic rule produced an "
            "output that, absent the rule, this configuration would have obtained by an "
            "LLM call. The agents' own counter credits every rule that fired; two of "
            r"those rules stand in for calls this configuration cannot make -- with "
            r"\texttt{heuristic\_shortcut: true} the LLM router is unreachable for every "
            r"sub-query, and with \texttt{entity\_linker: alias\_match} there is no LLM "
            "entity linker to skip -- and the identity plan of the no-planner ablation is "
            "the ablation itself, not a rule gating a planner call. A verifier skip for "
            "an empty or already-insufficient answer is a call nobody would make, not one "
            "replaced. The extraction ladder, which replaces a real rung-4 LLM call on "
            "every rung-1 to rung-3 success, never incremented the counter at all. "
            "Per-question decomposition, counted terms in bold, so any other definition "
            "can be applied from the same numbers: " + "; ".join(saved_breakdown) + "."
        )
    depth_note = (
        r" \textit{Plan depth} is the depth of the \textit{selected} plan (the cycle "
        r"FINALIZE chose, \texttt{best\_cycle}), as \texttt{docs/architecture.md} "
        r"\S6 defines it; the orchestrator's own metrics block records the "
        r"\textit{latest} plan's depth and the two differ wherever a re-plan executed "
        "and the first cycle's answer was kept"
        + (f" ({', '.join(depth_notes)})" if depth_notes else "")
        + "."
    )
    replan_note = (
        r" \textit{Re-plan rate} is the fraction of questions on which the verifier "
        r"triggered at least one re-plan (\textit{trig.}) and the fraction on which a "
        r"re-planned cycle actually executed (\textit{exec.}); the two differ when the "
        "planner's re-plan was a near-duplicate of a plan already run and transition T2b "
        "discarded it, which the trace counts as a re-plan but not as a cycle"
        + (
            ". Questions with a discarded re-plan: " + ", ".join(discard_notes)
            if discard_notes
            else ""
        )
        + "."
    )
    return table(
        colspec="lrrrrrrrrrr",
        header=_header(
            "System", "LLM calls", "Saved", "Tool calls", "Latency (s) med.",
            "Latency (s) max", "Plan depth", "Sub-queries", "Re-plan rate (trig.)",
            "Re-plan rate (exec.)", "Cite grounding",
        ),
        groups=groups,
        caption=(
            f"Agent-specific cost on {_tt(dataset)}, per question. "
            r"\textit{LLM calls} counts logical calls, cache hits included; "
            r"\textit{Saved} counts calls a deterministic rule replaced, under the "
            "definition stated below; "
            r"\textit{Re-plan rate} is a fraction of questions, not a mean count; "
            r"\textit{Cite grounding} is averaged only "
            "over questions that produced a non-empty answer, so it cannot be inflated by "
            r"abstentions. \textbf{Latency is the median per question, with the maximum "
            r"beside it}, and every other column is a mean: latency is the one unbounded "
            "quantity here, so it is the one a single hung question can dominate, and a "
            "mean of it would report that question rather than the system. The maximum is "
            "printed rather than trimmed so the outlier stays in view. Latency is only "
            "comparable between cold-cache runs."
            + caveat
            + hang_note
            + saved_note
            + depth_note
            + replan_note
            + f" {_unrun_note()}"
            + _partial_note(runs, config_names=config_names, expected=expected)
        ),
        label=f"tab:agent-metrics-{dataset}",
    )


def _saved_breakdown(name: str, run: RunData, m: AgentMetrics) -> str:
    """``agentic_full: routing 3.31, KG alias 2.54, verifier gate 1.21, ...``.

    Every term the agents credited, per question, with the ones that count
    under this configuration in bold and the agents' own total in brackets.
    A reader who wants to count routing as a save can add it back; a reader
    who wants to drop extraction can take it out. Nothing is hidden in the
    single number.
    """
    reachable = run.reachable()
    terms: list[tuple[str, float, bool]] = [
        ("routing", m.saved_routing, reachable["router"]),
        ("KG alias", m.saved_kg_alias, reachable["kg_linker"]),
        ("verifier gate", m.saved_verifier_gate, reachable["verifier_llm"]),
        ("verifier pointless", m.saved_verifier_pointless, False),
        ("extraction", m.saved_extraction, reachable["extractor_llm"]),
        ("planner template", m.saved_planner_template, reachable["planner_llm"]),
        ("planner ablation", m.saved_planner_ablation, False),
    ]
    parts = []
    for label, value, counted in terms:
        if value <= 0:
            continue
        cell = f"{label} {fmt(value, 2)}"
        parts.append(r"\textbf{" + cell + "}" if counted else cell)
    body = ", ".join(parts) if parts else "none"
    return f"{_tt(name)}: {body} (agents recorded {fmt(m.llm_calls_saved_recorded, 2)})"


# ---------------------------------------------------------------------------
# ablations.tex
# ---------------------------------------------------------------------------

def ablations_table(
    runs: Mapping[str, RunData],
    dataset: str,
    *,
    cfg: Config,
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
    expected = eval_sample_size(dataset, cfg=cfg)
    comparisons = _compare_all(
        runs, reference=ABLATION_REFERENCE, metrics=("em", "f1"),
        samples=samples, seed=seed, expected=expected,
    )

    ordered: list[str] = []
    for name in (ABLATION_REFERENCE, *AGENTIC_CONFIGS, REFERENCE_CONFIG):
        if name not in ordered:
            ordered.append(name)

    rows: list[list[str]] = []
    for name in ordered:
        run = runs.get(name)
        is_ref = name == ABLATION_REFERENCE
        answers = run is not None and run.attempts_answers
        partial = _is_partial(run, expected)
        agent = run.agent_metrics() if run is not None else None
        label = _tt(name)
        if name == REFERENCE_CONFIG:
            label += r" \textit{(non-agentic floor)}"
        n_cell = MISSING if run is None else fmt_int(run.n_questions)
        if partial:
            n_cell = _flagged(n_cell, PARTIAL_MARK)
        rows.append([
            label,
            n_cell,
            mark(run.mean("em") if answers else None,
                 comparisons.get(name, "em"), reference=is_ref),
            mark(run.mean("f1") if answers else None,
                 comparisons.get(name, "f1"), reference=is_ref),
            MISSING if is_ref else fmt_delta(comparisons.get(name, "f1")),
            MISSING if is_ref else fmt_p(comparisons.get(name, "f1")),
            fmt(agent.llm_calls, 2) if agent else MISSING,
            # Median, matching tab:agent-metrics: the mean of an unbounded
            # per-question duration is one hung question divided by n.
            fmt(run.median("latency_s") if run is not None else None, 1),
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
    # The subsample sentence Chapter 2 promises the reader will find here. It
    # is built from the runs rather than from the plan, so a subsample that was
    # designed and then not used cannot be reported as though it had been.
    present = [(name, runs[name].n_questions) for name in ordered if name in runs]
    if present:
        def stated_size(name: str, n: int) -> str:
            # One math group, not two: ``$n=123$$^{\ddagger}$`` typesets a gap
            # between the count and the mark that belongs to it.
            mark = r"^{\ddagger}" if expected is not None and n < expected else ""
            return f"{_tt(name)} $n={n}{mark}$"

        stated = ", ".join(stated_size(name, n) for name, n in present)
        sizes = {n for _, n in present}
        if expected is not None and sizes == {expected}:
            sample_note = (
                f" Every row covers the same frozen {expected}-question evaluation slice "
                f"as the main results -- no ablation was run on a reduced subsample "
                f"({stated})."
            )
        else:
            sample_note = (
                f" Sample sizes are stated per row and are not assumed equal: {stated}"
                + (f", against a {expected}-question evaluation slice." if expected else ".")
            )
    else:
        sample_note = ""

    return table(
        colspec="lrrrrrrr",
        header=_header(
            "System", "$n$", "EM", "F1", r"$\Delta$F1 vs.\ full [95\% CI]", "$p$",
            "LLM calls", "Latency (s) med.",
        ),
        groups=[(None, rows)],
        caption=(
            f"Ablation study on {_tt(dataset)}. {note}"
            + sample_note
            + f" {_tt(REFERENCE_CONFIG)} is repeated from the main results as the "
            "non-agentic floor, since removing planning, synthesis and verification "
            "reduces the pipeline to it. Latency is the median per question, matching "
            r"Table~\ref{tab:agent-metrics-" + dataset + "}; the maximum and the "
            "wall-clock overruns are reported there. "
            + _unrun_note()
            + _no_answer_note(
                runs, config_names=ordered, columns="EM, F1, $\\Delta$F1 and $p$"
            )
            + _partial_note(runs, config_names=ordered, expected=expected)
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
    expected = eval_sample_size(dataset, cfg=cfg)
    silent = [c for c in present if not runs[c].attempts_answers]
    no_answer_note = (
        ""
        if not silent
        else (
            f" {_names(silent)} emit no answer at all, so their "
            r"\textit{Correct} is structurally $0$ and every question is counted as an "
            "error; the informative rows for them are the retrieval labels, not the "
            "accuracy."
        )
    )
    return table(
        colspec="l" + "r" * len(columns),
        header=_header(
            "Label",
            *[
                _flagged(_tt(c), PARTIAL_MARK if _is_partial(runs.get(c), expected) else "")
                if c
                else MISSING
                for c in columns
            ],
        ),
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
            + no_answer_note
            + _partial_note(runs, config_names=present, expected=expected)
        ),
        label=f"tab:error-analysis-{dataset}",
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _preamble(
    title: str,
    runs: Mapping[str, Mapping[str, RunData]],
    expected: Mapping[str, int | None] | None = None,
) -> str:
    """A provenance header naming the runs the numbers came from.

    Deliberately carries no wall-clock timestamp: regenerating from unchanged
    runs must produce a byte-identical file, so that a diff under
    ``results/tables/`` always means a number actually moved.

    A source line whose ``n`` is short of the evaluation slice is annotated
    ``PRELIMINARY`` here as well as in the caption. The header is where a
    reader -- or ``tests/test_report_integrity.py`` -- looks to find out which
    run a number came from, so it is the one place where a partial run must not
    look like a finished one.
    """
    expected = expected or {}
    lines = [
        f"% {title}",
        "% GENERATED by src/agentic_ir/eval/tables.py -- do not edit by hand.",
        "% Regenerate with: python -m agentic_ir.cli tables",
        "% Requires \\usepackage{booktabs} and \\usepackage{graphicx};",
        "% report/main.tex already loads both.",
    ]

    def source_line(dataset: str, config: str, run: RunData) -> str:
        line = f"%   {dataset}/{config}: {run.run_id} ({run.n_questions} questions)"
        want = expected.get(dataset)
        if _is_partial(run, want):
            line += (
                f"  PRELIMINARY: short of the {want}-question evaluation slice; "
                "deltas and significance withheld"
            )
        return line

    sources = [
        source_line(dataset, config, run)
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

    expected = {d: eval_sample_size(d, cfg=cfg) for d in datasets}
    written: dict[str, Path] = {}

    def write(filename: str, title: str, body: str) -> None:
        path = target / filename
        path.write_text(_preamble(title, runs, expected) + body + "\n", encoding="utf-8")
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
            agent_metrics_table(runs[d], d, cfg=cfg, config_names=config_names)
            for d in datasets
        ),
    )
    write(
        "ablations.tex",
        "Chapter 4 -- ablation study",
        "\n\n".join(
            ablations_table(runs[d], d, cfg=cfg, samples=samples, seed=seed)
            for d in datasets
        ),
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
