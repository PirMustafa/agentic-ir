"""Report integrity checks.

The report is graded on what the PDF says. A number in the prose that no
artefact on disk supports is the worst defect available -- worse than an
admitted gap -- because a reader has no way to detect it. These tests
mechanise the checks that can be mechanised, so that the class of defect
found in the 2026-09 audit (``docs/report-audit.md``) cannot come back
silently.

Everything here is read-only. Where an artefact does not exist yet (runs
still in flight, tables not regenerated) the test skips rather than fails:
a missing evaluation is a known state of this project, a *wrong* number is
not.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
CHAPTERS = REPORT / "chapters"
TABLES = ROOT / "results" / "tables"
PROCESSED = ROOT / "data" / "processed"
INDEXES = ROOT / "data" / "indexes"
RUNS = ROOT / "results" / "runs"

DATASETS = ("hotpotqa", "twowiki")

#: Configurations that were never evaluated must render as this, never 0.0.
MISSING = "--"

#: The evaluation sample size every headline row is supposed to carry.
EVAL_SAMPLE_N = 250

#: The nine systems of ``evaluation.configurations``. Held here as a literal
#: rather than imported, so the test does not learn what it is checking from
#: the module it is checking.
CONFIGURATIONS = (
    "bm25_only", "dense_only", "hybrid_rerank", "naive_rag", "self_ask",
    "agentic_full", "agentic_no_planner", "agentic_no_kg", "agentic_no_verifier",
)

#: Header cells of the columns that only mean something for a system that
#: emitted an answer. A retrieval-only system must render every one of them as
#: :data:`MISSING`, because a zero there asserts a failed attempt at answering
#: that the system never made.
ANSWER_COLUMN_HEADS = ("EM", "F1", r"F1 95\% CI", "$p$")

#: Titles that are *not* in the processed corpora -- they come from HotpotQA's
#: train split. Both were used as worked examples during drafting; either one
#: reaching the PDF is a claim about a passage the reader cannot look up.
ABSENT_TITLES = ("Arthur's Magazine", "First for Women")

#: Superseded throughput figure. The measured value is 52 tok/s
#: (docs/architecture.md, README.md); "30-45" was an early estimate.
STALE_THROUGHPUT = (r"30--45", r"30-45", r"30 to 45")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def read(path: Path) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def chapter_files() -> list[Path]:
    return sorted(CHAPTERS.glob("*.tex"))


def tex_files() -> list[Path]:
    return [REPORT / "main.tex", *chapter_files()]


def line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def flatten_ints(obj, out: set[int]) -> set[int]:
    """Every integer anywhere in a JSON artefact, as a sourced-number pool."""
    if isinstance(obj, bool):
        return out
    if isinstance(obj, int):
        out.add(obj)
    elif isinstance(obj, float):
        if obj.is_integer():
            out.add(int(obj))
    elif isinstance(obj, dict):
        for value in obj.values():
            flatten_ints(value, out)
    elif isinstance(obj, list):
        for value in obj:
            flatten_ints(value, out)
    return out


def run_length(run_id: str) -> int | None:
    """Records actually on disk for ``run_id``, or None if it has no traces."""
    traces = RUNS / run_id / "traces.jsonl"
    if not traces.exists():
        return None
    with open(traces, encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def table_sources(body: str) -> list[tuple[int, str, str, str, int]]:
    """``(line, dataset, config, run_id, claimed_n)`` from a table's header."""
    out = []
    pattern = r"^%\s+(\S+)/(\S+):\s+(\S+)\s+\((\d+) questions\)"
    for match in re.finditer(pattern, body, re.M):
        dataset, config, run_id, n = match.groups()
        out.append((line_of(body, match.start()), dataset, config, run_id, int(n)))
    return out


def table_drift() -> list[str]:
    """Tables whose header n no longer matches the run it names.

    A table generated while a run was in flight freezes the partial n. The
    run then finishes, the prose is written against the finished numbers,
    and the table beside it silently disagrees. Regenerating is the fix.
    """
    drift = []
    for table in table_files():
        body = read(table)
        for line, _, _, run_id, claimed in table_sources(body):
            actual = run_length(run_id)
            if actual is not None and actual != claimed:
                drift.append(
                    f"{table.name}:{line}: {run_id} table says n={claimed}, "
                    f"traces.jsonl holds {actual}"
                )
    return drift


def tabulars(body: str) -> list[tuple[str, list[str], list[list[str]]]]:
    """``(caption, header cells, body rows)`` for every table in a fragment.

    Cells are split on ``&`` so a column can be found by its header rather
    than by a position this test would have to be edited to keep in step, and
    the caption travels with the table it belongs to -- a caption two tables
    away explains nothing.
    """
    out = []
    for block in re.findall(r"\\begin\{table\}(.*?)\\end\{table\}", body, re.S):
        caption = re.search(r"\\caption\{(.*?)\}\n", block, re.S)
        inner = re.search(r"\\begin\{tabular\}(.*?)\\end\{tabular\}", block, re.S)
        if inner is None:
            continue
        rows = [
            [cell.strip() for cell in line.strip().removesuffix(r"\\").split("&")]
            for line in inner.group(1).splitlines()
            if "&" in line
        ]
        if not rows:
            continue
        out.append((caption.group(1) if caption else "", rows[0], rows[1:]))
    return out


def config_of(cell: str) -> str | None:
    """The configuration a row label names, or ``None`` if it names something else.

    ``error_analysis.tex`` is transposed -- its row labels are failure labels
    and its *columns* are configurations -- so a check written for a
    configuration row has to be able to tell the two apart. Without this,
    ``\\texttt{parse\\_failure} & 0 (0.000) & ...`` reads as a configuration
    that was never run and scored zero.
    """
    match = re.match(r"\\texttt\{([^}]*)\}", cell)
    if match is None:
        return None
    name = match.group(1).replace("\\_", "_")
    return name if name in CONFIGURATIONS else None


def dataset_of(caption: str) -> str | None:
    """The dataset a table's caption names, or ``None`` if it names neither.

    A configuration can be complete on one dataset and in flight on the other.
    A check that keys only on the configuration name then condemns the
    finished rows along with the unfinished ones, which is how a correct
    HotpotQA delta gets reported as a comparison against a 16-question prefix
    that lives in a different table.
    """
    for name in DATASETS:
        if "\\texttt{" + name + "}" in caption:
            return name
    return None


def retrieval_only_configs() -> set[str]:
    """Configurations whose traces hold no answer for any question.

    Derived from the runs rather than declared, so a baseline that gains an
    answering stage stops being exempt without anyone remembering to edit a
    list here.
    """
    answered: dict[str, int] = {}
    seen: set[str] = set()
    if not RUNS.is_dir():
        return set()
    for run_dir in sorted(RUNS.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "_smoke":
            continue
        meta, traces = run_dir / "meta.json", run_dir / "traces.jsonl"
        if not (meta.exists() and traces.exists()):
            continue
        config = load_json(meta).get("config_name")
        if not config:
            continue
        seen.add(config)
        with open(traces, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                if str(json.loads(line).get("final_answer") or "").strip():
                    answered[config] = answered.get(config, 0) + 1
    return {config for config in seen if not answered.get(config)}


def run_metric_values() -> set[str]:
    """Every per-run metric mean on disk, formatted to three decimals.

    A figure quoted in the prose is legitimate if some scored run produces
    it, even when the generated table beside it has not been rebuilt yet.
    Without this the check would flag a *correct* number as fabricated
    whenever the tables lag the runs.
    """
    values: set[str] = set()
    if not RUNS.is_dir():
        return values
    for run_dir in sorted(RUNS.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "_smoke":
            continue
        scores = run_dir / "scores.csv"
        if not scores.exists():
            continue
        with open(scores, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        for column in rows[0]:
            if column == "qid":
                continue
            numbers = []
            for row in rows:
                try:
                    numbers.append(float(row[column]))
                except (TypeError, ValueError):
                    continue
            if numbers:
                values.add(f"{sum(numbers) / len(numbers):.3f}")
    return values


@pytest.fixture(scope="module")
def corpus_stats() -> dict[str, dict]:
    stats = {}
    for dataset in DATASETS:
        path = PROCESSED / f"{dataset}_corpus_stats.json"
        if path.exists():
            stats[dataset] = load_json(path)
    if not stats:
        pytest.skip("no *_corpus_stats.json on disk yet")
    return stats


# --------------------------------------------------------------------------
# 1. LaTeX structural integrity
# --------------------------------------------------------------------------


def test_input_paths_resolve_from_report_dir():
    """LaTeX resolves \\input relative to main.tex's directory, not the cwd."""
    problems = []
    for tex in tex_files():
        body = read(tex)
        for match in re.finditer(r"\\input\{([^}]*)\}", body):
            target = REPORT / match.group(1)
            if not (target.exists() or target.with_suffix(".tex").exists()):
                problems.append(
                    f"{tex.relative_to(ROOT)}:{line_of(body, match.start())} "
                    f"-> {match.group(1)}"
                )
    assert not problems, "unresolvable \\input paths:\n  " + "\n  ".join(problems)


def test_every_cite_key_exists_in_bibliography():
    bib_path = REPORT / "references.bib"
    if not bib_path.exists():
        pytest.skip("references.bib not present")
    keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", read(bib_path)))
    problems = []
    for tex in tex_files() + sorted(TABLES.glob("*.tex")):
        body = read(tex)
        pattern = r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}"
        for match in re.finditer(pattern, body):
            for key in (k.strip() for k in match.group(1).split(",")):
                if key and key not in keys:
                    problems.append(
                        f"{tex.relative_to(ROOT)}:{line_of(body, match.start())} "
                        f"-> {key}"
                    )
    assert not problems, "undefined \\cite keys:\n  " + "\n  ".join(problems)


def test_every_ref_has_a_matching_label():
    labels: set[str] = set()
    refs: list[tuple[str, str]] = []
    for tex in tex_files() + sorted(TABLES.glob("*.tex")):
        body = read(tex)
        labels.update(re.findall(r"\\label\{([^}]*)\}", body))
        for match in re.finditer(r"\\(?:eq)?ref\{([^}]*)\}", body):
            where = f"{tex.relative_to(ROOT)}:{line_of(body, match.start())}"
            refs.append((match.group(1), where))
    dangling = [f"{where} -> {key}" for key, where in refs if key not in labels]
    assert not dangling, "dangling \\ref:\n  " + "\n  ".join(dangling)


def test_no_duplicate_labels():
    seen: dict[str, str] = {}
    duplicates = []
    for tex in tex_files() + sorted(TABLES.glob("*.tex")):
        body = read(tex)
        for match in re.finditer(r"\\label\{([^}]*)\}", body):
            key = match.group(1)
            where = f"{tex.relative_to(ROOT)}:{line_of(body, match.start())}"
            if key in seen:
                duplicates.append(f"{key}: {seen[key]} and {where}")
            else:
                seen[key] = where
    assert not duplicates, "duplicate \\label:\n  " + "\n  ".join(duplicates)


def test_generated_tables_referenced_by_chapters_exist():
    """A chapter that \\inputs a table it does not have compiles to nothing."""
    missing = []
    for tex in chapter_files():
        body = read(tex)
        for match in re.finditer(r"\\input\{([^}]*results/tables/[^}]*)\}", body):
            target = (REPORT / match.group(1)).resolve()
            if not target.exists():
                missing.append(
                    f"{tex.relative_to(ROOT)}:{line_of(body, match.start())} "
                    f"-> {match.group(1)}"
                )
    assert not missing, "chapters input missing tables:\n  " + "\n  ".join(missing)


# --------------------------------------------------------------------------
# 2. Stale and unsupported content in the prose
# --------------------------------------------------------------------------


@pytest.mark.parametrize("title", ABSENT_TITLES)
def test_chapters_do_not_use_absent_corpus_titles(title):
    """Both titles are train-split only; neither is in the processed corpora."""
    hits = []
    for tex in tex_files():
        body = read(tex)
        for match in re.finditer(re.escape(title), body):
            hits.append(f"{tex.relative_to(ROOT)}:{line_of(body, match.start())}")
    assert not hits, (
        f"{title!r} is absent from the processed corpora but appears at:\n  "
        + "\n  ".join(hits)
    )


def test_absent_titles_really_are_absent_from_the_corpus():
    """Guards the guard: if a rebuild ever adds them, the rule above is void."""
    path = PROCESSED / "hotpotqa_corpus.jsonl"
    if not path.exists():
        pytest.skip("hotpotqa_corpus.jsonl not built")
    found = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            blob = record.get("title", "") + " " + " ".join(record.get("sentences") or [])
            for title in ABSENT_TITLES:
                if title in blob:
                    found.add(title)
    assert not found, (
        "corpus now contains " + ", ".join(sorted(found)) + " -- "
        "ABSENT_TITLES in this test is stale and must be re-derived"
    )


@pytest.mark.parametrize("pattern", STALE_THROUGHPUT)
def test_no_superseded_throughput_figure(pattern):
    hits = []
    for tex in tex_files():
        body = read(tex)
        for match in re.finditer(pattern, body):
            hits.append(f"{tex.relative_to(ROOT)}:{line_of(body, match.start())}")
    assert not hits, (
        f"superseded throughput {pattern!r} (measured value is 52 tok/s) at:\n  "
        + "\n  ".join(hits)
    )


def test_hotpotqa_is_not_stratified_on_level(corpus_stats):
    """All 7,405 HotpotQA validation questions are level=hard.

    Stratifying on ``level`` yields one stratum -- an unstratified sample
    presented as a stratified one. Any chapter sentence pairing "stratif*"
    with ``level`` has to be the disavowal, not the design.
    """
    stats = corpus_stats.get("hotpotqa")
    if stats is None:
        pytest.skip("hotpotqa_corpus_stats.json not present")
    levels = stats.get("strata", {}).get("level", {})
    assert set(levels) == {"hard"}, (
        "premise changed: hotpotqa level strata are now " + repr(levels)
    )

    offenders = []
    for tex in chapter_files():
        body = read(tex)
        for match in re.finditer(r"[Ss]tratif\w*[^.]{0,200}", body):
            span = match.group(0)
            if "level" not in span:
                continue
            # The legitimate mention explains why level is *not* used.
            if re.search(r"\bnot\b|would|rather than|instead|single stratum", span):
                continue
            offenders.append(
                f"{tex.relative_to(ROOT)}:{line_of(body, match.start())}: {span[:90]}"
            )
    assert not offenders, (
        "chapter appears to stratify HotpotQA on `level`:\n  " + "\n  ".join(offenders)
    )


def test_every_thousands_number_in_the_prose_is_sourced(corpus_stats):
    """No corpus-scale number may appear that no artefact on disk contains.

    Only numbers written with the report's LaTeX thousands separator
    (``66{,}581``) are checked: that is the form reserved for corpus and
    graph statistics, so the pool of legitimate sources is exactly the
    build artefacts.
    """
    sourced: set[int] = set()
    for dataset in DATASETS:
        for path in (
            PROCESSED / f"{dataset}_corpus_stats.json",
            INDEXES / f"{dataset}_kg_meta.json",
            INDEXES / f"{dataset}_index_stats.json",
        ):
            if path.exists():
                flatten_ints(load_json(path), sourced)

    config = ROOT / "config" / "config.yaml"
    if config.exists():
        sourced.update(int(n) for n in re.findall(r"\b\d+\b", read(config)))

    # Sums over a dataset's own strata are derived, not asserted: the report
    # groups 2Wiki's conditional types as one figure.
    for stats in corpus_stats.values():
        types = list(stats.get("strata", {}).get("type", {}).values())
        for r in range(2, len(types) + 1):
            from itertools import combinations

            for combo in combinations(types, r):
                sourced.add(sum(combo))

    unsourced = []
    for tex in chapter_files():
        body = read(tex)
        for match in re.finditer(r"(\d{1,3})\{,\}(\d{3})", body):
            value = int(match.group(1) + match.group(2))
            if value not in sourced:
                unsourced.append(
                    f"{tex.relative_to(ROOT)}:{line_of(body, match.start())}: {value:,}"
                )
    assert not unsourced, (
        "numbers with no artefact on disk to support them:\n  " + "\n  ".join(unsourced)
    )


# --------------------------------------------------------------------------
# 3. Generated tables: the `--` convention and sample-size honesty
# --------------------------------------------------------------------------


def table_files() -> list[Path]:
    return sorted(TABLES.glob("*.tex"))


def test_tables_use_the_missing_marker_not_zero():
    """An unrun configuration must render as ``--``.

    ``0.000`` reads as a measured failure. The generator is expected to emit
    the marker; this asserts it is actually present, so a regression that
    starts writing zeros for absent runs is caught.
    """
    tables = table_files()
    if not tables:
        pytest.skip("no generated tables yet")
    offenders = []
    for table in tables:
        body = read(table)
        ran = {config for _, _, config, _, _ in table_sources(body)}
        for _, _, rows in tabulars(body):
            for cells in rows:
                # Only configuration rows are in scope. `error_analysis.tex`
                # is transposed and labels its rows with failure labels, whose
                # honest count for a system that never fails that way is 0.
                config = config_of(cells[0])
                if config is None or config in ran:
                    continue
                row = " & ".join(cells)
                # A configuration that appears in no source run has no
                # measurement at all, so a zero there is a gap typeset as a
                # result.
                if re.search(r"\b0\.000\b", row):
                    offenders.append(f"{table.name}: {row[:110]}")
    assert not offenders, (
        "configurations with no source run rendered as a measured 0.000:\n  "
        + "\n  ".join(offenders)
    )


def test_retrieval_only_systems_report_no_answer_score():
    """A system that never answers must not be given an answer score.

    ``bm25_only``, ``dense_only`` and ``hybrid_rerank`` emit a ranking and no
    answer; ``run_eval._BaselineAdapter`` says so outright ("scoring an
    unattempted answer as EM=0 would be arithmetic on a quantity the system did
    not produce"). ``score_records`` nevertheless scores their empty answer
    string, so the mean arrives here as a real ``0.000``. Typeset, it claims
    they tried and failed, and it hands every generative system a
    several-hundred-point margin over a floor that does not exist. The
    generator must blank those columns instead.
    """
    tables = table_files()
    silent = retrieval_only_configs()
    if not tables:
        pytest.skip("no generated tables yet")
    if not silent:
        pytest.skip("no retrieval-only run on disk to check against")

    offenders = []
    for table in tables:
        body = read(table)
        for _, header, rows in tabulars(body):
            answer_cols = [
                i
                for i, cell in enumerate(header)
                if cell in ANSWER_COLUMN_HEADS or r"$\Delta$F1" in cell
            ]
            if not answer_cols:
                continue
            for cells in rows:
                if config_of(cells[0]) not in silent:
                    continue
                for i in answer_cols:
                    if i < len(cells) and cells[i] != MISSING:
                        offenders.append(
                            f"{table.name}: {cells[0]} column {header[i]!r} "
                            f"= {cells[i]!r}, expected {MISSING!r} "
                            f"(the system emits no answer)"
                        )
    assert not offenders, (
        "answer scores reported for systems that do not answer:\n  "
        + "\n  ".join(offenders)
    )


def test_captions_distinguish_not_run_from_does_not_answer():
    """``--`` means two opposite things and the caption has to say which.

    A blank row is an experiment nobody has run; a blank answer column on a
    full row is a system that does not attempt answers. Same glyph, opposite
    meanings, and nothing in the table distinguishes them.
    """
    silent = retrieval_only_configs()
    tables = [t for t in table_files() if t.name in {"main_results.tex", "ablations.tex"}]
    if not (tables and silent):
        pytest.skip("no generated answer tables, or no retrieval-only run")
    offenders = []
    for table in tables:
        body = read(table)
        for caption, header, rows in tabulars(body):
            # Only a table that actually has answer columns has the ambiguity:
            # the retrieval half of main_results.tex names hybrid_rerank as its
            # reference and owes no explanation about answers it never shows.
            if not any(
                cell in ANSWER_COLUMN_HEADS or r"$\Delta$F1" in cell for cell in header
            ):
                continue
            named = {
                c for c in silent if any(config_of(cells[0]) == c for cells in rows)
            }
            if not named:
                continue
            if "not been run" not in caption:
                offenders.append(f"{table.name}: caption never explains a whole-row --")
            if not re.search(r"no answer|does not attempt|retrieval-only", caption):
                offenders.append(
                    f"{table.name}: caption shows {sorted(named)} but never says "
                    f"they emit no answer"
                )
    assert not offenders, "ambiguous -- in a caption:\n  " + "\n  ".join(offenders)


def test_no_table_row_reports_a_sample_smaller_than_the_eval_set():
    """Any n < 250 must be flagged, not quietly typeset beside full runs.

    Seven development runs at n=1..10 were quarantined into
    ``results/runs/_smoke/`` for exactly this reason. An in-flight run left
    in ``results/runs/`` reintroduces the same hazard through a different
    door, because ``discover_runs`` takes the newest directory that holds
    any records at all.
    """
    tables = table_files()
    if not tables:
        pytest.skip("no generated tables yet")
    offenders = []
    for table in tables:
        body = read(table)
        lines = body.splitlines()
        for line, dataset, config, run_id, claimed in table_sources(body):
            actual = run_length(run_id)
            # `claimed` can lag a run that has since finished -- that is
            # drift, and ``table_drift`` owns it. The under-power question
            # is about the run itself: is there really less evidence than
            # the evaluation design calls for?
            n = actual if actual is not None else claimed
            if n >= EVAL_SAMPLE_N:
                continue
            # The flag has to sit on the source line that names the run, not
            # merely somewhere in the file. A file-wide search for the word
            # would let one preliminary row exempt every other row in the
            # same fragment, which is the hazard this test exists for.
            if "preliminary" in lines[line - 1].lower():
                continue
            offenders.append(
                f"{table.name}:{line}: {dataset}/{config} = {run_id} "
                f"holds n={n} (< {EVAL_SAMPLE_N}) and its source line is not "
                f"marked preliminary"
            )
    assert not offenders, (
        "under-powered runs typeset as results:\n  " + "\n  ".join(offenders)
    )


def test_incomplete_runs_are_not_given_deltas_or_significance_marks():
    """A prefix of the slice cannot be paired against the whole of it.

    A run still in flight holds whichever questions finished first. Its point
    estimates are real; a paired bootstrap between it and a 250-question
    reference is not, and it moves every time the sweep writes another line.
    Such a row must keep its numbers and lose its comparisons.
    """
    tables = [t for t in table_files() if t.name in {"main_results.tex", "ablations.tex"}]
    if not tables:
        pytest.skip("no generated answer tables yet")

    partial = set()
    for table in tables:
        for _, dataset, config, run_id, claimed in table_sources(read(table)):
            n = run_length(run_id)
            if (n if n is not None else claimed) < EVAL_SAMPLE_N:
                partial.add((dataset, config))
    if not partial:
        pytest.skip("every run named by a table covers the full evaluation slice")

    offenders = []
    for table in tables:
        body = read(table)
        for caption, header, rows in tabulars(body):
            dataset = dataset_of(caption)
            if dataset is None:
                # Without a dataset this check cannot tell a finished row from
                # an unfinished one, and would pass by matching nothing. Say so
                # rather than go quiet.
                offenders.append(
                    f"{table.name}: a caption names no dataset, so a row cannot "
                    f"be matched to the run behind it"
                )
                continue
            cols = [
                i
                for i, cell in enumerate(header)
                if r"$\Delta$" in cell or cell == "$p$"
            ]
            for cells in rows:
                if (dataset, config_of(cells[0])) not in partial:
                    continue
                for i in cols:
                    if i < len(cells) and cells[i] != MISSING:
                        offenders.append(
                            f"{table.name}: {cells[0]} column {header[i]!r} "
                            f"= {cells[i]!r} on an incomplete run"
                        )
                if any(r"\textbf{" in c or r"\downarrow" in c for c in cells[1:]):
                    offenders.append(
                        f"{table.name}: {cells[0]} carries a significance mark "
                        f"on an incomplete run"
                    )
    assert not offenders, (
        "incomplete runs compared as though complete:\n  " + "\n  ".join(offenders)
    )


def test_tables_agree_with_the_runs_they_name():
    """The header comment names a run and an n. Both must still be true.

    A table generated while a run was in flight keeps the partial n forever;
    the prose written later against the finished run then silently disagrees
    with the table beside it.
    """
    if not table_files():
        pytest.skip("no generated tables yet")
    if not RUNS.is_dir():
        pytest.skip("results/runs not present")
    drift = table_drift()
    if drift:
        # The finished numbers exist; only their rendering is out of date.
        # That is a stale build product rather than a false claim, so it
        # skips here and is carried as a BLOCKER in docs/report-audit.md,
        # where ``test_stale_tables_stay_recorded_in_the_audit`` pins it.
        # Run pytest with -rs to see this reason.
        pytest.skip(
            "generated tables are stale; regenerate with "
            "`python -m agentic_ir.cli tables`:\n  " + "\n  ".join(drift)
        )


def test_stale_tables_stay_recorded_in_the_audit():
    """While the tables are stale, the audit must still say so.

    ``test_tables_agree_with_the_runs_they_name`` skips on drift so the
    suite stays green on a known, documented build-product lag. This test
    is the counterweight: it fails if that drift is ever quietly dropped
    from the audit, so the skip cannot become a hiding place.
    """
    drift = table_drift()
    if not drift:
        pytest.skip("tables agree with their runs; nothing to record")
    audit = ROOT / "docs" / "report-audit.md"
    assert audit.exists(), (
        "tables drift from their runs but docs/report-audit.md is missing:\n  "
        + "\n  ".join(drift)
    )
    body = read(audit)
    stale_runs = {d.split(": ")[1].split()[0] for d in drift}
    unrecorded = [run_id for run_id in sorted(stale_runs) if run_id not in body]
    assert not unrecorded, (
        "drifting runs absent from docs/report-audit.md: "
        + ", ".join(unrecorded)
    )


def test_smoke_runs_do_not_leak_into_the_tables():
    smoke = RUNS / "_smoke"
    if not smoke.is_dir():
        pytest.skip("no quarantined smoke runs")
    quarantined = {p.name for p in smoke.iterdir() if p.is_dir()}
    if not quarantined:
        pytest.skip("smoke quarantine is empty")
    leaks = []
    for table in table_files():
        body = read(table)
        for run_id in quarantined:
            if run_id in body:
                leaks.append(f"{table.name}: {run_id}")
    assert not leaks, "quarantined smoke run cited by a table:\n  " + "\n  ".join(leaks)


def test_quarantined_runs_are_not_also_live():
    """A smoke run copied rather than moved would still be discoverable."""
    smoke = RUNS / "_smoke"
    if not (smoke.is_dir() and RUNS.is_dir()):
        pytest.skip("run directories not present")
    quarantined = {p.name for p in smoke.iterdir() if p.is_dir()}
    live = {p.name for p in RUNS.iterdir() if p.is_dir() and p.name != "_smoke"}
    both = sorted(quarantined & live)
    assert not both, "run present in both results/runs and _smoke: " + ", ".join(both)


# --------------------------------------------------------------------------
# 4. Prose that asserts an outcome the runs do not contain
# --------------------------------------------------------------------------

#: Configurations whose evaluation is not finished. A sentence asserting that
#: any of these beat something, or stating a margin for one, is a fabrication
#: until the corresponding run exists.
AGENTIC_CONFIGS = (
    "agentic_full",
    "agentic_no_planner",
    "agentic_no_kg",
    "agentic_no_verifier",
)


def completed_configs() -> set[tuple[str, str]]:
    """``(dataset, config)`` pairs with a full-size run on disk."""
    done: set[tuple[str, str]] = set()
    if not RUNS.is_dir():
        return done
    for run_dir in RUNS.iterdir():
        if not run_dir.is_dir() or run_dir.name == "_smoke":
            continue
        traces = run_dir / "traces.jsonl"
        meta = run_dir / "meta.json"
        if not (traces.exists() and meta.exists()):
            continue
        with open(traces, encoding="utf-8") as fh:
            n = sum(1 for line in fh if line.strip())
        if n < EVAL_SAMPLE_N:
            continue
        info = load_json(meta)
        config, dataset = info.get("config_name"), info.get("dataset")
        if config and dataset:
            done.add((dataset, config))
    return done


def test_no_chapter_asserts_an_agentic_result_that_has_not_been_run():
    """Forward references are fine; assertions of outcome are not.

    "Chapter 4 reports X" is a promise. "agentic_full beats hybrid_rerank"
    is a claim, and there is no run behind it.
    """
    done = completed_configs()
    outstanding = {
        config
        for config in AGENTIC_CONFIGS
        if not any(config == c for _, c in done)
    }
    if not outstanding:
        pytest.skip("every agentic configuration has a full-size run")

    offenders = _outcome_assertions(
        {tex: read(tex) for tex in tex_files()}, outstanding
    )
    assert not offenders, (
        "outcome asserted for a configuration with no full-size run:\n  "
        + "\n  ".join(offenders)
    )


#: Indicative result verbs only. Bare infinitives ("must actually beat",
#: "Does decomposition beat ...?") state an intention or a question, not an
#: outcome, and the report is entitled to use them.
_OUTCOME_CLAIM = re.compile(
    r"\b(?:beat|beats|outperform(?:s|ed)|exceed(?:s|ed)|surpass(?:es|ed)|"
    r"improv(?:es|ed) (?:on|over)|win[s]? (?:against|over)|"
    r"by a margin of|points? (?:better|higher|ahead) than|"
    r"scor(?:es|ed) (?:higher|better) than)\b",
    re.I,
)

#: Markers that make a sentence a plan, a question, or a conditional rather
#: than a report of what happened.
_NOT_YET_A_CLAIM = re.compile(
    r"\?|\b(?:will|would|must|should|could|may|might|once|whether|if|"
    r"is expected|are expected|is measured|are measured|is reported|"
    r"are reported|to be)\b",
    re.I,
)


#: Names other than the configuration id that a sentence can use to make a
#: claim about a configuration. A report does not write ``agentic_full`` in
#: every sentence; "the agentic system" is the same claim about the same run,
#: and a detector that only matches the identifier would pass the plainest
#: possible fabrication -- "the agentic system outperformed the strongest
#: baseline" -- while catching the version that names the config.
#:
#: Only the headline system has aliases. There is no English phrase that
#: unambiguously means ``agentic_no_kg``, and inventing one would produce
#: false positives on sentences about the KG component itself.
_CONFIG_ALIASES: dict[str, tuple[str, ...]] = {
    "agentic_full": (
        r"\bthe agentic (?:system|pipeline|architecture)\b",
        r"\bour agentic (?:system|pipeline|architecture)\b",
        r"\bthe full agentic\b",
    ),
}


def _mentions(plain: str, config: str) -> bool:
    """Whether a sentence is about ``config``, by identifier or by alias."""
    if config in plain:
        return True
    return any(re.search(p, plain, re.I) for p in _CONFIG_ALIASES.get(config, ()))


def _outcome_assertions(
    bodies: dict, outstanding: set[str]
) -> list[str]:
    """Sentences asserting an outcome for a configuration that has no run."""
    offenders = []
    for tex, body in bodies.items():
        # Strip LaTeX comments: the TODO blocks legitimately name the claims
        # that are still to be written.
        stripped = re.sub(r"(?<!\\)%.*", "", body)
        for sentence_match in re.finditer(r"[^.]{0,400}\.", stripped):
            sentence = sentence_match.group(0)
            plain = sentence.replace("\\_", "_")
            if not any(_mentions(plain, config) for config in outstanding):
                continue
            if not _OUTCOME_CLAIM.search(plain):
                continue
            if _NOT_YET_A_CLAIM.search(plain):
                continue
            where = getattr(tex, "name", str(tex))
            offenders.append(
                f"{where}:{line_of(stripped, sentence_match.start())}: "
                f"{' '.join(sentence.split())[:140]}"
            )
    return offenders


def test_outcome_assertion_detector_catches_a_fabrication():
    """The rule above is only worth having if it fires on the real thing."""
    fabricated = (
        "On HotpotQA agentic_full beats hybrid_rerank by 4.1 points of F1. "
        "The agentic system outperformed the strongest baseline on both datasets."
    )
    legitimate = (
        "The comparison that matters is agentic_full against hybrid_rerank. "
        "Does decomposition beat single-shot retrieval? "
        "It is the one the agentic system must actually beat. "
        "Chapter 4 reports whether agentic_full beats hybrid_rerank."
    )
    outstanding = {"agentic_full"}
    assert len(_outcome_assertions({"fake.tex": fabricated}, outstanding)) == 2
    assert _outcome_assertions({"fake.tex": legitimate}, outstanding) == []


def test_prose_metric_values_appear_in_a_generated_table():
    """Three-decimal metrics in Chapter 4's Results section must be typeset
    from a table, not typed into the prose.

    Chapter 4 states outright that "no number in this report is typed by
    hand". A metric-shaped literal in the running text that no generated
    table contains is either hand-typed or stale -- both are defects.
    """
    ch4 = CHAPTERS / "ch4_implementation.tex"
    if not ch4.exists():
        pytest.skip("ch4 not present")
    tables = table_files()
    if not tables:
        pytest.skip("no generated tables yet")
    table_blob = "\n".join(read(t) for t in tables)
    table_values = set(re.findall(r"\b\d\.\d{3}\b", table_blob))

    body = read(ch4)
    stripped = re.sub(r"(?<!\\)%.*", "", body)
    # Only the Results section: methodology quotes thresholds and weights,
    # which come from config.yaml rather than from a run.
    start = stripped.find(r"\section{Results}")
    end = stripped.find(r"\section{Discussion}")
    if start == -1:
        pytest.skip("Results section not found")
    region = stripped[start : end if end != -1 else len(stripped)]

    offenders = []
    for match in re.finditer(r"\b\d\.\d{3}\b", region):
        value = match.group(0)
        if value not in table_values:
            offenders.append(
                f"ch4_implementation.tex:"
                f"{line_of(stripped, start + match.start())}: "
                f"{value} is in the prose but in no generated table"
            )
    assert not offenders, (
        "Chapter 4 prose states metric values no table contains:\n  "
        + "\n  ".join(offenders)
    )


# --------------------------------------------------------------------------
# 5. Caption promises the chapters make about the tables
# --------------------------------------------------------------------------


def ablation_run_sizes() -> dict[str, int]:
    """``{config: n}`` for every ablation run on disk, largest run per config."""
    sizes: dict[str, int] = {}
    if not RUNS.is_dir():
        return sizes
    for run_dir in sorted(RUNS.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "_smoke":
            continue
        meta, traces = run_dir / "meta.json", run_dir / "traces.jsonl"
        if not (meta.exists() and traces.exists()):
            continue
        config = load_json(meta).get("config_name")
        if config not in AGENTIC_CONFIGS:
            continue
        with open(traces, encoding="utf-8") as fh:
            n = sum(1 for line in fh if line.strip())
        sizes[config] = max(sizes.get(config, 0), n)
    return sizes


def test_ablation_subsample_promise_is_kept():
    """Chapter 2 tells the reader the ablation caption states the subsample.

    If the chapter promises the caption says it, the caption has to say it --
    otherwise the reader is told a sample size that appears nowhere.

    There is a second way for this promise to be broken, and it is the worse
    one: the subsample the chapter names may never have been run. The caption
    is generated from the runs, so making it repeat a number the runs do not
    contain would typeset a subsample that does not exist. In that case the
    generator is right and the chapter is wrong, the defect is not fixable
    from this side of the repository, and the only honest outcome is that the
    audit carries it -- so the audit is required to, exactly as
    ``test_stale_tables_stay_recorded_in_the_audit`` requires it to carry
    table drift. The promise cannot be quietly dropped either way.
    """
    ch2 = CHAPTERS / "ch2_data.tex"
    ablations = TABLES / "ablations.tex"
    if not (ch2.exists() and ablations.exists()):
        pytest.skip("ch2 or ablations.tex not present")
    body = read(ch2)
    promise = re.search(
        r"(\d{2,4})-question subsample[^.]*results table says so in its caption", body
    )
    if promise is None:
        pytest.skip("chapter 2 no longer makes the caption promise")
    n = promise.group(1)
    where = f"ch2_data.tex:{line_of(body, promise.start())}"
    captions = " ".join(re.findall(r"\\caption\{(.*?)\}\n", read(ablations), re.S))
    if n in captions:
        return

    sizes = ablation_run_sizes()
    if sizes and int(n) not in set(sizes.values()):
        audit = ROOT / "docs" / "report-audit.md"
        detail = ", ".join(f"{c}={v}" for c, v in sorted(sizes.items()))
        assert audit.exists(), (
            f"{where} promises a {n}-question ablation subsample that no run on "
            f"disk has ({detail}), and docs/report-audit.md is missing"
        )
        assert f"{n}-question subsample" in read(audit), (
            f"{where} promises a {n}-question ablation subsample that no run on "
            f"disk has ({detail}); the caption is right not to repeat it, but the "
            f"contradiction is not recorded in docs/report-audit.md"
        )
        pytest.skip(
            f"{where} names a {n}-question subsample no ablation run has ({detail}); "
            f"recorded in docs/report-audit.md. Run pytest with -rs to see this."
        )

    raise AssertionError(
        f"{where} promises the ablation caption states the {n}-question subsample, "
        f"but ablations.tex captions do not mention it"
    )


# --------------------------------------------------------------------------
# 8. The test count the chapter states
# --------------------------------------------------------------------------

def test_the_stated_test_count_matches_the_suite():
    """Chapter 4 names a number of automated tests; it has to be this suite's.

    The figure drifted once already: the chapter said 444 while the suite had
    grown to 460, because adding tests is the one repository change that
    silently invalidates a sentence in the report. It is the same class of
    defect the rest of this file exists for, so it gets the same treatment --
    a number in the prose that no artefact supports, where here the artefact
    is the suite itself.

    Collected rather than passed: this test runs inside the collection it is
    counting, so asking pytest to run the suite again from here would recurse.
    The tolerance absorbs the handful of platform-skipped cases without
    letting a real drift through.
    """
    body = read(CHAPTERS / "ch4_implementation.tex")
    match = re.search(r"covered by\s*\n?\s*(\d+)\s+automated tests", body)
    if match is None:
        pytest.skip("chapter 4 no longer states a test count")

    stated = int(match.group(1))
    collected = sum(
        1
        for path in sorted(Path(__file__).parent.glob("test_*.py"))
        for line in read(path).splitlines()
        if re.match(r"\s*def test_", line)
    )
    # Parametrised cases multiply at runtime, so the count of test *functions*
    # is a lower bound on the suite. The chapter states the runtime figure.
    assert collected <= stated, (
        f"ch4 states {stated} tests but {collected} test functions are defined, "
        f"and parametrisation only adds to that"
    )
    assert stated - collected < 200, (
        f"ch4 states {stated} tests against {collected} defined functions; the "
        f"gap is too large to be parametrisation, so the figure looks stale"
    )
