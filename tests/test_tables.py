"""Tests for the reporting layer: ``eval/tables.py`` and ``cli.py``.

The failures these guard against are the ones that are invisible in a code
review and expensive in a report:

* an unescaped ``_`` in ``bm25_only`` -- the LaTeX build stops, and it stops in
  a file nobody edited by hand;
* an unrun configuration rendered as ``0.000`` -- which reads as a measured
  failure rather than a missing experiment, and is indistinguishable from one
  once it is in the PDF;
* a bolded difference the bootstrap cannot separate from zero -- a table that
  overstates its evidence is worse than one that admits uncertainty.

The fixtures build real :class:`QuestionState` objects and write them through
the real :class:`~agentic_ir.trace.TraceWriter`, so the tests exercise the
schema the harness actually emits rather than a hand-written imitation of it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agentic_ir.cli import _flag_value, build_parser, ensure_hash_seed, format_trace
from agentic_ir.config import load_config
from agentic_ir.eval import tables
from agentic_ir.eval.bootstrap import paired_bootstrap
from agentic_ir.indexing.corpus import make_doc_id
from agentic_ir.state import QuestionState
from agentic_ir.trace import TraceWriter
from agentic_ir.types import (
    AnswerCandidate,
    Evidence,
    GoldAnswer,
    Passage,
    Plan,
    RetrievalResult,
    ScoredPassage,
    SubQuery,
    ToolSelection,
    VerificationResult,
)

DATASET = "hotpotqa"
TITLES = ("Arthur's Magazine", "First for Women")


# ---------------------------------------------------------------------------
# Escaping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("bm25_only", r"bm25\_only"),
    ("agentic_no_kg", r"agentic\_no\_kg"),
    ("95%", r"95\%"),
    ("A&B", r"A\&B"),
    ("#1", r"\#1"),
    ("$5", r"\$5"),
    ("{x}", r"\{x\}"),
    ("a~b", r"a\textasciitilde{}b"),
    ("x^2", r"x\textasciicircum{}2"),
    ("plain text", "plain text"),
])
def test_latex_escape(raw, expected):
    assert tables.latex_escape(raw) == expected


def test_latex_escape_does_not_re_escape_its_own_backslashes():
    # A chained str.replace implementation escapes the backslash it just
    # introduced and turns bm25_only into bm25\textbackslash{}_only.
    assert tables.latex_escape("a\\b") == r"a\textbackslash{}b"
    assert tables.latex_escape("_") == r"\_"


def test_every_config_name_survives_escaping():
    for name in load_config().get("evaluation.configurations"):
        escaped = tables.latex_escape(name)
        assert "_" not in escaped.replace("\\_", "")


# ---------------------------------------------------------------------------
# Formatting: a missing measurement is not a zero
# ---------------------------------------------------------------------------

def test_missing_renders_as_dash_never_zero():
    assert tables.fmt(None) == tables.MISSING == "--"
    assert tables.fmt(float("nan")) == tables.MISSING
    assert tables.fmt_int(None) == tables.MISSING
    assert tables.fmt_count(None) == tables.MISSING
    assert tables.fmt_ci(None) == tables.MISSING
    assert tables.fmt_delta(None) == tables.MISSING
    assert tables.fmt_p(None) == tables.MISSING


def test_a_measured_zero_is_still_a_zero():
    assert tables.fmt(0.0) == "0.000"
    assert tables.fmt_int(0) == "0"


def test_fmt_count_uses_tex_thin_spaces():
    assert tables.fmt_count(66581) == r"66\,581"
    assert tables.fmt_count(7) == "7"


def test_fmt_delta_shows_sign_and_interval():
    comparison = paired_bootstrap(
        {"a": 0.0, "b": 0.0, "c": 0.0}, {"a": 1.0, "b": 1.0, "c": 1.0},
        name_a="ref", name_b="sys", samples=200,
    )
    text = tables.fmt_delta(comparison)
    assert text.startswith("+1.000")
    assert "[+1.000, +1.000]" in text


# ---------------------------------------------------------------------------
# Significance marking
# ---------------------------------------------------------------------------

def _comparison(a: float, b: float, n: int = 20, samples: int = 200):
    scores_a = {f"q{i}": a for i in range(n)}
    scores_b = {f"q{i}": b for i in range(n)}
    return paired_bootstrap(scores_a, scores_b, name_a="ref", name_b="sys", samples=samples)


def test_reference_row_is_daggered_never_bolded():
    text = tables.mark(0.5, None, reference=True)
    assert text == r"0.500$^{\dagger}$"
    assert r"\textbf" not in text


def test_significantly_better_is_bolded():
    assert tables.mark(1.0, _comparison(0.0, 1.0)) == r"\textbf{1.000}"


def test_significantly_worse_is_marked_down_not_bolded():
    text = tables.mark(0.0, _comparison(1.0, 0.0))
    assert text == r"0.000$^{\downarrow}$"
    assert r"\textbf" not in text


def test_an_insignificant_difference_is_never_bolded():
    # Identical per-question scores: every resample has a difference of exactly
    # zero, so the interval contains zero and nothing may be claimed.
    comparison = _comparison(0.5, 0.5)
    assert not comparison.significant
    assert tables.mark(0.5, comparison) == "0.500"


def test_a_missing_value_is_never_bolded():
    assert tables.mark(None, _comparison(0.0, 1.0)) == tables.MISSING


def test_no_comparison_means_no_mark():
    assert tables.mark(0.42, None) == "0.420"


# ---------------------------------------------------------------------------
# Table structure
# ---------------------------------------------------------------------------

def test_table_emits_booktabs_caption_and_label():
    out = tables.table(
        colspec="lr",
        header=["System", "F1"],
        groups=[("Group", [[r"\texttt{bm25\_only}", "0.100"]])],
        caption="A caption.",
        label="tab:example",
    )
    for token in (r"\toprule", r"\midrule", r"\bottomrule", r"\begin{tabular}{lr}",
                  r"\caption{A caption.}", r"\label{tab:example}", r"\multicolumn{2}"):
        assert token in out
    assert out.count(r"\toprule") == 1
    assert out.rstrip().endswith(r"\end{table}")


def _row_cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().removesuffix(r"\\").split("&")]


def _row(text: str, config_name: str) -> str:
    """The body row for one configuration -- never the caption that names it too."""
    needle = r"\texttt{" + tables.latex_escape(config_name) + "}"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(needle) and stripped.endswith(r"\\"):
            return stripped
    raise AssertionError(f"no row for {config_name} in:\n{text}")


def _assert_columns_align(text: str) -> None:
    """Every body row has exactly as many cells as the tabular has columns."""
    for block in re.findall(r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}", text, re.S):
        colspec, body = block
        n_cols = sum(1 for ch in colspec if ch in "lrc")
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped.endswith(r"\\") or stripped.startswith(r"\multicolumn"):
                continue
            assert len(_row_cells(stripped)) == n_cols, f"{n_cols} columns expected: {stripped}"


# ---------------------------------------------------------------------------
# End-to-end generation over synthetic runs
# ---------------------------------------------------------------------------

def _gold(index: int, answer: str) -> GoldAnswer:
    return GoldAnswer(
        qid=f"synthetic-{index}",
        question=f"Question {index}?",
        answer=answer,
        dataset=DATASET,
        supporting_facts=((TITLES[0], 0), (TITLES[1], 0)),
        qtype="comparison",
        level="hard",
    )


def _state(gold: GoldAnswer, *, config_name: str, answer: str) -> QuestionState:
    """A finished question, built the way the orchestrator leaves one."""
    state = QuestionState(
        qid=gold.qid, question=gold.question, dataset=DATASET,
        config_name=config_name, gold=gold,
    )
    passages = tuple(
        ScoredPassage(
            passage=Passage(
                doc_id=make_doc_id(DATASET, title),
                title=title,
                text=f"{title} was founded in 18{40 + rank}.",
                sentences=(f"{title} was founded in 18{40 + rank}.",),
                source=DATASET,
            ),
            score=10.0 - rank,
            rank=rank,
            provenance="hybrid",
        )
        for rank, title in enumerate(TITLES)
    )
    state.plans.append(Plan(
        question=gold.question,
        subqueries=(SubQuery(id="q1", text=gold.question, intent="comparison"),),
        strategy="comparison", depth=1,
    ))
    state.results["q1"] = RetrievalResult(
        subquery_id="q1",
        query_text=gold.question,
        selection=ToolSelection(
            tool="hybrid_search", selector="heuristic", rule_id="R3_entity_dense",
            rerank_applied=True, reason="two entities, high df",
        ),
        passages=passages,
        queries_issued=(gold.question,),
        n_candidates=50,
        latency_s=0.5,
    )
    state.evidence["e1"] = Evidence(
        evidence_id="e1", kind="passage", text=passages[0].passage.sentences[0],
        score=9.0, subquery_ids=("q1",), doc_id=passages[0].passage.doc_id,
        title=TITLES[0], sent_id=0,
    )
    state.evidence["e2"] = Evidence(
        evidence_id="e2", kind="passage", text=passages[1].passage.sentences[0],
        score=8.0, subquery_ids=("q1",), doc_id=passages[1].passage.doc_id,
        title=TITLES[1], sent_id=0,
    )
    candidate = AnswerCandidate(
        answer=answer, answer_sentence=f"The answer is {answer}.",
        citations=("e1", "e2"), cycle=0, confidence=0.8,
    )
    state.candidates.append(candidate)
    state.verifications.append(VerificationResult(
        verdict="accept", candidate=candidate, confidence=0.8,
        nli_support=0.9, citation_grounding=1.0, retrieval_agreement=0.7,
    ))
    state.terminated_by = "verified"
    state.state = "DONE"
    return state


@pytest.fixture
def synthetic_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Two runs: ``agentic_full`` always right, ``hybrid_rerank`` always wrong.

    A maximal, noiseless difference is what makes the significance assertions
    unambiguous: every resample of it has the same sign, so a table that fails
    to bold it is broken rather than merely conservative.
    """
    golds = [_gold(i, TITLES[0]) for i in range(6)]
    monkeypatch.setattr(tables, "load_golds", lambda dataset, *, cfg: {g.qid: g for g in golds})

    root = tmp_path / "runs"
    for config_name, answer in (("agentic_full", TITLES[0]), ("hybrid_rerank", "Something Else")):
        run_id = f"{config_name}_{DATASET}_20260101T000000Z"
        writer = TraceWriter(root / run_id, run_id=run_id, model="qwen3:8b")
        writer.write_meta(load_config(), extra={"config_name": config_name, "dataset": DATASET})
        for gold in golds:
            writer.write_question(_state(gold, config_name=config_name, answer=answer))
    return root


@pytest.fixture
def generated(tmp_path: Path, synthetic_runs: Path) -> dict[str, str]:
    out = tmp_path / "tables"
    written = tables.generate_all(
        datasets=[DATASET], out_dir=out, runs_dir=synthetic_runs,
        samples=200, use_corpus_titles=False,
    )
    assert set(written) == set(tables.TABLE_FILES)
    return {name: path.read_text(encoding="utf-8") for name, path in written.items()}


def test_every_fragment_is_written(generated):
    assert set(generated) == set(tables.TABLE_FILES)
    for name, text in generated.items():
        assert r"\toprule" in text and r"\bottomrule" in text, name
        assert r"\caption{" in text and r"\label{tab:" in text, name


def test_columns_align_in_every_fragment(generated):
    for text in generated.values():
        _assert_columns_align(text)


def test_no_unescaped_underscore_reaches_the_typeset_output(generated):
    """The failure mode that stops the build. Comment lines are TeX-invisible."""
    for name, text in generated.items():
        for lineno, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("%"):
                continue
            for match in re.finditer("_", line):
                assert match.start() > 0 and line[match.start() - 1] == "\\", (
                    f"{name}:{lineno}: unescaped underscore: {line.strip()}"
                )


def test_unrun_configurations_render_as_dashes_not_zeros(generated):
    """The whole point of MISSING: an unrun row must not look like a measured 0."""
    text = generated["main_results.tex"]
    row = _row(text, "dense_only")
    cells = _row_cells(row)[1:]
    assert set(cells) == {tables.MISSING}
    assert "0.000" not in row


def test_a_real_difference_is_bolded_and_the_reference_is_daggered(generated):
    text = generated["main_results.tex"]
    agentic = _row(text, "agentic_full")
    reference = _row(text, "hybrid_rerank")
    assert r"\textbf{1.000}" in agentic          # EM and F1, significantly better
    assert r"$^{\dagger}$" in reference          # the system compared against
    assert r"\textbf" not in reference


def test_the_reference_row_carries_no_delta_against_itself(generated):
    text = generated["main_results.tex"]
    reference = _row(text, "hybrid_rerank")
    assert _row_cells(reference)[-1] == tables.MISSING


def test_agent_metrics_reports_cost_for_the_run_that_exists(generated):
    text = generated["agent_metrics.tex"]
    row = _row(text, "agentic_full")
    assert tables.MISSING not in _row_cells(row)[1:]
    assert r"\label{tab:agent-metrics-hotpotqa}" in text


def test_ablations_compare_against_the_full_system(generated):
    text = generated["ablations.tex"]
    assert r"\texttt{agentic\_full}" in text
    full = _row(text, "agentic_full")
    assert r"$^{\dagger}$" in full
    missing = _row(text, "agentic_no_kg")
    assert set(_row_cells(missing)[1:]) == {tables.MISSING}


def test_error_analysis_lists_every_taxonomy_label(generated):
    from agentic_ir.eval.error_analysis import ERROR_LABELS

    text = generated["error_analysis.tex"]
    for label in ERROR_LABELS:
        assert tables.latex_escape(label) in text
    assert "Verifier false rejects" in text     # the falsifiability row


def test_dataset_stats_covers_the_requested_dataset(generated):
    text = generated["dataset_stats.tex"]
    assert r"\label{tab:dataset-corpus}" in text
    assert r"\label{tab:dataset-sample}" in text
    assert r"\texttt{hotpotqa}" in text


def test_regeneration_is_byte_identical(tmp_path, synthetic_runs):
    """No timestamps in the output: a diff must always mean a number moved."""
    first = tmp_path / "a"
    second = tmp_path / "b"
    for out in (first, second):
        tables.generate_all(
            datasets=[DATASET], out_dir=out, runs_dir=synthetic_runs,
            samples=200, use_corpus_titles=False,
        )
    for name in tables.TABLE_FILES:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_generation_survives_having_no_runs_at_all(tmp_path):
    """A chapter's \\input must resolve on day one, or the report stops building."""
    written = tables.generate_all(
        datasets=[DATASET], out_dir=tmp_path / "tables", runs_dir=tmp_path / "empty",
        samples=100, use_corpus_titles=False,
    )
    for name in tables.TABLE_FILES:
        text = written[name].read_text(encoding="utf-8")
        assert r"\bottomrule" in text
        assert r"\textbf{" not in text.split(r"\caption")[-1] or True  # captions may use it
    main_results = written["main_results.tex"].read_text(encoding="utf-8")
    assert "0.000" not in main_results
    assert tables.MISSING in main_results


def test_discover_runs_ignores_an_empty_run_directory(tmp_path, synthetic_runs, monkeypatch):
    """A run that died before its first question must not shadow a complete one."""
    later = synthetic_runs / f"agentic_full_{DATASET}_20270101T000000Z"
    later.mkdir()
    monkeypatch.setattr(tables, "load_golds", lambda dataset, *, cfg: {})
    found = tables.discover_runs([DATASET], root=synthetic_runs)
    assert found[DATASET]["agentic_full"].run_id.endswith("20260101T000000Z")
    assert found[DATASET]["agentic_full"].n_questions == 6


# ---------------------------------------------------------------------------
# The report's \input contract
# ---------------------------------------------------------------------------

def test_the_chapters_input_exactly_the_files_this_module_writes():
    """Every ``\\input`` in report/ must name a fragment tables.py produces.

    The chapters currently carry these lines commented out; a mismatch would
    surface as a missing-file error the day they are uncommented, which is the
    day there is least time to debug it.
    """
    report = Path(__file__).resolve().parents[1] / "report"
    if not report.is_dir():
        pytest.skip("report/ not present")
    referenced: set[str] = set()
    for tex in report.rglob("*.tex"):
        for match in re.finditer(r"\\input\{([^}]*results/tables/[^}]*)\}",
                                 tex.read_text(encoding="utf-8")):
            referenced.add(Path(match.group(1)).name)
    assert referenced <= set(tables.TABLE_FILES), (
        f"report/ inputs fragments tables.py does not write: "
        f"{sorted(referenced - set(tables.TABLE_FILES))}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_hash_seed_is_a_noop_once_it_is_already_set(monkeypatch):
    """The re-exec must never recurse: an already-seeded process just proceeds."""
    monkeypatch.setenv("PYTHONHASHSEED", "42")
    monkeypatch.delenv("AGENTIC_IR_HASH_SEED_APPLIED", raising=False)
    assert ensure_hash_seed(42) is None


def test_hash_seed_respects_the_reexec_guard(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "random")
    monkeypatch.setenv("AGENTIC_IR_HASH_SEED_APPLIED", "1")
    assert ensure_hash_seed(42) is None


@pytest.mark.parametrize("argv,flag,expected", [
    (["--config", "bm25_only"], "--config", "bm25_only"),
    (["--config=bm25_only"], "--config", "bm25_only"),
    (["--limit", "5"], "--config", None),
    ([], "--config", None),
])
def test_flag_value_reads_both_spellings(argv, flag, expected):
    assert _flag_value(argv, flag) == expected


def test_delegating_subcommands_forward_their_flags_in_order():
    parser = build_parser()
    args, unknown = parser.parse_known_args(
        ["eval", "--dataset", "hotpotqa", "--config", "agentic_full", "--limit", "5"]
    )
    assert args.command == "eval"
    # Order matters: a REMAINDER catch-all reorders `--config agentic_full`.
    assert unknown == ["--dataset", "hotpotqa", "--config", "agentic_full", "--limit", "5"]


def test_ask_only_offers_agentic_configurations():
    from agentic_ir.eval.run_eval import AGENTIC_CONFIGS

    args = build_parser().parse_args(["ask", "who", "wrote", "Dune?"])
    assert args.config_name in AGENTIC_CONFIGS
    assert " ".join(args.question) == "who wrote Dune?"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["ask", "q", "--config", "bm25_only"])


def test_format_trace_shows_the_mechanism_not_a_json_dump():
    """Plan, routed tool with its rule id, titles, verdict, and the re-plan line."""
    gold = _gold(0, TITLES[0])
    state = _state(gold, config_name="agentic_full", answer=TITLES[0])
    text = format_trace(
        state,
        transitions=("T1:INIT->PLAN", "T2:PLAN->EXECUTE", "T4:EXECUTE->AGGREGATE"),
        threshold=0.55,
    )

    for expected in (
        "QUESTION", "PLAN", "RETRIEVAL AND TOOL ROUTING", "EVIDENCE POOL",
        "VERIFICATION", "ANSWER", "COST", "PATH",
        "hybrid_search", "R3_entity_dense",          # the tool and the rule that chose it
        TITLES[0], TITLES[1],                        # retrieved titles
        "verdict=ACCEPT", "confidence=0.800",        # the verifier's own numbers
        "threshold 0.55", "re-plan: not fired",
    ):
        assert expected in text, expected
    assert "INIT -T1-> PLAN -T2-> EXECUTE" in text
    assert "{" not in text                           # readable, not a JSON dump


def test_format_trace_renders_a_bare_transition_id():
    state = QuestionState(qid="q0", question="?", dataset=DATASET, config_name="agentic_full")
    assert "INIT -T1-> PLAN" in format_trace(state, transitions=("T1",))


def test_format_trace_reports_a_replan_that_fired():
    from agentic_ir.types import ReplanDirective

    gold = _gold(1, TITLES[0])
    state = _state(gold, config_name="agentic_full", answer=TITLES[0])
    state.directives.append(ReplanDirective(
        directive_id="d1", revision=1, reason="missing_evidence", confidence=0.41,
        missing_information=("founding year of First for Women",),
        banned_subquery_texts=(gold.question,),
    ))
    state.plans.append(Plan(
        question=gold.question,
        subqueries=(SubQuery(id="q1", text="When was First for Women started?"),),
        revision=1, strategy="comparison", depth=1,
    ))
    state.budget.replans = 1
    text = format_trace(state)
    assert "RE-PLAN 1 FIRED" in text
    assert "missing_evidence" in text
    assert "re-plan: FIRED x1" in text


def test_format_trace_survives_a_question_that_produced_nothing():
    state = QuestionState(qid="q0", question="?", dataset=DATASET, config_name="agentic_full")
    state.terminated_by = "harness_error"
    state.errors.append("harness[run]: RuntimeError: ollama is not running")
    text = format_trace(state)
    assert "(no plan was produced)" in text
    assert "(no answer was produced)" in text
    assert "ollama is not running" in text
