"""The project entry point: ``ask``, ``eval``, ``tables``.

Three commands, one for each thing this project has to be able to do in front
of someone: answer one question with the whole machine visible, reproduce a
measured run, and regenerate every number in the report.

``ask`` is the demo, and it is deliberately not a JSON dump. The claim this
project makes is about *mechanism* -- a plan, a routed tool per sub-query, a
verifier that can send the planner back around -- and a mechanism nobody can
read has not been demonstrated. So the trace is rendered: the plan and where it
came from, each sub-query's tool with the rule id that chose it, the titles that
came back, the verifier's confidence against its own threshold, and, when it
fires, the re-plan directive that closed the loop. Every line of it is read back
off :class:`~agentic_ir.state.QuestionState`, so what is printed is exactly what
the evaluation harness would have written to ``traces.jsonl`` -- not a
prettier parallel story.

**Determinism (architecture section 9).** ``PYTHONHASHSEED`` only takes effect
at interpreter start-up, so setting it from inside a running process is
theatre. :func:`ensure_hash_seed` therefore re-executes the interpreter once
with the variable set, before anything heavier than the standard library is
imported, and then ``random``/``numpy``/``torch`` are seeded through the
harness's own :func:`~agentic_ir.eval.run_eval.seed_everything` so that the CLI
and the evaluation cannot drift into seeding differently.

Every import of this package is deferred into the command that needs it, for
the same reason: an import at module scope would run before the re-exec and
could sample from an unseeded RNG.

Usage::

    python -m agentic_ir.cli ask "Which magazine was started first, ...?"
    python -m agentic_ir.cli eval --dataset hotpotqa --config agentic_full --limit 5
    python -m agentic_ir.cli tables

The package must be importable: ``pip install -e .`` or ``PYTHONPATH=src``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import textwrap
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "HASH_SEED",
    "build_parser",
    "cmd_ask",
    "cmd_eval",
    "cmd_tables",
    "ensure_hash_seed",
    "format_trace",
    "main",
]

#: The seed ``PYTHONHASHSEED`` is pinned to. Matches ``project.seed``; kept as a
#: literal because it has to be applied before the config can be read.
HASH_SEED = 42

#: Set by :func:`ensure_hash_seed` in the child, so a re-exec can never recurse.
_REEXEC_GUARD = "AGENTIC_IR_HASH_SEED_APPLIED"

_RULE = "=" * 78
_THIN = "-" * 78


def _numeric_id(sq_id: str) -> tuple[int, str]:
    """Sort ``q10`` after ``q9``, matching the trace writer's own ordering."""
    digits = sq_id[1:]
    return (int(digits), sq_id) if digits.isdigit() else (10**6, sq_id)


# ---------------------------------------------------------------------------
# Determinism (architecture section 9)
# ---------------------------------------------------------------------------

def ensure_hash_seed(seed: int = HASH_SEED, argv: Sequence[str] | None = None) -> int | None:
    """Guarantee ``PYTHONHASHSEED``, re-executing the interpreter if need be.

    Assigning to ``os.environ["PYTHONHASHSEED"]`` inside a live process changes
    nothing: the hash randomisation it controls is fixed when the interpreter
    starts. The only honest ways to satisfy section 9 are to make the caller
    remember to set it -- which is exactly the kind of step that gets forgotten
    on the day the numbers are generated -- or to re-exec once with it set. This
    does the latter.

    Returns ``None`` when nothing was needed (already set, or the guard is in
    place); on Windows it returns the child's exit code, and on POSIX it never
    returns at all because ``execve`` replaces the process.
    """
    want = str(seed)
    if os.environ.get("PYTHONHASHSEED") == want or os.environ.get(_REEXEC_GUARD):
        return None

    env = dict(os.environ)
    env["PYTHONHASHSEED"] = want
    env[_REEXEC_GUARD] = "1"
    args = [sys.executable, "-m", "agentic_ir.cli", *(argv if argv is not None else sys.argv[1:])]

    if os.name == "nt":
        # execv on Windows detaches the console from the surviving process and
        # hands the shell its prompt back while the run is still printing. A
        # child process keeps the terminal coherent, which matters because this
        # is the command that gets demonstrated live.
        import subprocess  # noqa: PLC0415 - local: only this path needs it

        return subprocess.run(args, env=env, check=False).returncode
    os.execve(sys.executable, args, env)
    return None  # pragma: no cover - execve does not return


def _seed_everything(seed: int) -> dict[str, Any]:
    """Seed the RNGs through the harness's own function, never a second copy."""
    from .eval.run_eval import seed_everything

    return seed_everything(seed)


def _use_utf8_stdio() -> None:
    """Force UTF-8 on stdout/stderr.

    Both corpora are full of names like ``Xawery Zulawski``, and a Windows
    console defaulting to cp1252 turns printing one into a ``UnicodeEncodeError``
    that kills the command after the work is already done. ``errors="replace"``
    because a mangled character is a far better outcome than a lost answer.
    """
    import contextlib

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):  # a detached stream
                reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _wrap(text: str, *, indent: str = "  ", width: int = 78) -> str:
    body = textwrap.fill(
        " ".join(str(text).split()), width=width,
        initial_indent=indent, subsequent_indent=indent,
    )
    return body or f"{indent}(empty)"


def _clip(text: str, limit: int) -> str:
    """Collapse whitespace and truncate. ASCII ellipsis: every decoration this
    module adds stays inside cp1252, so the only characters that can trouble a
    Windows console are the ones the corpus itself contains."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: max(0, limit - 3)] + "..."


def _yn(value: object) -> str:
    return "yes" if value else "no"


def _plan_block(plan: Any, lines: list[str]) -> None:
    """One plan revision: the DAG, and where each node came from."""
    claimed = ""
    if getattr(plan, "strategy_llm", None) and plan.strategy_llm != plan.strategy:
        # The model mislabels the strategy most of the time and the shape of the
        # DAG is authoritative; showing both is how that stays visible.
        claimed = f", model said {plan.strategy_llm!r}"
    lines.append(
        f"  revision {plan.revision}  strategy={plan.strategy}{claimed}  "
        f"origin={plan.origin}  depth={plan.depth}  nodes={len(plan.subqueries)}"
    )
    if plan.repairs:
        lines.append(f"  repairs: {', '.join(plan.repairs)}")
    for sq in plan.subqueries:
        depends = ", ".join(sq.depends_on) if sq.depends_on else "-"
        flags = [f"hop {sq.hop}", sq.intent, f"expects {sq.answer_type}"]
        if sq.is_combiner:
            flags.append("combiner")
        if sq.tool_hint:
            flags.append(f"hint {sq.tool_hint}")
        lines.append(f"    {sq.id:<4}{_clip(sq.text, 66)}")
        lines.append(f"        [{', '.join(flags)}]  depends on: {depends}")
        if sq.rewrites:
            lines.append(f"        rewrites: {'; '.join(_clip(r, 50) for r in sq.rewrites)}")


def _retrieval_block(state: Any, lines: list[str], *, top: int) -> None:
    """Per sub-query: the routed tool, the rule that chose it, and what came back."""
    for sq_id in sorted(state.results, key=_numeric_id):
        result = state.results[sq_id]
        sel = result.selection
        lines.append(f"  {sq_id}  {_clip(result.query_text, 70)}")
        rule = sel.rule_id or "-"
        lines.append(
            f"      tool     {sel.tool}  (selector={sel.selector}, rule_id={rule}, "
            f"rerank={_yn(sel.rerank_applied)})"
        )
        if sel.reason:
            lines.append(f"      why      {_clip(sel.reason, 66)}")
        if result.queries_issued:
            issued = "; ".join(_clip(q, 44) for q in result.queries_issued)
            lines.append(f"      queries  {len(result.queries_issued)}: {issued}")
        lines.append(
            f"      pool     {result.n_candidates} candidates -> "
            f"{len(result.passages)} kept in {result.latency_s:.2f}s"
            + ("  [DEGRADED]" if result.degraded else "")
        )
        if result.error:
            lines.append(f"      error    {_clip(result.error, 66)}")
        for scored in result.passages[:top]:
            lines.append(
                f"        {scored.rank + 1:>2}. {_clip(scored.passage.title, 52):<52} "
                f"{scored.score:8.4f}  {scored.provenance}"
            )
        kg = state.kg_results.get(sq_id)
        if kg is not None:
            seeds = ", ".join(e.name for e in kg.seeds[:3]) or "none"
            lines.append(
                f"      kg       seeds=[{_clip(seeds, 40)}] linked_by={kg.linked_by} "
                f"bridge={kg.bridge_entity or '-'} paths={len(kg.paths)} "
                f"evidence={len(kg.evidence)}"
                + ("  [DEGRADED]" if kg.degraded else "")
            )
        answer = state.answers.get(sq_id)
        if answer:
            bridge = state.bridge_entities.get(sq_id)
            suffix = f"   (bridge entity: {bridge})" if bridge else ""
            lines.append(f"      answer   {_clip(answer, 60)}{suffix}")


def _verification_block(state: Any, lines: list[str], *, threshold: float) -> None:
    """Every cycle's candidate and verdict, and whether the loop closed."""
    for verification in state.verifications:
        candidate = verification.candidate
        lines.append(
            f"  cycle {candidate.cycle}: verdict={verification.verdict.upper()}  "
            f"confidence={verification.confidence:.3f} "
            f"(threshold {threshold:.2f})  method={verification.method}"
            + ("  [DEGRADED]" if verification.degraded else "")
        )
        lines.append(
            f"      support  nli={verification.nli_support:.3f}  "
            f"citations={verification.citation_grounding:.3f}  "
            f"retrieval={verification.retrieval_agreement:.3f}  "
            f"llm={'-' if verification.llm_support is None else f'{verification.llm_support:.3f}'}"
        )
        lines.append(f"      answer   {_clip(candidate.answer, 60) or '(none)'}")
        if verification.hallucinated_citations:
            lines.append(
                f"      HALLUCINATED CITATIONS: {', '.join(verification.hallucinated_citations)}"
            )
        if verification.missing_information:
            missing = "; ".join(_clip(m, 40) for m in verification.missing_information[:3])
            lines.append(f"      missing  {missing}")
        if verification.reason:
            lines.append(f"      reason   {verification.reason}")


def format_trace(state: Any, *, transitions: Sequence[str] = (), top: int = 3,
                 threshold: float = 0.55, evidence: int = 6) -> str:
    """Render one :class:`QuestionState` as a readable trace.

    Reads only the state, so what is shown and what the harness would have
    written to ``traces.jsonl`` are the same object seen twice. Anything the run
    degraded on is called out rather than smoothed over -- a demo that hides its
    fallbacks is a demo of something else.
    """
    metrics = state.metrics()
    best = state.best_candidate()
    lines: list[str] = [_RULE, f"QUESTION  [{state.qid}]  {state.dataset}/{state.config_name}",
                        _wrap(state.question), _THIN]

    lines.append(f"PLAN  ({len(state.plans)} revision(s))")
    if not state.plans:
        lines.append("  (no plan was produced)")
    for index, plan in enumerate(state.plans):
        if index:
            directive = next(
                (d for d in state.directives if d.revision == plan.revision), None
            )
            lines.append("")
            if directive is not None:
                lines.append(
                    f"  >> RE-PLAN {index} FIRED: reason={directive.reason}, "
                    f"confidence={directive.confidence:.3f}, "
                    f"{len(directive.banned_subquery_texts)} sub-query text(s) banned"
                )
                if directive.missing_information:
                    lines.append(
                        "     missing: "
                        + "; ".join(_clip(m, 46) for m in directive.missing_information[:3])
                    )
                if directive.suggested_subqueries:
                    lines.append(
                        "     suggested: "
                        + "; ".join(_clip(s, 46) for s in directive.suggested_subqueries[:3])
                    )
        _plan_block(plan, lines)

    lines += [_THIN, "RETRIEVAL AND TOOL ROUTING"]
    if state.results:
        _retrieval_block(state, lines, top=top)
    else:
        lines.append("  (nothing was retrieved)")

    lines += [_THIN, f"EVIDENCE POOL  ({len(state.evidence)} citable units)"]
    cited = set(best.citations) if best else set()
    for eid in sorted(state.evidence, key=_numeric_id)[:evidence]:
        item = state.evidence[eid]
        star = "*" if eid in cited else " "
        where = f"{item.title or '?'} s{item.sent_id}" if item.kind == "passage" else "kg triple"
        lines.append(f"  {star}{eid:<4}[{_clip(where, 40)}]  {_clip(item.text, 60)}")
    if len(state.evidence) > evidence:
        lines.append(f"  ... {len(state.evidence) - evidence} more (* = cited in the answer)")

    lines += [_THIN, "VERIFICATION"]
    if state.verifications:
        _verification_block(state, lines, threshold=threshold)
    else:
        lines.append("  (the verifier did not run in this configuration)")
    replans = int(metrics.get("replans", 0) or 0)
    lines.append(
        f"  re-plan: {'FIRED x' + str(replans) if replans else 'not fired'}   "
        f"answer selected from cycle {best.cycle if best else '-'} of "
        f"{len(state.candidates)} candidate(s) by argmax confidence"
    )

    lines += [_THIN, "ANSWER"]
    if best is not None and best.answer:
        lines.append(_wrap(best.answer, indent="  "))
        if best.answer_sentence:
            lines.append(_wrap(best.answer_sentence, indent="    "))
        lines.append(
            f"  confidence {best.confidence:.3f}   citations: "
            f"{', '.join(best.citations) or 'none'}   origin={best.origin}"
        )
    else:
        lines.append("  (no answer was produced)")

    lines += [_THIN,
              f"COST  llm_calls={metrics.get('llm_calls')} "
              f"(saved {metrics.get('llm_calls_saved')}, cache hits "
              f"{metrics.get('llm_cache_hits')}, parse failures "
              f"{metrics.get('parse_failures')})  "
              f"tool_calls={metrics.get('tool_calls')}  "
              f"latency={float(metrics.get('latency_s') or 0.0):.1f}s "
              f"(llm {float(metrics.get('llm_latency_s') or 0.0):.1f}s)",
              f"      terminated_by={state.terminated_by}  "
              f"degraded_steps={metrics.get('degraded_steps')}  "
              f"budget_exhausted={_yn(metrics.get('budget_exhausted'))}"]
    if transitions:
        lines.append(_path_line(transitions))
    if state.errors:
        lines.append(_THIN)
        lines.append("ERRORS")
        for err in state.errors:
            lines.append(f"  {_clip(err, 74)}")
    lines.append(_RULE)
    return "\n".join(lines)


def _path_line(transitions: Sequence[str]) -> str:
    """The state-machine path actually taken, as ``INIT -T1-> PLAN -T2-> ...``.

    The orchestrator records each hop as ``"T2:PLAN->EXECUTE"``, which already
    carries the endpoints; the static table is only consulted for a bare id, so
    that a hand-built ``("T1", "T2")`` from a test still renders.
    """
    from .orchestrator import TRANSITIONS

    parts: list[str] = []
    for token in transitions:
        tid, _, hop = str(token).partition(":")
        src, _, dst = hop.partition("->")
        if not dst:
            entry = TRANSITIONS.get(tid)
            if entry is None:
                parts.append(f"-{tid}-> ?")
                continue
            src, dst = entry[0], entry[1]
        if not parts:
            parts.append(src)
        parts.append(f"-{tid}-> {dst}")
    return "PATH  " + " ".join(parts)


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------

def _qid_for(question: str) -> str:
    """A stable id for an ad-hoc question, so two runs of it are comparable."""
    return "cli-" + hashlib.sha1(question.encode("utf-8")).hexdigest()[:12]


def cmd_ask(args: argparse.Namespace) -> int:
    """Run one question through the full agentic pipeline and print the trace."""
    from .config import load_config
    from .eval.run_eval import build_system, config_for, load_pipeline
    from .state import QuestionState

    question = " ".join(args.question).strip()
    if not question:
        print("ask: the question is empty", file=sys.stderr)
        return 2

    cfg = config_for(args.config_name, load_config())
    qid = args.qid or _qid_for(question)

    print(f"[1/3] loading {args.dataset} corpus, indexes and knowledge graph ...",
          file=sys.stderr, flush=True)
    started = time.perf_counter()
    pipeline = load_pipeline(
        args.dataset, cfg, with_kg=bool(cfg.get("agents.kg.enabled", True))
    )
    print(f"      ready in {time.perf_counter() - started:.1f}s "
          f"({len(pipeline.corpus):,} passages)", file=sys.stderr)
    for note in pipeline.notes:
        # A degraded component changes what the trace below means, so it is
        # reported before the run rather than left to be inferred from it.
        print(f"      DEGRADED: {note}", file=sys.stderr)

    print("[2/3] connecting to the local LLM ...", file=sys.stderr, flush=True)
    from .llm import get_client

    client = get_client()
    system = build_system(
        args.config_name, args.dataset, cfg=cfg, pipeline=pipeline, client=client
    )

    print(f"[3/3] running {args.config_name} ...\n", file=sys.stderr, flush=True)
    state = QuestionState(
        qid=qid, question=question, dataset=args.dataset, config_name=args.config_name
    )
    state = system.run(qid, question, gold=None, state=state)

    print(format_trace(
        state,
        transitions=tuple(getattr(system, "transitions", ()) or ()),
        top=args.passages,
        threshold=float(cfg.get("agents.verifier.confidence_threshold", 0.55)),
        evidence=args.evidence,
    ))

    if args.json is not None:
        import orjson

        from .trace import build_trace_record

        record = build_trace_record(
            state, run_id=f"ask_{qid}", seed=args.seed,
            model=str(cfg.get("llm.default_model", "")),
        )
        Path(args.json).write_bytes(orjson.dumps(record, option=orjson.OPT_INDENT_2))
        print(f"trace record written to {args.json}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

def _baseline_is_present(config_name: str) -> bool | None:
    """Whether a baseline module exists, without importing it.

    ``baselines/`` is a sibling milestone, so this uses the registry in
    ``baselines/base.py`` and tolerates its absence entirely: ``None`` means
    "cannot tell", which is treated as "go ahead and let the harness say so".
    ``find_spec`` is used rather than an import because a module being written
    right now may exist and not yet be importable, and a preflight check has no
    business failing on that.
    """
    import importlib.util

    try:
        from .baselines.base import BASELINE_NAMES
    except Exception:  # noqa: BLE001 - the whole package may not be written yet
        return None
    if config_name not in BASELINE_NAMES:
        return None
    try:
        return importlib.util.find_spec(f".baselines.{config_name}", package=__package__) is not None
    except (ImportError, ValueError):
        return False


def cmd_eval(args: argparse.Namespace) -> int:
    """Delegate to the evaluation harness, which owns every run artefact."""
    from .eval import run_eval as harness

    argv = list(args.forwarded)
    config_name = _flag_value(argv, "--config")
    if config_name and _baseline_is_present(config_name) is False:
        print(
            f"eval: configuration {config_name!r} needs "
            f"src/agentic_ir/baselines/{config_name}.py, which is not present yet. "
            "It is owned by a sibling milestone; run an agentic configuration, or "
            "wait for it to land.",
            file=sys.stderr,
        )
        return 2
    return harness.main(argv)


def _flag_value(argv: Sequence[str], flag: str) -> str | None:
    """``--flag value`` or ``--flag=value`` out of a raw argument list."""
    for index, token in enumerate(argv):
        if token == flag and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith(f"{flag}="):
            return token.split("=", 1)[1]
    return None


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def cmd_tables(args: argparse.Namespace) -> int:
    """Regenerate every LaTeX fragment the report ``\\input``s."""
    from .eval import tables

    return tables.main(list(args.forwarded))


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    from .eval.run_eval import AGENTIC_CONFIGS

    parser = argparse.ArgumentParser(
        prog="python -m agentic_ir.cli",
        description="Agentic AI for Information Retrieval -- multi-hop retrieval "
                    "with a Verifier->Planner feedback loop.",
        epilog="Runs are fully local: Ollama on localhost, no API keys anywhere.",
    )
    parser.add_argument("--seed", type=int, default=None,
                        help=f"RNG seed (default: project.seed, PYTHONHASHSEED={HASH_SEED})")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser(
        "ask", help="answer one question with the full agentic pipeline, showing the trace"
    )
    ask.add_argument("question", nargs="+", help="the question, quoted")
    ask.add_argument("--dataset", default="hotpotqa", choices=("hotpotqa", "twowiki"),
                     help="which corpus, index and knowledge graph to search")
    ask.add_argument("--config", dest="config_name", default="agentic_full",
                     choices=AGENTIC_CONFIGS,
                     help="agentic configuration; the ablations are available here too")
    ask.add_argument("--qid", default=None, help="override the generated question id")
    ask.add_argument("--passages", type=int, default=3,
                     help="retrieved titles to show per sub-query (default: 3)")
    ask.add_argument("--evidence", type=int, default=6,
                     help="evidence units to show (default: 6)")
    ask.add_argument("--json", type=Path, default=None,
                     help="also write the full trace record to this file")
    ask.set_defaults(func=cmd_ask)

    # Both delegating sub-commands declare NO arguments of their own and
    # `add_help=False`, so every token after the sub-command name -- `--help`
    # included -- comes back from `parse_known_args` in original order and is
    # handed to the module that owns those flags. Declaring a
    # `nargs=REMAINDER` catch-all instead reorders `--config bm25_only` into
    # `bm25_only --config`, which is worse than not accepting it at all.
    evaluate = sub.add_parser(
        "eval", add_help=False,
        help="run one configuration on one dataset; all flags are passed through to "
             "eval/run_eval.py (try: eval --help)",
    )
    evaluate.set_defaults(func=cmd_eval)

    tables = sub.add_parser(
        "tables", add_help=False,
        help="regenerate results/tables/*.tex from results/runs/; all flags are passed "
             "through to eval/tables.py (try: tables --help)",
    )
    tables.set_defaults(func=cmd_tables)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse, seed, dispatch. Does not re-exec: see :func:`ensure_hash_seed`."""
    _use_utf8_stdio()
    parser = build_parser()
    args, unknown = parser.parse_known_args(list(argv) if argv is not None else None)
    if unknown and args.command == "ask":
        parser.error("unrecognized arguments: " + " ".join(unknown))
    args.forwarded = unknown

    from .config import load_config

    seed = args.seed if args.seed is not None else int(load_config().get("project.seed", 42))
    args.seed = seed
    _seed_everything(seed)
    return int(args.func(args))


def _console_main() -> int:
    """Entry point for ``python -m``: pin ``PYTHONHASHSEED``, then run.

    The re-exec lives here and not in :func:`main` so that importing this module
    and calling ``main([...])`` -- which is what the tests do -- can never spawn
    a second interpreter.
    """
    code = ensure_hash_seed()
    return code if code is not None else main()


if __name__ == "__main__":
    raise SystemExit(_console_main())
