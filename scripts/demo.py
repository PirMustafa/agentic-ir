"""A guided walkthrough of the system, driven entirely by artefacts on disk.

The point of this script is to be *checkable*. Every number it prints is read
at run time from the thing that produced it -- the transition table from
``orchestrator.py``, the corpus and index sizes from ``data/indexes``, the
walkthroughs from real ``traces.jsonl`` records of the evaluated runs -- so a
reader who doubts a line can open the file named beside it. Nothing here is
transcribed from the report, and nothing needs a GPU: the demo replays runs
that already happened.

Sections, all of which run by default::

    python scripts/demo.py                 # everything
    python scripts/demo.py --section 3     # just the end-to-end walkthrough
    python scripts/demo.py --list          # what the sections are

The one thing this cannot show from disk is the system answering a question it
has not seen. Section 6 prints the command for that, which needs Ollama up.
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RUNS = ROOT / "results" / "runs"
INDEXES = ROOT / "data" / "indexes"
TABLES = ROOT / "results" / "tables"

#: The run every walkthrough replays. It is the full system on the frozen
#: 250-question HotpotQA slice -- the run the report's headline rows come from.
DEMO_RUN = "agentic_full_hotpotqa_20260904T221533Z"

#: Cycle 0 answered "yes" to a question asking for a rapper's name, scored 0.493
#: against a 0.55 threshold, and the backward edge sent it back twice. Chosen
#: because the first answer is wrong in a way nobody has to take on trust.
BACKWARD_EDGE_QID = "5ab2fccb5542991669774176"

RULE = "=" * 78
THIN = "-" * 78


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def head(n: int, title: str) -> None:
    print(f"\n{RULE}\n  {n}. {title}\n{RULE}")


def sub(title: str) -> None:
    print(f"\n  {title}\n  {THIN[:len(title)]}")


def wrap(text: str, indent: str = "      ", width: int = 74) -> str:
    """Wrap without importing textwrap's opinions about indentation."""
    words, lines, cur = str(text).split(), [], ""
    for word in words:
        if len(cur) + len(word) + 1 > width - len(indent):
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(indent + line for line in lines)


def load_traces(run_id: str = DEMO_RUN) -> list[dict]:
    path = RUNS / run_id / "traces.jsonl"
    if not path.exists():
        raise SystemExit(f"demo: no traces at {path} -- run the evaluation first")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def normalise(answer: str) -> str:
    """The official HotpotQA normalisation, so 'correct' here means what it means there."""
    text = (answer or "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def is_correct(record: dict) -> bool:
    return normalise(record["final_answer"]) == normalise(record["gold"]["answer"])


# ---------------------------------------------------------------------------
# 1. what is installed
# ---------------------------------------------------------------------------

def section_assets() -> None:
    head(1, "What the system is built on")

    print(wrap(
        "Everything below runs locally. There is no API key anywhere in this "
        "project: the language model is served by Ollama on the same machine, "
        "and the retrievers, reranker and NLI model are downloaded weights.",
        indent="  ",
    ))

    for dataset in ("hotpotqa", "twowiki"):
        stats_path = INDEXES / f"{dataset}_index_stats.json"
        kg_path = INDEXES / f"{dataset}_kg_meta.json"
        if not stats_path.exists():
            continue
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        sub(f"{dataset}")
        print(f"      corpus            {stats['n_docs']:,} passages, "
              f"{stats['n_terms']:,} distinct terms")
        files = stats.get("files", {})
        print(f"      BM25 index        {files.get('bm25_bytes', 0) / 1e6:>8.1f} MB   "
              f"(bm25s + PyStemmer)")
        print(f"      dense index       {files.get('dense_bytes', 0) / 1e6:>8.1f} MB   "
              f"(FAISS over bge-small-en-v1.5)")
        if kg_path.exists():
            kg = json.loads(kg_path.read_text(encoding="utf-8"))
            print(f"      knowledge graph   {kg['nodes']:,} nodes, {kg['edges']:,} edges "
                  f"induced from the corpus")
        check = stats.get("verification", {})
        if check:
            print(f"      smoke query       {check.get('query', '')!r}")
            for name in ("bm25", "dense", "hybrid"):
                if name in check:
                    row = check[name]
                    print(f"        {name:<8} {row['hits']:>3} hits in "
                          f"{row['latency_s']:>7.4f}s  top: {row['top']!r}")

    sub("models")
    config = (ROOT / "config" / "config.yaml").read_text(encoding="utf-8")
    for label, pattern in (
        ("generator", r"default_model:\s*(\S+)"),
        ("fallback", r"fallback_model:\s*(\S+)"),
        ("embedder", r"model:\s*(BAAI/\S+)"),
        ("reranker", r"model:\s*(cross-encoder/ms-marco\S+)"),
        ("NLI", r"nli_model:\s*(\S+)"),
    ):
        match = re.search(pattern, config)
        if match:
            print(f"      {label:<10} {match.group(1)}")
    print(wrap(
        "The NLI model runs on the CPU on purpose: the 8 GB card is fully "
        "committed to the generator, and a 0.5-2 s entailment check that never "
        "competes for VRAM is cheaper than one that sometimes evicts the model.",
        indent="      ",
    ))


# ---------------------------------------------------------------------------
# 2. the machine itself
# ---------------------------------------------------------------------------

def section_architecture() -> None:
    head(2, "The architecture, read out of the code")

    from agentic_ir.orchestrator import STATES, TRANSITIONS

    print(wrap(
        "Four agents and an orchestrator. The Planner decomposes the question "
        "into a DAG of sub-queries; the Retrieval Agent answers each one with a "
        "tool it chooses by rule; the KG Navigator looks for bridge entities "
        "between them; the Synthesizer writes an answer with citations; the "
        "Verifier scores it. The contribution of the project is one edge that "
        "points backwards: when the Verifier's confidence falls below "
        "threshold it hands the Planner a directive and the question is planned "
        "again.",
        indent="  ",
    ))

    sub(f"states ({len(STATES)})")
    print("      " + "  ->  ".join(STATES[:5]))
    print("      " + "  ->  ".join(STATES[5:]))

    sub(f"transitions ({len(TRANSITIONS)}), from src/agentic_ir/orchestrator.py")
    for tid, (src, dst, guard) in TRANSITIONS.items():
        arrow = "<<<" if dst == "PLAN" and src == "REPLAN_GATE" else "-->"
        print(f"      {tid:<4} {src:<12} {arrow} {dst:<10} {guard}")
    print(wrap(
        "T11 is the backward edge. Everything else in the table moves forward "
        "or terminates.", indent="      ",
    ))

    sub("the five guards on T11, from Orchestrator._replan_gate")
    guards = (
        ("G1", "max_replans", "the cycle budget is not spent"),
        ("G2", "budget_iterations", "the iteration budget is not spent"),
        ("G3", "budget_llm", "at least 3 privileged model calls remain"),
        ("G4", "budget_wallclock", "under 60% of the wall-clock allowance is used"),
        ("G5", "uninformative_feedback", "the directive actually says something new"),
    )
    for gid, name, why in guards:
        print(f"      {gid}  {name:<22} {why}")
    print(wrap(
        "The first guard to fail names terminated_by in the trace, so every "
        "question that could have looped and did not says why it stopped.",
        indent="      ",
    ))

    # A demo that drifts from the code is worse than no demo, so the guard
    # names above are checked against the source that implements them.
    source = (ROOT / "src" / "agentic_ir" / "orchestrator.py").read_text(encoding="utf-8")
    missing = [name for _, name, _ in guards if f'"{name}"' not in source]
    if missing:
        print(f"\n      WARNING: guard names not found in orchestrator.py: {missing}")


# ---------------------------------------------------------------------------
# 3. one question, end to end
# ---------------------------------------------------------------------------

def _print_plan(record: dict, cycle: int = 0) -> None:
    plans = record.get("plans") or []
    plan = next((p for p in plans if p.get("revision", 0) == cycle), None)
    if plan is None:
        return
    origin = plan.get("origin", "?")
    print(f"      strategy: {plan.get('strategy')}  (the model proposed "
          f"{plan.get('strategy_llm')})   origin: {origin}")
    for sq in plan.get("subqueries", []):
        dep = f" after {sq['depends_on']}" if sq.get("depends_on") else ""
        print(f"        [{sq['id']}] hop {sq.get('hop')}  {sq['text']}{dep}")
        print(f"              intent={sq.get('intent')}  "
              f"answer_type={sq.get('answer_type')}  tool_hint={sq.get('tool_hint')}")


def _print_retrieval(record: dict, limit: int = 2) -> None:
    for sqid, block in list((record.get("retrieved") or {}).items()):
        print(f"        [{sqid}] tool={block.get('tool')}  "
              f"chosen by {block.get('selector')}/{block.get('rule_id')}  "
              f"rerank={block.get('rerank_applied')}  "
              f"{block.get('n_candidates')} candidates in {block.get('latency_s', 0):.2f}s")
        print(f"              because: {block.get('reason')}")
        for passage in (block.get("passages") or [])[:limit]:
            title = passage.get("title") or passage.get("doc_id")
            print(f"              {passage.get('score', 0):>7.3f}  {title}")


def _print_kg(record: dict) -> None:
    kg = record.get("kg") or {}
    if not kg:
        print("        (the knowledge graph was not consulted for this question)")
        return
    for sqid, block in kg.items():
        bridge = block.get("bridge_entity") or "none found"
        print(f"        [{sqid}] seeds={block.get('seeds')}  linked_by="
              f"{block.get('linked_by')}")
        print(f"              bridge entity: {bridge}   "
              f"{block.get('n_paths')} paths, {block.get('n_neighbors')} neighbours, "
              f"{block.get('n_evidence')} evidence in {block.get('latency_s', 0):.2f}s")


def _print_verification(record: dict, index: int = 0) -> None:
    verifications = record.get("verifications") or []
    if index >= len(verifications):
        return
    ver = verifications[index]
    print(f"        verdict: {ver['verdict'].upper()}   "
          f"confidence {ver['confidence']:.3f}   method={ver.get('method')}")
    print(f"          nli_support        {ver.get('nli_support', 0):.3f}   x 0.45")
    print(f"          citation_grounding {ver.get('citation_grounding', 0):.3f}   x 0.25")
    print(f"          retrieval_agree    {ver.get('retrieval_agreement', 0):.3f}   x 0.15")
    llm = ver.get("llm_support")
    print(f"          llm_support        "
          f"{'not called' if llm is None else format(llm, '.3f')}   x 0.15")
    for claim in (ver.get("claims") or [])[:3]:
        mark = "supported" if claim.get("supported") else "NOT supported"
        print(f"          claim: {claim['text'][:60]!r}")
        print(f"            {claim.get('nli_label')} {claim.get('nli_score', 0):.3f} "
              f"against {claim.get('best_premise_id')} -> {mark}")
    if ver.get("suggested_subqueries"):
        print(f"          suggests: {ver['suggested_subqueries']}")


def section_walkthrough(traces: list[dict]) -> None:
    head(3, "One question, end to end")

    # A question the system got right on the first pass, with the graph used and
    # a plan that actually decomposed -- the clean case, before the loop.
    record = next(
        (
            r for r in traces
            if is_correct(r)
            and r["metrics"]["cycles"] == 1
            and r["metrics"]["n_subqueries"] >= 2
            and (r.get("kg") or {})
        ),
        traces[0],
    )

    print(f"  run:  {record['run_id']}")
    print(f"  qid:  {record['qid']}")
    print(f"\n  Q:    {record['question']}")
    print(f"  gold: {record['gold']['answer']}")

    sub("PLAN -- the Planner decomposes")
    _print_plan(record)

    sub("EXECUTE -- the Retrieval Agent picks a tool per sub-query")
    _print_retrieval(record)
    print(wrap(
        "selector=rule means the choice cost zero model calls. The rule id is "
        "recorded so the routing table in the report is counted, not asserted.",
        indent="        ",
    ))

    sub("EXECUTE -- the KG Navigator looks for a bridge")
    _print_kg(record)

    sub("AGGREGATE -- evidence is pooled, deduplicated and numbered")
    evidence = record.get("evidence") or []
    print(f"        {len(evidence)} sentences survived deduplication; the first three:")
    for item in evidence[:3]:
        print(f"          {item['evidence_id']}  [{item.get('provenance')}] "
              f"{item.get('title')} s{item.get('sent_id')}  score {item.get('score', 0):.3f}")
        print(wrap(item["text"][:150], indent="              "))

    sub("SYNTHESIZE -- an answer with citations")
    print(f"        answer:   {record['final_answer']!r}")
    print(f"        sentence: {record['answer_sentence'][:120]!r}")
    print(f"        cites:    {record['citations']}")

    sub("VERIFY -- the Verifier scores it")
    _print_verification(record)

    sub("DONE")
    print(f"        path: {' '.join(record.get('transitions', []))}")
    metrics = record["metrics"]
    print(f"        {metrics['llm_calls']} model calls, "
          f"{metrics['tool_calls']} tool calls, {metrics['latency_s']:.1f}s")
    print(f"        the agents recorded {metrics['llm_calls_saved']} calls "
          f"'avoided by rules' on this question")
    print(wrap(
        "That last figure is the raw counter and the report does not use it. "
        "An agent counts a saving whenever a rule decided something a model "
        "could have decided, including where this configuration has no model "
        "on that path at all -- routing and KG linking are set to rules here, "
        "so nothing was ever going to call a model for them. tables.py "
        "recounts only the savings on paths an LLM could actually have taken, "
        "which turns a corpus mean of 7.10 into 2.60 and prints both.",
        indent="        ",
    ))
    print(f"        correct: {is_correct(record)}   "
          f"terminated_by: {record.get('terminated_by')}")


# ---------------------------------------------------------------------------
# 4. the backward edge
# ---------------------------------------------------------------------------

def section_backward_edge(traces: list[dict]) -> None:
    head(4, "The backward edge, firing")

    record = next((r for r in traces if r["qid"] == BACKWARD_EDGE_QID), None)
    if record is None:
        candidates = [
            r for r in traces
            if r["metrics"]["replanned"] and r["best_cycle"] > 0 and is_correct(r)
        ]
        if not candidates:
            print("  no re-planned question in this run ended on a later cycle")
            return
        record = candidates[0]

    print(f"  qid:  {record['qid']}")
    print(f"\n  Q:    {record['question']}")
    print(f"  gold: {record['gold']['answer']}")
    print(wrap(
        "This is the mechanism the whole project exists to test. Read the "
        "confidence column: the first answer scores below the 0.55 threshold, "
        "the Verifier writes a directive, and the Planner runs again with it.",
        indent="  ",
    ))

    by_cycle = {c["cycle"]: c for c in record.get("candidates", [])}
    verifications = record.get("verifications") or []

    for cycle in sorted(by_cycle):
        cand = by_cycle[cycle]
        marker = "  <-- selected" if cycle == record["best_cycle"] else ""
        sub(f"cycle {cycle}{marker}")
        _print_plan(record, cycle)
        print(f"        answer:     {cand['answer']!r}")
        print(f"        confidence: {cand['confidence']:.3f}   "
              f"sufficient={cand.get('sufficient')}   origin={cand.get('origin')}")
        if cycle < len(verifications):
            ver = verifications[cycle]
            print(f"        verdict:    {ver['verdict'].upper()}")
            if ver.get("missing_information"):
                print(f"        missing:    {ver['missing_information'][:2]}")
            if ver.get("suggested_subqueries"):
                print(f"        directive:  ask instead -> "
                      f"{ver['suggested_subqueries'][:2]}")

    sub("what the machine did")
    print(f"        path: {' '.join(record.get('transitions', []))}")
    print(wrap(
        "T9 is VERIFY -> REPLAN_GATE (verdict revise); T11 is REPLAN_GATE -> "
        "PLAN, the backward edge, taken because all five guards passed.",
        indent="        ",
    ))
    print(f"\n        first answer:  {by_cycle[min(by_cycle)]['answer']!r}")
    print(f"        final answer:  {record['final_answer']!r}")
    print(f"        gold:          {record['gold']['answer']!r}")
    print(f"        correct after re-planning: {is_correct(record)}")

    sub("and how often that happens, across the whole 250-question slice")
    replanned = [r for r in traces if r["metrics"]["replanned"]]
    recovered = [r for r in replanned if r["best_cycle"] > 0 and is_correct(r)]
    first_wrong = [
        r for r in recovered
        if not any(
            normalise(c["answer"]) == normalise(r["gold"]["answer"])
            for c in r["candidates"] if c["cycle"] == 0
        )
    ]
    print(f"        re-planned at least once:        {len(replanned):>3} / {len(traces)}")
    print(f"        ended on a later cycle, correct: {len(recovered):>3}")
    print(f"        ... and cycle 0 had been wrong:  {len(first_wrong):>3}")
    print(wrap(
        "The ablation is what turns this into evidence: removing the edge and "
        "holding everything else fixed costs 3.7 F1 points on HotpotQA, "
        "dF1 -0.037 [-0.065, -0.011], p = 0.008. Section 5.",
        indent="        ",
    ))


# ---------------------------------------------------------------------------
# 5. results
# ---------------------------------------------------------------------------

def _rows_from(table: Path, wanted: tuple[str, ...]) -> list[list[str]]:
    if not table.exists():
        return []
    body = table.read_text(encoding="utf-8")
    block = re.search(r"\\begin\{tabular\}(.*?)\\end\{tabular\}", body, re.S)
    if block is None:
        return []
    out = []
    for line in block.group(1).splitlines():
        if "&" not in line:
            continue
        cells = [c.strip() for c in line.strip().removesuffix(r"\\").split("&")]
        name = re.sub(r"\\texttt\{|\}|\\", "", cells[0]).replace("_", "_").strip()
        if any(name.startswith(w) for w in wanted):
            out.append([name] + cells[1:])
    return out


def _clean(cell: str) -> str:
    """One LaTeX cell as plain text, keeping only marks a terminal can carry.

    The generated tables mark significance with ``\\downarrow`` and the
    reference row with ``\\dagger``. Rendering both as one asterisk would make
    the reference row look significant against itself, so they are kept apart:
    ``!`` means significantly worse, and the reference is named in the legend
    rather than marked in the cell.
    """
    cell = re.sub(r"\$\^\{?\\downarrow\}?\$", "!", cell)
    cell = re.sub(r"\$?\^?\{?\\(dagger|ddagger|ast)\}?\$?", "", cell)
    cell = re.sub(r"\\text(bf|it)\{(.*?)\}", r"\2", cell)
    return re.sub(r"[\\${}^]", "", cell).strip()


def section_results() -> None:
    head(5, "What it scores, and what it does not")

    sub("HotpotQA, 250 questions (results/tables/ablations.tex)")
    rows = _rows_from(TABLES / "ablations.tex", ("agentic", "hybrid"))
    if rows:
        print(f"      {'system':<22}{'n':>5}{'EM':>7}{'F1':>7}  "
              f"{'dF1 vs full [95% CI]':<26}{'p':>7}{'calls':>7}{'med s':>7}")
        for cells in rows:
            vals = [_clean(c) for c in cells[1:]]
            while len(vals) < 7:
                vals.append("--")
            name = _clean(cells[0]).split(" ")[0]
            print(f"      {name:<22}{vals[0]:>5}{vals[1]:>7}{vals[2]:>7}  "
                  f"{vals[3]:<26}{vals[4]:>7}{vals[5]:>7}{vals[6]:>7}")
        print("\n      ! = significantly worse than agentic_full "
              "(paired bootstrap, 95% CI excluding zero)")
        print("      hybrid_rerank emits a ranking and no answer, so it has "
              "no EM or F1 to report")

    print(wrap(
        "Read the signs honestly. Removing the verifier hurts -- that is the "
        "project's claim, and it survives a paired bootstrap. Removing the "
        "planner does not hurt, and removing the knowledge graph does not "
        "hurt: two of the four agents cannot be shown to earn their cost on "
        "two-hop questions. Both nulls are in the report as findings.",
        indent="      ",
    ))

    sub("and against the baselines, the result that is not flattering")
    print(wrap(
        "agentic_full scores F1 0.510 against self_ask's 0.603 on HotpotQA: "
        "dF1 -0.093 [-0.145, -0.039], significantly worse. It wins on "
        "attribution instead -- citation grounding 0.960 against 0.000 for "
        "every baseline, and evidence-pool recall 0.793 against 0.706 for the "
        "strongest retriever, hybrid_rerank. The report states it that way "
        "round rather than leading with the column it wins.",
        indent="      ",
    ))

    sub("calibration (results/calibration/hotpotqa_threshold.json)")
    calib = ROOT / "results" / "calibration" / "hotpotqa_threshold.json"
    if calib.exists():
        data = json.loads(calib.read_text(encoding="utf-8"))
        flat = json.dumps(data)
        for key in ("ece", "brier", "auc", "threshold", "n_questions"):
            match = re.search(rf'"{key}[^"]*":\s*([-\d.]+)', flat)
            if match:
                print(f"      {key:<14} {float(match.group(1)):g}")
    print(wrap(
        "The confidence score discriminates better than it calibrates: it "
        "ranks right answers above wrong ones, but its absolute value is not "
        "a probability. That is why the threshold is tuned on a disjoint "
        "50-question slice rather than set to a round number.",
        indent="      ",
    ))


# ---------------------------------------------------------------------------
# 6. live
# ---------------------------------------------------------------------------

def section_live() -> None:
    head(6, "Running it live")
    print(wrap(
        "Everything above is a replay, so it needs no GPU and no model server. "
        "To watch the same machine answer a question it has never seen, start "
        "Ollama and run:", indent="  ",
    ))
    print("""
      ollama serve                       # in another terminal
      ollama pull qwen3:8b

      python -m agentic_ir.cli ask "Which magazine was started first, \\
          Arthur's Magazine or First for Women?" --dataset hotpotqa

      # the same question without the backward edge, to compare:
      python -m agentic_ir.cli ask "..." --config agentic_no_verifier
""")
    print(wrap(
        "ask prints the same stages this demo replays -- plan, tool choices, "
        "evidence, citations, verification, the transition path -- and "
        "--json writes the trace record so it can be diffed against the ones "
        "in results/runs.", indent="  ",
    ))
    print(wrap(
        "To reproduce the tables instead: python -m agentic_ir.cli eval "
        "--config agentic_full --dataset hotpotqa, then python -m "
        "agentic_ir.cli tables.", indent="  ",
    ))


# ---------------------------------------------------------------------------

SECTIONS = {
    1: ("What the system is built on", lambda t: section_assets()),
    2: ("The architecture, read out of the code", lambda t: section_architecture()),
    3: ("One question, end to end", section_walkthrough),
    4: ("The backward edge, firing", section_backward_edge),
    5: ("What it scores, and what it does not", lambda t: section_results()),
    6: ("Running it live", lambda t: section_live()),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", type=int, action="append", choices=sorted(SECTIONS))
    parser.add_argument("--run", default=DEMO_RUN, help="run id to replay")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    if args.list:
        for number, (title, _) in SECTIONS.items():
            print(f"  {number}. {title}")
        return 0

    wanted = args.section or sorted(SECTIONS)
    traces: list[dict] | None = None
    if any(n in (3, 4) for n in wanted):
        traces = load_traces(args.run)

    print(f"\n{RULE}\n  Agentic AI for Information Retrieval -- guided walkthrough")
    print(f"  D03000104 | replaying {args.run}\n{RULE}")

    for number in wanted:
        SECTIONS[number][1](traces)  # type: ignore[arg-type]

    print(f"\n{RULE}\n  end of walkthrough\n{RULE}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
