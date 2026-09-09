# Agentic AI for Information Retrieval

A multi-agent retrieval system that decomposes complex multi-hop questions, selects
retrieval tools autonomously, traverses an entity knowledge graph, and self-validates
its own answers before returning them.

> **Course:** Information Retrieval, A.Y. 2024/2025
> **Università degli Studi di Napoli Federico II** — IKNOS-DIETI Lab
> **Professor:** Antonio Maria Rinaldi · **Instructor:** Dr. Domenico Benfenati
> **Student ID:** D03000104

---

## What this is

Most RAG systems are pipelines: retrieve once, generate once. This project builds an
**agentic** system instead — one where a verification step can send work *back* to the
planner, so the system re-queries when its own confidence is low. That feedback edge is
the difference between a pipeline and an agent, and measuring what it is worth is what the
report is for. It turned out to be worth something on one benchmark and nothing on the
other; see [Headline results](#headline-results).

Four agents, coordinated by an orchestrator, covering **all five** tasks in Table 1 of
the assignment brief:

| Agent | Assignment task covered | Role |
|---|---|---|
| **Planner** | Autonomous Query Refinement · Multi-Step Retrieval Planning | Decomposes a question into a DAG of dependent sub-queries; rewrites and expands terms; re-plans on verifier pushback |
| **Retrieval Agent** | Tool-Augmented Retrieval | Chooses *which* retrieval tool fits each sub-query: BM25, dense, hybrid RRF, cross-encoder rerank |
| **KG Navigator** | Knowledge Graph Traversal | Entity linking and bounded graph traversal to surface the *bridge entity* multi-hop questions hinge on |
| **Verifier** | Self-Reflective Validation & Fact Checking | NLI entailment + citation grounding + confidence scoring; below threshold, triggers a re-plan |

```
                  ┌──────────────────────────────────────┐
                  │            Orchestrator              │
                  └──────────────────────────────────────┘
                       │                            ▲
                       ▼                            │ low confidence
            ┌────────────────────┐                  │  → re-plan
            │   1. Planner       │◀─────────────────┘
            └────────────────────┘
                       │ sub-query DAG
          ┌────────────┴────────────┐
          ▼                         ▼
┌────────────────────┐   ┌────────────────────┐
│ 2. Retrieval Agent │   │ 3. KG Navigator    │
│  bm25 · dense      │   │  entity_link       │
│  hybrid · rerank   │   │  neighbors · path  │
└────────────────────┘   └────────────────────┘
          └────────────┬────────────┘
                       ▼
            ┌────────────────────┐
            │   4. Verifier      │──► answer + citations + confidence
            └────────────────────┘
```

## Datasets

| Dataset | Why it's here | Size used |
|---|---|---|
| **HotpotQA** (distractor) | Multi-hop by construction. Its *supporting facts* annotations are a genuine retrieval ground truth, not just answer strings. Also present in BEIR, so nDCG@10 is comparable to published numbers. | ~66k passages, 250-question stratified eval sample |
| **2WikiMultihopQA** | Ships gold **Wikidata evidence triples**, which is what lets the KG Navigator be scored against ground truth instead of only described. Reasoning-type labels enable per-type result tables. | ~55k passages, 250-question stratified eval sample |

**A note on the brief's Table 2.** The datasets listed there (The Pile, Common Crawl,
WikiText, OpenWebText, LAION-5B) are language-model *pretraining* corpora — they contain
no queries and no relevance judgments. Chapter 4 requires relevance metrics compared
against baseline IR systems, which is not computable without qrels. Following the brief's
own instruction to *"check for eventual updates of links and datasets via a web search"*,
we substitute two standard multi-hop IR benchmarks. This is justified in Chapter 2.

## Running locally — no API keys

Everything runs on-device via [Ollama](https://ollama.com). There are no API keys in this
repository and no network calls to any model provider.

### Requirements
- Python 3.11+ (developed on 3.13)
- NVIDIA GPU with ≥8 GB VRAM (developed on an RTX 5060 Laptop, 8 GB)
- ~15 GB disk for datasets, indexes, and model weights

### Setup

```bash
# 1. Install Ollama, then pull the agent model
winget install Ollama.Ollama          # Windows
ollama pull qwen3:8b

# 2. Python environment
python -m venv .venv
.venv\Scripts\activate                # Windows

# 3. PyTorch FIRST, from the cu128 index -- order matters, see below
pip install "torch>=2.7,<2.12" --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())"
#   -> must print True and a list containing 'sm_120'

# 4. Everything else
pip install -r requirements.txt
pip install -e .                      # see below -- `python -m agentic_ir...` needs this

# 5. Data, indexes, knowledge graph, frozen eval slices -- in this order
python scripts/download_data.py   --dataset all
python scripts/build_corpus.py    --dataset all
python scripts/build_indexes.py   --dataset all
python scripts/build_kg.py        --dataset all
python scripts/sample_eval_set.py --dataset all
```

The order is not cosmetic. `build_corpus` reads the frozen split
`download_data` wrote; `build_indexes` and `build_kg` both read
`data/processed/{dataset}_corpus.jsonl`; `sample_eval_set` draws from the
qrels `build_corpus` produced. Each stage is a no-op on re-run unless you pass
`--force`, so re-running the whole block after a failure is cheap.

`build_kg.py` is what the KG Navigator traverses -- without it the KG agent
returns empty results and degrades silently rather than failing, so a missing
graph looks like a weak ablation instead of a missing step.
`sample_eval_set.py` writes `data/processed/{dataset}_eval_250.jsonl` and a
disjoint `{dataset}_calib_50.jsonl`; the eval harness reads those files and
never re-samples, so the identity of the sample is part of the result.

**Why `pip install -e .`.** `pyproject.toml` puts the package under `src/`, so
`python -m agentic_ir.cli` and `python -m agentic_ir.eval.run_eval` fail with
`ModuleNotFoundError` from a clean clone until the package is installed.
`pytest` works without it (`pythonpath = ["src"]` in `pyproject.toml`) and so
do the `scripts/*.py`, which put `src/` on `sys.path` themselves -- which is
exactly why the omission is easy to miss until the first `python -m` command.

**If `C:` is tight.** Ollama stores models under `%USERPROFILE%\.ollama\models`
and Hugging Face caches under `%USERPROFILE%\.cache\huggingface`, both on the
system drive. Set `OLLAMA_MODELS` and `HF_HOME` to a roomier drive *before*
pulling the model or running the download script; ~15 GB lands in those two
caches.

**Why torch is installed first, and why cu128.** The RTX 5060 is Blackwell,
compute capability **sm_120**. CUDA 12.8 is the first toolkit that targets it,
so `cu124` and `cu126` wheels contain no machine code for this device. The trap
is that `pip install` from those indexes *succeeds* -- you get a green install
and a dead GPU, visible only as mysteriously slow indexing. Run the verify line;
it turns a silent 10x slowdown into a loud failure.

Order matters because `sentence-transformers` pulls `torch` from PyPI, whose
Windows wheel is CPU-only. Installing the cu128 build first means pip finds the
requirement already satisfied and leaves it alone.

### Environment variables

Set these **persistently**, not with `$env:`. Ollama runs as a background
service started at login, so a variable assigned in a shell session never
reaches it. After setting them, restart Ollama for the change to take effect.

```powershell
[Environment]::SetEnvironmentVariable("OLLAMA_NUM_PARALLEL",      "1",   "User")
[Environment]::SetEnvironmentVariable("OLLAMA_KEEP_ALIVE",        "30m", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_MAX_LOADED_MODELS", "1",   "User")
[Environment]::SetEnvironmentVariable("PYTHONUTF8",               "1",   "User")
[Environment]::SetEnvironmentVariable("OMP_NUM_THREADS",          "4",   "User")
[Environment]::SetEnvironmentVariable("KMP_DUPLICATE_LIB_OK",     "TRUE","User")
[Environment]::SetEnvironmentVariable("TOKENIZERS_PARALLELISM",   "false","User")
```

`OLLAMA_NUM_PARALLEL=1` matters more than it looks. Ollama allocates
`num_ctx x num_parallel` of KV cache, so at the default 4 slots our configured
8192-token context becomes 32768 tokens -- **4.5 GB of KV instead of 1.1 GB** --
which forces CPU offload on an 8 GB card and destroys throughput.

`OLLAMA_KEEP_ALIVE=30m` prevents model reloads between questions. At up to 20
LLM calls per question, ~15-second reloads would dominate the very latency
metric Chapter 4 reports.

`OMP_NUM_THREADS=4`, `KMP_DUPLICATE_LIB_OK=TRUE` and
`TOKENIZERS_PARALLELISM=false` prevent Windows segfaults during long runs. Two
were observed here -- one mid index-build, one mid-evaluation -- traced to
commit-charge pressure and a faiss/torch OpenMP conflict. Set them before any
250-question sweep; a segfault four hours into a run is expensive.

`PYTHONUTF8=1` because Windows still defaults to cp1252 and both datasets are
full of Unicode entity names (`Xawery Zulawski`). Without it the trace writer
dies mid-run on a `UnicodeEncodeError`.

### VRAM budget

The card reports 8151 MiB, but Windows already holds ~747 MiB, so the real
budget is **6.99 GiB**. `qwen3:8b` at Q4_K_M plus its KV cache needs ~6.5 GiB --
it fits, with almost nothing to spare. Every other model therefore runs on CPU
at query time:

| Component | Placement | Cost |
|---|---|---|
| `qwen3:8b` | **GPU**, exclusively | ~6.5 GiB |
| `bge-small` (index build) | GPU, offline only, Ollama stopped | ~1.5 GiB peak |
| `bge-small` (query encoding) | CPU | <10 ms per query |
| MiniLM cross-encoder | CPU | **5.0 s median, 7.2 s p90** per 50 pairs |
| DeBERTa NLI | CPU | ~0.5-2 s |

The cross-encoder row is the expensive one, and it was wrong here for a long
time: an earlier estimate of 0.3-0.6 s was off by a factor of ten. Measured
over 722 calls of the headline run it is about 15 s of a 28 s median question
-- the single largest cost per question, larger than generation. Moving it to
the GPU is still not the answer: the card has ~1.5 GiB free while Ollama holds
the model, and on Windows an over-commit spills to shared memory and makes
generation crawl. So the placement above is a deliberate trade of latency for
never OOMing mid-run, not a free win, and `max_wall_clock_s: 300` is what
absorbs it. Full analysis in [docs/environment-validation.md](docs/environment-validation.md).

### Measured performance

Verified on this machine, warm model, `qwen3:8b` Q4_K_M at 100% GPU:

| | |
|---|---|
| Generation throughput | **52 tok/s** |
| Schema-constrained planner call | **3.75 s** mean (2.8-6.7 s) |
| VRAM with model resident | 6707 / 8151 MiB — no CPU offload, 1.4 GiB spare |
| Structured-output reliability | **1 parse failure in 1065 calls** over the headline 250-question run |

The reliability row is the one worth trusting: it is the whole
`agentic_full_hotpotqa_20260904T221533Z` run, not an eight-call probe. An 8B
model at `temperature: 0` was expected to emit malformed JSON on a nontrivial
minority of structured calls; with the five-rung extraction ladder and two
repair retries it emitted one, which is why the fallback rules stay a safety
net rather than a co-author of the results.

The first call after a model load costs an extra ~9.5 s in CUDA warmup — which
is the whole reason `OLLAMA_KEEP_ALIVE` matters. Don't benchmark a cold model
and conclude the hardware is slow; that mistake was made once already here.

### Usage

```bash
# Single question, full agentic pipeline, with trace
python -m agentic_ir.cli ask "Which country is the firm that owns Babycham located?"

# One configuration on one dataset -- this is the unit of work, not the whole grid
python -m agentic_ir.cli eval --dataset hotpotqa --config agentic_full

# Resume a run that died at question 194 rather than restarting it
python -m agentic_ir.cli eval --dataset twowiki --config agentic_no_kg --resume

# Regenerate every table in the report from the runs on disk
python -m agentic_ir.cli tables
```

The full grid is nine `--config` values (`bm25_only`, `dense_only`,
`hybrid_rerank`, `naive_rag`, `self_ask`, `agentic_full`,
`agentic_no_planner`, `agentic_no_kg`, `agentic_no_verifier`) against two
`--dataset` values, one invocation each; there is no command that runs all
eighteen. `--split calib` selects the disjoint 50-question calibration slice
instead of the frozen 250. `--limit N` is for smoke tests only -- a limited run
is not comparable to a full one and the tables mark it by its `n`.

Ask the running system for the rest: `python -m agentic_ir.cli eval --help`
documents `--resume`, `--retry-failed`, `--run-id`, `--limit` and `--size`.

### Running `agentic_v2`, the improvement loop's configuration

`agentic_v2` is the system the improvement loop kept: the one-node plan plus
reasoning on the synthesis call. It is **not** one of the nine, and it is
absent from `config/config.yaml` on purpose -- `tables.py` renders a row for
every name in `evaluation.configurations`, so listing it there would put a
tenth row in a report it was never run for. It is declared instead in
`config/config.v2.yaml`, which is that file plus exactly two lines:
`agentic_v2` in the configuration list, and `trace.dir: results/runs_v2` so
`discover_runs` cannot see its output.

```bash
export AGENTIC_IR_CONFIG=config/config.v2.yaml
python -m agentic_ir.cli eval --dataset hotpotqa --config agentic_v2
python -m agentic_ir.cli eval --dataset twowiki  --config agentic_v2
```

Expect exact match 0.524 (HotpotQA) and 0.456 (2Wiki) at n=250, against
`agentic_full`'s 0.432 and 0.224 -- but expect them *approximately*. Identical
prompts at `temperature: 0` do not give identical completions on this stack:
Ollama's KV-cache prefix reuse changes the batch shape between calls, and
6-14% of answers move between runs of byte-identical code. Seed pinning does
not fix it. `docs/improvement-loop.log.md` has the measurement and what it
implies for every comparison in the report.

Without `AGENTIC_IR_CONFIG` pointing at the variant file, `--config
agentic_v2` is rejected by argparse. That is the intended behaviour, not a
bug: the shipped configuration is the report's, and it stays that way.

**Run evaluations through `cli eval`, not through `run_eval` directly.** Both
accept the same flags — `cli eval` passes everything through untouched — but
only the `cli` path re-executes the interpreter once with `PYTHONHASHSEED=42`
set, which is the only moment that variable can take effect. Calling
`python -m agentic_ir.eval.run_eval` skips that: it seeds `random`, `numpy` and
`torch` correctly, then records whatever `PYTHONHASHSEED` happened to be in the
environment into `meta.json` (`null` if it was unset). Check that field before
treating two runs as comparable.

Do not use the example question from `docs/architecture.md`
(*"Arthur's Magazine or First for Women"*) as a smoke test. Neither title is in
the processed corpus -- it is a train-split question and the corpus is built
from validation -- so it retrieves the wrong passages and looks like a routing
bug when nothing is wrong.

### Dashboard

```bash
python scripts/dashboard.py                      # http://127.0.0.1:8000
python scripts/dashboard.py --sweep-pid <pid>    # also report whether a sweep is alive
python scripts/dashboard.py --live               # enable the Ask tab (uses the GPU)
```

Four tabs. **Sweep** is the full 9×2 configuration grid with progress bars, so a
run nobody has started shows as an empty row rather than as an absence.
**Questions** browses every question of a run with its scores attached and
filters for the interesting ones — wrong, re-planned, recovered by re-planning,
abstained — and clicking one opens the full trace: every cycle's plan,
sub-queries, candidate and verification, the retrieval with the rule that chose
each tool, the evidence with the cited sentences marked, and the state-machine
path with `T9` and `T11` highlighted. **Results** renders the generated tables.
**Ask** runs the live pipeline.

Standard library only — no framework, nothing to install. The Ask tab is off by
default because answering one question loads the reranker, embedder and NLI
model onto the same 8 GB card an evaluation run uses.

### Guided walkthrough

```bash
python scripts/demo.py            # all six sections
python scripts/demo.py --list     # what they are
python scripts/demo.py --section 4
```

No GPU, no model server, no network: the walkthrough replays evaluation runs
that already happened and reads the state machine out of `orchestrator.py`, so
every number it prints can be checked against the file named beside it. The six
sections are the corpora and models, the architecture and its transition table,
one question traced end to end, the re-plan loop firing and recovering a wrong
answer, the results with their negative findings, and the commands for running
it live.

## Evaluation design

Baselines run from weakest to genuinely competitive, so the agentic system has to earn
its result rather than beat a strawman:

`bm25_only` → `dense_only` → `hybrid_rerank` → `naive_rag` → `self_ask` → **`agentic_full`**

Plus three ablations — `−planner`, `−kg`, `−verifier` — to attribute the gains.

Metrics fall into three groups:
- **Retrieval:** Recall@{2,5,10}, nDCG@10, MRR, Supporting-Fact EM/F1
- **Answer:** EM, F1
- **Agent-specific:** LLM calls, tool calls, latency, plan depth, re-plan rate, citation grounding %

That third group is what the brief means by *"agent-specific measures"* — an agentic
system that wins on accuracy while making 20× more LLM calls has not obviously won, and
the report says so.

## Repository layout

```
config/config.yaml         Every tunable knob; values referenced in the report
scripts/                   The five pipeline stages, in order:
                             download_data · build_corpus · build_indexes
                             build_kg · sample_eval_set
scripts/demo.py            Guided walkthrough; replays real runs, needs no GPU
scripts/dashboard.py       Local web dashboard: sweep progress, trace browser, live ask
scripts/dashboard.html     Its single-page front end; standard library only
src/agentic_ir/
  ├── agents/              Planner · Retriever · KG Navigator · Verifier · Synthesizer
  ├── tools/               Tool registry exposed to the Retrieval agent
  ├── indexing/            BM25, dense (FAISS), hybrid RRF, cross-encoder rerank
  ├── kg/                  Graph build, entity linking, traversal
  ├── baselines/           Non-agentic comparison systems
  ├── eval/                Metrics, bootstrap CIs, run harness, threshold calibration
  └── orchestrator.py      The agent loop and its budget caps
report/                    LaTeX source, Chapters 1–5
results/tables/            Generated .tex fragments the chapters \input{}
results/calibration/       Verifier threshold sweep, per dataset
results/runs/              Per-question traces — gitignored, see "Reproducing"
docs/architecture.md       What the code actually does, checked against the code
docs/report-audit.md       Integrity audit: every number traced to the run that made it
docs/assignment-brief.md   The original assignment specification
```

### Reproducing from a clean clone

`results/runs/` is gitignored, and it is the input to
`python -m agentic_ir.cli tables`. A clean clone therefore ships the generated
tables and the report that reads them, but not the traces they were computed
from: regenerating a table means re-running the configuration that produced it.
The frozen eval slices under `data/processed/` are gitignored too, but they are
reproducible — `sample_eval_set.py` draws them deterministically from
`project.seed: 42` against the frozen download, and prints the SHA-256 that
`meta.json` records, so a re-drawn slice can be checked against the one the
report used rather than assumed identical.

## Project status

- [x] **M0** — Repository scaffold, configuration, environment
- [x] **M1** — Corpus, BM25 + dense indexes, non-agentic baselines
- [x] **M2** — Planner and Retrieval agents
- [x] **M3** — KG Navigator
- [x] **M4** — Verifier and the re-plan feedback loop
- [x] **M5** — Full evaluation, ablations, result tables *(16 of the 18 grid cells —
      9 configurations × 2 datasets — are scored on all 250 questions. Complete on
      HotpotQA. On 2WikiMultihopQA the five baselines, `agentic_full` and
      `agentic_no_verifier` are done; `agentic_no_planner` is mid-run and
      `agentic_no_kg` has not started. Both render as `--` rows and nothing is
      concluded from them.)*
- [x] **M6** — Report Chapters 1–5, final PDF

### Headline results

On the frozen 250-question HotpotQA slice, against a `self_ask` reference:

| | EM | F1 | Citation grounding | Median latency |
|---|---|---|---|---|
| `self_ask` | 0.504 | 0.603 | 0.000 | 4.5 s |
| `agentic_full` | 0.432 | 0.510 | **0.960** | 28.3 s |

The system **loses** on answer accuracy — ΔF1 −0.093 [−0.145, −0.039], significant —
and wins on attribution and evidence recall: citation grounding 0.960 against 0.000,
and pooled supporting-fact recall 0.793 against 0.696.

**The re-plan loop does not replicate across datasets, and that is the headline
finding.** Removing the verifier costs 3.7 F1 points on HotpotQA
(ΔF1 −0.037 [−0.065, −0.011], *p* = 0.008) and nothing at all on 2WikiMultihopQA
(ΔF1 −0.001 [−0.025, +0.025], *p* = 1.000) — as null as a result can be, and not for
want of firing: the loop ran on 98 of 250 questions there. The defensible claim is the
conditional one, *the backward edge helps on HotpotQA and not on 2WikiMultihopQA*, and
that is what Chapter 4 says. Removing the planner or the knowledge graph costs nothing
measurable on either. All of this is reported as found rather than smoothed; see
`docs/report-audit.md` for the integrity audit that produced these corrections.

## License

Academic coursework. Datasets retain their original licenses (HotpotQA: CC BY-SA 4.0;
2WikiMultihopQA: Apache 2.0).
