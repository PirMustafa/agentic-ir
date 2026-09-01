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
the difference between a pipeline and an agent, and it is the central claim of the report.

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
| **2WikiMultihopQA** | Ships gold **Wikidata evidence triples**, which is what lets the KG Navigator be scored against ground truth instead of only described. Reasoning-type labels enable per-type result tables. | 250-question stratified eval sample |

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

# 5. Data + indexes
python scripts/download_data.py
python scripts/build_corpus.py
python scripts/build_indexes.py
```

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
| MiniLM cross-encoder | CPU | ~0.3-0.6 s per 50 passages |
| DeBERTa NLI | CPU | ~0.5-2 s |

Against `max_wall_clock_s: 300` the CPU encoder latencies are irrelevant, so
this costs essentially nothing and removes a whole class of mid-run OOM
failures. Full analysis in [docs/environment-validation.md](docs/environment-validation.md).

### Measured performance

Verified on this machine, warm model, `qwen3:8b` Q4_K_M at 100% GPU:

| | |
|---|---|
| Generation throughput | **52 tok/s** |
| Schema-constrained planner call | **3.75 s** mean (2.8-6.7 s) |
| VRAM with model resident | 6707 / 8151 MiB — no CPU offload, 1.4 GiB spare |
| Structured-output reliability | **8/8** parsed, 0 retries, 0 thinking leakage |

The first call after a model load costs an extra ~9.5 s in CUDA warmup — which
is the whole reason `OLLAMA_KEEP_ALIVE` matters. Don't benchmark a cold model
and conclude the hardware is slow; that mistake was made once already here.

### Usage

```bash
# Single question, full agentic pipeline, with trace
python -m agentic_ir.cli ask "Which magazine was started first, Arthur's Magazine or First for Women?"

# Full evaluation across all configurations
python -m agentic_ir.eval.run_eval --dataset hotpotqa --config agentic_full
```

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
scripts/                   Data download, corpus construction, index building
src/agentic_ir/
  ├── agents/              Planner · Retriever · KG Navigator · Verifier · Synthesizer
  ├── tools/               Tool registry exposed to the Retrieval agent
  ├── indexing/            BM25, dense (FAISS), hybrid RRF, cross-encoder rerank
  ├── baselines/           Non-agentic comparison systems
  ├── eval/                Metrics, bootstrap CIs, run harness
  └── orchestrator.py      The agent loop and its budget caps
report/                    LaTeX source, Chapters 1–5
results/                   Tables and figures (raw traces are gitignored)
docs/assignment-brief.md   The original assignment specification
```

## Project status

- [x] **M0** — Repository scaffold, configuration, environment
- [x] **M1** — Corpus, BM25 + dense indexes, non-agentic baselines
- [x] **M2** — Planner and Retrieval agents
- [x] **M3** — KG Navigator
- [x] **M4** — Verifier and the re-plan feedback loop
- [ ] **M5** — Full evaluation, ablations, result tables *(harness ready; runs pending)*
- [ ] **M6** — Report Chapters 1 & 5, final PDF

## License

Academic coursework. Datasets retain their original licenses (HotpotQA: CC BY-SA 4.0;
2WikiMultihopQA: Apache 2.0).
