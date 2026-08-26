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
pip install -r requirements.txt

# 3. PyTorch with CUDA (do NOT rely on the default CPU wheel)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 4. Data + indexes
python scripts/download_data.py
python scripts/build_corpus.py
python scripts/build_indexes.py
```

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
- [ ] **M1** — Corpus, BM25 + dense indexes, non-agentic baselines
- [ ] **M2** — Planner and Retrieval agents
- [ ] **M3** — KG Navigator
- [ ] **M4** — Verifier and the re-plan feedback loop
- [ ] **M5** — Full evaluation, ablations, result tables
- [ ] **M6** — Report Chapters 1 & 5, final PDF

## License

Academic coursework. Datasets retain their original licenses (HotpotQA: CC BY-SA 4.0;
2WikiMultihopQA: Apache 2.0).
