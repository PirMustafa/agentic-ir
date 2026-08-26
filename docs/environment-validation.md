# Environment Validation Report

**Reviewer pass:** M0 foundation de-risking, before real code is layered on top.
**Target machine:** Windows 11 (10.0.26200), Python 3.13.0 (conda-forge, MSC v.1942, win-amd64),
pip 25.1, NVIDIA RTX 5060 Laptop GPU (8151 MiB, driver 592.82), 31.6 GB RAM, 119 GB free on `D:`.
**Date of checks:** 2026-08-26

Every claim below is tagged **[VERIFIED]** (I ran it on this machine / queried the live API and
pasted the result) or **[INFERRED]** (reasoned from verified facts plus documented behaviour).

---

## Summary

| # | Issue | Severity |
|---|---|---|
| 1 | `xanhho/2WikiMultihopQA` is a script-loader dataset — cannot load under `datasets>=4.0` | **BLOCKER** |
| 2 | README's `cu124` PyTorch index cannot drive an RTX 5060 (sm_120) | **BLOCKER** |
| 3 | `requirements.txt` installs a **CPU-only** torch from PyPI, then README downgrades it | **HIGH** |
| 4 | Unpinned `>=` floors resolve to breaking majors; `sentence-transformers` 6.x forces `transformers` 5.x | **HIGH** |
| 5 | VRAM budget is over-subscribed by ~1.4 GiB if all four models are GPU-resident | **HIGH** |
| 6 | `Config` is documented "read-only" but is a globally mutable cached singleton | **HIGH** |
| 7 | `AGENTIC_IR_CONFIG` env var is silently ignored after the first `load_config()` | MEDIUM |
| 8 | `lru_cache` creates 4 distinct `Config` objects for the same file and exhausts `maxsize=4` | MEDIUM |
| 9 | Reranker model ID `ms-marco-MiniLM-L-6-v2` is a stale redirect | MEDIUM |
| 10 | Windows cp1252 default encoding will crash on dataset text | MEDIUM |
| 11 | `retrieval.dense.batch_size: 256` is unvalidated and risks OOM during index build | MEDIUM |
| 12 | README omits `pip install -e .`, so `python -m agentic_ir.cli` fails | MEDIUM |
| 13 | Ollama is not installed; `qwen3:8b` not pulled | LOW |
| 14 | `resolve_path` raises a bare `TypeError` that does not name the key | LOW |
| 15 | `yaml.safe_load` can return `None`; `Config(None, ...)` is not guarded | LOW |
| 16 | `Paths.from_config` does filesystem writes (`mkdir`) as a read-path side effect | LOW |

**Good news up front — the four packages flagged as "suspect" are all fine.** Every one of
`bm25s`, `PyStemmer`, `faiss-cpu`, `pytrec-eval-terrier` ships a prebuilt `cp313` / `win_amd64`
wheel. No C or C++ compiler is needed, and no substitute (pure-Python BM25, hand-rolled
nDCG/MRR) is required. See §1.

---

## 1. Dependency installability on Python 3.13 / Windows — **NO BLOCKER**

**[VERIFIED]** Method: `pip download --no-deps --only-binary=:all: -d <scratch> <pkg>`. Forcing
`--only-binary=:all:` means a success proves a prebuilt wheel exists; a failure would prove a
source build (and therefore a compiler) is required. All six succeeded:

| Package | Resolved wheel | Compiler needed? |
|---|---|---|
| `bm25s` | `bm25s-0.3.11-py3-none-any.whl` | No (pure Python; only depends on `numpy`) |
| `PyStemmer` | `pystemmer-3.1.0-cp313-cp313-win_amd64.whl` | **No** — cp313 Windows wheel is published |
| `faiss-cpu` | `faiss_cpu-1.15.0-cp313-cp313-win_amd64.whl` | No |
| `pytrec-eval-terrier` | `pytrec_eval_terrier-0.5.10-cp313-cp313-win_amd64.whl` | **No** — cp313 Windows wheel is published |
| `sentence-transformers` | `6.0.0` (py3-none-any) | No |
| `datasets` | `5.0.1` (py3-none-any) | No |

Notes on the two that were expected to break:

- **`PyStemmer`** — the concern was real for `2.2.0.x` (the version floor in `requirements.txt`),
  but the resolver picks `3.1.0`, which publishes cp313 win_amd64 binaries. Keep the floor at
  `>=2.2.0`; it will land on 3.1.0 and need no Snowball C build.
- **`pytrec-eval-terrier`** — `0.5.10` publishes a cp313 win_amd64 wheel. **Do not** replace it
  with hand-computed nDCG/MRR. Hand-rolling those metrics is a correctness liability in a report
  that claims comparability with published BEIR numbers; `pytrec_eval` is the same C++ core as
  `trec_eval`, which is what the published numbers were produced with. It covers everything
  `evaluation.retrieval_metrics` asks for: `recall.2,5,10`, `ndcg_cut.10`, `recip_rank`.

**No substitutions are needed for any dependency.** The real dependency problems are §3 and §4,
which are about *which versions* resolve, not about whether they build.

---

## 2. BLOCKER — `xanhho/2WikiMultihopQA` cannot be loaded

**[VERIFIED]** The repo exists (HTTP 200) and is public and ungated, so a naive existence check
passes. But its file listing is:

```
.gitattributes, .gitignore, 2WikiMultihopQA.py, README.md,
convert_to_jsonl.py, convert_to_parquet.py, dev.parquet, test.parquet, train.parquet
```

`2WikiMultihopQA.py` is a **dataset loading script**. Two independent confirmations:

1. **[VERIFIED]** The HF dataset-viewer API refuses it:
   `{"error":"The dataset viewer doesn't support this dataset because it runs arbitrary python code."}`
2. **[VERIFIED]** Loading it on this machine with the installed `datasets` 4.5.0 fails for
   *every* split name I tried (`validation`, `dev`, `train`):

   ```
   RuntimeError: Dataset scripts are no longer supported, but found 2WikiMultihopQA.py
   ```

The presence of `dev.parquet` / `train.parquet` does **not** save it: `datasets` detects the
script first and hard-errors rather than falling back to the parquet files. `requirements.txt`
resolves to `datasets` 5.0.1, which is stricter still. There is no `trust_remote_code` escape
hatch — that parameter was removed along with script support.

### Fix — replace with `framolfese/2WikiMultihopQA`

**[VERIFIED]** I loaded it on this machine and it is a clean drop-in:

```yaml
  twowiki:
    enabled: true
    hf_id: framolfese/2WikiMultihopQA   # was: xanhho/2WikiMultihopQA (script loader, unloadable)
    hf_config: default
    split: validation
    eval_sample: 250
    strata: [type]
```

- Public, ungated, **no `.py` script**; data is `data/{train,validation,test}-*.parquet`.
- Splits: `train` (167,454), `validation` (**12,576**), `test` (12,576). The configured
  `split: validation` works as-is — no config change needed there.
- Columns **[VERIFIED]** by streaming one row:
  `id`, `question`, `answer`, `type`, `evidences`, `supporting_facts`, `context`
  - `evidences`: `list[list[str]]` — the gold Wikidata triples, e.g.
    `[['Polish-Russian War', 'director', 'Xawery Żuławski'], ['Xawery Żuławski', 'mother', 'Małgorzata Braunek']]`.
    This is exactly the ground truth the KG Navigator needs, and it is a **flat triple list**.
  - `supporting_facts`: `{title: list[str], sent_id: list[int64]}`
  - `context`: `{title: list[str], sentences: list[list[str]]}` — 10 paragraphs per question
- `type` strata **[VERIFIED]** over the first 500 validation rows:
  `compositional` 213, `bridge_comparison` 119, `comparison` 103, `inference` 65. All four
  values in the `config.yaml` comment are correct.
- **Encoding is clean.** I specifically checked for the double-encoded UTF-8 that plagues some
  2Wiki mirrors and there is none — `answer` is `'Małgorzata Braunek'` and evidences carry
  `'Xawery Żuławski'` correctly. (An apparent hit on `'Prince Nicolas, Duke of Ångermanland'`
  was a false positive from my own detector: that `Å` is genuine Swedish orthography.)

**Why this one over the alternatives** [VERIFIED — I compared all four]:
- `ohjoonhee/2WikiMultihopQA` is also parquet and script-free, but its schema differs: `_id`
  instead of `id`, `context.content` instead of `context.sentences`, and `evidences` is a
  dict-of-parallel-lists (`{fact, relation, entity}`) rather than triples. Usable, but it forces
  a second code path.
- **`framolfese`'s `supporting_facts` and `context` shapes are byte-identical to HotpotQA's**, so
  one loader function handles both datasets. That is worth real implementation time.
- `voidful/2WikiMultihopQA` is raw `.json` (no README/schema declaration); `kamelliao/2wikimultihopqa`
  and `thinkall/2WikiMultihopQA` are also script loaders and fail the same way `xanhho` does.

### HotpotQA — no change needed

**[VERIFIED]** `hotpotqa/hotpot_qa` is correct as configured.
- Public, ungated, `lastModified` 2025-08-11, **parquet only** (`distractor/train-00000-of-00002.parquet`,
  `distractor/validation-00000-of-00001.parquet`, …). No loading script — safe under `datasets>=4.0`.
- `hf_config: distractor` is valid. Splits for that config: `train` (90,447), `validation` (**7,405**) —
  which matches the "Full split is 7,405 questions" comment in `config.yaml` exactly.
- Columns **[VERIFIED]** by streaming one row:
  `id`, `question`, `answer`, `type`, `level`, `supporting_facts{title, sent_id}`,
  `context{title, sentences}`; 10 context paragraphs per question; `level` ∈ {easy, medium, hard},
  so `strata: [level]` is valid.

---

## 3. BLOCKER — the `cu124` PyTorch index cannot drive an RTX 5060

README step 3 says:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

This is wrong in two compounding ways, and — critically — **it fails silently**, which is the
worst possible failure mode.

**[VERIFIED]** The GPU on this machine:

```
device      : NVIDIA GeForce RTX 5060 Laptop GPU
capability  : (12, 0)          -> sm_120 (Blackwell)
total VRAM  : 7.96 GiB (8151 MiB per nvidia-smi)
```

**[VERIFIED]** The `cu124` index tops out at **torch 2.6.0+cu124** — that is the only version
published there. And it *does* publish `torch-2.6.0+cu124-cp313-cp313-win_amd64.whl`, so
**`pip install` will succeed**. The user gets a green install and a broken GPU.

**[VERIFIED]** The currently-installed `torch 2.8.0+cu128` on this machine works correctly:

```
torch.version.cuda        = 12.8
torch.cuda.is_available() = True
torch.cuda.get_arch_list() = ['sm_61','sm_70','sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']
```

Note `sm_120` in that list. **[INFERRED]** CUDA 12.8 is the first toolkit release to add the
Blackwell consumer SM 12.0 target; `cu124` and `cu126` builds are compiled with arch lists that
stop at `sm_90`, so a `cu124` wheel has no SASS for this device. At best a `cu124` build falls
back to PTX JIT (slow, unreliable); at worst it raises
`CUDA error: no kernel image is available for execution on the device`, and in the common
sentence-transformers path it degrades to CPU with no error at all — the 10× slowdown the brief
warns about.

**[VERIFIED]** Index availability, for choosing the replacement:

| Index | Max torch | sm_120? |
|---|---|---|
| `cu124` | 2.6.0+cu124 | No **[INFERRED]** |
| `cu126` | 2.13.0+cu126 | No **[INFERRED]** |
| `cu128` | **2.11.0+cu128** | **Yes [VERIFIED via installed 2.8.0+cu128 arch_list]** |
| `cu130` | 2.13.0+cu130 | Yes **[INFERRED]** |

### Fix

Replace README step 3 with `cu128`, and **move it to run BEFORE `pip install -r requirements.txt`**
(see §4 for why the ordering matters):

```bash
# 2a. PyTorch with CUDA 12.8 — REQUIRED for Blackwell / RTX 50-series (sm_120).
#     cu124 and cu126 wheels have no sm_120 kernels and fall back to CPU.
#     Install this BEFORE requirements.txt so pip does not pull the CPU wheel from PyPI.
pip install "torch>=2.7,<2.12" --index-url https://download.pytorch.org/whl/cu128

# 2b. Verify — this MUST print True and a list containing sm_120, or the GPU is not being used:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())"
```

`cu128` is the right choice over `cu130`: it is what is already installed and proven working on
this exact machine, and `torchvision` (0.26.0+cu128) and `torchaudio` are available there too.
Add the 2b verification line to the README verbatim — a one-line check is what turns a silent
10× slowdown into a loud failure.

---

## 4. HIGH — `requirements.txt` installs a CPU-only torch, then the README downgrades it

Two distinct problems, both from `requirements.txt` having no torch constraint and no index pin.

**Problem A — PyPI's Windows torch wheel is CPU-only. [VERIFIED]**
A full `pip install --dry-run --ignore-installed -r requirements.txt` resolves
`torch 2.13.0` from `files.pythonhosted.org`, pulled in transitively by `sentence-transformers`.
Its `requires_dist` shows every CUDA runtime dependency gated to Linux:

```
nvidia-cudnn-cu13==9.20.0.48; platform_system == "Linux"
nvidia-cusparselt-cu13==0.8.1; platform_system == "Linux"
nvidia-nccl-cu13==2.29.7;     platform_system == "Linux"
nvidia-nvshmem-cu13==3.4.5;   platform_system == "Linux"
```

There is no Windows CUDA variant on PyPI, and the wheel is 122.1 MB — a CUDA-bundled Windows
wheel is measured in gigabytes. So `pip install -r requirements.txt` gives you a **CPU-only
torch**. The README's ordering (requirements first, CUDA torch second) then has pip *downgrade*
2.13.0 → 2.6.0+cu124, which is both the wrong CUDA version (§3) and a large unintended version
jump for every other library that depends on torch.

**Fix:** install torch from the `cu128` index **first** (§3), then `pip install -r requirements.txt`;
pip will see torch already satisfied and leave it alone. Add an explanatory comment where
`requirements.txt` currently says `# torch: install separately …` so the ordering is not
accidentally "fixed" later by someone moving the line.

**Problem B — unpinned `>=` floors resolve to breaking majors. [VERIFIED]**
The same dry-run resolves this set:

```
datasets 5.0.1 · transformers 5.16.1 · sentence-transformers 6.0.0 · huggingface_hub 1.28.0
pandas 3.0.5 · numpy 2.5.2 · pyarrow 25.0.1 · scipy 1.18.1 · matplotlib 3.11.1
```

Every one of those is a *major* version ahead of what the file's own comments target. The sharpest
edge **[VERIFIED from wheel metadata]**: `sentence-transformers 6.0.0` declares

```
transformers<6.0.0,>=5.0.0
huggingface-hub<2.0.0,>=1.3.0
```

So `sentence-transformers>=3.0` silently drags in the **transformers 5.x major line**, directly
contradicting the `transformers>=4.44` line three rows above it in the same file. `pandas>=2.2`
likewise lands on **pandas 3.0.5**, a breaking major.

This is not hypothetical drift — it is what `pip install -r requirements.txt` does *today*. For a
graded project whose report must be reproducible from the repo, unbounded floors mean the numbers
in Chapter 4 cannot be regenerated later.

**Fix — add upper bounds.** This set is coherent, and the `transformers`/`sentence-transformers`
pair matches what is already installed and working on this machine (4.57.6 / 5.2.2):

```
# --- Core ---
numpy>=1.26,<3
pandas>=2.2,<3                 # pandas 3.0 is a breaking major
pyyaml>=6.0
tqdm>=4.66
orjson>=3.10

# --- Datasets ---
datasets>=4.0,<5               # >=4.0: script loaders already removed, so 2Wiki choice is honest
huggingface-hub>=0.34,<1.0     # hub 1.x is a breaking major

# --- Sparse retrieval ---
bm25s>=0.2.0,<1
PyStemmer>=2.2.0

# --- Dense retrieval / reranking ---
sentence-transformers>=3.0,<6  # 6.x hard-requires transformers>=5.0
transformers>=4.44,<5
faiss-cpu>=1.8.0,<2
# torch: install FIRST from the cu128 index (see README) — the PyPI Windows wheel is CPU-only

# --- Evaluation ---
pytrec-eval-terrier>=0.5.6
scipy>=1.13,<2

# --- Reporting / plots ---
matplotlib>=3.9,<4
```

Once M1 runs green, regenerate a fully-pinned `requirements.lock.txt` via `pip freeze` and ship
both files. The report should cite the lock.

---

## 5. HIGH — VRAM budget does not fit; concrete placement required

**[VERIFIED]** Live headroom on this machine, with a normal desktop session running:

```
nvidia-smi: total 8151 MiB | used 747 MiB | free 7153 MiB (6.99 GiB)
```

So the real budget is **~6.99 GiB**, not 8 GiB. Windows WDDM permanently reserves the difference.

**[VERIFIED]** Component sizes, from primary sources rather than estimates:

| Component | Source | Size |
|---|---|---|
| `qwen3:8b` Q4_K_M weights | Ollama registry manifest, model layer | 5.23 GB = **4.87 GiB** |
| KV cache @ `num_ctx: 8192`, fp16 | computed from `Qwen/Qwen3-8B/config.json` | **1.13 GiB** |
| llama.cpp compute buffers + CUDA context | **[INFERRED]** typical for an 8B | ~0.50 GiB |
| `BAAI/bge-small-en-v1.5` fp32 | safetensors metadata: 33,360,512 params | **0.13 GiB** |
| `cross-encoder/ms-marco-MiniLM-L6-v2` fp32 | safetensors metadata: 22,714,113 params | **0.09 GiB** |
| `cross-encoder/nli-deberta-v3-base` fp32 | safetensors metadata: 184,424,963 params | **0.69 GiB** |
| encoder activations @ `batch_size: 256` | **[INFERRED]** | ~1.00 GiB |
| **Total if all co-resident** | | **~8.4 GiB** |
| **Available** | | **6.99 GiB** |
| **Verdict** | | **over by ~1.4 GiB** |

The KV figure is exact, not a guess. **[VERIFIED]** From `Qwen/Qwen3-8B/config.json`:
36 layers, 8 KV heads (GQA), head_dim 128 → `2 × 8 × 128 × 2 bytes × 36 = 147,456 B/token = 144 KiB/token`.

| `num_ctx` | KV cache |
|---|---|
| 4096 | 0.56 GiB |
| **8192 (configured)** | **1.13 GiB** |
| 16384 | 2.25 GiB |
| 32768 | 4.50 GiB |

**They cannot coexist.** `qwen3:8b` alone at `num_ctx: 8192` needs ~6.5 GiB of the 6.99 GiB
available — it fits, but with only ~0.5 GiB to spare. Adding even the smallest cross-encoder
pushes Ollama into partial CPU offload, which is the 10× slowdown by a different route.

### Recommended placement

**Phase A — offline index build (`build_indexes.py`).** No LLM is loaded. Stop the Ollama service
first (`ollama stop qwen3:8b`, or just don't start it). Put `bge-small-en-v1.5` on the **GPU**;
peak ~1.5 GiB, enormous headroom, and this is the one throughput-bound step (66k passages).

**Phase B — online agent loop.** `qwen3:8b` gets the GPU **alone**. All three encoders go to
**CPU** (`SentenceTransformer(..., device="cpu")`, `CrossEncoder(..., device="cpu")`):

- Query encoding is one short string per sub-query — **[INFERRED]** <10 ms on CPU. GPU placement
  buys nothing here.
- `ms-marco-MiniLM-L6-v2` reranking `top_n: 50` short passages — **[INFERRED]** ~0.3–0.6 s on
  CPU. Acceptable against a 300 s `max_wall_clock_s` budget.
- `nli-deberta-v3-base` runs a handful of claims per question — **[INFERRED]** ~0.5–2 s on CPU.

**Two Ollama environment variables that matter more than anything above:**

```powershell
setx OLLAMA_NUM_PARALLEL 1      # CRITICAL
setx OLLAMA_KEEP_ALIVE 30m
```

**[INFERRED]** `OLLAMA_NUM_PARALLEL` allocates `num_ctx × num_parallel` of KV cache. If Ollama
picks 4 parallel slots, the configured 8192 context becomes 32768 tokens of KV = **4.50 GiB**
instead of 1.13 GiB, and `qwen3:8b` will *definitely* spill to CPU. Pinning it to 1 is the single
highest-value setting on an 8 GB card. `OLLAMA_KEEP_ALIVE` avoids a ~15 s model reload between
agent calls — with `max_llm_calls_per_question: 20`, reloads would dominate the latency metric
the report is trying to measure.

**Optional headroom lever.** If you want the reranker on GPU, drop `llm.options.num_ctx` from
8192 → **4096**. That frees 0.56 GiB and is low-risk: `top_k: 10` passages of ~100 tokens each,
plus instructions, is ~1.5k tokens — well inside 4096. The 8192 setting is only needed if you
later feed all `top_n: 50` reranked passages to the LLM.

**Two more sequencing notes:**
- `llm.fallback_model: qwen2.5:7b-instruct` is **[VERIFIED]** a real tag (4.68 GB). But if the
  fallback fires while `qwen3:8b` is still within its keep-alive window, Ollama will hold **both**
  (5.23 + 4.68 = 9.9 GB) and thrash. Whoever writes `llm.py` should explicitly unload the primary
  before falling back.
- Good design already present: all five agents in `llm.models` use the *same* `qwen3:8b`, so there
  is no per-agent model swapping. Keep it that way.
- **[VERIFIED]** System RAM available at check time was only 5.5 GiB of 31.6 GiB. If Ollama does
  offload layers, it needs several GiB of host RAM — close other applications before an eval run.

---

## 6. `src/agentic_ir/config.py` review

I constructed a `Config` and exercised every path. The dotted-path logic itself is **correct** —
present keys, absent keys with a default, absent keys without a default, mid-path non-dict nodes,
and `default=None` all behave properly. **[VERIFIED]**:

```
project.seed              -> 42
retrieval.rerank.top_n    -> 50
cfg.get("nope.nope", "D") -> 'D'
cfg.get("llm.host.port","D") -> 'D'      # mid-path non-dict, correctly falls to default
cfg.get("nope.nope")      -> KeyError: "'nope.nope' not found in ...config.yaml"
cfg.get("nope", None)     -> None        # sentinel correctly distinguishes None from missing
```

That last one is the bug most such helpers have, and this one gets it right — the `_MISSING`
sentinel is properly used. The real defects are elsewhere.

### 6.1 HIGH — `Config` is not read-only, and mutations leak across the whole process

The class docstring says *"Read-only view over the YAML config"*. It is not. `get()` returns live
references into the parsed dict, and `raw` hands out the dict itself. Combined with `@lru_cache`
on `load_config`, one agent mutating a returned sub-dict silently reconfigures every other agent.

**[VERIFIED]:**

```
d = cfg.get("retrieval.rerank")
cfg.get("retrieval.rerank.top_n")   -> 50
d["top_n"] = 9999
cfg.get("retrieval.rerank.top_n")   -> 9999      # mutated
load_config().get("retrieval.rerank.top_n") -> 9999   # ...for every other caller too
cfg.raw["project"]["seed"] = 1
cfg.get("project.seed")             -> 1
```

In a five-agent codebase this is a genuine correctness hazard, not style. The natural idiom
`opts = cfg.get("llm.options"); opts["temperature"] = 0.7` — exactly what a per-agent override
would look like — permanently changes the temperature for every agent, and the config values the
report cites will no longer match what actually ran.

**Fix.** Deep-freeze at load time so mutation fails loudly rather than corrupting silently. Convert
nested `dict`s to `MappingProxyType` and `list`s to tuples in `Config.__init__`, and return
`MappingProxyType(self._data)` from `raw`. If per-agent overrides are wanted, add an explicit
`Config.with_overrides(**kw)` that returns a *new* `Config`, so the intent is visible at the call site.

### 6.2 MEDIUM — `AGENTIC_IR_CONFIG` is silently ignored after the first call

`load_config` reads `os.environ` *inside* a function decorated with `@lru_cache`, but the cache key
is the `path` argument only. The env var is therefore honoured exactly once — on the first call —
and ignored forever after. **[VERIFIED]:**

```
load_config()                                   # caches the default config
os.environ["AGENTIC_IR_CONFIG"] = "Z:/does/not/exist.yaml"
load_config()  ->  returns the CACHED config from D:\IR_Project_D03000104\config\config.yaml
```

No error, no warning. A test that points the env var at a fixture config will silently run against
the production config and pass for the wrong reason. Given the docstring explicitly advertises this
override, it should work.

**Fix.** Resolve the path *outside* the cached function:

```python
def load_config(path: str | Path | None = None) -> Config:
    resolved = Path(path or os.environ.get("AGENTIC_IR_CONFIG", DEFAULT_CONFIG_PATH)).resolve()
    return _load_config_cached(resolved)

@lru_cache(maxsize=8)
def _load_config_cached(resolved: Path) -> Config: ...
```

Adding `.resolve()` also fixes 6.3.

### 6.3 MEDIUM — `lru_cache` yields four different `Config` objects for one file

`functools.lru_cache` keys on the arguments *as passed*; it does not fill in defaults or normalise
types. So four spellings of the same call produce four separate cache entries and four distinct
`Config` instances. **[VERIFIED]:**

```
load_config() is load_config(None)                  -> False
load_config() is load_config(DEFAULT_CONFIG_PATH)   -> False
load_config(Path(p)) is load_config(str(p))         -> False
cache_info: CacheInfo(hits=2, misses=4, maxsize=4, currsize=4)
```

Note `currsize=4, maxsize=4`: **a single config file has already filled the entire cache**, so the
next distinct call evicts an entry and re-parses. Worse, combined with §6.1, a mutation made
through one instance is invisible to the other three — the hardest class of bug to reproduce.

**Fix.** The `.resolve()` normalisation in 6.2 collapses all four spellings to one key. Raise
`maxsize` to 8 while you are there.

### 6.4 LOW — `resolve_path` raises an unhelpful `TypeError`

**[VERIFIED]:**

```
cfg.resolve_path("project.seed")       -> TypeError: argument should be a str or an os.PathLike
                                          object where __fspath__ returns a str, not 'int'
cfg.resolve_path("retrieval.rerank")   -> TypeError: ... not 'dict'
```

The message never names the offending key, so in a stack trace from deep inside a script this is a
scavenger hunt. (`resolve_path` on a genuinely missing key correctly raises `KeyError` naming the
key — that path is fine. `paths.raw` and `paths.results` resolve correctly to
`D:\IR_Project_D03000104\data\raw` and `D:\IR_Project_D03000104\results`.)

**Fix:**

```python
def resolve_path(self, dotted: str) -> Path:
    value = self.get(dotted)
    if not isinstance(value, str):
        raise TypeError(f"{dotted!r} must be a string path, got {type(value).__name__}: {value!r}")
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p
```

### 6.5 LOW — `yaml.safe_load` can return `None`

An empty or all-comments config file makes `yaml.safe_load` return `None`, producing
`Config(None, path)`. `get()` then reports every key as "not found in <path>" — technically true,
but it sends the reader hunting for a typo when the real problem is an unparseable file. `raw`
returns `None`, which will `TypeError` in any caller that iterates it.

**Fix:** after loading, `if not isinstance(data, dict): raise ValueError(f"{resolved} did not parse
to a mapping (got {type(data).__name__})")`.

### 6.6 LOW — `Paths.from_config` writes to disk on a read path

`from_config` calls `p.mkdir(parents=True, exist_ok=True)` for all four paths. A function that
looks like a config accessor creates four directories as a side effect, so any import, test, or
`--dry-run` that touches it litters the working tree. The docstring ("created on access") documents
the behaviour but does not justify it.

**Fix:** `Paths.from_config(cls, cfg, *, create: bool = False)` and have the entry-point scripts
call it with `create=True` explicitly.

---

## 7. MEDIUM — reranker model ID is a stale redirect

**[VERIFIED]** `config.yaml` sets `retrieval.rerank.model: cross-encoder/ms-marco-MiniLM-L-6-v2`.
That repo was renamed:

```
GET /api/models/cross-encoder/ms-marco-MiniLM-L-6-v2
  -> HTTP 307 Temporary Redirect
  -> Location: /api/models/cross-encoder/ms-marco-MiniLM-L6-v2   (note: L6, not L-6)
```

Canonical ID is **`cross-encoder/ms-marco-MiniLM-L6-v2`** (22,714,113 params, 87M downloads,
ungated). The old ID still resolves through the redirect today, so this is not currently breaking —
but it is a *temporary* redirect on a renamed repo, and pinning a project's report to a name that
HF has already moved off is an avoidable dependency. `cross-encoder/ms-marco-MiniLM-L-12-v2` also
307s, if the larger variant is considered later.

**Fix:** in `config.yaml`, set `model: cross-encoder/ms-marco-MiniLM-L6-v2`.

Also **[VERIFIED]** the other two model IDs are correct and ungated as written:
`BAAI/bge-small-en-v1.5` (33.4M params, `dim: 384` in config is correct) and
`cross-encoder/nli-deberta-v3-base` (184.4M params).

**Optional (ties to §5):** `cross-encoder/nli-deberta-v3-base` is 184M params / 738 MB — by far the
largest of the three encoders and the most expensive on CPU. **[VERIFIED]**
`cross-encoder/nli-distilroberta-base` exists and is ungated at ~82M params; it is a reasonable
swap if verifier latency becomes the bottleneck. Worth a sentence in the report either way, since
`verifier.method: nli_plus_llm` makes this model load-bearing for the central claim.

---

## 8. MEDIUM — Windows cp1252 will crash on dataset text

**[VERIFIED]** I hit this for real while inspecting the 2Wiki data on this machine. Python 3.13 on
Windows still defaults `sys.stdout` and `open()` to cp1252, and both directions fail on ordinary
dataset content:

```
print(row["answer"])                      # 'Małgorzata Braunek'
  -> UnicodeEncodeError: 'charmap' codec can't encode character '\u0142'
json.load(open("report.json"))
  -> UnicodeDecodeError: 'charmap' codec can't decode byte 0x8d
```

Both HotpotQA and 2WikiMultihopQA are full of Unicode entity names, so this will hit
`scripts/download_data.py`, every `print`/log of a retrieved passage, and — most damagingly —
the `orchestrator.trace: true` JSONL writer, which would corrupt or crash mid-run after a long
agentic evaluation.

**Fix (do all three):**
1. Add `encoding="utf-8"` to **every** `open()` in the codebase. `config.py` already does this
   correctly — make it the house rule.
2. Set `PYTHONUTF8=1` in the README setup steps (`setx PYTHONUTF8 1`), which flips Python to UTF-8
   mode process-wide.
3. When writing traces, use `json.dump(..., ensure_ascii=False)` with a UTF-8 file handle, or
   `orjson` (already a dependency) which emits UTF-8 bytes directly — write with `"wb"`.

---

## 9. MEDIUM — `retrieval.dense.batch_size: 256` is unvalidated

`bge-small-en-v1.5` has `max_seq_length` 512. **[INFERRED]** At batch 256 × 512 tokens the
attention workspace is large; it is fine with SDPA/flash kernels but can OOM with the eager
attention path, and it competes with anything else on the GPU. In practice HotpotQA passages are
single paragraphs (~100 tokens) and sentence-transformers sorts by length before batching, so the
effective sequence length is much shorter and 256 will probably work.

**Fix:** set `batch_size: 128` as the default — the throughput difference on 66k short passages is
small, and it removes the only unbounded memory knob in the offline path. Note in
`build_indexes.py` that it can be raised if the GPU is otherwise idle.

---

## 10. MEDIUM — README omits `pip install -e .`

`pyproject.toml` now exists with `[tool.setuptools.packages.find] where = ["src"]`, and
`[tool.pytest.ini_options] pythonpath = ["src"]` makes `pytest` work. But the README's usage
examples are:

```bash
python -m agentic_ir.cli ask "..."
python -m agentic_ir.eval.run_eval --dataset hotpotqa --config agentic_full
```

**[INFERRED]** Both fail with `ModuleNotFoundError: No module named 'agentic_ir'` when run from the
repo root, because `src/` is not on `sys.path` outside of pytest. (I confirmed the package imports
only when `PYTHONPATH=src` is set explicitly.)

**Fix:** add to README setup, after `pip install -r requirements.txt`:

```bash
pip install -e .          # puts agentic_ir on the path for `python -m agentic_ir.cli`
```

---

## 11. LOW — Ollama not installed

**[VERIFIED]** `ollama` is not on PATH (`command not found`), so nothing has been pulled yet.
**[VERIFIED]** against the Ollama registry, all configured tags are real:

| Tag | Download size |
|---|---|
| `qwen3:8b` | **5.23 GB** |
| `qwen2.5:7b-instruct` | 4.68 GB |

**[VERIFIED]** 119 GB free on `D:`, so the README's "~15 GB disk" estimate holds
(5.2 GB model + ~1.5 GB HotpotQA parquet + ~2.2 GB 2Wiki + indexes + torch/CUDA wheels ~3 GB).

One caveat: Ollama installs to `%LOCALAPPDATA%` on **`C:`** by default, and models go to
`%USERPROFILE%\.ollama\models`. If `C:` is tight, set `OLLAMA_MODELS` to a `D:` path *before*
pulling. Likewise the HF cache defaults to `C:\Users\<user>\.cache\huggingface` — set `HF_HOME`
to `D:` if needed. Worth two lines in the README given the project is on `D:`.

**[VERIFIED]** Unrelated but worth knowing: the HF cache emits a symlink warning on this machine
("your machine does not support them"). Downloads still work, just with extra disk use. Enabling
Windows Developer Mode removes it, or set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` to quiet it.

---

## 12. LOW — installing into the base conda environment would break other projects

**[VERIFIED]** `sys.prefix` is `C:\Users\pirgh\miniconda3` with no `CONDA_DEFAULT_ENV` set — this is
the **base** environment, and it already contains an unrelated audio/ML stack
(`pytorch-lightning`, `torchaudio`, `torch-audiomentations`, `torch_pitch_shift`,
`pytorch-metric-learning`, `sentencepiece`), plus `torch 2.8.0+cu128`, `transformers 4.57.6`,
`sentence-transformers 5.2.2`, `datasets 4.5.0`, `numpy 2.3.5`, `pandas 2.3.3`.

Running `pip install -r requirements.txt` here would upgrade torch to a CPU build (§4), transformers
to 5.x, pandas to 3.x and numpy to 2.5.2 — very likely breaking those other projects.

The README already says `python -m venv .venv`, which is correct. **Fix:** strengthen it to an
explicit warning, since the base env is currently active and the temptation to skip the venv is
real. Note that `python -m venv .venv` from a conda base gives a clean, isolated 3.13 env — that is
fine, and preferable here to `conda create` because all the pins above are pip-resolved.

---

## Verification checklist for whoever implements these

Run these after applying the fixes; each maps to a finding above.

```bash
# §3 — GPU is actually being used (must print True and a list containing 'sm_120')
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())"

# §2 — both datasets load and expose the expected splits/columns
python -c "from datasets import load_dataset as L; d=L('hotpotqa/hotpot_qa','distractor',split='validation',streaming=True); print(list(next(iter(d)).keys()))"
python -c "from datasets import load_dataset as L; d=L('framolfese/2WikiMultihopQA',split='validation',streaming=True); e=next(iter(d)); print(list(e.keys())); print(e['evidences'][:1])"

# §7 — canonical reranker ID resolves without a redirect
python -c "from huggingface_hub import model_info as m; print(m('cross-encoder/ms-marco-MiniLM-L6-v2').id)"

# §5 — after `ollama pull qwen3:8b`, confirm 100% GPU offload (must NOT say 'CPU')
ollama ps

# §8 — Unicode round-trips
python -c "print('Ma\u0142gorzata Braunek / Xawery \u017bu\u0142awski')"

# §10 — package is importable as installed
python -m agentic_ir.cli --help
```
