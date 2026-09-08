# Raising answer accuracy: an execution plan

# ROUND 2 — read this first

Round 1 of the plan below was executed by two isolated agents on 2026-09-07/08.
This section records what happened, corrects the plan where the execution
proved it wrong, and states exactly what to do next. The original plan follows
unchanged as the reference for the diagnosis; where the two disagree, this
section wins.

## R2.1 What Round 1 found

Four plan items were built and measured. **None moved exact match.**

| item | built | offline (250 traces) | calib n=50 | verdict |
|---|---|---|---|---|
| 1A reconcile `answer` vs `answer_sentence` | yes, 26 tests | **6 recovered, 0 broken** across 300 q | unresolved (see R2.2) | defensible, unproven |
| 1B yes/no answer-type guard | yes | — | +1 / −1, exactly neutral | measured null |
| 1C rank-major evidence key | yes, 5 tests | **+3 complete gold pools**, 1 lost | SP-F1 +0.065 [−0.013, +0.148] | real bug fix, unproven on EM |
| 2A sentence-level selection | yes, **disabled** | **net 0** once KG counted | EM −0.04, p=0.69 | rejected until KG quota exists |

Two findings matter more than the table:

1. **Generation is the bottleneck, confirmed a second time from the other
   direction.** Every EM loss in the evidence agent's runs was a span-selection
   error with the gold evidence present — 1A's territory. Improving evidence
   does not convert to answers with this model.
2. **2A without 1C loses 36 questions.** The plan's ordering was right and is
   now measured, not argued.

## R2.2 Corrections to the plan below

- **§5's measurement protocol is underpowered by an order of magnitude.**
  The effects available are 1–3%. Run-to-run answer churn on byte-identical
  code was measured at **6–14% of questions** (3–7 of 50). On n=50, a
  2-question difference — every result in Round 1 — is indistinguishable from
  noise. +3/250 has an expected count of 0.6 questions on the calib slice.
  **Do not measure anything else on calib.** Use offline replay over the 250
  traces to choose, and the 250-question eval slice under a new config name to
  confirm.
- **§1's 149 complete-gold-pool baseline is correct.** An agent reported 140
  and claimed byte-exact reproduction; 140 is the pool with the 745 KG
  evidence rows excluded. Any offline harness that rebuilds the pool from
  `retrieved` alone is KG-blind and will overstate widening gains — the
  evidence agent's first "+9" was exactly this and is really 0.
- **Replays must count breakages first.** Both agents' first replays counted
  recoveries and validated against a set that could not expose their failure
  mode (the eval slice contains zero instances of "a correct answer absent
  from its own sentence"). A trigger that recovers 15 and breaks 10 is worse
  than nothing. Replay over eval250 **and** the calib traces together.
- **§2A's "10-sentence document eats half the budget" is largely
  counterfactual.** Retrieved passages average 3.9 sentences; 1.6% have ≥10.
  The binding constraint is `max_evidence`, not document coverage
  (20→30 gives 152→155 offline at docs=3). Per-document caps measured worse.

## R2.3 The one lever Round 1 did not touch, and it is the biggest

`agentic_no_planner` is already measured at n=250 on both datasets:

| | EM | F1 | vs `agentic_full` | vs `self_ask` |
|---|---|---|---|---|
| HotpotQA | **0.456** | 0.543 | +0.033 (n.s.) | −0.060 |
| 2Wiki | **0.324** | 0.388 | **+0.121, p<0.001** | **+0.078, significant** |

It is the only configuration in the grid that beats a baseline on answer
quality, at a third of the model calls and an eighth of the latency. It was
Tier 3 in the plan because it is a *different configuration* rather than a
fix. That is the right reason to keep it out of `agentic_full` and the wrong
reason not to build on it.

## R2.4 What to do: build `agentic_v2`

**`agentic_v2` = the one-node plan + 1A + 1C.** Nothing else.

### Step 1 — gate both fixes behind config, defaulted OFF

`agentic_full` must keep behaving exactly as it did, or the report's 18 runs
stop being reproducible from the code. Follow the precedent already in the
tree: `agents.verifier.entail_target` ships defaulted to the evaluated
behaviour with the correction selectable. Do the same:

```yaml
agents:
  synthesizer:
    reconcile_answer: false        # 1A; true in agentic_v2
  verifier:
    evidence_ranking: lexical_major # 1C; rank_major in agentic_v2
```

Then `run_eval.py::ABLATION_OVERRIDES` gets one new entry:

```python
"agentic_v2": {
    "agents.planner.template_shortcut": False,   # = agentic_no_planner
    "agents.synthesizer.reconcile_answer": True,   # 1A
    "agents.verifier.evidence_ranking": "rank_major",  # 1C
},
```

**Known snag:** `run_eval.py`'s argparse pins `--config` to `CONFIGURATIONS`,
and `tables.py` renders every name in `CONFIGURATIONS` into the report's
tables. Adding `agentic_v2` to that tuple would put a row in the report.
Either (a) accept `--config` from `evaluation.configurations` in the config
file — `configurations(cfg)` already exists for this — and list `agentic_v2`
only in your variant config, or (b) add it to `CONFIGURATIONS` and exclude it
in `tables.py::discover_runs`. (a) is smaller. `tests/test_report_integrity.py`
pins the nine-config grid; if a test fails, it is protecting the report —
read it before changing it.

### Step 2 — harvest the diffs; do not rewrite them

Both changes exist, are tested, and carry docstrings that record why they
look the way they do. Take them as they are:

```
D:\ir_agent_worktrees\answer   (branch exec-answer)
    src/agentic_ir/agents/synthesizer.py   reconcile_answer + infer_answer_type guard
    src/agentic_ir/types.py                one Origin literal ("reconciled")
    tests/test_synthesizer_reconcile.py    26 tests, untracked — copy it
    → wrap the reconcile call in the new config flag; keep the guard (1B) unconditional,
      it measured neutral and is a correctness fix

D:\ir_agent_worktrees\pool     (branch exec-pool)
    src/agentic_ir/orchestrator.py         build_evidence rank-major scoring, evidence_docs knob
    config/config.yaml                     evidence_docs: 3 declared
    tests/test_orchestrator.py             +5 tests
    → the scoring change is unconditional there; wrap it in evidence_ranking.
      Leave evidence_docs at 3. Do NOT enable 2A.
```

`git -C D:\ir_agent_worktrees\answer diff` and the same for `pool` show
exactly what changed. Both worktrees also have `results/runs/` with the calib
runs, for reference only.

### Step 3 — replay offline before any GPU

Over `results/runs/agentic_full_hotpotqa_20260904T221533Z/traces.jsonl` (250)
and `…/agentic_no_planner_hotpotqa_20260905T224906Z/traces.jsonl` (250):
apply 1A's rule to each record's (`question`, `final_answer`,
`answer_sentence`) and count **broken first, recovered second**. The answer
agent's harness for this is in its worktree; the bar is 0 broken. Include the
KG rows if you replay 1C.

### Step 4 — one run per dataset, n=250, new config, separate directory

```bash
# variant config: copy config.yaml, set trace.dir to a directory that is NOT
# results/runs, list agentic_v2 under evaluation.configurations
export AGENTIC_IR_CONFIG=config/config.v2.yaml PYTHONPATH=src
bash /d/ir_agent_worktrees/gpu-lock.sh acquire v2
python -m agentic_ir.eval.run_eval --dataset hotpotqa --config agentic_v2 --split eval --no-progress
python -m agentic_ir.eval.run_eval --dataset twowiki  --config agentic_v2 --split eval --no-progress
bash /d/ir_agent_worktrees/gpu-lock.sh release v2
```

`--split eval` is correct **here and only here**: the changes were chosen on
offline replay and the calib slice, and this is the confirmation, not the
selection. It is ~2 GPU hours per dataset. `run_eval` now pins
`PYTHONHASHSEED=42`; check `meta.json` says so.

The GPU lock is mandatory — two concurrent runs segfaulted last round. The
machine has ~3 GB RAM free; one Python workload at a time. The DeBERTa
cross-encoder load segfaults intermittently on this machine before any
question runs; wrap the run in a retry that resumes (`--resume <run-id>`).

### Step 5 — compare against the right baseline

`agentic_v2` is `agentic_no_planner` plus two fixes. The comparison that
isolates the fixes is **v2 vs `agentic_no_planner`**, paired, on the same 250.
Also report v2 vs `agentic_full` and v2 vs `self_ask` because those are the
numbers a reader will ask for. Use `eval/replication.py` for the paired view
of what flipped, and the paired bootstrap in `eval/tables.py` for intervals.

Expected, honestly: HotpotQA **0.47–0.50**, 2Wiki **0.33–0.36**. Clearing
`self_ask` on 2Wiki is likely; on HotpotQA it is a coin flip. Anything above
0.55 on HotpotQA would be surprising and should be re-checked for leakage
before being believed.

## R2.5 Definition of done

- `agentic_full`'s 18 evaluated runs are untouched and `python -m
  agentic_ir.cli tables` regenerates byte-identical tables.
- Both fixes are behind flags that default to the evaluated behaviour, with
  the flag values quoted in their docstrings.
- Offline replay of 1A over 500 traces: breakages reported, and 0.
- Two n=250 runs of `agentic_v2` in a directory `discover_runs` cannot see,
  `pythonhashseed: '42'` in both `meta.json`.
- A paired comparison v2 vs `agentic_no_planner`, v2 vs `agentic_full`, v2 vs
  `self_ask`: EM, F1, gained/lost, latency, citation grounding.
- Nothing under `report/` modified. No commit to `main` without being asked.
- Nulls written down as nulls.

## R2.6 What not to spend time on

- 2A, until `_kg_evidence` has a reserved quota instead of the remainder.
- The confidence blend. At chance on every component; the correction is
  measured at nothing.
- Any calib-slice measurement. See R2.2.
- Surface-form scoring (21 questions). That is a metric property, not a
  system defect; report human-judged accuracy beside EM if it matters, do not
  "fix" the normaliser.
- 95% exact match. Human agreement on this benchmark is ~83%; the best
  published systems are ~70–75%. It is not a target.

---

**For:** whoever executes this next (a coding agent or a person).
**Status of the codebase:** complete and committed at `03b8550`. The report is
finished and is **out of scope** — nothing in this plan touches `report/`.
**Goal:** move `agentic_full` exact match from 0.432 (HotpotQA) / 0.224 (2Wiki)
past the `self_ask` baseline at 0.504 / 0.264, without invalidating the evaluated
grid.

Everything below is grounded in measurements already taken. The numbers are
reproducible from `results/runs/` and the two agent worktrees named in §6.
Where a step is speculative it says so.

---

## 0. Rules of engagement

These are not style preferences. Each one prevents a specific way of producing
a number that looks like an improvement and is not.

1. **Measure on the calibration slice first.** `--split calib --size 50`. It is
   disjoint from the 250-question evaluation slice by construction. Tuning
   anything on `--split eval` is leakage and invalidates all 18 evaluated runs.
   Promote to the eval slice only once a change is frozen.
2. **Write runs somewhere other than `results/runs/`.**
   `tables.py::discover_runs` takes the lexicographically greatest run id
   matching `<config>_<dataset>_`. A new `agentic_full_hotpotqa_*` run written
   there gets a newer timestamp and **silently replaces the reported run in
   every table**. Use a config variant with a different `trace.dir`
   (`config/config.answer_bearing.yaml` shows the pattern) or `--run-id` with a
   prefix that does not match a configuration name.
3. **Pin the hash seed and re-baseline.** `run_eval.py` now pins
   `PYTHONHASHSEED=42` on launch. The evaluated grid was **not** run under it
   (`meta.json` records `pythonhashseed: null`). Re-running the identical
   configuration moved 15/250 answers and 0.014 F1. So: run your own baseline
   under the pinned seed on the same slice, and compare against *that*, never
   against the numbers in the report.
4. **One change at a time, then combine.** Both agents in §6 found their change
   was a null in isolation; the plan below expects gains to come from
   combinations, and you cannot attribute a combined gain without the singles.
5. **One GPU workload at a time.** The machine has ~3 GB of RAM free of 32 and
   an 8 GB card. Two concurrent `run_eval` processes killed one of them with a
   segfault. Check for a live `run_eval` before launching. A segfault means
   memory, not your change.
6. **Report nulls.** A measured null is a result. Do not revert on judgement;
   revert on measurement.
7. **No git writes to `main`** unless the user asks. Work on a branch or a
   worktree; the diff is the deliverable.

---

## 1. Where the accuracy goes

`agentic_full` on HotpotQA, 250 questions. All counts are from
`results/runs/agentic_full_hotpotqa_20260904T221533Z`.

```
250 questions
├─ 108 correct (EM 0.432)
└─ 142 wrong
    ├─  21  surface form only: pred ⊇ gold or gold ⊇ pred
    │       ("King George I" vs "George I"). A human marks these right.
    │       Exact-match does not. Human-judged accuracy ≈ 0.516.
    ├─  24  gold answer is INSIDE the model's own answer_sentence,
    │       but `answer` is a different span. Pure slot error.
    │       ("PRED '1939'  SENT 'Krzysztof Zanussi was born in 1939.'  GOLD 'Krzysztof Zanussi'")
    │       Ceiling if all recovered: EM 0.528.
    ├─   5  bare yes/no emitted for a question that is not yes/no
    │       (subset of the 24; the deterministically detectable part)
    └─  rest: genuine reasoning failures, or evidence never reached the model
```

Evidence side, same run:

```
149 / 250  every gold sentence present in the final evidence pool
            └─ 63 of these STILL answered wrong  → generation, not retrieval
101 / 250  at least one gold sentence missing
            ├─ 74  never retrieved by ANY sub-query   → decomposition dilution
            ├─ 22  retrieved at rank 3-9, dropped by evidence_passages: 3
            └─  8  document in pool, sentence cut by max_evidence: 20
            (re-plan overwrite: 0 — it corrupts state but loses no evidence)
All 500 gold documents are in the index. Every loss is ranking, never coverage.
```

Two structural facts that shape the plan:

- **Decomposition is net harmful.** `agentic_no_planner` beats `agentic_full`
  by +0.033 on HotpotQA (n.s.) and **+0.121 on 2Wiki (p < 0.001)**, at a third
  of the model calls. The one-node system already beats every baseline on 2Wiki.
- **The confidence signal is at chance** (AUC 0.538 / 0.513, every component
  independently at chance). The re-plan loop picks questions at random with
  respect to correctness. Do not build on it.

---

## 2. The plan, ranked by expected value per hour

### Tier 1 — deterministic, no extra model calls, measured headroom

#### 1A. Reconcile `answer` against `answer_sentence`  ★ highest value

**Headroom:** 24/250 = 9.6 points ceiling. Realistic 4–7.
**Where:** post-processing in `src/agentic_ir/agents/synthesizer.py` after the
JSON is parsed, before the candidate is built. No prompt change.
**What:** the model writes a sentence and then fills `answer` with the wrong
constituent of it. Re-derive the span from the sentence using the question's
interrogative when the two disagree:

```
if answer is bare yes/no  and  question is not a yes/no question:
    → answer must be an entity/value from answer_sentence
if answer (normalised) is not a substring of answer_sentence (normalised):
    → the model contradicted itself; prefer the sentence
selection rule by interrogative:
    who / which person / what band … → the capitalised multi-word span
                                       that is not already in the question
    when / what year                 → the date / 4-digit year
    how many / how much              → the number (+ following noun if present)
    where                            → the location-like span
```

Use `src/agentic_ir/eval/metrics.py::normalize_answer` for the substring test so
the check matches how EM will score it. Only fire when the question and the
sentence disagree with the emitted `answer`; when they agree, do nothing.

**Measure:** the 24 questions are identifiable offline — replay the rule over
`traces.jsonl` and count recoveries **before** spending GPU. Then one calib run.
**Risk:** over-firing on correct answers. Mitigate by logging every
reconciliation into the trace (`origin: "reconciled"`) so the false-positive
rate is countable.

#### 1B. Never let a plan node flip the answer type into or out of yes/no

**Headroom:** measured +1 on calib (fired once, turned `'no'` into the gold
`'Fort Worth'`). Across the eval run, the plan flips the yes/no bit on 9
questions → EM 0.333 vs 0.444 where plan and surface agree.
**Where:** `synthesizer.py::infer_answer_type`. The last plan node's
`answer_type` describes *its own* sub-answer; a bridge chain often ends in a
yes/no verification node ("Is {{q2.answer}} a retired soccer player?") whose type
leaks into the final answer.
**Status:** already implemented and measured in the `agent-gen` worktree (§6).
Harvest the code guard; **discard the prompt rewrite from the same diff** — it
measured −1 by making answers longer (`'Topeka'` → `'Topeka, the state capital
of Kansas'`).

#### 1C. Fix the evidence ranking key

**Headroom:** invisible today, load-bearing the moment the pool widens (2A).
**Where:** `src/agentic_ir/orchestrator.py::build_evidence`. The score is
`1/(rrf_k + rank + 1) + 0.05 * jaccard(sentence, query)`. Verified: across the
three ranks the code uses, the rank term spans **0.00052** and the jaccard term
spans **0.05** — 96×. Lexical overlap is the primary sort key and the
cross-encoder's ordering is the tiebreaker. Backwards.
**What:** make rank dominant; make jaccard a genuine tiebreaker
(e.g. `score = (n_ranks - rank) + 0.001 * jaccard`, or sort by `(rank, -jaccard)`).
**Measure:** replay over all 250 traces — count gold sentences that move into
the top-20 — before any GPU. Evidence agent's simulation: worth +11 complete
pools at `evidence_passages: 10`.

### Tier 2 — evidence pipeline, structural

#### 2A. Select evidence by sentence, not by document

**Headroom:** simulation 169 → 183 complete gold pools / 250 (+14). Converts at
roughly the current 40% → +3–5 EM.
**Where:** `orchestrator.py::build_evidence`. Today it takes **every sentence**
of the top-3 documents per sub-query. A 10-sentence document eats half the
20-sentence budget regardless of relevance. That is why raising
`evidence_passages` alone **hurts** (140 → 132 → 118 complete pools at 3/5/10):
more documents flood a fixed cap.
**What:** score sentences individually across the top-10 documents (rank-major
per 1C, then lexical), keep the best 20. Decouples document coverage from the
sentence budget.
**Prerequisite:** 1C, or the widened pool sorts by the wrong key.

#### 2B. Anchor the original question as an extra retrieval node

**Headroom:** measured on calib — gold pool complete 27 → 32 (+5, 0 lost,
monotone), nDCG@10 0.785 → 0.822. **EM null in isolation** (0.380 → 0.400,
p = 1.00). Latency +67% (16.4 → 27.4 s).
**Where:** implemented in the `agent-evidence` worktree as
`orchestrator.py::_anchor_original_question`, reserved id `q0`, config flag
`agents.retriever.anchor_original_question`.
**Why null alone:** the recovered documents get crowded out of the 20-sentence
cap by whole-document ingestion. Expect this to convert **only after 2A**.
**Cost:** the latency is the reranker running once more (5 s/call). Pair with 3B
or it is not worth shipping.

### Tier 3 — architecture

#### 3A. Default to the one-node plan; decompose only when it pays

**Headroom:** the largest measured effect in the project. On 2Wiki, deleting the
Planner is +10 EM points and clears every baseline.
**Where:** `src/agentic_ir/agents/planner.py`. Today the Planner always
decomposes. Invert the default: issue the verbatim question as a single node,
and decompose **only** when (a) the question carries an explicit bridge marker
(a relative clause, "the X that/who/whose …") **and** (b) single-query retrieval
is weak (fused top-1 margin below a threshold — note the existing
`rerank_margin_gate` cannot be reused as-is; it is dead by construction, see §4).
**Measure:** this is a configuration ablation, so it goes on calib first, then
becomes a new config name (`agentic_adaptive`) rather than a change to
`agentic_full` — the report's `agentic_full` row must keep meaning what it meant.

#### 3B. Cut rerank depth

`retrieval.rerank.top_n: 50 → 20`. Saves ~9 s per question (the reranker is
5.0 s median per call, ~3 calls per question). Only `top_k: 10` survives anyway.
Small recall risk; measure nDCG@10 on calib. Needed to make 2B affordable.

### Tier 4 — deprioritise

The Verifier's confidence blend. Every component is at chance; the
`answer_bearing` correction was implemented and measured at 0.538 → 0.541 /
0.513 → 0.499 — nothing. Any further work here should replace the signal, not
patch it: a plausible replacement is "does the answer occur in a cited premise,
and does its type match the interrogative" — which is 1A's machinery reused as a
score. Low expected value; do last if at all.

---

## 3. Expected outcome, honestly

| step | HotpotQA EM | 2Wiki EM | confidence |
|---|---|---|---|
| today | 0.432 | 0.224 | measured |
| + 1A + 1B | 0.47–0.50 | 0.25–0.28 | ceiling is 0.528; realistic is below it |
| + 1C + 2A (+2B) | 0.49–0.53 | 0.28–0.32 | simulation-backed, not yet run |
| + 3A | 0.50–0.54 | **0.34–0.40** | 3A's 2Wiki gain is already measured at +0.10 |

Clearing `self_ask` (0.504 / 0.264) on both datasets is plausible. Reaching 0.60
is not, with this model. `synthesis_error` is the largest error bucket and the
model is decode-bound; the remaining headroom after this plan is the model.

---

## 4. Dead ends — measured, do not retry

- **Raising `evidence_passages` alone.** Hurts (140 → 132 → 118). Fix 2A first.
- **Lowering `rerank_margin_gate` so it fires.** It is dead by construction
  (RRF top-1/top-2 margins are median 0.0006 against a 0.15 threshold) and that
  is *fortunate*: always reranking finds +40 gold documents at top-3. Leave it.
- **The `answer_bearing` entailment veto.** Implemented, replayed, null.
- **Rewriting the synthesiser prompt for "verbatim spans".** Measured −1; it
  lengthens answers. 1A does the same job deterministically.
- **A bigger model.** Would help — 63 failures with perfect evidence say so —
  but it invalidates every number in the grid. If tried, it is a separate
  single-row experiment, not a change to `agentic_full`.
- **Fixing the re-plan state overwrite for accuracy.** It is a real defect but
  loses zero gold evidence. Worth fixing for trace hygiene; do not expect EM.

---

## 5. Measurement protocol

```bash
# baseline, under the now-pinned hash seed, on calib, in a separate runs dir
AGENTIC_IR_CONFIG=config/config.<yours>.yaml PYTHONPATH=src \
python -m agentic_ir.eval.run_eval --dataset hotpotqa --config agentic_full \
  --split calib --size 50 --run-id base_calib_<stamp> --no-progress

# one change → one run, same flags, new run-id
# compare with eval/replication.py for a paired view of what flipped:
python -m agentic_ir.eval.replication <base_run_dir> <treatment_run_dir>
```

Report per change: EM/F1 before and after, questions gained/lost, gold-pool
completeness before and after, latency before and after. Promote to
`--split eval` only for the frozen combination, into a **new config name**.

Offline replays (no GPU) exist for 1A, 1C and 2A — do them first. A rule that
recovers the 24 slot errors on the existing traces is worth more than a calib
run that guesses.

---

## 6. Work already done — harvest, don't redo

Two isolated worktrees hold measured diffs. Read them before writing anything:

```
D:\ir_agent_worktrees\gen        branch agent-gen
    synthesizer.py::infer_answer_type yes/no guard   → KEEP (1B)
    prompts/synth.answer.v1.txt rewrite              → DISCARD (measured −1)
    runs: results/runs/gen_calib_baseline, gen_calib_synthfix

D:\ir_agent_worktrees\evidence   branch agent-evidence
    orchestrator.py::_anchor_original_question       → KEEP, gate behind 2A (2B)
    config flag agents.retriever.anchor_original_question
    3 new tests
    runs: results/runs/evidence_calib_before, evidence_calib_after
```

`git diff main..agent-gen` and `git diff main..agent-evidence` from the main
checkout show both. Each worktree has `config/config.worktree.yaml` pointing at
the shared corpus — reuse that pattern for your own runs.

Calibration baseline both agents measured, n = 50: **EM 0.380, F1 0.446**.

---

## 7. Definition of done

- Each of 1A, 1B, 1C, 2A has a measured calib result, singly.
- The frozen combination has one eval-slice run under a **new config name**,
  in a separate runs directory, with `pythonhashseed: '42'` in its `meta.json`.
- `agentic_full`'s reported rows are byte-identical to before.
- A paired comparison against the re-baselined `agentic_full` (also under the
  pinned seed) states EM, F1, gained/lost, and latency.
- Nulls are written down as nulls.
