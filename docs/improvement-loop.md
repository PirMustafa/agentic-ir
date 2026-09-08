# The improvement loop

**For:** the coordinating agent (Opus) and the subagents it dispatches.
**Reference:** `docs/accuracy-plan.md` — the diagnosis (why accuracy is where
it is) and Round 2 (how to build `agentic_v2`). Read it first. This file is
the driver: it says what to try, in what order, how to measure each one, when
to keep it, and when to stop.
**Constraint from the owner:** local Ollama only. No paid APIs. 8 GB of VRAM.
`report/` is not modified by anything in this loop.

---

## 1. Targets, and the one that is not a target

| level | HotpotQA EM | 2Wiki EM | meaning |
|---|---|---|---|
| where it is | 0.432 | 0.224 | `agentic_full`, evaluated |
| **floor** | > 0.504 | > 0.264 | beats `self_ask`; the loop must reach this |
| **target** | ≥ 0.55 | ≥ 0.38 | reachable with a better generation step |
| stretch | ≥ 0.60 | ≥ 0.42 | needs both a stronger model and the pipeline gains |
| not a target | 0.80 | — | above the best published system on this benchmark; human agreement is ~0.83 |

Every number here is exact match on the frozen 250-question evaluation slice,
scored by the official HotpotQA normaliser. Human-judged accuracy runs ~8
points higher (21 of the current 142 errors are surface-form only); report it
beside EM if it helps, do not change the scorer.

## 2. What was learned in Round 1 that governs this loop

1. **Generation is the bottleneck.** 149/250 questions had every gold sentence
   in the evidence pool; 63 of them were still answered wrong. Two agents
   working from opposite directions both landed here. Evidence improvements
   do not convert into answers with the current generation step.
2. **The instrument must match the effect.** Run-to-run churn on identical
   code is 6–14% of answers. So:
   - an effect of **≥ 8 points** (a model change, thinking mode) can be
     screened on the 50-question calib slice;
   - an effect of **1–4 points** (a pipeline fix) cannot; it needs the
     250-question eval slice, and only as a one-shot confirmation.
   Do not screen small effects on calib. Half of Round 1's GPU time went to
   noise that way.
3. **Offline replay first, always, where the change is deterministic.** The
   traces record every retrieved passage and every model output; a scoring or
   selection rule can be replayed over 250 questions in seconds. Count
   breakages before recoveries.
4. **`agentic_no_planner` is the best configuration in the grid** (0.456 /
   0.324) and the base everything here builds on.

## 3. The loop protocol

One iteration = one hypothesis, one measurement, one decision.

```
for each hypothesis H in the backlog (§5), in order:
    1. Branch a fresh worktree from the current integration branch `v2`
       (so gains compound and the baseline is always "everything kept so far").
    2. Dispatch ONE subagent for H with the brief in §6. If a second
       hypothesis is file-disjoint and offline-only, a second subagent may run
       it in parallel; GPU work is serialised by the lock regardless.
    3. The subagent: offline replay if H is deterministic → n=50 screen if the
       expected effect is ≥ 8 points → one n=250 confirmation run per dataset.
       Never more than one eval-slice run per hypothesis: choosing on the eval
       slice is leakage.
    4. The coordinator VERIFIES before deciding: recompute at least one
       headline number from the subagent's scores.csv/traces.jsonl yourself.
       Round 1 found one agent's harness silently dropping 745 KG rows and
       another validating a trigger against a slice that could not expose its
       failure. Neither would have been caught by reading the report.
    5. Decide:
         KEEP  — paired 95% CI on EM excludes zero on at least one dataset,
                 OR the change is a correctness fix with 0 measured breakages
                 and no significant loss;
                 AND latency stays inside the budget in §4.
         DROP  — anything else. A null is a result; write it down and move on.
    6. On KEEP: merge into `v2`, record the new baseline, run the test suite.
    7. Stop when any of §7's conditions is met.
```

Keep a running log at `docs/improvement-loop.log.md` with one block per
iteration in the format of §8. That log is the deliverable, more than the
final number.

## 4. Budgets

- **GPU:** one workload at a time. `bash /d/ir_agent_worktrees/gpu-lock.sh
  acquire <name>` before every `run_eval`, `release` after, always.
- **Latency ceiling:** a full 250-question run must finish in **≤ 8 hours**
  per dataset, or the loop cannot iterate. Median per-question latency must
  stay **≤ 90 s**. A hypothesis that breaks this is DROPPED even if accurate,
  unless §5's latency valve recovers it.
- **RAM:** ~3 GB free of 32. One Python workload at a time. A segfault is
  memory pressure, not a bug; the DeBERTa cross-encoder also segfaults
  intermittently on load — wrap runs in a retry that uses `--resume`.
- **Disk:** 214 GB free. Pulling a 14B model (~9 GB) is fine.

## 5. The backlog, ranked

Each entry: what, why, expected size, how to measure, where, risk. Take them
in order; a later entry's expected size may change after an earlier KEEP.

### L0 — `agentic_v2` = one-node plan + 1A + 1C

Fully specified in `accuracy-plan.md` Round 2. Expected HotpotQA 0.47–0.50,
2Wiki 0.33–0.36. This establishes the `v2` branch and the baseline every
later loop is measured against. **Do this first; nothing else stacks without
it.** Confirmation: n=250, both datasets.

### L1 — thinking mode on, same model  ★ highest expected gain per hour

**What.** `llm.think: true`, remove the `/no_think` system suffix, raise
`llm.options.num_predict` 1024 → 2048 so reasoning cannot starve the answer.
The plumbing exists: `llm.py` passes `chat(think=...)` with version detection
and strips `<think>` blocks before parsing (Ollama 0.32.15 is installed;
`think` is supported).
**Why.** Thinking was disabled for *latency* — "10–20 s of pure waste per
call" in `config.yaml` — and never measured for *accuracy*. Qwen3's reasoning
gains come from exactly this mode. The 63 perfect-evidence failures are
reasoning errors about which span answers the question; that is what
thinking is for.
**Expected.** ≥ 8 points if it works at all. Screen on calib n=50 first
(this effect size is detectable there); confirm at n=250.
**Variants to try, in this order:** (a) thinking on the synthesiser only —
`llm.models` already routes per agent, add a per-agent `think` if needed;
(b) thinking on planner + synthesiser; (c) all agents. Stop at the first
variant that clears the screen.
**Risk.** Latency: +200–800 reasoning tokens per call at ~45 tok/s. Measure
it; if a 250-run exceeds 8 h, apply the latency valve below before dropping.
Parse failures: `<think>` handling exists but was tuned at 128 tokens; check
`parse_failures` stays ≈ 0 on the screen. Also check `think_chars` in the
trace is non-zero — if it is zero the flag did not reach the model.

### L2 — a larger local model with partial GPU offload

**What.** Pull `qwen3:14b` (Q4_K_M ≈ 9 GB). It does not fit 8 GB; Ollama
offloads the remainder to CPU automatically. Set `llm.default_model` and
every `llm.models.*` entry. Also drop `num_ctx` 8192 → 4096 (max observed
prompt ≈ 1,130 tokens; frees ~0.4 GB of KV cache for more GPU layers).
**Gate before any evaluation:** measure throughput on a 20-token prompt with
`ollama run` — if it is **< 15 tok/s, DROP** without running; the loop cannot
iterate at that speed. If the 14B is too slow, try `gemma3:12b` (≈ 8.1 GB,
may fit with num_ctx 4096) with the same gate.
**Why.** Every result is conditional on a 4-bit 8B model; the report says so.
The perfect-evidence failure rate is the direct measure of what a stronger
model buys.
**Expected.** 8–15 points. Screen on calib n=50; confirm at n=250.
**Interaction with L1:** measure L2 *with* whatever L1 decided. If L1 kept
thinking on, the 14B runs with thinking on.
**Risk.** Throughput and RAM. The 14B's CPU-offloaded layers compete with the
CPU-resident encoders and the NLI model. Check `nvidia-smi` and free RAM
before committing to a 250-run.

### L3 — `max_evidence` 20 → 30

**What.** One config value. **Why.** Offline (KG-inclusive replay), it is
the binding constraint on gold-pool completeness: 152 → 155 at docs=3.
**Expected.** 1–2 points. Too small for calib; **n=250 only**, and only once
L1/L2 are settled (a stronger generator uses extra evidence better). Check
that the synthesiser prompt stays inside `num_ctx`.

### L4 — self-consistency on the synthesis call

**What.** Sample the synthesiser 3× at `temperature: 0.7`, normalise each
`answer` with the official normaliser, take the majority; ties → the
lowest-temperature sample. Only the synthesis call; planner/verifier
unchanged. **Why.** The slot errors (24/250: gold inside the model's own
sentence, wrong span in `answer`) are exactly the kind of error voting
suppresses. **Expected.** 2–4 points. **n=250 only.** **Cost.** +2 model
calls per question; check the latency ceiling.

### L5 — 2B (anchor the original question) with a KG quota

Only if L3 kept and `_kg_evidence` has been given a reserved share of
`max_evidence` (see `accuracy-plan.md` Round 2, R2.6). Measured +5 gold
pools on calib when it does not evict the graph. **n=250 only.** Expected
1–3 points.

### Latency valve — `retrieval.rerank.top_n` 50 → 20

Not an accuracy hypothesis. Saves ~9 s per question (the reranker is 5.0 s
per call, ~3 calls per question). Apply it **only** to bring an otherwise
KEEP-worthy hypothesis (L1, L2, L4) back under the latency ceiling, and
measure nDCG@10 and Recall@10 on the same n=250 run to confirm the recall
cost is small. Never lower `rerank_margin_gate` — it is dead by construction
and that is load-bearing (`accuracy-plan.md` §4).

### Not in the backlog, and why

- The Verifier's confidence blend: at chance on every component, correction
  measured at nothing. Do not spend loops here.
- Prompt rewrites of the synthesiser: measured −1; lengthens answers.
- Any API model. Owner's constraint.
- Fine-tuning: not feasible on this hardware in this project's timeframe.

## 6. The subagent brief (template)

Every subagent gets this, filled in:

```
HYPOTHESIS: <L-number and one sentence>
WORKTREE: D:\ir_agent_worktrees\<name>, branched from `v2` at <sha>.
  cd there. export AGENTIC_IR_CONFIG=config/config.worktree.yaml PYTHONPATH=src
  (that config points data paths at D:\IR_Project_D03000104 and trace.dir at
  the worktree; create it by copying the pattern from an existing worktree).
FILES YOU OWN: <list>. Touch nothing else.
MEASURE: <offline replay | calib n=50 screen | n=250 confirm>, per §2.
  One eval-slice run per dataset, maximum. GPU lock around every run.
BASELINE: run it yourself under the current `v2` branch, same slice, same
  seed. Do not compare against numbers in the report or in this file.
REPORT (≤ 600 words): the §8 block, plus the diff summary and what you'd try
  next. Count breakages before recoveries. A null is a result.
DO NOT: edit report/; run git write commands; touch D:\IR_Project_D03000104;
  pass --split eval more than once per dataset; run two Python workloads
  at once.
```

Two subagents may run concurrently only when file-disjoint and at most one
needs the GPU at a time (the lock enforces the second part).

## 7. Stop conditions

Stop the loop and write the final summary when the first of these is true:

1. **Target reached** on both datasets, confirmed at n=250.
2. **Backlog exhausted** (L0–L5 all decided).
3. **Two consecutive DROPs after the floor is reached** — the remaining
   hypotheses are below the noise floor for this model.
4. **A confirmation run would exceed the latency ceiling** with the valve
   already applied.
5. **The owner says stop.**

Do not keep iterating past the backlog inventing hypotheses. If the target is
not reached when the backlog is exhausted, the honest conclusion is "this is
the ceiling of this model on this hardware", and that conclusion is worth
more than a fifth variant of a prompt.

## 8. Per-iteration log block

```
## Loop <n> — <L-number>: <hypothesis>
date · subagent · worktree · branch sha

baseline  (v2 @ <sha>)   HotpotQA EM x.xxx F1 x.xxx | 2Wiki EM x.xxx F1 x.xxx   run ids
treatment                HotpotQA EM x.xxx F1 x.xxx | 2Wiki EM x.xxx F1 x.xxx   run ids
paired ΔEM   HotpotQA +x.xxx [lo, hi] p=…   gained/lost g/l
             2Wiki    +x.xxx [lo, hi] p=…   gained/lost g/l
cost         median latency  before → after s;  model calls before → after
verified     <which number the coordinator recomputed, and it matched / did not>
decision     KEEP | DROP — one sentence
new baseline <v2 sha after merge, or unchanged>
```

## 9. Definition of done

- `docs/improvement-loop.log.md` has one block per iteration, including the
  DROPs.
- The `v2` branch contains every KEEP, tests pass, and `agentic_full` is
  byte-identical in behaviour (all fixes flag-gated, defaulted off).
- `python -m agentic_ir.cli tables` from `main` regenerates the report's
  tables byte-identically.
- A final summary states: the best configuration, its n=250 numbers on both
  datasets with paired intervals against `agentic_full`, `agentic_no_planner`
  and `self_ask`, its latency, and which stop condition ended the loop.
- If a model other than `qwen3:8b` was kept, the summary says so in the
  first line, because every number in the report is conditional on the
  model and a reader has to know the new ones are not.
