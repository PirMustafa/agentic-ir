# Improvement loop — running log

Driver: `docs/improvement-loop.md`. Diagnosis: `docs/accuracy-plan.md`.
One block per iteration, DROPs included. Numbers are exact match on the frozen
slices, official HotpotQA normaliser, `PYTHONHASHSEED=42`.

## Loop 0 — setup

2026-09-08 · coordinator

- Integration branch `v2` created at `03b8550` (= `main`). Every KEEP merges
  here; `main` is untouched.
- Worktrees `L0-build` and `L1-think` cut from `v2`, each with a
  `config.worktree.yaml` pointing corpus/indexes at the shared checkout and
  `trace.dir` at the worktree, so no run can be picked up by
  `tables.py::discover_runs`.
- Starting state: GPU idle (251 MiB / 8151), RAM 15.5 GB free of 31.6, lock
  free, `qwen3:8b` the only installed model.

**Ordering deviation from §5, and why.** The backlog runs L0 → L1. L0's
confirmation is two n=250 runs (~4-6 h each); L1's screen is two n=50 runs
(~1 h total). L1 is a global generation-mode change, so if it is KEPT the L0
confirmation would have to be re-run under it — paying the expensive
measurement twice. So L1's *screen* runs in parallel with L0's *build and
offline replay*, and L0's n=250 confirmation is held until L1 has decided.
This respects §3 (parallel dispatch permitted when file-disjoint, GPU
serialised by the lock) and §5's own instruction that L2 must be measured
under L1's decision. No hypothesis is decided out of order; only the cheap
screen is brought forward.

Baseline for the loop, from the evaluated grid (`main`):

| config | HotpotQA EM / F1 | 2Wiki EM / F1 |
|---|---|---|
| `agentic_full` | 0.432 / 0.510 | 0.224 / 0.267 |
| `agentic_no_planner` (v2 base) | 0.456 / 0.543 | 0.324 / 0.388 |
| `self_ask` (floor to beat) | 0.504 / 0.603 | 0.264 / 0.310 |

### Loop 1 (L1) — interim verifications by the coordinator

**Variant space collapses from three to two — confirmed.** The subagent
reported that under `agentic_no_planner` the Planner issues no LLM calls, so
§5's variant (b) "planner + synthesiser" is identical to variant (a)
"synthesiser only". Recounted from the evaluated traces:

    agentic_no_planner/hotpotqa  n=250  {synthesizer: 250, verifier: 88}
    agentic_full/hotpotqa        n=250  {planner: 456, synthesizer: 452,
                                         verifier: 140, extractor: 17}

Correct. Only `{synthesizer}` and `{synthesizer, verifier}` are
distinguishable, which fits the 6-run budget as baseline x2 + two variants x2.
(Side note worth keeping: in `agentic_full` the Planner is the single largest
consumer of model calls, 456 of 1065 — and it is the component the ablation
grid says is harmful.)

**A defect in the brief I wrote, corrected by the subagent.** I located
`SYSTEM_PROMPT` in `llm.py`; it is in `agents/base.py`, outside the files L1
was given. Rather than reach outside its ownership it implemented the
`/no_think` strip inside `llm.py`, applied only when thinking is enabled for
that call, so with `llm.think: false` outgoing messages stay byte-identical to
the evaluated grid. That is the right call and preserves the L0 invariant.

**Think plumbing fires.** Live one-call probe on `qwen3:8b`:
`thinking_chars=764` with thinking on, `0` with it off, structured output
parsing cleanly both ways. The §5 "if `think_chars` is zero you measured
nothing" trap is cleared.

## Loop 1 — L0: build `agentic_v2` (one-node plan + 1A + 1C)

2026-09-08 · subagent `L0-build` · worktree `D:\ir_agent_worktrees\L0-build`
· merged to `v2` @ `1443038`

**Measurement: offline replay only. No GPU run.** See the decision below for
why the n=250 confirmation was not spent.

    1A over agentic_full's 250 traces        8 fires,  0 broken,  6 recovered
    1A over agentic_no_planner's 250 traces  1 fire,   0 broken,  0 recovered
    1C, KG-inclusive, agentic_full base      149 logged -> 149 replayed -> 152 rank-major (+4, -1)
    1C on agentic_no_planner base            159 -> 158 (0 gained, 1 lost)

**verified** — coordinator re-ran 1A independently against both trace sets,
importing `reconcile_answer` from the L0 worktree and scoring against
`scores.csv`: `agentic_full` 8 fires / 0 broken / 6 recovered,
`agentic_no_planner` 1 fire / 0 broken / 0 recovered. Matches the subagent
exactly.

**decision — KEEP the build, CANCEL its confirmation run.**

KEEP under §3.5's second limb: both fixes are correctness fixes with 0 measured
breakages across 500 questions, the `agentic_full` invariant is demonstrated by
differential replay (identical sha256 over 500 questions, harness proven
non-blind), and 424 tests pass. The build is also the infrastructure every
later loop needs.

CANCEL the two n=250 runs, which is a departure from §5 L0 and needs its
reason on the record. **The fixes were designed against `agentic_full`'s
failure modes, and `agentic_v2` is built on `agentic_no_planner`, which does
not have them.** 1A repairs answer-type leakage from the Planner into the
synthesis prompt; with no Planner there is no leakage, and the rule recovers
nothing. 1C is a wash-to-negative on that base. So `agentic_v2` is predicted
to land at `agentic_no_planner`, whose n=250 numbers are already known
(0.456 / 0.324). Spending 8-12 GPU hours to re-measure a configuration we
have already measured, in order to confirm a null the replay already predicts,
buys nothing the loop needs. That budget goes to L1 and L2, which are the
hypotheses with a real expected effect.

R2.4's 0.47-0.50 expectation for `agentic_v2` is **withdrawn**; it was
extrapolated from `agentic_full`-base replays. Expect ~0.456 / ~0.324.

**Three errors in the plan, found by building it** (all fixed, all recorded in
the commit): the `template_shortcut` override is a no-op so `agentic_v2` would
have run the full LLM planner; keeping 1B unconditional is incompatible with
the byte-identity invariant (it changes the prompt on 12/250 of
`agentic_full`); and `AGENTIC_CONFIGS` gates the LLM client and NLI preflight,
so `agentic_v2` would have run with `client=None` and silently degraded to
extractive answers with no error raised.

**new baseline** `v2` @ `1443038` — behaviourally identical to `main` at the
default flags; `agentic_v2` selectable.

## Loop 2 — L1: thinking mode on (CALIB SCREEN)

2026-09-08 · subagent `L1-think` · worktree `D:\ir_agent_worktrees\L1-think`
· screened against `v2` @ 03b8550, config `agentic_no_planner`, calib n=50

```
baseline   HotpotQA EM 0.480 F1 0.546 | 2Wiki EM 0.280 F1 0.364
           L1_base_hotpotqa / L1_base_twowiki
treatment  HotpotQA EM 0.520 F1 0.598 | 2Wiki EM 0.380 F1 0.465
           L1_synth_hotpotqa / L1_synth_twowiki
paired dEM HotpotQA +0.040 [-0.060, +0.140]  g/l 4/2
           2Wiki    +0.100 [+0.000, +0.200]  g/l 6/1
cost       median latency 12.4 -> 25.5 s (Hotpot), 15.1 -> 36.8 s (2Wiki)
           llm_calls 1.38 -> 1.44, 1.16 -> 1.18
           projected n=250: 2.2 h / 3.4 h against an 8 h ceiling
```

**verified** — coordinator recomputed both paired deltas, gained/lost counts,
parse failures and median latencies from `scores.csv` + `traces.jsonl`: all
match. My first `think_chars` probe read the question-level `metrics` block and
returned 0; the field is per-call. Re-checked at the right level:
`L1_synth_twowiki` synthesiser 52/52 calls non-zero (median 6955, max 27585),
verifier 0/7 — per-agent routing does what it claims, and the subagent's figure
was right where mine was wrong.

**decision — PROMOTE to n=250 confirmation.** The screen bar in §5 was >= 8 EM
points; 2Wiki returned +10 with 6 gained against 1 lost. Neither interval
excludes zero at n=50 and the subagent said so plainly rather than claiming a
direction — which is the correct reading of a screen, whose job is to decide
whether the confirmation is worth buying, not to establish an effect.

Latency is comfortably inside §4 (25.5 / 36.8 s median against a 90 s cap).

**Attribution checked**: the baseline's longest completion was 271 chars
against a 1024-token cap, so `num_predict` 1024 -> 2048 cannot by itself have
moved an answer. The effect is thinking, not the raised cap.

**A defect the screen exposed, to be fixed before the confirmation.** 2 of
2Wiki's 50 questions produced `completion_chars: 0` after 27585 and 23335
characters of thinking across three attempts each — the synthesiser reasoned
until it had nothing left for an answer and fell back. `num_predict: 2048`
raises the starvation threshold, it does not remove it, and median thinking is
already ~1700 tokens. This *depresses* the treatment, so the true effect is at
least what was measured.

**Variant space**: settled on (a) synthesiser-only. (b) is not a distinct
experiment under a one-node plan — verified separately in Loop 1's interim
note.

**new baseline** unchanged, `v2` @ `1443038`. L1 is not merged until the
confirmation decides it.

### Loop 2 (L1) — starvation fix before confirmation

VRAM measured rather than assumed, and the obvious option was rejected on the
measurement:

    num_ctx  8192   6.19 GB   100% GPU
    num_ctx 16384   7.81 GB   80% GPU / 20% CPU, 1.53 GB spilled

Doubling the context spills 1.53 GB to CPU on a run that is already
latency-bound, so `num_ctx` stays at 8192 and the damage is capped with
`num_predict: 2048 -> 5120`. Sized from the traces, not guessed: max
synthesiser prompt is 5472 chars (~1368 tokens), leaving ~6800 inside the
window, so 5120 keeps ~1700 tokens of headroom for the repair ladder's extra
turns. **verified** by the coordinator over 102 synthesiser calls: median
prompt 3703, p90 4803, max 5472.

Re-screened on the 25-question 2Wiki calib slice containing both known
starvation cases:

    num_predict 2048   2 starved   median 38.6 s   total 24.7 min
    num_predict 5120   1 starved   median 35.4 s   total 23.6 min

Both original starvations now succeed on the **first** attempt (retries 0),
and their latency fell 256.5 -> 51.3 s and 199.9 -> 69.2 s. Total wall clock
went *down* despite the larger budget, because a starved call previously burnt
2048 x 3 attempts and still failed — the repair turn instructs the model to
omit a `<think>` block, which does nothing when thinking is returned
out-of-band. That retry loop was pure waste.

**Residual, carried forward honestly.** One *different* question starved at
5120 having survived at 2048: it drives qwen3 into an unbounded reasoning loop
(67374 thinking chars across three attempts, no content), so a larger budget
only lengthens the loop. It terminated on the 300 s wall-clock guard, which is
the orchestrator containing the damage correctly. 1 starved call per 25 on
2Wiki remains as a known depressant on the treatment. `max_format_retries` was
deliberately left alone — it would cut the same waste but changes error
handling for every agent and would confound the confirmation.

Both confirmation arms share `num_ctx: 8192`, `num_predict: 5120`. With
thinking off, completions are ~70 tokens, so the raised cap is inert in the
baseline — which is what makes sharing it safe.

## Loop 3 — L2: larger local model (PHASE 1, preparation only, no inference)

2026-09-08 · subagent `L2-model` · worktree `D:\ir_agent_worktrees\L2-model`
· cut from `v2` @ `1443038`

`qwen3:14b` pulled: 9.3 GB, Q4_K_M, 14.8B params, 40 blocks, native context
40960, thinking-capable. No inference run; `ollama ps` shows only the other
agent's `qwen3:8b`.

**Two plan items struck on evidence.**

*`num_ctx` 8192 -> 4096 is REJECTED under L1-thinking.* §5 proposed it to free
KV cache for more GPU layers. It does not survive: max synthesiser prompt is
1368 tokens and L1's adopted `num_predict` is 5120, so the worst case needs
6488 > 4096. The decisive argument is not the arithmetic though — on hitting
the window Ollama shifts it and evicts the prompt head mid-reasoning, turning
L1's *visible* `completion_chars: 0` fallback into a *silently ungrounded*
answer. §5's "max prompt ~1130 tokens" was measured with thinking off, where
the generation term does not bind. Conditional: if L1 DROPs, 4096 becomes
viable again (1368 + 1024 = 2392) and the suggestion stands.

*`llm.fallback_model` must NOT be repointed.* It fires only when the primary is
unpulled, and `qwen2.5:7b-instruct` has never been installed — coordinator
confirmed `ollama list` carries no such tag. Making it live for the first time
inside the one experiment that contrasts two models would let a 14B failure be
answered by the 8B, invisibly. Left as dead code.

**A coordinator error, caught rather than run with.** All worktree configs were
generated from the main checkout's `config.yaml`, which sits on `main` at
03b8550 (pre-L0). For `L1-think`, also at 03b8550, that is correct. For
`L2-model` at 1443038 it was stale, missing `reconcile_answer`,
`answer_type_guard`, `evidence_ranking` and `evidence_docs`. Harmless at
runtime — every key has a code default equal to evaluated behaviour — but the
trace's config snapshot would have been wrong. Regenerated from `v2`. Scope is
exactly one file.

**Arms defined.** `config/config.worktree.yaml` (8B) and `config/config.14b.yaml`
(14B), verified on the parsed trees to differ by exactly six keys:
`llm.default_model` and `llm.models.{planner,retriever,kg,verifier,synthesizer}`.

**A calibrated prediction for the gate, made before measuring.** Verified
`OLLAMA_KV_CACHE_TYPE` and `OLLAMA_FLASH_ATTENTION` unset (f16 KV, 2 B) and
`OLLAMA_NUM_PARALLEL=1`, then from GGUF metadata:

    qwen3:14b  40 blocks x 8 kv heads x (128+128) x 2 = 163,840 B/tok = 160 KiB/tok
    qwen3:8b   36 x 8 x 256 x 2 = 147,456 B/tok = 144 KiB/tok

**verified** by the coordinator against `/api/show`: block_count 40,
head_count_kv 8, key_length 128, value_length 128 — the 160 KiB/token figure is
exact.

Calibrated against the only real measurement available (L1's 8B: 8192 ->
6.19 GB, 16384 -> 7.81 GB, delta 1.62 GB) where pure KV predicts 1.21 GB, giving
a compute-buffer overhead factor of **1.34**. That factor is what turns a
textbook number into one that matched hardware.

    prediction @8192   VRAM need 10.29 GiB, ~30/40 layers on GPU, 10.5-14.5 tok/s
    prediction @4096   VRAM need  9.57 GiB, ~32/40 layers,        12.1-16.1 tok/s

**The gate is 15 tok/s, so the prediction straddles it and leans to FAIL.**
Note the third independent reason not to adopt 4096: it moves throughput only
~1.5-2 tok/s and cannot rescue a failing gate.

Falsifiers were stated in advance: >20 tok/s falsifies the bandwidth model;
`size_vram` @8192 well under 7.5 GiB means the split is worse than predicted;
a 8192->4096 delta outside 0.67-0.90 GB falsifies the f16 KV assumption.

**Phase 2 blocked** until L1's confirmation releases the GPU. Loading the 14B
would evict `qwen3:8b` and corrupt L1's timings mid-run.

**Open item for when L1 lands:** both L2 config files carry `v2`'s
`num_predict: 1024, think: false`. If L1 KEEPs, its settings must be applied to
*both* arms simultaneously or they stop differing by only the model.

## Loop 2 — L1: thinking mode on (CONFIRMATION) — **KEEP**

2026-09-08 · subagent `L1-think` · four runs, one attempt each, no resumes
· both arms `agentic_no_planner`, `num_ctx` 8192, `num_predict` 5120,
`PYTHONHASHSEED=42`

```
baseline   HotpotQA EM 0.452 F1 0.538   L1c_base_hotpotqa
treatment  HotpotQA EM 0.524 F1 0.618   L1c_think_hotpotqa
paired dEM +0.072 [+0.028, +0.116]  EXCLUDES ZERO   gained/lost 25/7

baseline   2Wiki    EM 0.316 F1 0.379   L1c_base_twowiki
treatment  2Wiki    EM 0.456 F1 0.511   L1c_think_twowiki
paired dEM +0.140 [+0.088, +0.192]  EXCLUDES ZERO   gained/lost 42/7

cost       median latency 3.6 -> 18.7 s (Hotpot), 3.6 -> 32.2 s (2Wiki)
           max 22.2 -> 424.1 s, 17.4 -> 418.2 s
           250-question run 2.45 h / 4.32 h against an 8 h ceiling
           mean llm_calls 1.35 -> 1.37, 1.25 -> 1.21 (unchanged)
           parse_failures 0 -> 2, 0 -> 8; starved 10/500
```

**verified** — coordinator recomputed both paired bootstrap intervals and both
floor comparisons independently from `scores.csv`: +0.072 [+0.028, +0.116] and
+0.140 [+0.088, +0.192], gained/lost 25/7 and 42/7. All match.

**decision — KEEP.** Both paired intervals exclude zero under §3.5's first
limb, and latency is inside §4 with the valve unused.

**Against the `self_ask` floor, the two datasets read differently and the
subagent said so rather than rounding it up:**

    HotpotQA  0.524 vs 0.504   +0.020 [-0.032, +0.072]   TIE, interval spans zero
    2Wiki     0.456 vs 0.264   +0.192 [+0.128, +0.256]   CLEARS, decisively

Both verified by the coordinator. Against §1's ladder: the floor is met on
2Wiki and tied on HotpotQA; the **target** (>= 0.38) is met on 2Wiki, which at
0.456 is past even the stretch (0.42); HotpotQA at 0.524 is short of its 0.55
target. §7's stop condition 1 is therefore *not* met — the HotpotQA gap is
still open.

Against the reported system, `agentic_full`: **+0.092 HotpotQA, +0.232 2Wiki.**

**Starvation residual, unfixed and honest.** 10 of 500 questions, every one the
same signature: `retries=2`, 66-70k thinking chars across three attempts,
`completion_chars=0`. The model enters an unbounded reasoning loop that no
generation budget reaches; the 300 s wall-clock guard contains it. This
depresses the treatment, so the measured gain is a floor rather than an
inflation.

**A coordinator error, recorded.** `L1-think` was cut from `v2` *before* L0
landed, so its commit's parent is 03b8550. Force-moving `v2` onto it silently
dropped L0's commit. Caught by checking `git merge-base --is-ancestor`
immediately after; fixed by resetting `v2` to L0 and cherry-picking L1 on top
(clean — the two touch disjoint files). `v2` @ `84bcc92` now contains both;
422 tests pass. The lesson is the one §3 already implies: verify the merge, not
just the measurement.

**new baseline** `v2` @ `84bcc92` — L0 + L1. `main` untouched at 03b8550,
`config.yaml` unmodified, so `agentic_full` remains byte-identical.

## Loop 3 — L2: larger local model — **DROP**

2026-09-08 · subagent `L2-model` · worktree re-cut onto `v2` @ `84bcc92`
· gate only, no evaluation run

```
qwen3:14b @8192, thinking on   11.06 tok/s   size_vram 5.857 GiB   58% GPU / 42% CPU
qwen3:14b @4096                11.90-12.65   size_vram 5.722 GiB   61% GPU
gemma3:12b @8192               13.61-15.52   size_vram 4.791 GiB   58% GPU
```

**verified** — coordinator loaded `qwen3:14b` independently and measured
**11.28 tok/s** over 300 decoded tokens, `size_vram` 5.86 GiB, 58% GPU.
Matches the subagent's range and split exactly.

**decision — DROP, on two independent counts, with no evaluation run.**

1. *Throughput*: 11.06 tok/s against §5's 15 tok/s gate.
2. *Latency*: at 4.70x slower than the 8B and against L1's confirmed 2.45 h /
   4.32 h, a 250-question run projects to **16.5-20.3 h on 2Wiki** against §4's
   8 h ceiling; median per question 237-292 s against the 90 s cap. The latency
   valve saves ~9 s of a 237 s question and cannot close a 2x gap.

**The general finding is worth more than the specific one.** This card admits
~5.9 GiB resident and Ollama holds the GPU fraction near 58% for anything
larger, regardless of `num_ctx`. So **"a larger local model" is closed on this
hardware, not merely for the 14B.** The 8B's 52 tok/s comes precisely from
fitting entirely on the card.

**Prediction versus measurement — right answer, partly wrong reasons, caught by
its own falsifiers.** Predicted 10.5-14.5 tok/s, measured 11.06-12.41.
Falsifier 2 fired: predicted `size_vram` ~7.5 GiB at a 75/25 split, measured
5.857 GiB at 58/42. Falsifier 1 did not (11-12, not >20) but the implied CPU
penalty was R~4.7 against an assumed 5. A worse split and a milder CPU penalty
cancelled, putting throughput in-band by coincidence. Falsifier 3 did not fire:
the KV arithmetic was near-exact, footprint 10.077 -> 9.456 GiB, a delta of
**0.621 GiB against 0.625 predicted** (0.6% error).

**The failed sub-prediction retires a plan idea for good.** Freeing 0.62 GiB of
KV bought **~0 extra GPU layers**, not the 3-4 predicted: `size_vram` went
*down* (5.857 -> 5.722) and the entire saving came off the CPU side. "Shrink
`num_ctx` to buy layers" does not work on this hardware. §5's L2 suggestion is
struck permanently, not merely declined.

**§5's 15 tok/s gate is too lenient.** Inverting the budget: the 8 h run
ceiling requires **>= 24.5 tok/s** and the 90 s median cap requires
**>= 32.7 tok/s**. A model could clear the stated gate and still be unrunnable.

**Fallback closed independently.** `gemma3:12b` grazes 15 tok/s on a short
prompt but sits at 13.6-14.0 realistically and projects to 13.4-16.3 h on
2Wiki. It is closed regardless: `ollama show` reports capabilities
`completion, vision` with **no `thinking`**, so it cannot reproduce L1's
mechanism, and adopting it would forfeit L1's banked +0.072 / +0.140 to chase a
speculative gain.

**new baseline** unchanged, `v2` @ `84bcc92`. Models left on disk (C:, 205 GB
free); say the word to remove them.


## Coordinator fix — L1's KEEP was not on the branch
2026-09-08 · coordinator · `v2` `84bcc92` -> `76f8551`

Setting up L3's worktree surfaced a defect in the branch. `v2` carried L1's
*mechanism* -- `LLMSettings.think_for`, the per-agent thinking router, and the
`/no_think` strip -- but none of the three settings that produced the measured
gain. Those lived in `D:\ir_agent_worktrees\L1-think\config\config.worktree.yaml`
and nowhere else:

| setting | config.yaml on `v2` | the confirmed run |
|---|---|---|
| `llm.think_agents` | absent | `[synthesizer]` |
| `llm.think` | `false` | `true` |
| `llm.options.num_predict` | 1024 | 5120 |

So `git checkout v2 && run_eval --config agentic_v2` ran the **pre-L1** system.
Left alone, that would have become the baseline every later hypothesis was
measured against, and L3's +1-2 points would have been reported on top of a
system 7 points below the one we had already banked.

Fixed in `76f8551`: the three settings move into
`ABLATION_OVERRIDES["agentic_v2"]`, beside 1A/1B/1C and for the same reason --
`config.yaml` keeps the evaluated values, so `agentic_full` is untouched, and
the override is what defines the new system. `llm.think_agents` is now declared
in `config.yaml` as `null`, which is what `think_for` already assumed for the
grid (`[]` would mean "no agent reasons", a different statement); declaring it
lets `cfg.get` answer instead of raising. Four new assertions; 426 pass, 20
skip. `1443038` (L0) and `84bcc92` (L1) both still ancestors -- checked, after
the merge error two loops ago.

**`num_predict` is part of the decision, not incidental to it.** At 1024 the
reasoning block consumes the whole completion budget and the synthesiser
returns no answer; run `L1_starve_np5120` exists to record exactly that. The
per-call budgets elsewhere (retriever 48/64, kg 96, verifier 256) override the
global, so only the synthesiser sees 5120.

**What was measured, precisely.** All four L1 confirmation runs record
`config_name: agentic_no_planner` -- thinking was switched on in the worktree
config, which applies to whatever configuration runs, with 1A/1B/1C off. The
confirmed system is therefore the one-node plan plus reasoning. `agentic_v2` as
it now stands is that *plus* the three fixes, a combination no run has covered.
It should land on the same number, because L0 measured 1A recovering 0
questions on this base and 1B changing 0 answer types under a one-node plan --
but that is an inference from two offline replays, not a measurement. L3's own
baseline run settles it, which is one more reason §6 requires the executor to
run its own baseline rather than read one.

## Loop 4 — L3: `max_evidence` 20 -> 30 — setup
2026-09-08 · subagent · `D:\ir_agent_worktrees\L3-evidence` @ `76f8551`

**HotpotQA only.** 2Wiki is at 0.456 against a 0.38 target and a 0.42 stretch;
spending ~4.3 h to move a dataset that has already passed is poor value against
the one open gap (HotpotQA 0.524 vs 0.55). Two runs, n=250: baseline
`agentic_v2` at max_evidence 20, treatment at 30.

**Pre-flight the executor must clear before any GPU time.** `num_ctx` is 8192
and the synthesiser's `num_predict` is 5120, leaving roughly 3000 tokens for
the prompt. Twenty evidence items may already be near that. If a material
fraction of prompts overflows at 30, Ollama truncates context from the left
silently and the run measures truncation rather than evidence. The prompt-token
distribution at both settings is due **before** the treatment runs.

**Budget.** L1's HotpotQA thinking run: 2.45 h wall, 18.7 s median. The
treatment is slower. §4's ceilings are 8 h per run and 90 s median.

The 14B left resident in VRAM by L2 was unloaded first (6.6 GiB -> 484 MiB).


## Loop 4 — L3: `max_evidence` 20 -> 30 — **DROP**
2026-09-08 · subagent `L3-evidence` · `D:\ir_agent_worktrees\L3-evidence` @ `76f8551`

```
pre-flight (offline, no GPU, real Qwen3 tokenizer over the real chat template)
  synth prompt tokens   me=20  median 1064  p95 1369  max 1489   over 3072: 0/250
                        me=30  median 1064  p95 1867  max 2018   over 3072: 0/250
  harness validated: 248/250 rebuilt prompts match L1c's logged prompt_chars

treatment  NOT RUN -- dropped on the pre-flight premise (~3 GPU hours saved)
baseline   L3_base_hotpotqa, killed at 31/250 on instruction
           paired vs L1c_think_hotpotqa, same 31 questions:
           EM 0.5484 vs 0.5484   F1 0.6184 vs 0.6184   dEM +0.0000   g/l 1/1
cost       median latency 17.4 -> 34.9 s;  mean llm_calls 1.39 -> 1.87
           projected 250q wall 1.74 h -> 2.82 h (inside the 8 h ceiling)
verified   coordinator recomputed the synthesis-failure census independently
           from traces (below): matched the subagent exactly, 0/0, 2/26, 0/0, 8/49
decision   DROP -- the binding constraint the hypothesis assumed does not exist
           on this base
new baseline  v2 @ `48ec924` (see the unbundling commit below)
```

**The premise, and my error in the backlog.** `agentic_v2`'s one-node plan
issues one sub-query -> 3 documents -> a **median of 10 passage sentences**;
only **3 of 250** questions exceed 20. So 20 -> 30 adds **14 passage sentences
across the whole slice** and 828 KG triples, moving complete gold pools
**142 -> 143**. §5's "152 -> 155" is `agentic_full`'s five-sub-query pool. I
carried a number from the wrong system into the backlog, and the pre-flight
caught it for the price of no GPU time at all. That is the whole argument for
§3's offline-first rule, made concrete.

**L5 closed on measurement, not on a guess.** KG evidence scores 0.2255-1.0000;
passage sentences under `rank_major` normalise into 0.0333-0.1000. Every KG row
outranks every passage row in `Synthesizer._pool`'s sort, so widening the budget
fills the new slots with triples placed *above* the gold-bearing sentences.
L5's premise was that a KG quota would fix exactly this; the measurement says
the ranking is the defect and L5 as specified does not address it.

**The bundle defect, and the limit of what the probe establishes.** 3/30
questions lost synthesis outright against **0/30** for L1c on identical
questions; `prompt_chars` identical, `prompt_sha1` different, which is 1C
permuting the evidence. But the hash-gated probe (5 reps/arm) **does not
confirm 1C as the cause**: both arms answered 5/5. It shows a directional
difference only -- `rank_major` 7 empty attempts in 22 and 3/15 reps needing a
retry, `lexical_major` 0 in 15 -- and 3-in-15 against 0-in-15 is Fisher
p ~ 0.22. Suggestive, not established. `rank_major` is out of `agentic_v2`
because it cost 2x median latency and 10% of synthesis calls for dEM +0.0000,
which is a cost argument and does not need the causal one.

**What actually ends the turn.** `done_reason='length'` with
`eval_count=5120`: the entire generation budget spent inside the thinking
channel, the content channel never opened. Both the subagent and I reported
earlier that this was *not* the cap; both of us were wrong, and the trace misled
us because `llm.py::chat` deduplicates `think_chars` across attempts, which
makes one attempt look like a fifth of its real size. The trace does not record
`done_reason` at all -- that is why it took a probe wrapping the client to see
it, and it is the single most useful thing to add to `LLMCallTrace`.

**Census -- thinking introduces the failure, `rank_major` amplifies it.**
Recomputed by the coordinator from `steps[*].llm_calls[*]`, matching the
subagent exactly:

| run | synth calls | failed | questions with a failure | lottery won (`retries>0`, ended OK) |
|---|---|---|---|---|
| `L1c_base_hotpotqa` (think off) | 250 | **0** | **0** | 0 |
| `L1c_think_hotpotqa` | 251 | 2 | 2 (0.8%) | **26 (10.4%)** |
| `L1c_base_twowiki` (think off) | 250 | **0** | **0** | 0 |
| `L1c_think_twowiki` | 251 | 8 | 8 (3.2%) | **49 (19.6%)** |

All 10 failures appear in `scores.csv` at EM 0, so **0.524 and 0.456 are net of
them**; the ceiling if every one were recovered correctly is 0.532 / 0.488. The
75 lottery-won questions all ended `parse_ok=True` -- a latency cost, not an
accuracy one.

**The thinking-off fallback: a clean null, and it closes the question.** The
first attempt at this probe measured nothing: `LLMSettings.think_for` checks
`think_agents` *first*, so `chat(agent="synthesizer", think=False)` returns
True and the arm ran with reasoning on. Corrected by clearing the routing rather
than passing an argument past it, and witnessed by `thinking_chars=0` on all
ten. The corrected arm recovers the content channel **10/10 on the first
attempt** and produces **0/10 exact matches**. The +0.008 / +0.032 ceiling does
not materialise: the model answers, and the answers are wrong. No override
channel is needed and none was built.

**Two defects recorded and deliberately not fixed.** `think_for` makes
`think_agents` unoverridable by any caller, including a repair path that
specifically needs to turn reasoning off; the obvious fix -- letting an explicit
`requested` win -- would switch reasoning on for *every* agent, because
`agents/base.py::_think()` passes a concrete value on every call, so it must
not be applied. And the repair rung's *"no `<think>` block"* instruction is
inert when thinking returns out-of-band. Both are cheap to fix and neither is
worth an unmeasured change at the end of a loop.

## Coordinator — the bundle unwound
2026-09-08 · `v2` `76f8551` -> `48ec924`

Loop 4's baseline was the first run of `agentic_v2` as the branch defined it,
and it exposed what bundling had hidden. 1A recovers 0 questions on this base
and 1B changes 0 answer types under a one-node plan, both from Loop 1's replay
over 500 traces; 1C costs 2x median latency and 10% of synthesis calls for
dEM +0.0000. None of the three earned a place in a configuration whose numbers
someone will cite, so `agentic_v2` is now exactly what the confirmation runs
measured: the one-node plan plus reasoning on the synthesis call.

All three keep their code, their flags and their tests. A new test builds a
system with them forced on, because nothing in the tree enables them any more
and that seam would otherwise rot unnoticed; another asserts that no
configuration -- `agentic_v2` included -- turns one on by accident.


---

# Final summary

**The model did not change.** Every number below is `qwen3:8b` at Q4_K_M on an
RTX 5060 Laptop GPU, the same model the report's grid ran, so these numbers and
the report's are on the same footing. L2 tried `qwen3:14b` and `gemma3:12b` and
both are closed on this hardware (Loop 3).

## The kept configuration

`agentic_v2` = the one-node plan (`NoPlanner`) + **reasoning on the synthesis
call only** (`llm.think_agents: [synthesizer]`, `llm.options.num_predict: 5120`).
One change, one mechanism. It is the only hypothesis of the five in the backlog
that survived measurement.

n=250 on the frozen eval slice of each dataset, paired on question id, 2000
bootstrap resamples:

### HotpotQA — exact match 0.524, F1 0.618

| against | its EM | ΔEM | 95% CI | p | gained/lost |
|---|---|---|---|---|---|
| `agentic_full` (the report's system) | 0.432 | **+0.092** | [+0.036, +0.144] | 0.000 | +36 / −13 |
| `agentic_no_planner` (its own base) | 0.456 | **+0.068** | [+0.024, +0.112] | 0.002 | +25 / −8 |
| `self_ask` (strongest baseline) | 0.504 | +0.020 | [−0.032, +0.072] | 0.499 | +25 / −20 |

### 2WikiMultihopQA — exact match 0.456, F1 0.511

| against | its EM | ΔEM | 95% CI | p | gained/lost |
|---|---|---|---|---|---|
| `agentic_full` | 0.224 | **+0.232** | [+0.168, +0.296] | 0.000 | +67 / −9 |
| `agentic_no_planner` | 0.324 | **+0.132** | [+0.080, +0.188] | 0.000 | +42 / −9 |
| `self_ask` | 0.264 | **+0.192** | [+0.128, +0.256] | 0.000 | +64 / −16 |

Five of the six comparisons exclude zero. **The exception is the one that
matters most for the headline claim**: on HotpotQA the kept system does not
beat `self_ask`, a single-model baseline with no agents in it. It ties. 2Wiki
is where the architecture earns its place, and it earns it decisively — +0.192
against the same baseline, on a dataset built to need multiple hops.

## What it costs

| | median | p90 | mean LLM calls |
|---|---|---|---|
| kept, HotpotQA | **18.7 s** | 115.6 s | **1.37** |
| `agentic_full`, HotpotQA | 28.3 s | 55.4 s | 4.26 |
| kept, 2Wiki | **32.2 s** | 144.7 s | **1.21** |
| `agentic_full`, 2Wiki | 35.4 s | 69.2 s | 3.80 |

A faster median and **a third of the model calls**, bought at roughly double
the p90. The tail is the reasoning budget: a synthesis call that reasons can
run to 5,120 tokens, and the questions that do are the slow ones.

*(The report's `agentic_full` HotpotQA run also contains one question logged at
18.3 hours. That is a machine stall, not compute — one question ≥ 250 s in the
whole run — and any total wall-clock figure that includes it is meaningless.)*

## The caveat this configuration ships with

Reasoning introduces a failure mode that thinking-off does not have. Measured
over 500 questions per condition:

| | questions losing a synthesis call | questions needing a silent retry |
|---|---|---|
| thinking **off** | **0 / 500** | **0 / 500** |
| thinking **on** | 10 / 500 (0.8% HotpotQA, 3.2% 2Wiki) | 75 / 500 (10.4%, 19.6%) |

The mechanism is exact: `done_reason='length'` with `eval_count=5120`. The
model spends its entire generation budget inside the thinking channel and never
opens the content channel, and the orchestrator's extractive `fallback_rule`
then emits a confident wrong answer rather than nothing.

**Those 10 losses are already inside 0.524 and 0.456** — all 10 score EM = 0 in
the runs that produced those numbers, so the gains are reported net of them.
Recovering every one correctly is a ceiling of 0.532 / 0.488. The 75 retries
all ended `parse_ok=True`: a latency cost, not an accuracy one.

**That ceiling was tested and does not materialise.** Re-issuing each of the 10
failed calls with reasoning off recovers the content channel **10 out of 10, on
the first attempt** — and yields **0 out of 10 exact matches**. The model
answers; the answers are simply wrong. So the failure mode costs less than it
appears to: these are questions the system was going to get wrong anyway, and
the extractive fallback that currently answers them is no worse than the model
would have been. The obvious repair is not worth building, which is the most
useful thing a null can tell you.

## What was measured and dropped

| | hypothesis | outcome |
|---|---|---|
| **L0** | one-node plan + fixes 1A/1B/1C | built, flag-gated, **not shipped** — 1A recovers 0 questions on this base, 1B changes 0 answer types, 1C costs latency and synthesis calls for ΔEM +0.0000 |
| **L1** | reasoning on the synthesis call | **KEPT** — the result above |
| **L2** | a larger local model | **DROPPED** — `qwen3:14b` at 11.1 tok/s against the 24.5 tok/s the budget requires; `gemma3:12b` has no thinking capability at all |
| **L3** | `max_evidence` 20 → 30 | **DROPPED on its premise**, before any GPU time |
| **L4** | self-consistency, 3× sampling | **not attempted** — its cost estimate predates L1, and ×3 synthesis with reasoning breaches the 8 h ceiling |
| **L5** | question anchoring + KG quota | **CLOSED on L3's evidence** |

L3 deserves its own line because the pre-flight is what the loop exists for.
The backlog claimed 20 was the binding constraint on evidence (152 → 155
complete gold pools). That figure is `agentic_full`'s five-sub-query pool. On
the one-node base the plan issues **one** sub-query → 3 documents → a median of
**10** passage sentences, with only 3 of 250 questions above 20. So 20 → 30
buys 14 passage sentences across the entire slice and moves complete gold pools
**142 → 143**. I wrote that number into the backlog from the wrong system, and
the pre-flight caught it for the price of no GPU time at all.

The same pre-flight closed L5: KG rows score 0.23–1.00 against a passage
ceiling of 0.10, so widening the budget fills the new slots with graph triples
ranked *above* the gold-bearing sentences. L5's premise was that a KG quota
would fix exactly this; the measurement says the ranking is the problem, and
L5 as specified does not address it.

## The finding that outlives the loop

**Seed pinning does not make this system reproducible.** Identical prompt,
`temperature: 0`, `seed: 42`, same process, same model — and different
completions across repetitions. `prompt_eval_count` for the same first-attempt
prompt varies 1,218 / 2,977 / 3,671: Ollama's KV-cache prefix reuse changes the
batch shape, and the batch shape changes the arithmetic.

This is a concrete mechanism for the 6–14% run-to-run answer churn measured
earlier on byte-identical code, which until now was an observed fact with no
explanation. It sets a floor on what is measurable here: an effect smaller than
about 3 points cannot be distinguished from the harness on n=250, whatever the
seed says. Every "unresolved" verdict in this log is downstream of it.

## Which stop condition ended the loop

**§7 condition 3 — two consecutive DROPs after the floor was reached.** L2 and
L3 both dropped; the floor (`self_ask`) was cleared on 2Wiki at Loop 2 and tied
on HotpotQA.

The original text here added that "condition 2 is also effectively met". It was
not: L4 had never been attempted, and "effectively" was doing work that a stop
condition does not allow. **Loop 5 decided L4 on an offline gate** (below), so
condition 2 is now met outright — L0 through L5 are all decided — and the
sentence no longer needs the hedge.

§1's ladder, honestly scored: 2Wiki **passed its target (≥0.38) and its stretch
(≥0.42)** at 0.456. HotpotQA at 0.524 **did not reach its 0.55 target** — it
cleared the floor and stopped 0.026 short. The loop ends with one dataset past
its stretch goal and one short of its target, and no remaining hypothesis
credibly worth the GPU hours to close a 2.6-point gap on a harness whose own
noise floor is around 3 points.

## What the branch holds

`v2`, on top of `main` at `03b8550`, untouched:

- `1443038` — L0's fixes, flag-gated, **all defaulted off**
- `84bcc92` — L1's per-agent thinking router
- `76f8551` — L1's confirmed settings moved onto the branch
- `48ec924` — the unmeasured fixes unbundled from `agentic_v2`

Verified: 430 tests pass; `python -m agentic_ir.cli tables` from `main`
regenerates the report's tables byte-identically; and the four evaluated
agentic configurations resolve to **identical** config trees on `main` and
`v2`, so `agentic_full` is the system the report describes in behaviour and not
merely in intent.


## Loop 5 — rectification, after an independent audit
2026-09-09 · `v2` `5e4b722` → see below

An audit re-derived every number this log reports, from the traces, without
reusing the loop's own scripts. **All of them reproduce**: exact match
0.524 / 0.456 and F1 0.618 / 0.511; the six paired intervals to four decimals;
the synthesis-failure census 0/0/2/8 with 0/0/26/49 retries; byte-identical
report tables from `main`; identical resolved configurations for the four
evaluated agentic systems on `main` and `v2`. What it found was not a wrong
number. It was a set of things the branch could not tell a reader.

### Provenance of the runs behind 0.524 / 0.456

The four `L1c_*` runs record `git_commit 03b8550`, which is `main`, because
L1's diff was still uncommitted when they were made. The code they ran is the
patch now on the branch as `84bcc92`, and that is checkable rather than
asserted:

```
diff <(git diff 29fbdba~1 29fbdba -- src/ tests/) \
     <(git diff 1443038 84bcc92   -- src/ tests/)   # empty: identical patch
```

### `agentic_v2` could not be run from a checkout of the branch

`--config` takes its choices from `evaluation.configurations`; `config.yaml`
must not name `agentic_v2` or `tables.py` grows a tenth row; and the variant
config that declared it lived in a scratch worktree deleted when the loop
ended. So the branch quoted numbers for a configuration nobody could run.
`config/config.v2.yaml` is now committed — `config.yaml` plus exactly two
lines, verified by diff — with a test pinning both halves and a README
section giving the commands. (`14c3a1e`)

### The trace could not say why a call ended

`LLMCallTrace.truncated` was declared with the schema and assigned by nothing:
`False` on every call ever recorded, including the ten the loop lost to a
spent budget. **This is why this log twice states, wrongly, that the empty
completions were "not the cap"** — the field that would have settled it was in
the record, reading False, because nobody wrote to it. `llm.py` now reads
Ollama's `done_reason`; `LLMResponse` and `LLMFormatError` carry it plus
`hit_length_cap` (any attempt, not just the last, so a call that burns 5,120
tokens and then parses is still visible); `base.py` writes it into `truncated`
and adds `completion_tokens`. Seven tests, and no behaviour change: the five
probe questions produce **byte-identical synthesis prompt hashes** before and
after (4656 / 4450 / 4257 / 2972 / 2417). (`36c4817`)

### And the instrumentation immediately earned itself

The first run under it reported `completion_tokens` **3072** on a failing
synthesis call. That is three attempts of **1024**, not of the 5120 the
configuration is supposed to use. Checking per-agent `think_chars` on the same
five questions against the run that produced 0.524:

| | synthesiser | verifier |
|---|---|---|
| `L1c_think_hotpotqa` (measured) | reasons | **does not reason** |
| `agentic_v2` via `ABLATION_OVERRIDES` | reasons | **reasons** |

`run_eval` built its client with `get_client()`, which reads the *shipped*
config, so **every `llm.*` key in `ABLATION_OVERRIDES` was discarded** -- the
client is where `llm.options` and `llm.think_agents` are read and it had never
seen them. `llm.think` reaches agents by a second route
(`agents/base.py::_think`, which does read the run config), so what actually
ran was reasoning on **every** agent at **1024** tokens. That is precisely the
starvation setting Loop 2 measured as returning no answer at all, and it is why
2 of these 5 questions failed where the measured run answered both.

**The configuration on the branch was not the configuration that produced
0.524**, and it had been that way since `76f8551` moved those settings out of
the config file and into the override -- a commit written by the coordinator to
*fix* provenance.

**The audit's 5/5 prompt-hash probe could not have caught this**, and that is
the lesson worth keeping. None of `think`, `think_agents` or `num_predict`
takes part in building a prompt, so the hashes matched perfectly while the
generation behaviour was a different system's. A prompt-identity check
establishes that the retrieval and evidence path is unchanged; it says nothing
about how the model is then asked to generate. `completion_tokens` -- a field
that did not exist that morning -- is what gave it away.

Fixed by `client_for(run_cfg)`, which builds the client from the run's own
configuration, with tests pinning both directions: `agentic_v2` reasons on the
synthesiser alone at 5120, and none of the nine evaluated configurations picks
up either setting. (`36c4817`)


Implementing it surfaced a trap worth recording. Read as plain attributes,
`response.hit_length_cap` raised `AttributeError` on every duck-typed stub in
the suite — caught by axiom 2, so nothing crashed; the agent simply returned
`ok=False` and the Planner produced `fallback_rule` plans. **A telemetry field
silently degraded working agents, and the only symptom was a worse answer.**
Both reads are `getattr` with a default now, and a test pins that a response
lacking the fields still answers.

## Loop 5 — L4: self-consistency on the synthesis call — **DROP**
2026-09-09 · offline gate, no GPU time

The loop had left L4 "not attempted", which is not a decision. The audit
costed it from the kept runs' own traces: synthesis is 83% of HotpotQA's wall
time (2.04 h of 2.45 h) and 74% of 2Wiki's (3.19 h of 4.32 h), so ×3 sampling
projects to **6.53 h** and **10.69 h**. Against §4's 8 h ceiling that made it
feasible on HotpotQA — the dataset with the open gap — and infeasible on 2Wiki.

So the question was whether 6.5 GPU hours could buy the ~2 points HotpotQA
needs. The gate: voting suppresses *variance*, so it can only fix a wrong
answer where a different draw would have been right. Count those first.

Of 119 wrong answers in `L1c_think_hotpotqa`, **19** have the shape L4 targets
— gold present in the model's own `answer_sentence`, a different span in
`answer`. That clears the plan's bar of 15, and on the plan's own rule L4
should have run. **It should not, and the bar was measuring the wrong thing.**
Fifteen of the nineteen are surface-form artifacts with no variance to
suppress:

| gold | model's answer |
|---|---|
| `ten` | `10` |
| `five books` | `5` |
| `650 locations` | `650` |
| `over 150 films` | `over 150` |
| `Washington State` | `Washington State Cougars` |
| `Los Alamos` | `Los Alamos Laboratory` |

Three draws at `temperature: 0.7` would agree with each other and still
disagree with gold; these are deterministic formatting choices, and
`accuracy-plan.md` R2.6 already ruled surface-form scoring out of scope as a
metric property rather than a system defect. **Four** are genuinely different
spans — and one of those is `19th century` against gold `19th-century`, a
hyphen, and another is `yes` against an airport name, an answer-type error
rather than a span choice.

**Ceiling if voting fixed every genuine one: +0.016 EM — 1.6 points.** The
measured noise floor on this harness is around 3 points. The effect cannot be
resolved at n=250 *even if it is entirely real*, so the run's interval would
have to include zero, and §3's own decision rule would then DROP it. Six and a
half GPU hours to buy an interval whose verdict is already known is not a
measurement.

**DROP on the offline gate**, the same way L3 was dropped, and for the same
reason: the hypothesis's premise did not survive contact with the actual
failure distribution. The backlog L0–L5 is now fully decided and §7's
condition 2 is met outright.

**A note on the gate itself.** The plan set the bar at a *count of a shape*,
and 19 of that shape existed. Counting whether the shape was the *mechanism*
took one more filter and reversed the answer. That is the same error as L3's
premise — a number that described a different situation than the one it was
applied to — and it is worth stating that the rectification plan's own gate
had it too.


## Loop 5 — invariants, re-verified after every change
2026-09-09 · `v2` @ `5134e77`

| invariant | result |
|---|---|
| `python -m agentic_ir.cli tables` from `main` | no tracked table modified |
| the four evaluated agentic configurations, `main` vs `v2` | **0 keys changed value, 0 removed**; 20 keys added (5 distinct), every one at the evaluated default |
| `--config agentic_v2` under `config/config.yaml` | rejected |
| `--config agentic_v2` under `config/config.v2.yaml` | accepted |
| `tests/test_report_integrity.py` | 16 passed — still nine configurations |
| synthesis `prompt_sha1`, five probe questions, before/after R4 | 5/5 identical |
| full suite | **444 passed**, 20 skipped |

The keys `v2` adds and `main` does not have are `llm.think_agents` (null),
`agents.synthesizer.reconcile_answer` and `.answer_type_guard` (both false),
`agents.verifier.evidence_ranking` (`lexical_major`) and `.evidence_docs` (3).
Each is L0's or L1's, each is declared at the behaviour the grid ran, and no
key that existed before changed value. That is the invariant that makes
`agentic_full` the system the report describes, and it holds.

**Two corrections to how this was checked.** The audit reported the
configuration trees "IDENTICAL" twice. That result came from a shell in which
`cd <worktree> && dump` was followed by a second `dump` in the same command —
the working directory persists between commands, so the second ran in the
worktree too and the check **compared `v2` with itself**. Whole-tree identity
was never the right property either, since L0 and L1 legitimately *declare*
keys that did not exist before; the property that matters is the one in the
table above. And the first replacement check reported "0 keys added", which was
also wrong: the dumper rendered each configuration with `default=str`, so the
comparison walked strings rather than trees and could not see a key at all. A
green result from a broken check is worth less than a red one, and both of
these were green.
