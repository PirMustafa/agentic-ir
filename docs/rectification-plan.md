# Rectification plan

For the executing model (Opus). Written 2026-09-09 after an independent audit of
the improvement loop that ended the night before. It lists what the audit
verified, what it found wrong or missing, the fix for each, and the one backlog
item the loop left undecided. Read `docs/improvement-loop.md` §2–§4 and §6–§7
first; every rule there still applies. Where this document and that one
disagree, this one wins, because it is later and it was written against the
evidence.

## 0. Ground rules (unchanged, restated because they are load-bearing)

- Nothing under `report/` is modified. Not a character. A draft chapter is
  written *outside* `report/` (§4) and the owner decides.
- No commit to `main`. Everything lands on `v2`. `main` stays at `03b8550`.
- Sole author on every commit. No `Co-Authored-By` trailer of any kind.
- Local Ollama only. No API model.
- One Python workload at a time; the GPU lock (`scripts/gpu-lock.sh`, path
  `/d/ir_agent_worktrees/gpu-lock.sh` if it still exists there) around every
  run. `PYTHONHASHSEED=42` is pinned by `run_eval` — check `meta.json` says so.
- Offline before GPU. A hypothesis that can be killed by replay is killed by
  replay. L3 was dropped for the price of no GPU time; that is the standard.
- Verify a subagent's number before writing it down. Recompute it from the
  traces yourself. The loop's log records two wrong conclusions ("not the
  cap", twice) that came from reading a deduplicated field at face value.
- Never report an action before the tool call that performs it has returned.
- Commit only from a worktree checked out *on* `v2` (`git worktree add <dir>
  v2`, never `--detach`), and after every commit run `git log --oneline v2 -1`
  and confirm it advanced. The audit itself committed `3766c0a` onto a
  detached HEAD, removed the worktree, and had to recover the commit from the
  object store by hand — the same class of error the loop's log records at
  Loop 2. A commit that `git log v2` does not show did not happen.
- Nulls are results. Write them as nulls.

## 1. What the audit verified (do not redo; the numbers stand)

| claim | check | result |
|---|---|---|
| EM 0.524 / 0.456, F1 0.618 / 0.511 | recomputed from `results/runs_v2/L1c_think_*/scores.csv` | exact |
| paired ΔEM vs `agentic_full` +0.092 [+0.036, +0.144], vs `self_ask` +0.020 [−0.032, +0.072] p=0.499, vs own baseline +0.072 [+0.028, +0.116] | `eval/bootstrap.py::paired_bootstrap`, 2000 resamples | exact to 4 dp |
| `self_ask` is the strongest answer baseline on HotpotQA | `results/tables/main_results.tex`: `self_ask` 0.504, `naive_rag` 0.496 | true |
| 430 tests pass on `v2` | fresh worktree at `df52430` | true |
| `agentic_full` unchanged | the four evaluated agentic configs resolve to identical trees on `main` and `v2`; `python -m agentic_ir.cli tables` from `main` is byte-identical | true |
| `agentic_v2` is the measured system | config diff against `L1c_think_hotpotqa/meta.json`: only four L0 keys that did not exist then (all at evaluated defaults) and file paths differ | true at config level |
| L1's cherry-pick carries the code the run used | `git diff 29fbdba~1 29fbdba` == `git diff 1443038 84bcc92` on `src/ tests/` | identical patch |
| synthesis-failure census 0/0/2/8, retries 0/0/26/49 | recomputed from `steps[*].llm_calls[*]` | exact |
| `docs/*.md` untracked in the main checkout == the blobs committed on `v2` | diff, modulo CRLF | identical |

**Dynamic check — done, 5/5.** `agentic_v2` as committed (`df52430`), run on
the first 5 HotpotQA eval questions under `config/config.v2.yaml` (absolute
data paths for the worktree), compared to `L1c_think_hotpotqa` on
first-attempt synthesis `prompt_sha1`. Retrieval is deterministic, so the
prompt hash is a code-path fingerprint that run-to-run sampling cannot touch.
All five hashes and all five `prompt_chars` match (4656 / 4450 / 4257 / 2972 /
2417). The equivalence is established at the code level; **R0 does not fire**,
and `L1c_think_hotpotqa` may serve as the L4 baseline (§3). Run preserved as
`results/runs_v2/audit_v2_probe`; `meta.json` records `git df52430`,
`pythonhashseed 42`, `evidence_ranking lexical_major`, `reconcile False`.

One observation from the same five, recorded because it bears on the log's
1C paragraph and *not* because five questions make a rate: two of the five
(`5a712beb…`, `5a713dcb…`) ended with `parse_ok=False, retries=2` — the
length-cap failure — under `lexical_major`, where the kept run had answered
both. `5a712beb…` is the same question that failed in `L3_base` under
`rank_major`. So the failure reproduces on identical prompts without 1C,
which is consistent with the log's conclusion that thinking introduces it
and 1C at most amplifies it. Do not update the census with these; they were
not drawn the same way.

## 2. What the audit found, and the fix for each

Severity is about what a reader of the branch would be misled by, not about
how hard the fix is.

| # | finding | severity | fix |
|---|---|---|---|
| R0 | (conditional) probe shows `agentic_v2` builds different prompts from the measured run | **blocking** if it fires | §2.0 |
| R1 | `agentic_v2` cannot be run from a fresh checkout: `--config` choices come from `evaluation.configurations`, the shipped config must not list it, and the variant config that enabled it lived only in deleted worktrees. `accuracy-plan.md` names `config/config.v2.yaml`; it was never committed | **high** — the headline result is not reproducible from the branch | §2.1 |
| R2 | `run_eval.py` comment cites the gain as 0.452→0.524 / 0.316→0.456 (the loop's own paired baseline, `L1c_base_*`) while the docs cite 0.456 / 0.324 (the report's `agentic_no_planner` runs). Both true; the comment does not say which | low | §2.2 |
| R3 | provenance: `L1c_*` `meta.json` records `git_commit 03b8550` because L1's code was uncommitted when the runs were made; the log does not say so. The log's stop-condition paragraph says "condition 2 is also effectively met" while L4 was never attempted | low | §2.3 |
| R4 | the trace cannot see why a call ended. `LLMCallTrace.truncated` exists and is never assigned; `done_reason` is not captured at all. This is what let two wrong conclusions stand until a probe wrapped the client | **medium** — instrumentation gap that already cost the loop | §2.4 |
| R5 | the repair rung on an empty completion appends an empty assistant turn and a correction ending "no `<think>` block", which is inert when thinking returns out-of-band; the retry burns another full 5,120-token budget. Measured: re-issuing with thinking off recovers the content channel 10/10 and is correct 0/10 | low — it is a latency defect, not an accuracy one | §2.5, optional |
| R6 | L4 was never costed against the post-L1 system; the summary says "not attempted" | **medium** — the backlog is not honestly closed | §3 |
| R7 | nothing is on GitHub: `main` is 7 commits ahead of `origin/main`, `v2` is 13 | high, but **owner's action** | §5 |

### 2.0 — R0, only if the probe is not 5/5

Do not touch anything else first. Find which of the four L0 keys, at its
evaluated default, still changes the synthesis prompt. Method: the L3 agent's
`results/runs_v2/_probes/l3_preflight.py` rebuilds prompts from recorded pools
under a stub client and validated 248/250 against `L1c_think`; drive it with
each L0 key flipped in turn and diff the hash. The two questions that did not
match in the pre-flight are the first place to look. Whatever it is, the fix
is on the L0 side (make the default path byte-identical), never on the L1 side,
and it needs a test that pins the prompt hash for at least the five probe
questions.

### 2.1 — R1: commit the recipe

`config/config.v2.yaml` is committed on `v2` together with this plan. It is
`config.yaml` plus a header comment and exactly two lines: `- agentic_v2`
under `evaluation.configurations`, and `trace.dir: results/runs_v2`. The
audit verified the two-line property by diff and used the file (with absolute
data paths, for the worktree) for the §1 probe, so it is known to run.

What remains for R1 is one test and one README paragraph: `tests/test_agentic_v2.py` already asserts the shipped config
does *not* name `agentic_v2`; add the mirror, that `config/config.v2.yaml`
does, that `configurations(load_config(path))` yields it, and that
`trace.dir` under it is not `results/runs` (so `discover_runs` cannot see its
output). Add the two-line recipe to the README's evaluation section — the
README does not mention `agentic_v2` at all.

### 2.2 — R2: say which baseline

In `run_eval.py`, the intro comment to `ABLATION_OVERRIDES["agentic_v2"]`.
Replace the bare figures with: "HotpotQA 0.452 → 0.524 and 2Wiki 0.316 →
0.456 against the loop's own paired `agentic_no_planner` baseline (runs
`L1c_base_*`, same code, same slice); against the report's `agentic_no_planner`
runs the deltas are +0.068 and +0.132. The two baselines are two runs of one
system and differ by the run-to-run churn `docs/improvement-loop.log.md`
documents." One sentence each; no other change to that block.

### 2.3 — R3: three lines in the log

Append a short coordinator block to `docs/improvement-loop.log.md`:

1. Provenance: the four `L1c_*` runs record `git_commit 03b8550` because the
   L1 diff was uncommitted at run time; the code they ran is the patch in
   `29fbdba`, which is byte-identical to `84bcc92` on `v2` (verified by the
   audit; state the command).
2. The probe result from §1, with its 5 qids and the hash outcome.
3. Correct the stop-condition paragraph in the final summary: condition 3
   ended the loop; condition 2 was *not* met because L4 was never attempted.
   After §3 below is done, add one line saying how L4 was decided, at which
   point condition 2 is genuinely met and the sentence can say so.

### 2.4 — R4: give `truncated` its meaning

Zero behaviour change. In `llm.py::chat`'s attempt loop, read
`_get(payload, "done_reason")` on every attempt. Carry it out two ways:

- `LLMResponse` gains `done_reason: str | None` (the last attempt's) and
  `hit_length_cap: bool` (any attempt's `done_reason == "length"`).
  `LLMFormatError` gains the same two, so the failure path records them too —
  the failing calls are the ones that matter.
- `agents/base.py::_trace_call` (both the success branch near line 400 and the
  error branch near line 386) sets `truncated=` from `hit_length_cap`. Also
  record `completion_tokens` — the attempt loop already sums `eval_count`,
  so this is a field on `LLMCallTrace` and one assignment. Two new fields on
  a frozen dataclass need defaults so old traces still load; check
  `eval/replication.py` and `eval/confidence.py`, which read traces, still
  run against `results/runs_v2/L1c_think_hotpotqa/traces.jsonl` unchanged.

Test: a stub client returning `done_reason: "length"` with empty content and a
full `thinking` field produces a trace with `truncated=True`,
`completion_chars=0`, `think_chars>0`. And the inverse.

Verify byte-identity the way L0 did: the change must not alter any prompt or
any answer, only what is *recorded*. `agentic_full` on the same five probe
questions, prompt hashes identical before and after.

### 2.5 — R5 (optional, flag off): stop paying twice for a spent budget

Only after R4, because it keys on `done_reason`. Behind
`llm.retry_after_length_cap: with_thinking_off | none | correction` defaulted
to `correction` (the evaluated behaviour). Under `with_thinking_off`, when an
attempt ends with `done_reason == "length"` and empty visible text, the next
attempt is issued with `think=False` passed straight to
`_invoke_with_fallback` — a loop-local decision inside `chat()`. **Do not
change `think_for`.** Its precedence is what keeps every other agent from
reasoning, and `agents/base.py::_think` passes a concrete value on every call,
so reordering it would switch reasoning on for the planner, retriever, KG and
verifier at once.

Expected value, stated so nobody mistakes it for an accuracy fix: EM +0.000
(measured 0/10 correct on the known failures), latency −290 s on each of the
~0.8–3.2% of questions that hit the cap, because two more 5,120-token attempts
are not made. Measure on the ten known failing qids (`results/runs_v2/_probes/
l3_probe_nothink_fallback.py` has the harness and the hash gate) and on
nothing else. It is a null with a latency benefit; write it as one.

## 3. L4, decided properly

The loop wrote "not attempted — its cost estimate predates L1". The audit
costed it from the kept runs' traces (`steps[*].llm_calls[*]` where
`agent == "synthesizer"`, summed per question):

| run | wall | synthesis share | L4 projection (×3 synthesis) | verdict against the 8 h ceiling |
|---|---|---|---|---|
| `L1c_think_hotpotqa` | 2.45 h | 2.04 h (83%) | **6.53 h** | fits |
| `L1c_think_twowiki` | 4.32 h | 3.19 h (74%) | **10.69 h** | does not fit |

So L4 is feasible on exactly the dataset with the open gap (HotpotQA 0.524
against a 0.55 target), and infeasible on the one that has already passed.
2Wiki is out of scope for L4; do not run it there.

**Design** (the accuracy plan's, made precise):

- The existing temperature-0 synthesis call stays and is sample 0. Two more
  samples at `temperature: 0.7`, same prompt, same evidence, thinking on (L1
  is kept; L4 is measured *with* it). +2 calls per question.
- Each sample's `answer` is normalised with the official HotpotQA
  `normalize_answer`; majority of three wins. **Ties go to sample 0.** This
  makes the baseline answer a strict fallback: only two samples agreeing
  *against* it can change the output, so the treatment can only differ from
  the baseline where two independent draws disagree with the deterministic
  one.
- A sample whose call ends with `done_reason == "length"` (R4 makes that
  visible) is dropped from the vote. Note the count.
- `answer_sentence`, citations and everything downstream come from whichever
  sample won. If sample 0 wins, the record is byte-identical to the baseline's
  except for `llm_calls` and latency.

Gate it as `agents.synthesizer.self_consistency: {samples: 3, temperature:
0.7}` with `samples: 1` the shipped default, and list a configuration
`agentic_v2_sc` in `config.v2.yaml` only, driven by an `ABLATION_OVERRIDES`
entry that adds the block to `agentic_v2`'s. `agentic_v2` itself does not
change.

**Offline first.** Before any GPU: replay the 26 HotpotQA questions where the
kept run needed a retry, and the 24 "slot error" questions the accuracy plan
identified (gold inside the model's own sentence, wrong span in `answer`), to
confirm the failure mode L4 targets is actually present in the kept run's
traces and count how many of the 119 wrong answers are of that kind. If it
is under ~15, L4 cannot move EM by 2 points even if it fixes every one, and
the run is not worth 6.5 h. Write the count down either way.

**Measure.** One run: HotpotQA, `--split eval`, n=250, `--config agentic_v2_sc`
under `config.v2.yaml`, `--run-id L4_sc_hotpotqa`, GPU lock held. Baseline:
`L1c_think_hotpotqa` *if* the §1 probe was 5/5 (same code path, same slice;
the reproducibility finding applies equally to both arms). Otherwise run
`agentic_v2` fresh first (2.45 h) and pair against that. Report the §8 block:
paired ΔEM with interval and McNemar, gained/lost, median and p90 latency, mean
`llm_calls`, the number of questions where the vote overturned sample 0 and
how many of those were right, and the count of samples dropped for hitting
the cap.

**Decision rule.** KEEP only if the interval excludes zero *and* the median
latency stays under 90 s. A point estimate of +2 with an interval spanning
zero is the noise floor the log already documents — DROP, and say the floor
is why. Either way the backlog is then fully decided and §2.3's sentence
about condition 2 becomes true.

**Budget.** Kill the run if it projects past 8 h at n=30. Expect roughly
6.5 h; the tail (p90 111 s per synthesis call) will make it longer than the
mean suggests.

## 4. The report chapter — draft, outside `report/`

Write `docs/report-chapter-draft.tex`: a chapter in the report's existing
style (read `report/chapters/ch4_implementation.tex` for conventions, do not
edit it) titled along the lines of "Improving the system, and what it cost".
Content, in this order: the kept configuration and the one mechanism; the
paired table against `agentic_full`, `agentic_no_planner` and `self_ask` on
both datasets with intervals; the cost table (median, p90, calls); the
`self_ask` tie stated in the first paragraph that mentions HotpotQA, not in a
footnote; the synthesis-failure caveat with the census; the reproducibility
finding with its mechanism (`prompt_eval_count` varying 1,218 / 2,977 / 3,671
for one prompt under Ollama's prefix cache) and what it implies for every
n=250 comparison in the report; the drops, one paragraph each, with L3's
wrong-system premise named as the coordinator's error; and, if §3 was run,
L4's outcome. Every number cites the run id it came from. The tables are
generated, not typed: extend `eval/tables.py` with a function that reads
`results/runs_v2` explicitly by path and writes `results/tables/v2_*.tex`,
so the chapter can `\input` them and a reader can regenerate them — but do
**not** add those tables to `discover_runs` or to the existing table set.

Hand the draft to the owner. It is their decision whether any of it enters the
report, and they said so.

## 5. Push — owner's action

`origin` is `https://github.com/PirMustafa/agentic-ir.git`. `main` is 7
commits ahead of `origin/main`; `v2` is 13. Nothing from this loop, and
nothing from the seven commits before it, exists anywhere but this laptop.
Do not push. Tell the owner the two commands (`git push origin main`,
`git push origin v2`) and stop.

## 6. Order of work, and definition of done

1. Read the probe result. If not 5/5 → R0, then stop and report.
2. R1, R2, R4 as three commits on `v2`, each with the full suite green
   (`430 passed` today; the count goes up).
3. R3's log block (a fourth commit; docs only).
4. §3's offline count. If it clears the bar, the L4 run; then its §8 block
   and the final line of R3.
5. R5 only if there is time and only as described; it is the lowest-value
   item here.
6. §4's draft chapter and generated tables (commit on `v2`).
7. Re-verify the invariants that must survive all of the above, and paste the
   outputs into the log:
   - `python -m agentic_ir.cli tables` from `main`: `git status --short
     results/tables` is empty.
   - The four evaluated agentic configs resolve identically on `main` and
     `v2` (the audit's `dump_cfg.py` method, or equivalent).
   - `--config agentic_v2` is rejected under `config/config.yaml` and
     accepted under `config/config.v2.yaml`.
   - `tests/test_report_integrity.py` still pins nine configurations.
   - First-attempt synthesis `prompt_sha1` on the five probe questions is
     unchanged by R4 (and by R5 with the flag off).
8. Report to the owner: what was committed (shas), what §3 decided, where the
   draft chapter is, and the two push commands.

Done means: every row of §2's table has a commit or a written reason it was
skipped; the backlog has no "not attempted" entry; and a reader who checks
out `v2` can reproduce 0.524 / 0.456 from the README alone.

## 7. Do not

- Edit anything under `report/`. The draft goes in `docs/`.
- Commit to `main`, or push anything.
- Change `think_for`'s precedence (see R5 for why).
- Run 2Wiki for L4. It does not fit and it has already passed.
- Run two Python workloads at once, or a run without the lock.
- Round a null up. The reproducibility finding puts the resolvable effect at
  roughly three points; an interval that includes zero is "unresolved", not
  "a trend".
- Pass `--split eval` more than once per configuration per dataset.
- Add `agentic_v2` or `agentic_v2_sc` to `CONFIGURATIONS`, to
  `config.yaml`'s `evaluation.configurations`, or to anything `tables.py`
  discovers. `tests/test_agentic_v2.py::test_agentic_v2_is_not_one_of_the_nine_reported_systems`
  is there to catch it.
