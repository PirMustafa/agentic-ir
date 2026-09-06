# Report integrity audit

**Scope.** Every quantitative claim in `report/main.tex` and `report/chapters/*.tex`,
checked against the artefacts under `results/` that should produce it.

**Method.** Chapters were read read-only. Generated fragments were regenerated with
`python -m agentic_ir.cli tables` and compared cell by cell; per-question values were
recomputed from `results/runs/*/traces.jsonl` where a claim asserts something the
aggregate cannot show. The mechanised subset of these checks lives in
`tests/test_report_integrity.py`.

**Status.** Sections 1-9 were written after the first `tables.py` corrections (section 9)
and describe the prose as it stood *before* the chapters were rewritten against them.
Section 10 is the second pass: it records what the rewrite changed, the findings the
first pass missed, and the run inventory as of that pass. Where sections 1-8 say a
sentence "must be rewritten", section 10 says whether it was.

**A moving target.** The evaluation sweep is running. Row counts, and therefore some
numbers here, change while it advances. Everything below carries the run id it was read
from, so a later reader can tell drift from disagreement.

---

## 1. Blockers

Ordered by how much of the report's argument rests on them.

### B1. "The generative baselines emit no supporting facts" is false. It is a harness bug.

| | |
|---|---|
| **Claim** | `report/main.tex:62-63` "0.000 for both generative ones, **which emit no supporting facts at all**" |
| | `ch4_implementation.tex:333-335` "`naive_rag` and `self_ask` both record 0.000, **because neither emits supporting facts at all**. That zero is structural" |
| | `ch4_implementation.tex:527-528` "`self_ask` and `naive_rag` supply no supporting facts whatsoever" |
| | `ch5_conclusions.tex:82-84` "0.000 for `self_ask` and `naive_rag`, **which emit no supporting facts to score**" |
| **Source** | `results/runs/naive_rag_*/traces.jsonl`, `results/runs/self_ask_*/traces.jsonl` |
| **Status** | **CONTRADICTED BY THE TRACES.** |

Both systems emit supporting facts on all 250 questions of all four runs. Every record
carries an `evidence` array whose entries have both `title` and `sent_id` -- exactly the
`(title, sent_id)` pairs SP-EM and SP-F1 are computed over.

The zero comes from an identifier-namespace mismatch inside the harness.
`run_eval.predicted_supporting_facts` (`src/agentic_ir/eval/run_eval.py:1102`) intersects
a record's `citations` with its `evidence` by `evidence_id`. `_BaselineAdapter`
(`src/agentic_ir/eval/run_eval.py:363`) stores baseline evidence as `e1..en`, while the
baselines cite **passage** ids `p1, p2`. The intersection is empty on every question, and
the documented fallback ("the whole evidence pool otherwise") never fires, because it is
guarded by `if cited` and `cited` is non-empty -- just unresolvable.

Rescored from the same traces using the evidence pool the fallback intends:

| run | SP-F1 as reported | SP-F1 the trace actually supports |
|---|---|---|
| `naive_rag_hotpotqa_20260902T120313Z` | 0.000 | **0.412** |
| `self_ask_hotpotqa_20260904T194457Z` | 0.000 | **0.405** |
| `naive_rag_twowiki_20260904T201400Z` | 0.000 | **0.327** |
| `self_ask_twowiki_20260904T210355Z` | 0.000 | **0.318** |

**What this costs the report.** The attribution result is the one defensible claim the
report makes, and it is stated as categorical. It is not. `agentic_full` reaches SP-F1
0.481 on HotpotQA against a corrected 0.412 for `naive_rag` -- a margin of 0.069, on the
same order as its margin over `hybrid_rerank`, and not the 0.481-against-nothing the
chapters describe. On 2WikiMultihopQA the corrected `naive_rag` figure (0.327) equals
`hybrid_rerank` exactly, which is the expected result since it retrieves through the same
hybrid pipeline, and is a good check that the rescoring is right rather than an artefact
of a different one.

Every sentence that turns this zero into a qualitative difference has to go, including
the practitioner recommendation at `ch4_implementation.tex:523-528` ("On that axis it is
not a marginal improvement over the baselines but a categorical one") and the abstract.

**Fix.** In `run_eval.py` (not owned by this pass): either write baseline citations in the
same namespace as baseline evidence in `_BaselineAdapter`, or make
`predicted_supporting_facts` fall back to the pool when the cited set resolves to
nothing. Then re-run `tables`. Until then `tables.py` marks these cells with a star and
the caption states that the zero describes the trace writer and not the system.

### B2. The paired bootstrap was run against a system that never answers.

| | |
|---|---|
| **Claim** | `ch4_implementation.tex:286-299`, the whole "Two qualifications" paragraph |
| **Source** | `results/tables/main_results.tex`, `tab:main-answer-hotpotqa` |
| **Status** | **FIXED IN THE TABLE; THE PROSE NOW DESCRIBES A TABLE THAT NO LONGER EXISTS.** |

The chapter is right about the defect and describes it accurately: the reference was
`hybrid_rerank`, which has no generation stage, so the "significant +0.510 improvement"
credited to `agentic_full` was its own F1 with a plus sign in front. The diagnosis was
correct, and the remedy available from the prose side -- warn the reader -- was the only
one there was.

The generator has now been corrected (section 9). Consequences for this paragraph:

* "Its 0.000 EM" -- there is no longer a 0.000 EM. Retrieval-only rows render `--`.
* "the significant +0.510 credited to `agentic_full`" -- no longer exists.
* "**No paired interval is generated between `agentic_full` and `self_ask`**" -- one is
  now generated, because `self_ask` is the chosen answer reference:
  HotpotQA dF1 = -0.093 [-0.145, -0.039]; 2Wiki -0.043 [-0.100, +0.014].
* "**so the gap cannot be called significant on this evidence**" -- **on HotpotQA it now
  can, and it goes against the agentic system.** The 95% interval of the paired
  difference excludes zero. `agentic_full` is *significantly worse* than `self_ask` on
  HotpotQA F1 and EM, and the table marks it accordingly. On 2WikiMultihopQA the interval
  still spans zero and the gap remains unresolved.

The revised paragraph is a stronger version of the chapter's own argument, not a weaker
one, but it must be rewritten: as it stands it tells the reader an interval was not
computed, beside a table that prints it.

### B3. The HotpotQA mean latency is one hung question, and the wall-clock budget did not stop it.

| | |
|---|---|
| **Claim** | `ch4_implementation.tex:358-374`; `ch5_conclusions.tex:148-149` |
| **Source** | `results/runs/agentic_full_hotpotqa_20260904T221533Z/traces.jsonl` |
| **Status** | **REPORTING BUG FIXED; UNDERLYING SYSTEM BUG STANDS AND IS NOT YET STATED AS ONE.** |

Measured over the 250 records of that run:

| statistic | value |
|---|---|
| median `latency_s` | 28.3 s |
| mean excluding the maximum | 32.3 s |
| second-worst question | 84.3 s |
| **mean as previously reported** | **295.1 s** |
| **single worst question** | **65,744.3 s = 18.3 h**, qid `5a7af32e55429931da12c99c` |

One question is about 20.5 h of a roughly 20.5 h run. The reported mean was that question
divided by 250 and described nothing that happens per question; it inflated the agentic
system's cost by roughly an order of magnitude, in the column a reader is most likely to
quote.

`tables.py` now reports the median with the maximum beside it, on both datasets and in
both `tab:agent-metrics-*` and `tab:ablations-*` (section 9). Every other column in that
table is a bounded count -- the orchestrator caps calls, sub-queries and cycles -- so no
other column is exposed to this. That was checked across every run rather than assumed.

**This is a system defect, not only a reporting one.** `config/config.yaml` sets
`orchestrator.max_wall_clock_s: 300`. One question ran **220 times that budget**. The
guard is evidently evaluated only at state transitions and cannot pre-empt a call that
blocks inside one. `ch4_implementation.tex:369-374` says this correctly and should keep
saying it, with the concrete qid attached -- a claim of this kind is worth much more when
a reader can go and look at the record. The correction it names (bound a single tool
invocation, not only the question) remains unimplemented and belongs in the limitations.

### B4. Chapter 2 and Chapter 5 disagree about the ablation sample, and Chapter 2 is wrong.

| | |
|---|---|
| **Claim** | `ch2_data.tex:294-296` "The three component ablations run on a fixed **150-question subsample** of the same 250 to keep the grid inside its time budget, and the results table says so in its caption." |
| **Counter-claim** | `ch5_conclusions.tex:218-219` "the ablations run against the **same 250-question evaluation split rather than a subsample**" |
| **Source** | `results/runs/agentic_*/meta.json`, `traces.jsonl` |
| **Status** | **INTERNAL CONTRADICTION. Chapter 5 is correct.** |

No ablation run on disk has n = 150. Sizes at the time of writing: `agentic_full` 250,
`agentic_no_verifier` 250, `agentic_no_planner` 58 (in flight), `agentic_no_kg` not
started. The `ablations.tex` caption is generated from the runs and now states their
sizes explicitly, so it correctly refuses to repeat the 150. `ch2_data.tex:296` therefore
promises the reader a caption sentence that cannot honestly be written.

`tests/test_report_integrity.py::test_ablation_subsample_promise_is_kept` pins this: it
requires either that the caption keep the promise or that this document record that it
cannot. Deleting the subsample sentence from `ch2_data.tex:294-296` retires both the
contradiction and the skip.

---

## 2. Numbers in the prose that no generated table contains

Checked by extracting every decimal literal from the chapters and testing membership in
the union of `results/tables/*.tex`. Corpus, index, KG and hardware figures resolve
against `data/processed/*_corpus_stats.json`, `data/indexes/*_index_stats.json`,
`data/indexes/*_kg_meta.json` and `docs/architecture.md`, and are excluded below. What
remains:

| location | value | verdict |
|---|---|---|
| `ch4_implementation.tex:359` | `41.6 s` (2Wiki `agentic_full` latency) | **stale** -- a mean; the table now prints median **35.4** |
| `ch4_implementation.tex:360` | `17.1 s` (`self_ask`) | **stale** -- median is **13.2** |
| `ch4_implementation.tex:360` | `7.4 s` (`hybrid_rerank`) | **stale** -- median is **8.3** |
| `ch4_implementation.tex:363` | `295.1 s` | **withdrawn** -- see B3; in no table now |
| `ch5_conclusions.tex:149` | `41.6` / `17.1` / `7.4` | the same three, repeated |
| `ch5_conclusions.tex:261-275` | `0.332`, `0.636`, `0.340`, `0.339`, `-0.510`, `0.633`, `0.54`, `[0.45, 0.64]`, `0.19` | **sourced but hand-typed** -- see 2.1 |
| `ch4_implementation.tex:460`, `ch5_conclusions.tex:305` | `0.0008` NLI entailment on the worked example | **unsourced** -- a development observation held in no artefact under `results/` |
| `ch4_implementation.tex:50` | Jaccard `0.85` re-plan similarity gate | a design constant that is not in `config/config.yaml` either; worth exposing as config |

The consequential ones are the latency figures. `ch4_implementation.tex:361-362` also
derives ratios from them -- "roughly two and a half times the decomposing baseline and
five and a half times the non-agentic one". On medians those become 2.7x and **4.3x**, so
the second ratio changes materially.

### 2.1 The calibration section is hand-typed beside a generated fragment nobody inputs

`results/tables/calibration.tex` exists, is generated by `src/agentic_ir/eval/calibrate.py`
from `results/calibration/hotpotqa_threshold.json`, defines `tab:calibration` and
`tab:calibration-reliability`, and is **never used with \input** by any chapter. Meanwhile
`ch5_conclusions.tex:258-290` types nine of its numbers into the running text.

Every one of them checks out against the JSON (ECE 0.3316 -> 0.332, MCE 0.6355 -> 0.636,
base rate 0.34, Brier 0.33883 -> 0.339, Brier skill -0.5099 -> -0.510, AUC 0.6328 ->
0.633, Youden threshold 0.54, CI [0.45, 0.64], width 0.19, grid [0.40, 0.75] step 0.01,
n = 50). So this is not a wrong-number defect. It is a violation of the rule the report
states in its own voice at `ch4_implementation.tex:258-260` -- "no number in this report
is typed by hand" -- and that rule is what makes the other numbers trustworthy. Input the
fragment in Chapter 5 and cite the table.

Note also that Chapter 5's calibration numbers come from
`agentic_full_hotpotqa_20260904T184256Z`, the **50-question calibration slice** run, and
not from either 250-question evaluation run. That is correct and deliberate -- the point
of the slice is disjointness -- but the run id differs from every other number in the
report and the text should say which run it is reading.

---

## 3. Claims of outcome the paired bootstrap does not support at 95%

| location | claim | what the bootstrap says |
|---|---|---|
| `ch4_implementation.tex:296-297` | the `agentic_full`/`self_ask` gap "cannot be called significant on this evidence" | **Now it can, on HotpotQA**: dF1 -0.093 [-0.145, -0.039], interval excludes zero. On 2Wiki (-0.043 [-0.100, +0.014]) the sentence is still right. |
| `ch5_conclusions.tex:215-217` | "the system's deficit against `self_ask` cannot be separated from noise" | Same. It argues from the *overlap of per-system CIs*, which is the wrong test for paired data and is more conservative than the paired interval now printed. |
| `ch4_implementation.tex:280-281` | "the full system clears `naive_rag`'s 0.152 and 0.200 **by a wide margin**" (2Wiki) | **Not established.** No paired bootstrap between `agentic_full` and `naive_rag` is computed. Their deltas against the common reference are -0.043 [-0.100, +0.014] and -0.110 [-0.160, -0.063], which overlap. "Wide margin" is a point-estimate reading. |
| `ch4_implementation.tex:495-497` | `synthesis_error` "roughly **double** the corresponding rate for `self_ask`" | **Understated by a factor of about 2.** HotpotQA 0.228 vs 0.064 (3.6x); 2Wiki 0.236 vs 0.076 (3.1x). Directionally right and numerically too kind to the agentic system. |
| `ch4_implementation.tex:307-308` | both agentic nDCG@10 figures "carry the marker for significantly worse" | **Correct**, verified in `tab:main-retrieval-*`. |
| `ch4_implementation.tex:328-329` | both SP-F1 margins over the strongest retrieval baseline "are significant" | **Correct**, both bold. Note this is the margin over `hybrid_rerank`; the margin over the *generative* baselines is the one B1 dissolves. |

Nothing in the chapters asserts an outcome for a configuration that has no run. The
mechanised check (`test_no_chapter_asserts_an_agentic_result_that_has_not_been_run`)
passes, and the placeholder blocks at `ch4_implementation.tex:403-407` and
`ch5_conclusions.tex:132-135` correctly hold the ablation conclusions open.

---

## 4. Prose that is stale relative to the runs

| location | says | is now |
|---|---|---|
| `ch4_implementation.tex:391-394` | `agentic_no_planner` and `agentic_no_kg` "have not been run on either dataset"; `agentic_no_verifier` "scored on 202 of the 250" | `agentic_no_verifier` on HotpotQA is **complete at 250**; `agentic_no_planner` on HotpotQA is **in flight** |
| `ch4_implementation.tex:396-401` | the `agentic_no_verifier` row is "provisional in a specific and disqualifying way", its delta "not yet a paired comparison at all", markers "should be disregarded" | **No longer true.** n = 250 and the paired comparison is valid: dF1 -0.037 [-0.065, -0.011], p = 0.008. Removing the verifier makes the system *significantly worse* -- the project's central measurement, now available on HotpotQA. |
| `ch4_implementation.tex:407` | "Do not reuse the partial n=202 row" | superseded |
| `ch5_conclusions.tex:123-131` | "The marginal effect ... cannot yet be reported" | reportable on HotpotQA; still open on 2Wiki |
| `ch5_conclusions.tex:219-222` | "202 of those questions on HotpotQA and none on 2WikiMultihopQA, while `agentic_no_planner` and `agentic_no_kg` had not been run" | as above |
| `ch5_conclusions.tex:223` | "Six of the nine configurations promised in Chapter 1 are complete on both datasets" | true today (the five baselines plus `agentic_full`), but it will need re-counting as the sweep lands, and `agentic_no_verifier` is now complete on HotpotQA alone |

These are the good kind of defect: the numbers moved in the direction the chapters
predicted, so the caveats can be replaced with results rather than deleted.

---

## 5. Table staleness

`results/tables/*.tex` were regenerated during this pass. They go stale again within
minutes, because the sweep is writing. Drift at the time of writing, from
`test_report_integrity.py::table_drift`:

```
ablations.tex:8       agentic_no_planner_hotpotqa_20260905T224906Z  table n=44, traces.jsonl 58
agent_metrics.tex:8   agentic_no_planner_hotpotqa_20260905T224906Z  table n=44, traces.jsonl 58
dataset_stats.tex:8   agentic_no_planner_hotpotqa_20260905T224906Z  table n=44, traces.jsonl 58
error_analysis.tex:8  agentic_no_planner_hotpotqa_20260905T224906Z  table n=44, traces.jsonl 58
main_results.tex:8    agentic_no_planner_hotpotqa_20260905T224906Z  table n=44, traces.jsonl 58
```

This is a build-product lag and not a false claim: the affected row is marked preliminary
in every table and carries no delta, no p-value and no significance mark. **Regenerate
with `python -m agentic_ir.cli tables` immediately before the final PDF build, and re-read
section 4 afterwards**, because the numbers the chapters quote will have moved.

Run inventory at the time of writing (largest run per configuration and dataset; earlier
same-configuration directories are superseded and hold no traces):

| run id | n |
|---|---|
| `agentic_full_hotpotqa_20260904T221533Z` | 250 |
| `agentic_full_twowiki_20260905T184526Z` | 250 |
| `agentic_no_verifier_hotpotqa_20260905T213927Z` | 250 |
| `agentic_no_planner_hotpotqa_20260905T224906Z` | 58 and rising |
| `agentic_full_hotpotqa_20260904T184256Z` | 50, calibration slice; feeds `calibration.tex` only |
| `bm25_only_hotpotqa_20260902T113736Z`, `bm25_only_twowiki_20260902T113905Z` | 250, 250 |
| `dense_only_hotpotqa_20260902T113746Z`, `dense_only_twowiki_20260902T113913Z` | 250, 250 |
| `hybrid_rerank_hotpotqa_20260902T114415Z`, `hybrid_rerank_twowiki_20260904T184235Z` | 250, 250 |
| `naive_rag_hotpotqa_20260902T120313Z`, `naive_rag_twowiki_20260904T201400Z` | 250, 250 |
| `self_ask_hotpotqa_20260904T194457Z`, `self_ask_twowiki_20260904T210355Z` | 250, 250 |

Seven development runs at n = 1..10 are quarantined in `results/runs/_smoke/` and are
verified absent from every table and from the live run directory.

---

## 6. Evaluation configurations that are still incomplete

Everything in this section is provisional. A reader should treat any conclusion resting
on it as unsupported.

| configuration | HotpotQA | 2WikiMultihopQA |
|---|---|---|
| `bm25_only` | complete (250) | complete (250) |
| `dense_only` | complete (250) | complete (250) |
| `hybrid_rerank` | complete (250) | complete (250) |
| `naive_rag` | complete (250) | complete (250) |
| `self_ask` | complete (250) | complete (250) |
| `agentic_full` | complete (250) | complete (250) |
| `agentic_no_verifier` | **complete (250)** -- new since the chapters were written | **not started** |
| `agentic_no_planner` | **in flight (58/250)** | **not started** |
| `agentic_no_kg` | **not started** | **not started** |

**What this means for the report's central claim.** The backward edge's contribution is
now measurable on HotpotQA and only on HotpotQA: dF1 = -0.037 [-0.065, -0.011], p = 0.008
against `agentic_full`. It is not yet a two-dataset result, and the report's own standard
elsewhere is that a single dataset does not settle a direction. The knowledge-graph
ablation, on which `ch4_implementation.tex:414-419` explicitly reserves judgement, has no
evidence at all.

One provisional signal is worth flagging in advance so it is not a surprise. At the
time of writing the in-flight `agentic_no_planner` row on HotpotQA sits *above*
`agentic_full` on both answer metrics (EM 0.472 / F1 0.575 at n = 126, against
0.432 / 0.510 at n = 250), at roughly a third of the model calls. It is withheld from
every comparison, and a prefix of the slice is not the slice, so nothing follows from
it yet. But if it holds to 250 it says the planner is costing the system accuracy,
which is a different and larger conclusion than the one Chapter 4 currently
anticipates at lines 411-414, where the planner ablation is described as fixing the
floor. Chapter 4 should not be written to assume the ablation lands below the full
system.

Preliminary rows are marked in every generated table, are excluded from every paired
bootstrap, and are annotated `PRELIMINARY` on their provenance line in each fragment's
header. That annotation is what
`test_no_table_row_reports_a_sample_smaller_than_the_eval_set` enforces.

---

## 7. Claims verified as correct

Recorded so a reader knows the scope of what was checked, not only what failed. All
verified against the regenerated fragments and, where marked with a dagger, recomputed
from the traces.

* `main.tex:58-59`, `ch4:275-280` -- EM/F1 for `self_ask` (0.504/0.603, 0.264/0.310),
  `naive_rag` (0.496/0.581, 0.152/0.200), `agentic_full` (0.432/0.510, 0.224/0.267).
* `ch4:265-268` -- Recall@10 ladder 0.774 / 0.816 / 0.876 and SP-F1 0.330 -> 0.412.
* `ch4:305-307` -- nDCG@10 0.763 [0.734, 0.791] and 0.723 [0.697, 0.748] against 0.835,
  0.837, 0.756, 0.790.
* `ch4:316-317`, `ch5:88` -- `agentic_full` and `hybrid_rerank` tie exactly on 2Wiki
  Recall@10 at 0.727 while nDCG@10 differs by 3.3 points. A precise, well-chosen
  observation.
* `ch4:325-330` -- SP-F1 0.481 / 0.421 against 0.412, 0.377, 0.330 and 0.327, 0.314,
  0.273; SP-EM 0.016 -> 0.088 and 0.016 -> 0.124. (The comparison against the
  *generative* baselines in the same paragraph is B1.)
* `ch4:338-339` -- citation grounding 0.960 and 0.976.
* `ch4:344-349`, `ch5:145-147`, `ch5:156-157` -- 4.26 / 3.80 model calls, 1.24 / 1.46 for
  `self_ask`, 14.83 / 16.00 tool calls against 2.41 / 2.86, plan depth 1.56 / 1.76,
  sub-queries 1.80 / 2.26, Saved 7.10 / 7.19. The "more calls avoided than made" claim
  holds on both datasets.
* `ch4:422-423`, `ch5:105`, `main.tex:65` -- re-plan rate 0.472 and 0.392 (47.2%, 39.2%).
* `ch4:441-442` -- 142 and 194 errors, 16 and 23 unlabelled.
* `ch4:444-446` -- zero parse failures and zero budget exhaustions in every configuration
  on both datasets.
* `ch4:449-452`, `ch5:109-111` (dagger) -- 2 recoverable and 2 false rejects on HotpotQA,
  5 and 5 on 2Wiki; the counts are the same questions, as claimed.
* `ch4:489-492` -- `decomposition_error` 0.248 -> 0.064 and 0.552 -> 0.088;
  `bridge_link_failure` 0.552 -> 0.068 and 0.200 -> 0.028.
* `ch4:505-506`, `ch5:117` -- 92 and 154 verifier false accepts, 2 and 5 false rejects.
* `ch2:287` -- the 1.96 * sqrt(0.25/250) = 0.062 arithmetic.
* `ch1:91-92` (dagger) -- "of the 66,581 passages ... exactly zero contain both dates"
  holds against `data/processed/hotpotqa_corpus.jsonl`.
* Corpus, index and KG statistics throughout Chapters 2 and 4 resolve against
  `*_corpus_stats.json`, `*_index_stats.json` and `*_kg_meta.json`.
* Every input resolves, every cite key is defined, every ref has a label, no label is
  duplicated. Neither train-split-only worked-example title reaches the PDF, and the
  superseded 30-45 tok/s throughput figure does not appear.

---

## 8. Smaller items

* **23 labels are defined and never referenced**, including `tab:main-retrieval-hotpotqa`,
  `tab:main-answer-twowiki`, both `tab:agent-metrics-*`, both `tab:ablations-*`, both
  `tab:error-analysis-*`, `tab:dataset-corpus`, `sec:results` and `sec:errors`. Nine of
  the report's generated tables are printed but never pointed at from the prose that
  discusses them. Not a correctness defect, but it makes the tables harder to find than
  they should be.
* `ch4:96` and `ch2:258` both quote a 3.75 s mean planner call, derived from 52 tok/s.
  Consistent with each other and with `docs/architecture.md`, but derived rather than
  measured -- worth saying so once.
* `ch4:110` "the first call after a model load costs approximately 9.5 s in CUDA warm-up"
  and `ch4:101` "roughly 1.4 GiB" free VRAM are hardware observations with no artefact
  behind them. Fine as reported observations; they should not sit in the same register as
  measured results.
* `ch5:118` says the loop fires on "47%" where the table says 0.472 and `main.tex:65` says
  47.2%. Harmless rounding, inconsistent presentation.

---

## 9. Corrections made to `tables.py` this pass

For traceability; the rationale is in the module and function docstrings.

1. **Retrieval-only systems no longer report an answer score.** A run whose trace holds no
   non-empty `final_answer` renders `--` in EM, F1, the F1 CI and dF1, and is excluded
   from every paired bootstrap on an answer metric. Detected from the trace, not from a
   hard-coded list. The captions state that this `--` means *does not attempt answers*,
   distinctly from the whole-row `--` of an unrun configuration.
2. **EM and F1 are referenced to a system that answers.** `hybrid_rerank` remains the
   reference for retrieval and supporting facts; the answer columns are referenced to the
   strongest complete, non-agentic, answering run, chosen from the data and named in the
   caption. Currently `self_ask`. This is what makes B2's interval exist.
3. **Under-powered runs keep their estimates and lose their comparisons.** A run below
   `datasets.{dataset}.eval_sample` is marked preliminary, is excluded from every paired
   bootstrap in either direction, cannot be a reference, and renders `--` for delta and p.
   The rule is a comparison against the configured slice size, so a growing run rejoins
   the comparison by itself -- nothing is hardcoded, and the `agentic_no_verifier` row
   demonstrated it by rejoining mid-pass.
4. **Latency is a median with its maximum beside it**, in `tab:agent-metrics-*` and
   `tab:ablations-*`, with a generated caption sentence naming any question that overran
   `orchestrator.max_wall_clock_s`, with its qid and duration. Every other column was
   checked for the same exposure across every run and is a bounded count.
5. **Supporting-fact scores that are harness artefacts are marked with a star** and
   explained in the caption, rather than blanked. Blanking would have hidden B1.
6. **Provenance lines annotate under-powered runs** with `PRELIMINARY` and the slice size
   they fall short of.

Corresponding checks in `tests/test_report_integrity.py`:
`test_retrieval_only_systems_report_no_answer_score`,
`test_captions_distinguish_not_run_from_does_not_answer`,
`test_incomplete_runs_are_not_given_deltas_or_significance_marks`, and a tightened
`test_no_table_row_reports_a_sample_smaller_than_the_eval_set` that requires the
preliminary marking on the source line naming the run rather than anywhere in the file.

**Not fixed here, because the code is not owned by this pass:** B1 lives in `run_eval.py`;
B3's wall-clock guard lives in the orchestrator.


---

## 10. Second pass: the rewrite, and what the first pass missed

Written 2026-09-06 after `report/main.tex` and every chapter were revised against the
regenerated tables. Everything in sections 1-8 that named a stale sentence has been
acted on; the disposition of each is listed first, then the findings the first pass did
not catch, then the run inventory.

### 10.1 Disposition of sections 1-8

| finding | disposition |
|---|---|
| B1 "generative baselines emit no supporting facts" | **Rewritten** in `main.tex` (abstract), `ch4` results and discussion, `ch5` findings. The prose now says it was a harness defect, quotes the starred pool-scored figures, states that the agentic (cited-subset) and baseline (whole-pool) protocols trade precision against recall in opposite directions and that one SP-F1 does not rank them, and keeps only citation grounding (0.960/0.976 vs 0.000) as categorical. |
| B2 bootstrap against a non-answering reference | **Rewritten** in `ch4` and `ch5`. Reference is `self_ask`; the paired dF1 -0.093 [-0.145, -0.039] (HotpotQA, significant) and -0.043 [-0.100, +0.014] (2Wiki, not) are quoted; the "cannot be called significant" sentence is inverted for HotpotQA. |
| B3 latency mean | **Rewritten** in `ch4` and `ch5` on medians (28.3 / 35.4 s vs 4.5 / 13.2 s `self_ask`, 4.9 / 8.3 s `hybrid_rerank`), the 65,744.3 s maximum and its qid quoted, the cooperative guard stated as a system defect in `ch4` results and `ch5` limitations. |
| B4 150-question ablation subsample | **Deleted** from `ch2`; replaced by the statement that ablations run on the same 250 and that short rows are marked preliminary. |
| section 2 stale latency literals | all replaced; `295.1`, `41.6`, `17.1`, `7.4` no longer appear. |
| section 2.1 calibration fragment never input | **Fixed**: `ch4` gains a `Verifier Threshold Calibration` section that inputs `calibration.tex` and names the run `agentic_full_hotpotqa_20260904T184256Z`; `ch5` refs `tab:calibration` and `tab:calibration-reliability` and names the run. |
| section 3 "wide margin" over `naive_rag` | softened to a point-estimate ordering with the note that no paired interval exists. |
| section 3 `synthesis_error` "roughly double" | corrected to "more than three times" with the `self_ask` rates (0.064 / 0.076) quoted. |
| section 4 stale ablation prose | **Rewritten**; see 10.2 (a). |
| section 8 unreferenced tables | every generated table is now referenced from the prose that discusses it (`tab:main-*` x4, `tab:agent-metrics-*` x2, `tab:ablations-*` x2, `tab:error-analysis-*` x2, `tab:dataset-corpus`, `tab:calibration*` x2). |
| section 8 "47%" | now 47.2% everywhere. |

### 10.2 Findings the first pass missed

**(a) The central ablation is answered and the report said it was not.**
`agentic_no_verifier` on HotpotQA is complete at n=250: EM 0.396 / F1 0.474 against
0.432 / 0.510, dF1 -0.037 [-0.065, -0.011], p = 0.008, at 2.06 LLM calls and 16.5 s
median against 4.26 and 28.3 s. `agentic_no_planner` is also complete at n=250 and
lands *above* the full system (EM 0.456 / F1 0.543, dF1 +0.033 [-0.017, +0.086],
p = 0.186, not significant) at 1.35 calls and 9.7 s. Both are now written into `ch4`
(Ablation Study) and `ch5` (Empirical Findings) with their sample sizes. One
`% PLACEHOLDER` remains, in `ch4`, for `agentic_no_kg` (HotpotQA, n=162 at the time of
writing) and all three ablations on 2WikiMultihopQA (not started).

**(b) "Removing the planner reduces the system to `hybrid_rerank` by construction" was
false.** Stated in `ch3:251-256`, `ch4:381-384`, `ch4:411-414`, `ch5:52-55`. The
agent-metrics table contradicts it: `agentic_no_planner` makes 1.35 LLM calls, re-plans
on 18.8% of questions and grounds citations at 0.936; `hybrid_rerank` makes 0 calls and
no answer. The ablations caption itself says the floor requires removing planning,
synthesis *and* verification. Corrected in all four places; `ch3` and `ch5` say the
earlier draft made the claim and withdraw it.

**(c) `ch5:213-217` mislabelled the F1 interval as an EM interval and argued from the
overlap of marginal intervals**, the wrong test for paired data. Corrected: the paired
interval is quoted and the overlap argument is described as the mistake it was.

**(d) `ch3:402-405` said the threshold was "swept on the calibration slice and frozen
before the evaluation run".** `ch5:251-257` and `calibration.tex` say the sweep was
post hoc, over confidences produced at 0.55. `ch3` now agrees with `ch5`; `ch2` now
says the slice's pre-run purpose was "intended" and points to `ch5`.

**(e) The rule-vs-LLM routing agreement experiment (`ch3:298-304`) was written in the
present tense with three numbers.** It was never run, and cannot be run as coded:
`RetrievalAgent._select` returns on `selector == "planner_hint"` *before* it consults
`heuristic_shortcut`, so with a hint on every sub-query the selection prompt is
unreachable. Rewritten as unrun and unrunnable; the `ch5` placeholder is deleted.

**(f) The seven-rule router was a one-rule router on this data.** In
`agentic_full_hotpotqa_20260904T221533Z` the Planner attached a `tool_hint` to 854/854
sub-queries; R1 fired on 575/575 routing decisions and R2-R7 never executed. Every
place that narrated the seven rules as operating (`main.tex` abstract, `ch1:165`,
`ch3:271-295`, `ch4:151`, `ch4:534-543`, `ch5:13`) now carries the qualification, and
`ch3`/`ch4` write it as a finding about the planner/router division of labour.
These counts are trace-derived and appear in no generated table; `ch3` attributes them
to the run.

**(g) The LLM response cache does not exist.** `config/config.yaml` declares
`llm.cache`, but no sqlite code exists anywhere, `cache_hit=False` is hard-coded in
`agents/base.py:444`, `llm_cache_hits` is always 0 and `meta.json` reports
`cache_cold: true` only because no file exists. `ch4:109-114` ("the harness records
whether the response cache was cold") and `ch4:168` ("cache state") are rewritten to
say no cache is implemented and every latency is uncached by absence, not by
configuration. The agent-metrics captions ("cache hits included", "only comparable
between cold-cache runs") are owned by `tables.py`.

**(h) `llm_calls_saved` was inflated.** Of the 1,774 credited saves in the HotpotQA
`agentic_full` run, 635 were KG entity-linking calls this configuration never makes,
and 347 extraction-ladder saves were uncounted; the regenerated `agent_metrics.tex` prints 2.60 (HotpotQA) and 3.10 (2Wiki) per
question rather than 7.10 / 7.19, with the per-rule decomposition in its caption. `ch4:348-356`, `ch5:156-163` and the abstract's "displaced
more model calls than the system spent" are rewritten to the corrected definition and
the sentence is withdrawn. `ch4` and `ch5` quote the new figures from the regenerated table.

**(i) The Verifier entails `answer_sentence`, never `answer`.** `verifier.py:318`:
`hypothesis = cand.answer_sentence.strip() or cand.answer.strip()`. Demonstration in
the trace: qid `5a7af32e55429931da12c99c` answered "New York City" (gold: Brooklyn,
New York) with an answer sentence copied from unrelated evidence; nli_support 0.995,
citation_grounding 1.0, retrieval_agreement 0.0, accepted at confidence 0.82. Written
into `ch4` error analysis and `ch5` limitations with the qid. The component scores are
trace-derived and in no table.

**(j) Re-planning overwrites cycle-0 state.** Sub-query ids restart at `q1` per
revision and `QuestionState.results/kg_results/answers/bridge_entities` are keyed by id.
Of 118 re-planned HotpotQA questions, 115 lost cycle-0 retrieval and 92 carry stale
ids. Consequences: retrieval metrics for the 47.2% of questions that re-planned are over
a last-cycle-plus-stale pool, and the trace is not the complete audit record
`architecture.md` section 1.3 promises. Added to `ch4` results (retrieval paragraph),
`ch5` limitations, and `docs/architecture.md` section 1.10 as a known defect with the
fix (namespace state by cycle). Counts are trace-derived.

**(k) The wall-clock budget is cooperative.** `Budget.wallclock_exceeded()` is checked
at `orchestrator.py:241` (between DAG nodes) and `:561` (re-plan guard G4); no call can
be pre-empted, and the NLI / reranker paths have no timeout. Stated in `ch4` results
and as a `ch5` limitation.

**(l) `ch1:202-204` "nine configurations are run"** -- nine on HotpotQA (one partial),
six on 2Wiki. Rephrased to stay true and to defer to `ch4` for the exact status.

**(m) `ch2:258-259` "22-25 s" per agentic question** was a pre-run estimate; the
measured medians are 28.3 / 35.4 s. Now labelled as the estimate it was.

**(n) `ch1` section 1.4 had no mapping table for the brief's Table 1.** Added
`tab:task-mapping` (task, agent/component, section and table) and one paragraph
acknowledging the brief's "external APIs for real-time enrichment" clause and stating
the deliberate fully-local deviation. `ch3` (Architectural Summary) points at it.

**(o) `report/README.md` still called every chapter a skeleton**, and `report/main.pdf`
in the tree was built from the pre-rewrite chapters. README fixed; PDF rebuilt at the
end of this pass.

### 10.3 Numbers in the prose that no generated table contains (second pass)

All corpus/index/KG/hardware figures are as in section 2. Trace-derived figures that
the chapters now quote, each attributed to its run in the text:

| location | value | source |
|---|---|---|
| `ch3` Tool-Augmented Retrieval | 854 sub-queries with a hint; R1 fired 575/575 | `agentic_full_hotpotqa_20260904T221533Z/traces.jsonl` |
| `ch4` Error Analysis, `ch5` Limitations | qid `5a7af32e55429931da12c99c`; NLI 0.995, grounding 1.0, retrieval agreement 0.0, confidence 0.82; "New York City" vs "Brooklyn, New York" | same run |
| `ch5` Limitations, `architecture.md` 1.10 | 118 re-planned, 115 lost cycle-0 results, 92 stale ids | same run |
| `ch4` Results | "roughly 220 times its budget" | 65,744.3 s / 300 s, both in `agent_metrics.tex` |
| `ch5` Limitations | ECE 0.332, MCE 0.636, base rate 0.340, Brier 0.339, skill -0.510, AUC 0.633, J-optimal 0.54, CI [0.45, 0.64], width 0.19 | now backed by `calibration.tex`, which `ch4` inputs |

### 10.4 Run inventory (second pass)

| configuration | HotpotQA | 2WikiMultihopQA |
|---|---|---|
| `bm25_only` | 250 (`20260902T113736Z`) | 250 (`20260902T113905Z`) |
| `dense_only` | 250 (`20260902T113746Z`) | 250 (`20260902T113913Z`) |
| `hybrid_rerank` | 250 (`20260902T114415Z`) | 250 (`20260904T184235Z`) |
| `naive_rag` | 250 (`20260902T120313Z`) | 250 (`20260904T201400Z`) |
| `self_ask` | 250 (`20260904T194457Z`) | 250 (`20260904T210355Z`) |
| `agentic_full` | 250 (`20260904T221533Z`) | 250 (`20260905T184526Z`) |
| `agentic_no_verifier` | **250** (`20260905T213927Z`) | not started |
| `agentic_no_planner` | **250** (`20260905T224906Z`) | not started |
| `agentic_no_kg` | in flight: `agentic_no_kg_hotpotqa_20260905T232702Z`, 162 rows when the tables were regenerated at 02:06, 184 at 02:11, still rising; every table marks the row preliminary and it drifts from `traces.jsonl` between regenerations | not started |
| `agentic_full` calibration slice | 50 (`20260904T184256Z`), feeds `calibration.tex` only | -- |

Sentences that depend on the rows still open are all in `ch4` (Ablation Study, the one
`% PLACEHOLDER`) and `ch5` (Empirical Findings, Limitations), and each says so.

### 10.5 Run inventory (third pass, 2026-09-06)

The HotpotQA ablation grid is now complete. `agentic_no_kg_hotpotqa_20260905T232702Z`
finished at 250 and is no longer preliminary anywhere: the `% PLACEHOLDER` in
`ch4` is filled from it, and `ch5`'s Empirical Findings and Limitations are
rewritten against it. Its result is a null -- EM 0.404, F1 0.501, paired
dF1 -0.009 [-0.048, +0.029], p = 0.646, with Recall@10 0.820 against the full
system's 0.812 and nDCG@10 0.766 [0.737, 0.793] against 0.763 [0.734, 0.791] --
so the knowledge graph cannot be shown to help on this dataset while accounting
for almost half the tool calls and more than half the median latency.

| configuration | HotpotQA | 2WikiMultihopQA |
|---|---|---|
| `bm25_only` | 250 (`20260902T113736Z`) | 250 (`20260902T113905Z`) |
| `dense_only` | 250 (`20260902T113746Z`) | 250 (`20260902T113913Z`) |
| `hybrid_rerank` | 250 (`20260902T114415Z`) | 250 (`20260904T184235Z`) |
| `naive_rag` | 250 (`20260902T120313Z`) | 250 (`20260904T201400Z`) |
| `self_ask` | 250 (`20260904T194457Z`) | 250 (`20260904T210355Z`) |
| `agentic_full` | 250 (`20260904T221533Z`) | 250 (`20260905T184526Z`) |
| `agentic_no_verifier` | **250** (`20260905T213927Z`) | **in flight**: `agentic_no_verifier_twowiki_20260906T103608Z`, 12 rows when the sweep was inspected and 16 when the tables were regenerated, still rising; every table marks the row preliminary, withholds its deltas and significance marks, and it drifts from `traces.jsonl` between regenerations |
| `agentic_no_planner` | **250** (`20260905T224906Z`) | not started |
| `agentic_no_kg` | **250** (`20260905T232702Z`) | not started |
| `agentic_full` calibration slice | 50 (`20260904T184256Z`), feeds `calibration.tex` only | -- |

**Guard fixed in this pass.**
`test_incomplete_runs_are_not_given_deltas_or_significance_marks` keyed its
`partial` set on the configuration name alone. `agentic_no_verifier` is complete
on HotpotQA and in flight on 2WikiMultihopQA, so the guard condemned the finished
HotpotQA deltas -- dF1 -0.037 [-0.065, -0.011], p = 0.008, the central result of
the report -- as comparisons against a 16-question prefix that lives in a
different table. The set is now keyed on `(dataset, configuration)`, the dataset
read from the table's own caption, and the check fails loudly rather than
silently if a caption ever stops naming one.

Sentences that depend on the rows still open are now confined to `ch4`
(Ablation Study, 2WikiMultihopQA paragraph) and `ch5` (Empirical Findings,
Limitations), and each says so.

### 10.6 The verifier ablation on 2WikiMultihopQA does not replicate

`agentic_no_verifier_twowiki_20260906T103608Z` completed at 250. The central
claim of the report does not hold on the second dataset:

| | HotpotQA | 2WikiMultihopQA |
|---|---|---|
| `agentic_full` EM / F1 | 0.432 / 0.510 | 0.224 / 0.267 |
| `agentic_no_verifier` EM / F1 | 0.396 / 0.474 | 0.216 / 0.266 |
| paired dF1 | **-0.037 [-0.065, -0.011], p = 0.008** | **-0.001 [-0.025, +0.025], p = 1.000** |
| model calls, full vs ablated | 4.26 vs 2.06 | 3.80 vs 2.03 |
| median latency, full vs ablated | 28.3s vs 16.5s | 35.4s vs 18.4s |

It is not a failure to fire. On 2WikiMultihopQA the loop ran on 98 of 250
questions and a later cycle was selected on 64, so the mechanism executed on
more than a third of the dataset and moved the aggregate by one thousandth of
an F1 point at roughly double the cost.

Note also that the ablated system is BETTER attributed on 2WikiMultihopQA:
SP-precision 0.612 against 0.558, SP-F1 0.465 against 0.421. The extra cycles
add evidence that dilutes the cited set rather than sharpening it.

**Rewritten in consequence:** `main.tex` abstract (the backward-edge sentence
now carries both datasets and the word "conditional"), `ch4` Ablation Study
(new paragraph stating the disagreement and the two readings the grid
permits), `ch5` Empirical Findings (same, plus the "one dataset does not
settle a direction" sentence inverted -- the second dataset was run and the
two disagree), and `ch5` Limitations (the completeness paragraph now says the
verifier is the only component measured twice, and that this is why the
planner's and graph's single nulls are not treated as settled either).

The honest headline is the conditional one: **the backward edge helps on
HotpotQA and not on 2WikiMultihopQA.** Anything stronger reads one row and
ignores the other.

### 10.7 Removing the Planner is the largest effect in the grid, and it is positive

`agentic_no_planner_twowiki_20260906T121324Z` completed at 250.

| | HotpotQA | 2WikiMultihopQA |
|---|---|---|
| `agentic_full` EM / F1 | 0.432 / 0.510 | 0.224 / 0.267 |
| `agentic_no_planner` EM / F1 | 0.456 / 0.543 | **0.324 / 0.388** |
| paired dF1 vs full | +0.033 [-0.017, +0.086], p = 0.186 | **+0.121 [+0.067, +0.173], p < 0.001** |
| model calls, full vs ablated | 4.26 vs 1.35 | 3.80 vs 1.26 |
| median latency, full vs ablated | 28.3s vs 9.7s | 35.4s vs 4.5s |

Two things make this the study's most consequential row.

**It is the only configuration that significantly beats a baseline on answer
quality.** Against `self_ask` on 2WikiMultihopQA: dF1 +0.078 [+0.021, +0.135],
at 1.26 model calls against 1.24, a median of 4.5s against 13.2s, and citation
grounding 0.900 against 0.000. The report's headline -- that the system trades
accuracy for accountability -- is true of `agentic_full` and false of this
ablation of it, which gives up nothing.

**It is not `naive_rag`.** With the Planner removed the plan is one identity
node (depth 1.00, one sub-query, one cycle), so it issues a single query
exactly as `naive_rag` does -- and `naive_rag` scores 0.152 / 0.200 on the same
questions. The 0.172 exact-match gap between two one-query systems is the rest
of the stack: KG evidence, sentence-granular pooling, citation-constrained
synthesis, verification. The agentic machinery earns its cost; the
decomposition was spending it.

The effect grows with difficulty (+0.033 -> +0.121), which rules out the
dilution explanation offered for the HotpotQA row: 2WikiMultihopQA's rankings
are the worse of the two (Recall@10 0.727 vs 0.876), so a cost arising from
diluting a good ranking should have shrunk there, not tripled. What grows with
chain length is the number of sub-answers an 8B model must compose across.

**Rewritten in consequence:** `main.tex` abstract (new sentences carrying both
deltas, the baseline win, and the naive_rag comparison), `ch4` Ablation Study
(the planner paragraph now ends with the 2Wiki result rather than deferring to
it), `ch5` Empirical Findings and Limitations.

Also recorded: every component of the confidence blend is at chance on both
datasets -- nli_support 0.529 / 0.508, citation_grounding 0.494 / 0.504,
retrieval_agreement 0.521 / 0.533, blend 0.538 / 0.513. The blend cannot
discriminate because none of its inputs does. See §10.6 and
`results/tables/confidence_diagnostic.tex`.

**Run inventory at this point.** 17 of 18 cells complete.
`agentic_no_kg_twowiki_20260906T124506Z` is the last, in flight and marked
preliminary in every table; its deltas and significance marks are withheld.

### 10.8 Grid complete: 18 of 18. The knowledge graph is a null twice.

`agentic_no_kg_twowiki_20260906T124506Z` completed at 250. Every cell of the
9x2 grid is now over the same frozen 250 questions.

| removed | HotpotQA dF1 | 2WikiMultihopQA dF1 |
|---|---|---|
| Planner | +0.033 [-0.017, +0.086], p = 0.186 | **+0.121 [+0.067, +0.173], p < 0.001** |
| KG Navigator | -0.009 [-0.048, +0.029], p = 0.646 | +0.018 [-0.015, +0.053], p = 0.252 |
| Verifier | **-0.037 [-0.065, -0.011], p = 0.008** | -0.001 [-0.025, +0.025], p = 1.000 |

The graph is the one component whose reading did not change between datasets:
a null both times, and on 2WikiMultihopQA -- the dataset whose longer chains
were supposed to be where a bridge-finder earns its keep -- the sign is
positive. Chain length was the variable that should have unlocked it.
Lengthening the chains changed nothing.

Its cost is consistent and large. Tool calls 16.00 -> 8.65 and median latency
35.4s -> 15.8s on 2WikiMultihopQA; 14.83 -> 8.15 and 28.3s -> 11.8s on
HotpotQA. Roughly half the system's tool calls and more than half its median
latency, for no effect either evaluation can detect. Unlike the planner
ablation this is cost-without-effect rather than harm: the graph is not
damaging the answer, it is being paid for and ignored.

**The completed picture.** Two of three components were read differently by
their second dataset -- the verifier's significant gain vanished, the
planner's harmless null became significant damage. Only the graph read the
same way twice. That is the strongest argument in this report for why a
single-dataset ablation should not be trusted, and it is made by the report's
own grid rather than asserted.

**Rewritten in consequence:** `ch4` graph-ablation paragraph (which previously
deferred to this row), `ch5` Empirical Findings and Limitations. The
completeness paragraph now reads "eighteen of eighteen" rather than listing
what is in flight.
