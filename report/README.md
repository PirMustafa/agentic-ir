# Project Report

Chapter structure follows Section 4 of the assignment brief exactly. The
compiled report is 28 pages including the bibliography, at 11pt on A4.

| File | Chapter | Inputs |
|---|---|---|
| `chapters/ch1_introduction.tex` | 1, Introduction to Agentic AI in IR | the Table 1 task-mapping table |
| `chapters/ch2_data.tex` | 2, Data and Preprocessing | `dataset_stats.tex` |
| `chapters/ch3_methodology.tex` | 3, Retrieval Methodologies | the architecture figure |
| `chapters/ch4_implementation.tex` | 4, System Implementation and Evaluation | `main_results`, `agent_metrics`, `ablations`, `error_analysis`, and the improvement loop's `v2_results`, `v2_cost`, `v2_failures` (Section 4.8) |
| `chapters/ch5_conclusions.tex` | 5, Conclusions and Future Work | `replication.tex` |

The evaluation grid is complete: nine configurations on two datasets, all
eighteen cells over the same frozen 250 questions. Every table is generated
from the run artefacts by `src/agentic_ir/eval/tables.py` and inserted
verbatim; no number in the report is typed by hand, and
`tests/test_report_integrity.py` checks that prose numbers trace to artefacts
on disk. `docs/report-audit.md` records which corrections that audit produced.

Two of the report's findings are deliberately conditional and any edit that
restates them as general is wrong. Removing the verifier costs 0.037 F1 on
HotpotQA (*p* = 0.008) and 0.001 on 2WikiMultihopQA (*p* = 1.000), so the
chapter claims the backward edge helps on one dataset and not the other. And
the improvement loop's kept configuration ties `self_ask` on HotpotQA
(+0.020, interval spanning zero) while beating it by 0.192 on 2WikiMultihopQA,
so the architecture is said to earn its place on the dataset built to need
multiple hops and not on the one that does not.

## Building

Locally with `latexmk`:
```bash
cd report && latexmk -pdf main.tex
```

Or upload the `report/` folder to [Overleaf](https://overleaf.com); no local
TeX installation is needed. The chapters `\input` the generated tables by
relative path (`../results/tables/`), so regenerate those first if any run has
changed:

```bash
python -m agentic_ir.cli tables                 # the nine-system grid
python -c "from agentic_ir.eval.tables import write_v2_tables as w; w()"   # the loop's three
```

## Style

Chapter headings are centred (`titlesec`). The contents lists chapters and
sections only. The prose uses no dashes as punctuation; ranges are written
out ("22 to 25 seconds") and asides are sentences. Generated table captions
are kept short; the caveats they once restated live once, in the chapter that
owns the table.
