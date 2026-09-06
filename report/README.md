# Project Report

Chapter structure follows Section 4 of the assignment brief exactly.

| File | Chapter | Status |
|---|---|---|
| `chapters/ch1_introduction.tex` | 1 — Introduction to Agentic AI in IR | written; includes the Table 1 task-mapping table |
| `chapters/ch2_data.tex` | 2 — Data and Preprocessing | written; inputs `dataset_stats.tex` |
| `chapters/ch3_methodology.tex` | 3 — Retrieval Methodologies | written |
| `chapters/ch4_implementation.tex` | 4 — System Implementation and Evaluation | written; inputs `main_results`, `agent_metrics`, `ablations`, `error_analysis`, `calibration`; one `% PLACEHOLDER` remains for the ablation rows still running (`agentic_no_kg` on HotpotQA, all three ablations on 2WikiMultihopQA) |
| `chapters/ch5_conclusions.tex` | 5 — Conclusions and Future Work | written; findings and limitations track the current generated tables |

The prose is complete. What is still open is the evaluation grid, not the
writing: `docs/report-audit.md` records which runs are finished and which
sentences depend on the ones that are not.

## Building

Locally with `latexmk`:
```bash
cd report && latexmk -pdf main.tex
```

Or upload the `report/` folder to [Overleaf](https://overleaf.com) — no local
TeX installation needed.

## Writing order

Chapters are **not** written front to back. Chapter 4 depends on results, so
the order that avoids rewriting is: 2 → 3 → 4 → 1 → 5. The introduction is
written last because it should promise exactly what the results delivered.

Numbers must never be typed by hand into the report. `results/tables/` holds
generated `.tex` fragments that the chapters `\input{}`, so a re-run of the
evaluation updates the report automatically and the two can never disagree.
Prose that quotes a table cell has to be re-read whenever the tables are
regenerated (`python -m agentic_ir.cli tables`); `tests/test_report_integrity.py`
mechanises part of that check and `docs/report-audit.md` records the rest.
The few figures that come from a trace rather than a table (a qid, a count of
re-planned questions, the verifier's component scores on one record) are
attributed to their run in the text.
