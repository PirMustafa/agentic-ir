# Project Report

Chapter structure follows Section 4 of the assignment brief exactly.

| File | Chapter | Status |
|---|---|---|
| `chapters/ch1_introduction.tex` | 1 — Introduction to Agentic AI in IR | skeleton |
| `chapters/ch2_data.tex` | 2 — Data and Preprocessing | skeleton |
| `chapters/ch3_methodology.tex` | 3 — Retrieval Methodologies | skeleton |
| `chapters/ch4_implementation.tex` | 4 — System Implementation and Evaluation | skeleton |
| `chapters/ch5_conclusions.tex` | 5 — Conclusions and Future Work | skeleton |

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
