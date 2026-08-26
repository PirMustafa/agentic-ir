"""Build the deduplicated passage corpus and the qrels.

    python scripts/build_corpus.py --dataset all

Reads the frozen splits under ``data/raw/`` and writes, per dataset:

* ``data/processed/{dataset}_corpus.jsonl``      one ``Passage`` per line
* ``data/processed/{dataset}_qrels.jsonl``       one ``GoldAnswer`` per question
* ``data/processed/{dataset}_corpus_stats.json`` the numbers quoted in Chapter 3

The one thing that must not go wrong
------------------------------------
Sentence splitting happens **here, once, and is persisted**. Supporting facts
are ``(title, sent_id)`` pairs, so if sentence ids drift by even one between
the corpus and the qrels, ``sp_em``/``sp_f1`` silently measure nothing while
still producing plausible-looking numbers.

Both datasets already ship ``context`` as a list of sentences per title, so the
split is *theirs* -- nothing here calls nltk, pysbd or a regex. Each sentence is
whitespace-normalised (HotpotQA prefixes continuation sentences with a space,
2Wiki does not) but the **list length is never changed**, not even for empty
sentences, because the length is the id space.

After writing, the corpus is re-read from disk through
``indexing.corpus.Corpus`` -- the same reader every other module uses -- and
every gold ``(title, sent_id)`` is resolved against it. Failures are counted,
sampled into the report, and by default make this script exit non-zero.

Deduplication, and why it is not "keep the first"
-------------------------------------------------
The distractor setting repeats paragraphs across questions: HotpotQA is 10
paragraphs x 7,405 questions before dedupe. Passages are keyed by
``doc_id = f"{source}:{quote(title, safe='')}"``.

HotpotQA is clean -- measured, every one of its 66,581 titles has exactly one
body text. 2Wiki is not: 1,720 of its 54,957 titles ship two or more
detokenisations of the same article across different questions ("Feo( c. 1471"
vs "Feo (c. 1471"), and 453 of those split into *different numbers of
sentences*. Since ``sent_id`` is an index into that list, the variant we keep
decides whether a gold fact lands on the right sentence.

Measured over 2Wiki's 30,687 gold facts, keeping the first occurrence leaves 70
gold ids out of range and 211 pointing at a different sentence. Keeping the
variant with the **most sentences** -- the widest id space, decided without
looking at any gold data -- leaves 0 out of range and 150 mispointed. That is
the rule, and both residuals are counted and reported below; ties go to the
first occurrence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tqdm import tqdm

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:  # sibling scripts, however this file is invoked
    sys.path.insert(0, str(_HERE))

from download_data import force_utf8_stdout, load_raw_split, rule  # noqa: E402

from agentic_ir.config import Config, Paths, load_config  # noqa: E402
from agentic_ir.indexing.corpus import (  # noqa: E402
    DATASETS,
    Corpus,
    corpus_path,
    gold_to_json,
    load_qrels,
    make_doc_id,
    passage_to_json,
    qrels_path,
    write_jsonl,
)
from agentic_ir.types import GoldAnswer, Passage  # noqa: E402

#: How many unresolved gold facts to print before truncating the list.
MAX_EXAMPLES = 10

#: Above this fraction of unresolvable gold facts the build fails. The failure
#: mode this guards against -- sentence ids drifting between the corpus and the
#: qrels -- breaks essentially *every* fact, so anything past a rounding error
#: is it. Below the threshold what remains is upstream annotation noise:
#: HotpotQA's validation split has exactly one such fact (sent_id 902 for a
#: 5-sentence paragraph), which no amount of correct code here can fix.
MAX_UNRESOLVED_RATE = 0.001

#: Same idea for the *silent* half of that failure: a gold id that still
#: resolves, but to different text than the question's own context showed. A
#: re-split corpus would score ~100% here while every count above stayed clean.
#: Measured: HotpotQA 0.00%, 2Wiki 0.49% (its duplicate-title variants).
MAX_DRIFT_RATE = 0.02

#: Comparison for the drift check: ignore case, whitespace and punctuation, so
#: that 2Wiki's detokenisation artefacts ("Feo( c." vs "Feo (c.") do not count
#: as drift while genuinely different sentences do.
_SQUASH = re.compile(r"[^0-9a-z]+")


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def clean_sentence(text: str) -> str:
    """Normalise whitespace inside one sentence, preserving nothing else.

    HotpotQA ships continuation sentences with a leading space and 2Wiki does
    not; some 2Wiki sentences carry embedded newlines. Collapsing runs of
    whitespace makes the two sources look alike to BM25, to bge and to the NLI
    premise encoder, and it is the only transformation applied to corpus text.
    """
    return " ".join(text.split())


def squash(text: str) -> str:
    """Text reduced to its letters and digits, for the drift comparison."""
    return _SQUASH.sub("", text.lower())


def paragraph_text(sentences: Sequence[str]) -> str:
    """The paragraph, rebuilt from its own sentences.

    Empty sentences keep their slot in ``sentences`` (the slot *is* the
    ``sent_id``) but contribute nothing here, so ``text`` never contains a
    double space.
    """
    return " ".join(s for s in sentences if s)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(dataset: str, cfg: Config) -> tuple[dict[str, Any], list[tuple[str, str, int, str]]]:
    """Read the raw split and write corpus + qrels.

    Returns ``(stats, drift_probe)``, where the probe carries, for every gold
    fact, the sentence text the question's **own** context showed at that id --
    the only reference against which a silently mis-resolved id can be caught.
    """
    ds = load_raw_split(dataset, cfg)
    columns = set(ds.column_names)

    passages: dict[str, Passage] = {}
    golds: list[GoldAnswer] = []
    drift_probe: list[tuple[str, str, int, str]] = []

    n_paragraphs = 0
    n_repeat_identical = 0
    n_variant_occurrences = 0
    n_replacements = 0
    n_malformed_triples = 0
    n_duplicate_facts = 0
    variant_titles: set[str] = set()
    collision_examples: list[dict[str, str]] = []
    strata: dict[str, dict[str, int]] = {"level": {}, "type": {}}

    for row in tqdm(ds, total=len(ds), desc=f"{dataset}: reading", unit="q"):
        context = row.get("context") or {}
        titles = list(context.get("title") or ())
        sentence_lists = list(context.get("sentences") or ())
        qid = row["id"]
        own_context: dict[str, tuple[str, ...]] = {}

        for title, raw_sentences in zip(titles, sentence_lists, strict=False):
            n_paragraphs += 1
            # The dataset's own split, verbatim in length. Never re-split.
            sentences = tuple(clean_sentence(s) for s in raw_sentences)
            own_context.setdefault(title, sentences)
            doc_id = make_doc_id(dataset, title)
            text = paragraph_text(sentences)

            existing = passages.get(doc_id)
            if existing is not None:
                # Compared on the sentence tuple, not the joined text: ("A.",
                # "B.") and ("A. B.",) are the same paragraph but not the same
                # id space, and only the tuple says so.
                if existing.sentences == sentences:
                    n_repeat_identical += 1
                    continue
                # Same title, two detokenisations. Keep the one with the most
                # sentences: sent_id indexes this list, so the finest split is
                # the only variant whose id space can hold every gold id.
                # Decided on shape alone -- no gold data is consulted.
                n_variant_occurrences += 1
                variant_titles.add(doc_id)
                if len(sentences) <= len(existing.sentences):
                    if len(collision_examples) < MAX_EXAMPLES:
                        collision_examples.append({
                            "title": title,
                            "kept": f"[{len(existing.sentences)} sents] {existing.text[:140]}",
                            "dropped": f"[{len(sentences)} sents] {text[:140]}",
                            "seen_at_qid": qid,
                        })
                    continue
                n_replacements += 1

            passages[doc_id] = Passage(
                doc_id=doc_id,
                title=title,
                text=text,
                sentences=sentences,
                source=dataset,  # type: ignore[arg-type]
                meta={"first_seen_qid": qid},
            )

        facts_block = row.get("supporting_facts") or {}
        facts = tuple(
            (title, int(sent_id))
            for title, sent_id in zip(
                facts_block.get("title") or (),
                facts_block.get("sent_id") or (),
                strict=False,
            )
        )
        n_duplicate_facts += len(facts) - len(set(facts))
        for title, sent_id in facts:
            own = own_context.get(title)
            if own is not None and 0 <= sent_id < len(own):
                drift_probe.append((qid, title, sent_id, own[sent_id]))

        triples: list[tuple[str, str, str]] = []
        for evidence in row.get("evidences") or ():
            parts = list(evidence)
            if len(parts) == 3:
                triples.append((parts[0], parts[1], parts[2]))
            else:
                n_malformed_triples += 1

        level = row.get("level") if "level" in columns else None
        qtype = row.get("type") if "type" in columns else None
        strata["level"][str(level)] = strata["level"].get(str(level), 0) + 1
        strata["type"][str(qtype)] = strata["type"].get(str(qtype), 0) + 1

        golds.append(GoldAnswer(
            qid=qid,
            question=row["question"],
            answer=row["answer"],
            dataset=dataset,  # type: ignore[arg-type]
            supporting_facts=facts,
            evidence_triples=tuple(triples),
            level=level,
            qtype=qtype,
            context_titles=tuple(titles),
        ))

    # Deterministic file order: doc_id for the corpus, qid for the qrels.
    ordered_docs = [passages[doc_id] for doc_id in sorted(passages)]
    golds.sort(key=lambda g: g.qid)

    c_path = corpus_path(dataset, cfg)
    q_path = qrels_path(dataset, cfg)
    n_written = write_jsonl(c_path, (passage_to_json(p) for p in ordered_docs))
    n_qrels = write_jsonl(q_path, (gold_to_json(g) for g in golds))
    print(f"  wrote {n_written:,} passages -> {c_path}")
    print(f"  wrote {n_qrels:,} qrels     -> {q_path}")

    sentence_counts = [len(p.sentences) for p in ordered_docs]
    stats: dict[str, Any] = {
        "dataset": dataset,
        "questions": len(golds),
        "paragraphs_read": n_paragraphs,
        "passages": len(ordered_docs),
        "dedupe": {
            "repeat_identical": n_repeat_identical,
            "collisions_same_title_different_text": n_variant_occurrences,
            "titles_with_variants": len(variant_titles),
            "replaced_by_finer_split": n_replacements,
            "collision_examples": collision_examples,
        },
        "sentences": {
            "total": sum(sentence_counts),
            "mean_per_passage": round(statistics.fmean(sentence_counts), 3),
            "median_per_passage": statistics.median(sentence_counts),
            "min_per_passage": min(sentence_counts),
            "max_per_passage": max(sentence_counts),
            # Counted over the passages actually written, not every paragraph
            # read: an empty sentence still occupies a sent_id.
            "empty": sum(1 for p in ordered_docs for s in p.sentences if not s),
            "mean_chars": round(
                statistics.fmean([len(s) for p in ordered_docs for s in p.sentences]), 2
            ),
        },
        "passage_chars": {
            "mean": round(statistics.fmean([len(p.text) for p in ordered_docs]), 2),
            "max": max(len(p.text) for p in ordered_docs),
        },
        "gold": {
            "duplicate_supporting_facts": n_duplicate_facts,
            "malformed_evidence_triples": n_malformed_triples,
            "questions_with_triples": sum(1 for g in golds if g.evidence_triples),
            "triples_total": sum(len(g.evidence_triples) for g in golds),
        },
        "strata": {key: dict(sorted(counts.items())) for key, counts in strata.items()},
        "files": {
            "corpus": {"path": str(c_path), "lines": n_written,
                       "bytes": c_path.stat().st_size, "sha256": sha256_of(c_path)},
            "qrels": {"path": str(q_path), "lines": n_qrels,
                      "bytes": q_path.stat().st_size, "sha256": sha256_of(q_path)},
        },
    }
    return stats, drift_probe


# ---------------------------------------------------------------------------
# Validation -- against the files that were actually written
# ---------------------------------------------------------------------------

def validate(
    dataset: str,
    cfg: Config,
    stats: dict[str, Any],
    drift_probe: Sequence[tuple[str, str, int, str]],
) -> dict[str, Any]:
    """Resolve every gold ``(title, sent_id)`` in the corpus just written.

    Deliberately re-reads both files through the production reader rather than
    checking the in-memory objects: what matters is that the artefacts on disk
    agree, including the JSON round trip.
    """
    corpus = Corpus.load(dataset, cfg=cfg, progress=True)  # type: ignore[arg-type]
    qrels = load_qrels(dataset, cfg=cfg)  # type: ignore[arg-type]

    if len(corpus) != stats["passages"]:
        raise SystemExit(
            f"corpus round trip lost passages: wrote {stats['passages']}, read {len(corpus)}"
        )
    unique_ids = len({p.doc_id for p in corpus})
    if unique_ids != len(corpus):
        raise SystemExit(f"duplicate doc_ids on disk: {len(corpus) - unique_ids}")

    n_facts = 0
    missing_title = 0
    bad_sent_id = 0
    empty_sentence = 0
    off_context = 0
    failed_qids: set[str] = set()
    examples: list[dict[str, Any]] = []

    for gold in tqdm(qrels.values(), total=len(qrels), desc=f"{dataset}: gold facts", unit="q"):
        context_titles = set(gold.context_titles)
        for title, sent_id in gold.supporting_facts:
            n_facts += 1
            if title not in context_titles:
                # Gold cites a paragraph that is not among this question's own
                # ten. Not itself a failure -- the corpus is global -- but it
                # tells us whether a per-question oracle would be complete.
                off_context += 1
            passage = corpus.by_title(title)
            if passage is None:
                missing_title += 1
                failed_qids.add(gold.qid)
                reason = "title not in corpus"
            else:
                sentence = corpus.sentence(passage.doc_id, sent_id)
                if sentence is None:
                    bad_sent_id += 1
                    failed_qids.add(gold.qid)
                    reason = f"sent_id out of range (0..{len(passage.sentences) - 1})"
                elif not sentence.strip():
                    empty_sentence += 1
                    failed_qids.add(gold.qid)
                    reason = "resolves to an empty sentence"
                else:
                    continue
            if len(examples) < MAX_EXAMPLES:
                examples.append({
                    "qid": gold.qid, "title": title, "sent_id": sent_id, "reason": reason,
                })

    # The silent half: ids that resolve, but not to the sentence the question's
    # own context showed there. Re-splitting the corpus would light this up
    # while every counter above stayed clean.
    drifted = 0
    drift_examples: list[dict[str, Any]] = []
    for qid, title, sent_id, expected in tqdm(
        drift_probe, desc=f"{dataset}: sentence drift", unit="fact"
    ):
        passage = corpus.by_title(title)
        if passage is None:
            continue
        actual = corpus.sentence(passage.doc_id, sent_id)
        if actual is None or squash(actual) == squash(expected):
            continue
        drifted += 1
        if len(drift_examples) < MAX_EXAMPLES:
            drift_examples.append({
                "qid": qid, "title": title, "sent_id": sent_id,
                "in_question_context": expected[:120], "in_corpus": actual[:120],
            })

    unresolved = missing_title + bad_sent_id + empty_sentence
    result = {
        "gold_facts": n_facts,
        "drift_checked": len(drift_probe),
        "drifted": drifted,
        "drift_rate": round(drifted / len(drift_probe), 6) if drift_probe else 0.0,
        "drift_examples": drift_examples,
        "unresolved": unresolved,
        "unresolved_rate": round(unresolved / n_facts, 6) if n_facts else 0.0,
        "missing_title": missing_title,
        "bad_sent_id": bad_sent_id,
        "empty_sentence": empty_sentence,
        "questions_affected": len(failed_qids),
        "facts_outside_own_context": off_context,
        "examples": examples,
    }
    stats["validation"] = result
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(stats: dict[str, Any]) -> None:
    dedupe = stats["dedupe"]
    sents = stats["sentences"]
    val = stats["validation"]
    print(f"\n  questions                    : {stats['questions']:,}")
    print(f"  paragraphs read (with repeats): {stats['paragraphs_read']:,}")
    print(f"  passages after dedupe        : {stats['passages']:,}")
    print(f"    identical repeats dropped  : {dedupe['repeat_identical']:,}")
    print(f"    variant collisions         : {dedupe['collisions_same_title_different_text']:,} "
          f"over {dedupe['titles_with_variants']:,} titles")
    print(f"      resolved to finer split  : {dedupe['replaced_by_finer_split']:,} replacements")
    print(f"  sentences total              : {sents['total']:,}")
    print(f"    mean / median per passage  : {sents['mean_per_passage']} / "
          f"{sents['median_per_passage']}")
    print(f"    min / max per passage      : {sents['min_per_passage']} / "
          f"{sents['max_per_passage']}")
    print(f"    empty sentences (id kept)  : {sents['empty']:,}")
    print(f"    mean chars per sentence    : {sents['mean_chars']}")
    print(f"  mean chars per passage       : {stats['passage_chars']['mean']}")
    print(f"  gold supporting facts        : {val['gold_facts']:,}")
    print(f"    UNRESOLVED                 : {val['unresolved']:,} "
          f"({val['unresolved_rate']:.4%}) over {val['questions_affected']:,} questions")
    print(f"      missing title            : {val['missing_title']:,}")
    print(f"      sent_id out of range     : {val['bad_sent_id']:,}")
    print(f"      empty sentence           : {val['empty_sentence']:,}")
    print(f"    DRIFTED (resolves, wrong)  : {val['drifted']:,} of {val['drift_checked']:,} "
          f"checked ({val['drift_rate']:.4%})")
    print(f"    outside own 10-para context: {val['facts_outside_own_context']:,}")
    print(f"  gold evidence triples        : {stats['gold']['triples_total']:,} over "
          f"{stats['gold']['questions_with_triples']:,} questions")
    for key, counts in stats["strata"].items():
        if set(counts) != {"None"}:
            print(f"  strata[{key}]{' ' * (21 - len(key))}: {counts}")
    if dedupe["collision_examples"]:
        print("\n  collision examples (kept | dropped):")
        for ex in dedupe["collision_examples"]:
            print(f"    {ex['title']!r}\n      kept   : {ex['kept']!r}\n"
                  f"      dropped: {ex['dropped']!r}")
    if val["examples"]:
        print("\n  unresolved gold facts:")
        for ex in val["examples"]:
            print(f"    {ex['qid']} ({ex['title']!r}, {ex['sent_id']}): {ex['reason']}")
    if val["drift_examples"]:
        print("\n  drifted gold facts (question's own context | corpus):")
        for ex in val["drift_examples"]:
            print(f"    {ex['qid']} ({ex['title']!r}, {ex['sent_id']})\n"
                  f"      context: {ex['in_question_context']!r}\n"
                  f"      corpus : {ex['in_corpus']!r}")
    for name, info in stats["files"].items():
        print(f"  {name:<7} {info['bytes'] / 1e6:8.2f} MB  sha256={info['sha256'][:16]}...")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the deduplicated passage corpus and qrels from data/raw/.",
    )
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument(
        "--max-unresolved-rate", type=float, default=MAX_UNRESOLVED_RATE,
        help=f"fail if more than this fraction of gold facts does not resolve "
             f"(default {MAX_UNRESOLVED_RATE})",
    )
    parser.add_argument(
        "--max-drift-rate", type=float, default=MAX_DRIFT_RATE,
        help=f"fail if more than this fraction of gold facts resolves to a different sentence "
             f"than the question's own context showed (default {MAX_DRIFT_RATE})",
    )
    parser.add_argument(
        "--allow-unresolved", action="store_true",
        help="exit 0 whatever the unresolved and drift counts are (both are still reported)",
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    force_utf8_stdout()
    args = parse_args(argv)
    cfg = load_config(args.config)
    processed = Paths.from_config(cfg).processed

    targets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    exit_code = 0
    for dataset in targets:
        rule(f"{dataset}: building corpus")
        try:
            stats, drift_probe = build(dataset, cfg)
        except FileNotFoundError as exc:
            print(f"\nFAILED: {exc}", file=sys.stderr)
            exit_code = 1
            continue
        validate(dataset, cfg, stats, drift_probe)
        report(stats)

        stats_path = processed / f"{dataset}_corpus_stats.json"
        with stats_path.open("w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2, ensure_ascii=False)
        print(f"  stats   {stats_path}")

        val = stats["validation"]
        if val["unresolved"]:
            print(f"\n  WARNING: {val['unresolved']} of {val['gold_facts']:,} gold supporting "
                  f"facts ({val['unresolved_rate']:.4%}) do not resolve in the corpus just "
                  "written; those questions cannot score sp_em/sp_f1 above their remaining facts.")
            if val["unresolved_rate"] > args.max_unresolved_rate and not args.allow_unresolved:
                print(f"\nFAILED: unresolved rate {val['unresolved_rate']:.4%} exceeds "
                      f"--max-unresolved-rate {args.max_unresolved_rate:.4%}.\n"
                      "        At this scale it is sentence-id drift, not upstream annotation "
                      "noise: the\n        corpus and the qrels disagree about what sent_id "
                      "means, and every\n        supporting-fact metric computed from them "
                      "would be meaningless.", file=sys.stderr)
                exit_code = 2
        if val["drifted"]:
            print(f"\n  WARNING: {val['drifted']} gold facts resolve to a different sentence "
                  f"than the\n           question's own context showed at that id "
                  f"({val['drift_rate']:.4%}).")
            if val["drift_rate"] > args.max_drift_rate and not args.allow_unresolved:
                print(f"\nFAILED: sentence drift {val['drift_rate']:.4%} exceeds "
                      f"--max-drift-rate {args.max_drift_rate:.4%}.\n"
                      "        The ids still resolve, so nothing downstream would complain -- "
                      "sp_em/sp_f1\n        would simply be scored against the wrong sentences.",
                      file=sys.stderr)
                exit_code = 2

    if exit_code == 0:
        print("\nNext: python scripts/sample_eval_set.py --dataset all")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
