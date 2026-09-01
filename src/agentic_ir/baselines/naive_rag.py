"""``naive_rag`` -- ``hybrid_rerank`` plus exactly one LLM call.

Retrieve once with the *same* pipeline ``hybrid_rerank`` uses, show the model
the top-``retrieval.top_k`` passages, ask for a short answer. No decomposition,
no second retrieval, no verification, no re-planning. One question in, one
generation out.

Why this rung exists
--------------------
It isolates **agency** from **merely having an LLM**. Without it, every gap
between ``hybrid_rerank`` and ``agentic_full`` is open to the reading "you added
a language model, of course it answers better" -- which would be true and would
say nothing about planning, tool routing, the knowledge graph or the
verification loop. ``naive_rag`` is the system that has the model and none of
the machinery. Whatever separates it from ``agentic_full`` is the machinery.

That argument only holds if the retrieval underneath is *identical*, so this
class **subclasses** :class:`~agentic_ir.baselines.hybrid_rerank.HybridRerankBaseline`
rather than reimplementing its pipeline. Retrieval is literally the same method
running on the same stack, which makes two things true that would otherwise
need checking: their retrieval columns must agree exactly (a difference is a
bug, never a finding), and ``naive_rag - hybrid_rerank`` is a clean measurement
of one generation step. Fusing, reranking or truncating differently here --
even "better" -- would confound the only comparison this row exists to make.

The deterministic fallback
--------------------------
Design axiom 3, and the same shape the Synthesizer uses (3.4): when the model
is unreachable, unparseable, or the call budget is gone, an extractive ladder
still produces an EM/F1-comparable span from the passages that were retrieved.
A generative baseline that returned ``""`` on a transport error would report a
*retrieval* system's failure as an *answering* failure, and the degraded
questions would silently drag down a column that has nothing to do with them.
``degraded`` and ``answer_origin`` record which path ran, so Chapter 4 can say
how often it mattered instead of guessing.

The ladder deliberately stops short of the Synthesizer's rung 1, the
comparison-arithmetic shortcut (5.4): that rung reads ``Plan.strategy`` and
groups evidence by sub-query, and a baseline with no planner has neither. It is
a feature *of the agentic system*, and lending it to the opponent would be as
dishonest in that direction as weakening the retrieval was in the other.

``self_ask`` imports the rendering, citation-resolution and fallback helpers
from here, so the two generative baselines differ in exactly one thing --
whether the question is decomposed -- and not in how they present passages or
recover from a failed call.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from typing import Any

from ..agents.synthesizer import declarative_sentence, infer_answer_type
from ..llm import LLMError, LLMFormatError, get_client
from ..types import ScoredPassage
from .hybrid_rerank import HybridRerankBaseline

__all__ = [
    "ANSWER_SCHEMA",
    "MAX_PASSAGE_CHARS",
    "NaiveRAGBaseline",
    "SYSTEM_PROMPT",
    "extractive_answer",
    "passage_label",
    "render_passages",
    "resolve_citations",
]

#: Same system turn the agents use (``agents/base.py``). ``/no_think`` is belt
#: and braces beside ``think=False``: an older ollama client silently drops the
#: keyword, and qwen3 then spends its whole completion budget reasoning.
SYSTEM_PROMPT = (
    "You are one component of a local information-retrieval system. "
    "Reply with a single JSON value matching the requested schema. "
    "No prose, no explanation, no markdown fences. /no_think"
)

#: The three fields the task specifies. ``sufficient`` is deliberately absent:
#: it exists on the agentic path so the Verifier can act on self-reported
#: insufficiency, and a baseline with no verifier would only collect it.
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "answer_sentence": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "answer_sentence", "citations"],
}

#: Longest passage shown to the model. HotpotQA paragraphs average ~460 chars,
#: so this bites on the long tail only, and it keeps ten passages plus the
#: instructions comfortably inside ``llm.options.num_ctx`` (8192).
MAX_PASSAGE_CHARS = 700

#: Rung 4 of the fallback ladder: how much of the top passage to return when
#: nothing more specific matched. Matches the Synthesizer's cap.
MAX_FALLBACK_TOKENS = 20

_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_MONTH = (
    r"(?:January|February|March|April|May|June|July|August|September|October"
    r"|November|December)"
)
_DATE_RE = re.compile(
    rf"\b(?:\d{{1,2}}\s+{_MONTH}\s+\d{{4}}|{_MONTH}\s+\d{{1,2}},?\s+\d{{4}}"
    rf"|{_MONTH}\s+\d{{4}}|\d{{4}}-\d{{2}}-\d{{2}})\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")

_PROMPT = """\
Answer the question using ONLY the numbered passages below.

Rules:
1. The answer must be a SHORT SPAN copied from the passages -- a name, a date, a
   number, or "yes"/"no". Never a sentence, never an explanation, never a
   preamble like "The answer is".
2. answer_sentence is ONE declarative sentence stating the answer as a fact.
3. citations lists the passage ids you actually used, e.g. ["p1","p4"]. Cite
   only ids that appear below.
4. If the passages do not contain the answer, still give your best short span
   from them. Do not invent a passage id.

Return ONLY a JSON object of this exact shape:
{{"answer": "Arthur's Magazine", "answer_sentence": "Arthur's Magazine was \
started before First for Women.", "citations": ["p1", "p4"]}}

Question: {question}

Passages:
{passages}"""


# ---------------------------------------------------------------------------
# Passage rendering and citation resolution
# ---------------------------------------------------------------------------

def passage_label(rank: int) -> str:
    """The id the model is asked to cite for the passage at 0-based ``rank``.

    ``p1``-style rather than the agentic ``e3``-style evidence id, because a
    baseline cites *passages* and the agentic system cites *sentences*. Using
    the same prefix for two different units would make the citation columns
    look comparable when they are not.
    """
    return f"p{rank + 1}"


def render_passages(
    passages: Sequence[ScoredPassage], *, max_chars: int = MAX_PASSAGE_CHARS
) -> str:
    """The numbered passage block shown to the model, in ranking order."""
    lines: list[str] = []
    for rank, scored in enumerate(passages):
        text = " ".join(scored.passage.text.split())
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + " ..."
        lines.append(f"[{passage_label(rank)}] ({scored.passage.title}) {text}")
    return "\n".join(lines)


def resolve_citations(
    raw: Any, passages: Sequence[ScoredPassage]
) -> tuple[tuple[str, ...], tuple[str, ...], int]:
    """``(citations, cited_doc_ids, n_unresolvable)`` from the model's list.

    Citations are kept **verbatim**, including ids that name no shown passage.
    Silently dropping them would erase the one failure mode the
    ``hallucinated_citations`` column exists to count -- and on the agentic path
    it is the Verifier, not the generator, that decides what to do about them
    (3.5 step 1). Only the *doc-id* mapping drops the unresolvable ones, since
    there is nothing to map them to.
    """
    if not isinstance(raw, (list, tuple)):
        return (), (), 0
    by_label = {passage_label(i): sp.passage.doc_id for i, sp in enumerate(passages)}
    citations: list[str] = []
    doc_ids: list[str] = []
    unresolvable = 0
    for item in raw:
        label = str(item).strip()
        if not label or label in citations:
            continue
        citations.append(label)
        doc_id = by_label.get(label)
        if doc_id is None:
            unresolvable += 1
        elif doc_id not in doc_ids:
            doc_ids.append(doc_id)
    return tuple(citations), tuple(doc_ids), unresolvable


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------

def extractive_answer(
    question: str, passages: Sequence[ScoredPassage]
) -> dict[str, Any]:
    """Rungs 2-4 of the Synthesizer's ladder (3.4), with no model and no plan.

    Returns the same ``{answer, answer_sentence, citations, cited_doc_ids}``
    shape the LLM path produces, plus ``rung`` naming which step fired, so the
    error analysis can separate "the model was wrong" from "the model never
    ran". Rung 5 -- an empty pool -- returns an empty answer rather than
    ``None``: the baseline is generative, so it *attempted* an answer and
    scoring it as EM=0 is arithmetic on something it really claimed.
    """
    if not passages:
        return {
            "answer": "",
            "answer_sentence": "",
            "citations": (),
            "cited_doc_ids": (),
            "rung": "empty",
        }

    answer_type = infer_answer_type(question)

    def emit(text: str, rank: int, rung: str) -> dict[str, Any]:
        """One rung's result. ``rank`` is the citation, not a lookup key --
        two ranks can hold equal passages, and ``list.index`` would then cite
        the wrong one."""
        answer = text.strip()
        return {
            "answer": answer,
            "answer_sentence": declarative_sentence(question, answer),
            "citations": (passage_label(rank),),
            "cited_doc_ids": (passages[rank].passage.doc_id,),
            "rung": rung,
        }

    if answer_type in ("date", "number"):  # rung 2: a typed surface pattern
        pattern = _DATE_RE if answer_type == "date" else _NUMBER_RE
        for rank, scored in enumerate(passages):
            match = pattern.search(scored.passage.text) or (
                _YEAR_RE.search(scored.passage.text) if answer_type == "date" else None
            )
            if match:
                return emit(match.group(0), rank, f"pattern:{answer_type}")

    if answer_type == "entity":  # rung 3: the top passage's title
        for rank, scored in enumerate(passages):
            if scored.passage.title:
                return emit(scored.passage.title, rank, "title")

    # rung 4: the leading span of the best passage
    span = " ".join(passages[0].passage.text.split()[:MAX_FALLBACK_TOKENS])
    return emit(span, 0, "leading_span")


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

class NaiveRAGBaseline(HybridRerankBaseline):
    """Retrieve once with ``hybrid_rerank``'s pipeline, then generate once."""

    name = "naive_rag"
    generative = True

    #: One retrieval, one generation. Named rather than inlined because
    #: ``self_ask`` sets the same knob to a larger number, and the two figures
    #: sitting side by side in ``metrics.csv`` are the cost half of the
    #: accuracy/cost trade-off Chapter 4 argues about.
    max_llm_calls: int = 1

    def __init__(
        self,
        *args: Any,
        client: Any = None,
        max_llm_calls: int | None = None,
        **kwargs: Any,
    ) -> None:
        """``client`` is injected so the tests can run against a stub, offline.

        It is resolved lazily rather than in ``__init__``: constructing a
        baseline must not require a running Ollama, or listing the available
        configurations would fail on a machine with the server stopped.
        """
        super().__init__(*args, **kwargs)
        self._client = client
        if max_llm_calls is not None:
            self.max_llm_calls = int(max_llm_calls)
        self._calls = 0
        self._llm_latency_s = 0.0

    # -- per-question state -------------------------------------------------
    def _begin(self) -> None:
        """Reset the call counter. One question's budget never funds another's."""
        self._calls = 0
        self._llm_latency_s = 0.0

    # -- LLM plumbing -------------------------------------------------------
    def client(self) -> Any:
        """The shared :class:`~agentic_ir.llm.OllamaClient`, or the injected stub."""
        if self._client is None:
            self._client = get_client()
        return self._client

    @property
    def calls_left(self) -> int:
        return max(0, self.max_llm_calls - self._calls)

    def plan_depth(self) -> int:
        """Retrieval rounds this question took. Always one: there is no plan.

        A hook rather than a constant because ``self_ask`` reports the same
        column with a real number, and ``metrics.csv`` needs one definition of
        ``plan_depth`` across every row it holds.
        """
        return 1

    def ask(
        self, prompt: str, schema: dict[str, Any], *, agent: str | None = None
    ) -> tuple[dict[str, Any] | None, str]:
        """One budgeted, schema-constrained call. Returns ``(parsed, reason)``.

        Never raises: a baseline sweep that dies on question 194 of 250 costs
        hours, and every failure mode here (server down, model missing, timeout,
        unparseable JSON after ``llm.max_format_retries``) is one the fallback
        can answer around. ``reason`` is ``""`` on success and otherwise names
        the path taken, which is what ``errors`` in the trace records.

        ``agent`` defaults to the baseline's own name so the
        :class:`~agentic_ir.llm.CallLedger` keeps baseline calls out of the
        per-agent table; ``llm.models`` has no entry for it, so it falls through
        to ``llm.default_model`` -- the same model the agents use, which is the
        point.
        """
        if self.calls_left <= 0:
            return None, "budget_exhausted"
        self._calls += 1
        started = time.perf_counter()
        try:
            response = self.client().chat(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                agent=agent or self.name,
                schema=schema,
                think=bool(self.cfg.get("llm.think", False)),
            )
        except LLMFormatError:
            # Classified apart from the transport errors, exactly as
            # ``agents/base.py`` classifies it: the model answered, it just
            # never produced valid JSON after ``llm.max_format_retries``. Folding
            # it into ``llm_error`` would zero the ``parse_failures`` column,
            # which is the one number that says whether the schema is holding.
            self._llm_latency_s += time.perf_counter() - started
            return None, "parse_failure"
        except LLMError as exc:
            self._llm_latency_s += time.perf_counter() - started
            return None, f"llm_error:{type(exc).__name__}"
        except Exception as exc:  # noqa: BLE001 - axiom 2: never raise mid-sweep
            self._llm_latency_s += time.perf_counter() - started
            return None, f"llm_error:{type(exc).__name__}: {exc}"
        self._llm_latency_s += getattr(response, "latency_s", 0.0) or (
            time.perf_counter() - started
        )
        parsed = response.parsed if isinstance(response.parsed, dict) else None
        return (parsed, "") if parsed is not None else (None, "parse_failure")

    # -- answering ----------------------------------------------------------
    def answer(
        self, question: str, hops: Sequence[Any], passages: Sequence[ScoredPassage]
    ) -> dict[str, Any]:
        """The single generation step over the ranked pool."""
        parsed: dict[str, Any] | None = None
        reason = "no_passages"
        if passages:
            parsed, reason = self.ask(
                _PROMPT.format(question=question, passages=render_passages(passages)),
                ANSWER_SCHEMA,
            )
        return self._payload(question, passages, parsed, reason)

    def _payload(
        self,
        question: str,
        passages: Sequence[ScoredPassage],
        parsed: dict[str, Any] | None,
        reason: str,
    ) -> dict[str, Any]:
        """Assemble the ``run()`` payload from a parsed reply, or fall back.

        An empty ``answer`` counts as a failed generation and drops to the
        ladder. The model answering ``""`` and the model never running are the
        same thing for the answer columns, and both are honestly ``degraded``.
        """
        errors: list[str] = []
        origin = "llm"
        answer = ""
        sentence = ""
        citations: tuple[str, ...] = ()
        cited: tuple[str, ...] = ()
        hallucinated = 0

        if parsed is not None:
            answer = str(parsed.get("answer") or "").strip()
            sentence = str(parsed.get("answer_sentence") or "").strip()
            citations, cited, hallucinated = resolve_citations(
                parsed.get("citations"), passages
            )
            if not answer:
                errors.append(f"{self.name}: model returned an empty answer")
        elif reason:
            errors.append(f"{self.name}: {reason}")

        degraded = not answer
        if degraded:
            fallback = extractive_answer(question, passages)
            origin = f"fallback:{fallback['rung']}"
            answer = str(fallback["answer"])
            sentence = str(fallback["answer_sentence"])
            citations = tuple(fallback["citations"])
            cited = tuple(fallback["cited_doc_ids"])
            hallucinated = 0
        elif not sentence:
            sentence = declarative_sentence(question, answer)

        return {
            "answer": answer,
            "answer_sentence": sentence,
            "citations": citations,
            "cited_doc_ids": cited,
            "llm_calls": self._calls,
            "llm_latency_s": self._llm_latency_s,
            "degraded": degraded,
            "errors": tuple(errors),
            "extra": {
                "answer_origin": origin,
                "parse_failures": int(reason == "parse_failure"),
                "budget_exhausted": reason == "budget_exhausted",
                # The weak, non-NLI quantity a baseline can actually measure:
                # the share of emitted citations that name a shown passage.
                # ``citation_grounding`` stays None -- that column is claim-level
                # and NLI-backed, and inventing a number for it here would make
                # two different things look like one.
                "citation_resolution": (
                    None if not citations else 1.0 - hallucinated / len(citations)
                ),
                "hallucinated_citations": hallucinated,
                "plan_depth": self.plan_depth(),
            },
        }
