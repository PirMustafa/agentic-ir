"""Verifier: entailment, grounding, a gated adjudicator, and the feedback edge.

Contract: ``docs/architecture.md`` 3.5. Input is an ``AnswerCandidate`` and the
evidence map; output is a ``VerificationResult``, and -- when the verdict is
``revise`` -- a ``ReplanDirective``, which is the only object in the system that
flows backwards.

**This component must never block.** A Verifier that raises takes the whole
contribution of the project with it, so every stage below has a defined
degraded outcome: the NLI model failing to load falls back to token
containment, an unparseable adjudication drops the LLM term and renormalises,
and an exhausted budget skips the call entirely. ``abstain`` is a *label*, not
a refusal -- the best candidate is still returned, so EM and F1 stay computable
for every question in the run.

Where the LLM is and is not spent
---------------------------------
The adjudicator is issued only inside the uncertainty band
``|conf - 0.55| <= 0.15``. Outside it the arithmetic has already decided, and a
call could not change the verdict -- only the latency. On this hardware that is
2-5 s saved per confident question, which over 250 questions and nine
configurations is hours.

Three signals, deliberately not three views of one signal
---------------------------------------------------------
``nli_support`` asks whether the cited text entails the claim; the encoder is
the authority. ``citation_grounding`` asks the weaker, independent question of
whether the citations *resolve at all* and are not contradicted -- a local 8B
model inventing ``e14`` when only ``e1..e9`` exist is a real and common failure
that entailment scoring cannot see, because a hallucinated id has no premise to
score. ``retrieval_agreement`` asks whether the answer used what retrieval
actually found. Making the middle term depend on the first would collapse the
blend to one number wearing three hats.

The DeBERTa encoder runs on **CPU** (``agents.verifier.nli_device``). The 8 GB
card is fully committed to qwen3:8b at query time; an encoder beside it is the
OOM that kills the run at question 194.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from ..config import Config
from ..state import QuestionState
from ..types import (
    AnswerCandidate,
    Claim,
    Evidence,
    ReplanDirective,
    ReplanReason,
    Verdict,
    VerificationResult,
)
from .base import BaseAgent

__all__ = [
    "CrossEncoderNLI",
    "NLIScorer",
    "PROMPT_ID",
    "Verifier",
    "build_directive",
    "preflight_nli",
]

PROMPT_ID = "verifier.adjudicate.v1"


def preflight_nli(cfg: Any = None) -> tuple[bool, str]:
    """Load the NLI encoder now and report whether it worked.

    Call this BEFORE a long evaluation run. The degraded path is safe in the
    sense that it never raises, but it is not safe for the *results*: token
    containment scored 1.000 on an answer where the real model scores 0.0008
    entailment (a comparison claim supported by only one operand's date).
    Because ``nli_support`` carries 0.45 of the confidence blend, a silent
    fallback pushes nearly everything above the 0.55 threshold, the verdict is
    always ``accept``, and the re-plan loop -- the contribution this project is
    built to measure -- never fires. The run completes, the numbers look
    plausible, and the central result is an artefact.

    So: fail loudly here rather than 250 questions later. Returns
    ``(ok, message)`` instead of raising, matching the module's no-blocking
    contract.
    """
    from ..config import load_config

    cfg = cfg or load_config()
    name = str(cfg.get("agents.verifier.nli_model"))
    device = str(cfg.get("agents.verifier.nli_device", "cpu"))
    try:
        scorer = CrossEncoderNLI(name, device)
        rows = scorer.score([("A cat sat on the mat.", "A cat sat on the mat.")])
    except Exception as exc:  # noqa: BLE001 - preflight reports, never raises
        return False, f"NLI unavailable ({type(exc).__name__}: {exc}) -- verifier would run DEGRADED"
    if not rows:
        return False, "NLI returned no scores -- verifier would run DEGRADED"
    entail = rows[0].get("entailment", 0.0)
    if entail < 0.5:
        return False, (
            f"NLI loaded but scored a self-entailment at {entail:.3f}; "
            "labels are probably misaligned"
        )
    return True, f"NLI ok ({name} on {device}, self-entailment {entail:.3f})"

_ADJUDICATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "supported": {"type": "boolean"},
        "confidence": {"type": "number"},
        "missing_information": {"type": "array", "items": {"type": "string"}},
        "contradictions": {"type": "array", "items": {"type": "string"}},
        "verdict": {"type": "string", "enum": ["accept", "revise"]},
    },
    "required": ["supported", "confidence", "missing_information", "contradictions",
                 "verdict"],
}

#: A premise counts as supporting its hypothesis above this entailment
#: probability. Used for ``Claim.supported`` and for the best-premise label,
#: not for ``citation_grounding`` -- see the module docstring.
DEFAULT_ENTAIL_THRESHOLD = 0.5

#: Uncited evidence above this contradiction probability is recorded (3.5 step
#: 2). Deliberately high: a false contradiction would push a correct answer into
#: a re-plan it cannot benefit from.
DEFAULT_CONTRADICTION_THRESHOLD = 0.7

#: How many uncited pieces of evidence to screen for contradictions.
MAX_CONTRADICTION_PREMISES = 5

#: Longest premise or evidence snippet shown to a model, characters.
MAX_PREMISE_CHARS = 600

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _clip(text: str, limit: int = MAX_PREMISE_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + " ..."


# ---------------------------------------------------------------------------
# Entailment
# ---------------------------------------------------------------------------

class NLIScorer(Protocol):
    """Anything that scores ``(premise, hypothesis)`` pairs into label probabilities.

    A protocol rather than a concrete class so that tests can inject a stub and
    run offline in milliseconds, which is the difference between a test suite
    that gets run and one that does not.
    """

    def score(
        self, pairs: Sequence[tuple[str, str]]
    ) -> list[dict[str, float]]:  # pragma: no cover - interface
        ...


class CrossEncoderNLI:
    """``cross-encoder/nli-deberta-v3-base`` on CPU, loaded on first use.

    The model is loaded lazily and cached process-wide: a 250-question run must
    pay the ~2 s load once, not 250 times. Raw logits are softmaxed here rather
    than trusting an activation default, because sentence-transformers has
    changed that keyword's name and behaviour across major versions and a
    silent identity activation would turn probabilities into unbounded scores.
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any | None = None
        self._labels: tuple[str, ...] = ("contradiction", "entailment", "neutral")

    def _ensure(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
            id2label = getattr(getattr(self._model, "config", None), "id2label", None)
            if isinstance(id2label, Mapping) and id2label:
                self._labels = tuple(
                    str(id2label[i]).lower() for i in sorted(id2label, key=int)
                )
        return self._model

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[dict[str, float]]:
        if not pairs:
            return []
        import numpy as np

        model = self._ensure()
        raw = np.asarray(model.predict(list(pairs)), dtype=float)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        shifted = raw - raw.max(axis=1, keepdims=True)
        probs = np.exp(shifted)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return [
            {label: float(row[i]) for i, label in enumerate(self._labels)}
            for row in probs
        ]


@lru_cache(maxsize=2)
def _shared_nli(model_name: str, device: str) -> CrossEncoderNLI:
    return CrossEncoderNLI(model_name, device)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Verifier(BaseAgent):
    """Score a candidate, decide a verdict, and say what is missing."""

    name = "verifier"

    def __init__(
        self,
        cfg: Config | None = None,
        *,
        client: Any | None = None,
        nli: NLIScorer | None = None,
        prompt_dir: Path | None = None,
    ) -> None:
        super().__init__(cfg, client=client, prompt_dir=prompt_dir)
        self._nli = nli
        self.threshold = float(self.cfg.get("agents.verifier.confidence_threshold", 0.55))
        self.band = float(self.cfg.get("agents.verifier.uncertainty_band", 0.15))
        self.method = str(self.cfg.get("agents.verifier.method", "nli_plus_llm"))
        self.max_evidence = int(self.cfg.get("agents.verifier.max_evidence", 20))
        self.entail_threshold = float(
            self.cfg.get("agents.verifier.entail_threshold", DEFAULT_ENTAIL_THRESHOLD)
        )
        self.contradiction_threshold = float(
            self.cfg.get(
                "agents.verifier.contradiction_threshold", DEFAULT_CONTRADICTION_THRESHOLD
            )
        )
        weights = dict(self.cfg.get("agents.verifier.weights", {}) or {})
        self.weights = {
            "nli": float(weights.get("nli", 0.45)),
            "citation": float(weights.get("citation", 0.25)),
            "retrieval": float(weights.get("retrieval", 0.15)),
            "llm": float(weights.get("llm", 0.15)),
        }

    # -- entry point -------------------------------------------------------

    def run(
        self,
        state: QuestionState,
        candidate: AnswerCandidate | None = None,
        *,
        evidence: Mapping[str, Evidence] | None = None,
    ) -> VerificationResult:
        """Verify one candidate. Never raises.

        The result is returned, not appended: ``state.verifications`` is a
        cycle-level artefact list owned by the orchestrator.
        """
        cand = candidate or (state.candidates[-1] if state.candidates else None)
        if cand is None:
            cand = AnswerCandidate(
                answer="", answer_sentence="", cycle=state.cycle,
                origin="fallback_rule", sufficient=False,
            )
        pool = dict(evidence) if evidence is not None else dict(state.evidence)
        result: VerificationResult | None = None

        with state.step(
            self.name, answer=cand.answer[:120], n_citations=len(cand.citations)
        ) as rec:
            result = self._verify(state, cand, pool, rec)
            rec.output_summary = {
                "verdict": result.verdict,
                "confidence": round(result.confidence, 4),
                "nli_support": round(result.nli_support, 4),
                "citation_grounding": round(result.citation_grounding, 4),
                "retrieval_agreement": round(result.retrieval_agreement, 4),
                "llm_support": result.llm_support,
                "method": result.method,
                "hallucinated": len(result.hallucinated_citations),
                "reason": result.reason,
            }
        if result is None:  # the step swallowed an exception
            result = self._degraded(state, cand, "verifier_exception")
        return result

    # -- pipeline ----------------------------------------------------------

    def _verify(
        self,
        state: QuestionState,
        cand: AnswerCandidate,
        pool: Mapping[str, Evidence],
        rec: Any,
    ) -> VerificationResult:
        started = time.perf_counter()

        # 1. Citation resolution. A cited id that does not exist contributes no
        #    support and is reported, rather than quietly dropped.
        resolved = [cid for cid in cand.citations if cid in pool]
        hallucinated = tuple(cid for cid in cand.citations if cid not in pool)

        hypothesis = cand.answer_sentence.strip() or cand.answer.strip()
        method: str = "nli"
        nli_support = 0.0
        best_premise: str | None = None
        nli_label = "neutral"
        contradictions: set[str] = set()

        # 2. Entailment over the cited premises, plus a contradiction screen
        #    over the highest-ranked uncited evidence.
        if hypothesis and self.method != "llm":
            scores, ok = self._entail(
                [(pool[cid].text, hypothesis) for cid in resolved]
            )
            if not ok:
                method = "heuristic"
                nli_support = self._containment(cand.answer, resolved, pool)
                rec.degrade("nli_unavailable")
            else:
                for cid, row in zip(resolved, scores, strict=False):
                    entail = row.get("entailment", 0.0)
                    if entail > nli_support:
                        nli_support = entail
                        best_premise = cid
                        nli_label = max(row, key=lambda k: row[k])
                    if row.get("contradiction", 0.0) > self.contradiction_threshold:
                        contradictions.add(cid)
                uncited = [
                    ev for ev in sorted(
                        pool.values(), key=lambda e: (-e.score, e.evidence_id)
                    )
                    if ev.evidence_id not in resolved
                ][:MAX_CONTRADICTION_PREMISES]
                extra, extra_ok = self._entail(
                    [(ev.text, hypothesis) for ev in uncited]
                )
                if extra_ok:
                    for ev, row in zip(uncited, extra, strict=False):
                        if row.get("contradiction", 0.0) > self.contradiction_threshold:
                            contradictions.add(ev.evidence_id)
        elif hypothesis:
            method = "llm"
            nli_support = self._containment(cand.answer, resolved, pool)

        claim = Claim(
            claim_id="c1",
            text=hypothesis,
            cited_evidence_ids=tuple(resolved),
            nli_label=nli_label,  # type: ignore[arg-type]
            nli_score=round(nli_support, 6),
            best_premise_id=best_premise,
            supported=nli_support >= self.entail_threshold,
        )

        # 3. Independent signals.
        supporting = [cid for cid in resolved if cid not in contradictions]
        citation_grounding = 1.0 if supporting else 0.0
        retrieval_agreement = self._retrieval_agreement(state, resolved, pool)

        # 4/5. Blend without the LLM term, then decide whether to buy one.
        conf_no_llm = self._blend(nli_support, citation_grounding, retrieval_agreement, None)
        llm_support: float | None = None
        adjudication: dict[str, Any] | None = None

        skip_reason = self._adjudication_skip_reason(state, cand, conf_no_llm)
        if skip_reason is None:
            adjudication = self._adjudicate(state, cand, pool, resolved, rec)
            if adjudication is not None:
                llm_support = adjudication["llm_support"]
                contradictions.update(adjudication["contradictions"])
                method = "nli_plus_llm" if method == "nli" else method
        else:
            if skip_reason != "budget_exhausted":
                # A call the gate decided against is a saved call; one the
                # budget refused is not, and conflating them would overstate
                # the efficiency claim in the results table.
                state.budget.note_saved()
            rec.input_summary["adjudication_skipped"] = skip_reason

        confidence = self._blend(
            nli_support, citation_grounding, retrieval_agreement, llm_support
        )

        # 6. Verdict. Abstention is a label: the candidate still comes back.
        failed = self._failed_subqueries(state, resolved, pool)
        reason = self._reason(cand, pool, resolved, contradictions, citation_grounding)
        verdict = self._verdict(state, cand, confidence)

        missing = self._missing_information(state, cand, failed, adjudication)
        suggested = self._suggestions(state, adjudication, failed)

        _ = time.perf_counter() - started
        return VerificationResult(
            verdict=verdict,
            candidate=AnswerCandidate(
                answer=cand.answer,
                answer_sentence=cand.answer_sentence,
                citations=cand.citations,
                cycle=cand.cycle,
                origin=cand.origin,
                sufficient=cand.sufficient,
                confidence=round(confidence, 6),
            ),
            confidence=round(confidence, 6),
            claims=(claim,),
            nli_support=round(nli_support, 6),
            citation_grounding=round(citation_grounding, 6),
            retrieval_agreement=round(retrieval_agreement, 6),
            llm_support=llm_support,
            hallucinated_citations=hallucinated,
            contradictions=tuple(sorted(contradictions)),
            missing_information=missing,
            failed_subquery_ids=failed,
            suggested_subqueries=suggested,
            reason=reason if verdict != "accept" else None,
            method=method,  # type: ignore[arg-type]
            degraded=rec.degraded,
        )

    # -- entailment helpers ------------------------------------------------

    def _scorer(self) -> NLIScorer:
        if self._nli is None:
            self._nli = _shared_nli(
                str(self.cfg.get("agents.verifier.nli_model",
                                 "cross-encoder/nli-deberta-v3-base")),
                str(self.cfg.get("agents.verifier.nli_device", "cpu")),
            )
        return self._nli

    def _entail(
        self, pairs: Sequence[tuple[str, str]]
    ) -> tuple[list[dict[str, float]], bool]:
        """Score pairs; ``(rows, ok)`` where ``ok`` is False on any failure.

        The encoder failing to load is a configuration problem on a laptop, not
        a reason to fail a 250-question run, so it degrades to containment.
        """
        if not pairs:
            return [], True
        try:
            return self._scorer().score(
                [(_clip(p), _clip(h)) for p, h in pairs]
            ), True
        except Exception:  # noqa: BLE001 - the Verifier never blocks
            return [], False

    @staticmethod
    def _containment(
        answer: str, resolved: Sequence[str], pool: Mapping[str, Evidence]
    ) -> float:
        """Heuristic stand-in for entailment: ``|answer ∩ evidence| / |answer|``."""
        answer_tokens = _tokens(answer)
        if not answer_tokens or not resolved:
            return 0.0
        evidence_tokens: set[str] = set()
        for cid in resolved:
            evidence_tokens |= _tokens(pool[cid].text)
        return len(answer_tokens & evidence_tokens) / len(answer_tokens)

    # -- signals -----------------------------------------------------------

    @staticmethod
    def _retrieval_agreement(
        state: QuestionState, resolved: Sequence[str], pool: Mapping[str, Evidence]
    ) -> float:
        """Fraction of executed sub-queries whose top-1 title was cited."""
        cited_titles = {pool[cid].title for cid in resolved if pool[cid].title}
        executed = [r for r in state.results.values() if r.passages]
        if not executed:
            return 0.0
        hits = sum(1 for r in executed if r.passages[0].passage.title in cited_titles)
        return hits / len(executed)

    def _blend(
        self, nli: float, citation: float, retrieval: float, llm: float | None
    ) -> float:
        """Weighted blend; the LLM term is dropped and the rest renormalised."""
        terms = [
            (self.weights["nli"], nli),
            (self.weights["citation"], citation),
            (self.weights["retrieval"], retrieval),
        ]
        if llm is not None:
            terms.append((self.weights["llm"], llm))
        total = sum(w for w, _ in terms)
        if total <= 0:
            return 0.0
        return sum(w * v for w, v in terms) / total

    # -- adjudication ------------------------------------------------------

    def _adjudication_skip_reason(
        self, state: QuestionState, cand: AnswerCandidate, conf_no_llm: float
    ) -> str | None:
        """Why the LLM call is not being made, or ``None`` to make it.

        The band check is 3.5 step 5. The other rungs are cheaper than the
        band: an answer the synthesiser already flagged insufficient is a
        re-plan trigger on its own (3.4) and adjudicating it would buy nothing.
        """
        if self.method == "nli":
            return "method_nli_only"
        if not cand.answer:
            return "empty_answer"
        if not cand.sufficient:
            return "synthesizer_insufficient"
        if self.method != "llm" and abs(conf_no_llm - self.threshold) > self.band:
            return "outside_uncertainty_band"
        if state.budget.remaining_llm(privileged=True) <= 0:
            return "budget_exhausted"
        return None

    def _adjudicate(
        self,
        state: QuestionState,
        cand: AnswerCandidate,
        pool: Mapping[str, Evidence],
        resolved: Sequence[str],
        rec: Any,
    ) -> dict[str, Any] | None:
        """``verifier.adjudicate.v1``. Returns ``None`` on any failure.

        The gate has already established that a call is worth making and that
        the budget can afford one; ``call_json`` is what actually spends it. A
        parse failure therefore costs the call but not the verdict:
        ``llm_support`` stays ``None`` and the blend renormalises without it.
        """
        cited = "\n".join(
            f"[{cid}] {_clip(pool[cid].text)}" for cid in resolved
        ) or "(none cited)"
        uncited_items = [
            ev for ev in sorted(pool.values(), key=lambda e: (-e.score, e.evidence_id))
            if ev.evidence_id not in resolved
        ][:MAX_CONTRADICTION_PREMISES]
        uncited = "\n".join(
            f"[{ev.evidence_id}] {_clip(ev.text)}" for ev in uncited_items
        ) or "(none)"
        call = self.call_json(
            state,
            rec,
            prompt_id=PROMPT_ID,
            variables={
                "question": state.question,
                "answer": cand.answer,
                "claim": cand.answer_sentence or cand.answer,
                "cited": cited,
                "uncited": uncited,
            },
            schema=_ADJUDICATE_SCHEMA,
            purpose="adjudicate",
            privileged=True,
            num_predict=256,
        )
        if call.failed:
            rec.degrade(f"adjudication_failed:{call.reason}")
            return None

        parsed = call.parsed or {}
        supported = bool(parsed.get("supported"))
        try:
            claimed = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            claimed = 0.0
        claimed = min(1.0, max(0.0, claimed))
        return {
            # The term measures *support*: an adjudicator that says the answer
            # is unsupported contributes none, whatever confidence it attaches
            # to that judgement.
            "llm_support": claimed if supported else 0.0,
            "missing_information": tuple(
                str(x).strip() for x in (parsed.get("missing_information") or ())
                if str(x).strip()
            )[:5],
            "contradictions": tuple(
                str(x).strip() for x in (parsed.get("contradictions") or ())
                if str(x).strip() in pool
            ),
            "verdict": str(parsed.get("verdict") or ""),
        }

    # -- verdict and feedback ---------------------------------------------

    def _verdict(
        self, state: QuestionState, cand: AnswerCandidate, confidence: float
    ) -> Verdict:
        """3.5 step 6, plus the synthesiser's own insufficiency flag."""
        replans_left = state.budget.replans < state.budget.max_replans
        if not cand.sufficient or not cand.answer:
            return "revise" if replans_left else "abstain"
        if confidence >= self.threshold:
            return "accept"
        return "revise" if replans_left else "abstain"

    @staticmethod
    def _reason(
        cand: AnswerCandidate,
        pool: Mapping[str, Evidence],
        resolved: Sequence[str],
        contradictions: set[str],
        citation_grounding: float,
    ) -> ReplanReason:
        """First matching cause, most specific first, so the trace is diagnostic."""
        if not cand.sufficient:
            return "synthesizer_insufficient"
        if not pool:
            return "empty_retrieval"
        if not cand.citations or not resolved:
            return "no_citations"
        if contradictions:
            return "contradiction"
        if citation_grounding <= 0.0:
            return "missing_evidence"
        return "low_confidence"

    @staticmethod
    def _failed_subqueries(
        state: QuestionState, resolved: Sequence[str], pool: Mapping[str, Evidence]
    ) -> tuple[str, ...]:
        """Sub-queries that returned nothing the answer could use.

        Combiner nodes are exempt: they are never retrieved for by design
        (architecture 3.1), so counting them as failures would send the Planner
        chasing a sub-query it was right not to run.
        """
        plan = state.plan
        if plan is None:
            return ()
        cited_subqueries = {
            sq for cid in resolved for sq in pool[cid].subquery_ids
        }
        failed: list[str] = []
        for sq in plan.subqueries:
            if sq.is_combiner:
                continue
            result = state.results.get(sq.id)
            if result is None or not result.passages or sq.id not in cited_subqueries:
                failed.append(sq.id)
        return tuple(failed)

    @staticmethod
    def _missing_information(
        state: QuestionState,
        cand: AnswerCandidate,
        failed: Sequence[str],
        adjudication: Mapping[str, Any] | None,
    ) -> tuple[str, ...]:
        """What could not be established, in plain English.

        The adjudicator's list is preferred when it exists -- it is specific
        ("the founding year of First for Women") in a way a rule cannot be --
        and the deterministic list is what keeps guard G5 satisfiable when no
        call was made.
        """
        if adjudication and adjudication.get("missing_information"):
            return tuple(adjudication["missing_information"])
        out: list[str] = []
        plan = state.plan
        if plan is not None:
            by_id = {sq.id: sq for sq in plan.subqueries}
            out.extend(by_id[sq_id].text for sq_id in failed if sq_id in by_id)
        if not cand.citations:
            out.append(f"supporting evidence for: {cand.answer_sentence or state.question}")
        if not state.evidence:
            out.append(f"any passage relevant to: {state.question}")
        return tuple(dict.fromkeys(x for x in out if x))[:5]

    @staticmethod
    def _suggestions(
        state: QuestionState,
        adjudication: Mapping[str, Any] | None,
        failed: Sequence[str],
    ) -> tuple[str, ...]:
        """Sub-queries the Planner *may* try next. Never a repeat of a banned one.

        The deterministic proposals splice an entity already established into
        the original question. That is the cheapest way to produce something
        materially different from every previous sub-query, which is exactly
        what the re-plan prompt demands of the model.
        """
        banned = {t.strip().lower() for t in state.banned_texts()}
        out: list[str] = []
        for entity in state.covered_entities()[:2]:
            proposal = f"{entity} {state.question}"
            if proposal.strip().lower() not in banned:
                out.append(proposal)
        if adjudication and adjudication.get("missing_information"):
            for item in adjudication["missing_information"]:
                if item.strip().lower() not in banned:
                    out.append(item)
        if not out and failed:
            plan = state.plan
            if plan is not None:
                by_id = {sq.id: sq for sq in plan.subqueries}
                for sq_id in failed:
                    sq = by_id.get(sq_id)
                    if sq is not None and sq.entities:
                        out.append(f"{sq.entities[0]} {state.question}")
        return tuple(dict.fromkeys(out))[:3]

    # -- degraded outcome --------------------------------------------------

    def _degraded(
        self, state: QuestionState, cand: AnswerCandidate, reason: str
    ) -> VerificationResult:
        """The floor: something above raised, but a verdict is still produced."""
        replans_left = state.budget.replans < state.budget.max_replans
        return VerificationResult(
            verdict="revise" if replans_left else "abstain",
            candidate=cand,
            confidence=0.0,
            missing_information=(f"verification failed ({reason})",),
            reason="low_confidence",
            method="heuristic",
            degraded=True,
        )

    # -- feedback edge -----------------------------------------------------

    @staticmethod
    def build_directive(
        state: QuestionState,
        verification: VerificationResult,
        *,
        revision: int | None = None,
    ) -> ReplanDirective:
        """Convenience wrapper over the module-level :func:`build_directive`."""
        return build_directive(state, verification, revision=revision)


def build_directive(
    state: QuestionState,
    verification: VerificationResult,
    *,
    revision: int | None = None,
) -> ReplanDirective:
    """Turn a ``revise`` verdict into instructions the Planner can act on.

    This is the contribution made concrete: the only backwards edge in the
    system, and the one object that makes the loop *closed* rather than merely
    iterative.

    ``banned_subquery_texts`` is the load-bearing field. Measured behaviour of
    a local 8B model told "that was insufficient, try again" is to re-emit the
    same decomposition, which burns both re-plans and produces an identical
    answer -- and, because of the near-duplicate test (2.4), terminates the
    question as ``degenerate_replan``. Handing the model the exact list of
    texts it has already tried is what turns the second cycle into a different
    search rather than a slower copy of the first.

    ``covered_entities`` and ``seen_doc_ids`` carry the complementary pressure:
    do not re-establish what is already established, and prefer documents not
    yet seen.
    """
    next_revision = revision if revision is not None else len(state.plans)
    return ReplanDirective(
        directive_id=f"{state.qid}:r{next_revision}",
        revision=next_revision,
        reason=verification.reason or "low_confidence",
        confidence=round(verification.confidence, 6),
        missing_information=verification.missing_information,
        failed_subquery_ids=verification.failed_subquery_ids,
        suggested_subqueries=verification.suggested_subqueries,
        covered_entities=state.covered_entities(),
        seen_doc_ids=state.seen_doc_ids(),
        banned_subquery_texts=state.banned_texts(),
        previous_plan_summary=_plan_summary(state),
    )


def _plan_summary(state: QuestionState) -> str:
    """Compact ``q1: ... -> q2: ...`` rendering of the latest plan.

    Titles and sub-query texts only -- never retrieved passage text. At 52
    tok/s prompt length is latency, and this keeps the re-plan prompt at a few
    hundred tokens instead of several thousand (3.1).
    """
    plan = state.plan
    if plan is None:
        return ""
    parts = []
    for sq in plan.subqueries:
        deps = f"<-{','.join(sq.depends_on)}" if sq.depends_on else ""
        parts.append(f"{sq.id}{deps}: {sq.text}")
    return " -> ".join(parts)
