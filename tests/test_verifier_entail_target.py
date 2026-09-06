"""The entailment hypothesis, and the veto that makes it mean something.

The Verifier's original design entails ``answer_sentence`` and never
``answer``. Measured over both evaluated datasets that makes the entailment
score a coin flip -- AUC 0.529 and 0.508, both intervals containing chance --
because the Synthesizer frequently copies a premise verbatim into the answer
sentence, and a premise entails its own copy at ~1.0 whatever the answer says.

These tests pin two things. First, that the default is still ``sentence``, so
every number in the evaluated grid stays reproducible. Second, that
``answer_bearing`` rejects the exact record the report quotes as a false
accept, using that record's own strings rather than an invented example.
"""

from __future__ import annotations

import pytest

from agentic_ir.agents.verifier import Verifier, _normalise_for_containment
from agentic_ir.config import load_config
from agentic_ir.types import AnswerCandidate

#: qid 5a7af32e55429931da12c99c. The Synthesizer answered "New York City" and
#: offered, as the sentence supporting it, a verbatim copy of the cited premise
#: -- which is about a different song and does not contain the answer at all.
#: NLI scored 0.995, the blend reached 0.821, and the answer was accepted.
FALSE_ACCEPT_ANSWER = "New York City"
FALSE_ACCEPT_SENTENCE = (
    "Fallin' is a collaboration between Scottish power pop band Teenage "
    "Fanclub and American alternative hip hop trio De La Soul."
)

#: qid 5a7557d75542992d0ec05f68, for contrast: a genuine synthesis that names
#: its own answer. Whatever else is wrong with that record -- the citations are
#: poor and NLI scored 0.009 -- the veto must not fire on it.
GENUINE_ANSWER = "Laurie Metcalf"
GENUINE_SENTENCE = "Laurie Metcalf is in Scream 2."


def verifier(target: str | None = None) -> Verifier:
    cfg = load_config()
    if target is not None:
        cfg = cfg.override({"agents.verifier.entail_target": target}) \
            if hasattr(cfg, "override") else cfg
    v = Verifier(cfg)
    if target is not None:
        v.entail_target = target
    return v


def candidate(answer: str, sentence: str) -> AnswerCandidate:
    return AnswerCandidate(
        answer=answer, answer_sentence=sentence, citations=("e1",),
        cycle=0, origin="llm", sufficient=True,
    )


def test_the_default_is_still_the_evaluated_behaviour():
    """The grid ran under `sentence`. Changing the default silently would make
    every reported number irreproducible from this code."""
    assert Verifier(load_config()).entail_target == "sentence"


def test_sentence_mode_does_not_veto_anything():
    v = verifier("sentence")
    hypothesis, veto = v._hypothesis(candidate(FALSE_ACCEPT_ANSWER, FALSE_ACCEPT_SENTENCE))
    assert veto is None
    assert hypothesis == FALSE_ACCEPT_SENTENCE


def test_answer_bearing_vetoes_the_documented_false_accept():
    """The record the report quotes must be rejected by the fix."""
    v = verifier("answer_bearing")
    _, veto = v._hypothesis(candidate(FALSE_ACCEPT_ANSWER, FALSE_ACCEPT_SENTENCE))
    assert veto == "answer_absent_from_its_own_sentence"


def test_answer_bearing_passes_a_sentence_that_names_its_answer():
    v = verifier("answer_bearing")
    hypothesis, veto = v._hypothesis(candidate(GENUINE_ANSWER, GENUINE_SENTENCE))
    assert veto is None
    assert hypothesis == GENUINE_SENTENCE


def test_answer_bearing_declines_a_bare_answer_with_no_sentence():
    """A bare entity is not a proposition; scoring NLI over it is noise."""
    v = verifier("answer_bearing")
    hypothesis, veto = v._hypothesis(candidate("Vince Staples", ""))
    assert veto == "no_answer_sentence"
    assert hypothesis == ""


def test_containment_respects_word_boundaries():
    """Substring containment on raw text would match 'Ann' inside 'Anne'."""
    assert _normalise_for_containment("Ann") not in _normalise_for_containment(
        "Anne Boleyn was queen."
    )
    assert _normalise_for_containment("Anne") in _normalise_for_containment(
        "Anne Boleyn was queen."
    )


def test_containment_is_phrase_not_token_set():
    """A token-set test would accept the answer's words scattered anywhere."""
    scattered = "The city of new administration in york was a soul project."
    assert _normalise_for_containment("New York City") not in _normalise_for_containment(
        scattered
    )


@pytest.mark.parametrize("punctuation", ['"Laurie Metcalf" is in Scream 2.',
                                         "Laurie Metcalf, is in Scream 2.",
                                         "It is LAURIE METCALF."])
def test_containment_ignores_case_and_punctuation(punctuation):
    assert _normalise_for_containment("Laurie Metcalf") in _normalise_for_containment(
        punctuation
    )
