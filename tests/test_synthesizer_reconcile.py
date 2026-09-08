"""Reconciling ``answer`` against the model's own ``answer_sentence``.

Accuracy plan 1A. The model writes a correct sentence and then fills ``answer``
with the wrong constituent of it: 24 of the 250 questions in
``agentic_full_hotpotqa_20260904T221533Z`` carry the gold answer inside the
sentence the model itself produced. Every case pinned below is one of those
records, quoted verbatim rather than invented, so the tests fail if the rule
stops recovering the questions it was measured on.

The other half of the contract is restraint, and it is the half that has already
failed once. The first version of this rule also fired when the answer was
absent from the sentence, which rewrote a correct "football" into "Bayer
Leverkusen" and made the change a net regression on calib. The negative cases
below therefore carry more weight than the positive ones, and
``test_absence_is_not_contradiction`` pins that specific record.
"""

from __future__ import annotations

import pytest

from agentic_ir.agents.synthesizer import (
    infer_answer_type,
    question_shape,
    reconcile_answer,
)
from agentic_ir.types import Plan, SubQuery

# ---------------------------------------------------------------------------
# 1A: recoveries. (question, answer, sentence, expected, reason)
# ---------------------------------------------------------------------------

RECOVERED = [
    pytest.param(
        "Andrea Camplone is a football coach in which 22-team competition?",
        "22",
        "The Serie B competition has 22 teams.",
        "Serie B",
        "type_mismatch",
        id="bare-number-for-which-question",
    ),
    pytest.param(
        "Who was born first, Krzysztof Zanussi or Thom Andersen?",
        "1939",
        "Krzysztof Zanussi was born in 1939.",
        "Krzysztof Zanussi",
        "type_mismatch",
        id="year-instead-of-person",
    ),
    pytest.param(
        "Which director, John Schlesinger or Barbara Albert, was also a writer "
        "and film producer?",
        "yes",
        "Barbara Albert was also a writer and film producer.",
        "Barbara Albert",
        "yesno_slot",
        id="yes-for-a-which-question",
    ),
    pytest.param(
        "Both Mulled wine and Blue Lagoon are considered primarily alcoholic "
        "or non-alcoholic?",
        "no",
        "Blue Lagoon is a cocktail and Mulled wine is an alcoholic drink.",
        "alcoholic",
        "yesno_slot",
        id="no-for-an-alternative-question",
    ),
    pytest.param(
        "What date in 2010 was a South Korean film starring Kim Hyang-gi "
        "released?",
        "January 14",
        "Wedding Dress was released on January 14, 2010.",
        "January 14, 2010",
        "underspecified_date",
        id="truncated-date",
    ),
]


@pytest.mark.parametrize("question,answer,sentence,expected,reason", RECOVERED)
def test_reconciles_a_measured_slot_error(question, answer, sentence, expected, reason):
    got, why = reconcile_answer(question, answer, sentence)
    assert got == expected
    assert why == reason


# ---------------------------------------------------------------------------
# Restraint. Each of these is a record the rule must leave alone.
# ---------------------------------------------------------------------------

UNTOUCHED = [
    pytest.param(
        "Which Italian-American composer and librettist wrote the English "
        "language opera, Maria Golovin?",
        "Gian Carlo Menotti",
        "Gian Carlo Menotti wrote the English language opera, Maria Golovin.",
        id="already-correct-and-consistent",
    ),
    pytest.param(
        # A genuine yes/no question: "yes" is never literally in the sentence,
        # and without the yes/no exclusion the contradiction trigger eats it.
        "Are Dogo Cubano and the Dutch Shepherd both breeds of dog?",
        "yes",
        "Dogo Cubano and the Dutch Shepherd are both breeds of dog.",
        id="genuine-yesno-answer",
    ),
    pytest.param(
        # The interrogative that governs is the trailing "in what year?", not
        # the relative "who". Reading the relative pronoun as the question's
        # focus overwrote this correct 1999 with the actress's name.
        "The Duke Steps Out stars an actress who was ranked tenth on a list of "
        "greatest female Hollywood stars in what year?",
        "1999",
        "Joan Crawford was ranked tenth on the list of greatest female "
        "Hollywood stars in 1999.",
        id="relative-pronoun-is-not-the-interrogative",
    ),
    pytest.param(
        # Both arms are asserted by the sentence, so the disjunction does not
        # decide the question and the rule must not pick one.
        "Who is younger Jenny Bae or Lionel Richie ?",
        "no",
        "Lionel Richie is older than Jenny Bae.",
        id="both-arms-present-declines",
    ),
    pytest.param(
        # The sentence is a copy of an unrelated premise and names no
        # alternative; nothing in it is a defensible year.
        "In what year was the band formed?",
        "1985",
        "The band has released several albums.",
        id="nothing-to-select",
    ),
    pytest.param(
        # qid 5a7ae2f2554299042af8f6aa. The answer is correct and simply does
        # not appear in the sentence. See test_absence_is_not_contradiction.
        "Which sport has been played at the BayArena in Leverkusen, Germany, "
        "since 1958?",
        "football",
        "The BayArena has been the home ground of Bayer Leverkusen since 1958.",
        id="correct-answer-absent-from-its-own-sentence",
    ),
    pytest.param(
        # The same shape with a hyphenated near-miss. The first version of the
        # rule recovered this one ("Lithuanian" -> "Lithuanian-born French",
        # which is the gold), but only via the trigger that broke "football".
        # One recovery does not buy back a trigger that rewrites correct
        # answers, so this is now deliberately left alone.
        "Jacques Sernas, actor in Fugitive in Trieste, was of what nationality?",
        "Lithuanian",
        "Jacques Sernas was a Lithuanian-born French actor.",
        id="hyphenated-near-miss-left-alone",
    ),
]


@pytest.mark.parametrize("question,answer,sentence", UNTOUCHED)
def test_declines_rather_than_guesses(question, answer, sentence):
    got, why = reconcile_answer(question, answer, sentence)
    assert got == answer
    assert why is None


def test_absence_is_not_contradiction():
    """The regression that this rule's first version shipped, pinned.

    qid 5a7ae2f2554299042af8f6aa, calib. "football" is the gold answer. The
    model answered it from the evidence pool and then wrote a sentence about the
    stadium's tenant instead of about the sport, so the answer does not appear
    in its own answer_sentence. Inferring self-contradiction from that absence
    rewrote a correct answer into "Bayer Leverkusen".

    No trigger may fire on this record. The property is general -- the model may
    always answer correctly while writing a sentence that does not restate the
    answer -- so this is not an exception list of one.
    """
    question = (
        "Which sport has been played at the BayArena in Leverkusen, Germany, "
        "since 1958?"
    )
    sentence = (
        "The BayArena has been the home ground of Bayer Leverkusen since 1958."
    )
    assert reconcile_answer(question, "football", sentence) == ("football", None)


def test_empty_inputs_are_passed_through():
    assert reconcile_answer("Who?", "", "A sentence.") == ("", None)
    assert reconcile_answer("Who?", "Someone", "") == ("Someone", None)


# ---------------------------------------------------------------------------
# question_shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "question,shape",
    [
        ("Who wrote Maria Golovin?", "entity"),
        ("Are they both dogs?", "yesno"),
        ("How many copies did it sell?", "number"),
        ("In what year did it open?", "date"),
        # Mid-question interrogatives: the last one governs.
        ("The publication of which magazine ended first?", "entity"),
        ("An actress who was ranked tenth did so in what year?", "date"),
        ("Andrea Camplone coaches in which 22-team competition?", "entity"),
        ("Nothing interrogative here at all.", "string"),
    ],
)
def test_question_shape(question, shape):
    assert question_shape(question) == shape


# ---------------------------------------------------------------------------
# 1B: a plan node may refine the answer type but never flip the yes/no bit.
# ---------------------------------------------------------------------------

def _plan(question: str, answer_type: str) -> Plan:
    return Plan(
        question=question,
        subqueries=(
            SubQuery(id="q1", text="who?", hop=1, answer_type="entity"),
            SubQuery(
                id="q2",
                text="Is {{q1.answer}} a retired soccer player?",
                depends_on=("q1",),
                hop=2,
                answer_type=answer_type,
            ),
        ),
    )


def test_plan_may_not_flip_an_entity_question_into_yesno():
    """The last node of a bridge chain is often a yes/no verification hop.

    Its type describes that hop's own sub-answer, not the final answer, and
    "yesno" is the one label that makes the model emit the literal string "yes"
    for a question whose answer is a name.
    """
    question = "Which player did the club sign in 1998?"
    plan = _plan(question, "yesno")
    assert infer_answer_type(question, plan, yesno_guard=True) == "entity"


def test_plan_may_not_flip_a_yesno_question_out_of_yesno():
    question = "Are both bands from Japan?"
    plan = _plan(question, "entity")
    assert infer_answer_type(question, plan, yesno_guard=True) == "yesno"


def test_the_yesno_guard_is_off_unless_asked_for():
    """1B is a flag, and its default is the behaviour the report was written on.

    ``infer_answer_type`` sets the ``answer_type`` rendered into the synthesis
    prompt, so this is not a post-processing repair that can be applied for
    free: it changes what the model is asked. Replaying both 250-question eval
    traces, the guard changes the type on 12 of ``agentic_full``'s 250 questions
    and on 0 of ``agentic_no_planner``'s, whose plans are all a single "string"
    node. So it ships off, ``agentic_v2`` turns it on, and the twelve questions
    stay where the report left them.
    """
    question = "Which player did the club sign in 1998?"
    plan = _plan(question, "yesno")
    assert infer_answer_type(question, plan) == "yesno"
    assert infer_answer_type(question, plan, yesno_guard=False) == "yesno"


def test_a_single_node_string_plan_makes_the_guard_unreachable():
    """Why gating 1B costs ``agentic_v2`` nothing.

    Every one of the 250 plans in ``agentic_no_planner_hotpotqa_20260905T224906Z``
    is one node of type "string". The refinement branch the guard sits inside is
    never entered, so both settings agree -- which is what makes the flag a
    statement of intent in a one-node system rather than a behaviour.
    """
    for question in ("Who wrote Maria Golovin?", "Are both bands from Japan?"):
        plan = _plan(question, "string")
        assert infer_answer_type(question, plan) == infer_answer_type(
            question, plan, yesno_guard=True
        )


def test_plan_still_refines_within_the_same_yesno_bit():
    """The guard is about the yes/no bit only; other refinements survive."""
    question = "What did the treaty establish?"
    assert infer_answer_type(question, _plan(question, "date")) == "date"


def test_string_is_still_treated_as_no_information():
    question = "Who wrote Maria Golovin?"
    assert infer_answer_type(question, _plan(question, "string")) == "entity"
