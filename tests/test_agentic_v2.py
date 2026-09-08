"""``agentic_v2``: the registration, and the invariant it must not break.

``agentic_v2`` is the accuracy plan's Round 2 system -- the one-node plan
carrying fix 1A (answer/sentence reconciliation) and fix 1C (rank-major
evidence ranking). It is *not* one of the nine systems the report evaluates,
and the point of these tests is that it stays that way in both directions:

* nothing about its existence may change what ``agentic_full`` does, because
  eighteen evaluated runs are what the report's tables are built from;
* nothing about the report's protective machinery may quietly stop
  ``agentic_v2`` from running, because a configuration that silently degrades
  to no LLM client still produces a number.

The behavioural half of the first claim -- that the changed code paths return
bit-identical results under the shipped defaults -- is not asserted here but
demonstrated by differential replay over the two 250-question eval traces; see
``test_evidence_ranking_defaults_to_the_evaluated_key`` and
``test_lexical_major_reproduces_the_pre_1c_scores_exactly`` in
``tests/test_orchestrator.py`` for the unit-level pins.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_ir.config import load_config
from agentic_ir.eval.run_eval import (
    ABLATION_OVERRIDES,
    AGENTIC_CONFIGS,
    BASELINE_CONFIGS,
    CONFIGURATIONS,
    ONE_NODE_CONFIGS,
    NoPlanner,
    Pipeline,
    build_parser,
    build_system,
    config_for,
    configurations,
    is_agentic,
)
from agentic_ir.tools.registry import ToolRegistry

#: The keys ``agentic_v2`` flips, and the value each must hold by default.
EVALUATED_DEFAULTS = {
    "agents.synthesizer.reconcile_answer": False,
    "agents.synthesizer.answer_type_guard": False,
    "agents.verifier.evidence_ranking": "lexical_major",
    # L1's three. ``think_agents`` being absent is what makes ``think_for``
    # fall back to the caller's argument, which is what the grid ran; naming
    # any agent there would change every evaluated run.
    "llm.think": False,
    "llm.think_agents": None,
    "llm.options.num_predict": 1024,
}


#: The report's configuration, by path: these tests must reach it whatever
#: ``AGENTIC_IR_CONFIG`` happens to be pointing at.
SHIPPED_CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# ---------------------------------------------------------------------------
# The report is not to grow a tenth row
# ---------------------------------------------------------------------------

def test_agentic_v2_is_not_one_of_the_nine_reported_systems():
    """``tables.py`` renders a row per name in both of these tuples.

    ``discover_runs`` iterates ``configurations(cfg)`` and the ablation table
    iterates :data:`AGENTIC_CONFIGS`, so a name in either becomes a row -- an
    empty one if no run exists, which is worse than no row at all.
    """
    assert "agentic_v2" not in CONFIGURATIONS
    assert "agentic_v2" not in AGENTIC_CONFIGS
    assert "agentic_v2" not in BASELINE_CONFIGS


def test_the_shipped_config_still_names_exactly_the_nine():
    """``config/config.yaml`` is the report's configuration and stays so.

    ``agentic_v2`` is enabled by listing it in a *variant* config; if it ever
    reaches this list, ``discover_runs`` starts looking for its runs and the
    grid becomes ten systems.
    """
    shipped = configurations(load_config(SHIPPED_CONFIG))
    assert shipped == CONFIGURATIONS


# ---------------------------------------------------------------------------
# ... and is not to be silently unrunnable either
# ---------------------------------------------------------------------------

def test_agentic_v2_runs_through_the_orchestrator():
    """The runtime question is "is it a baseline?", not "is it in the grid?".

    :func:`is_agentic` gates the LLM client and the NLI preflight. Had those
    kept asking for :data:`AGENTIC_CONFIGS` membership, ``agentic_v2`` would
    have run with ``client=None``: an orchestrator that degrades to extractive
    answers without raising, and reports a number for it.
    """
    assert is_agentic("agentic_v2")
    assert all(is_agentic(name) for name in AGENTIC_CONFIGS)
    assert not any(is_agentic(name) for name in BASELINE_CONFIGS)


def test_agentic_v2_gets_the_one_node_plan_from_build_system_not_the_override():
    """The one-node plan is a Planner substitution, not a config switch.

    The accuracy plan (R2.4) specifies ``agentic_v2`` as
    ``agents.planner.template_shortcut: False`` with the comment
    "= agentic_no_planner". It is not: ``false`` is already the shipped value,
    so the override is a no-op, and ``agentic_no_planner``'s behaviour comes
    from :class:`NoPlanner` in :func:`build_system`. Taking the plan literally
    would have produced a system that runs the full LLM Planner under a name
    that promises it does not.
    """
    assert "agentic_v2" in ONE_NODE_CONFIGS
    assert "agentic_no_planner" in ONE_NODE_CONFIGS
    assert load_config().get("agents.planner.template_shortcut") is False


@pytest.mark.parametrize(
    "config_name,one_node,reconcile,ranking",
    [
        ("agentic_full", False, False, "lexical_major"),
        ("agentic_no_planner", True, False, "lexical_major"),
        ("agentic_v2", True, False, "lexical_major"),
    ],
)
def test_build_system_wires_the_flags_through(
    cfg, config_name, one_node, reconcile, ranking
):
    """End to end from the configuration name to the objects that run.

    The one place all three seams meet: which Planner, which evidence key, and
    whether the Synthesizer reconciles. The indexes are stubbed -- nothing here
    retrieves -- but the Orchestrator, Synthesizer and Verifier are the real
    ones, built the way ``run_eval`` builds them.
    """
    pipeline = Pipeline(dataset="hotpotqa", registry=ToolRegistry())
    system = build_system(
        config_name, "hotpotqa", cfg=config_for(config_name, cfg), pipeline=pipeline
    )
    assert isinstance(system.planner, NoPlanner) is one_node
    assert system.evidence_ranking == ranking
    assert system.evidence_docs == 3
    synth = system._optional("synthesizer")
    assert synth is not None and synth.reconcile is reconcile


def test_the_unshipped_fixes_still_wire_through_when_asked_for(cfg):
    """No configuration sets them, so this is what keeps their wiring honest.

    L0's two fixes are code the loop measured and declined to ship, not code
    it deleted: the flags remain, and a run that wants to measure one can
    still switch it on. Nothing else in the suite now builds a system with
    them enabled, so without this test the seam between the flag and the
    object could rot unnoticed.
    """
    derived = cfg.with_overrides(
        {
            "agents.synthesizer.reconcile_answer": True,
            "agents.verifier.evidence_ranking": "rank_major",
        }
    )
    pipeline = Pipeline(dataset="hotpotqa", registry=ToolRegistry())
    system = build_system("agentic_v2", "hotpotqa", cfg=derived, pipeline=pipeline)
    assert system.evidence_ranking == "rank_major"
    synth = system._optional("synthesizer")
    assert synth is not None and synth.reconcile is True


# ---------------------------------------------------------------------------
# The flags
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dotted,evaluated", sorted(EVALUATED_DEFAULTS.items()))
def test_every_fix_defaults_to_the_evaluated_behaviour(cfg, dotted, evaluated):
    """Round 2's whole premise: the corrections are selectable, not imposed."""
    assert cfg.get(dotted) == evaluated


@pytest.mark.parametrize("config_name", ["agentic_full", *AGENTIC_CONFIGS])
def test_no_evaluated_configuration_turns_a_fix_on(cfg, config_name):
    derived = config_for(config_name, cfg)
    for dotted, evaluated in EVALUATED_DEFAULTS.items():
        assert derived.get(dotted) == evaluated, (config_name, dotted)


#: The three fixes L0 built and the loop declined to ship. Their flags exist,
#: their tests pass, and no configuration in the tree sets them.
UNSHIPPED_FIXES = (
    "agents.synthesizer.reconcile_answer",
    "agents.synthesizer.answer_type_guard",
    "agents.verifier.evidence_ranking",
)


@pytest.mark.parametrize("dotted", UNSHIPPED_FIXES)
def test_no_configuration_at_all_turns_an_unshipped_fix_on(cfg, dotted):
    """Including ``agentic_v2``, which is the one that used to."""
    for config_name in ("agentic_full", *AGENTIC_CONFIGS, "agentic_v2"):
        derived = config_for(config_name, cfg)
        assert derived.get(dotted) == EVALUATED_DEFAULTS[dotted], config_name


def test_agentic_v2_ships_none_of_the_three_unmeasured_fixes(cfg):
    """The configuration is what the confirmation runs measured, and no more.

    All three shipped here until Loop 4. 1A recovers 0 questions on this base
    and 1B changes 0 answer types under a one-node plan, both from offline
    replay over 500 traces. 1C is the one that mattered: on 30 paired
    questions it cost 2x the median latency and lost a synthesis call on 10%
    of them against 0%, for a paired exact-match difference of +0.0000. None
    of the three earned a place in a configuration whose numbers get cited,
    and this test is what stops one drifting back in.
    """
    derived = config_for("agentic_v2", cfg)
    assert derived.get("agents.synthesizer.reconcile_answer") is False
    assert derived.get("agents.synthesizer.answer_type_guard") is False
    assert derived.get("agents.verifier.evidence_ranking") == "lexical_major"
    # 2A stays off: sentence-level widening measured net zero once KG rows were
    # counted, and is explicitly out of scope for this configuration.
    assert derived.get("agents.verifier.evidence_docs") == 3


def test_agentic_v2_carries_the_confirmed_thinking_settings(cfg):
    """L1's KEEP is a property of the branch, not of one worktree's config.

    The +0.072 (HotpotQA) / +0.140 (2Wiki) exact-match gain was measured with
    reasoning enabled on the synthesiser and the completion budget raised to
    hold the reasoning block. Both settings sat in the executing worktree's
    config file, so for a while ``git checkout v2`` gave you the code that can
    route thinking per agent beside a configuration that never asks it to.
    This pins the decision to the branch.
    """
    from agentic_ir.llm import LLMSettings

    derived = config_for("agentic_v2", cfg)
    settings = LLMSettings.from_config(derived)
    assert settings.think_for("synthesizer") is True
    for other in ("planner", "retriever", "kg_navigator", "verifier"):
        assert settings.think_for(other) is False, other
    # Thinking without the budget is measurably worse than not thinking: the
    # block runs to the cap and the answer never starts.
    assert derived.get("llm.options.num_predict") == 5120


def test_the_override_does_not_reach_into_the_shared_config(cfg):
    """``config_for`` derives; the process-wide instance must not be mutated."""
    config_for("agentic_v2", cfg)
    for dotted, evaluated in EVALUATED_DEFAULTS.items():
        assert cfg.get(dotted) == evaluated


def test_the_override_touches_nothing_else():
    """Anything beyond these four keys is a change nobody asked for."""
    assert set(ABLATION_OVERRIDES["agentic_v2"]) == {
        "agents.planner.template_shortcut",
        "llm.think",
        "llm.think_agents",
        "llm.options.num_predict",
    }


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------

def test_config_choices_come_from_the_config_file(cfg):
    """``--config agentic_v2`` is legal exactly when the config file says so.

    Both directions, and neither read off whichever config the suite happens to
    be running under: the point is that the *file* is what decides, so the
    listing is supplied explicitly here.
    """
    listing = [*CONFIGURATIONS, "agentic_v2"]
    permissive = build_parser(cfg.with_overrides({"evaluation.configurations": listing}))
    assert permissive.parse_args(["--config", "agentic_v2"]).config_name == "agentic_v2"

    shipped = build_parser(
        cfg.with_overrides({"evaluation.configurations": list(CONFIGURATIONS)})
    )
    with pytest.raises(SystemExit):
        shipped.parse_args(["--config", "agentic_v2"])
    assert shipped.parse_args(["--config", "self_ask"]).config_name == "self_ask"


def test_the_default_configuration_is_unchanged(cfg):
    assert build_parser(cfg).parse_args([]).config_name == "agentic_full"
