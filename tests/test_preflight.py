"""The NLI preflight, and the supporting-fact protocol fields of ``score_records``.

Both guard the same class of defect: a run that completes, produces plausible
numbers, and measures something other than the system. ``verifier.preflight_nli``
documents the first case -- a silently degraded NLI scores 1.000 where DeBERTa
scores 0.0008 and switches the re-plan loop off for the whole run -- and it was
never called by the harness. The second is the cited-subset / whole-pool
protocol split that ``predicted_supporting_facts`` performs per record and the
tables have to be able to see.

The encoder is never loaded here: ``CrossEncoderNLI`` is replaced by a double,
so the tests are CPU-only and run beside a GPU sweep without touching it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_ir.agents import verifier as verifier_module
from agentic_ir.agents.verifier import preflight_nli
from agentic_ir.config import load_config
from agentic_ir.eval import run_eval as harness
from agentic_ir.eval.run_eval import (
    RunSpec,
    config_for,
    pool_supporting_facts,
    predicted_supporting_facts,
    preflight_or_abort,
    resolved_citations,
    run_eval,
    score_records,
    uses_verifier,
)
from agentic_ir.types import GoldAnswer


# ---------------------------------------------------------------------------
# NLI doubles
# ---------------------------------------------------------------------------

class _HealthyNLI:
    def __init__(self, name: str, device: str) -> None:
        self.name, self.device = name, device

    def score(self, pairs):
        return [{"entailment": 0.99, "neutral": 0.005, "contradiction": 0.005} for _ in pairs]


class _MisalignedNLI(_HealthyNLI):
    def score(self, pairs):
        return [{"entailment": 0.01, "neutral": 0.01, "contradiction": 0.98} for _ in pairs]


class _SilentNLI(_HealthyNLI):
    def score(self, pairs):
        return []


class _BrokenNLI:
    def __init__(self, name: str, device: str) -> None:
        raise OSError("model weights not found")


@pytest.fixture
def cfg():
    return load_config()


# ---------------------------------------------------------------------------
# preflight_nli itself
# ---------------------------------------------------------------------------

def test_preflight_passes_on_a_healthy_encoder(monkeypatch, cfg):
    monkeypatch.setattr(verifier_module, "CrossEncoderNLI", _HealthyNLI)
    ok, message = preflight_nli(cfg)
    assert ok
    assert "NLI ok" in message


@pytest.mark.parametrize("double,fragment", [
    (_BrokenNLI, "NLI unavailable"),
    (_SilentNLI, "returned no scores"),
    (_MisalignedNLI, "misaligned"),
])
def test_preflight_reports_each_degraded_state(monkeypatch, cfg, double, fragment):
    monkeypatch.setattr(verifier_module, "CrossEncoderNLI", double)
    ok, message = preflight_nli(cfg)
    assert not ok
    assert fragment in message


# ---------------------------------------------------------------------------
# The harness: abort, not warn
# ---------------------------------------------------------------------------

def test_uses_verifier_mirrors_build_system(cfg):
    assert uses_verifier("agentic_full", config_for("agentic_full", cfg))
    assert uses_verifier("agentic_no_planner", config_for("agentic_no_planner", cfg))
    assert not uses_verifier("agentic_no_verifier", config_for("agentic_no_verifier", cfg))
    assert not uses_verifier("hybrid_rerank", cfg)
    assert not uses_verifier("self_ask", cfg)


def test_preflight_or_abort_does_not_apply_without_a_verifier(monkeypatch, cfg):
    # Even a broken encoder is irrelevant to a configuration that never loads one.
    monkeypatch.setattr(verifier_module, "CrossEncoderNLI", _BrokenNLI)
    assert preflight_or_abort("bm25_only", cfg) is None
    assert preflight_or_abort(
        "agentic_no_verifier", config_for("agentic_no_verifier", cfg)
    ) is None


def test_preflight_or_abort_raises_system_exit_not_a_warning(monkeypatch, cfg):
    monkeypatch.setattr(verifier_module, "CrossEncoderNLI", _BrokenNLI)
    with pytest.raises(SystemExit) as excinfo:
        preflight_or_abort("agentic_full", config_for("agentic_full", cfg))
    message = str(excinfo.value)
    assert "NLI preflight FAILED" in message
    assert "agentic_full" in message
    assert "Refusing to start" in message


def test_preflight_or_abort_returns_the_verdict_when_healthy(monkeypatch, cfg):
    monkeypatch.setattr(verifier_module, "CrossEncoderNLI", _HealthyNLI)
    verdict = preflight_or_abort("agentic_full", config_for("agentic_full", cfg))
    assert verdict is not None and "NLI ok" in verdict


class _Sentinel(Exception):
    """Raised by a patched loader to prove control reached it."""


def test_run_eval_aborts_before_loading_anything_when_nli_is_degraded(monkeypatch, cfg, tmp_path):
    """The abort has to come first: before the eval slice, the indexes, the client.

    A preflight that ran after a fifteen-minute index load would be a check
    nobody waits for. Every loader is patched to a sentinel; the sentinel must
    not be reached.
    """
    monkeypatch.setattr(verifier_module, "CrossEncoderNLI", _BrokenNLI)

    def never(*_a, **_k):
        raise _Sentinel("loaded something before the preflight")

    monkeypatch.setattr(harness, "load_eval_set", never)
    monkeypatch.setattr(harness, "load_pipeline", never)
    spec = RunSpec(config_name="agentic_full", dataset="hotpotqa", limit=1, root=tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        run_eval(spec, cfg=cfg, progress=False)
    assert "NLI preflight FAILED" in str(excinfo.value)


def test_run_eval_skips_the_preflight_for_an_injected_system(monkeypatch, cfg, tmp_path):
    """A test double has no verifier to check, so the harness must not load one.

    With a broken encoder and an injected ``system``, control must get past
    the preflight and reach the slice loader -- which is patched to raise a
    sentinel that is *not* ``SystemExit``.
    """
    monkeypatch.setattr(verifier_module, "CrossEncoderNLI", _BrokenNLI)

    def sentinel(*_a, **_k):
        raise _Sentinel("reached the loader")

    monkeypatch.setattr(harness, "load_eval_set", sentinel)
    spec = RunSpec(config_name="agentic_full", dataset="hotpotqa", limit=1, root=tmp_path)
    with pytest.raises(_Sentinel):
        run_eval(spec, cfg=cfg, system=object(), progress=False)


def test_run_eval_skips_the_preflight_for_a_baseline(monkeypatch, cfg, tmp_path):
    monkeypatch.setattr(verifier_module, "CrossEncoderNLI", _BrokenNLI)

    def sentinel(*_a, **_k):
        raise _Sentinel("reached the loader")

    monkeypatch.setattr(harness, "load_eval_set", sentinel)
    spec = RunSpec(config_name="bm25_only", dataset="hotpotqa", limit=1, root=tmp_path)
    with pytest.raises(_Sentinel):
        run_eval(spec, cfg=cfg, progress=False)


# ---------------------------------------------------------------------------
# Supporting-fact protocols in score_records
# ---------------------------------------------------------------------------

def _evidence(n: int) -> list[dict]:
    return [
        {"evidence_id": f"e{i}", "title": f"T{i}", "sent_id": 0, "text": f"sentence {i}"}
        for i in range(1, n + 1)
    ]


def _gold(qid: str) -> GoldAnswer:
    return GoldAnswer(
        qid=qid, question="?", answer="x", dataset="hotpotqa",
        supporting_facts=(("T1", 0), ("T2", 0)),
    )


def test_resolved_citations_distinguish_namespace_mismatch_from_no_citation():
    cited = {"qid": "a", "citations": ["e1"], "evidence": _evidence(3)}
    mismatched = {"qid": "b", "citations": ["p1"], "evidence": _evidence(3)}
    silent = {"qid": "c", "citations": [], "evidence": _evidence(3)}
    assert resolved_citations(cited) == {"e1"}
    assert resolved_citations(mismatched) == set()
    assert resolved_citations(silent) == set()


def test_pool_facts_ignore_citations_and_predicted_facts_honour_them():
    record = {"qid": "a", "citations": ["e1"], "evidence": _evidence(5)}
    assert predicted_supporting_facts(record) == [("T1", 0)]
    assert pool_supporting_facts(record) == [(f"T{i}", 0) for i in range(1, 6)]


def test_score_records_reports_both_protocols_side_by_side(cfg):
    """Hand-computed. Gold = {T1, T2}.

    Cited record: cites e1 only -> cited set {T1}: P=1, R=0.5, F1=0.667.
    Its whole pool is e1..e5 -> {T1..T5}: P=0.4, R=1.0, F1=0.571.
    Uncited record (p1 against e1..e5) falls back to the pool for BOTH.
    """
    records = [
        {"qid": "a", "final_answer": "x", "citations": ["e1"], "evidence": _evidence(5)},
        {"qid": "b", "final_answer": "x", "citations": ["p1"], "evidence": _evidence(5)},
    ]
    golds = {"a": _gold("a"), "b": _gold("b")}
    scores = score_records(records, golds, cfg=cfg)

    a = scores["a"]
    assert a["sp_cited"] == 1.0
    assert a["sp_precision"] == pytest.approx(1.0)
    assert a["sp_recall"] == pytest.approx(0.5)
    assert a["sp_f1"] == pytest.approx(2 / 3)
    assert a["sp_n_predicted"] == 1.0
    assert a["sp_pool_precision"] == pytest.approx(0.4)
    assert a["sp_pool_recall"] == pytest.approx(1.0)
    assert a["sp_pool_f1"] == pytest.approx(2 * 0.4 / 1.4)
    assert a["sp_n_pool"] == 5.0

    b = scores["b"]
    assert b["sp_cited"] == 0.0
    assert b["sp_n_predicted"] == 5.0
    assert b["sp_precision"] == pytest.approx(b["sp_pool_precision"]) == pytest.approx(0.4)
    assert b["sp_recall"] == pytest.approx(b["sp_pool_recall"]) == pytest.approx(1.0)
    assert b["sp_f1"] == pytest.approx(b["sp_pool_f1"])
