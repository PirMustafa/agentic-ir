"""Why a model call ended, and the trace field that records it.

``LLMCallTrace.truncated`` was declared with the schema and assigned by
nothing. Its value was ``False`` on every call ever traced, including the ten
questions the improvement loop lost to a synthesis call that spent its entire
``num_predict`` budget inside the thinking channel and never opened the content
channel. The loop concluded twice, in writing, that those failures were *not*
the completion cap -- and both times the field that would have settled it was
sitting in the record reading ``False`` because nobody set it.

Ollama says exactly what happened in ``done_reason``: ``"stop"`` when the model
ended its own turn, ``"length"`` when the budget ended it. These tests pin that
it is read, that it survives into the trace on both the success and the failure
path, and -- the part that bit during implementation -- that reading it can
never cost a caller its answer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agentic_ir.agents.base import BaseAgent
from agentic_ir.llm import LLMFormatError, LLMResponse, OllamaClient
from agentic_ir.state import QuestionState


SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}}
PROMPT_ID = "probe.answer.v1"


@pytest.fixture
def prompts(tmp_path: Path) -> Path:
    (tmp_path / f"{PROMPT_ID}.txt").write_text("Answer $question", encoding="utf-8")
    return tmp_path


def _payload(
    *,
    content: str,
    done_reason: str,
    thinking: str | None = None,
    eval_count: int = 12,
) -> dict[str, Any]:
    return {
        "message": {"content": content, "thinking": thinking},
        "prompt_eval_count": 100,
        "eval_count": eval_count,
        "done_reason": done_reason,
    }


class ScriptedClient(OllamaClient):
    """A real client with only the network hop replaced.

    Subclassing rather than duck-typing is deliberate: the logic under test is
    ``chat()``'s attempt loop -- the thinking split, the token sums, the repair
    rung, the ``done_reason`` bookkeeping -- and a stub that reimplemented any
    of it would be testing the stub.
    """

    def __init__(self, *payloads: dict[str, Any]) -> None:
        super().__init__()
        self.payloads = list(payloads)
        self.seen: list[dict[str, Any]] = []

    def _invoke_with_fallback(self, *, model: str, **kwargs: Any):
        self.seen.append({"model": model, **kwargs})
        payload = self.payloads.pop(0) if self.payloads else _payload(
            content='{"answer": "ok"}', done_reason="stop"
        )
        return payload, model


# ---------------------------------------------------------------------------
# The client reads it
# ---------------------------------------------------------------------------

def test_a_spent_budget_is_reported_as_length_not_as_silence():
    """The exact failure the loop lost ten questions to, in one call.

    Empty content channel, a full thinking channel, ``done_reason: length``.
    Before this, the only trace evidence was ``completion_chars=0`` beside a
    large ``think_chars``, which is equally consistent with a model that
    finished and said nothing.
    """
    spent = _payload(content="", done_reason="length", thinking="x" * 22_000, eval_count=5120)
    client = ScriptedClient(spent, spent, spent)
    with pytest.raises(LLMFormatError) as excinfo:
        client.chat([{"role": "user", "content": "q"}], agent="synthesizer", schema=SCHEMA)

    exc = excinfo.value
    assert exc.hit_length_cap is True
    assert exc.done_reason == "length"
    # Summed across all three attempts, which is what makes it a cost figure.
    assert exc.completion_tokens == 3 * 5120
    assert exc.raw == ""
    assert exc.thinking and len(exc.thinking) > 1000


def test_a_model_that_ends_its_own_turn_is_not_marked_truncated():
    client = ScriptedClient(_payload(content='{"answer": "Naples"}', done_reason="stop"))
    response = client.chat([{"role": "user", "content": "q"}], agent="synthesizer", schema=SCHEMA)

    assert response.done_reason == "stop"
    assert response.hit_length_cap is False
    assert response.parsed == {"answer": "Naples"}


def test_a_cap_on_an_earlier_attempt_survives_a_later_success():
    """``hit_length_cap`` is "any attempt", and the distinction is the point.

    A call that burns 5,120 tokens, fails to parse, then succeeds on the repair
    rung has cost a question roughly 145 seconds. A last-attempt-only flag would
    record that as a clean call and the cost would stay invisible -- which is
    how 26 HotpotQA and 49 2Wiki questions came to be described as a "silent
    retry" rather than as a measured latency defect.
    """
    client = ScriptedClient(
        _payload(content="", done_reason="length", thinking="t" * 9_000, eval_count=5120),
        _payload(content='{"answer": "Naples"}', done_reason="stop", eval_count=9),
    )
    response = client.chat([{"role": "user", "content": "q"}], agent="synthesizer", schema=SCHEMA)

    assert response.parsed == {"answer": "Naples"}
    assert response.retries == 1
    assert response.done_reason == "stop", "the last attempt ended cleanly"
    assert response.hit_length_cap is True, "but an earlier one did not"
    assert response.completion_tokens == 5120 + 9


# ---------------------------------------------------------------------------
# The trace records it
# ---------------------------------------------------------------------------

def _one_trace(client: Any, prompts: Path):
    agent = BaseAgent(client=client, prompt_dir=prompts)
    agent.name = "synthesizer"
    state = QuestionState(qid="q1", question="who?", dataset="hotpotqa", config_name="agentic_v2")
    with state.step(agent.name) as rec:
        agent.call_json(
            state,
            rec,
            prompt_id=PROMPT_ID,
            variables={"question": state.question},
            schema=SCHEMA,
            purpose="answer",
        )
    return state.traces[-1].llm_calls[-1]


def test_the_failure_path_writes_truncated_into_the_trace(prompts: Path):
    spent = _payload(content="", done_reason="length", thinking="x" * 22_000, eval_count=5120)
    call = _one_trace(ScriptedClient(spent, spent, spent), prompts)

    assert call.parse_ok is False
    assert call.truncated is True
    assert call.completion_chars == 0, "nothing reached the content channel"
    assert call.think_chars > 1000, "and everything reached the thinking one"
    assert call.completion_tokens == 3 * 5120


def test_the_success_path_writes_it_too(prompts: Path):
    client = ScriptedClient(_payload(content='{"answer": "Naples"}', done_reason="stop"))
    call = _one_trace(client, prompts)

    assert call.parse_ok is True
    assert call.truncated is False
    assert call.completion_tokens == 12


# ---------------------------------------------------------------------------
# ... and reading it can never cost a caller its answer
# ---------------------------------------------------------------------------

class _Antique:
    """A response object from before these fields existed."""

    def __init__(self) -> None:
        self.parsed = {"answer": "Naples"}
        self.text = '{"answer": "Naples"}'
        self.thinking = None
        self.thinking_chars = 0
        self.model = "stub"
        self.agent = "synthesizer"
        self.retries = 0
        self.latency_s = 0.01


class _AntiqueClient:
    def model_for(self, agent: str) -> str:
        return "stub"

    def chat(self, messages: Any, *, agent: str, **kwargs: Any) -> _Antique:
        return _Antique()


def test_a_response_without_the_new_fields_still_answers(prompts: Path):
    """The regression this file exists for as much as for ``done_reason``.

    Implemented as plain attribute access, the trace call raised
    ``AttributeError`` on every duck-typed response in the suite. Axiom 2 caught
    it -- agents never raise -- so nothing crashed; the agent just returned
    ``ok=False`` and the Planner quietly produced ``fallback_rule`` plans. A
    telemetry field turned working agents into degraded ones, and the only
    symptom was a worse answer. Telemetry reads defensively for that reason.
    """
    call = _one_trace(_AntiqueClient(), prompts)

    assert call.parse_ok is True, "the answer survives a response that lacks the fields"
    assert call.truncated is False
    assert call.completion_tokens == 0


def test_llmresponse_defaults_keep_older_construction_working():
    response = LLMResponse(
        text="x", parsed=None, thinking=None, tool_calls=None, model="m",
        agent="synthesizer", prompt_tokens=1, completion_tokens=2, latency_s=0.1, retries=0,
    )
    assert response.done_reason is None
    assert response.hit_length_cap is False
    assert response.to_dict()["done_reason"] is None
