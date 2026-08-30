"""Shared agent scaffolding (architecture section 3, preamble).

Four things every agent needs and none should re-implement:

* **versioned prompts** -- templates are files under ``prompts/``, rendered with
  :class:`string.Template` and hashed, so a trace can prove which text produced
  a completion (section 9);
* **one guarded LLM entry point** -- :meth:`BaseAgent.call_json` checks the
  budget, spends it, calls the model, records an :class:`LLMCallTrace` and
  returns ``ok=False`` instead of raising. Axiom 2 says no agent may raise, and
  the cheapest way to keep that promise is to give every agent one door;
* **think suppression** -- ``llm.think`` is passed explicitly on every call.
  ``llm.chat()`` defaults it to ``None`` (leave the model alone), so an agent
  that forgets it gets qwen3's reasoning block and, at ``num_predict`` 48, no
  answer at all;
* **text features** -- capitalised-run and content-token extraction, used by the
  Planner's fallback ladder and the Retriever's routing table. They live here
  rather than in either agent because agents may not import one another.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Protocol

from .. import llm as llm_module
from ..config import Config, load_config
from ..llm import LLMFormatError
from ..state import QuestionState, StepRecorder
from ..types import PLACEHOLDER_RE, LLMCallTrace

__all__ = [
    "Agent",
    "BaseAgent",
    "JsonCall",
    "PROMPT_DIR",
    "SYSTEM_PROMPT",
    "collapse_whitespace",
    "content_tokens",
    "entity_runs",
    "load_prompt",
    "prompt_sha1",
    "render_prompt",
    "strip_placeholder",
    "word_tokens",
]

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"

#: ``/no_think`` is belt and braces alongside ``think=False``: older ollama
#: clients silently drop the keyword, and qwen3 then spends the whole
#: completion budget reasoning (measured: 503 chars of think, no answer).
SYSTEM_PROMPT = (
    "You are one component of a local information-retrieval system. "
    "Reply with a single JSON value matching the requested schema. "
    "No prose, no explanation, no markdown fences. /no_think"
)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class Agent(Protocol):
    """Every agent: a name, and a ``run`` that never raises."""

    name: str

    def run(self, state: QuestionState, **kwargs: Any) -> Any: ...


# ---------------------------------------------------------------------------
# Shared text features
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-\.]*")
_CAP_RE = re.compile(r"[A-Z][\w'\-\.]*")
_WS_RE = re.compile(r"\s+")

#: The 25-word list section 2.4 fixes for normalisation. Frozen deliberately:
#: growing it later would silently change every duplicate-plan decision.
STOPWORDS = frozenset(
    ["a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "is", "was", "were", "are", "be", "been", "by", "with", "from", "as", "that", "this", "it", "its", "which", "who", "what", "when", "where", "how"]
)


def word_tokens(text: str) -> tuple[str, ...]:
    """Whitespace/punctuation tokens, preserving case for entity detection."""
    return tuple(_WORD_RE.findall(text or ""))


def content_tokens(text: str) -> tuple[str, ...]:
    """Lowercased tokens with stopwords removed. The ``n_tokens`` feature."""
    return tuple(t.lower() for t in word_tokens(text) if t.lower() not in STOPWORDS)


def entity_runs(text: str) -> tuple[str, ...]:
    """Maximal runs of capitalised tokens, excluding the sentence-initial one.

    Sentence-initial capitals carry no entity signal ("Which magazine..."), and
    counting them would push nearly every question over rule R3's threshold.
    Runs are joined on the original spacing so the surface form stays usable as
    a query on its own.
    """
    runs: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", (text or "").strip()):
        tokens = list(_WORD_RE.finditer(sentence))
        current: list[str] = []
        previous_end = 0
        for position, match in enumerate(tokens):
            token = match.group(0)
            # Punctuation between two capitalised tokens ends the run:
            # "Titanic, James Cameron" is two entities, not one.
            separated = match.start() > previous_end and sentence[previous_end:match.start()].strip()
            capitalised = bool(_CAP_RE.fullmatch(token)) and position > 0
            if capitalised and not (separated and current):
                current.append(token)
            elif capitalised:
                runs.append(" ".join(current))
                current = [token]
            elif current:
                runs.append(" ".join(current))
                current = []
            previous_end = match.end()
        if current:
            runs.append(" ".join(current))
    return tuple(runs)


def collapse_whitespace(text: str) -> str:
    """Squeeze runs of whitespace and tidy space before terminal punctuation.

    Used by placeholder rung 4: deleting ``{{q1.answer}}`` from "When was
    {{q1.answer}} born?" must leave "When was born?", not "When was  born ?".
    """
    cleaned = _WS_RE.sub(" ", text or "").strip()
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return re.sub(r"([(\[])\s+", r"\1", cleaned)


def strip_placeholder(text: str, qid: str, field: str) -> str:
    """Delete one ``{{qN.field}}`` token and collapse the surrounding space."""
    return collapse_whitespace(re.sub(rf"\{{\{{{re.escape(qid)}\.{field}\}}\}}", " ", text))


def placeholders_of(text: str) -> tuple[tuple[str, str], ...]:
    """``((qid, field), ...)`` for a raw string (``SubQuery`` has its own)."""
    return tuple((m.group(1), m.group(2)) for m in PLACEHOLDER_RE.finditer(text or ""))


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_PROMPT_CACHE: dict[str, str] = {}


def load_prompt(prompt_id: str, *, directory: Path | None = None) -> str:
    """Read ``{prompt_id}.txt``, cached. Raises ``OSError`` if absent.

    Prompts are versioned files (``planner.decompose.v1``); changing one means
    bumping the version, so caching by id for the life of the process is safe
    and saves a stat per LLM call.
    """
    root = directory or PROMPT_DIR
    key = f"{root}/{prompt_id}"
    cached = _PROMPT_CACHE.get(key)
    if cached is None:
        with (root / f"{prompt_id}.txt").open("r", encoding="utf-8") as fh:
            cached = fh.read().strip()
        _PROMPT_CACHE[key] = cached
    return cached


def render_prompt(
    prompt_id: str,
    variables: Mapping[str, Any],
    *,
    directory: Path | None = None,
) -> str:
    """Render a template with ``$name`` substitution.

    ``string.Template`` rather than ``str.format``: every prompt here contains a
    literal JSON example, and ``{"strategy": ...}`` makes ``format`` raise.
    ``safe_substitute`` leaves an unknown ``$name`` in place rather than
    exploding mid-run -- a visibly wrong prompt is easier to spot in the trace
    than a lost question.
    """
    template = Template(load_prompt(prompt_id, directory=directory))
    return template.safe_substitute(
        {k: "" if v is None else str(v) for k, v in variables.items()}
    )


def prompt_sha1(text: str) -> str:
    """SHA-1 of the rendered prompt; makes a stale-prompt run detectable."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# One guarded LLM call
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JsonCall:
    """The outcome of one schema-constrained call. Never an exception."""

    ok: bool
    parsed: dict[str, Any] | None = None
    raw: str | None = None
    reason: str | None = None
    retries: int = 0
    latency_s: float = 0.0

    @property
    def failed(self) -> bool:
        return not self.ok


class BaseAgent:
    """Config, client, prompts and the guarded call. Subclasses add ``run``."""

    name: str = "agent"

    def __init__(
        self,
        cfg: Config | None = None,
        *,
        client: Any = None,
        prompt_dir: Path | None = None,
    ) -> None:
        self.cfg: Config = cfg or load_config()
        self.client = client
        self.prompt_dir = prompt_dir

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"

    # -- plumbing ----------------------------------------------------------
    def _llm(self) -> Any:
        """The shared client, resolved lazily.

        Lazily, and through the module rather than a bound import, so that a
        test can monkeypatch ``agentic_ir.llm.get_client`` and so that importing
        an agent never touches Ollama.
        """
        if self.client is None:
            self.client = llm_module.get_client()
        return self.client

    def _think(self) -> bool:
        return bool(self.cfg.get("llm.think", False))

    def _raw_limit(self) -> int:
        return int(self.cfg.get("trace.raw_output_chars", 2000))

    # -- the door ----------------------------------------------------------
    def call_json(
        self,
        state: QuestionState,
        rec: StepRecorder,
        *,
        prompt_id: str,
        variables: Mapping[str, Any],
        schema: Mapping[str, Any] | None,
        purpose: str = "",
        privileged: bool = False,
        repair: bool = True,
        system: str = SYSTEM_PROMPT,
        **options: Any,
    ) -> JsonCall:
        """Render, spend budget, call, trace. Returns a verdict, never raises.

        On unparseable output one explicit repair turn (``repair.json.v1``,
        ladder rung 5) is issued when the budget allows. The client's own
        corrective retries live *inside* the first logical call, so this is the
        only rung that costs a second one -- and it is accounted for as such.
        """
        try:
            prompt = render_prompt(prompt_id, variables, directory=self.prompt_dir)
        except OSError as exc:
            rec.degrade(f"prompt_missing:{prompt_id}")
            return JsonCall(ok=False, reason=f"prompt_missing: {exc}")
        return self.call_json_prompt(
            state, rec, prompt_id=prompt_id, prompt=prompt, schema=schema, purpose=purpose,
            privileged=privileged, repair=repair, system=system, **options,
        )

    def call_json_prompt(
        self,
        state: QuestionState,
        rec: StepRecorder,
        *,
        prompt_id: str,
        prompt: str,
        schema: Mapping[str, Any] | None,
        purpose: str = "",
        privileged: bool = False,
        repair: bool = True,
        system: str = SYSTEM_PROMPT,
        **options: Any,
    ) -> JsonCall:
        """:meth:`call_json` for an already-rendered prompt.

        The seam for a prompt that is assembled from live objects rather than a
        file -- the tool-routing prompt is built from the registry's own
        descriptions, which is what stops the documentation and the executable
        tools from drifting apart.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        outcome = self._attempt(
            state, rec, messages, prompt_id=prompt_id, prompt=prompt, schema=schema,
            purpose=purpose, privileged=privileged, **options,
        )
        if outcome.ok or not repair or outcome.reason == "budget_exhausted":
            return outcome

        try:
            repair_prompt = render_prompt(
                "repair.json.v1",
                {
                    "schema": _schema_text(schema),
                    "bad_output": (outcome.raw or "")[: self._raw_limit()],
                },
                directory=self.prompt_dir,
            )
        except OSError:
            return outcome
        retried = self._attempt(
            state,
            rec,
            [*messages,
             {"role": "assistant", "content": outcome.raw or ""},
             {"role": "user", "content": repair_prompt}],
            prompt_id="repair.json.v1",
            prompt=repair_prompt,
            schema=schema,
            purpose=f"{purpose}:repair",
            privileged=privileged,
            **options,
        )
        return retried if retried.ok else outcome

    def _attempt(
        self,
        state: QuestionState,
        rec: StepRecorder,
        messages: Sequence[Mapping[str, Any]],
        *,
        prompt_id: str,
        prompt: str,
        schema: Mapping[str, Any] | None,
        purpose: str,
        privileged: bool,
        **options: Any,
    ) -> JsonCall:
        """One logical call: budget check, invoke, record, classify."""
        if not state.budget.try_spend_llm(privileged=privileged):
            rec.degrade("budget_exhausted")
            return JsonCall(ok=False, reason="budget_exhausted")

        model = ""
        try:
            client = self._llm()
            model = str(client.model_for(self.name))
            response = client.chat(
                messages,
                agent=self.name,
                schema=dict(schema) if schema is not None else None,
                think=self._think(),
                **options,
            )
        except LLMFormatError as exc:
            self._trace_call(
                rec, prompt_id=prompt_id, prompt=prompt, model=exc.model or model,
                purpose=purpose, parse_ok=False, raw=exc.raw, retries=max(0, exc.attempts - 1),
                error=str(exc), think_chars=len(exc.thinking or ""),
            )
            return JsonCall(ok=False, raw=exc.raw, reason="parse_failure")
        except Exception as exc:  # noqa: BLE001 - axiom 2: agents never raise
            self._trace_call(
                rec, prompt_id=prompt_id, prompt=prompt, model=model, purpose=purpose,
                parse_ok=False, raw=None, error=f"{type(exc).__name__}: {exc}",
            )
            return JsonCall(ok=False, reason=f"llm_error: {type(exc).__name__}")

        parsed = response.parsed if isinstance(response.parsed, dict) else None
        self._trace_call(
            rec, prompt_id=prompt_id, prompt=prompt, model=response.model, purpose=purpose,
            parse_ok=parsed is not None, raw=response.text, retries=response.retries,
            think_chars=response.thinking_chars, latency_s=response.latency_s,
            completion_chars=len(response.text or ""),
        )
        if parsed is None:
            return JsonCall(ok=False, raw=response.text, reason="parse_failure")
        return JsonCall(
            ok=True,
            parsed=parsed,
            raw=response.text,
            retries=response.retries,
            latency_s=response.latency_s,
        )

    def _trace_call(
        self,
        rec: StepRecorder,
        *,
        prompt_id: str,
        prompt: str,
        model: str,
        purpose: str,
        parse_ok: bool,
        raw: str | None,
        retries: int = 0,
        think_chars: int = 0,
        latency_s: float = 0.0,
        completion_chars: int = 0,
        error: str | None = None,
    ) -> None:
        limit = self._raw_limit()
        rec.note_llm(
            LLMCallTrace(
                call_id=f"c{rec.step}_{len(rec.llm_calls)}",
                agent=self.name,
                prompt_id=prompt_id,
                prompt_sha1=prompt_sha1(prompt),
                model=model,
                latency_s=round(latency_s, 4),
                parse_ok=parse_ok,
                purpose=purpose,
                prompt_chars=len(prompt),
                completion_chars=completion_chars,
                think_chars=think_chars,
                retries=retries,
                cache_hit=False,
                truncated=False,
                raw_output=None if raw is None else raw[:limit],
                error=error,
            )
        )


def _schema_text(schema: Mapping[str, Any] | None) -> str:
    """Compact JSON-schema rendering for the repair prompt."""
    if not schema:
        return "a single JSON object"
    return json.dumps(dict(schema), indent=2, ensure_ascii=False)
