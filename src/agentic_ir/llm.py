"""Local LLM client (Ollama).

Every model call in this project runs on-device through Ollama: there are no
API keys and no hosted providers anywhere in the codebase. This module wraps
the ``ollama`` package with the four things the agents need:

* per-agent model selection (``llm.models.<agent>`` falling back to
  ``llm.default_model``),
* JSON-Schema structured output with a bounded, self-correcting retry loop,
* ``<think>`` block handling for qwen3 -- stripped before parsing, kept on the
  response so Chapter 4 can do qualitative reasoning analysis,
* a :class:`CallLedger` recording the per-agent call counts, token totals and
  latencies that feed the agent-specific results table.

The ``ollama`` package is imported lazily inside :class:`OllamaClient` so that
importing this module never fails on a machine where Ollama is not installed
yet; the failure surfaces at call time as :class:`LLMUnavailableError` with an
actionable message instead.
"""

from __future__ import annotations

import inspect
import json
import re
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import Config, load_config

__all__ = [
    "CallLedger",
    "CallRecord",
    "CallStats",
    "LLMError",
    "LLMFormatError",
    "LLMResponse",
    "LLMSettings",
    "LLMTimeoutError",
    "LLMUnavailableError",
    "OllamaClient",
    "QuestionScope",
    "get_client",
    "get_ledger",
    "reset_client",
    "split_thinking",
]

# qwen3 emits reasoning inside <think>...</think>. Ollama >=0.5 can also return
# it out-of-band on ``message.thinking``; both routes are handled.
_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)
_FENCED = re.compile(r"```[a-zA-Z0-9_+-]*[ \t]*\r?\n?(.*?)```", re.DOTALL)

_CORRECTION = (
    "Your previous reply could not be parsed as JSON ({error}). "
    "Reply again with the corrected JSON value only: no prose, no explanation, "
    "no markdown code fences and no <think> block."
)


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------
class LLMError(RuntimeError):
    """Base class for every failure raised by this module."""


class LLMUnavailableError(LLMError):
    """Ollama is unreachable, not installed, or the model is not pulled."""

    def __init__(self, message: str, *, model_missing: bool = False) -> None:
        super().__init__(message)
        self.model_missing = model_missing


class LLMTimeoutError(LLMError):
    """The request exceeded ``llm.request_timeout_s``."""


class LLMFormatError(LLMError):
    """Structured output stayed unparseable after ``llm.max_format_retries``.

    Carries the last raw completion so the caller can log it and apply its own
    deterministic fallback; this module never invents a substitute value.
    """

    def __init__(
        self,
        message: str,
        *,
        raw: str,
        agent: str,
        model: str,
        attempts: int,
        thinking: str | None = None,
    ) -> None:
        super().__init__(message)
        self.raw = raw
        self.agent = agent
        self.model = model
        self.attempts = attempts
        self.thinking = thinking


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LLMSettings:
    """The ``llm:`` block of ``config/config.yaml``, materialised."""

    provider: str
    host: str
    default_model: str
    fallback_model: str | None
    models: dict[str, str]
    options: dict[str, Any]
    max_format_retries: int
    request_timeout_s: float

    @classmethod
    def from_config(cls, cfg: Config) -> "LLMSettings":
        return cls(
            provider=str(cfg.get("llm.provider", "ollama")),
            host=str(cfg.get("llm.host", "http://localhost:11434")),
            default_model=str(cfg.get("llm.default_model")),
            fallback_model=cfg.get("llm.fallback_model", None),
            models=dict(cfg.get("llm.models", {}) or {}),
            options=dict(cfg.get("llm.options", {}) or {}),
            max_format_retries=int(cfg.get("llm.max_format_retries", 2)),
            request_timeout_s=float(cfg.get("llm.request_timeout_s", 180)),
        )

    def model_for(self, agent: str) -> str:
        """Model for ``agent``, falling back to ``llm.default_model``."""
        return str(self.models.get(agent) or self.default_model)

    def merged_options(
        self,
        options: Mapping[str, Any] | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Config options overlaid with per-call options, then keyword overrides."""
        merged = dict(self.options)
        if options:
            merged.update(options)
        if overrides:
            merged.update({k: v for k, v in overrides.items() if v is not None})
        return merged


# --------------------------------------------------------------------------
# Response
# --------------------------------------------------------------------------
@dataclass
class LLMResponse:
    """One completed model call."""

    text: str
    parsed: dict[str, Any] | None
    thinking: str | None
    tool_calls: list[dict[str, Any]] | None
    model: str
    agent: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    retries: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def thinking_chars(self) -> int:
        """Length of the stripped reasoning trace; a cheap thinking-effort proxy."""
        return len(self.thinking or "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "model": self.model,
            "text": self.text,
            "parsed": self.parsed,
            "thinking": self.thinking,
            "tool_calls": self.tool_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_s": round(self.latency_s, 4),
            "retries": self.retries,
        }


# --------------------------------------------------------------------------
# Ledger
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CallRecord:
    """A single ledger row: one logical ``chat()`` call."""

    agent: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    retries: int
    format_retry: bool
    parse_failure: bool
    question_id: str | None
    timestamp: str

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def model_calls(self) -> int:
        """Actual HTTP round-trips to Ollama (a format retry is a second real call)."""
        return self.retries + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "model": self.model,
            "question_id": self.question_id,
            "timestamp": self.timestamp,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_s": round(self.latency_s, 4),
            "retries": self.retries,
            "format_retry": self.format_retry,
            "parse_failure": self.parse_failure,
        }


@dataclass
class CallStats:
    """Aggregates over a set of :class:`CallRecord` rows."""

    n_calls: int = 0
    n_model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_latency_s: float = 0.0
    format_retries: int = 0
    parse_failures: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def mean_latency_s(self) -> float:
        return self.total_latency_s / self.n_calls if self.n_calls else 0.0

    def add(self, record: CallRecord) -> None:
        self.n_calls += 1
        self.n_model_calls += record.model_calls
        self.prompt_tokens += record.prompt_tokens
        self.completion_tokens += record.completion_tokens
        self.total_latency_s += record.latency_s
        self.format_retries += int(record.format_retry)
        self.parse_failures += int(record.parse_failure)

    @classmethod
    def over(cls, records: Sequence[CallRecord]) -> "CallStats":
        stats = cls()
        for record in records:
            stats.add(record)
        return stats

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_calls": self.n_calls,
            "n_model_calls": self.n_model_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_latency_s": round(self.total_latency_s, 4),
            "mean_latency_s": round(self.mean_latency_s, 4),
            "format_retries": self.format_retries,
            "parse_failures": self.parse_failures,
        }


@dataclass
class QuestionScope:
    """Calls made while answering one evaluation question.

    Serialised straight into the per-question trace JSONL by the orchestrator.
    """

    question_id: str
    started_at: float = field(default_factory=time.perf_counter)
    ended_at: float | None = None
    records: list[CallRecord] = field(default_factory=list)

    @property
    def wall_clock_s(self) -> float:
        end = self.ended_at if self.ended_at is not None else time.perf_counter()
        return end - self.started_at

    def stats(self) -> CallStats:
        return CallStats.over(self.records)

    def per_agent(self) -> dict[str, CallStats]:
        by_agent: dict[str, CallStats] = {}
        for record in self.records:
            by_agent.setdefault(record.agent, CallStats()).add(record)
        return by_agent

    def to_dict(self, *, include_calls: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "question_id": self.question_id,
            "wall_clock_s": round(self.wall_clock_s, 4),
            "overall": self.stats().to_dict(),
            "per_agent": {a: s.to_dict() for a, s in self.per_agent().items()},
        }
        if include_calls:
            payload["calls"] = [r.to_dict() for r in self.records]
        return payload


class CallLedger:
    """Thread-safe record of every LLM call made in a run.

    Retrieval may fan sub-queries out across threads, so all mutation happens
    under one lock. The active question is process-wide rather than
    thread-local, precisely so calls made on worker threads spawned inside
    ``with ledger.question(...)`` are still attributed to that question.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[CallRecord] = []
        self._stack: list[QuestionScope] = []
        self._questions: dict[str, QuestionScope] = {}

    # -- recording ---------------------------------------------------------
    def record(self, response: LLMResponse, *, parse_failure: bool = False) -> CallRecord:
        """Record a completed call from its :class:`LLMResponse`."""
        return self.record_call(
            agent=response.agent,
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_s=response.latency_s,
            retries=response.retries,
            parse_failure=parse_failure,
        )

    def record_call(
        self,
        *,
        agent: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_s: float = 0.0,
        retries: int = 0,
        parse_failure: bool = False,
    ) -> CallRecord:
        """Record a call from raw values (also used for calls that ended in error)."""
        with self._lock:
            question_id = self._stack[-1].question_id if self._stack else None
            record = CallRecord(
                agent=agent,
                model=model,
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                latency_s=float(latency_s),
                retries=int(retries),
                format_retry=retries > 0,
                parse_failure=parse_failure,
                question_id=question_id,
                timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
            )
            self._records.append(record)
            for scope in self._stack:
                scope.records.append(record)
            return record

    # -- scoping -----------------------------------------------------------
    @contextmanager
    def question(self, question_id: str) -> Iterator[QuestionScope]:
        """Scope calls to one question: ``with ledger.question(qid) as q: ...``."""
        with self._lock:
            scope = self._questions.get(question_id)
            if scope is None:
                scope = QuestionScope(question_id=question_id)
                self._questions[question_id] = scope
            self._stack.append(scope)
        try:
            yield scope
        finally:
            with self._lock:
                scope.ended_at = time.perf_counter()
                if scope in self._stack:
                    self._stack.remove(scope)

    @property
    def current_question_id(self) -> str | None:
        with self._lock:
            return self._stack[-1].question_id if self._stack else None

    def scope_for(self, question_id: str) -> QuestionScope | None:
        """The recorded scope for ``question_id``, if that question has been seen."""
        with self._lock:
            return self._questions.get(question_id)

    # -- aggregation -------------------------------------------------------
    def records(self, agent: str | None = None) -> list[CallRecord]:
        with self._lock:
            rows = list(self._records)
        return [r for r in rows if agent is None or r.agent == agent]

    def stats(self, agent: str | None = None) -> CallStats:
        """Aggregate over the whole run, or over a single agent."""
        return CallStats.over(self.records(agent))

    def per_agent(self) -> dict[str, CallStats]:
        by_agent: dict[str, CallStats] = {}
        for record in self.records():
            by_agent.setdefault(record.agent, CallStats()).add(record)
        return by_agent

    def to_dict(
        self,
        *,
        include_calls: bool = False,
        include_questions: bool = False,
    ) -> dict[str, Any]:
        """JSON-serialisable view of the ledger."""
        payload: dict[str, Any] = {
            "overall": self.stats().to_dict(),
            "per_agent": {a: s.to_dict() for a, s in self.per_agent().items()},
        }
        if include_questions:
            with self._lock:
                scopes = list(self._questions.values())
            payload["questions"] = [s.to_dict(include_calls=include_calls) for s in scopes]
        if include_calls:
            payload["calls"] = [r.to_dict() for r in self.records()]
        return payload

    def summary(self) -> str:
        """Fixed-width table of the per-agent aggregates, for the CLI."""
        header = (
            f"{'agent':<14}{'calls':>7}{'model':>7}{'tokens':>10}"
            f"{'mean s':>9}{'total s':>10}{'retry':>7}{'fail':>6}"
        )
        lines = [header, "-" * len(header)]
        rows: list[tuple[str, CallStats]] = sorted(self.per_agent().items())
        for agent, stats in rows:
            lines.append(_summary_row(agent, stats))
        lines.append("-" * len(header))
        lines.append(_summary_row("ALL", self.stats()))
        return "\n".join(lines)

    def reset(self) -> None:
        """Forget every recorded call (between evaluation configurations)."""
        with self._lock:
            self._records.clear()
            self._stack.clear()
            self._questions.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


def _summary_row(agent: str, stats: CallStats) -> str:
    return (
        f"{agent:<14}{stats.n_calls:>7}{stats.n_model_calls:>7}"
        f"{stats.total_tokens:>10}{stats.mean_latency_s:>9.2f}"
        f"{stats.total_latency_s:>10.2f}{stats.format_retries:>7}"
        f"{stats.parse_failures:>6}"
    )


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------
def split_thinking(text: str) -> tuple[str, str | None]:
    """Split a completion into ``(visible, thinking)``.

    Handles the three shapes qwen3 produces in practice: a complete
    ``<think>...</think>`` block, a bare closing tag (the opener came from the
    chat template), and an unterminated block (``num_predict`` cut it short).
    """
    if not text:
        return "", None
    parts = [m.group(1) for m in _THINK_BLOCK.finditer(text)]
    visible = _THINK_BLOCK.sub("", text)
    close = _THINK_CLOSE.search(visible)
    if close:
        parts.append(visible[: close.start()])
        visible = visible[close.end() :]
    opener = _THINK_OPEN.search(visible)
    if opener:
        parts.append(visible[opener.end() :])
        visible = visible[: opener.start()]
    thinking = "\n\n".join(p.strip() for p in parts if p.strip()) or None
    return visible.strip(), thinking


def _strip_fences(text: str) -> str:
    """Return the body of the first markdown code fence, or the text unchanged."""
    match = _FENCED.search(text)
    return match.group(1) if match else text


def _first_json_value(text: str) -> str | None:
    """Return the first balanced ``{...}``/``[...]`` span, ignoring string bodies."""
    candidates = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not candidates:
        return None
    start = min(candidates)
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _coerce_json(text: str) -> dict[str, Any]:
    """Parse a completion into a dict, raising ``ValueError`` if impossible.

    A top-level array is wrapped as ``{"items": [...]}`` and a bare scalar as
    ``{"value": ...}`` so callers always receive a mapping.
    """
    candidate = _strip_fences(text).strip()
    if not candidate:
        raise ValueError("the reply was empty")
    for snippet in (candidate, _first_json_value(candidate)):
        if not snippet:
            continue
        try:
            value = json.loads(snippet)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
        if isinstance(value, list):
            return {"items": value}
        return {"value": value}
    raise ValueError("no valid JSON value found in the reply")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dict (ollama 0.3.x) or an object (ollama >=0.4)."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_format(schema: Any) -> Any | None:
    """Normalise ``schema`` into Ollama's ``format`` argument."""
    if schema is None:
        return None
    if isinstance(schema, str):
        return schema  # "json" -- free-form JSON mode
    if isinstance(schema, Mapping):
        return dict(schema)
    to_schema = getattr(schema, "model_json_schema", None)
    if callable(to_schema):  # a pydantic model class
        return to_schema()
    raise TypeError(f"unsupported schema type: {type(schema).__name__}")


def _normalise_tool_calls(message: Any) -> list[dict[str, Any]] | None:
    """Flatten Ollama tool calls into ``{id, name, arguments}`` dicts."""
    calls = _get(message, "tool_calls")
    if not calls:
        return None
    flattened: list[dict[str, Any]] = []
    for call in calls:
        function = _get(call, "function")
        arguments = _get(function, "arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        flattened.append(
            {
                "id": _get(call, "id"),
                "name": _get(function, "name"),
                "arguments": arguments if arguments is not None else {},
            }
        )
    return flattened or None


def _model_names(listing: Any) -> list[str]:
    """Extract model names from ``Client.list()`` across ollama versions."""
    models = _get(listing, "models") or []
    names: list[str] = []
    for entry in models:
        name = _get(entry, "model") or _get(entry, "name")
        if name:
            names.append(str(name))
    return names


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------
class OllamaClient:
    """Thin, retrying wrapper over a local Ollama server."""

    def __init__(self, cfg: Config | None = None, *, ledger: CallLedger | None = None) -> None:
        self.config = cfg or load_config()
        self.settings = LLMSettings.from_config(self.config)
        self.ledger = ledger if ledger is not None else CallLedger()
        self._backend_client: Any = None
        self._backend_lock = threading.Lock()
        self._supports_think: bool | None = None

    # -- plumbing ----------------------------------------------------------
    def _unavailable_hint(self, model: str) -> str:
        return (
            f"Cannot reach Ollama at {self.settings.host}. "
            "Start the server with `ollama serve`, then check the model is "
            f"pulled with `ollama pull {model}`."
        )

    def _backend(self) -> Any:
        """Lazily build the ``ollama.Client``; import errors become LLM errors."""
        if self._backend_client is not None:
            return self._backend_client
        with self._backend_lock:
            if self._backend_client is None:
                try:
                    import ollama  # imported here so this module imports without it
                except ImportError as exc:
                    raise LLMUnavailableError(
                        "The `ollama` Python package is not installed. Run "
                        "`pip install ollama` (see requirements.txt), install the "
                        "Ollama server, and start it with `ollama serve`."
                    ) from exc
                try:
                    self._backend_client = ollama.Client(
                        host=self.settings.host,
                        timeout=self.settings.request_timeout_s,
                    )
                except Exception as exc:  # malformed host, bad kwargs, ...
                    raise LLMUnavailableError(
                        f"Could not create an Ollama client for {self.settings.host}: {exc}"
                    ) from exc
        return self._backend_client

    def _accepts_think(self, backend: Any) -> bool:
        """Whether the installed ollama version exposes ``chat(think=...)``."""
        if self._supports_think is None:
            try:
                self._supports_think = "think" in inspect.signature(backend.chat).parameters
            except (TypeError, ValueError):
                self._supports_think = False
        return self._supports_think

    def _classify(self, exc: Exception, model: str) -> LLMError:
        """Map a transport or server exception onto this module's error types."""
        name = type(exc).__name__
        text = str(exc)
        lowered = text.lower()
        status = _get(exc, "status_code")
        if "timeout" in name.lower() or "timed out" in lowered:
            return LLMTimeoutError(
                "Ollama did not respond within llm.request_timeout_s="
                f"{self.settings.request_timeout_s}s for model {model!r}. "
                "Raise the timeout or lower llm.options.num_predict."
            )
        if status == 404 or "not found" in lowered or "try pulling" in lowered:
            return LLMUnavailableError(
                f"Model {model!r} is not available on {self.settings.host}. "
                f"Pull it with `ollama pull {model}`.",
                model_missing=True,
            )
        if (
            isinstance(exc, (ConnectionError, OSError))
            or "connect" in name.lower()
            or "connection" in lowered
            or "refused" in lowered
            or "max retries" in lowered
        ):
            return LLMUnavailableError(self._unavailable_hint(model))
        return LLMError(f"Ollama call failed for model {model!r}: {name}: {text}")

    def _invoke(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Any] | None,
        fmt: Any,
        options: Mapping[str, Any],
        think: bool | None,
        keep_alive: Any,
    ) -> Any:
        """One raw round-trip to ``/api/chat``."""
        backend = self._backend()
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [dict(m) for m in messages],
            "stream": False,
        }
        if tools:
            kwargs["tools"] = list(tools)
        if fmt is not None:
            kwargs["format"] = fmt
        if options:
            kwargs["options"] = dict(options)
        if keep_alive is not None:
            kwargs["keep_alive"] = keep_alive
        if think is not None and self._accepts_think(backend):
            kwargs["think"] = think
        try:
            return backend.chat(**kwargs)
        except LLMError:
            raise
        except Exception as exc:
            raise self._classify(exc, model) from exc

    def _invoke_with_fallback(self, *, model: str, **kwargs: Any) -> tuple[Any, str]:
        """Call ``model``; if it is not pulled, retry once on ``llm.fallback_model``."""
        try:
            return self._invoke(model=model, **kwargs), model
        except LLMUnavailableError as exc:
            fallback = self.settings.fallback_model
            if not exc.model_missing or not fallback or fallback == model:
                raise
            return self._invoke(model=fallback, **kwargs), fallback

    # -- public API --------------------------------------------------------
    def model_for(self, agent: str) -> str:
        """Model configured for ``agent`` (``llm.models.<agent>`` or the default)."""
        return self.settings.model_for(agent)

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        agent: str,
        tools: Sequence[Any] | None = None,
        schema: Any | None = None,
        model: str | None = None,
        options: Mapping[str, Any] | None = None,
        think: bool | None = None,
        keep_alive: Any = None,
        **overrides: Any,
    ) -> LLMResponse:
        """Run one chat completion and record it in the ledger.

        ``schema`` may be a JSON Schema dict (passed as Ollama's ``format``),
        the string ``"json"``, or a pydantic model class. When given, the reply
        is parsed into :attr:`LLMResponse.parsed`; if parsing fails the call is
        retried up to ``llm.max_format_retries`` times with a corrective user
        turn appended, after which :class:`LLMFormatError` is raised carrying
        the last raw text. Remaining keyword arguments are merged into Ollama's
        ``options`` (``temperature``, ``num_predict``, ``seed``, ...).
        """
        target_model = model or self.model_for(agent)
        fmt = _as_format(schema)
        call_options = self.settings.merged_options(options, overrides)
        conversation: list[dict[str, Any]] = [dict(m) for m in messages]
        max_attempts = 1 + (self.settings.max_format_retries if schema is not None else 0)

        started = time.perf_counter()
        prompt_tokens = 0
        completion_tokens = 0
        thinking_parts: list[str] = []
        visible = ""
        raw = ""
        tool_calls: list[dict[str, Any]] | None = None
        parsed: dict[str, Any] | None = None
        used_model = target_model
        parse_error: ValueError | None = None
        attempt = 0

        for attempt in range(max_attempts):
            payload, used_model = self._invoke_with_fallback(
                model=used_model,
                messages=conversation,
                tools=tools,
                fmt=fmt,
                options=call_options,
                think=think,
                keep_alive=keep_alive,
            )
            message = _get(payload, "message")
            raw = str(_get(message, "content") or "")
            visible, inline_thinking = split_thinking(raw)
            out_of_band = _get(message, "thinking")  # ollama >=0.5 thinking models
            for chunk in (out_of_band, inline_thinking):
                if chunk:
                    thinking_parts.append(str(chunk).strip())
            prompt_tokens += int(_get(payload, "prompt_eval_count") or 0)
            completion_tokens += int(_get(payload, "eval_count") or 0)
            tool_calls = _normalise_tool_calls(message) or tool_calls

            if schema is None:
                parse_error = None
                break
            try:
                parsed = _coerce_json(visible or raw)
                parse_error = None
                break
            except ValueError as exc:
                parse_error = exc
                parsed = None
                if attempt < max_attempts - 1:
                    conversation.append({"role": "assistant", "content": visible or raw})
                    conversation.append(
                        {"role": "user", "content": _CORRECTION.format(error=exc)}
                    )

        latency_s = time.perf_counter() - started
        thinking = "\n\n".join(dict.fromkeys(p for p in thinking_parts if p)) or None
        response = LLMResponse(
            text=visible,
            parsed=parsed,
            thinking=thinking,
            tool_calls=tool_calls,
            model=used_model,
            agent=agent,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=latency_s,
            retries=attempt,
        )

        if parse_error is not None:
            self.ledger.record(response, parse_failure=True)
            raise LLMFormatError(
                f"{agent}: model {used_model!r} returned unparseable structured output "
                f"after {attempt + 1} attempt(s) ({parse_error}).",
                raw=visible or raw,
                agent=agent,
                model=used_model,
                attempts=attempt + 1,
                thinking=thinking,
            )

        self.ledger.record(response)
        return response

    def complete(
        self,
        prompt: str,
        *,
        agent: str,
        system: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Single-turn convenience wrapper over :meth:`chat`."""
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, agent=agent, **kwargs)

    def ping(self) -> dict[str, Any]:
        """Check the server is up and report which configured models are pulled."""
        backend = self._backend()
        try:
            listing = backend.list()
        except LLMError:
            raise
        except Exception as exc:
            raise self._classify(exc, self.settings.default_model) from exc
        available = _model_names(listing)
        wanted: dict[str, str] = {"default": self.settings.default_model}
        wanted.update({agent: self.model_for(agent) for agent in self.settings.models})
        if self.settings.fallback_model:
            wanted["fallback"] = str(self.settings.fallback_model)
        missing = sorted({m for m in wanted.values() if m not in available})
        return {
            "host": self.settings.host,
            "available_models": available,
            "configured_models": wanted,
            "missing_models": missing,
        }

    def is_available(self) -> bool:
        """``True`` if Ollama answers; never raises."""
        try:
            self.ping()
        except LLMError:
            return False
        return True


# --------------------------------------------------------------------------
# Process-wide singleton
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _cached_client(config_path: str | None) -> OllamaClient:
    """Cached constructor behind :func:`get_client`."""
    return OllamaClient(load_config(config_path))


def get_client(config_path: str | Path | None = None) -> OllamaClient:
    """Return the process-wide client, so every agent shares one ledger.

    The path is normalised to a string first, so ``get_client()`` and
    ``get_client(path)`` for the default path hit the same cache entry.
    """
    return _cached_client(str(config_path) if config_path is not None else None)


def get_ledger(config_path: str | Path | None = None) -> CallLedger:
    """Shortcut to the shared client's ledger."""
    return get_client(config_path).ledger


def reset_client() -> None:
    """Drop the cached client (tests, or after editing the config)."""
    _cached_client.cache_clear()


# --------------------------------------------------------------------------
# Smoke test: `python -m agentic_ir.llm`
# --------------------------------------------------------------------------
_SMOKE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "capital": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["capital", "confidence"],
}


def _smoke_test() -> int:
    """Ping the configured model and print the ledger summary."""
    client = get_client()
    settings = client.settings
    print("Ollama smoke test")
    print(f"  host           : {settings.host}")
    print(f"  default model  : {settings.default_model}")
    print(f"  per-agent      : {settings.models}")
    print(f"  timeout        : {settings.request_timeout_s}s")
    print(f"  format retries : {settings.max_format_retries}")

    try:
        info = client.ping()
    except LLMError as exc:
        print(f"\nFAILED: {exc}")
        return 1
    print(f"\n  models pulled  : {', '.join(info['available_models']) or '(none)'}")
    if info["missing_models"]:
        print(f"  MISSING        : {', '.join(info['missing_models'])}")
        print("  -> pull them before running the evaluation.")

    try:
        with client.ledger.question("smoke-test") as scope:
            plain = client.chat(
                [{"role": "user", "content": "Reply with exactly one word: pong."}],
                agent="planner",
            )
            print(f"\n  plain reply    : {plain.text[:120]!r}")
            print(f"  thinking chars : {plain.thinking_chars}")

            structured = client.chat(
                [
                    {
                        "role": "user",
                        "content": "What is the capital of Italy? Answer as JSON "
                        "with keys `capital` and `confidence` (0-1).",
                    }
                ],
                agent="verifier",
                schema=_SMOKE_SCHEMA,
            )
            print(f"  parsed JSON    : {structured.parsed}")
            print(f"  retries        : {structured.retries}")
    except LLMFormatError as exc:
        print(f"\nFAILED (structured output): {exc}")
        print(f"  last raw text  : {exc.raw[:300]!r}")
        return 1
    except LLMError as exc:
        print(f"\nFAILED: {exc}")
        return 1

    print("\nLedger summary")
    print(client.ledger.summary())
    print("\nPer-question trace record")
    print(json.dumps(scope.to_dict(include_calls=False), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke_test())
