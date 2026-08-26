"""Configuration loading.

Single source of truth: ``config/config.yaml``. Values are reachable by
dotted path so the report can cite exact settings, e.g. ``retrieval.rerank.top_n``.

The loaded tree is **deep-frozen**. Chapter 4 cites configuration values as the
settings that produced its numbers, so a caller doing the natural thing --
``opts = cfg.get("llm.options"); opts["temperature"] = 0.7`` -- must not be able
to silently reconfigure every agent through the shared cache and leave the
report describing a run that never happened. Use :meth:`Config.with_overrides`
to derive a modified configuration instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

_MISSING = object()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def _freeze(value: Any) -> Any:
    """Recursively convert mappings to read-only views and lists to tuples."""
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    """Inverse of :func:`_freeze`, for building a modified copy."""
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


class Config:
    """Read-only view over the YAML config with dotted-path access."""

    def __init__(self, data: Mapping[str, Any], path: Path) -> None:
        self._data = _freeze(data)
        self.path = path

    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        """Look up a dotted path.

        Raises ``KeyError`` when the path is absent and no default is given.
        A stored ``None`` is returned as ``None`` rather than treated as absent.
        """
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                if default is _MISSING:
                    raise KeyError(f"{dotted!r} not found in {self.path}")
                return default
            node = node[part]
        return node

    def __getitem__(self, dotted: str) -> Any:
        return self.get(dotted)

    def __contains__(self, dotted: str) -> bool:
        return self.get(dotted, _MISSING) is not _MISSING

    def resolve_path(self, dotted: str) -> Path:
        """Resolve a configured relative path against the project root."""
        p = Path(self.get(dotted))
        return p if p.is_absolute() else PROJECT_ROOT / p

    def with_overrides(self, overrides: Mapping[str, Any]) -> "Config":
        """Return a NEW Config with dotted-path ``overrides`` applied.

        The receiver is unchanged, so ablation runs and tests can vary settings
        without leaking the change into the process-wide cached instance.
        """
        data = _thaw(self._data)
        for dotted, value in overrides.items():
            parts = dotted.split(".")
            node = data
            for part in parts[:-1]:
                nxt = node.get(part)
                if not isinstance(nxt, dict):
                    nxt = {}
                    node[part] = nxt
                node = nxt
            node[parts[-1]] = value
        return Config(data, self.path)

    @property
    def raw(self) -> Mapping[str, Any]:
        """The frozen configuration tree. Safe to serialise into a run's meta.json."""
        return self._data

    def as_dict(self) -> dict[str, Any]:
        """A mutable deep copy, for snapshotting into trace metadata."""
        return _thaw(self._data)

    def __repr__(self) -> str:
        return f"Config(path={self.path!s})"


def _resolve_config_path(path: str | Path | None) -> Path:
    """Resolve the config location. Read OUTSIDE the cache.

    ``AGENTIC_IR_CONFIG`` is consulted here rather than inside the cached
    loader, so that changing it actually takes effect and so that the four
    spellings of the same file (``None``, ``str``, ``Path``, absolute) collapse
    to a single cache entry instead of four.
    """
    return Path(path or os.environ.get("AGENTIC_IR_CONFIG") or DEFAULT_CONFIG_PATH).resolve()


@lru_cache(maxsize=8)
def _load_cached(resolved: Path) -> Config:
    if not resolved.exists():
        raise FileNotFoundError(f"Config not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return Config(data, resolved)


def load_config(path: str | Path | None = None) -> Config:
    """Load and cache the project configuration.

    Override the location with the ``AGENTIC_IR_CONFIG`` environment variable.
    """
    return _load_cached(_resolve_config_path(path))


def reset_config_cache() -> None:
    """Drop the cached configuration. For tests."""
    _load_cached.cache_clear()


@dataclass(frozen=True)
class Paths:
    """Materialised project paths, created on access."""

    raw: Path
    processed: Path
    indexes: Path
    results: Path

    @classmethod
    def from_config(cls, cfg: Config) -> "Paths":
        paths = cls(
            raw=cfg.resolve_path("paths.raw"),
            processed=cfg.resolve_path("paths.processed"),
            indexes=cfg.resolve_path("paths.indexes"),
            results=cfg.resolve_path("paths.results"),
        )
        for p in (paths.raw, paths.processed, paths.indexes, paths.results):
            p.mkdir(parents=True, exist_ok=True)
        return paths
