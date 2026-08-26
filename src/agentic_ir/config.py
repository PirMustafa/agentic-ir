"""Configuration loading.

Single source of truth: ``config/config.yaml``. Values are reachable by
dotted path so the report can cite exact settings, e.g. ``retrieval.rerank.top_n``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_MISSING = object()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class Config:
    """Read-only view over the YAML config with dotted-path access."""

    def __init__(self, data: dict[str, Any], path: Path) -> None:
        self._data = data
        self.path = path

    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is _MISSING:
                    raise KeyError(f"{dotted!r} not found in {self.path}")
                return default
            node = node[part]
        return node

    def __getitem__(self, dotted: str) -> Any:
        return self.get(dotted)

    def resolve_path(self, dotted: str) -> Path:
        """Resolve a configured relative path against the project root."""
        p = Path(self.get(dotted))
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def raw(self) -> dict[str, Any]:
        return self._data


@lru_cache(maxsize=4)
def load_config(path: str | Path | None = None) -> Config:
    """Load and cache the project configuration.

    Override the location with the ``AGENTIC_IR_CONFIG`` environment variable.
    """
    resolved = Path(path or os.environ.get("AGENTIC_IR_CONFIG", DEFAULT_CONFIG_PATH))
    if not resolved.exists():
        raise FileNotFoundError(f"Config not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config(data, resolved)


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
