"""Minimal YAML-backed config with attribute access and dotted overrides."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


class Config(dict):
    """A dict that also supports attribute access and nested `get`.

    Nested dicts are wrapped lazily so `cfg.data.n_ids` works.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[name] = value
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: Dict[str, Any] = self
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value


def _coerce(text: str) -> Any:
    """Best-effort scalar coercion for command-line overrides."""
    low = text.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none"}:
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a YAML config and apply `key.subkey=value` overrides."""
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    cfg = Config(raw)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got: {item!r}")
        key, value = item.split("=", 1)
        cfg.set_path(key.strip(), _coerce(value.strip()))
    return cfg


@dataclass
class RegimeSpec:
    """Description of one synthetic evaluation regime (E1/E2 grid cell)."""

    name: str
    n_ids: int = 200
    imgs_per_id: int = 20
    dim: int = 128
    attr_dim: int = 16
    individuation: float = 1.0  # p in gap(p) = g0 - kappa * p
    noise: float = 1.5
    seed: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)
