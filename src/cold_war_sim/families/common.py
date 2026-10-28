"""Shared validation and configuration helpers for the mechanism families.

Families are configured from plain mappings (for example, parsed JSON or
YAML) and convert them into frozen parameter objects. Validation is strict:
unknown keys, non-finite numbers and out-of-range probabilities raise when the
game is built, never during play.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ..games.protocol import ImmutableParams


class ConfigError(ValueError):
    """Raised for an invalid family configuration."""


def finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{name} must be finite")
    return result


def nonnegative(name: str, value: Any) -> float:
    result = finite(name, value)
    if result < 0.0:
        raise ConfigError(f"{name} must be nonnegative, got {result}")
    return result


def probability(name: str, value: Any) -> float:
    result = finite(name, value)
    if not 0.0 <= result <= 1.0:
        raise ConfigError(f"{name} must lie in [0, 1], got {result}")
    return result


def positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{name} must be a positive integer, got {value!r}")
    return value


def identifier(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ConfigError(f"{name} must be a non-empty trimmed string, got {value!r}")
    if any(ch in value for ch in ":=\n\t "):
        raise ConfigError(f"{name} may not contain spaces, ':' or '=': {value!r}")
    return value


def reject_unknown(name: str, data: Mapping[str, Any], allowed: Iterable[str]) -> None:
    unknown = set(data) - set(allowed)
    if unknown:
        raise ConfigError(f"{name}: unknown keys {sorted(unknown)}")


def require(name: str, data: Mapping[str, Any], key: str) -> Any:
    if key not in data:
        raise ConfigError(f"{name}: missing required key {key!r}")
    return data[key]


def as_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return value


def as_list(name: str, value: Any) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{name} must be a list")
    return list(value)


@dataclass(frozen=True, slots=True)
class ValueTable(ImmutableParams):
    """Payoff of one role for each outcome label."""

    entries: tuple[tuple[str, float], ...]

    def __getitem__(self, label: str) -> float:
        for key, value in self.entries:
            if key == label:
                return value
        raise KeyError(label)


def value_table(name: str, data: Any, required: Iterable[str]) -> ValueTable:
    mapping = as_mapping(name, data)
    required = tuple(required)
    missing = [label for label in required if label not in mapping]
    if missing:
        raise ConfigError(f"{name}: missing values for {missing}")
    reject_unknown(name, mapping, required)
    return ValueTable(
        tuple((label, finite(f"{name}.{label}", mapping[label])) for label in required)
    )


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


Risk = tuple[float, float, float]  # prospect theory's (alpha, beta, lambda)


def risk_attitude(name: str, data: Any) -> Risk:
    """A role's risk attitude: ``{alpha, beta, lambda}`` (each defaults to 1, risk-neutral)."""

    mapping = as_mapping(name, data)
    reject_unknown(name, mapping, ("alpha", "beta", "lambda"))
    alpha, beta, lam = (
        finite(f"{name}.{key}", mapping.get(key, 1.0)) for key in ("alpha", "beta", "lambda")
    )
    if not (0.0 < alpha <= 1.0 and 0.0 < beta <= 1.0 and lam > 0.0):
        raise ConfigError(f"{name}: alpha and beta must lie in (0, 1] and lambda be positive")
    return (alpha, beta, lam)


def prospect(x: float, risk: Risk | None) -> float:
    """Prospect theory's value of a payoff *x* measured from the status quo (Tversky and
    Kahneman 1992): x^alpha for gains, -lambda (-x)^beta for losses; ``None`` is linear."""

    if risk is None:
        return x
    alpha, beta, lam = risk
    return float(x**alpha) if x >= 0.0 else float(-lam * (-x) ** beta)
