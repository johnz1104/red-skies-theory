"""Shared validation and configuration helpers for the mechanism families.

Families are configured from plain mappings (for example, parsed JSON or
YAML) and convert them into parameter objects. Validation is strict:
unknown keys, non-finite numbers and out-of-range probabilities raise when the
game is built, never during play.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from types import TracebackType
from typing import Any


class ConfigError(ValueError):
    """Raised for an invalid family configuration."""


# ------------------------------------------------------------------- single values
def finite(name: str, value: Any) -> float:
    """*value* as a float; it must be a finite number (booleans are not numbers)."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{name} must be finite")
    return result


def nonnegative(name: str, value: Any) -> float:
    """*value* as a float that is zero or more."""

    result = finite(name, value)
    if result < 0.0:
        raise ConfigError(f"{name} must be nonnegative, got {result}")
    return result


def probability(name: str, value: Any) -> float:
    """*value* as a float in [0, 1]."""

    result = finite(name, value)
    if not 0.0 <= result <= 1.0:
        raise ConfigError(f"{name} must lie in [0, 1], got {result}")
    return result


def positive_int(name: str, value: Any) -> int:
    """*value*, which must be an integer of at least 1."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{name} must be a positive integer, got {value!r}")
    return value


def identifier(name: str, value: Any) -> str:
    """*value*, which must be a name usable in log tokens: no spaces, ``:`` or ``=``."""

    if not isinstance(value, str) or not value or value != value.strip():
        raise ConfigError(f"{name} must be a non-empty trimmed string, got {value!r}")
    if any(ch in value for ch in ":=\n\t "):
        raise ConfigError(f"{name} may not contain spaces, ':' or '=': {value!r}")
    return value


def clamp(value: float, low: float, high: float) -> float:
    """*value* held within [low, high]."""

    return min(high, max(low, value))


def logistic(x: float) -> float:
    """1 / (1 + exp(-x)), computed without overflow."""

    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


# ----------------------------------------------------------------------- mappings
def reject_unknown(name: str, data: Mapping[str, Any], allowed: Iterable[str]) -> None:
    """Raise if *data* has a key outside *allowed*."""

    unknown = set(data) - set(allowed)
    if unknown:
        raise ConfigError(f"{name}: unknown keys {sorted(unknown)}")


def require(name: str, data: Mapping[str, Any], key: str) -> Any:
    """The value of *key*, which must be present."""

    if key not in data:
        raise ConfigError(f"{name}: missing required key {key!r}")
    return data[key]


def as_mapping(name: str, value: Any) -> Mapping[str, Any]:
    """*value*, which must be a mapping."""

    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return value


def as_list(name: str, value: Any) -> list[Any]:
    """*value* as a list; it must be a list or a tuple."""

    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{name} must be a list")
    return list(value)


_REQUIRED: Any = object()  # the default of a key that has none


class Section:
    """One mapping of a config while it is being read.

    It carries the mapping's dotted name for error messages and remembers which keys
    were asked for. Used in a ``with`` block, it rejects on the way out any key nobody
    asked for, so a reader never lists its keys twice::

        with Section("crisis.roles.USA", data) as s:
            cost = s.nonnegative("escalation_cost", 0.0)   # optional, default 0
            side = s.choice("side", ("a", "b", "neutral"))  # required

    A reader method without a default requires its key.
    """

    def __init__(self, name: str, data: Any) -> None:
        self.name = name
        self.data = as_mapping(name, data)
        self._asked: set[str] = set()

    def __enter__(self) -> Section:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        if error is None:
            self.finish()

    def finish(self) -> None:
        """Reject every key that was never asked for."""

        reject_unknown(self.name, self.data, self._asked)

    def at(self, key: str) -> str:
        """The dotted name of *key* inside this mapping."""

        return f"{self.name}.{key}"

    def has(self, key: str) -> bool:
        """Whether *key* is present (asking counts as reading it)."""

        self._asked.add(key)
        return key in self.data

    def get(self, key: str, default: Any = _REQUIRED) -> Any:
        """The unchecked value of *key*."""

        self._asked.add(key)
        if default is _REQUIRED:
            return require(self.name, self.data, key)
        return self.data.get(key, default)

    def number(self, key: str, default: Any = _REQUIRED) -> float:
        """A finite number."""

        return finite(self.at(key), self.get(key, default))

    def nonnegative(self, key: str, default: Any = _REQUIRED) -> float:
        """A number that is zero or more."""

        return nonnegative(self.at(key), self.get(key, default))

    def probability(self, key: str, default: Any = _REQUIRED) -> float:
        """A number in [0, 1]."""

        return probability(self.at(key), self.get(key, default))

    def positive_int(self, key: str, default: Any = _REQUIRED) -> int:
        """An integer of at least 1."""

        return positive_int(self.at(key), self.get(key, default))

    def flag(self, key: str, default: bool = False) -> bool:
        """A switch: ``true`` or ``false``, never another value read by its truth."""

        value = self.get(key, default)
        if not isinstance(value, bool):
            raise ConfigError(f"{self.at(key)} must be true or false, got {value!r}")
        return value

    def choice(self, key: str, allowed: tuple[str, ...], default: Any = _REQUIRED) -> str:
        """One of *allowed*."""

        value = self.get(key, default)
        if value not in allowed:
            raise ConfigError(f"{self.at(key)} must be one of {allowed}")
        return str(value)

    def role(self, key: str, role_ids: Iterable[str]) -> str:
        """A required key naming one of the game's roles."""

        value = self.get(key)
        if value not in role_ids:
            raise ConfigError(f"{self.at(key)} refers to unknown role {value!r}")
        return str(value)

    def sequence(self, key: str, default: Any = _REQUIRED) -> list[Any]:
        """A list or tuple, as a list."""

        return as_list(self.at(key), self.get(key, default))

    def section(self, key: str, default: Any = _REQUIRED) -> Section:
        """The mapping under *key*, to be read in its own ``with`` block."""

        return Section(self.at(key), self.get(key, default))


def distribution(name: str, data: Any, levels: tuple[str, ...]) -> dict[str, float]:
    """A probability for each of *levels* (0 when left out), in their order, summing to 1."""

    mapping = as_mapping(name, data)
    reject_unknown(name, mapping, levels)
    weights = {level: probability(f"{name}.{level}", mapping.get(level, 0.0)) for level in levels}
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ConfigError(f"{name} must sum to 1")
    return weights


# ------------------------------------------------------------------------ payoffs
ValueTable = dict[str, float]  # one role's payoff for each outcome label


def value_table(name: str, data: Any, required: Iterable[str]) -> ValueTable:
    """A payoff for exactly the *required* outcome labels, in their order."""

    mapping = as_mapping(name, data)
    required = tuple(required)
    missing = [label for label in required if label not in mapping]
    if missing:
        raise ConfigError(f"{name}: missing values for {missing}")
    reject_unknown(name, mapping, required)
    return {label: finite(f"{name}.{label}", mapping[label]) for label in required}


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
    """Value payoff x relative to the status quo using prospect theory.

    Gains are worth x^alpha; losses are worth -lambda * (-x)^beta
    (Tversky and Kahneman 1992). With risk=None, return x unchanged.
    """

    if risk is None:
        return x
    alpha, beta, lam = risk
    return float(x**alpha) if x >= 0.0 else float(-lam * (-x) ** beta)
