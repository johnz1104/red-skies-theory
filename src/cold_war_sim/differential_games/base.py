"""Shared validation and status types for supported differential games."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import NoReturn


class BoundarySystemStatus(StrEnum):
    """Classification of the linear two-point boundary system."""

    UNIQUE_WELL_CONDITIONED = "UNIQUE_WELL_CONDITIONED"
    INCONSISTENT = "INCONSISTENT_BOUNDARY_SYSTEM"
    NONUNIQUE = "NONUNIQUE_BOUNDARY_SYSTEM_UNSUPPORTED"
    ILL_CONDITIONED = "ILL_CONDITIONED_BOUNDARY_SYSTEM"


@dataclass(frozen=True, slots=True)
class BoundarySystemDiagnostics:
    """Numerical diagnostics for the initial-costate linear system."""

    status: BoundarySystemStatus
    rank: int
    augmented_rank: int
    dimension: int
    condition_number: float | None
    residual_norm: float | None


class OpenLoopSolveError(ValueError):
    """A supported model has no uniquely and reliably computable solution."""

    def __init__(self, message: str, diagnostics: BoundarySystemDiagnostics) -> None:
        super().__init__(message)
        self.status = diagnostics.status
        self.diagnostics = diagnostics


def finite_float(value: object, *, name: str) -> float:
    """Validate a scalar real input without accepting booleans."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def reject_unknown_keys(
    values: object,
    *,
    allowed: set[str],
    required: set[str] | frozenset[str] = frozenset(),
    name: str,
) -> dict[str, object]:
    """Return a plain string-keyed mapping after strict key validation."""

    if not isinstance(values, dict):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in values):
        raise TypeError(f"{name} keys must be strings")
    data = dict(values)
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} fields: {unknown}")
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"missing {name} fields: {missing}")
    return data


def unsupported(message: str) -> NoReturn:
    """Raise a consistent error for cases outside the deliberately narrow class."""

    raise ValueError(f"unsupported differential-game model: {message}")
