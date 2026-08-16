"""Typed deterministic solver and diagnostic records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .probability import ensure_finite
from .types import (
    SerializableMixin,
    frozen_mapping,
    to_serializable,
    validate_stable_id,
)


def _strings(values: tuple[str, ...] | list[str], *, name: str) -> tuple[str, ...]:
    result = tuple(values)
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{name} must contain only non-empty strings")
    return result


@dataclass(frozen=True)
class SolverResult(SerializableMixin):
    """Common result envelope for exact solvers and labeled heuristics."""

    solver_name: str
    solver_version: str
    equilibrium_concept: str
    found: bool
    solutions: tuple[Any, ...] = ()
    status: str = "SOLUTIONS_FOUND"
    multiple_solutions: bool | None = None
    exactness_status: str = "EXACT"
    convergence_status: str | None = None
    best_response_gap: float | None = None
    off_path_belief_convention: str | None = None
    runtime_seconds: float = 0.0
    warnings: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    seed: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value, name in (
            (self.solver_name, "solver name"),
            (self.solver_version, "solver version"),
            (self.equilibrium_concept, "equilibrium concept"),
            (self.status, "solver status"),
            (self.exactness_status, "exactness status"),
        ):
            validate_stable_id(value, field_name=name)
        if not isinstance(self.found, bool):
            raise TypeError("found must be boolean")
        solutions = tuple(self.solutions)
        if self.found and not solutions:
            raise ValueError("found=True requires at least one solution")
        if not self.found and solutions:
            raise ValueError("found=False cannot contain solutions")
        object.__setattr__(self, "solutions", solutions)
        observed_multiple = len(solutions) > 1
        if (
            self.multiple_solutions is not None
            and self.multiple_solutions != observed_multiple
        ):
            raise ValueError(
                "multiple_solutions is inconsistent with the solution count"
            )
        object.__setattr__(self, "multiple_solutions", observed_multiple)
        if self.best_response_gap is not None:
            gap = ensure_finite(self.best_response_gap, name="best-response gap")
            if gap < -1e-12:
                raise ValueError("best-response gap must be nonnegative")
            object.__setattr__(self, "best_response_gap", max(0.0, gap))
        runtime = ensure_finite(self.runtime_seconds, name="solver runtime")
        if runtime < 0.0:
            raise ValueError("solver runtime must be nonnegative")
        object.__setattr__(self, "runtime_seconds", runtime)
        object.__setattr__(self, "warnings", _strings(self.warnings, name="warnings"))
        object.__setattr__(
            self, "assumptions", _strings(self.assumptions, name="assumptions")
        )
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("seed must be an integer or None")
        if self.convergence_status is not None and not self.convergence_status:
            raise ValueError("convergence status must be non-empty when supplied")
        if (
            self.off_path_belief_convention is not None
            and not self.off_path_belief_convention
        ):
            raise ValueError(
                "off-path-belief convention must be non-empty when supplied"
            )
        metadata = {} if self.metadata is None else dict(self.metadata)
        to_serializable(metadata)
        object.__setattr__(self, "metadata", frozen_mapping(metadata))

    @property
    def multiple(self) -> bool:
        return bool(self.multiple_solutions)

    @property
    def exactness(self) -> str:
        return self.exactness_status

    @property
    def equilibria(self) -> tuple[Any, ...]:
        """Semantic alias used by equilibrium-enumeration callers."""

        return self.solutions


@dataclass(frozen=True)
class DiagnosticResult(SerializableMixin):
    diagnostic_name: str
    diagnostic_version: str
    definition: str
    values: Mapping[str, float]
    per_player: Mapping[str, Mapping[str, float]]
    runtime_seconds: float = 0.0
    warnings: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    seed: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_stable_id(self.diagnostic_name, field_name="diagnostic name")
        validate_stable_id(self.diagnostic_version, field_name="diagnostic version")
        if not self.definition:
            raise ValueError("diagnostic definition must be non-empty")
        values = {
            validate_stable_id(key, field_name="diagnostic value key"): ensure_finite(
                value, name=f"diagnostic value {key!r}"
            )
            for key, value in self.values.items()
        }
        per_player: dict[str, Mapping[str, float]] = {}
        for player, diagnostics in self.per_player.items():
            validate_stable_id(player, field_name="diagnostic player id")
            per_player[player] = frozen_mapping(
                {
                    validate_stable_id(
                        key, field_name="player diagnostic key"
                    ): ensure_finite(value, name=f"{player} diagnostic {key!r}")
                    for key, value in diagnostics.items()
                }
            )
        object.__setattr__(self, "values", frozen_mapping(values))
        object.__setattr__(self, "per_player", frozen_mapping(per_player))
        runtime = ensure_finite(self.runtime_seconds, name="diagnostic runtime")
        if runtime < 0.0:
            raise ValueError("diagnostic runtime must be nonnegative")
        object.__setattr__(self, "runtime_seconds", runtime)
        object.__setattr__(self, "warnings", _strings(self.warnings, name="warnings"))
        object.__setattr__(
            self, "assumptions", _strings(self.assumptions, name="assumptions")
        )
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("seed must be an integer or None")
        metadata = {} if self.metadata is None else dict(self.metadata)
        to_serializable(metadata)
        object.__setattr__(self, "metadata", frozen_mapping(metadata))
