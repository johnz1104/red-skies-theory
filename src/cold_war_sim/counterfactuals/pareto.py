"""Player, Pareto, constrained, robust, and maximin comparisons."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from cold_war_sim.core.types import SerializableMixin, frozen_mapping, validate_stable_id


def _validate_tolerance(tolerance: float) -> float:
    value = float(tolerance)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("comparison tolerance must be finite and nonnegative")
    return value


def _finite_values(values: Mapping[str, float], *, name: str) -> dict[str, float]:
    if not values:
        raise ValueError(f"{name} must not be empty")
    converted = {}
    for identifier, raw in values.items():
        validate_stable_id(identifier, field_name=f"{name} id")
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"{name} values must be finite")
        converted[identifier] = value
    return converted


def player_improvement(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    player_id: str,
    *,
    tolerance: float = 1e-9,
) -> bool:
    tolerance = _validate_tolerance(tolerance)
    baseline_values = _finite_values(baseline, name="baseline utilities")
    candidate_values = _finite_values(candidate, name="candidate utilities")
    if set(baseline_values) != set(candidate_values):
        raise ValueError("player comparisons require the same players")
    validate_stable_id(player_id, field_name="focal player id")
    return candidate_values[player_id] > baseline_values[player_id] + tolerance


def pareto_dominates(
    candidate: Mapping[str, float],
    reference: Mapping[str, float],
    *,
    tolerance: float = 1e-9,
) -> bool:
    tolerance = _validate_tolerance(tolerance)
    candidate_values = _finite_values(candidate, name="candidate utilities")
    reference_values = _finite_values(reference, name="reference utilities")
    if set(candidate_values) != set(reference_values):
        raise ValueError("Pareto comparisons require the same players")
    weak = all(
        candidate_values[player] >= reference_values[player] - tolerance
        for player in candidate_values
    )
    strict = any(
        candidate_values[player] > reference_values[player] + tolerance
        for player in candidate_values
    )
    return weak and strict


@dataclass(frozen=True)
class ParetoPoint(SerializableMixin):
    id: str
    utilities: Mapping[str, float]

    def __post_init__(self) -> None:
        validate_stable_id(self.id, field_name="Pareto-point id")
        object.__setattr__(
            self,
            "utilities",
            frozen_mapping(_finite_values(self.utilities, name="Pareto utilities")),
        )


def pareto_frontier(
    points: Sequence[ParetoPoint], *, tolerance: float = 1e-9
) -> tuple[ParetoPoint, ...]:
    tolerance = _validate_tolerance(tolerance)
    ids = tuple(point.id for point in points)
    if len(ids) != len(set(ids)):
        raise ValueError("Pareto-point ids must be unique")
    if points:
        players = set(points[0].utilities)
        if any(set(point.utilities) != players for point in points[1:]):
            raise ValueError("Pareto points must report the same players")
    unique = {point.id: point for point in points}
    ordered = tuple(unique[key] for key in sorted(unique))
    return tuple(
        point
        for point in ordered
        if not any(
            other.id != point.id
            and pareto_dominates(other.utilities, point.utilities, tolerance=tolerance)
            for other in ordered
        )
    )


def robust_improvement(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    tolerance: float = 1e-9,
) -> bool:
    tolerance = _validate_tolerance(tolerance)
    baseline_values = _finite_values(baseline, name="baseline scenario values")
    candidate_values = _finite_values(candidate, name="candidate scenario values")
    if set(baseline_values) != set(candidate_values):
        raise ValueError("robust comparisons require identical scenarios")
    return all(
        candidate_values[key] > baseline_values[key] + tolerance
        for key in baseline_values
    )


def maximin_improvement(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    tolerance: float = 1e-9,
) -> bool:
    tolerance = _validate_tolerance(tolerance)
    baseline_values = _finite_values(baseline, name="baseline scenario values")
    candidate_values = _finite_values(candidate, name="candidate scenario values")
    if set(baseline_values) != set(candidate_values):
        raise ValueError("maximin comparisons require identical scenarios")
    return min(candidate_values.values()) > min(baseline_values.values()) + tolerance
