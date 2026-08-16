"""Strict probability validation and explicit normalization helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .types import SerializableMixin, frozen_mapping, validate_stable_id

DEFAULT_TOLERANCE = 1e-10


def _validate_tolerance(tolerance: float) -> float:
    value = float(tolerance)
    if not math.isfinite(value) or value <= 0.0 or value >= 1.0:
        raise ValueError("tolerance must be finite and lie strictly between 0 and 1")
    return value


def validate_probability(value: float, *, name: str = "probability") -> float:
    """Validate and return one probability without clipping it."""

    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        probability = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    if not math.isfinite(probability):
        raise ValueError(f"{name} must be finite")
    if probability < 0.0 or probability > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return probability


def validate_probability_distribution(
    probabilities: Sequence[float] | Mapping[str, float],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    name: str = "probability distribution",
) -> tuple[float, ...]:
    """Validate a finite distribution and return its values in input order.

    This function never clips or normalizes.  Inputs whose total mass differs
    from one beyond *tolerance* are rejected with the observed total.
    """

    tolerance = _validate_tolerance(tolerance)
    raw_values = (
        list(probabilities.values())
        if isinstance(probabilities, Mapping)
        else list(probabilities)
    )
    if not raw_values:
        raise ValueError(f"{name} must contain at least one outcome")
    values = tuple(
        validate_probability(value, name=f"{name}[{index}]")
        for index, value in enumerate(raw_values)
    )
    total = math.fsum(values)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(
            f"{name} must sum to 1 within {tolerance:g}; observed {total:.17g}"
        )
    return values


validate_distribution = validate_probability_distribution


def normalize_weights(
    weights: Sequence[float] | Mapping[str, float],
) -> tuple[float, ...] | dict[str, float]:
    """Explicitly normalize nonnegative finite weights.

    Unlike distribution validation this operation is intentionally named for
    the transformation it performs, so callers cannot normalize bad model
    inputs accidentally.
    """

    if isinstance(weights, Mapping):
        is_mapping = True
        keys = list(weights)
        raw_values = list(weights.values())
    else:
        is_mapping = False
        keys = []
        raw_values = list(weights)
    if not raw_values:
        raise ValueError("weights must contain at least one value")
    values: list[float] = []
    for index, value in enumerate(raw_values):
        if isinstance(value, bool):
            raise TypeError(f"weight[{index}] must be a real number, not bool")
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"weight[{index}] must be a real number") from error
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"weight[{index}] must be finite and nonnegative")
        values.append(number)
    total = math.fsum(values)
    if total <= 0.0:
        raise ValueError("weights must have positive total mass")
    normalized = tuple(value / total for value in values)
    if is_mapping:
        return {str(key): normalized[index] for index, key in enumerate(keys)}
    return normalized


@dataclass(frozen=True)
class ProbabilityDistribution(SerializableMixin):
    """Immutable labeled probability distribution with deterministic ordering."""

    probabilities: Mapping[str, float]
    tolerance: float = DEFAULT_TOLERANCE

    def __post_init__(self) -> None:
        if not isinstance(self.probabilities, Mapping):
            raise TypeError(
                "probabilities must be a mapping from outcome IDs to probabilities"
            )
        sorted_items = sorted(self.probabilities.items(), key=lambda item: item[0])
        for outcome, _ in sorted_items:
            validate_stable_id(outcome, field_name="probability outcome id")
        validated = validate_probability_distribution(
            [value for _, value in sorted_items],
            tolerance=self.tolerance,
        )
        object.__setattr__(
            self,
            "probabilities",
            frozen_mapping(
                {
                    outcome: validated[index]
                    for index, (outcome, _) in enumerate(sorted_items)
                }
            ),
        )
        object.__setattr__(self, "tolerance", _validate_tolerance(self.tolerance))

    def __getitem__(self, outcome_id: str) -> float:
        return self.probabilities[outcome_id]

    def support(self, *, tolerance: float = 0.0) -> tuple[str, ...]:
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("support tolerance must be finite and nonnegative")
        return tuple(
            outcome
            for outcome, probability in self.probabilities.items()
            if probability > tolerance
        )

    @classmethod
    def from_sequences(
        cls,
        labels: Sequence[str],
        probabilities: Sequence[float],
        *,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> ProbabilityDistribution:
        labels_tuple = tuple(labels)
        if len(labels_tuple) != len(probabilities):
            raise ValueError("labels and probabilities must have the same length")
        if len(labels_tuple) != len(set(labels_tuple)):
            raise ValueError("distribution labels must be unique")
        return cls(
            dict(zip(labels_tuple, probabilities, strict=True)), tolerance=tolerance
        )


def ensure_finite(value: Any, *, name: str = "value") -> float:
    """Validate a generic finite scalar used in utilities or diagnostics."""

    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a finite real number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result
