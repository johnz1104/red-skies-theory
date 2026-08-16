"""Explicit policy and outcome constraints for exact search."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from cold_war_sim.core.types import SerializableMixin, validate_stable_id

from .objectives import CandidateMetrics


@dataclass(frozen=True)
class ConstraintContext(SerializableMixin):
    metrics: CandidateMetrics
    policy: Mapping[str, str]
    baseline_policy: Mapping[str, str]

    @property
    def policy_changes(self) -> int:
        keys = set(self.policy) | set(self.baseline_policy)
        return sum(self.policy.get(key) != self.baseline_policy.get(key) for key in keys)


class Constraint(Protocol):
    def check(self, context: ConstraintContext, *, tolerance: float) -> bool: ...

    def to_dict(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class MinimumExpectedUtility(SerializableMixin):
    player_id: str
    minimum: float

    def __post_init__(self) -> None:
        validate_stable_id(self.player_id, field_name="constraint player id")
        if not math.isfinite(float(self.minimum)):
            raise ValueError("minimum expected utility must be finite")

    def check(self, context: ConstraintContext, *, tolerance: float) -> bool:
        return context.metrics.expected_utilities[self.player_id] >= self.minimum - tolerance

    def to_dict(self) -> dict[str, object]:
        return {"type": type(self).__name__, "player": self.player_id, "minimum": self.minimum}


@dataclass(frozen=True)
class MaximumEscalationProbability(SerializableMixin):
    maximum: float

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.maximum)) or not 0.0 <= self.maximum <= 1.0:
            raise ValueError("maximum escalation probability must lie in [0, 1]")

    def check(self, context: ConstraintContext, *, tolerance: float) -> bool:
        return context.metrics.escalation_probability <= self.maximum + tolerance

    def to_dict(self) -> dict[str, object]:
        return {"type": type(self).__name__, "maximum": self.maximum}


@dataclass(frozen=True)
class MaximumCatastropheProbability(SerializableMixin):
    maximum: float

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.maximum)) or not 0.0 <= self.maximum <= 1.0:
            raise ValueError("maximum catastrophe probability must lie in [0, 1]")

    def check(self, context: ConstraintContext, *, tolerance: float) -> bool:
        return context.metrics.catastrophe_probability <= self.maximum + tolerance

    def to_dict(self) -> dict[str, object]:
        return {"type": type(self).__name__, "maximum": self.maximum}


@dataclass(frozen=True)
class MinimumSettlementProbability(SerializableMixin):
    minimum: float

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.minimum)) or not 0.0 <= self.minimum <= 1.0:
            raise ValueError("minimum settlement probability must lie in [0, 1]")

    def check(self, context: ConstraintContext, *, tolerance: float) -> bool:
        return context.metrics.negotiated_settlement_probability >= self.minimum - tolerance

    def to_dict(self) -> dict[str, object]:
        return {"type": type(self).__name__, "minimum": self.minimum}


@dataclass(frozen=True)
class MaximumPolicyChanges(SerializableMixin):
    maximum: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.maximum, bool)
            or not isinstance(self.maximum, int)
            or self.maximum < 0
        ):
            raise ValueError("maximum policy changes must be a nonnegative integer")

    def check(self, context: ConstraintContext, *, tolerance: float) -> bool:
        del tolerance
        return context.policy_changes <= self.maximum

    def to_dict(self) -> dict[str, object]:
        return {"type": type(self).__name__, "maximum": self.maximum}


@dataclass(frozen=True)
class AllowedInformationSets(SerializableMixin):
    information_set_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        identifiers = tuple(sorted(self.information_set_ids))
        if not identifiers:
            raise ValueError("allowed information sets must not be empty")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("allowed information sets must not contain duplicates")
        for identifier in identifiers:
            validate_stable_id(identifier, field_name="allowed information-set id")
        object.__setattr__(self, "information_set_ids", identifiers)

    def check(self, context: ConstraintContext, *, tolerance: float) -> bool:
        del tolerance
        allowed = set(self.information_set_ids)
        keys = set(context.policy) | set(context.baseline_policy)
        changed = {key for key in keys if context.policy.get(key) != context.baseline_policy.get(key)}
        return changed <= allowed

    def to_dict(self) -> dict[str, object]:
        return {"type": type(self).__name__, "information_sets": list(self.information_set_ids)}


ConstraintType = (
    MinimumExpectedUtility
    | MaximumEscalationProbability
    | MaximumCatastropheProbability
    | MinimumSettlementProbability
    | MaximumPolicyChanges
    | AllowedInformationSets
)
