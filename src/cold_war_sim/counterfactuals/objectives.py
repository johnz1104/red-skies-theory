"""Modeled utility functions and explicit counterfactual objectives."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from cold_war_sim.core.types import SerializableMixin, frozen_mapping, validate_stable_id

from .outcomes import (
    DEFAULT_FEATURE_REGISTRY,
    FeatureRegistry,
    OutcomeDistribution,
    OutcomeFeatures,
)


def _finite(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


class UtilityModel(Protocol):
    def evaluate(
        self,
        player: str,
        outcome: OutcomeFeatures,
        parameters: Mapping[str, float],
    ) -> float: ...


@dataclass(frozen=True)
class LinearUtilityModel(SerializableMixin):
    """Replaceable linear utility over registered outcome features."""

    common_weights: Mapping[str, Mapping[str, float]]
    player_feature_weights: Mapping[str, Mapping[str, float]]
    intercepts: Mapping[str, float]
    registry: FeatureRegistry = DEFAULT_FEATURE_REGISTRY

    def __post_init__(self) -> None:
        players = set(self.common_weights) | set(self.player_feature_weights) | set(
            self.intercepts
        )
        for player in players:
            validate_stable_id(player, field_name="utility player id")
        for player, weights in self.common_weights.items():
            unknown = set(weights) - set(self.registry.common)
            if unknown:
                raise ValueError(
                    f"unregistered common utility features for {player!r}: {sorted(unknown)}"
                )
            for feature, weight in weights.items():
                _finite(weight, name=f"utility weight {player}.{feature}")
        for player, weights in self.player_feature_weights.items():
            unknown = set(weights) - set(self.registry.by_player)
            if unknown:
                raise ValueError(
                    f"unregistered player utility features for {player!r}: {sorted(unknown)}"
                )
            for feature, weight in weights.items():
                _finite(weight, name=f"utility weight {player}.{feature}")
        for player, intercept in self.intercepts.items():
            _finite(intercept, name=f"utility intercept {player}")
        object.__setattr__(
            self,
            "common_weights",
            frozen_mapping(
                {player: frozen_mapping(weights) for player, weights in self.common_weights.items()}
            ),
        )
        object.__setattr__(
            self,
            "player_feature_weights",
            frozen_mapping(
                {
                    player: frozen_mapping(weights)
                    for player, weights in self.player_feature_weights.items()
                }
            ),
        )
        object.__setattr__(self, "intercepts", frozen_mapping(self.intercepts))

    def evaluate(
        self,
        player: str,
        outcome: OutcomeFeatures,
        parameters: Mapping[str, float] | None = None,
    ) -> float:
        validate_stable_id(player, field_name="player id")
        if outcome.registry != self.registry:
            raise ValueError(
                "utility and outcome feature registries must match exactly"
            )
        common_weights = self.common_weights.get(player, {})
        own_weights = self.player_feature_weights.get(player, {})
        value = float(self.intercepts.get(player, 0.0))
        for feature, weight in common_weights.items():
            value += float(weight) * float(outcome.common.get(feature, 0.0))
        for feature, weight in own_weights.items():
            value += float(weight) * float(outcome.by_player.get(player, {}).get(feature, 0.0))
        for name, adjustment in (parameters or {}).items():
            if not name.startswith("intercept:"):
                raise ValueError(f"unknown utility parameter {name!r}")
            parameter_player = name.removeprefix("intercept:")
            known_players = (
                set(self.common_weights)
                | set(self.player_feature_weights)
                | set(self.intercepts)
            )
            if parameter_player not in known_players:
                raise ValueError(f"unknown utility-parameter player {parameter_player!r}")
            if parameter_player == player:
                value += float(adjustment)
        if not math.isfinite(value):
            raise ArithmeticError("utility evaluation produced a non-finite value")
        return value


def expected_utilities(
    distribution: OutcomeDistribution,
    players: tuple[str, ...],
    utility_model: UtilityModel,
    parameters: Mapping[str, float] | None = None,
) -> Mapping[str, float]:
    return frozen_mapping(
        {
            player: sum(
                item.probability
                * utility_model.evaluate(player, item.features, parameters or {})
                for item in distribution.outcomes
            )
            for player in players
        }
    )


class ObjectiveKind(StrEnum):
    MAXIMIZE_EXPECTED_UTILITY = "MAXIMIZE_EXPECTED_UTILITY"
    MAXIMIZE_WORST_CASE_UTILITY = "MAXIMIZE_WORST_CASE_UTILITY"
    MINIMIZE_ESCALATION_PROBABILITY = "MINIMIZE_ESCALATION_PROBABILITY"
    MINIMIZE_CATASTROPHE_PROBABILITY = "MINIMIZE_CATASTROPHE_PROBABILITY"
    MAXIMIZE_NEGOTIATED_SETTLEMENT_PROBABILITY = (
        "MAXIMIZE_NEGOTIATED_SETTLEMENT_PROBABILITY"
    )
    LEXICOGRAPHIC_OBJECTIVE = "LEXICOGRAPHIC_OBJECTIVE"


@dataclass(frozen=True)
class CandidateMetrics(SerializableMixin):
    expected_utilities: Mapping[str, float]
    scenario_utilities: Mapping[str, Mapping[str, float]]
    escalation_probability: float
    catastrophe_probability: float
    negotiated_settlement_probability: float

    def __post_init__(self) -> None:
        if not self.expected_utilities:
            raise ValueError("candidate metrics require at least one player utility")
        players = set(self.expected_utilities)
        for player, value in self.expected_utilities.items():
            validate_stable_id(player, field_name="candidate-metric player id")
            _finite(value, name=f"expected utility for {player}")
        for scenario, values in self.scenario_utilities.items():
            validate_stable_id(scenario, field_name="parameter-scenario id")
            if set(values) != players:
                raise ValueError(
                    "every scenario utility mapping must cover the expected-utility players"
                )
            for player, value in values.items():
                _finite(value, name=f"scenario utility {scenario}.{player}")
        object.__setattr__(
            self,
            "expected_utilities",
            frozen_mapping(
                {player: float(value) for player, value in self.expected_utilities.items()}
            ),
        )
        object.__setattr__(
            self,
            "scenario_utilities",
            frozen_mapping(
                {
                    scenario: frozen_mapping(
                        {player: float(value) for player, value in values.items()}
                    )
                    for scenario, values in self.scenario_utilities.items()
                }
            ),
        )
        for name in (
            "escalation_probability",
            "catastrophe_probability",
            "negotiated_settlement_probability",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, value)


class Objective(Protocol):
    kind: ObjectiveKind

    def score(self, metrics: CandidateMetrics) -> tuple[float, ...]: ...

    def to_dict(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class ExpectedUtilityObjective(SerializableMixin):
    player_id: str
    kind: ObjectiveKind = ObjectiveKind.MAXIMIZE_EXPECTED_UTILITY

    def __post_init__(self) -> None:
        validate_stable_id(self.player_id, field_name="objective player id")

    def score(self, metrics: CandidateMetrics) -> tuple[float, ...]:
        return (metrics.expected_utilities[self.player_id],)

    def to_dict(self) -> dict[str, object]:
        return {"type": self.kind.value, "player": self.player_id}


@dataclass(frozen=True)
class WorstCaseUtilityObjective(SerializableMixin):
    player_id: str
    kind: ObjectiveKind = ObjectiveKind.MAXIMIZE_WORST_CASE_UTILITY

    def __post_init__(self) -> None:
        validate_stable_id(self.player_id, field_name="objective player id")

    def score(self, metrics: CandidateMetrics) -> tuple[float, ...]:
        if not metrics.scenario_utilities:
            return (metrics.expected_utilities[self.player_id],)
        return (
            min(values[self.player_id] for values in metrics.scenario_utilities.values()),
        )

    def to_dict(self) -> dict[str, object]:
        return {"type": self.kind.value, "player": self.player_id}


@dataclass(frozen=True)
class FeatureProbabilityObjective(SerializableMixin):
    kind: ObjectiveKind

    def __post_init__(self) -> None:
        permitted = {
            ObjectiveKind.MINIMIZE_ESCALATION_PROBABILITY,
            ObjectiveKind.MINIMIZE_CATASTROPHE_PROBABILITY,
            ObjectiveKind.MAXIMIZE_NEGOTIATED_SETTLEMENT_PROBABILITY,
        }
        if self.kind not in permitted:
            raise ValueError("unsupported feature-probability objective")

    def score(self, metrics: CandidateMetrics) -> tuple[float, ...]:
        if self.kind is ObjectiveKind.MINIMIZE_ESCALATION_PROBABILITY:
            return (-metrics.escalation_probability,)
        if self.kind is ObjectiveKind.MINIMIZE_CATASTROPHE_PROBABILITY:
            return (-metrics.catastrophe_probability,)
        return (metrics.negotiated_settlement_probability,)

    def to_dict(self) -> dict[str, object]:
        return {"type": self.kind.value}


@dataclass(frozen=True)
class LexicographicObjective(SerializableMixin):
    objectives: tuple[Objective, ...]
    kind: ObjectiveKind = ObjectiveKind.LEXICOGRAPHIC_OBJECTIVE

    def __post_init__(self) -> None:
        if not self.objectives:
            raise ValueError("lexicographic objective must contain at least one objective")
        if any(objective.kind is ObjectiveKind.LEXICOGRAPHIC_OBJECTIVE for objective in self.objectives):
            raise ValueError("nested lexicographic objectives are not supported")

    def score(self, metrics: CandidateMetrics) -> tuple[float, ...]:
        return tuple(value for objective in self.objectives for value in objective.score(metrics))

    def to_dict(self) -> dict[str, object]:
        return {"type": self.kind.value, "objectives": [item.to_dict() for item in self.objectives]}
