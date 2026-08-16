"""Outcome features and distributions, deliberately separate from utility."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from cold_war_sim.core.types import SerializableMixin, frozen_mapping, validate_stable_id

FeatureValue = float | int | bool


@dataclass(frozen=True)
class FeatureDefinition(SerializableMixin):
    name: str
    value_type: str = "number"
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.name, field_name="feature name")
        if self.value_type not in {"number", "integer", "boolean"}:
            raise ValueError("feature value_type must be number, integer, or boolean")
        if self.minimum is not None and not math.isfinite(float(self.minimum)):
            raise ValueError("feature minimum must be finite")
        if self.maximum is not None and not math.isfinite(float(self.maximum)):
            raise ValueError("feature maximum must be finite")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("feature minimum cannot exceed maximum")

    def validate(self, value: FeatureValue) -> None:
        if self.value_type == "boolean":
            if not isinstance(value, bool):
                raise TypeError(f"feature {self.name!r} must be boolean")
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"feature {self.name!r} must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"feature {self.name!r} must be finite")
        if self.value_type == "integer" and not isinstance(value, int):
            raise TypeError(f"feature {self.name!r} must be an integer")
        if self.minimum is not None and number < self.minimum:
            raise ValueError(f"feature {self.name!r} is below its minimum")
        if self.maximum is not None and number > self.maximum:
            raise ValueError(f"feature {self.name!r} is above its maximum")


@dataclass(frozen=True)
class FeatureRegistry(SerializableMixin):
    common: Mapping[str, FeatureDefinition]
    by_player: Mapping[str, FeatureDefinition]

    def __post_init__(self) -> None:
        common = dict(sorted(self.common.items()))
        by_player = dict(sorted(self.by_player.items()))
        for key, definition in (*common.items(), *by_player.items()):
            if key != definition.name:
                raise ValueError("feature registry keys must match definition names")
        overlap = set(common) & set(by_player)
        if overlap:
            raise ValueError(
                "common and per-player feature names must be distinct: "
                f"{sorted(overlap)}"
            )
        object.__setattr__(self, "common", frozen_mapping(common))
        object.__setattr__(self, "by_player", frozen_mapping(by_player))


DEFAULT_FEATURE_REGISTRY = FeatureRegistry(
    common={
        "peaceful_settlement": FeatureDefinition("peaceful_settlement", "boolean"),
        "negotiated_agreement": FeatureDefinition("negotiated_agreement", "boolean"),
        "concession": FeatureDefinition("concession"),
        "military_escalation": FeatureDefinition("military_escalation", "boolean"),
        "catastrophic_escalation": FeatureDefinition("catastrophic_escalation", "boolean"),
        "intervention": FeatureDefinition("intervention", "boolean"),
        "bargaining_duration": FeatureDefinition("bargaining_duration", "integer", 0.0),
        "escalation_probability": FeatureDefinition(
            "escalation_probability", minimum=0.0, maximum=1.0
        ),
        "catastrophe_probability": FeatureDefinition(
            "catastrophe_probability", minimum=0.0, maximum=1.0
        ),
        "settlement_probability": FeatureDefinition(
            "settlement_probability", minimum=0.0, maximum=1.0
        ),
        "aggregate_armament": FeatureDefinition("aggregate_armament", minimum=0.0),
        "missile_removal": FeatureDefinition("missile_removal"),
        "war_intensity": FeatureDefinition("war_intensity", minimum=0.0),
        "receiver_advance": FeatureDefinition("receiver_advance"),
    },
    by_player={
        "political_cost": FeatureDefinition("political_cost"),
        "military_cost": FeatureDefinition("military_cost"),
        "economic_cost": FeatureDefinition("economic_cost"),
        "credibility_effect": FeatureDefinition("credibility_effect"),
        "reputation_effect": FeatureDefinition("reputation_effect"),
    },
)


@dataclass(frozen=True)
class OutcomeFeatures(SerializableMixin):
    common: Mapping[str, FeatureValue]
    by_player: Mapping[str, Mapping[str, float]]
    registry: FeatureRegistry = DEFAULT_FEATURE_REGISTRY

    def __post_init__(self) -> None:
        common = dict(sorted(self.common.items()))
        for key, value in common.items():
            if key not in self.registry.common:
                raise ValueError(f"unregistered common outcome feature {key!r}")
            self.registry.common[key].validate(value)
        players: dict[str, Mapping[str, float]] = {}
        for player, features in sorted(self.by_player.items()):
            validate_stable_id(player, field_name="player id")
            converted: dict[str, float] = {}
            for key, value in sorted(features.items()):
                if key not in self.registry.by_player:
                    raise ValueError(f"unregistered per-player outcome feature {key!r}")
                self.registry.by_player[key].validate(value)
                converted[key] = float(value)
            players[player] = MappingProxyType(converted)
        object.__setattr__(self, "common", frozen_mapping(common))
        object.__setattr__(self, "by_player", frozen_mapping(players))

    def to_dict(self) -> dict[str, object]:
        return {
            "common": dict(self.common),
            "by_player": {
                player: dict(features) for player, features in self.by_player.items()
            },
        }


@dataclass(frozen=True)
class WeightedOutcome(SerializableMixin):
    outcome_id: str
    probability: float
    features: OutcomeFeatures

    def __post_init__(self) -> None:
        validate_stable_id(self.outcome_id, field_name="outcome id")
        probability = float(self.probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("outcome probability must lie in [0, 1]")
        object.__setattr__(self, "probability", probability)

    def to_dict(self) -> dict[str, object]:
        return {
            "outcome_id": self.outcome_id,
            "probability": self.probability,
            "features": self.features.to_dict(),
        }


@dataclass(frozen=True)
class OutcomeDistribution(SerializableMixin):
    outcomes: tuple[WeightedOutcome, ...]
    tolerance: float = 1e-9

    def __post_init__(self) -> None:
        tolerance = float(self.tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("outcome-distribution tolerance must be finite and positive")
        outcomes = tuple(sorted(self.outcomes, key=lambda item: item.outcome_id))
        if not outcomes:
            raise ValueError("outcome distribution must not be empty")
        identifiers = tuple(item.outcome_id for item in outcomes)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("outcome identifiers must be unique")
        total = sum(item.probability for item in outcomes)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError("outcome probabilities must sum to one")
        registry = outcomes[0].features.registry
        if any(item.features.registry != registry for item in outcomes[1:]):
            raise ValueError(
                "all outcomes in a distribution must use the same feature registry"
            )
        players = set(outcomes[0].features.by_player)
        if any(set(item.features.by_player) != players for item in outcomes[1:]):
            raise ValueError(
                "all outcomes in a distribution must report the same players"
            )
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "tolerance", tolerance)

    def expected_common(self, feature: str) -> float:
        if feature not in self.outcomes[0].features.registry.common:
            raise ValueError(f"unregistered outcome feature {feature!r}")
        return sum(
            item.probability * float(item.features.common.get(feature, 0.0))
            for item in self.outcomes
        )

    def probability(self, feature: str) -> float:
        return self.expected_common(feature)

    def to_dict(self) -> dict[str, object]:
        return {
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
            "tolerance": self.tolerance,
        }
