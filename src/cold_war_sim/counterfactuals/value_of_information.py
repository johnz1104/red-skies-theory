"""Reusable strategic values of information and enforceable commitment."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from cold_war_sim.core.types import SerializableMixin, frozen_mapping, validate_stable_id


class InformationChange(StrEnum):
    PRIVATE_INFORMATION_IMPROVEMENT = "PRIVATE_INFORMATION_IMPROVEMENT"
    PUBLIC_SIGNAL_IMPROVEMENT = "PUBLIC_SIGNAL_IMPROVEMENT"
    SIGNAL_ACCURACY_CHANGE = "SIGNAL_ACCURACY_CHANGE"
    INFORMATION_REMOVAL = "INFORMATION_REMOVAL"
    INFORMATION_DISCLOSURE = "INFORMATION_DISCLOSURE"


class CommitmentMode(StrEnum):
    ONE_ACTION = "ONE_ACTION"
    COMPLETE_CONTINGENT_POLICY = "COMPLETE_CONTINGENT_POLICY"
    OBSERVABLE_NONBINDING_ANNOUNCEMENT = "OBSERVABLE_NONBINDING_ANNOUNCEMENT"
    BINDING_PUBLIC_COMMITMENT = "BINDING_PUBLIC_COMMITMENT"


@dataclass(frozen=True)
class StrategicValue(SerializableMixin):
    player_id: str
    baseline_values: tuple[float, ...]
    transformed_values: tuple[float, ...]
    minimum_difference: float
    maximum_difference: float
    selection_treatment: str

    def __post_init__(self) -> None:
        validate_stable_id(self.player_id, field_name="strategic-value player id")
        if not self.baseline_values or not self.transformed_values:
            raise ValueError("strategic value sets must be nonempty")
        baseline = tuple(float(value) for value in self.baseline_values)
        transformed = tuple(float(value) for value in self.transformed_values)
        values = (*baseline, *transformed)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("strategic values must be finite")
        minimum = float(self.minimum_difference)
        maximum = float(self.maximum_difference)
        if not math.isfinite(minimum) or not math.isfinite(maximum):
            raise ValueError("strategic value differences must be finite")
        if minimum > maximum:
            raise ValueError("minimum difference cannot exceed maximum difference")
        actual_differences = tuple(
            after - before for before in baseline for after in transformed
        )
        if not math.isclose(
            minimum, min(actual_differences), rel_tol=0.0, abs_tol=1e-12
        ) or not math.isclose(
            maximum, max(actual_differences), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "strategic-value difference bounds must match the retained solution sets"
            )
        if (
            not isinstance(self.selection_treatment, str)
            or not self.selection_treatment.strip()
        ):
            raise ValueError("selection treatment must not be empty")
        object.__setattr__(self, "baseline_values", tuple(sorted(baseline)))
        object.__setattr__(self, "transformed_values", tuple(sorted(transformed)))
        object.__setattr__(self, "minimum_difference", minimum)
        object.__setattr__(self, "maximum_difference", maximum)


@dataclass(frozen=True)
class ValueOfInformationResult(SerializableMixin):
    change: InformationChange
    by_player: Mapping[str, StrategicValue]
    public_information_benefits_every_player: bool

    def __post_init__(self) -> None:
        if not isinstance(self.change, InformationChange):
            raise TypeError("change must be an InformationChange")
        if not isinstance(self.public_information_benefits_every_player, bool):
            raise TypeError("public-information benefit flag must be a boolean")
        by_player = frozen_mapping(self.by_player)
        if not by_player:
            raise ValueError("value-of-information results must report players")
        if any(player != value.player_id for player, value in by_player.items()):
            raise ValueError("value-of-information player keys must match values")
        expected_benefit = all(
            item.minimum_difference >= 0.0 for item in by_player.values()
        )
        if self.public_information_benefits_every_player != expected_benefit:
            raise ValueError(
                "public-information benefit flag must match the strategic values"
            )
        object.__setattr__(self, "by_player", by_player)


@dataclass(frozen=True)
class ValueOfCommitmentResult(SerializableMixin):
    mode: CommitmentMode
    observable: bool
    binding: bool
    by_player: Mapping[str, StrategicValue]
    is_commitment: bool

    def __post_init__(self) -> None:
        if not isinstance(self.mode, CommitmentMode):
            raise TypeError("mode must be a CommitmentMode")
        if not all(isinstance(value, bool) for value in (
            self.observable,
            self.binding,
            self.is_commitment,
        )):
            raise TypeError("commitment flags must be booleans")
        if self.mode is CommitmentMode.OBSERVABLE_NONBINDING_ANNOUNCEMENT and (
            not self.observable or self.binding
        ):
            raise ValueError(
                "an observable nonbinding announcement must be observable and nonbinding"
            )
        expected_commitment = self.observable and self.binding and self.mode in {
            CommitmentMode.ONE_ACTION,
            CommitmentMode.COMPLETE_CONTINGENT_POLICY,
            CommitmentMode.BINDING_PUBLIC_COMMITMENT,
        }
        if self.is_commitment != expected_commitment:
            raise ValueError("is_commitment must match mode, observability, and binding")
        by_player = frozen_mapping(self.by_player)
        if not by_player:
            raise ValueError("value-of-commitment results must report players")
        if any(player != value.player_id for player, value in by_player.items()):
            raise ValueError("value-of-commitment player keys must match values")
        object.__setattr__(self, "by_player", by_player)


def _strategic_values(
    baseline: Sequence[Mapping[str, float]],
    transformed: Sequence[Mapping[str, float]],
    *,
    selection_treatment: str,
) -> Mapping[str, StrategicValue]:
    if not baseline or not transformed:
        raise ValueError("both solution sets must be nonempty")
    players = tuple(sorted(baseline[0]))
    if not players:
        raise ValueError("solution sets must report at least one player")
    for player in players:
        validate_stable_id(player, field_name="strategic-value player id")
    if any(set(item) != set(players) for item in (*baseline, *transformed)):
        raise ValueError("all solutions must report the same players")
    result = {}
    for player in players:
        old = tuple(sorted(float(item[player]) for item in baseline))
        new = tuple(sorted(float(item[player]) for item in transformed))
        if not all(math.isfinite(value) for value in (*old, *new)):
            raise ValueError("solution-set values must be finite")
        differences = tuple(after - before for before in old for after in new)
        result[player] = StrategicValue(
            player, old, new, min(differences), max(differences), selection_treatment
        )
    return frozen_mapping(result)


def value_of_information(
    baseline: Sequence[Mapping[str, float]],
    informed: Sequence[Mapping[str, float]],
    *,
    change: InformationChange,
    selection_treatment: str = "retain_all_pairwise_range",
) -> ValueOfInformationResult:
    if not isinstance(change, InformationChange):
        raise TypeError("change must be an InformationChange")
    by_player = _strategic_values(
        baseline, informed, selection_treatment=selection_treatment
    )
    return ValueOfInformationResult(
        change,
        by_player,
        all(item.minimum_difference >= 0.0 for item in by_player.values()),
    )


def value_of_commitment(
    baseline: Sequence[Mapping[str, float]],
    transformed: Sequence[Mapping[str, float]],
    *,
    mode: CommitmentMode,
    observable: bool,
    binding: bool,
    selection_treatment: str = "retain_all_pairwise_range",
) -> ValueOfCommitmentResult:
    if not isinstance(mode, CommitmentMode):
        raise TypeError("mode must be a CommitmentMode")
    if not isinstance(observable, bool) or not isinstance(binding, bool):
        raise TypeError("observable and binding must be booleans")
    is_commitment = observable and binding and mode in {
        CommitmentMode.ONE_ACTION,
        CommitmentMode.COMPLETE_CONTINGENT_POLICY,
        CommitmentMode.BINDING_PUBLIC_COMMITMENT,
    }
    if mode is CommitmentMode.OBSERVABLE_NONBINDING_ANNOUNCEMENT and (
        binding or not observable
    ):
        raise ValueError(
            "a nonbinding announcement must be observable and cannot be binding"
        )
    return ValueOfCommitmentResult(
        mode,
        observable,
        binding,
        _strategic_values(
            baseline, transformed, selection_treatment=selection_treatment
        ),
        is_commitment,
    )
