"""Side-by-side comparison without silently selecting one equilibrium."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from cold_war_sim.core.types import (
    SerializableMixin,
    deep_frozen_json,
    frozen_mapping,
    validate_stable_id,
)

from .outcomes import OutcomeDistribution
from .pareto import pareto_dominates
from .responses import EquilibriumCertificate
from .specs import EquilibriumSelectionRule


@dataclass(frozen=True)
class EquilibriumRecord(SerializableMixin):
    id: str
    classification: str
    strategy_profile: Mapping[str, object]
    expected_utilities: Mapping[str, float]
    outcome_distribution: OutcomeDistribution
    information_set_reach: Mapping[str, float]
    certificate: EquilibriumCertificate
    distance_from_baseline: int

    def __post_init__(self) -> None:
        validate_stable_id(self.id, field_name="equilibrium id")
        if not isinstance(self.classification, str) or not self.classification.strip():
            raise ValueError("equilibrium classification must not be empty")
        if (
            isinstance(self.distance_from_baseline, bool)
            or not isinstance(self.distance_from_baseline, int)
            or self.distance_from_baseline < 0
        ):
            raise ValueError("distance from baseline must be a nonnegative integer")
        if not self.expected_utilities:
            raise ValueError("equilibrium utilities must not be empty")
        utilities = {}
        for player, value in self.expected_utilities.items():
            validate_stable_id(player, field_name="equilibrium-utility player id")
            utilities[player] = float(value)
        if any(not math.isfinite(value) for value in utilities.values()):
            raise ValueError("equilibrium utilities must be finite")
        if set(self.certificate.deviation_gains) != set(utilities):
            raise ValueError(
                "certificate deviation gains must cover the equilibrium players"
            )
        strategy_profile = deep_frozen_json(self.strategy_profile)
        if strategy_profile != self.certificate.strategy_profile:
            raise ValueError(
                "equilibrium record strategy must match its certificate profile"
            )
        reaches = {}
        for identifier, value in self.information_set_reach.items():
            validate_stable_id(identifier, field_name="information-set reach id")
            reaches[identifier] = float(value)
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in reaches.values()
        ):
            raise ValueError("information-set reach must lie in [0, 1]")
        object.__setattr__(self, "strategy_profile", strategy_profile)
        object.__setattr__(self, "expected_utilities", frozen_mapping(utilities))
        object.__setattr__(self, "information_set_reach", frozen_mapping(reaches))

    @property
    def escalation_probability(self) -> float:
        return self.outcome_distribution.probability("military_escalation")

    @property
    def catastrophe_probability(self) -> float:
        return self.outcome_distribution.probability("catastrophic_escalation")

    @property
    def negotiated_settlement_probability(self) -> float:
        return self.outcome_distribution.probability("negotiated_agreement")


@dataclass(frozen=True)
class EquilibriumComparison(SerializableMixin):
    equilibria: tuple[EquilibriumRecord, ...]
    pareto_dominance: Mapping[str, tuple[str, ...]]
    selection_rule: EquilibriumSelectionRule
    selected_ids: tuple[str, ...]
    multiplicity: int

    def __post_init__(self) -> None:
        if not isinstance(self.selection_rule, EquilibriumSelectionRule):
            raise TypeError("selection_rule must be an EquilibriumSelectionRule")
        equilibria = tuple(self.equilibria)
        ids = tuple(item.id for item in equilibria)
        if len(ids) != len(set(ids)):
            raise ValueError("equilibrium ids must be unique")
        if (
            isinstance(self.multiplicity, bool)
            or not isinstance(self.multiplicity, int)
            or self.multiplicity != len(equilibria)
        ):
            raise ValueError("multiplicity must equal the equilibrium count")
        selected = tuple(self.selected_ids)
        if len(selected) != len(set(selected)) or not set(selected) <= set(ids):
            raise ValueError("selected equilibrium ids must be unique and known")
        dominance: dict[str, tuple[str, ...]] = {}
        if set(self.pareto_dominance) != set(ids):
            raise ValueError("Pareto-dominance keys must cover every equilibrium")
        for identifier, dominated_ids in self.pareto_dominance.items():
            values = tuple(dominated_ids)
            if (
                identifier in values
                or len(values) != len(set(values))
                or not set(values) <= set(ids)
            ):
                raise ValueError("Pareto-dominance ids must be unique, known, and nonreflexive")
            dominance[identifier] = tuple(sorted(values))
        if self.selection_rule is EquilibriumSelectionRule.RETAIN_ALL and set(selected) != set(ids):
            raise ValueError("RETAIN_ALL must select every retained equilibrium")
        object.__setattr__(self, "equilibria", equilibria)
        object.__setattr__(self, "selected_ids", selected)
        object.__setattr__(self, "pareto_dominance", frozen_mapping(dominance))


def compare_equilibria(
    equilibria: Sequence[EquilibriumRecord],
    *,
    selection_rule: EquilibriumSelectionRule,
    focal_player: str | None = None,
    tolerance: float = 1e-9,
) -> EquilibriumComparison:
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if not isinstance(selection_rule, EquilibriumSelectionRule):
        raise TypeError("selection_rule must be an EquilibriumSelectionRule")
    ordered = tuple(sorted(equilibria, key=lambda item: item.id))
    ids = tuple(item.id for item in ordered)
    if len(ids) != len(set(ids)):
        raise ValueError("equilibrium ids must be unique")
    if not ordered:
        return EquilibriumComparison((), {}, selection_rule, (), 0)
    players = set(ordered[0].expected_utilities)
    if any(set(item.expected_utilities) != players for item in ordered[1:]):
        raise ValueError("all equilibria must report the same players")
    dominance = {
        item.id: tuple(
            other.id
            for other in ordered
            if other.id != item.id
            and pareto_dominates(item.expected_utilities, other.expected_utilities, tolerance=tolerance)
        )
        for item in ordered
    }
    if selection_rule is EquilibriumSelectionRule.RETAIN_ALL:
        selected = ordered
    elif selection_rule in {
        EquilibriumSelectionRule.PLAYER_OPTIMAL,
        EquilibriumSelectionRule.PLAYER_PESSIMAL,
    }:
        if focal_player is None:
            raise ValueError("player selection requires focal_player")
        if focal_player not in players:
            raise ValueError("focal_player is absent from equilibrium utilities")
        values = [item.expected_utilities[focal_player] for item in ordered]
        target = max(values) if selection_rule is EquilibriumSelectionRule.PLAYER_OPTIMAL else min(values)
        selected = tuple(
            item
            for item in ordered
            if math.isclose(
                item.expected_utilities[focal_player],
                target,
                rel_tol=0.0,
                abs_tol=tolerance,
            )
        )
    elif selection_rule is EquilibriumSelectionRule.SOCIAL_WELFARE:
        target = max(sum(item.expected_utilities.values()) for item in ordered)
        selected = tuple(
            item
            for item in ordered
            if math.isclose(
                sum(item.expected_utilities.values()),
                target,
                rel_tol=0.0,
                abs_tol=tolerance,
            )
        )
    elif selection_rule is EquilibriumSelectionRule.MINIMUM_CATASTROPHE:
        target = min(item.catastrophe_probability for item in ordered)
        selected = tuple(
            item
            for item in ordered
            if math.isclose(
                item.catastrophe_probability,
                target,
                rel_tol=0.0,
                abs_tol=tolerance,
            )
        )
    else:
        raise ValueError("CUSTOM selection requires an external selector")
    return EquilibriumComparison(
        ordered,
        dominance,
        selection_rule,
        tuple(item.id for item in selected),
        len(ordered),
    )
