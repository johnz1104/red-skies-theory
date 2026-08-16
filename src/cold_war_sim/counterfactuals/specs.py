"""Typed counterfactual specifications with explicit strategic assumptions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from cold_war_sim.core.types import (
    SerializableMixin,
    deep_frozen_json,
    to_serializable,
    validate_stable_id,
)

if TYPE_CHECKING:
    from .constraints import Constraint
    from .interventions import Intervention
    from .objectives import Objective


class ResponseModel(StrEnum):
    FROZEN_OPPONENTS = "FROZEN_OPPONENTS"
    UNILATERAL_BEST_RESPONSE = "UNILATERAL_BEST_RESPONSE"
    DOWNSTREAM_BEST_RESPONSES = "DOWNSTREAM_BEST_RESPONSES"
    REEQUILIBRATE = "REEQUILIBRATE"
    STACKELBERG_COMMITMENT = "STACKELBERG_COMMITMENT"


class SolutionConcept(StrEnum):
    SUPPLIED_PROFILE = "SUPPLIED_PROFILE"
    BACKWARD_INDUCTION = "BACKWARD_INDUCTION"
    PURE_NASH = "PURE_NASH"
    SUBGAME_PERFECT = "SUBGAME_PERFECT"
    SUBGAME_PERFECT_EQUILIBRIUM = "SUBGAME_PERFECT_EQUILIBRIUM"
    PURE_PBE = "PURE_PBE"
    OPEN_LOOP_NASH = "OPEN_LOOP_NASH"


class EquilibriumSelectionRule(StrEnum):
    RETAIN_ALL = "RETAIN_ALL"
    PLAYER_OPTIMAL = "PLAYER_OPTIMAL"
    PLAYER_PESSIMAL = "PLAYER_PESSIMAL"
    SOCIAL_WELFARE = "SOCIAL_WELFARE"
    MINIMUM_CATASTROPHE = "MINIMUM_CATASTROPHE"
    CUSTOM = "CUSTOM"


class StackelbergTieBreaking(StrEnum):
    STRONG_STACKELBERG = "STRONG_STACKELBERG"
    WEAK_STACKELBERG = "WEAK_STACKELBERG"
    ALL_FOLLOWER_BEST_RESPONSES = "ALL_FOLLOWER_BEST_RESPONSES"


class CounterfactualStatus(StrEnum):
    VALID_COUNTERFACTUAL = "VALID_COUNTERFACTUAL"
    INFEASIBLE_POLICY = "INFEASIBLE_POLICY"
    CAPACITY_EXCEEDED = "CAPACITY_EXCEEDED"
    NO_SUPPORTED_SOLUTION = "NO_SUPPORTED_SOLUTION"
    NO_PURE_PBE_FOUND = "NO_PURE_PBE_FOUND"
    MULTIPLE_SUPPORTED_EQUILIBRIA = "MULTIPLE_SUPPORTED_EQUILIBRIA"
    DEPENDENT_ON_OFF_PATH_BELIEFS = "DEPENDENT_ON_OFF_PATH_BELIEFS"
    UNVERIFIED_APPROXIMATION = "UNVERIFIED_APPROXIMATION"


class ReachabilityStatus(StrEnum):
    MODEL_REACHABLE = "MODEL_REACHABLE"
    STRATEGY_FEASIBLE = "STRATEGY_FEASIBLE"
    EQUILIBRIUM_SUPPORTED = "EQUILIBRIUM_SUPPORTED"
    ROBUSTLY_PREFERRED = "ROBUSTLY_PREFERRED"


@dataclass(frozen=True)
class ScenarioReference(SerializableMixin):
    event: str
    configuration_id: str
    configuration: Mapping[str, Any]
    seed_policy: str = "deterministic_no_random_draws"

    def __post_init__(self) -> None:
        validate_stable_id(self.event, field_name="event")
        validate_stable_id(self.configuration_id, field_name="configuration id")
        validate_stable_id(self.seed_policy, field_name="seed policy")
        to_serializable(self.configuration)
        object.__setattr__(self, "configuration", deep_frozen_json(self.configuration))


@dataclass(frozen=True)
class ParameterScenario(SerializableMixin):
    id: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        validate_stable_id(self.id, field_name="parameter-scenario id")
        to_serializable(self.parameters)
        object.__setattr__(self, "parameters", deep_frozen_json(self.parameters))


@dataclass(frozen=True)
class UncertaintySet(SerializableMixin):
    scenarios: tuple[ParameterScenario, ...]
    aggregation: str = "ALL"

    def __post_init__(self) -> None:
        scenarios = tuple(sorted(self.scenarios, key=lambda item: item.id))
        if not scenarios:
            raise ValueError("an uncertainty set must contain at least one scenario")
        ids = tuple(item.id for item in scenarios)
        if len(ids) != len(set(ids)):
            raise ValueError("parameter-scenario ids must be unique")
        if self.aggregation not in {"ALL", "WORST_CASE", "MAXIMIN"}:
            raise ValueError("uncertainty aggregation must be ALL, WORST_CASE, or MAXIMIN")
        object.__setattr__(self, "scenarios", scenarios)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_scenarios": [scenario.to_dict() for scenario in self.scenarios],
            "aggregation": self.aggregation,
        }


@dataclass(frozen=True)
class EquilibriumHandling(SerializableMixin):
    selection_rule: EquilibriumSelectionRule
    focal_player: str | None = None
    stackelberg_tie_breaking: StackelbergTieBreaking | None = None
    custom_rule_id: str | None = None

    def __post_init__(self) -> None:
        if self.focal_player is not None:
            validate_stable_id(self.focal_player, field_name="focal player")
        if self.selection_rule is EquilibriumSelectionRule.CUSTOM:
            if self.custom_rule_id is None:
                raise ValueError("CUSTOM equilibrium selection requires custom_rule_id")
            validate_stable_id(self.custom_rule_id, field_name="custom rule id")
        elif self.custom_rule_id is not None:
            raise ValueError("custom_rule_id is only valid with CUSTOM selection")
        player_rules = {
            EquilibriumSelectionRule.PLAYER_OPTIMAL,
            EquilibriumSelectionRule.PLAYER_PESSIMAL,
        }
        if self.selection_rule in player_rules and self.focal_player is None:
            raise ValueError("player-specific equilibrium selection requires focal_player")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"selection_rule": self.selection_rule.value}
        if self.focal_player is not None:
            result["player"] = self.focal_player
        if self.stackelberg_tie_breaking is not None:
            result["stackelberg_tie_breaking"] = self.stackelberg_tie_breaking.value
        if self.custom_rule_id is not None:
            result["custom_rule"] = self.custom_rule_id
        return result


@dataclass(frozen=True)
class CounterfactualSpec(SerializableMixin):
    baseline: ScenarioReference
    intervention: Intervention
    response_model: ResponseModel
    solution_concept: SolutionConcept
    objective: Objective
    constraints: tuple[Constraint, ...]
    uncertainty_set: UncertaintySet | None
    equilibrium_handling: EquilibriumHandling
    tolerance: float = 1e-9
    schema_version: str = "1.0"
    document_type: str = "counterfactual_spec"

    def __post_init__(self) -> None:
        tolerance = float(self.tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")
        if (
            self.response_model is ResponseModel.STACKELBERG_COMMITMENT
            and self.equilibrium_handling.stackelberg_tie_breaking is None
        ):
            raise ValueError("Stackelberg analysis requires an explicit tie-breaking rule")
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "tolerance", tolerance)
        if self.schema_version != "1.0":
            raise ValueError("unsupported counterfactual specification schema version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "document_type": self.document_type,
            "baseline": self.baseline.to_dict(),
            "intervention": self.intervention.to_dict(),
            "response_model": self.response_model.value,
            "solution_concept": self.solution_concept.value,
            "objective": self.objective.to_dict(),
            "constraints": [constraint.to_dict() for constraint in self.constraints],
            "uncertainty_set": (
                None if self.uncertainty_set is None else self.uncertainty_set.to_dict()
            ),
            "equilibrium_handling": self.equilibrium_handling.to_dict(),
            "tolerance": self.tolerance,
        }
