"""Exact finite response calculations and equilibrium certificates."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from cold_war_sim.core.extensive_form import BehavioralStrategy, ExtensiveFormGame
from cold_war_sim.core.normal_form import NormalFormGame
from cold_war_sim.core.types import (
    SerializableMixin,
    deep_frozen_json,
    frozen_mapping,
    validate_stable_id,
)
from cold_war_sim.solvers.best_response import find_pure_nash

from .specs import SolutionConcept, StackelbergTieBreaking


@dataclass(frozen=True)
class EquilibriumCertificate(SerializableMixin):
    equilibrium_concept: str
    candidate_class: str
    strategy_profile: Mapping[str, object]
    beliefs: Mapping[str, Mapping[str, float]]
    reach_probabilities: Mapping[str, float]
    deviation_gains: Mapping[str, float]
    best_response_gap: float
    off_path_convention: str
    exact: bool
    tolerance: float
    warnings: tuple[str, ...] = ()
    depends_on_off_path_beliefs: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.equilibrium_concept, str):
            raise TypeError("equilibrium_concept must be a string")
        supported_concepts = {
            SolutionConcept.BACKWARD_INDUCTION.value,
            SolutionConcept.PURE_NASH.value,
            SolutionConcept.SUBGAME_PERFECT.value,
            SolutionConcept.SUBGAME_PERFECT_EQUILIBRIUM.value,
            SolutionConcept.PURE_PBE.value,
            SolutionConcept.OPEN_LOOP_NASH.value,
        }
        if self.equilibrium_concept not in supported_concepts:
            raise ValueError(
                f"unsupported certified equilibrium concept {self.equilibrium_concept!r}"
            )
        if not isinstance(self.candidate_class, str) or not self.candidate_class.strip():
            raise ValueError("candidate_class must not be empty")
        if not isinstance(self.exact, bool):
            raise TypeError("exact must be a boolean")
        if not isinstance(self.depends_on_off_path_beliefs, bool):
            raise TypeError("depends_on_off_path_beliefs must be a boolean")
        lowered = self.candidate_class.lower()
        if self.exact and ("mixed" in lowered or "behavioral" in lowered):
            raise ValueError(
                "this release does not issue exact mixed or behavioral certificates"
            )
        tolerance = float(self.tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("certificate tolerance must be finite and positive")
        gap = float(self.best_response_gap)
        if not math.isfinite(gap) or gap < 0.0:
            raise ValueError("best-response gap must be finite and nonnegative")
        gains = {}
        for player, value in self.deviation_gains.items():
            validate_stable_id(player, field_name="deviation-gain player id")
            gains[player] = float(value)
        if any(not math.isfinite(value) or value < 0.0 for value in gains.values()):
            raise ValueError("deviation gains must be finite and nonnegative")
        if self.exact and not gains:
            raise ValueError(
                "an exact equilibrium certificate must report deviation gains"
            )
        expected_gap = max(gains.values(), default=0.0)
        if not math.isclose(gap, expected_gap, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError("best-response gap must equal the maximum deviation gain")
        if self.exact and gap > tolerance:
            raise ValueError("an exact equilibrium certificate cannot have a profitable deviation")
        reaches = {}
        for key, value in self.reach_probabilities.items():
            validate_stable_id(key, field_name="reach-probability id")
            reaches[key] = float(value)
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in reaches.values()
        ):
            raise ValueError("reach probabilities must be finite and lie in [0, 1]")
        beliefs = {}
        for identifier, distribution in self.beliefs.items():
            validate_stable_id(identifier, field_name="belief information-set id")
            converted = {}
            for key, value in distribution.items():
                validate_stable_id(key, field_name="belief state id")
                converted[key] = float(value)
            if not converted or any(
                not math.isfinite(value) or value < 0.0 for value in converted.values()
            ):
                raise ValueError("belief distributions must be finite and nonnegative")
            if not math.isclose(
                sum(converted.values()), 1.0, rel_tol=0.0, abs_tol=tolerance
            ):
                raise ValueError("belief distributions must sum to one")
            beliefs[identifier] = frozen_mapping(converted)
        if (
            not isinstance(self.off_path_convention, str)
            or not self.off_path_convention.strip()
        ):
            raise ValueError("off_path_convention must not be empty")
        if self.depends_on_off_path_beliefs and self.off_path_convention == "NOT_APPLICABLE":
            raise ValueError(
                "off-path belief dependence requires a stated off-path convention"
            )
        object.__setattr__(self, "strategy_profile", deep_frozen_json(self.strategy_profile))
        object.__setattr__(
            self,
            "beliefs",
            frozen_mapping(beliefs),
        )
        object.__setattr__(self, "reach_probabilities", frozen_mapping(reaches))
        object.__setattr__(self, "deviation_gains", frozen_mapping(gains))
        object.__setattr__(self, "best_response_gap", gap)
        object.__setattr__(self, "tolerance", tolerance)
        warnings = tuple(self.warnings)
        if any(not isinstance(warning, str) or not warning.strip() for warning in warnings):
            raise ValueError("certificate warnings must be nonempty strings")
        object.__setattr__(self, "warnings", warnings)


@dataclass(frozen=True)
class BestResponseResult(SerializableMixin):
    player_id: str
    baseline_utility: float
    best_response_utility: float
    gain: float
    policies: tuple[Mapping[str, str], ...]
    exact: bool = True

    def __post_init__(self) -> None:
        validate_stable_id(self.player_id, field_name="best-response player id")
        if not isinstance(self.exact, bool):
            raise TypeError("exact must be a boolean")
        baseline = float(self.baseline_utility)
        best = float(self.best_response_utility)
        gain = float(self.gain)
        if not all(math.isfinite(value) for value in (baseline, best, gain)):
            raise ValueError("best-response utilities and gain must be finite")
        expected_gain = max(0.0, best - baseline)
        if not math.isclose(gain, expected_gain, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("best-response gain must match the reported utilities")
        if not self.policies:
            raise ValueError("a best-response result must retain at least one policy")
        converted_policies = []
        for policy in self.policies:
            converted = {}
            for information_set, action in policy.items():
                validate_stable_id(
                    information_set, field_name="best-response information-set id"
                )
                validate_stable_id(action, field_name="best-response action id")
                converted[information_set] = action
            converted_policies.append(frozen_mapping(converted))
        if len({tuple(policy.items()) for policy in converted_policies}) != len(
            converted_policies
        ):
            raise ValueError("best-response policies must be unique")
        object.__setattr__(self, "baseline_utility", baseline)
        object.__setattr__(self, "best_response_utility", best)
        object.__setattr__(self, "gain", gain)
        object.__setattr__(
            self,
            "policies",
            tuple(sorted(converted_policies, key=lambda policy: tuple(policy.items()))),
        )


@dataclass(frozen=True)
class BestResponseCapacityResult(SerializableMixin):
    player_id: str
    status: str
    estimated_policy_count: int
    maximum_policy_count: int
    narrower_legal_configuration: Mapping[str, object]

    def __post_init__(self) -> None:
        validate_stable_id(self.player_id, field_name="best-response player id")
        if self.status != "CAPACITY_EXCEEDED":
            raise ValueError("capacity result status must be CAPACITY_EXCEEDED")
        for name in ("estimated_policy_count", "maximum_policy_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.estimated_policy_count <= self.maximum_policy_count:
            raise ValueError(
                "a capacity result requires estimated_policy_count to exceed the maximum"
            )
        narrowed = deep_frozen_json(self.narrower_legal_configuration)
        if not isinstance(narrowed, Mapping):
            raise TypeError("narrower_legal_configuration must be a mapping")
        narrowed_size = narrowed.get("estimated_size")
        if (
            isinstance(narrowed_size, bool)
            or not isinstance(narrowed_size, int)
            or not 1 <= narrowed_size <= self.maximum_policy_count
        ):
            raise ValueError(
                "narrower configuration estimated_size must be within capacity"
            )
        object.__setattr__(
            self,
            "narrower_legal_configuration",
            narrowed,
        )


def extensive_form_best_response(
    game: ExtensiveFormGame,
    profile: BehavioralStrategy,
    player_id: str,
    *,
    tolerance: float = 1e-9,
    maximum_policy_count: int = 100_000,
) -> BestResponseResult | BestResponseCapacityResult:
    """Enumerate pure contingent plans while all other behavior remains fixed."""

    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if (
        isinstance(maximum_policy_count, bool)
        or not isinstance(maximum_policy_count, int)
        or maximum_policy_count < 1
    ):
        raise ValueError("maximum_policy_count must be a positive integer")
    game.validate_strategy(profile)
    if player_id not in game.player_ids:
        raise ValueError("unknown player")
    owned = tuple(
        identifier
        for identifier, info in game.information_set_map.items()
        if info.player_id == player_id
    )
    choices = tuple(game.information_set_map[key].action_ids for key in owned)
    policy_count = math.prod(len(actions) for actions in choices)
    if policy_count > maximum_policy_count:
        variable_information_sets: list[str] = []
        fixed: dict[str, str] = {}
        narrowed_size = 1
        for identifier, actions in zip(owned, choices, strict=True):
            if narrowed_size * len(actions) <= maximum_policy_count:
                variable_information_sets.append(identifier)
                narrowed_size *= len(actions)
            else:
                fixed[identifier] = actions[0]
        narrower: dict[str, object] = {
            "fixed_actions": fixed,
            "estimated_size": narrowed_size,
        }
        if variable_information_sets:
            narrower["allowed_information_sets"] = variable_information_sets
        return BestResponseCapacityResult(
            player_id,
            "CAPACITY_EXCEEDED",
            policy_count,
            maximum_policy_count,
            narrower,
        )
    plans = itertools.product(*choices) if choices else ((),)
    baseline = game.expected_utilities(profile).utilities[player_id]
    best = -float("inf")
    retained: list[Mapping[str, str]] = []
    for plan in plans:
        probabilities = {
            key: dict(distribution)
            for key, distribution in profile.probabilities.items()
        }
        policy = dict(zip(owned, plan, strict=True))
        for key, selected in policy.items():
            probabilities[key] = {
                action: float(action == selected)
                for action in game.information_set_map[key].action_ids
            }
        value = game.expected_utilities(BehavioralStrategy(probabilities)).utilities[player_id]
        if value > best + tolerance:
            best = value
            retained = [policy]
        elif abs(value - best) <= tolerance:
            retained.append(policy)
    retained.sort(key=lambda policy: tuple(policy.items()))
    return BestResponseResult(
        player_id,
        baseline,
        best,
        max(0.0, best - baseline),
        tuple(retained),
    )


@dataclass(frozen=True)
class StackelbergOutcome(SerializableMixin):
    leader_action: str
    follower_action: str
    leader_utility: float
    follower_utility: float

    def __post_init__(self) -> None:
        validate_stable_id(self.leader_action, field_name="leader action id")
        validate_stable_id(self.follower_action, field_name="follower action id")
        leader_utility = float(self.leader_utility)
        follower_utility = float(self.follower_utility)
        if not math.isfinite(leader_utility) or not math.isfinite(follower_utility):
            raise ValueError("Stackelberg utilities must be finite")
        object.__setattr__(self, "leader_utility", leader_utility)
        object.__setattr__(self, "follower_utility", follower_utility)


@dataclass(frozen=True)
class StackelbergResult(SerializableMixin):
    leader_id: str
    follower_id: str
    tie_breaking: StackelbergTieBreaking
    supported_outcomes: tuple[StackelbergOutcome, ...]
    all_follower_best_responses: Mapping[str, tuple[str, ...]]
    selection_treatment: str
    exact: bool = True

    def __post_init__(self) -> None:
        validate_stable_id(self.leader_id, field_name="Stackelberg leader id")
        validate_stable_id(self.follower_id, field_name="Stackelberg follower id")
        if self.leader_id == self.follower_id:
            raise ValueError("Stackelberg leader and follower must be distinct")
        if not isinstance(self.tie_breaking, StackelbergTieBreaking):
            raise TypeError("tie_breaking must be a StackelbergTieBreaking value")
        if not isinstance(self.exact, bool):
            raise TypeError("exact must be a boolean")
        outcomes = tuple(self.supported_outcomes)
        if not outcomes:
            raise ValueError("a Stackelberg result must retain supported outcomes")
        responses: dict[str, tuple[str, ...]] = {}
        for leader_action, actions in self.all_follower_best_responses.items():
            validate_stable_id(leader_action, field_name="leader action id")
            converted = tuple(actions)
            if not converted or len(converted) != len(set(converted)):
                raise ValueError(
                    "each leader action requires unique follower best responses"
                )
            for action in converted:
                validate_stable_id(action, field_name="follower action id")
            responses[leader_action] = tuple(sorted(converted))
        if not responses:
            raise ValueError("follower best-response correspondence must not be empty")
        for outcome in outcomes:
            if outcome.leader_action not in responses:
                raise ValueError("supported outcome references an unknown leader action")
            if outcome.follower_action not in responses[outcome.leader_action]:
                raise ValueError(
                    "supported outcome must use a follower best response"
                )
        if not isinstance(self.selection_treatment, str) or not self.selection_treatment.strip():
            raise ValueError("selection_treatment must not be empty")
        object.__setattr__(self, "supported_outcomes", outcomes)
        object.__setattr__(
            self, "all_follower_best_responses", frozen_mapping(responses)
        )


def solve_stackelberg(
    game: NormalFormGame,
    *,
    leader_id: str,
    tie_breaking: StackelbergTieBreaking,
    tolerance: float = 1e-9,
) -> StackelbergResult:
    """Solve a pure-action commitment game with explicit follower tie treatment."""

    if not isinstance(tie_breaking, StackelbergTieBreaking):
        raise TypeError("tie_breaking must be a StackelbergTieBreaking value")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if len(game.player_ids) != 2:
        raise ValueError("the pure Stackelberg solver supports exactly two players")
    if leader_id not in game.player_ids:
        raise ValueError("unknown leader")
    leader_index = game.player_ids.index(leader_id)
    follower_index = 1 - leader_index
    follower_id = game.player_ids[follower_index]
    action_sets = game.action_sets
    response_map: dict[str, tuple[str, ...]] = {}
    induced: list[StackelbergOutcome] = []
    for leader_action in action_sets[leader_index]:
        values: list[tuple[str, float, float]] = []
        for follower_action in action_sets[follower_index]:
            profile = (
                (leader_action, follower_action)
                if leader_index == 0
                else (follower_action, leader_action)
            )
            payoff = game.payoff(profile)
            values.append((follower_action, payoff[leader_index], payoff[follower_index]))
        best_follower = max(value[2] for value in values)
        responses = tuple(value[0] for value in values if value[2] >= best_follower - tolerance)
        response_map[leader_action] = responses
        tied = [value for value in values if value[0] in responses]
        if tie_breaking is StackelbergTieBreaking.STRONG_STACKELBERG:
            target = max(value[1] for value in tied)
            tied = [value for value in tied if value[1] >= target - tolerance]
        elif tie_breaking is StackelbergTieBreaking.WEAK_STACKELBERG:
            target = min(value[1] for value in tied)
            tied = [value for value in tied if value[1] <= target + tolerance]
        for follower_action, leader_value, follower_value in tied:
            induced.append(
                StackelbergOutcome(
                    leader_action, follower_action, leader_value, follower_value
                )
            )
    if tie_breaking is StackelbergTieBreaking.ALL_FOLLOWER_BEST_RESPONSES:
        # A set-valued follower response does not define a unique leader optimum.
        # Retain the complete correspondence instead of silently applying the
        # optimistic (strong) or pessimistic (weak) convention.
        supported = tuple(
            sorted(induced, key=lambda item: (item.leader_action, item.follower_action))
        )
        selection_treatment = (
            "all follower best responses retained for every leader commitment; "
            "no leader action selected"
        )
    else:
        best_leader = max(outcome.leader_utility for outcome in induced)
        supported = tuple(
            sorted(
                (
                    outcome
                    for outcome in induced
                    if outcome.leader_utility >= best_leader - tolerance
                ),
                key=lambda item: (item.leader_action, item.follower_action),
            )
        )
        selection_treatment = (
            "follower ties favor the leader before leader optimization"
            if tie_breaking is StackelbergTieBreaking.STRONG_STACKELBERG
            else "follower ties disfavor the leader before leader optimization"
        )
    return StackelbergResult(
        leader_id,
        follower_id,
        tie_breaking,
        supported,
        response_map,
        selection_treatment,
    )


def pure_nash_certificates(
    game: NormalFormGame, *, tolerance: float = 1e-9
) -> tuple[EquilibriumCertificate, ...]:
    result = find_pure_nash(game, tolerance=tolerance)
    return tuple(
        EquilibriumCertificate(
            SolutionConcept.PURE_NASH.value,
            "pure strategies",
            solution.action_profile,
            {},
            {},
            solution.deviation_gains,
            max(solution.deviation_gains.values(), default=0.0),
            "NOT_APPLICABLE",
            True,
            tolerance,
            result.warnings,
        )
        for solution in result.solutions
    )


DownstreamResolver = Callable[[object], object]


def downstream_best_responses(transformed_game: object, resolver: DownstreamResolver) -> object:
    """Invoke an event-justified continuation resolver; no global claim is added."""

    return resolver(transformed_game)
