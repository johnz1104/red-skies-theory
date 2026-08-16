"""Exact deviation diagnostics and an explicitly uncertified adjustment heuristic."""

from __future__ import annotations

import itertools
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cold_war_sim.core.extensive_form import BehavioralStrategy, ExtensiveFormGame
from cold_war_sim.core.normal_form import MixedStrategy, NormalFormGame, StrategyInput
from cold_war_sim.core.results import DiagnosticResult, SolverResult
from cold_war_sim.core.types import SerializableMixin, frozen_mapping

SOLVER_VERSION = "1.0.0"


@dataclass(frozen=True)
class PureNashEquilibrium(SerializableMixin):
    action_profile: Mapping[str, str]
    expected_utilities: Mapping[str, float]
    deviation_gains: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_profile", frozen_mapping(self.action_profile))
        object.__setattr__(
            self, "expected_utilities", frozen_mapping(self.expected_utilities)
        )
        object.__setattr__(
            self, "deviation_gains", frozen_mapping(self.deviation_gains)
        )


def find_pure_nash(
    game: NormalFormGame,
    *,
    tolerance: float = 1e-10,
) -> SolverResult:
    """Exhaustively enumerate all pure Nash equilibria of a two-player game."""

    started = time.perf_counter()
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    solutions: list[PureNashEquilibrium] = []
    for action_profile in game.pure_profiles():
        utilities = game.payoff(action_profile)
        gains: dict[str, float] = {}
        for player in range(2):
            opponent = 1 - player
            opponent_strategy = MixedStrategy.pure(
                game.player_ids[opponent],
                game.action_sets[opponent],
                action_profile[opponent],
            )
            _, best_value = game.best_responses(
                player, opponent_strategy, tolerance=tolerance
            )
            gains[game.player_ids[player]] = max(0.0, best_value - utilities[player])
        if max(gains.values()) <= tolerance:
            solutions.append(
                PureNashEquilibrium(
                    action_profile=dict(
                        zip(game.player_ids, action_profile, strict=True)
                    ),
                    expected_utilities=dict(
                        zip(game.player_ids, utilities, strict=True)
                    ),
                    deviation_gains=gains,
                )
            )
    solutions_tuple = tuple(solutions)
    return SolverResult(
        solver_name="pure_nash_enumerator",
        solver_version=SOLVER_VERSION,
        equilibrium_concept="pure-strategy Nash equilibrium",
        found=bool(solutions_tuple),
        solutions=solutions_tuple,
        status="SOLUTIONS_FOUND" if solutions_tuple else "NO_PURE_NASH_FOUND",
        exactness_status="EXACT_WITHIN_TOLERANCE",
        convergence_status="NOT_APPLICABLE",
        best_response_gap=max(
            (max(solution.deviation_gains.values()) for solution in solutions_tuple),
            default=None,
        ),
        off_path_belief_convention="NOT_APPLICABLE",
        runtime_seconds=time.perf_counter() - started,
        warnings=(
            () if solutions_tuple else ("mixed-strategy equilibria were not searched",)
        ),
        assumptions=("finite two-player normal-form game",),
        seed=None,
        metadata={
            "tolerance": tolerance,
            "profiles_checked": len(game.action_sets[0]) * len(game.action_sets[1]),
            "solution_count": len(solutions_tuple),
        },
    )


pure_nash = find_pure_nash


def strictly_dominant_actions(
    game: NormalFormGame,
    *,
    tolerance: float = 1e-10,
) -> dict[str, tuple[str, ...]]:
    """Return actions strictly dominating every alternative at every opponent action."""

    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    result: dict[str, tuple[str, ...]] = {}
    for player in range(2):
        actions = game.action_sets[player]
        dominant: list[str] = []
        for action_index, action in enumerate(actions):
            dominates_all = True
            for alternative_index in range(len(actions)):
                if alternative_index == action_index:
                    continue
                if player == 0:
                    differences = (
                        game.payoffs[action_index, :, 0]
                        - game.payoffs[alternative_index, :, 0]
                    )
                else:
                    differences = (
                        game.payoffs[:, action_index, 1]
                        - game.payoffs[:, alternative_index, 1]
                    )
                if not np.all(differences > tolerance):
                    dominates_all = False
                    break
            if dominates_all:
                dominant.append(action)
        result[game.player_ids[player]] = tuple(dominant)
    return result


def normal_form_nashconv(
    game: NormalFormGame,
    profile: Sequence[StrategyInput],
) -> DiagnosticResult:
    """Compute ex-ante player best-response gains and their sum (NashConv)."""

    started = time.perf_counter()
    vectors = game.validate_profile(profile)
    baseline = game.expected_payoffs(vectors)
    per_player: dict[str, dict[str, float]] = {}
    gains: list[float] = []
    for player in range(2):
        _, best_value = game.best_responses(player, vectors[1 - player])
        gain = max(0.0, best_value - baseline[player])
        gains.append(gain)
        per_player[game.player_ids[player]] = {
            "profile_expected_utility": baseline[player],
            "best_response_expected_utility": best_value,
            "best_response_gain": gain,
        }
    return DiagnosticResult(
        diagnostic_name="normal_form_nashconv",
        diagnostic_version=SOLVER_VERSION,
        definition=(
            "ex-ante player-form NashConv: sum over players of the gain from an exact "
            "pure best response to the opponent's fixed mixed strategy"
        ),
        values={
            "nashconv": sum(gains),
            "max_best_response_gain": max(gains),
        },
        per_player=per_player,
        runtime_seconds=time.perf_counter() - started,
        assumptions=("finite two-player normal-form game",),
        metadata={
            "profile": {
                game.player_ids[player]: dict(
                    zip(game.action_sets[player], vectors[player], strict=True)
                )
                for player in range(2)
            }
        },
    )


nashconv = normal_form_nashconv


def extensive_form_nashconv(
    game: ExtensiveFormGame,
    profile: BehavioralStrategy,
) -> DiagnosticResult:
    """Compute exact ex-ante player-form gains for perfect-information games.

    The supported domain is a finite perfect-information tree.  A player's
    best response is found by exhaustive enumeration of all pure contingent
    plans across that player's information sets, holding every other player's
    behavioral probabilities fixed.  Perfect information gives perfect recall,
    so a pure best response attains the behavioral best-response value.
    """

    started = time.perf_counter()
    if not game.is_perfect_information:
        raise ValueError(
            "extensive-form NashConv currently supports perfect-information games only"
        )
    game.validate_strategy(profile)
    baseline = game.expected_utilities(profile).utilities
    per_player: dict[str, dict[str, float]] = {}
    gains: list[float] = []
    for player_id in game.player_ids:
        owned_sets = tuple(
            set_id
            for set_id, information_set in game.information_set_map.items()
            if information_set.player_id == player_id
        )
        choices = tuple(
            game.information_set_map[set_id].action_ids for set_id in owned_sets
        )
        plans = itertools.product(*choices) if choices else ((),)
        best_value = -float("inf")
        for plan in plans:
            probabilities = {
                set_id: dict(distribution)
                for set_id, distribution in profile.probabilities.items()
            }
            for set_id, chosen_action in zip(owned_sets, plan, strict=True):
                probabilities[set_id] = {
                    action: float(action == chosen_action)
                    for action in game.information_set_map[set_id].action_ids
                }
            deviation_profile = BehavioralStrategy(probabilities)
            value = game.expected_utilities(deviation_profile).utilities[player_id]
            best_value = max(best_value, value)
        gain = max(0.0, best_value - baseline[player_id])
        gains.append(gain)
        per_player[player_id] = {
            "profile_expected_utility": baseline[player_id],
            "best_response_expected_utility": best_value,
            "best_response_gain": gain,
        }
    return DiagnosticResult(
        diagnostic_name="extensive_form_nashconv",
        diagnostic_version=SOLVER_VERSION,
        definition=(
            "ex-ante player-form NashConv in a perfect-information tree: sum of "
            "gains from exact exhaustive pure contingent-plan best responses"
        ),
        values={
            "nashconv": sum(gains),
            "max_best_response_gain": max(gains, default=0.0),
        },
        per_player=per_player,
        runtime_seconds=time.perf_counter() - started,
        assumptions=(
            "finite perfect-information game",
            "other players' behavioral strategies remain fixed during each deviation",
        ),
        metadata={
            "best_response_method": "exhaustive pure contingent-plan enumeration"
        },
    )


behavioral_nashconv = extensive_form_nashconv


@dataclass(frozen=True)
class DampedBehavioralBestResponseResult(SerializableMixin):
    """Output of a local adjustment diagnostic, never an equilibrium certificate."""

    algorithm_name: str
    algorithm_version: str
    final_profile: Mapping[str, Mapping[str, float]]
    iterations: int
    converged_by_gap: bool
    certified_equilibrium: bool
    final_nashconv: float
    max_best_response_gain: float
    runtime_seconds: float
    warnings: tuple[str, ...]
    assumptions: tuple[str, ...]
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.certified_equilibrium:
            raise ValueError("this heuristic cannot certify an equilibrium")
        object.__setattr__(
            self,
            "final_profile",
            frozen_mapping(
                {
                    player: frozen_mapping(distribution)
                    for player, distribution in self.final_profile.items()
                }
            ),
        )

    @property
    def status(self) -> str:
        return (
            "HEURISTIC_GAP_TOLERANCE_REACHED"
            if self.converged_by_gap
            else "HEURISTIC_MAX_ITERATIONS_REACHED"
        )


def damped_behavioral_best_response(
    game: NormalFormGame,
    initial_profile: Sequence[StrategyInput] | None = None,
    *,
    damping: float = 0.2,
    max_iterations: int = 1_000,
    tolerance: float = 1e-8,
) -> DampedBehavioralBestResponseResult:
    """Run simultaneous damped local best-response updates.

    This deterministic routine is useful as a behavioral adjustment diagnostic.
    Its stopping rule is not a sequential-equilibrium certificate (nor a
    general Nash-equilibrium certificate); exact solvers should support any
    equilibrium claim.
    """

    started = time.perf_counter()
    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must lie in (0, 1]")
    if isinstance(max_iterations, bool) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if initial_profile is None:
        vectors = tuple(
            np.ones(len(actions), dtype=float) / len(actions)
            for actions in game.action_sets
        )
    else:
        vectors = game.validate_profile(initial_profile)
    current = [vectors[0].copy(), vectors[1].copy()]
    converged = False
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        diagnostic = normal_form_nashconv(game, current)
        if diagnostic.values["nashconv"] <= tolerance:
            converged = True
            break
        targets: list[np.ndarray] = []
        for player in range(2):
            responses, _ = game.best_responses(player, current[1 - player])
            target = np.zeros(len(game.action_sets[player]), dtype=float)
            for action in responses:
                target[game.action_sets[player].index(action)] = 1.0 / len(responses)
            targets.append(target)
        current = [
            (1.0 - damping) * current[player] + damping * targets[player]
            for player in range(2)
        ]
    final_diagnostic = normal_form_nashconv(game, current)
    converged = converged or final_diagnostic.values["nashconv"] <= tolerance
    final_profile = {
        game.player_ids[player]: {
            action: float(current[player][action_index])
            for action_index, action in enumerate(game.action_sets[player])
        }
        for player in range(2)
    }
    return DampedBehavioralBestResponseResult(
        algorithm_name="damped_behavioral_best_response",
        algorithm_version=SOLVER_VERSION,
        final_profile=final_profile,
        iterations=iterations,
        converged_by_gap=converged,
        certified_equilibrium=False,
        final_nashconv=final_diagnostic.values["nashconv"],
        max_best_response_gain=final_diagnostic.values["max_best_response_gain"],
        runtime_seconds=time.perf_counter() - started,
        warnings=(
            "local damped best-response adjustment does not certify sequential equilibrium",
            "simultaneous best-response dynamics can cycle or converge to non-equilibrium behavior",
        ),
        assumptions=("deterministic uniform tie-breaking among pure best responses",),
        seed=None,
    )


def solve_sequential_equilibrium(
    *args: Any, **kwargs: Any
) -> DampedBehavioralBestResponseResult:
    """Deprecated compatibility wrapper for a historically misnamed heuristic."""

    warnings.warn(
        "solve_sequential_equilibrium was misnamed and is deprecated; it delegates to "
        "damped_behavioral_best_response and does not certify sequential equilibrium",
        DeprecationWarning,
        stacklevel=2,
    )
    return damped_behavioral_best_response(*args, **kwargs)
