"""Exact backward induction for finite perfect-information games."""

from __future__ import annotations

import itertools
import time
from collections.abc import Mapping
from dataclasses import dataclass

from cold_war_sim.core.extensive_form import (
    ChanceNode,
    DecisionNode,
    ExtensiveFormGame,
    TerminalNode,
)
from cold_war_sim.core.results import SolverResult
from cold_war_sim.core.types import SerializableMixin, canonical_json, frozen_mapping

SOLVER_NAME = "backward_induction"
SOLVER_VERSION = "1.0.0"


@dataclass(frozen=True)
class BackwardInductionSolution(SerializableMixin):
    """One complete pure subgame-perfect strategy profile."""

    strategy_profile: Mapping[str, str]
    expected_utilities: Mapping[str, float]
    node_values: Mapping[str, Mapping[str, float]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "strategy_profile", frozen_mapping(self.strategy_profile)
        )
        object.__setattr__(
            self, "expected_utilities", frozen_mapping(self.expected_utilities)
        )
        object.__setattr__(
            self,
            "node_values",
            frozen_mapping(
                {
                    node_id: frozen_mapping(utilities)
                    for node_id, utilities in self.node_values.items()
                }
            ),
        )

    @property
    def strategies(self) -> Mapping[str, str]:
        return self.strategy_profile


@dataclass
class _Continuation:
    utilities: tuple[float, ...]
    strategy: dict[str, str]
    node_values: dict[str, dict[str, float]]


def _merge_continuations(
    continuations: tuple[_Continuation, ...],
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    strategy: dict[str, str] = {}
    node_values: dict[str, dict[str, float]] = {}
    for continuation in continuations:
        overlap = set(strategy).intersection(continuation.strategy)
        if overlap:
            raise RuntimeError(
                f"game was not a tree; overlapping strategy nodes={sorted(overlap)}"
            )
        strategy.update(continuation.strategy)
        node_values.update(continuation.node_values)
    return strategy, node_values


def solve_backward_induction(
    game: ExtensiveFormGame,
    *,
    tolerance: float = 1e-10,
) -> SolverResult:
    """Enumerate every pure backward-induction solution.

    Chance nodes are evaluated in expectation.  At payoff ties, every optimal
    action is retained, including complete equilibrium continuation strategies
    in off-path subgames; consequently ``multiple_solutions`` reports genuine
    pure-strategy multiplicity rather than selecting an arbitrary action.
    """

    started = time.perf_counter()
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if not game.is_perfect_information:
        raise ValueError(
            "backward induction requires singleton information sets (perfect information)"
        )
    players = game.player_ids
    player_index = {player: index for index, player in enumerate(players)}
    tie_actions: dict[str, set[str]] = {}

    def solve_node(node_id: str) -> tuple[_Continuation, ...]:
        node = game.nodes[node_id]
        if isinstance(node, TerminalNode):
            utilities = tuple(node.utilities[player] for player in players)
            return (
                _Continuation(
                    utilities=utilities,
                    strategy={},
                    node_values={node_id: dict(zip(players, utilities, strict=True))},
                ),
            )

        if isinstance(node, ChanceNode):
            child_solution_sets = tuple(
                solve_node(branch.child_id) for branch in node.branches
            )
            solutions: list[_Continuation] = []
            for combination in itertools.product(*child_solution_sets):
                strategy, node_values = _merge_continuations(combination)
                utilities = tuple(
                    sum(
                        branch.probability * combination[branch_index].utilities[player]
                        for branch_index, branch in enumerate(node.branches)
                    )
                    for player in range(len(players))
                )
                node_values[node_id] = dict(zip(players, utilities, strict=True))
                solutions.append(_Continuation(utilities, strategy, node_values))
            return tuple(solutions)

        if not isinstance(node, DecisionNode):
            raise TypeError(f"unsupported node type {type(node).__name__}")
        action_items = tuple(node.actions.items())
        child_solution_sets = tuple(
            solve_node(child_id) for _, child_id in action_items
        )
        acting_player = player_index[node.player_id]
        solutions = []
        for combination in itertools.product(*child_solution_sets):
            action_values = tuple(
                combination[action_index].utilities[acting_player]
                for action_index in range(len(action_items))
            )
            best_value = max(action_values)
            best_indices = tuple(
                action_index
                for action_index, value in enumerate(action_values)
                if value >= best_value - tolerance
            )
            tie_actions.setdefault(node_id, set()).update(
                action_items[action_index][0] for action_index in best_indices
            )
            base_strategy, base_values = _merge_continuations(combination)
            for action_index in best_indices:
                action_id = action_items[action_index][0]
                continuation = combination[action_index]
                strategy = dict(base_strategy)
                strategy[node_id] = action_id
                node_values = dict(base_values)
                node_values[node_id] = dict(
                    zip(players, continuation.utilities, strict=True)
                )
                solutions.append(
                    _Continuation(continuation.utilities, strategy, node_values)
                )
        return tuple(solutions)

    raw_solutions = solve_node(game.root_id)
    deduplicated: dict[str, _Continuation] = {}
    for solution in raw_solutions:
        key = canonical_json(solution.strategy)
        deduplicated[key] = solution
    ordered = tuple(deduplicated[key] for key in sorted(deduplicated))
    public_solutions = tuple(
        BackwardInductionSolution(
            strategy_profile=solution.strategy,
            expected_utilities=dict(zip(players, solution.utilities, strict=True)),
            node_values=solution.node_values,
        )
        for solution in ordered
    )
    runtime = time.perf_counter() - started
    optimal_action_sets = {
        node_id: sorted(actions) for node_id, actions in sorted(tie_actions.items())
    }
    tied_nodes = {
        node_id: actions
        for node_id, actions in optimal_action_sets.items()
        if len(actions) > 1
    }
    warnings = (
        ("payoff ties generate multiple valid backward-induction continuations",)
        if tied_nodes
        else ()
    )
    return SolverResult(
        solver_name=SOLVER_NAME,
        solver_version=SOLVER_VERSION,
        equilibrium_concept="pure-strategy subgame-perfect equilibrium",
        found=bool(public_solutions),
        solutions=public_solutions,
        status="SOLUTIONS_FOUND" if public_solutions else "NO_SOLUTION_FOUND",
        exactness_status="EXACT_WITHIN_TOLERANCE",
        convergence_status="NOT_APPLICABLE",
        best_response_gap=0.0 if public_solutions else None,
        off_path_belief_convention="NOT_APPLICABLE",
        runtime_seconds=runtime,
        warnings=warnings,
        assumptions=(
            "finite game tree",
            "perfect information represented by singleton information sets",
            "chance probabilities are exogenous",
        ),
        seed=None,
        metadata={
            "tolerance": tolerance,
            "solution_count": len(public_solutions),
            "optimal_actions_by_node": optimal_action_sets,
            "tie_actions_by_node": tied_nodes,
        },
    )


backward_induction = solve_backward_induction
