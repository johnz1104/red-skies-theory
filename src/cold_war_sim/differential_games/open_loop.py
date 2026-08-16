"""Matrix-exponential solver for finite-horizon open-loop Nash equilibria."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any, cast

import numpy as np
from scipy.linalg import expm

from .base import (
    BoundarySystemDiagnostics,
    BoundarySystemStatus,
    OpenLoopSolveError,
    finite_float,
    reject_unknown_keys,
)
from .linear_quadratic import FloatArray, LinearQuadraticDifferentialGame
from .results import DifferentialGameVerification, OpenLoopNashResult
from .trajectories import DifferentialGameTrajectory


@dataclass(frozen=True, slots=True)
class OpenLoopSolverSettings:
    """Numerical controls for the supported matrix-exponential method."""

    solution_concept: str = "open_loop_nash"
    time_points: int = 201
    boundary_tolerance: float = 1e-9
    verification_tolerance: float = 1e-7
    condition_number_limit: float = 1e12
    matrix_exponential_tolerance: float = 1e-10
    unilateral_perturbation_size: float = 1e-4
    unilateral_perturbation_count: int = 6

    def __post_init__(self) -> None:
        if self.solution_concept != "open_loop_nash":
            raise ValueError("only the open_loop_nash solution concept is supported")
        for integer_value, name, minimum in (
            (self.time_points, "time_points", 2),
            (self.unilateral_perturbation_count, "unilateral_perturbation_count", 1),
        ):
            if (
                isinstance(integer_value, bool)
                or not isinstance(integer_value, int)
                or integer_value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in (
            "boundary_tolerance",
            "verification_tolerance",
            "matrix_exponential_tolerance",
            "unilateral_perturbation_size",
        ):
            numeric_value = finite_float(getattr(self, name), name=name)
            if numeric_value <= 0.0:
                raise ValueError(f"{name} must be strictly positive")
            object.__setattr__(self, name, numeric_value)
        limit = finite_float(self.condition_number_limit, name="condition_number_limit")
        if limit <= 1.0:
            raise ValueError("condition_number_limit must be greater than one")
        object.__setattr__(self, "condition_number_limit", limit)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> OpenLoopSolverSettings:
        if values is None:
            return cls()
        data = reject_unknown_keys(
            dict(values),
            allowed={
                "solution_concept",
                "time_points",
                "boundary_tolerance",
                "verification_tolerance",
                "condition_number_limit",
                "matrix_exponential_tolerance",
                "unilateral_perturbation_size",
                "unilateral_perturbation_count",
            },
            name="open-loop solver settings",
        )
        return cls(**data)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "solution_concept": self.solution_concept,
            "time_points": self.time_points,
            "boundary_tolerance": self.boundary_tolerance,
            "verification_tolerance": self.verification_tolerance,
            "condition_number_limit": self.condition_number_limit,
            "matrix_exponential_tolerance": self.matrix_exponential_tolerance,
            "unilateral_perturbation_size": self.unilateral_perturbation_size,
            "unilateral_perturbation_count": self.unilateral_perturbation_count,
        }


@dataclass(frozen=True, slots=True)
class _HamiltonianSystem:
    matrix: FloatArray
    augmented_matrix: FloatArray
    state_selector: FloatArray
    costate_selectors: Mapping[str, FloatArray]


def _hamiltonian_system(game: LinearQuadraticDifferentialGame) -> _HamiltonianSystem:
    n = game.state_dimension
    player_0, player_1 = game.player_ids
    blocks: list[FloatArray] = []
    for player in game.player_ids:
        objective = game.objectives[player]
        control = game.control_matrices[player]
        blocks.append(control @ np.linalg.solve(objective.control_cost, control.T))
    zeros = np.zeros((n, n), dtype=float)
    matrix = np.block(
        [
            [game.state_matrix, -blocks[0], -blocks[1]],
            [
                -game.objectives[player_0].state_cost,
                -game.state_matrix.T,
                zeros,
            ],
            [
                -game.objectives[player_1].state_cost,
                zeros,
                -game.state_matrix.T,
            ],
        ]
    )
    forcing = np.concatenate((game.affine_vector, np.zeros(2 * n, dtype=float)))
    augmented = np.zeros((3 * n + 1, 3 * n + 1), dtype=float)
    augmented[: 3 * n, : 3 * n] = matrix
    augmented[: 3 * n, -1] = forcing
    state_selector = np.zeros((n, 3 * n + 1), dtype=float)
    state_selector[:, :n] = np.eye(n)
    selectors: dict[str, FloatArray] = {}
    for index, player in enumerate(game.player_ids):
        selector = np.zeros((n, 3 * n + 1), dtype=float)
        start = n * (index + 1)
        selector[:, start : start + n] = np.eye(n)
        selectors[player] = selector
    return _HamiltonianSystem(matrix, augmented, state_selector, selectors)


def _initial_costates(
    game: LinearQuadraticDifferentialGame,
    settings: OpenLoopSolverSettings,
    system: _HamiltonianSystem,
) -> tuple[FloatArray, BoundarySystemDiagnostics]:
    n = game.state_dimension
    transition = expm(system.augmented_matrix * game.horizon)
    if not np.all(np.isfinite(transition)):
        raise FloatingPointError("matrix exponential produced non-finite values")
    half_transition = expm(system.augmented_matrix * (0.5 * game.horizon))
    semigroup_scale = max(1.0, float(np.linalg.norm(transition, ord=2)))
    semigroup_residual = float(
        np.linalg.norm(half_transition @ half_transition - transition, ord=2)
        / semigroup_scale
    )
    if semigroup_residual > settings.matrix_exponential_tolerance:
        raise FloatingPointError(
            "matrix exponential failed the configured semigroup consistency tolerance: "
            f"{semigroup_residual} > {settings.matrix_exponential_tolerance}"
        )
    physical_transition = transition[: 3 * n, : 3 * n]
    affine_transition = transition[: 3 * n, -1]
    identity = np.eye(n)
    zeros = np.zeros((n, n), dtype=float)
    player_0, player_1 = game.player_ids
    terminal_operator = np.block(
        [
            [-game.objectives[player_0].terminal_state_cost, identity, zeros],
            [-game.objectives[player_1].terminal_state_cost, zeros, identity],
        ]
    )
    boundary_matrix = terminal_operator @ physical_transition[:, n:]
    boundary_vector = terminal_operator @ (
        physical_transition[:, :n] @ game.initial_state + affine_transition
    )
    dimension = 2 * n
    rank = int(np.linalg.matrix_rank(boundary_matrix, tol=settings.boundary_tolerance))
    augmented_rank = int(
        np.linalg.matrix_rank(
            np.column_stack((boundary_matrix, -boundary_vector)),
            tol=settings.boundary_tolerance,
        )
    )
    if rank < dimension:
        status = (
            BoundarySystemStatus.INCONSISTENT
            if augmented_rank > rank
            else BoundarySystemStatus.NONUNIQUE
        )
        diagnostics = BoundarySystemDiagnostics(
            status=status,
            rank=rank,
            augmented_rank=augmented_rank,
            dimension=dimension,
            condition_number=None,
            residual_norm=None,
        )
        description = "inconsistent" if status is BoundarySystemStatus.INCONSISTENT else "nonunique"
        raise OpenLoopSolveError(
            f"open-loop Nash boundary system is {description}; "
            "the solver will not select an initial costate",
            diagnostics,
        )
    condition_number = float(np.linalg.cond(boundary_matrix))
    if not np.isfinite(condition_number) or condition_number > settings.condition_number_limit:
        diagnostics = BoundarySystemDiagnostics(
            status=BoundarySystemStatus.ILL_CONDITIONED,
            rank=rank,
            augmented_rank=augmented_rank,
            dimension=dimension,
            condition_number=(condition_number if np.isfinite(condition_number) else None),
            residual_norm=None,
        )
        raise OpenLoopSolveError(
            "open-loop Nash boundary system exceeds condition_number_limit",
            diagnostics,
        )
    costates = np.linalg.solve(boundary_matrix, -boundary_vector)
    residual = float(np.linalg.norm(boundary_matrix @ costates + boundary_vector, ord=2))
    if not np.all(np.isfinite(costates)) or residual > settings.boundary_tolerance:
        diagnostics = BoundarySystemDiagnostics(
            status=BoundarySystemStatus.ILL_CONDITIONED,
            rank=rank,
            augmented_rank=augmented_rank,
            dimension=dimension,
            condition_number=condition_number,
            residual_norm=residual,
        )
        raise OpenLoopSolveError(
            "initial-costate solution failed the configured boundary tolerance",
            diagnostics,
        )
    return cast(FloatArray, costates), BoundarySystemDiagnostics(
        status=BoundarySystemStatus.UNIQUE_WELL_CONDITIONED,
        rank=rank,
        augmented_rank=augmented_rank,
        dimension=dimension,
        condition_number=condition_number,
        residual_norm=residual,
    )


def _sample_trajectory(
    game: LinearQuadraticDifferentialGame,
    settings: OpenLoopSolverSettings,
    system: _HamiltonianSystem,
    initial_costates: FloatArray,
) -> tuple[DifferentialGameTrajectory, FloatArray]:
    n = game.state_dimension
    initial_augmented = np.concatenate((game.initial_state, initial_costates, [1.0]))
    times = np.linspace(0.0, game.horizon, settings.time_points, dtype=float)
    augmented_states = np.stack(
        [expm(system.augmented_matrix * time) @ initial_augmented for time in times]
    )
    if not np.all(np.isfinite(augmented_states)):
        raise FloatingPointError("trajectory matrix exponential produced non-finite values")
    states = augmented_states[:, :n]
    costates: dict[str, FloatArray] = {}
    controls: dict[str, FloatArray] = {}
    for index, player in enumerate(game.player_ids):
        player_costates = augmented_states[:, n * (index + 1) : n * (index + 2)]
        objective = game.objectives[player]
        control_matrix = game.control_matrices[player]
        player_controls = -np.linalg.solve(
            objective.control_cost, control_matrix.T @ player_costates.T
        ).T
        costates[player] = cast(FloatArray, player_costates)
        controls[player] = cast(FloatArray, player_controls)
    trajectory = DifferentialGameTrajectory.from_mapping(
        {
            "times": times,
            "states": states,
            "controls": controls,
            "costates": costates,
        },
        player_ids=game.player_ids,
        state_dimension=n,
        control_dimensions=game.control_dimensions,
    )
    return trajectory, cast(FloatArray, initial_augmented)


def _exact_objective_values(
    game: LinearQuadraticDifferentialGame,
    system: _HamiltonianSystem,
    initial_augmented: FloatArray,
) -> dict[str, float]:
    dimension = system.augmented_matrix.shape[0]
    terminal_augmented = expm(system.augmented_matrix * game.horizon) @ initial_augmented
    result: dict[str, float] = {}
    for player in game.player_ids:
        objective = game.objectives[player]
        control_matrix = game.control_matrices[player]
        costate_selector = system.costate_selectors[player]
        control_operator = -np.linalg.solve(
            objective.control_cost, control_matrix.T @ costate_selector
        )
        running_weight = (
            system.state_selector.T @ objective.state_cost @ system.state_selector
            + control_operator.T @ objective.control_cost @ control_operator
        )
        van_loan = np.zeros((2 * dimension, 2 * dimension), dtype=float)
        van_loan[:dimension, :dimension] = -system.augmented_matrix.T
        van_loan[:dimension, dimension:] = running_weight
        van_loan[dimension:, dimension:] = system.augmented_matrix
        exponential = expm(van_loan * game.horizon)
        integral_weight = np.linalg.solve(
            exponential[:dimension, :dimension], exponential[:dimension, dimension:]
        )
        integral_weight = 0.5 * (integral_weight + integral_weight.T)
        running_cost = float(initial_augmented @ integral_weight @ initial_augmented)
        terminal_state = system.state_selector @ terminal_augmented
        terminal_cost = float(terminal_state @ objective.terminal_state_cost @ terminal_state)
        value = 0.5 * (running_cost + terminal_cost)
        if value < 0.0 and abs(value) <= 1e-10:
            value = 0.0
        if not np.isfinite(value):
            raise FloatingPointError(f"objective for {player!r} is non-finite")
        result[player] = value
    return result


def solve_open_loop_nash(
    game: LinearQuadraticDifferentialGame,
    settings: OpenLoopSolverSettings | Mapping[str, Any] | None = None,
) -> OpenLoopNashResult:
    """Solve and independently verify the supported open-loop Nash system."""

    from .verification import verify_open_loop_nash

    parsed_settings = (
        settings
        if isinstance(settings, OpenLoopSolverSettings)
        else OpenLoopSolverSettings.from_mapping(settings)
    )
    started = perf_counter()
    system = _hamiltonian_system(game)
    initial_costates, boundary = _initial_costates(game, parsed_settings, system)
    trajectory, initial_augmented = _sample_trajectory(
        game, parsed_settings, system, initial_costates
    )
    objectives = _exact_objective_values(game, system, initial_augmented)
    preverification_record = DifferentialGameVerification(
        passed=False,
        boundary_residual=boundary.residual_norm or 0.0,
        first_order_residual=0.0,
        dynamics_residual=0.0,
        costate_residual=0.0,
        objective_residual=0.0,
        terminal_state_residual=0.0,
        boundary_rank_matches=True,
        boundary_condition_residual=0.0,
        boundary_record_residual=0.0,
        event_metric_residuals={},
        best_response_gaps={player: 0.0 for player in game.player_ids},
        unilateral_perturbations=(),
        warnings=("independent verification not yet run",),
    )
    provisional = OpenLoopNashResult(
        model=game,
        solver_settings=parsed_settings.to_dict(),
        trajectory=trajectory,
        objective_values=objectives,
        boundary_system=boundary,
        verification=preverification_record,
        runtime_seconds=0.0,
        warnings=(
            "Controls are unconstrained simultaneous open-loop policies fixed at time zero.",
        ),
    )
    verification = verify_open_loop_nash(game, provisional, parsed_settings)
    return OpenLoopNashResult(
        model=game,
        solver_settings=parsed_settings.to_dict(),
        trajectory=trajectory,
        objective_values=objectives,
        boundary_system=boundary,
        verification=verification,
        runtime_seconds=perf_counter() - started,
        warnings=provisional.warnings,
    )


__all__ = ["OpenLoopSolverSettings", "solve_open_loop_nash"]
