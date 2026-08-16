"""Independent verification for open-loop Nash differential-game solutions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_bvp, solve_ivp, trapezoid
from scipy.interpolate import CubicSpline
from scipy.linalg import expm

from .linear_quadratic import FloatArray, LinearQuadraticDifferentialGame
from .metrics import armament_trajectory_metrics
from .results import DifferentialGameVerification, OpenLoopNashResult

if TYPE_CHECKING:
    from .open_loop import OpenLoopSolverSettings


def _control_interpolator(
    times: FloatArray, values: FloatArray
) -> Callable[[float | NDArray[np.float64]], FloatArray]:
    spline = CubicSpline(times, values, axis=0)

    def interpolate(time: float | NDArray[np.float64]) -> FloatArray:
        raw = np.asarray(time, dtype=float)
        if raw.ndim == 0:
            return cast(FloatArray, np.asarray(spline(float(raw)), dtype=float))
        return cast(FloatArray, np.asarray(spline(raw), dtype=float).T)

    return interpolate


def _integrate_state_for_controls(
    game: LinearQuadraticDifferentialGame,
    control_functions: Mapping[str, Callable[[float], FloatArray]],
    evaluation_times: FloatArray,
) -> FloatArray:
    def rhs(time: float, state: FloatArray) -> FloatArray:
        derivative = game.state_matrix @ state + game.affine_vector
        for player in game.player_ids:
            derivative = derivative + game.control_matrices[player] @ control_functions[player](time)
        return cast(FloatArray, derivative)

    solution = solve_ivp(
        rhs,
        (0.0, game.horizon),
        game.initial_state,
        t_eval=evaluation_times,
        method="DOP853",
        rtol=2e-11,
        atol=2e-13,
    )
    if not solution.success or solution.y.shape[1] != evaluation_times.shape[0]:
        raise RuntimeError(f"independent state integration failed: {solution.message}")
    return cast(FloatArray, solution.y.T)


def _independent_objectives(
    game: LinearQuadraticDifferentialGame,
    control_functions: Mapping[str, Callable[[float], FloatArray]],
) -> dict[str, float]:
    """Integrate state and running costs independently with adaptive DOP853."""

    n = game.state_dimension

    def rhs(time: float, augmented: FloatArray) -> FloatArray:
        state = augmented[:n]
        derivative = game.state_matrix @ state + game.affine_vector
        running_costs: list[float] = []
        for player in game.player_ids:
            control = control_functions[player](time)
            derivative = derivative + game.control_matrices[player] @ control
            objective = game.objectives[player]
            running_costs.append(
                0.5
                * float(
                    state @ objective.state_cost @ state
                    + control @ objective.control_cost @ control
                )
            )
        return cast(FloatArray, np.concatenate((derivative, running_costs)))

    initial = np.concatenate((game.initial_state, np.zeros(2, dtype=float)))
    solution = solve_ivp(
        rhs,
        (0.0, game.horizon),
        initial,
        method="DOP853",
        rtol=2e-11,
        atol=2e-13,
    )
    if not solution.success:
        raise RuntimeError(
            f"independent state/objective integration failed: {solution.message}"
        )
    terminal_state = solution.y[:n, -1]
    values: dict[str, float] = {}
    for index, player in enumerate(game.player_ids):
        objective = game.objectives[player]
        terminal = 0.5 * float(
            terminal_state @ objective.terminal_state_cost @ terminal_state
        )
        values[player] = float(solution.y[n + index, -1] + terminal)
    return values


def _sampled_objective(
    game: LinearQuadraticDifferentialGame,
    player: str,
    times: FloatArray,
    states: FloatArray,
    controls: FloatArray,
) -> float:
    objective = game.objectives[player]
    state_terms = np.einsum("ti,ij,tj->t", states, objective.state_cost, states)
    control_terms = np.einsum(
        "ti,ij,tj->t", controls, objective.control_cost, controls
    )
    running = float(trapezoid(state_terms + control_terms, times))
    terminal = float(states[-1] @ objective.terminal_state_cost @ states[-1])
    return 0.5 * (running + terminal)


def _independent_best_response(
    game: LinearQuadraticDifferentialGame,
    player: str,
    result: OpenLoopNashResult,
    settings: OpenLoopSolverSettings,
) -> tuple[float, float]:
    """Solve one player's fixed-opponent convex BVP using SciPy's collocation method."""

    opponent = game.player_ids[1] if player == game.player_ids[0] else game.player_ids[0]
    times = result.trajectory.times
    opponent_control = _control_interpolator(times, result.trajectory.controls[opponent])
    objective = game.objectives[player]
    control_matrix = game.control_matrices[player]
    n = game.state_dimension

    def ode(time: FloatArray, values: FloatArray) -> FloatArray:
        states = values[:n]
        costates = values[n:]
        controls = -np.linalg.solve(objective.control_cost, control_matrix.T @ costates)
        opponent_controls = opponent_control(time)
        state_derivative = (
            game.state_matrix @ states
            + control_matrix @ controls
            + game.control_matrices[opponent] @ opponent_controls
            + game.affine_vector[:, None]
        )
        costate_derivative = -objective.state_cost @ states - game.state_matrix.T @ costates
        return cast(FloatArray, np.vstack((state_derivative, costate_derivative)))

    def boundary(left: FloatArray, right: FloatArray) -> FloatArray:
        return cast(
            FloatArray,
            np.concatenate(
                (
                    left[:n] - game.initial_state,
                    right[n:] - objective.terminal_state_cost @ right[:n],
                )
            ),
        )

    guess = np.vstack(
        (result.trajectory.states.T, result.trajectory.costates[player].T)
    )
    bvp = solve_bvp(
        ode,
        boundary,
        times,
        guess,
        tol=max(1e-8, 0.1 * settings.verification_tolerance),
        max_nodes=max(1000, 10 * settings.time_points),
    )
    if not bvp.success:
        raise RuntimeError(f"independent best-response BVP failed for {player}: {bvp.message}")
    values = bvp.sol(times)
    states = cast(FloatArray, values[:n].T)
    costates = cast(FloatArray, values[n:].T)
    controls = cast(
        FloatArray,
        -np.linalg.solve(objective.control_cost, control_matrix.T @ costates.T).T,
    )
    best_response_cost = _sampled_objective(game, player, times, states, controls)
    candidate_functions = {
        other: _control_interpolator(times, result.trajectory.controls[other])
        for other in game.player_ids
    }
    candidate_states = _integrate_state_for_controls(game, candidate_functions, times)
    candidate_cost = _sampled_objective(
        game,
        player,
        times,
        candidate_states,
        result.trajectory.controls[player],
    )
    return candidate_cost, best_response_cost


def _perturbations(
    game: LinearQuadraticDifferentialGame,
    result: OpenLoopNashResult,
    settings: OpenLoopSolverSettings,
) -> tuple[dict[str, Any], ...]:
    times = result.trajectory.times
    records: list[dict[str, Any]] = []
    perturbation_limit = settings.unilateral_perturbation_count
    for player in game.player_ids:
        control_dimension = game.control_dimensions[player]
        opponent = game.player_ids[1] if player == game.player_ids[0] else game.player_ids[0]
        baseline_controls = {
            other: _control_interpolator(times, result.trajectory.controls[other])
            for other in game.player_ids
        }
        baseline_states = _integrate_state_for_controls(game, baseline_controls, times)
        baseline_sampled = _sampled_objective(
            game,
            player,
            times,
            baseline_states,
            result.trajectory.controls[player],
        )
        patterns = (
            ("constant_positive", lambda time: 1.0),
            ("constant_negative", lambda time: -1.0),
            ("sine_positive", lambda time: np.sin(np.pi * time / game.horizon)),
            ("sine_negative", lambda time: -np.sin(np.pi * time / game.horizon)),
            (
                "early_smooth_pulse",
                lambda time: np.exp(
                    -((time - 0.2 * game.horizon) / (0.12 * game.horizon)) ** 2
                ),
            ),
            (
                "late_smooth_pulse",
                lambda time: -np.exp(
                    -((time - 0.8 * game.horizon) / (0.12 * game.horizon)) ** 2
                ),
            ),
        )
        for perturbation_index, (name, pattern) in enumerate(patterns[:perturbation_limit]):
            component = perturbation_index % control_dimension
            base_control = baseline_controls[player]

            def perturbed(
                time: float,
                *,
                component: int = component,
                pattern: Any = pattern,
                base_control: Callable[[float], FloatArray] = base_control,
            ) -> FloatArray:
                value = np.array(base_control(time), dtype=float, copy=True)
                value[component] += settings.unilateral_perturbation_size * pattern(time)
                return cast(FloatArray, value)

            control_functions: dict[str, Callable[[float], FloatArray]] = {
                player: perturbed,
                opponent: baseline_controls[opponent],
            }
            perturbed_states = _integrate_state_for_controls(game, control_functions, times)
            perturbed_samples = np.stack([perturbed(float(time)) for time in times])
            perturbed_cost = _sampled_objective(
                game, player, times, perturbed_states, perturbed_samples
            )
            records.append(
                {
                    "player": player,
                    "name": name,
                    "control_component": component,
                    "magnitude": settings.unilateral_perturbation_size,
                    "objective_change": perturbed_cost - baseline_sampled,
                    "improved": perturbed_cost < baseline_sampled - settings.verification_tolerance,
                }
            )
    return tuple(records)


def _independent_boundary_diagnostics(
    game: LinearQuadraticDifferentialGame,
    result: OpenLoopNashResult,
    settings: OpenLoopSolverSettings,
) -> tuple[int, int, float, float]:
    """Rebuild the boundary system without using saved solver diagnostics."""

    n = game.state_dimension
    player_0, player_1 = game.player_ids
    control_terms = []
    for player in game.player_ids:
        objective = game.objectives[player]
        control_matrix = game.control_matrices[player]
        control_terms.append(
            control_matrix @ np.linalg.solve(objective.control_cost, control_matrix.T)
        )
    zeros = np.zeros((n, n), dtype=float)
    hamiltonian = np.block(
        [
            [game.state_matrix, -control_terms[0], -control_terms[1]],
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
    augmented[: 3 * n, : 3 * n] = hamiltonian
    augmented[: 3 * n, -1] = forcing
    transition = expm(augmented * game.horizon)
    if not np.all(np.isfinite(transition)):
        raise RuntimeError("independent boundary matrix exponential is non-finite")
    physical_transition = transition[: 3 * n, : 3 * n]
    affine_transition = transition[: 3 * n, -1]
    identity = np.eye(n)
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
    rank = int(np.linalg.matrix_rank(boundary_matrix, tol=settings.boundary_tolerance))
    augmented_rank = int(
        np.linalg.matrix_rank(
            np.column_stack((boundary_matrix, -boundary_vector)),
            tol=settings.boundary_tolerance,
        )
    )
    raw_condition = float(np.linalg.cond(boundary_matrix))
    condition = (
        raw_condition if np.isfinite(raw_condition) else float(np.finfo(float).max)
    )
    initial_costates = np.concatenate(
        [result.trajectory.costates[player][0] for player in game.player_ids]
    )
    residual = float(
        np.linalg.norm(boundary_matrix @ initial_costates + boundary_vector, ord=2)
    )
    return rank, augmented_rank, condition, residual


def verify_open_loop_nash(
    game: LinearQuadraticDifferentialGame,
    result: OpenLoopNashResult,
    settings: OpenLoopSolverSettings | None = None,
) -> DifferentialGameVerification:
    """Independently verify state/costate equations and unilateral optimality."""

    from .open_loop import OpenLoopSolverSettings

    parsed = settings or OpenLoopSolverSettings.from_mapping(result.solver_settings)
    times = result.trajectory.times
    states = result.trajectory.states
    control_interpolators = {
        player: _control_interpolator(times, result.trajectory.controls[player])
        for player in game.player_ids
    }
    independently_integrated = _integrate_state_for_controls(
        game, control_interpolators, times
    )
    dynamics_residual = float(np.max(np.abs(independently_integrated - states)))
    independent_objectives = _independent_objectives(game, control_interpolators)
    objective_residual = max(
        abs(independent_objectives[player] - result.objective_values[player])
        for player in game.player_ids
    )
    reported_terminal = (
        np.asarray(result.reported_terminal_state, dtype=float)
        if result.reported_terminal_state is not None
        else states[-1]
    )
    terminal_state_residual = float(np.max(np.abs(reported_terminal - states[-1])))

    recomputed_rank, recomputed_augmented_rank, recomputed_condition, recomputed_boundary = (
        _independent_boundary_diagnostics(game, result, parsed)
    )
    boundary_rank_matches = result.boundary_system.rank == recomputed_rank
    saved_condition = float(result.boundary_system.condition_number or 0.0)
    boundary_condition_residual = abs(saved_condition - recomputed_condition) / max(
        1.0, abs(recomputed_condition)
    )
    saved_boundary_residual = float(result.boundary_system.residual_norm or 0.0)
    boundary_record_residual = abs(saved_boundary_residual - recomputed_boundary)

    event_metric_residuals: dict[str, float] = {}
    if (
        result.aggregate_armament_exposure is not None
        or result.escalation_risk_proxy is not None
    ):
        metrics = armament_trajectory_metrics(times, states, horizon=game.horizon)
        if result.aggregate_armament_exposure is not None:
            event_metric_residuals["aggregate_armament_exposure"] = abs(
                result.aggregate_armament_exposure
                - metrics.aggregate_armament_exposure
            )
        if result.escalation_risk_proxy is not None:
            event_metric_residuals["escalation_risk_proxy"] = abs(
                result.escalation_risk_proxy - metrics.escalation_risk_proxy
            )

    terminal_residuals = []
    first_order_residuals = []
    costate_residuals = []
    for player in game.player_ids:
        objective = game.objectives[player]
        control_matrix = game.control_matrices[player]
        costates = result.trajectory.costates[player]
        controls = result.trajectory.controls[player]
        terminal_residuals.append(
            float(
                np.linalg.norm(
                    costates[-1] - objective.terminal_state_cost @ states[-1], ord=2
                )
            )
        )
        stationarity = controls @ objective.control_cost.T + costates @ control_matrix
        first_order_residuals.append(float(np.max(np.linalg.norm(stationarity, axis=1))))
        derivative = np.gradient(
            costates,
            times,
            axis=0,
            edge_order=2 if times.shape[0] >= 3 else 1,
        )
        expected = -states @ objective.state_cost.T - costates @ game.state_matrix
        # Finite-difference derivatives are an independent, lower-accuracy check.
        costate_residuals.append(float(np.max(np.linalg.norm(derivative - expected, axis=1))))
    boundary_residual = max(
        float(np.linalg.norm(states[0] - game.initial_state, ord=2)),
        *terminal_residuals,
        recomputed_boundary,
    )
    first_order_residual = max(first_order_residuals)
    costate_residual = max(costate_residuals)

    gaps: dict[str, float] = {}
    warnings: list[str] = []
    try:
        for player in game.player_ids:
            candidate_cost, best_response_cost = _independent_best_response(
                game, player, result, parsed
            )
            # Positive means the candidate can lower its cost by deviating.
            gaps[player] = max(0.0, candidate_cost - best_response_cost)
    except RuntimeError as error:
        failure_gap = max(1.0, 1000.0 * parsed.verification_tolerance)
        gaps = {player: failure_gap for player in game.player_ids}
        warnings.append(str(error))
    perturbations = _perturbations(game, result, parsed)
    perturbation_failure = any(bool(item["improved"]) for item in perturbations)
    grid_matches_settings = (
        times.shape[0] == parsed.time_points
        and abs(float(times[0])) <= parsed.verification_tolerance
        and abs(float(times[-1]) - game.horizon) <= parsed.verification_tolerance
    )
    boundary_system_supported = (
        recomputed_rank == 2 * game.state_dimension
        and recomputed_augmented_rank == recomputed_rank
        and recomputed_condition <= parsed.condition_number_limit
    )
    saved_metric_values: dict[str, float] = {}
    if result.aggregate_armament_exposure is not None:
        saved_metric_values["aggregate_armament_exposure"] = (
            result.aggregate_armament_exposure
        )
    if result.escalation_risk_proxy is not None:
        saved_metric_values["escalation_risk_proxy"] = result.escalation_risk_proxy
    metric_checks_pass = all(
        residual
        <= 20.0
        * parsed.verification_tolerance
        * max(1.0, abs(saved_metric_values[name]))
        for name, residual in event_metric_residuals.items()
    )
    # Finite-difference costate derivatives converge quadratically with the grid.
    derivative_tolerance = max(
        50.0 * parsed.verification_tolerance,
        10.0 * (game.horizon / (parsed.time_points - 1)) ** 2,
    )
    passed = (
        boundary_residual <= parsed.verification_tolerance
        and first_order_residual <= parsed.verification_tolerance
        and dynamics_residual <= 20.0 * parsed.verification_tolerance
        and costate_residual <= derivative_tolerance
        and objective_residual <= 20.0 * parsed.verification_tolerance
        and terminal_state_residual <= parsed.verification_tolerance
        and boundary_rank_matches
        and boundary_condition_residual <= 20.0 * parsed.verification_tolerance
        and boundary_record_residual <= parsed.verification_tolerance
        and boundary_system_supported
        and grid_matches_settings
        and metric_checks_pass
        and all(gap <= 20.0 * parsed.verification_tolerance for gap in gaps.values())
        and not perturbation_failure
    )
    if not boundary_rank_matches:
        warnings.append("saved boundary-system rank does not match independent reconstruction")
    if boundary_condition_residual > 20.0 * parsed.verification_tolerance:
        warnings.append(
            "saved boundary-system condition number does not match independent reconstruction"
        )
    if boundary_record_residual > parsed.verification_tolerance:
        warnings.append(
            "saved boundary-system residual does not match independent reconstruction"
        )
    if terminal_state_residual > parsed.verification_tolerance:
        warnings.append("saved redundant terminal_state does not match the trajectory")
    if not boundary_system_supported:
        warnings.append("independently reconstructed boundary system is unsupported")
    if not grid_matches_settings:
        warnings.append("saved time grid does not match the embedded model and solver settings")
    if not metric_checks_pass:
        warnings.append("saved event metrics do not match their trajectory-derived definitions")
    if not passed and not warnings:
        warnings.append("one or more independent equilibrium checks exceeded tolerance")
    return DifferentialGameVerification(
        passed=passed,
        boundary_residual=boundary_residual,
        first_order_residual=first_order_residual,
        dynamics_residual=dynamics_residual,
        costate_residual=costate_residual,
        objective_residual=objective_residual,
        terminal_state_residual=terminal_state_residual,
        boundary_rank_matches=boundary_rank_matches,
        boundary_condition_residual=boundary_condition_residual,
        boundary_record_residual=boundary_record_residual,
        event_metric_residuals=event_metric_residuals,
        best_response_gaps=gaps,
        unilateral_perturbations=perturbations,
        warnings=tuple(warnings),
    )


__all__ = ["verify_open_loop_nash"]
