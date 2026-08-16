"""Side-by-side comparisons of supported differential-game solutions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .results import OpenLoopNashResult


@dataclass(frozen=True, slots=True)
class DifferentialSolutionComparison:
    """A numerical comparison, not a historical or causal claim."""

    baseline_model_id: str
    counterfactual_model_id: str
    changed_model_elements: tuple[str, ...]
    time_grid: tuple[float, ...]
    baseline_states: tuple[tuple[float, ...], ...]
    counterfactual_states: tuple[tuple[float, ...], ...]
    baseline_controls: Mapping[str, tuple[tuple[float, ...], ...]]
    counterfactual_controls: Mapping[str, tuple[tuple[float, ...], ...]]
    baseline_terminal_state: tuple[float, ...]
    counterfactual_terminal_state: tuple[float, ...]
    baseline_objective_costs: Mapping[str, float]
    counterfactual_objective_costs: Mapping[str, float]
    objective_cost_changes: Mapping[str, float]
    terminal_state_change: tuple[float, ...]
    maximum_state_trajectory_change: float
    maximum_control_changes: Mapping[str, float]
    baseline_aggregate_armament_exposure: float | None
    counterfactual_aggregate_armament_exposure: float | None
    aggregate_armament_exposure_change: float | None
    baseline_escalation_risk_proxy: float | None
    counterfactual_escalation_risk_proxy: float | None
    escalation_risk_proxy_change: float | None
    verification_residuals: Mapping[str, Mapping[str, float]]
    commitment_assumption: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_model_id": self.baseline_model_id,
            "counterfactual_model_id": self.counterfactual_model_id,
            "changed_model_elements": list(self.changed_model_elements),
            "time_grid": list(self.time_grid),
            "baseline_states": [list(row) for row in self.baseline_states],
            "counterfactual_states": [list(row) for row in self.counterfactual_states],
            "baseline_controls": {
                player: [list(row) for row in self.baseline_controls[player]]
                for player in sorted(self.baseline_controls)
            },
            "counterfactual_controls": {
                player: [list(row) for row in self.counterfactual_controls[player]]
                for player in sorted(self.counterfactual_controls)
            },
            "baseline_terminal_state": list(self.baseline_terminal_state),
            "counterfactual_terminal_state": list(self.counterfactual_terminal_state),
            "baseline_objective_costs": dict(sorted(self.baseline_objective_costs.items())),
            "counterfactual_objective_costs": dict(
                sorted(self.counterfactual_objective_costs.items())
            ),
            "objective_cost_changes": dict(sorted(self.objective_cost_changes.items())),
            "terminal_state_change": list(self.terminal_state_change),
            "maximum_state_trajectory_change": self.maximum_state_trajectory_change,
            "maximum_control_changes": dict(sorted(self.maximum_control_changes.items())),
            "baseline_aggregate_armament_exposure": (
                self.baseline_aggregate_armament_exposure
            ),
            "counterfactual_aggregate_armament_exposure": (
                self.counterfactual_aggregate_armament_exposure
            ),
            "aggregate_armament_exposure_change": self.aggregate_armament_exposure_change,
            "baseline_escalation_risk_proxy": self.baseline_escalation_risk_proxy,
            "counterfactual_escalation_risk_proxy": (
                self.counterfactual_escalation_risk_proxy
            ),
            "escalation_risk_proxy_change": self.escalation_risk_proxy_change,
            "verification_residuals": {
                name: dict(sorted(values.items()))
                for name, values in sorted(self.verification_residuals.items())
            },
            "commitment_assumption": self.commitment_assumption,
            "warnings": list(self.warnings),
        }


def _changes(baseline: OpenLoopNashResult, counterfactual: OpenLoopNashResult) -> tuple[str, ...]:
    first = baseline.model
    second = counterfactual.model
    changes: list[str] = []
    if not np.array_equal(first.initial_state, second.initial_state):
        changes.append("initial_state")
    if not np.array_equal(first.state_matrix, second.state_matrix) or not np.array_equal(
        first.affine_vector, second.affine_vector
    ):
        changes.append("state_dynamics")
    if any(
        not np.array_equal(first.control_matrices[player], second.control_matrices[player])
        for player in first.player_ids
    ):
        changes.append("control_effectiveness")
    if any(
        not np.array_equal(
            first.objectives[player].state_cost, second.objectives[player].state_cost
        )
        or not np.array_equal(
            first.objectives[player].control_cost, second.objectives[player].control_cost
        )
        or not np.array_equal(
            first.objectives[player].terminal_state_cost,
            second.objectives[player].terminal_state_cost,
        )
        for player in first.player_ids
    ):
        changes.append("cost_weights")
    return tuple(changes)


def _rows(values: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(item) for item in row) for row in values)


def compare_solutions(
    baseline: OpenLoopNashResult,
    counterfactual: OpenLoopNashResult,
) -> DifferentialSolutionComparison:
    """Compare two verified simultaneous-open-loop solutions.

    Initial-state, cost-weight, and control-effectiveness changes are supported.
    This function does not infer why a model changed and does not implement a
    leader-follower or Stackelberg commitment comparison.
    """

    if baseline.model.player_ids != counterfactual.model.player_ids:
        raise ValueError("solutions must use the same ordered player identifiers")
    if baseline.model.state_dimension != counterfactual.model.state_dimension:
        raise ValueError("solutions must use the same state dimension")
    if baseline.model.control_dimensions != counterfactual.model.control_dimensions:
        raise ValueError("solutions must use the same control dimensions")
    if baseline.model.horizon != counterfactual.model.horizon:
        raise ValueError(
            "pathwise comparison requires the same horizon; solve horizon changes as "
            "separate models without trajectory subtraction"
        )
    if baseline.trajectory.times.shape != counterfactual.trajectory.times.shape or not np.allclose(
        baseline.trajectory.times, counterfactual.trajectory.times, rtol=0.0, atol=1e-12
    ):
        raise ValueError("solutions must use the same sampled time grid")
    # Saved pass/fail flags are informational only. Re-run every numerical check
    # against the embedded models and paths before comparing them.
    from .open_loop import OpenLoopSolverSettings
    from .verification import verify_open_loop_nash

    baseline_verification = verify_open_loop_nash(
        baseline.model,
        baseline,
        OpenLoopSolverSettings.from_mapping(baseline.solver_settings),
    )
    counterfactual_verification = verify_open_loop_nash(
        counterfactual.model,
        counterfactual,
        OpenLoopSolverSettings.from_mapping(counterfactual.solver_settings),
    )
    if not baseline_verification.passed or not counterfactual_verification.passed:
        raise ValueError("both solutions must pass independent verification before comparison")
    terminal_delta = counterfactual.trajectory.terminal_state - baseline.trajectory.terminal_state
    exposure_delta = None
    if (
        baseline.aggregate_armament_exposure is not None
        and counterfactual.aggregate_armament_exposure is not None
    ):
        exposure_delta = (
            counterfactual.aggregate_armament_exposure
            - baseline.aggregate_armament_exposure
        )
    risk_delta = None
    if baseline.escalation_risk_proxy is not None and counterfactual.escalation_risk_proxy is not None:
        risk_delta = counterfactual.escalation_risk_proxy - baseline.escalation_risk_proxy
    residual_names = (
        "boundary_residual",
        "first_order_residual",
        "dynamics_residual",
        "costate_residual",
        "objective_residual",
        "terminal_state_residual",
        "boundary_condition_residual",
        "boundary_record_residual",
    )
    return DifferentialSolutionComparison(
        baseline_model_id=baseline.model.model_id,
        counterfactual_model_id=counterfactual.model.model_id,
        changed_model_elements=_changes(baseline, counterfactual),
        time_grid=tuple(float(value) for value in baseline.trajectory.times),
        baseline_states=_rows(baseline.trajectory.states),
        counterfactual_states=_rows(counterfactual.trajectory.states),
        baseline_controls={
            player: _rows(baseline.trajectory.controls[player])
            for player in baseline.model.player_ids
        },
        counterfactual_controls={
            player: _rows(counterfactual.trajectory.controls[player])
            for player in baseline.model.player_ids
        },
        baseline_terminal_state=baseline.terminal_state,
        counterfactual_terminal_state=counterfactual.terminal_state,
        baseline_objective_costs={
            player: baseline.objective_values[player]
            for player in baseline.model.player_ids
        },
        counterfactual_objective_costs={
            player: counterfactual.objective_values[player]
            for player in baseline.model.player_ids
        },
        objective_cost_changes={
            player: counterfactual.objective_values[player] - baseline.objective_values[player]
            for player in baseline.model.player_ids
        },
        terminal_state_change=tuple(float(value) for value in terminal_delta),
        maximum_state_trajectory_change=float(
            np.max(np.abs(counterfactual.trajectory.states - baseline.trajectory.states))
        ),
        maximum_control_changes={
            player: float(
                np.max(
                    np.abs(
                        counterfactual.trajectory.controls[player]
                        - baseline.trajectory.controls[player]
                    )
                )
            )
            for player in baseline.model.player_ids
        },
        baseline_aggregate_armament_exposure=baseline.aggregate_armament_exposure,
        counterfactual_aggregate_armament_exposure=(
            counterfactual.aggregate_armament_exposure
        ),
        aggregate_armament_exposure_change=exposure_delta,
        baseline_escalation_risk_proxy=baseline.escalation_risk_proxy,
        counterfactual_escalation_risk_proxy=counterfactual.escalation_risk_proxy,
        escalation_risk_proxy_change=risk_delta,
        verification_residuals={
            "baseline": {
                name: float(getattr(baseline_verification, name))
                for name in residual_names
            },
            "counterfactual": {
                name: float(getattr(counterfactual_verification, name))
                for name in residual_names
            },
        },
        commitment_assumption=(
            "Both solutions use simultaneous, binding open-loop policy paths fixed at time zero; "
            "no Stackelberg leader or follower is modeled."
        ),
        warnings=(
            "The comparison is conditional on two illustrative formal models and is not a "
            "historical or causal estimate.",
        ),
    )


__all__ = ["DifferentialSolutionComparison", "compare_solutions"]
