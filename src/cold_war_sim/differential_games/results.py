"""Structured results and verification records for differential games."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from cold_war_sim import __version__

from .base import BoundarySystemDiagnostics, BoundarySystemStatus, finite_float
from .linear_quadratic import LinearQuadraticDifferentialGame
from .metrics import armament_trajectory_metrics
from .trajectories import DifferentialGameTrajectory

SCHEMA_VERSION = "1.0"
SOLUTION_CONCEPT = "open_loop_nash"
SOLVER_METHOD = "coupled_linear_tpbvp_matrix_exponential"


@dataclass(frozen=True, slots=True)
class DifferentialGameVerification:
    """Independent residual, best-response, and perturbation checks."""

    passed: bool
    boundary_residual: float
    first_order_residual: float
    dynamics_residual: float
    costate_residual: float
    objective_residual: float
    terminal_state_residual: float
    boundary_rank_matches: bool
    boundary_condition_residual: float
    boundary_record_residual: float
    event_metric_residuals: Mapping[str, float]
    best_response_gaps: Mapping[str, float]
    unilateral_perturbations: tuple[Mapping[str, Any], ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "boundary_residual": self.boundary_residual,
            "first_order_residual": self.first_order_residual,
            "dynamics_residual": self.dynamics_residual,
            "costate_residual": self.costate_residual,
            "objective_residual": self.objective_residual,
            "terminal_state_residual": self.terminal_state_residual,
            "boundary_rank_matches": self.boundary_rank_matches,
            "boundary_condition_residual": self.boundary_condition_residual,
            "boundary_record_residual": self.boundary_record_residual,
            "event_metric_residuals": {
                metric: self.event_metric_residuals[metric]
                for metric in sorted(self.event_metric_residuals)
            },
            "best_response_gaps": {
                player: self.best_response_gaps[player]
                for player in sorted(self.best_response_gaps)
            },
            "unilateral_perturbations": [dict(item) for item in self.unilateral_perturbations],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class OpenLoopNashResult:
    """A sampled candidate together with independent numerical certification."""

    model: LinearQuadraticDifferentialGame
    solver_settings: Mapping[str, Any]
    trajectory: DifferentialGameTrajectory
    objective_values: Mapping[str, float]
    boundary_system: BoundarySystemDiagnostics
    verification: DifferentialGameVerification
    runtime_seconds: float
    reported_terminal_state: tuple[float, ...] | None = None
    warnings: tuple[str, ...] = ()
    aggregate_armament_exposure: float | None = None
    escalation_risk_proxy: float | None = None

    @property
    def status(self) -> str:
        return "VERIFIED_EQUILIBRIUM" if self.verification.passed else "VERIFICATION_FAILED"

    @property
    def terminal_state(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.trajectory.terminal_state)

    @property
    def initial_costates(self) -> Mapping[str, tuple[float, ...]]:
        """Initial first-order variables used to reproduce the candidate path.

        The serialized trajectory records these explicitly as the first row of
        each player's ``costates`` array.
        """

        return {
            player: tuple(float(value) for value in self.trajectory.costates[player][0])
            for player in self.model.player_ids
        }

    def with_event_metrics(
        self,
        *,
        aggregate_armament_exposure: float,
        escalation_risk_proxy: float,
    ) -> OpenLoopNashResult:
        recomputed = armament_trajectory_metrics(
            self.trajectory.times,
            self.trajectory.states,
            horizon=self.model.horizon,
        )
        supplied_exposure = finite_float(
            aggregate_armament_exposure, name="aggregate armament exposure"
        )
        supplied_risk = finite_float(
            escalation_risk_proxy, name="escalation risk proxy"
        )
        for supplied, expected, name in (
            (
                supplied_exposure,
                recomputed.aggregate_armament_exposure,
                "aggregate armament exposure",
            ),
            (supplied_risk, recomputed.escalation_risk_proxy, "escalation risk proxy"),
        ):
            tolerance = 1e-12 * max(1.0, abs(expected))
            if abs(supplied - expected) > tolerance:
                raise ValueError(
                    f"{name} does not match the documented trajectory-derived definition"
                )
        return replace(
            self,
            aggregate_armament_exposure=supplied_exposure,
            escalation_risk_proxy=supplied_risk,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "document_type": "differential_game_result",
            "package_version": __version__,
            "solution_concept": SOLUTION_CONCEPT,
            "solver_method": SOLVER_METHOD,
            "status": self.status,
            "model": self.model.to_dict(),
            "solver_settings": dict(self.solver_settings),
            "trajectory": self.trajectory.to_dict(),
            "objective_values": {
                player: self.objective_values[player]
                for player in sorted(self.objective_values)
            },
            "boundary_system": {
                "rank": self.boundary_system.rank,
                "condition_number": float(self.boundary_system.condition_number or 0.0),
                "residual_norm": float(self.boundary_system.residual_norm or 0.0),
            },
            "verification": self.verification.to_dict(),
            "terminal_state": list(
                self.reported_terminal_state
                if self.reported_terminal_state is not None
                else self.terminal_state
            ),
            "runtime_seconds": self.runtime_seconds,
            "warnings": list(self.warnings),
        }
        if self.aggregate_armament_exposure is not None:
            result["aggregate_armament_exposure"] = self.aggregate_armament_exposure
        if self.escalation_risk_proxy is not None:
            result["escalation_risk_proxy"] = self.escalation_risk_proxy
        return result

    def deterministic_payload(self) -> dict[str, Any]:
        """Return scientific output without the nondeterministic runtime field."""

        result = self.to_dict()
        result.pop("runtime_seconds")
        return result

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> OpenLoopNashResult:
        model = LinearQuadraticDifferentialGame.from_mapping(values["model"])
        trajectory = DifferentialGameTrajectory.from_mapping(
            values["trajectory"],
            player_ids=model.player_ids,
            state_dimension=model.state_dimension,
            control_dimensions=model.control_dimensions,
        )
        boundary = values["boundary_system"]
        verification = values["verification"]
        boundary_diagnostics = BoundarySystemDiagnostics(
            status=BoundarySystemStatus.UNIQUE_WELL_CONDITIONED,
            rank=int(boundary["rank"]),
            augmented_rank=int(boundary["rank"]),
            dimension=2 * model.state_dimension,
            condition_number=finite_float(
                boundary["condition_number"], name="boundary condition number"
            ),
            residual_norm=finite_float(boundary["residual_norm"], name="boundary residual"),
        )
        verification_record = DifferentialGameVerification(
            passed=bool(verification["passed"]),
            boundary_residual=finite_float(
                verification["boundary_residual"], name="verification boundary residual"
            ),
            first_order_residual=finite_float(
                verification["first_order_residual"], name="first-order residual"
            ),
            dynamics_residual=finite_float(
                verification["dynamics_residual"], name="dynamics residual"
            ),
            costate_residual=finite_float(
                verification["costate_residual"], name="costate residual"
            ),
            objective_residual=finite_float(
                verification["objective_residual"], name="objective residual"
            ),
            terminal_state_residual=finite_float(
                verification["terminal_state_residual"], name="terminal-state residual"
            ),
            boundary_rank_matches=bool(verification["boundary_rank_matches"]),
            boundary_condition_residual=finite_float(
                verification["boundary_condition_residual"],
                name="boundary-condition residual",
            ),
            boundary_record_residual=finite_float(
                verification["boundary_record_residual"], name="boundary-record residual"
            ),
            event_metric_residuals={
                str(metric): finite_float(residual, name=f"event metric residual for {metric}")
                for metric, residual in verification["event_metric_residuals"].items()
            },
            best_response_gaps={
                str(player): finite_float(gap, name=f"best-response gap for {player}")
                for player, gap in verification["best_response_gaps"].items()
            },
            unilateral_perturbations=tuple(
                dict(item) for item in verification["unilateral_perturbations"]
            ),
            warnings=tuple(str(item) for item in verification["warnings"]),
        )
        reported_terminal_state = tuple(
            finite_float(item, name="reported terminal state")
            for item in values["terminal_state"]
        )
        if len(reported_terminal_state) != model.state_dimension:
            raise ValueError(
                "terminal_state length must match the embedded model state dimension"
            )
        return cls(
            model=model,
            solver_settings=dict(values["solver_settings"]),
            trajectory=trajectory,
            objective_values={
                str(player): finite_float(value, name=f"objective value for {player}")
                for player, value in values["objective_values"].items()
            },
            boundary_system=boundary_diagnostics,
            verification=verification_record,
            runtime_seconds=finite_float(values["runtime_seconds"], name="runtime_seconds"),
            reported_terminal_state=reported_terminal_state,
            warnings=tuple(str(item) for item in values["warnings"]),
            aggregate_armament_exposure=(
                finite_float(values["aggregate_armament_exposure"], name="armament exposure")
                if "aggregate_armament_exposure" in values
                else None
            ),
            escalation_risk_proxy=(
                finite_float(values["escalation_risk_proxy"], name="escalation risk proxy")
                if "escalation_risk_proxy" in values
                else None
            ),
        )


__all__ = [
    "SCHEMA_VERSION",
    "SOLUTION_CONCEPT",
    "SOLVER_METHOD",
    "DifferentialGameVerification",
    "OpenLoopNashResult",
]
