"""Illustrative linear-quadratic arms-competition differential game.

This strategic-control model is separate from Richardson arms-race dynamics.
Its parameters are illustrative and its unconstrained controls may represent
either buildup or drawdown. It is not historically calibrated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np
from scipy.integrate import trapezoid

from cold_war_sim.differential_games.base import finite_float, reject_unknown_keys
from cold_war_sim.differential_games.linear_quadratic import LinearQuadraticDifferentialGame
from cold_war_sim.differential_games.open_loop import (
    OpenLoopSolverSettings,
    solve_open_loop_nash,
)
from cold_war_sim.differential_games.results import OpenLoopNashResult

MODEL_NAME = "Illustrative LQ arms-competition differential game"
PLAYER_IDS = ("united_states", "soviet_union")


@dataclass(frozen=True, slots=True)
class ArmsCompetitionGameParameters:
    """Illustrative coefficients for a symmetric two-stock strategic model."""

    horizon: float = 5.0
    initial_armament_0: float = 3.0
    initial_armament_1: float = 3.0
    depreciation_0: float = 0.12
    depreciation_1: float = 0.12
    control_effectiveness_0: float = 0.80
    control_effectiveness_1: float = 0.80
    investment_cost_0: float = 1.0
    investment_cost_1: float = 1.0
    strategic_imbalance_cost_0: float = 0.30
    strategic_imbalance_cost_1: float = 0.30
    aggregate_armament_cost_0: float = 0.03
    aggregate_armament_cost_1: float = 0.03
    terminal_imbalance_cost_0: float = 0.60
    terminal_imbalance_cost_1: float = 0.60
    terminal_aggregate_cost_0: float = 0.05
    terminal_aggregate_cost_1: float = 0.05

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = finite_float(getattr(self, name), name=name)
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        if self.horizon <= 0.0:
            raise ValueError("horizon must be strictly positive")
        for name in (
            "control_effectiveness_0",
            "control_effectiveness_1",
            "investment_cost_0",
            "investment_cost_1",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be strictly positive")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any] | None = None
    ) -> ArmsCompetitionGameParameters:
        if values is None:
            return cls()
        allowed = set(cls.__dataclass_fields__)
        data = reject_unknown_keys(
            dict(values), allowed=allowed, name="arms-competition parameters"
        )
        return cls(**data)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in asdict(self).items()}


def _quadratic_cost(imbalance_weight: float, aggregate_weight: float) -> list[list[float]]:
    # (x0-x1)^2 and (x0+x1)^2 are both positive-semidefinite proxies.
    imbalance = np.array([[1.0, -1.0], [-1.0, 1.0]])
    aggregate = np.ones((2, 2), dtype=float)
    return cast(
        list[list[float]],
        (imbalance_weight * imbalance + aggregate_weight * aggregate).tolist(),
    )


def build_arms_competition_game(
    parameters: ArmsCompetitionGameParameters | Mapping[str, Any] | None = None,
    *,
    model_id: str = "illustrative_lq_arms_competition",
) -> LinearQuadraticDifferentialGame:
    """Construct the separately named strategic-control arms model."""

    parsed = (
        parameters
        if isinstance(parameters, ArmsCompetitionGameParameters)
        else ArmsCompetitionGameParameters.from_mapping(parameters)
    )
    return LinearQuadraticDifferentialGame.from_mapping(
        {
            "model_type": "finite_horizon_two_player_linear_quadratic",
            "model_id": model_id,
            "player_ids": list(PLAYER_IDS),
            "horizon": parsed.horizon,
            "initial_state": [parsed.initial_armament_0, parsed.initial_armament_1],
            "dynamics": {
                "A": [
                    [-parsed.depreciation_0, 0.0],
                    [0.0, -parsed.depreciation_1],
                ],
                "B": {
                    PLAYER_IDS[0]: [[parsed.control_effectiveness_0], [0.0]],
                    PLAYER_IDS[1]: [[0.0], [parsed.control_effectiveness_1]],
                },
                "c": [0.0, 0.0],
            },
            "objectives": {
                PLAYER_IDS[0]: {
                    "Q": _quadratic_cost(
                        parsed.strategic_imbalance_cost_0,
                        parsed.aggregate_armament_cost_0,
                    ),
                    "R": [[parsed.investment_cost_0]],
                    "Q_terminal": _quadratic_cost(
                        parsed.terminal_imbalance_cost_0,
                        parsed.terminal_aggregate_cost_0,
                    ),
                },
                PLAYER_IDS[1]: {
                    "Q": _quadratic_cost(
                        parsed.strategic_imbalance_cost_1,
                        parsed.aggregate_armament_cost_1,
                    ),
                    "R": [[parsed.investment_cost_1]],
                    "Q_terminal": _quadratic_cost(
                        parsed.terminal_imbalance_cost_1,
                        parsed.terminal_aggregate_cost_1,
                    ),
                },
            },
            "parameter_status": "ILLUSTRATIVE",
        }
    )


def solve_arms_competition_game(
    parameters: ArmsCompetitionGameParameters | Mapping[str, Any] | None = None,
    settings: OpenLoopSolverSettings | Mapping[str, Any] | None = None,
) -> OpenLoopNashResult:
    """Solve the example and attach transparent arms-exposure proxy metrics."""

    result = solve_open_loop_nash(build_arms_competition_game(parameters), settings)
    aggregate_stock = np.sum(result.trajectory.states, axis=1)
    times = result.trajectory.times
    exposure = float(trapezoid(aggregate_stock, times))
    risk_proxy = float(trapezoid(np.square(aggregate_stock), times) / result.model.horizon)
    return result.with_event_metrics(
        aggregate_armament_exposure=exposure,
        escalation_risk_proxy=risk_proxy,
    )


def describe() -> dict[str, Any]:
    """Describe the example without conflating it with Richardson dynamics."""

    return {
        "model_name": MODEL_NAME,
        "model_type": "finite_horizon_two_player_linear_quadratic",
        "differential_game": True,
        "solution_concept": "finite-horizon open-loop Nash equilibrium",
        "strategic_controls": "unconstrained net buildup or drawdown rates",
        "player_objective_functionals": True,
        "parameter_status": "illustrative; not historically calibrated",
        "commitment_timing": "simultaneous open-loop policies fixed at time zero",
        "proxies": [
            (
                "squared strategic imbalance (a symmetric proxy, not a one-sided "
                "disadvantage loss)"
            ),
            "squared aggregate armament",
            "aggregate-armament exposure",
            "modeled escalation-risk proxy",
        ],
        "unsupported": [
            "control bounds and state constraints",
            "feedback Nash equilibrium",
            "Stackelberg commitment",
            "stochastic dynamics and incomplete information",
        ],
        "distinct_from": (
            "Richardson arms-race dynamics, which remains a nonstrategic affine ODE"
        ),
    }


__all__ = [
    "MODEL_NAME",
    "ArmsCompetitionGameParameters",
    "build_arms_competition_game",
    "describe",
    "solve_arms_competition_game",
]
