"""Strategic differential-game model and nonstrategic Richardson baseline."""

from __future__ import annotations

from typing import Any

from .parameters import RichardsonParameters
from .richardson import (
    MODEL_NAME,
    RichardsonArmsRaceDynamics,
    affine_fixed_point,
    analyze_richardson,
    eigenvalue_stability,
    integrate_rk4,
)


def describe() -> dict[str, Any]:
    """Describe Richardson dynamics without making a strategic-game claim."""

    parameters = RichardsonParameters()
    return {
        "event": "arms_race",
        "historical_scenario": "Cold War arms competition, historically inspired",
        "framework": "Richardson arms-race dynamics",
        "components": {
            "richardson_arms_race_dynamics": {
                "model_name": MODEL_NAME,
                "state_variables": ["player_0_armament", "player_1_armament"],
                "state_domain": "nonnegative orthant, with explicit projection diagnostics",
                "equations": [
                    "x' = -fatigue_0*x + reaction_0*y + grievance_0",
                    "y' = reaction_1*x - fatigue_1*y + grievance_1",
                ],
                "implemented_analysis": [
                    "RK4 integration",
                    "affine fixed-point calculation",
                    "matrix-eigenvalue stability classification",
                    "fixed-point nonnegative-domain validation",
                ],
                "strategic_controls": False,
                "player_objective_functionals": False,
                "nash_equilibrium_solver": False,
                "differential_game": False,
            }
        },
        "default_parameters": {
            "richardson": parameters.to_dict(),
            "example_settings": {
                "richardson_initial_state": [1.0, 1.0],
                "richardson_end_time": 5.0,
                "richardson_step_size": 0.05,
                "project_nonnegative": True,
                "numerical_tolerance": 1e-10,
                "provenance": "numerical convenience",
            },
        },
        "assumptions": [
            RichardsonParameters.ASSUMPTION_LABEL,
            "Richardson coefficients are illustrative and are not fitted to historical data.",
            "The separate LQ arms model contains the strategic controls and Nash solver.",
        ],
        "calibration": "not historically calibrated; defaults are illustrative",
    }


__all__ = [
    "MODEL_NAME",
    "RichardsonArmsRaceDynamics",
    "RichardsonParameters",
    "affine_fixed_point",
    "analyze_richardson",
    "describe",
    "eigenvalue_stability",
    "integrate_rk4",
]
