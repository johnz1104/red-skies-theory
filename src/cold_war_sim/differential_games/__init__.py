"""Scoped, verified differential-game support for Cold War Simulator."""

from .base import BoundarySystemStatus, OpenLoopSolveError
from .counterfactuals import DifferentialSolutionComparison, compare_solutions
from .linear_quadratic import LinearQuadraticDifferentialGame, LQPlayerObjective
from .open_loop import OpenLoopSolverSettings, solve_open_loop_nash
from .results import DifferentialGameVerification, OpenLoopNashResult
from .verification import verify_open_loop_nash

__all__ = [
    "BoundarySystemStatus",
    "DifferentialGameVerification",
    "DifferentialSolutionComparison",
    "LQPlayerObjective",
    "LinearQuadraticDifferentialGame",
    "OpenLoopNashResult",
    "OpenLoopSolveError",
    "OpenLoopSolverSettings",
    "compare_solutions",
    "solve_open_loop_nash",
    "verify_open_loop_nash",
]
