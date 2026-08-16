"""Richardson arms-race dynamics as a validated affine ODE system.

The model has no strategic controls or player objective functionals. It is a
two-state action-reaction baseline kept mathematically distinct from the
strategic differential game.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .parameters import NUMERICAL_CONVENIENCE, RichardsonParameters

MODEL_NAME = "Richardson arms-race dynamics"


def _real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _state_vector(
    state: Sequence[float] | NDArray[np.float64],
    *,
    require_nonnegative: bool,
) -> NDArray[np.float64]:
    array = np.asarray(state, dtype=float)
    if array.shape != (2,):
        raise ValueError(f"state must have shape (2,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("state must contain only finite values")
    if require_nonnegative and np.any(array < 0.0):
        raise ValueError("initial state must lie in the nonnegative modeled domain")
    return array


def coefficient_matrix(parameters: RichardsonParameters) -> NDArray[np.float64]:
    """Return the deterministic matrix ``A`` in ``state' = A state + b``."""

    return np.array(
        [
            [-parameters.fatigue_0, parameters.reaction_0],
            [parameters.reaction_1, -parameters.fatigue_1],
        ],
        dtype=float,
    )


def affine_vector(parameters: RichardsonParameters) -> NDArray[np.float64]:
    """Return the exogenous grievance vector ``b``."""

    return np.array([parameters.grievance_0, parameters.grievance_1], dtype=float)


def rhs(
    parameters: RichardsonParameters,
    state: Sequence[float] | NDArray[np.float64],
    time: float = 0.0,
) -> NDArray[np.float64]:
    """Evaluate the autonomous affine right-hand side.

    ``time`` is accepted for integration interfaces and validated even though
    the baseline coefficients do not vary with time.
    """

    _real("time", time)
    vector = _state_vector(state, require_nonnegative=False)
    result = coefficient_matrix(parameters) @ vector + affine_vector(parameters)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("Richardson derivative contains a non-finite value")
    return cast(NDArray[np.float64], result)


def rk4_step(
    parameters: RichardsonParameters,
    state: Sequence[float] | NDArray[np.float64],
    step_size: float,
    *,
    time: float = 0.0,
) -> NDArray[np.float64]:
    """Advance one classical fourth-order Runge--Kutta step."""

    dt = _real("step_size", step_size)
    if dt <= 0.0:
        raise ValueError("step_size must be strictly positive")
    t = _real("time", time)
    vector = _state_vector(state, require_nonnegative=False)
    k1 = rhs(parameters, vector, t)
    k2 = rhs(parameters, vector + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = rhs(parameters, vector + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = rhs(parameters, vector + dt * k3, t + dt)
    result = vector + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("RK4 step produced a non-finite state")
    return cast(NDArray[np.float64], result)


@dataclass(frozen=True, slots=True)
class RichardsonTrajectory:
    """JSON-serializable trajectory plus nonnegative-domain diagnostics."""

    times: tuple[float, ...]
    states: tuple[tuple[float, float], ...]
    projection_requested: bool
    projection_applied: bool
    projected_steps: int
    projected_components: int
    domain_violation_steps: int
    minimum_unprojected_state: float
    warnings: tuple[str, ...]

    @property
    def final_state(self) -> tuple[float, float]:
        return self.states[-1]

    @property
    def stayed_in_nonnegative_domain(self) -> bool:
        return self.domain_violation_steps == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": MODEL_NAME,
            "times": list(self.times),
            "states": [list(state) for state in self.states],
            "final_state": list(self.final_state),
            "nonnegative_domain": {
                "projection_requested": self.projection_requested,
                "projection_applied": self.projection_applied,
                "projected_steps": self.projected_steps,
                "projected_components": self.projected_components,
                "domain_violation_steps": self.domain_violation_steps,
                "minimum_unprojected_state": self.minimum_unprojected_state,
                "stayed_in_domain_without_projection": self.stayed_in_nonnegative_domain,
            },
            "warnings": list(self.warnings),
        }


def integrate_rk4(
    parameters: RichardsonParameters | Mapping[str, Any] | None,
    initial_state: Sequence[float] | NDArray[np.float64],
    *,
    end_time: float,
    step_size: float,
    start_time: float = 0.0,
    project_nonnegative: bool = False,
) -> RichardsonTrajectory:
    """Integrate to ``end_time``, optionally projecting negative components.

    Projection is never silent: the returned diagnostics count every proposed
    state that left the domain and every component set to zero.  Without
    projection, negative states are retained and explicitly warned about.
    """

    if isinstance(parameters, RichardsonParameters):
        params = parameters
    else:
        params = RichardsonParameters.from_mapping(parameters)
    start = _real("start_time", start_time)
    end = _real("end_time", end_time)
    dt_requested = _real("step_size", step_size)
    if end < start:
        raise ValueError("end_time must be greater than or equal to start_time")
    if dt_requested <= 0.0:
        raise ValueError("step_size must be strictly positive")
    if not isinstance(project_nonnegative, bool):
        raise TypeError("project_nonnegative must be a boolean")

    current = _state_vector(initial_state, require_nonnegative=True).copy()
    current_time = start
    times = [current_time]
    states = [(float(current[0]), float(current[1]))]
    projected_steps = 0
    projected_components = 0
    violation_steps = 0
    minimum_unprojected = float(np.min(current))

    # The subtraction guard also handles a final step shorter than the nominal
    # step size without accumulating an extra near-zero integration step.
    time_tolerance = max(1e-14, abs(end) * 1e-14)
    while current_time < end - time_tolerance:
        dt = min(dt_requested, end - current_time)
        proposed = rk4_step(params, current, dt, time=current_time)
        minimum_unprojected = min(minimum_unprojected, float(np.min(proposed)))
        negative = proposed < 0.0
        if np.any(negative):
            violation_steps += 1
            if project_nonnegative:
                projected_steps += 1
                projected_components += int(np.count_nonzero(negative))
                proposed = np.maximum(proposed, 0.0)
        current = proposed
        current_time = min(end, current_time + dt)
        times.append(float(current_time))
        states.append((float(current[0]), float(current[1])))

    warnings: list[str] = []
    if violation_steps and project_nonnegative:
        warnings.append(
            "Nonnegative projection altered the unconstrained RK4 trajectory; "
            "inspect projection diagnostics."
        )
    elif violation_steps:
        warnings.append("The trajectory left the modeled nonnegative domain.")

    return RichardsonTrajectory(
        times=tuple(times),
        states=tuple(states),
        projection_requested=project_nonnegative,
        projection_applied=projected_components > 0,
        projected_steps=projected_steps,
        projected_components=projected_components,
        domain_violation_steps=violation_steps,
        minimum_unprojected_state=minimum_unprojected,
        warnings=tuple(warnings),
    )


@dataclass(frozen=True, slots=True)
class FixedPointResult:
    exists: bool
    unique: bool
    singular: bool
    point: tuple[float, float] | None
    representative_point: tuple[float, float] | None
    residual_norm: float | None
    in_nonnegative_domain: bool | None
    rank_matrix: int
    rank_augmented: int
    tolerance: float
    warning: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "exists": self.exists,
            "unique": self.unique,
            "singular": self.singular,
            "mathematical_fixed_point": list(self.point) if self.point is not None else None,
            "representative_point": (
                list(self.representative_point)
                if self.representative_point is not None
                else None
            ),
            "residual_norm": self.residual_norm,
            "in_nonnegative_domain": self.in_nonnegative_domain,
            "rank_matrix": self.rank_matrix,
            "rank_augmented": self.rank_augmented,
            "tolerance": self.tolerance,
            "warning": self.warning,
        }


def affine_fixed_point(
    parameters: RichardsonParameters | Mapping[str, Any] | None = None,
    *,
    tolerance: float = 1e-10,
) -> FixedPointResult:
    """Solve ``A state + b = 0`` and report singular/domain cases honestly."""

    if isinstance(parameters, RichardsonParameters):
        params = parameters
    else:
        params = RichardsonParameters.from_mapping(parameters)
    tol = _real("tolerance", tolerance)
    if tol <= 0.0:
        raise ValueError("tolerance must be strictly positive")
    matrix = coefficient_matrix(params)
    vector = affine_vector(params)
    augmented = np.column_stack((matrix, -vector))
    rank_matrix = int(np.linalg.matrix_rank(matrix, tol=tol))
    rank_augmented = int(np.linalg.matrix_rank(augmented, tol=tol))
    singular = rank_matrix < 2

    if not singular:
        point_array = np.linalg.solve(matrix, -vector)
        residual = float(np.linalg.norm(matrix @ point_array + vector, ord=2))
        point = (float(point_array[0]), float(point_array[1]))
        in_domain = bool(np.all(point_array >= -tol))
        warning = None
        if not in_domain:
            warning = "The mathematical fixed point is outside the nonnegative modeled domain."
        return FixedPointResult(
            exists=True,
            unique=True,
            singular=False,
            point=point,
            representative_point=point,
            residual_norm=residual,
            in_nonnegative_domain=in_domain,
            rank_matrix=rank_matrix,
            rank_augmented=rank_augmented,
            tolerance=tol,
            warning=warning,
        )

    if rank_augmented > rank_matrix:
        return FixedPointResult(
            exists=False,
            unique=False,
            singular=True,
            point=None,
            representative_point=None,
            residual_norm=None,
            in_nonnegative_domain=None,
            rank_matrix=rank_matrix,
            rank_augmented=rank_augmented,
            tolerance=tol,
            warning="The singular affine system is inconsistent and has no fixed point.",
        )

    representative, *_ = np.linalg.lstsq(matrix, -vector, rcond=tol)
    residual = float(np.linalg.norm(matrix @ representative + vector, ord=2))
    representative_point = (float(representative[0]), float(representative[1]))
    return FixedPointResult(
        exists=True,
        unique=False,
        singular=True,
        point=None,
        representative_point=representative_point,
        residual_norm=residual,
        in_nonnegative_domain=None,
        rank_matrix=rank_matrix,
        rank_augmented=rank_augmented,
        tolerance=tol,
        warning="The singular affine system has a continuum of fixed points; no unique point exists.",
    )


@dataclass(frozen=True, slots=True)
class EigenvalueRecord:
    real: float
    imaginary: float

    def to_dict(self) -> dict[str, float]:
        return {"real": self.real, "imaginary": self.imaginary}


@dataclass(frozen=True, slots=True)
class StabilityResult:
    classification: str
    eigenvalues: tuple[EigenvalueRecord, EigenvalueRecord]
    tolerance: float
    hyperbolic: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "eigenvalues": [value.to_dict() for value in self.eigenvalues],
            "tolerance": self.tolerance,
            "hyperbolic": self.hyperbolic,
            "method": "eigenvalue real-part classification of the affine system matrix",
        }


def eigenvalue_stability(
    parameters: RichardsonParameters | Mapping[str, Any] | None = None,
    *,
    tolerance: float = 1e-10,
) -> StabilityResult:
    """Classify local/global affine stability using eigenvalue real parts."""

    if isinstance(parameters, RichardsonParameters):
        params = parameters
    else:
        params = RichardsonParameters.from_mapping(parameters)
    tol = _real("tolerance", tolerance)
    if tol <= 0.0:
        raise ValueError("tolerance must be strictly positive")
    raw_values = np.linalg.eigvals(coefficient_matrix(params))
    ordered = sorted(
        (complex(value) for value in raw_values),
        key=lambda value: (float(value.real), float(value.imag)),
    )
    real_parts = np.array([value.real for value in ordered], dtype=float)
    has_positive = bool(np.any(real_parts > tol))
    has_negative = bool(np.any(real_parts < -tol))
    has_zero = bool(np.any(np.abs(real_parts) <= tol))
    if has_positive and has_negative:
        classification = "saddle"
    elif has_positive:
        classification = "unstable"
    elif has_negative and not has_zero:
        classification = "asymptotically_stable"
    else:
        classification = "marginal_or_nonhyperbolic"
    records = tuple(
        EigenvalueRecord(real=float(value.real), imaginary=float(value.imag))
        for value in ordered
    )
    return StabilityResult(
        classification=classification,
        eigenvalues=(records[0], records[1]),
        tolerance=tol,
        hyperbolic=not has_zero,
    )


@dataclass(frozen=True, slots=True)
class RichardsonAnalysis:
    parameters: RichardsonParameters
    fixed_point: FixedPointResult
    stability: StabilityResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": MODEL_NAME,
            "parameters": self.parameters.to_dict(),
            "fixed_point": self.fixed_point.to_dict(),
            "stability": self.stability.to_dict(),
            "assumptions": [
                RichardsonParameters.ASSUMPTION_LABEL,
                f"stability tolerance is a {NUMERICAL_CONVENIENCE}",
            ],
        }


def analyze_richardson(
    parameters: RichardsonParameters | Mapping[str, Any] | None = None,
    *,
    tolerance: float = 1e-10,
) -> RichardsonAnalysis:
    """Return fixed-point and stability diagnostics for one parameter set."""

    if isinstance(parameters, RichardsonParameters):
        params = parameters
    else:
        params = RichardsonParameters.from_mapping(parameters)
    return RichardsonAnalysis(
        parameters=params,
        fixed_point=affine_fixed_point(params, tolerance=tolerance),
        stability=eigenvalue_stability(params, tolerance=tolerance),
    )


@dataclass(frozen=True, slots=True)
class RichardsonArmsRaceDynamics:
    """Small immutable facade around the validated Richardson functions."""

    parameters: RichardsonParameters = field(default_factory=RichardsonParameters)

    def rhs(self, state: Sequence[float], time: float = 0.0) -> NDArray[np.float64]:
        return rhs(self.parameters, state, time)

    def rk4_step(
        self,
        state: Sequence[float],
        step_size: float,
        *,
        time: float = 0.0,
    ) -> NDArray[np.float64]:
        return rk4_step(self.parameters, state, step_size, time=time)

    def integrate(
        self,
        initial_state: Sequence[float],
        *,
        end_time: float,
        step_size: float,
        start_time: float = 0.0,
        project_nonnegative: bool = False,
    ) -> RichardsonTrajectory:
        return integrate_rk4(
            self.parameters,
            initial_state,
            end_time=end_time,
            step_size=step_size,
            start_time=start_time,
            project_nonnegative=project_nonnegative,
        )

    def analyze(self, *, tolerance: float = 1e-10) -> RichardsonAnalysis:
        return analyze_richardson(self.parameters, tolerance=tolerance)


__all__ = [
    "MODEL_NAME",
    "EigenvalueRecord",
    "FixedPointResult",
    "RichardsonAnalysis",
    "RichardsonArmsRaceDynamics",
    "RichardsonTrajectory",
    "StabilityResult",
    "affine_fixed_point",
    "affine_vector",
    "analyze_richardson",
    "coefficient_matrix",
    "eigenvalue_stability",
    "integrate_rk4",
    "rhs",
    "rk4_step",
]
