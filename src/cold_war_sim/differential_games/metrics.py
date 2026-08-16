"""Auditable trajectory metrics for the illustrative arms-competition bridge."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid

from .linear_quadratic import FloatArray


@dataclass(frozen=True, slots=True)
class ArmamentTrajectoryMetrics:
    """Metrics whose definitions are stable enough to verify from a saved path."""

    aggregate_armament_exposure: float
    escalation_risk_proxy: float


def armament_trajectory_metrics(
    times: FloatArray,
    states: FloatArray,
    *,
    horizon: float,
) -> ArmamentTrajectoryMetrics:
    """Recompute the two documented arms-stock metrics from trajectory samples.

    Every state component is treated as one modeled armament/capability stock.
    The exposure is the sampled-time integral of their sum. The risk proxy is
    the sampled-time integral of the squared sum divided by the model horizon.
    These are formal-model summaries, not empirically calibrated risk measures.
    """

    if times.ndim != 1 or states.ndim != 2 or states.shape[0] != times.shape[0]:
        raise ValueError("armament metrics require aligned one-dimensional times and states")
    if states.shape[1] < 1:
        raise ValueError("armament metrics require at least one state component")
    if horizon <= 0.0 or not np.isfinite(horizon):
        raise ValueError("armament metric horizon must be finite and strictly positive")
    aggregate_stock = np.sum(states, axis=1)
    return ArmamentTrajectoryMetrics(
        aggregate_armament_exposure=float(trapezoid(aggregate_stock, times)),
        escalation_risk_proxy=float(trapezoid(np.square(aggregate_stock), times) / horizon),
    )


__all__ = ["ArmamentTrajectoryMetrics", "armament_trajectory_metrics"]
