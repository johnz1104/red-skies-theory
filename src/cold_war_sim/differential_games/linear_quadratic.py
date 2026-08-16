"""Validated deterministic two-player finite-horizon LQ differential games."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from cold_war_sim.core.types import validate_stable_id

from .base import finite_float, reject_unknown_keys

FloatArray = NDArray[np.float64]
MODEL_TYPE = "finite_horizon_two_player_linear_quadratic"


def _array(value: object, *, name: str, ndim: int) -> FloatArray:
    """Convert nested numeric input while rejecting booleans and ragged data."""

    def reject_boolean(item: object) -> None:
        if isinstance(item, bool):
            raise TypeError(f"{name} must contain real numbers, not booleans")
        if isinstance(item, (list, tuple)):
            for child in item:
                reject_boolean(child)

    reject_boolean(value)
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a rectangular numeric array") from error
    if result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got shape {result.shape}")
    if any(size == 0 for size in result.shape):
        raise ValueError(f"{name} dimensions must be nonempty")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(result, dtype=float, copy=True)
    result.setflags(write=False)
    return cast(FloatArray, result)


def _symmetric_matrix(value: object, *, name: str, size: int) -> FloatArray:
    matrix = _array(value, name=name, ndim=2)
    if matrix.shape != (size, size):
        raise ValueError(f"{name} must have shape {(size, size)}, got {matrix.shape}")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    symmetric = np.array(0.5 * (matrix + matrix.T), dtype=float)
    symmetric.setflags(write=False)
    return cast(FloatArray, symmetric)


@dataclass(frozen=True, slots=True)
class LQPlayerObjective:
    """One player's convex quadratic running and terminal costs."""

    state_cost: FloatArray
    control_cost: FloatArray
    terminal_state_cost: FloatArray

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        state_dimension: int,
        control_dimension: int,
        player_id: str,
    ) -> LQPlayerObjective:
        data = reject_unknown_keys(
            dict(values),
            allowed={"Q", "R", "Q_terminal"},
            required={"Q", "R", "Q_terminal"},
            name=f"objective for {player_id!r}",
        )
        state = _symmetric_matrix(data["Q"], name=f"{player_id}.Q", size=state_dimension)
        control = _symmetric_matrix(
            data["R"], name=f"{player_id}.R", size=control_dimension
        )
        terminal = _symmetric_matrix(
            data["Q_terminal"],
            name=f"{player_id}.Q_terminal",
            size=state_dimension,
        )
        for matrix, label in ((state, "Q"), (terminal, "Q_terminal")):
            minimum = float(np.min(np.linalg.eigvalsh(matrix)))
            if minimum < -1e-10:
                raise ValueError(
                    f"{player_id}.{label} must be positive semidefinite; "
                    f"minimum eigenvalue is {minimum}"
                )
        minimum_control = float(np.min(np.linalg.eigvalsh(control)))
        if minimum_control <= 1e-12:
            raise ValueError(
                f"{player_id}.R must be positive definite; "
                f"minimum eigenvalue is {minimum_control}"
            )
        return cls(state, control, terminal)

    def to_dict(self) -> dict[str, Any]:
        return {
            "Q": self.state_cost.tolist(),
            "R": self.control_cost.tolist(),
            "Q_terminal": self.terminal_state_cost.tolist(),
        }


@dataclass(frozen=True, slots=True)
class LinearQuadraticDifferentialGame:
    """The deliberately scoped class supported by the open-loop Nash solver.

    Controls are unconstrained, deterministic, and selected as complete
    open-loop time paths at time zero. State/control bounds, stochastic terms,
    incomplete information, and feedback strategies are not represented.
    """

    model_id: str
    player_ids: tuple[str, str]
    horizon: float
    initial_state: FloatArray
    state_matrix: FloatArray
    control_matrices: Mapping[str, FloatArray]
    affine_vector: FloatArray
    objectives: Mapping[str, LQPlayerObjective]
    parameter_status: str = "ILLUSTRATIVE"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> LinearQuadraticDifferentialGame:
        data = reject_unknown_keys(
            dict(values),
            allowed={
                "model_type",
                "model_id",
                "player_ids",
                "horizon",
                "initial_state",
                "dynamics",
                "objectives",
                "parameter_status",
            },
            required={
                "model_type",
                "model_id",
                "player_ids",
                "horizon",
                "initial_state",
                "dynamics",
                "objectives",
            },
            name="linear-quadratic model",
        )
        if data["model_type"] != MODEL_TYPE:
            raise ValueError(
                f"model_type must be {MODEL_TYPE!r}; got {data['model_type']!r}"
            )
        model_id = validate_stable_id(str(data["model_id"]), field_name="model id")
        raw_players = data["player_ids"]
        if not isinstance(raw_players, (list, tuple)) or len(raw_players) != 2:
            raise ValueError("player_ids must contain exactly two identifiers")
        if any(not isinstance(player, str) for player in raw_players):
            raise TypeError("player_ids must contain strings")
        players = tuple(
            validate_stable_id(player, field_name="player id") for player in raw_players
        )
        if players[0] == players[1]:
            raise ValueError("player_ids must be distinct")

        horizon = finite_float(data["horizon"], name="horizon")
        if horizon <= 0.0:
            raise ValueError("horizon must be strictly positive")
        initial = _array(data["initial_state"], name="initial_state", ndim=1)
        state_dimension = initial.shape[0]

        dynamics = reject_unknown_keys(
            data["dynamics"],
            allowed={"A", "B", "c"},
            required={"A", "B", "c"},
            name="dynamics",
        )
        state_matrix = _array(dynamics["A"], name="dynamics.A", ndim=2)
        if state_matrix.shape != (state_dimension, state_dimension):
            raise ValueError(
                "dynamics.A must be square and match initial_state; "
                f"expected {(state_dimension, state_dimension)}, got {state_matrix.shape}"
            )
        affine = _array(dynamics["c"], name="dynamics.c", ndim=1)
        if affine.shape != (state_dimension,):
            raise ValueError(
                f"dynamics.c must have shape {(state_dimension,)}, got {affine.shape}"
            )
        raw_controls = dynamics["B"]
        if not isinstance(raw_controls, dict) or set(raw_controls) != set(players):
            raise ValueError("dynamics.B keys must exactly match player_ids")
        controls: dict[str, FloatArray] = {}
        for player in players:
            matrix = _array(raw_controls[player], name=f"dynamics.B.{player}", ndim=2)
            if matrix.shape[0] != state_dimension:
                raise ValueError(
                    f"dynamics.B.{player} must have {state_dimension} rows, "
                    f"got {matrix.shape}"
                )
            controls[player] = matrix

        raw_objectives = data["objectives"]
        if not isinstance(raw_objectives, dict) or set(raw_objectives) != set(players):
            raise ValueError("objectives keys must exactly match player_ids")
        objectives: dict[str, LQPlayerObjective] = {}
        for player in players:
            raw_objective = raw_objectives[player]
            if not isinstance(raw_objective, Mapping):
                raise TypeError(f"objective for {player!r} must be a mapping")
            objectives[player] = LQPlayerObjective.from_mapping(
                raw_objective,
                state_dimension=state_dimension,
                control_dimension=controls[player].shape[1],
                player_id=player,
            )
        status = data.get("parameter_status", "ILLUSTRATIVE")
        if status not in {"ILLUSTRATIVE", "NORMALIZED"}:
            raise ValueError("parameter_status must be ILLUSTRATIVE or NORMALIZED")
        return cls(
            model_id=model_id,
            player_ids=(players[0], players[1]),
            horizon=horizon,
            initial_state=initial,
            state_matrix=state_matrix,
            control_matrices=MappingProxyType(controls),
            affine_vector=affine,
            objectives=MappingProxyType(objectives),
            parameter_status=status,
        )

    @property
    def state_dimension(self) -> int:
        return int(self.initial_state.shape[0])

    @property
    def control_dimensions(self) -> Mapping[str, int]:
        return MappingProxyType(
            {player: int(self.control_matrices[player].shape[1]) for player in self.player_ids}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": MODEL_TYPE,
            "model_id": self.model_id,
            "player_ids": list(self.player_ids),
            "horizon": self.horizon,
            "initial_state": self.initial_state.tolist(),
            "dynamics": {
                "A": self.state_matrix.tolist(),
                "B": {
                    player: self.control_matrices[player].tolist()
                    for player in self.player_ids
                },
                "c": self.affine_vector.tolist(),
            },
            "objectives": {
                player: self.objectives[player].to_dict() for player in self.player_ids
            },
            "parameter_status": self.parameter_status,
        }


__all__ = [
    "MODEL_TYPE",
    "LQPlayerObjective",
    "LinearQuadraticDifferentialGame",
]
