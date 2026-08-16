"""Immutable sampled trajectories for differential-game solutions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from .linear_quadratic import FloatArray, _array


@dataclass(frozen=True, slots=True)
class DifferentialGameTrajectory:
    times: FloatArray
    states: FloatArray
    controls: Mapping[str, FloatArray]
    costates: Mapping[str, FloatArray]

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        player_ids: tuple[str, str],
        state_dimension: int,
        control_dimensions: Mapping[str, int],
    ) -> DifferentialGameTrajectory:
        if set(values) != {"times", "states", "controls", "costates"}:
            raise ValueError(
                "trajectory fields must be exactly times, states, controls, and costates"
            )
        times = _array(values["times"], name="trajectory.times", ndim=1)
        if times.shape[0] < 2 or np.any(np.diff(times) <= 0.0):
            raise ValueError("trajectory.times must be strictly increasing with at least two points")
        states = _array(values["states"], name="trajectory.states", ndim=2)
        expected_state_shape = (times.shape[0], state_dimension)
        if states.shape != expected_state_shape:
            raise ValueError(
                f"trajectory.states must have shape {expected_state_shape}, got {states.shape}"
            )
        raw_controls = values["controls"]
        raw_costates = values["costates"]
        if not isinstance(raw_controls, Mapping) or set(raw_controls) != set(player_ids):
            raise ValueError("trajectory.controls keys must exactly match player_ids")
        if not isinstance(raw_costates, Mapping) or set(raw_costates) != set(player_ids):
            raise ValueError("trajectory.costates keys must exactly match player_ids")
        controls: dict[str, FloatArray] = {}
        costates: dict[str, FloatArray] = {}
        for player in player_ids:
            controls[player] = _array(
                raw_controls[player], name=f"trajectory.controls.{player}", ndim=2
            )
            expected_control_shape = (times.shape[0], control_dimensions[player])
            if controls[player].shape != expected_control_shape:
                raise ValueError(
                    f"trajectory.controls.{player} must have shape "
                    f"{expected_control_shape}, got {controls[player].shape}"
                )
            costates[player] = _array(
                raw_costates[player], name=f"trajectory.costates.{player}", ndim=2
            )
            if costates[player].shape != expected_state_shape:
                raise ValueError(
                    f"trajectory.costates.{player} must have shape "
                    f"{expected_state_shape}, got {costates[player].shape}"
                )
        return cls(
            times=times,
            states=states,
            controls=MappingProxyType(controls),
            costates=MappingProxyType(costates),
        )

    @property
    def terminal_state(self) -> FloatArray:
        return cast(FloatArray, self.states[-1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "times": self.times.tolist(),
            "states": self.states.tolist(),
            "controls": {
                player: self.controls[player].tolist() for player in sorted(self.controls)
            },
            "costates": {
                player: self.costates[player].tolist() for player in sorted(self.costates)
            },
        }


__all__ = ["DifferentialGameTrajectory"]
