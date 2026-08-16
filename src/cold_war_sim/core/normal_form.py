"""Strict two-player finite normal-form representation."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np

from .probability import DEFAULT_TOLERANCE, validate_probability_distribution
from .types import Player, SerializableMixin, frozen_mapping, labels, validate_stable_id


@dataclass(frozen=True)
class MixedStrategy(SerializableMixin):
    player_id: str
    probabilities: Mapping[str, float]
    tolerance: float = DEFAULT_TOLERANCE

    def __post_init__(self) -> None:
        validate_stable_id(self.player_id, field_name="strategy player id")
        if not self.probabilities:
            raise ValueError("a mixed strategy must contain at least one action")
        sorted_distribution = dict(sorted(self.probabilities.items()))
        for action_id in sorted_distribution:
            validate_stable_id(action_id, field_name="mixed-strategy action id")
        values = validate_probability_distribution(
            list(sorted_distribution.values()),
            tolerance=self.tolerance,
            name=f"mixed strategy for {self.player_id!r}",
        )
        object.__setattr__(
            self,
            "probabilities",
            frozen_mapping(
                {
                    action_id: values[index]
                    for index, action_id in enumerate(sorted_distribution)
                }
            ),
        )

    @classmethod
    def pure(
        cls, player_id: str, actions: Sequence[str], chosen_action: str
    ) -> MixedStrategy:
        actions_tuple = tuple(actions)
        if chosen_action not in actions_tuple:
            raise ValueError("chosen action must belong to the supplied action set")
        return cls(
            player_id,
            {action: float(action == chosen_action) for action in actions_tuple},
        )


StrategyInput: TypeAlias = (
    MixedStrategy | Mapping[str, float] | Sequence[float] | np.ndarray[Any, Any]
)


@dataclass(frozen=True)
class NormalFormGame(SerializableMixin):
    """A validated two-player finite normal-form game.

    ``payoffs`` has shape ``(actions_player_0, actions_player_1, 2)``.
    It is copied and marked read-only, preventing mutation after validation.
    """

    players: tuple[str | Player, str | Player]
    action_sets: tuple[Sequence[str], Sequence[str]]
    payoffs: Any

    def __post_init__(self) -> None:
        if len(self.players) != 2:
            raise ValueError("NormalFormGame supports exactly two players")
        player_ids = labels(self.players)
        if len(player_ids) != 2:
            raise ValueError("NormalFormGame requires exactly two distinct players")
        action_sets = tuple(labels(action_set) for action_set in self.action_sets)
        if len(action_sets) != 2:
            raise ValueError("NormalFormGame requires exactly two action sets")
        array = np.array(self.payoffs, dtype=float, copy=True)
        expected_shape = (len(action_sets[0]), len(action_sets[1]), 2)
        if array.shape != expected_shape:
            raise ValueError(
                f"payoffs must have shape {expected_shape}; observed {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("all normal-form payoffs must be finite")
        array.setflags(write=False)
        object.__setattr__(self, "players", player_ids)
        object.__setattr__(self, "action_sets", action_sets)
        object.__setattr__(self, "payoffs", array)

    @property
    def player_ids(self) -> tuple[str, str]:
        return self.players  # type: ignore[return-value]

    def player_index(self, player: int | str | Player) -> int:
        if isinstance(player, int) and not isinstance(player, bool):
            if player not in (0, 1):
                raise IndexError("player index must be 0 or 1")
            return player
        identifier = player.id if isinstance(player, Player) else str(player)
        try:
            return self.player_ids.index(identifier)
        except ValueError as error:
            raise KeyError(f"unknown player {identifier!r}") from error

    def action_index(self, player: int | str | Player, action: int | str) -> int:
        index = self.player_index(player)
        if isinstance(action, int) and not isinstance(action, bool):
            if action < 0 or action >= len(self.action_sets[index]):
                raise IndexError(
                    f"action index {action} is out of range for player {index}"
                )
            return action
        try:
            return self.action_sets[index].index(str(action))
        except ValueError as error:
            raise KeyError(
                f"unknown action {action!r} for player {self.player_ids[index]!r}"
            ) from error

    def payoff(self, action_profile: Sequence[int | str]) -> tuple[float, float]:
        if len(action_profile) != 2:
            raise ValueError("an action profile must contain exactly two actions")
        row = self.action_index(0, action_profile[0])
        column = self.action_index(1, action_profile[1])
        return (
            float(self.payoffs[row, column, 0]),
            float(self.payoffs[row, column, 1]),
        )

    def validate_profile(
        self,
        profile: Sequence[StrategyInput],
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(profile) != 2:
            raise ValueError("a mixed profile must contain one strategy per player")
        vectors: list[np.ndarray] = []
        for player, supplied in enumerate(profile):
            actions = self.action_sets[player]
            if isinstance(supplied, MixedStrategy):
                if supplied.player_id != self.player_ids[player]:
                    raise ValueError("mixed strategy is assigned to the wrong player")
                distribution = supplied.probabilities
                if set(distribution) != set(actions):
                    raise ValueError(
                        "mixed-strategy actions must exactly match the game"
                    )
                raw = [distribution[action] for action in actions]
            elif isinstance(supplied, Mapping):
                if set(supplied) != set(actions):
                    raise ValueError("profile actions must exactly match the game")
                raw = [supplied[action] for action in actions]
            else:
                raw = list(supplied)
                if len(raw) != len(actions):
                    raise ValueError("profile probability vector has the wrong length")
            values = validate_probability_distribution(
                raw, name=f"strategy for {self.player_ids[player]!r}"
            )
            vectors.append(np.array(values, dtype=float))
        return vectors[0], vectors[1]

    def expected_payoffs(
        self,
        profile: Sequence[StrategyInput],
    ) -> tuple[float, float]:
        row, column = self.validate_profile(profile)
        joint = np.outer(row, column)
        return (
            float(np.sum(joint * self.payoffs[:, :, 0])),
            float(np.sum(joint * self.payoffs[:, :, 1])),
        )

    def pure_action_payoffs(
        self,
        player: int | str | Player,
        opponent_strategy: StrategyInput,
    ) -> tuple[float, ...]:
        player_index = self.player_index(player)
        opponent = 1 - player_index
        dummy: list[StrategyInput] = [
            np.ones(len(self.action_sets[index])) / len(self.action_sets[index])
            for index in range(2)
        ]
        dummy[opponent] = opponent_strategy
        vectors = self.validate_profile(dummy)
        opponent_vector = vectors[opponent]
        if player_index == 0:
            return tuple(
                float(value) for value in self.payoffs[:, :, 0] @ opponent_vector
            )
        return tuple(
            float(value) for value in self.payoffs[:, :, 1].T @ opponent_vector
        )

    def best_responses(
        self,
        player: int | str | Player,
        opponent_strategy: StrategyInput,
        *,
        tolerance: float = 1e-10,
    ) -> tuple[tuple[str, ...], float]:
        index = self.player_index(player)
        values = self.pure_action_payoffs(index, opponent_strategy)
        best_value = max(values)
        responses = tuple(
            action
            for action, value in zip(self.action_sets[index], values, strict=True)
            if value >= best_value - tolerance
        )
        return responses, best_value

    def pure_profiles(self) -> tuple[tuple[str, str], ...]:
        return tuple(itertools.product(self.action_sets[0], self.action_sets[1]))

    def pure_nash(self, *, tolerance: float = 1e-10) -> tuple[tuple[str, str], ...]:
        equilibria: list[tuple[str, str]] = []
        for row_action, column_action in self.pure_profiles():
            row_best, _ = self.best_responses(
                0,
                MixedStrategy.pure(
                    self.player_ids[1], self.action_sets[1], column_action
                ),
                tolerance=tolerance,
            )
            column_best, _ = self.best_responses(
                1,
                MixedStrategy.pure(self.player_ids[0], self.action_sets[0], row_action),
                tolerance=tolerance,
            )
            if row_action in row_best and column_action in column_best:
                equilibria.append((row_action, column_action))
        return tuple(equilibria)

    def to_dict(self) -> dict[str, Any]:
        return {
            "players": list(self.player_ids),
            "action_sets": [list(action_set) for action_set in self.action_sets],
            "payoffs": self.payoffs.tolist(),
        }
