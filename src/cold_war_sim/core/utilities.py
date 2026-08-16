"""Validated terminal and expected-utility representations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .probability import ensure_finite, validate_probability_distribution
from .types import Player, SerializableMixin, frozen_mapping, validate_stable_id


def player_id(player: str | Player) -> str:
    return (
        player.id
        if isinstance(player, Player)
        else validate_stable_id(player, field_name="player id")
    )


@dataclass(frozen=True)
class TerminalUtilities(SerializableMixin):
    """Utilities at one terminal history, keyed by stable player ID."""

    utilities: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.utilities:
            raise ValueError("terminal utilities must contain at least one player")
        converted: dict[str, float] = {}
        for player, utility in self.utilities.items():
            validate_stable_id(player, field_name="utility player id")
            converted[player] = ensure_finite(utility, name=f"utility[{player!r}]")
        object.__setattr__(self, "utilities", frozen_mapping(converted))

    def validate_players(self, players: Sequence[str | Player]) -> None:
        expected = {player_id(player) for player in players}
        observed = set(self.utilities)
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ValueError(
                f"terminal utilities must contain every player exactly; missing={missing}, extra={extra}"
            )

    def __getitem__(self, player: str | Player) -> float:
        return self.utilities[player_id(player)]

    def as_tuple(self, players: Sequence[str | Player]) -> tuple[float, ...]:
        self.validate_players(players)
        return tuple(self[player] for player in players)


@dataclass(frozen=True)
class ExpectedUtilities(SerializableMixin):
    utilities: Mapping[str, float]

    def __post_init__(self) -> None:
        converted = {
            validate_stable_id(
                player, field_name="expected-utility player id"
            ): ensure_finite(value, name=f"expected utility[{player!r}]")
            for player, value in self.utilities.items()
        }
        if not converted:
            raise ValueError("expected utilities must contain at least one player")
        object.__setattr__(self, "utilities", frozen_mapping(converted))


def probability_weighted_utilities(
    probabilities: Sequence[float],
    outcomes: Sequence[TerminalUtilities],
    players: Sequence[str | Player],
) -> ExpectedUtilities:
    probabilities_tuple = validate_probability_distribution(probabilities)
    outcomes_tuple = tuple(outcomes)
    if len(probabilities_tuple) != len(outcomes_tuple):
        raise ValueError("probabilities and outcomes must have the same length")
    identifiers = tuple(player_id(player) for player in players)
    totals = {identifier: 0.0 for identifier in identifiers}
    for probability, outcome in zip(probabilities_tuple, outcomes_tuple, strict=True):
        outcome.validate_players(identifiers)
        for identifier in identifiers:
            totals[identifier] += probability * outcome[identifier]
    return ExpectedUtilities(totals)
