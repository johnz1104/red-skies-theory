"""Uniform decision requests: every choice is one index into labeled options."""

from __future__ import annotations

from ...records import Record

MAX_OPTIONS = 20

STRATEGY_KINDS = ("strategy_posture", "strategy_region", "strategy_mode", "strategy_diplomacy")
FAMILY_KINDS = (
    "subgame_crisis",
    "subgame_bargaining",
    "subgame_intervention",
    "subgame_attrition",
    "subgame_alignment",
)
EVENT_KINDS = ("event_decision",)  # a decision event's choice (catalog kind "decision")
DECISION_KINDS = STRATEGY_KINDS + FAMILY_KINDS + EVENT_KINDS


class Option(Record):
    """One choice of a decision; ``historical`` marks the choice history made."""

    def __init__(
        self,
        id: str,
        label: str,
        historical: bool = False,
    ) -> None:
        self.id = id
        self.label = label
        self.historical = historical


class DecisionRequest(Record):
    """A decision put to a power: a strategy turn, a move in a sub-game, or a decision event."""

    def __init__(
        self,
        power: str,
        kind: str,
        source: str,
        options: tuple[Option, ...],
        role: str | None = None,
        family: str | None = None,
        round: int = 0,
    ) -> None:
        self.power = power
        self.kind = kind
        self.source = source  # "strategy" or an event id
        self.options = options
        self.role = role
        self.family = family
        self.round = round
        self._check()

    def _check(self) -> None:
        if not self.options:
            raise ValueError("a decision needs at least one option")
        if len(self.options) > MAX_OPTIONS:
            raise ValueError(f"{self.source}: {len(self.options)} options exceed {MAX_OPTIONS}")
        if self.kind not in DECISION_KINDS:
            raise ValueError(f"unknown decision kind {self.kind!r}")

    def index_of(self, option_id: str) -> int:
        """The index of the option with id *option_id*."""

        for i, option in enumerate(self.options):
            if option.id == option_id:
                return i
        raise KeyError(option_id)

    def historical_index(self) -> int | None:
        """The index of the historical option, if one is marked."""

        for i, option in enumerate(self.options):
            if option.historical:
                return i
        return None
