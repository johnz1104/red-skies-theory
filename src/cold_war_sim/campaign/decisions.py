"""Uniform decision requests: every choice is one index into labeled options."""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class Option:
    id: str
    label: str
    historical: bool = False


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    power: str
    kind: str
    source: str  # "strategy" or an event id
    options: tuple[Option, ...]
    role: str | None = None
    family: str | None = None
    round: int = 0

    def __post_init__(self) -> None:
        if not self.options:
            raise ValueError("a decision needs at least one option")
        if len(self.options) > MAX_OPTIONS:
            raise ValueError(f"{self.source}: {len(self.options)} options exceed {MAX_OPTIONS}")
        if self.kind not in DECISION_KINDS:
            raise ValueError(f"unknown decision kind {self.kind!r}")

    def index_of(self, option_id: str) -> int:
        for i, option in enumerate(self.options):
            if option.id == option_id:
                return i
        raise KeyError(option_id)

    def historical_index(self) -> int | None:
        for i, option in enumerate(self.options):
            if option.historical:
                return i
        return None
