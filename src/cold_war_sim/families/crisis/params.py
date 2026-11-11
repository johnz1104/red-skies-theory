"""Players, available actions, and payoff settings for a crisis."""

from __future__ import annotations

from collections.abc import Callable, Collection
from typing import Any, TypeVar

from ...games.protocol import ImmutableParams
from ...records import Record
from ..common import (
    ConfigError,
    Risk,
    Section,
    ValueTable,
    as_mapping,
    finite,
    identifier,
    risk_attitude,
    value_table,
)

NAME = "crisis"
OUTCOMES = ("a_prevails", "b_prevails", "compromise", "limited_war", "nuclear_war")
WAR_OUTCOMES = frozenset({"limited_war", "nuclear_war"})
SIDES = ("a", "b", "neutral")
STANCES = ("hard", "soft", "keep", "flip")
MOVE_KINDS = ("concede", "hold", "escalate", "deal", "fight")
HIGH_LOW_TRAITS = ("resolve", "readiness")  # private traits that are high or low


class CrisisRole(ImmutableParams, Record):
    """A player: its side and its payoff parameters.

    ``side``                      ``a``, ``b`` or ``neutral``.
    ``values``                    its payoff at each outcome label.
    ``escalation_cost``           per rung its own side adds (sides ``a`` and ``b``).
    ``audience_cost``             paid on conceding after a hard stance.
    ``high_resolve_war_factor``   scales its war payoffs when its resolve is high.
    ``risk``                      ``{alpha, beta, lambda}``: its total payoff ``x``, measured
                                  from the status quo, is valued by prospect theory as
                                  ``x^alpha`` for gains and ``-lambda (-x)^beta`` for losses.
    """

    def __init__(
        self,
        id: str,
        side: str,
        values: ValueTable,
        escalation_cost: float = 0.0,
        audience_cost: float = 0.0,
        high_resolve_war_factor: float = 1.0,
        risk: Risk | None = None,
    ) -> None:
        self.id = id
        self.side = side
        self.values = values
        self.escalation_cost = escalation_cost
        self.audience_cost = audience_cost
        self.high_resolve_war_factor = high_resolve_war_factor
        self.risk = risk

    @classmethod
    def from_config(cls, role_id: str, data: Any) -> CrisisRole:
        with Section(f"crisis.roles.{role_id}", data) as s:
            side = s.choice("side", SIDES)
            return cls(
                id=identifier(s.name, role_id),
                side=side,
                values=value_table(s.at("values"), s.get("values"), OUTCOMES),
                escalation_cost=s.nonnegative("escalation_cost", 0.0),
                audience_cost=s.nonnegative("audience_cost", 0.0),
                high_resolve_war_factor=s.nonnegative("high_resolve_war_factor", 1.0),
                risk=risk_attitude(s.at("risk"), s.get("risk")) if s.has("risk") else None,
            )


class Option(ImmutableParams, Record):
    """An action at a decision node.

    ``costs: {role: x}`` is charged to the named roles when the action is taken (a negative
    cost is a gain).
    """

    def __init__(self, id: str, costs: dict[str, float]) -> None:
        self.id = id
        self.costs = costs


class SignalOption(Option):
    """A public message.

    ``stance`` sets the sender's stance to ``hard`` or ``soft``, keeps it (``keep``, the
    default) or flips it (``flip``); ``cost`` is the sender's cost of sending it.
    """

    def __init__(self, id: str, costs: dict[str, float], stance: str, cost: float) -> None:
        super().__init__(id, costs)
        self.stance = stance
        self.cost = cost

    @classmethod
    def from_config(cls, s: Section, role_ids: Collection[str]) -> SignalOption:
        return cls(
            id=_option_id(s),
            costs=_costs(s, role_ids),
            stance=s.choice("stance", STANCES, "keep"),
            cost=s.nonnegative("cost", 0.0),
        )


class MoveOption(Option):
    """A move on the escalation ladder.

    ``kind`` is ``concede``, ``hold``, ``escalate``, ``deal`` (offer one) or ``fight``;
    ``rungs`` is the size of an escalation (default 1); ``control`` is a control order, the
    share of the hold cost lifted from its side's officers (default 0).
    """

    def __init__(
        self, id: str, costs: dict[str, float], kind: str, rungs: int, control: float
    ) -> None:
        super().__init__(id, costs)
        self.kind = kind
        self.rungs = rungs
        self.control = control

    @classmethod
    def from_config(cls, s: Section, role_ids: Collection[str]) -> MoveOption:
        return cls(
            id=_option_id(s),
            costs=_costs(s, role_ids),
            kind=s.choice("kind", MOVE_KINDS),
            rungs=s.positive_int("rungs", 1),
            control=s.probability("control", 0.0),
        )


class PartnerOption(Option):
    """A third party's backing or pressure.

    ``target`` is the side it bears on (default ``neutral``); ``continue_cost`` is what
    that side pays unless it concedes; ``risk_multiplier`` scales the probability of
    accidents and incidents from then on (default 1).
    """

    def __init__(
        self,
        id: str,
        costs: dict[str, float],
        target: str,
        continue_cost: float,
        risk_multiplier: float,
    ) -> None:
        super().__init__(id, costs)
        self.target = target
        self.continue_cost = continue_cost
        self.risk_multiplier = risk_multiplier

    @classmethod
    def from_config(cls, s: Section, role_ids: Collection[str]) -> PartnerOption:
        return cls(
            id=_option_id(s),
            costs=_costs(s, role_ids),
            target=s.choice("target", SIDES, "neutral"),
            continue_cost=s.nonnegative("continue_cost", 0.0),
            risk_multiplier=s.nonnegative("risk_multiplier", 1.0),
        )


AnyOption = TypeVar("AnyOption", bound=Option)


def read_options(
    stage: Section,
    read: Callable[[Section, Collection[str]], AnyOption],
    role_ids: Collection[str],
) -> dict[str, AnyOption]:
    """A stage's options by id: at least one, with unique ids."""

    name = stage.at("options")
    options = []
    for index, raw in enumerate(stage.sequence("options")):
        with Section(f"{name}[{index}]", raw) as s:
            options.append(read(s, role_ids))
    by_id = {option.id: option for option in options}
    if not options or len(by_id) != len(options):
        raise ConfigError(f"{name} needs at least one option and unique ids")
    return by_id


def _option_id(s: Section) -> str:
    return identifier(s.at("id"), s.get("id"))


def _costs(s: Section, role_ids: Collection[str]) -> dict[str, float]:
    """An option's ``costs``; each role it names must be in the game."""

    name = s.at("costs")
    listed = as_mapping(name, s.get("costs", {}))
    for role in listed:
        if role not in role_ids:
            raise ConfigError(f"{name} refers to unknown role {role!r}")
    return {str(role): finite(f"{name}.{role}", cost) for role, cost in listed.items()}
