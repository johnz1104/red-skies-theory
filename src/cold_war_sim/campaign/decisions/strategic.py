"""Yearly strategy decisions: posture, region focus, focus mode, diplomacy.

With the crisis generator on, the focus mode may also be ``challenge`` (a standoff with
the region's leading power, which must be a hostile rival) or ``insurgency`` (backing an
insurgency against it): agents can start crises through their choices.

Every active power decides in the first quarter of each year. The decisions
form one simultaneous group: commits are held by the engine and applied only
in the resolve phase, so no power observes another's choice for the year.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..world import calendar
from ..world.state import POSTURES, POWERS, REGIONS, WorldState
from .requests import DecisionRequest, Option

STAGES = ("posture", "region", "mode", "diplomacy")  # the order of a power's turn


def historical_row(
    table: Mapping[str, tuple[Mapping[str, Any], ...]], power: str, year: int
) -> Mapping[str, Any] | None:
    """The row of the historical strategy table that covers *year* for *power*."""

    for row in table.get(power, ()):
        if int(row["from"]) <= year <= int(row["to"]):
            return row
    return None


def rival_leader(world: WorldState, power: str, region: str) -> str | None:
    """The region's leading active power if it is a hostile rival of *power*."""

    influence = world.regions[region].influence
    active = [p for p in POWERS if world.powers[p].status == "active"]
    if not active:
        return None
    leader = max(active, key=lambda p: (influence[p], p))
    if leader == power or world.powers[power].relations.get(leader, 0.0) >= 0.0:
        return None
    return leader


class _Turn:
    """One power's strategy turn: the world, the year's historical row and the commits so far."""

    def __init__(
        self,
        world: WorldState,
        power: str,
        row: Mapping[str, Any] | None,
        commits: Mapping[str, str],
        crises: bool,
    ) -> None:
        self.world = world
        self.power = power
        self.row = row
        self.commits = commits
        self.crises = crises

    def historical(self, key: str) -> Any:
        """The historical row's entry for *key* (``None`` without a row)."""

        return self.row.get(key) if self.row else None


def _posture(turn: _Turn) -> DecisionRequest:
    """The budget posture, one of ``POSTURES``."""

    hist = turn.historical("posture")
    options = tuple(Option(p, p.title(), p == hist) for p in POSTURES)
    return DecisionRequest(turn.power, "strategy_posture", "strategy", options)


def _region(turn: _Turn) -> DecisionRequest:
    """The region of focus, or none."""

    focus = turn.historical("focus")
    hist_region = focus[0] if focus else "none"
    options = (
        Option("none", "No focus", hist_region == "none"),
        *(Option(r, r.replace("_", " ").title(), r == hist_region) for r in REGIONS),
    )
    return DecisionRequest(turn.power, "strategy_region", "strategy", options)


def _mode(turn: _Turn) -> DecisionRequest | None:
    """The focus mode in the chosen region: aid or covert action, and with the generator a
    challenge or an insurgency against a hostile leading power. Skipped without a region.
    """

    region = turn.commits.get("region", "none")
    if region == "none":
        return None
    focus = turn.historical("focus")
    hist_mode = focus[1] if focus and focus[0] == region else "aid"
    options: tuple[Option, ...] = (
        Option("aid", "Aid", hist_mode == "aid"),
        Option("covert", "Covert action", hist_mode == "covert"),
    )
    if turn.crises and rival_leader(turn.world, turn.power, region) is not None:
        options += (
            Option("challenge", "Challenge the leading power", False),
            Option("insurgency", "Back an insurgency", False),
        )
    return DecisionRequest(turn.power, "strategy_mode", "strategy", options)


def _diplomacy(turn: _Turn) -> DecisionRequest:
    """The diplomatic initiative: engage or pressure one other active power, or none."""

    diplomacy = turn.historical("diplomacy")
    hist_key = f"{diplomacy[0]}:{diplomacy[1]}" if diplomacy else "none"
    options = [Option("none", "No initiative", hist_key == "none")]
    for other in POWERS:
        if other == turn.power or turn.world.powers[other].status != "active":
            continue
        for action in ("engage", "pressure"):
            key = f"{action}:{other}"
            options.append(Option(key, f"{action.title()} {other}", key == hist_key))
    return DecisionRequest(turn.power, "strategy_diplomacy", "strategy", tuple(options))


REQUESTS = {"posture": _posture, "region": _region, "mode": _mode, "diplomacy": _diplomacy}


def request_for(
    world: WorldState,
    power: str,
    stage: str,
    commits: Mapping[str, str],
    table: Mapping[str, tuple[Mapping[str, Any], ...]],
    crises: bool = False,
) -> DecisionRequest | None:
    """The request for *stage*, or ``None`` when the stage is skipped."""

    row = historical_row(table, power, calendar.year(world.quarter))
    return REQUESTS[stage](_Turn(world, power, row, commits, crises))


def apply_commits(world: WorldState, power: str, commits: Mapping[str, str]) -> None:
    """Set *power*'s posture, focus and diplomacy for the year from its *commits*, and
    record the focus and the stance in its eight-entry histories.
    """

    state = world.powers[power]
    year = calendar.year(world.quarter)
    state.posture = commits.get("posture", state.posture)
    region = commits.get("region", "none")
    if region == "none":
        state.focus_region, state.focus_mode = None, None
    else:
        state.focus_region, state.focus_mode = region, commits.get("mode", "aid")
        state.focus_history.append((year, region, state.focus_mode))
        del state.focus_history[:-8]
    diplomacy = commits.get("diplomacy", "none")
    if diplomacy == "none":
        state.diplomacy = None
    else:
        action, _, target = diplomacy.partition(":")
        state.diplomacy = diplomacy
        state.diplomacy_history.append((year, action, target))
        del state.diplomacy_history[:-8]
