"""Yearly strategy decisions: posture, region focus, focus mode, diplomacy.

With the crisis generator on, the focus mode may also be ``challenge`` (a standoff with
the region's leading power, which must be a hostile rival) or ``insurgency`` (backing an
insurgency against it): agents start crises (decision 17; E49).

Every active power decides in the first quarter of each year. The decisions
form one simultaneous group: commits are held by the engine and applied only
in the resolve phase, so no power observes another's choice for the year.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from . import calendar
from .decisions import DecisionRequest, Option
from .state import POSTURES, POWERS, REGIONS, WorldState

STAGES = ("posture", "region", "mode", "diplomacy")


def historical_row(
    table: Mapping[str, tuple[Mapping[str, Any], ...]], power: str, year: int
) -> Mapping[str, Any] | None:
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
    focus = row.get("focus") if row else None
    diplomacy = row.get("diplomacy") if row else None
    if stage == "posture":
        hist = row.get("posture") if row else None
        options = tuple(Option(p, p.title(), p == hist) for p in POSTURES)
        return DecisionRequest(power, "strategy_posture", "strategy", options)
    if stage == "region":
        hist_region = focus[0] if focus else "none"
        options = (
            Option("none", "No focus", hist_region == "none"),
            *(Option(r, r.replace("_", " ").title(), r == hist_region) for r in REGIONS),
        )
        return DecisionRequest(power, "strategy_region", "strategy", options)
    if stage == "mode":
        if commits.get("region", "none") == "none":
            return None
        hist_mode = focus[1] if focus and focus[0] == commits["region"] else "aid"
        options = (
            Option("aid", "Aid", hist_mode == "aid"),
            Option("covert", "Covert action", hist_mode == "covert"),
        )
        if crises and rival_leader(world, power, commits["region"]) is not None:
            options += (
                Option("challenge", "Challenge the leading power", False),
                Option("insurgency", "Back an insurgency", False),
            )
        return DecisionRequest(power, "strategy_mode", "strategy", options)
    hist_key = f"{diplomacy[0]}:{diplomacy[1]}" if diplomacy else "none"
    options_list = [Option("none", "No initiative", hist_key == "none")]
    for other in POWERS:
        if other == power or world.powers[other].status != "active":
            continue
        for action in ("engage", "pressure"):
            key = f"{action}:{other}"
            options_list.append(Option(key, f"{action.title()} {other}", key == hist_key))
    return DecisionRequest(power, "strategy_diplomacy", "strategy", tuple(options_list))


def apply_commits(world: WorldState, power: str, commits: Mapping[str, str]) -> None:
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
