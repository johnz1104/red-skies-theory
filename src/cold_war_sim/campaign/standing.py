"""The standing scoreboard: an Elo-style rating per identity (``docs/GOALS.md`` §8).

Standing never enters any reward or the trace; it measures how the powers are
doing. When a contest resolves, each power ``p`` that took part on side ``s``
gets::

    delta = K * c_p * (S_s - E_s)

* ``c_p``  commitment in [0, 1], reported by the family as ``commit:<role>``;
* ``S_s``  the result: 1 win, 1/2 compromise or stalemate, 0 loss (a bargaining
  agreement scores each side its share; an event may override with
  ``model.results``);
* ``E_s``  the expected result, the side's share of committed strength:
  military strength times commitment for powers, or, in attrition wars, local
  strength plus each sponsor's capacity times its average effort.

A space-race first (an event with ``race: {winner, loser}``) is a contest with
commitment 1 and an expected result of 1/2.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ..games import Outcome
from . import goals
from .state import POWERS, WorldState, identity

if TYPE_CHECKING:
    from .catalog import Catalog, EventSpec

HALF = 0.5


def delta(k: float, commitment: float, result: float, expected: float) -> float:
    return k * commitment * (result - expected)


def sides(family: str, config: Mapping[str, Any]) -> dict[str, list[str]]:
    """Power roles grouped by side (in alignment every patron is its own side)."""

    groups: dict[str, list[str]] = {}

    def add(side: str, role: str) -> None:
        if role in POWERS:
            groups.setdefault(side, []).append(role)

    if family == "crisis":
        for role, spec in config["roles"].items():
            if spec.get("side") in ("a", "b"):
                add(spec["side"], role)
    elif family == "attrition":
        for role, spec in config["sponsors"].items():
            add(spec["side"], role)
    elif family == "bargaining":
        add("a", config["roles"]["A"])
        add("b", config["roles"]["B"])
    elif family == "intervention":
        add("a", config["roles"]["patron"])
        add("b", config["roles"]["rival"])
    elif family == "alignment":
        for patron in config["patrons"]:
            add(patron, patron)
    return groups


def results(family: str, config: Mapping[str, Any], outcome: Outcome) -> dict[str, float] | None:
    """Each side's result, or None when the contest has no winner to score."""

    label = outcome.label
    if family == "crisis":
        if label == "nuclear_war":
            return None
        if label in ("a_prevails", "b_prevails"):
            winner = label[0]
            return {"a": float(winner == "a"), "b": float(winner == "b")}
        return {"a": HALF, "b": HALF}
    if family == "attrition":
        if label == "refused":  # the war never started
            return None
        if label in ("a_victory", "b_withdrew"):
            return {"a": 1.0, "b": 0.0}
        if label in ("b_victory", "a_withdrew"):
            return {"a": 0.0, "b": 1.0}
        if label == "partition" and "share" in outcome.metrics:
            share = float(outcome.metrics["share"])
            return {"a": share, "b": 1.0 - share}
        return {"a": HALF, "b": HALF}
    if family == "bargaining":
        if label == "agreement":
            share = float(outcome.metrics.get("share", HALF))
            return {"a": share, "b": 1.0 - share}
        return {"a": HALF, "b": HALF}
    if family == "intervention":
        if label == "clash":
            return None
        if label == "crushed":
            return {"a": 1.0, "b": 0.0}
        if label == "defected":
            return {"a": 0.0, "b": 1.0}
        return {"a": HALF, "b": HALF}
    if family == "alignment":
        choices = [
            choice
            for tag in outcome.tags
            if not tag.startswith("offer:")
            and ((choice := tag.split(":", 1)[1]).startswith("aligned_") or choice == "nonaligned")
        ]
        if not choices:
            return None
        scores = {}
        for patron in config["patrons"]:
            won = sum(1.0 for c in choices if c == f"aligned_{patron}")
            neutral = sum(HALF for c in choices if c == "nonaligned")
            scores[patron] = (won + neutral) / len(choices)
        return scores
    raise ValueError(f"unknown family {family!r}")


def expected(
    catalog: Catalog,
    family: str,
    config: Mapping[str, Any],
    outcome: Outcome,
    world: WorldState,
    groups: Mapping[str, list[str]],
) -> dict[str, float]:
    metrics = outcome.metrics
    strength: dict[str, float] = {}
    if family == "attrition":
        local = config.get("local_strength", {})
        for side in ("a", "b"):
            total = float(local.get(side, 1.0))
            for role, spec in config["sponsors"].items():
                if spec["side"] == side:
                    effort = float(metrics.get(f"effort:{role}", 0.0))
                    total += float(spec.get("capacity", 1.0)) * effort
            strength[side] = total
    else:
        for side, members in groups.items():
            strength[side] = sum(
                float(metrics.get(f"commit:{p}", 0.0))
                * goals.military_strength(world, p, catalog.goals)
                for p in members
            )
    total = sum(strength.values())
    if total <= 0.0:
        return dict.fromkeys(strength, 1.0 / max(1, len(strength)))
    return {side: value / total for side, value in strength.items()}


def contest(
    catalog: Catalog,
    event: EventSpec,
    config: Mapping[str, Any],
    outcome: Outcome,
    world: WorldState,
) -> dict[str, float]:
    """Standing changes by identity for a resolved major event."""

    major = event.major
    assert major is not None
    groups = sides(major.family, config)
    scored = results(major.family, config, outcome)
    if scored is None or len(groups) < 2:
        return {}
    override = major.results.get(outcome.label, {})
    expect = expected(catalog, major.family, config, outcome, world, groups)
    k = catalog.goals.standing_k
    changes: dict[str, float] = {}
    for side, members in groups.items():
        for power in members:
            if world.powers[power].status == "eliminated":
                continue
            result = override.get(power, scored.get(side, HALF))
            commitment = float(outcome.metrics.get(f"commit:{power}", 0.0))
            change = delta(k, commitment, result, expect.get(side, HALF))
            if change:
                ident = identity(world, power)
                changes[ident] = changes.get(ident, 0.0) + change
    return changes


def race(catalog: Catalog, event: EventSpec, world: WorldState) -> dict[str, float]:
    """Standing changes for a space-race first: commitment 1, expected result 1/2."""

    assert event.race is not None
    winner, loser = event.race
    k = catalog.goals.standing_k
    return {
        identity(world, winner): delta(k, 1.0, 1.0, HALF),
        identity(world, loser): delta(k, 1.0, 0.0, HALF),
    }


__all__ = ["contest", "delta", "expected", "race", "results", "sides"]
