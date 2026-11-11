"""An Elo-style scoreboard for each identity.

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

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any

from ...games import Outcome
from ..world.state import POWERS, WorldState, identity
from . import goals

if TYPE_CHECKING:
    from ..catalog.core import Catalog, EventSpec

HALF = 0.5


def delta(k: float, commitment: float, result: float, expected: float) -> float:
    """The Elo update K x commitment x (result - expected)."""

    return k * commitment * (result - expected)


def _group(pairs: Iterable[tuple[str, str]]) -> dict[str, list[str]]:
    """The power roles among (side, role) *pairs*, by side; a side with none is left out."""

    groups: dict[str, list[str]] = {}
    for side, role in pairs:
        if role in POWERS:
            groups.setdefault(side, []).append(role)
    return groups


def _crisis_sides(config: Mapping[str, Any]) -> dict[str, list[str]]:
    return _group(
        (spec["side"], role)
        for role, spec in config["roles"].items()
        if spec.get("side") in ("a", "b")
    )


def _attrition_sides(config: Mapping[str, Any]) -> dict[str, list[str]]:
    return _group((spec["side"], role) for role, spec in config["sponsors"].items())


def _bargaining_sides(config: Mapping[str, Any]) -> dict[str, list[str]]:
    return _group((("a", config["roles"]["A"]), ("b", config["roles"]["B"])))


def _intervention_sides(config: Mapping[str, Any]) -> dict[str, list[str]]:
    return _group((("a", config["roles"]["patron"]), ("b", config["roles"]["rival"])))


def _alignment_sides(config: Mapping[str, Any]) -> dict[str, list[str]]:
    return _group((patron, patron) for patron in config["patrons"])


SIDES: dict[str, Callable[[Mapping[str, Any]], dict[str, list[str]]]] = {
    "crisis": _crisis_sides,
    "attrition": _attrition_sides,
    "bargaining": _bargaining_sides,
    "intervention": _intervention_sides,
    "alignment": _alignment_sides,
}


def sides(family: str, config: Mapping[str, Any]) -> dict[str, list[str]]:
    """Power roles grouped by side (in alignment every patron is its own side)."""

    group = SIDES.get(family)
    return group(config) if group is not None else {}


WIN_A, WIN_B, DRAW = {"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}, {"a": HALF, "b": HALF}


def _split(share: float) -> dict[str, float]:
    """A settlement at *share* for side a scores each side its share."""

    return {"a": share, "b": 1.0 - share}


def _crisis_results(config: Mapping[str, Any], outcome: Outcome) -> dict[str, float] | None:
    """A side that prevails wins; a nuclear war has no winner to score; the rest draw."""

    if outcome.label == "nuclear_war":
        return None
    return dict({"a_prevails": WIN_A, "b_prevails": WIN_B}.get(outcome.label, DRAW))


def _attrition_results(config: Mapping[str, Any], outcome: Outcome) -> dict[str, float] | None:
    """Victory, or the other side's withdrawal, wins; a partition with a ``share`` scores
    the shares; a refused war never started."""

    label = outcome.label
    if label == "refused":
        return None
    if label in ("a_victory", "b_withdrew"):
        return dict(WIN_A)
    if label in ("b_victory", "a_withdrew"):
        return dict(WIN_B)
    if label == "partition" and "share" in outcome.metrics:
        return _split(float(outcome.metrics["share"]))
    return dict(DRAW)


def _bargaining_results(config: Mapping[str, Any], outcome: Outcome) -> dict[str, float] | None:
    """An agreement scores each side its share; the rest draw."""

    if outcome.label == "agreement":
        return _split(float(outcome.metrics.get("share", HALF)))
    return dict(DRAW)


def _intervention_results(config: Mapping[str, Any], outcome: Outcome) -> dict[str, float] | None:
    """A crushed upheaval is the patron's win, a defection the rival's; a clash is not scored."""

    if outcome.label == "clash":
        return None
    return dict({"crushed": WIN_A, "defected": WIN_B}.get(outcome.label, DRAW))


def _alignment_results(config: Mapping[str, Any], outcome: Outcome) -> dict[str, float] | None:
    """Each patron scores the share of clients that aligned with it, plus half of those
    that stayed non-aligned."""

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


RESULTS: dict[str, Callable[[Mapping[str, Any], Outcome], dict[str, float] | None]] = {
    "crisis": _crisis_results,
    "attrition": _attrition_results,
    "bargaining": _bargaining_results,
    "intervention": _intervention_results,
    "alignment": _alignment_results,
}


def results(family: str, config: Mapping[str, Any], outcome: Outcome) -> dict[str, float] | None:
    """Each side's result, or None when the contest has no winner to score."""

    if family not in RESULTS:
        raise ValueError(f"unknown family {family!r}")
    return RESULTS[family](config, outcome)


def expected(
    catalog: Catalog,
    family: str,
    config: Mapping[str, Any],
    outcome: Outcome,
    world: WorldState,
    groups: Mapping[str, list[str]],
) -> dict[str, float]:
    """Each side's expected result: its share of committed strength.

    In an attrition war a side's strength is its local strength plus each sponsor's
    capacity times its average effort; elsewhere it is military strength times commitment.
    """

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
