"""General-sum rewards from world state.

Each quarter a power receives the change in its *score*, a weighted sum of
security, influence, economic share, prestige, stability and credibility
(weights per power in ``powers.yaml``). Ending the campaign adds per-power
end bonuses (for example the USSR's dissolution), and nuclear war imposes a
large penalty on every power. Sub-game payoffs never enter the reward: they
are cardinal only within one event.
"""

from __future__ import annotations

from collections.abc import Mapping

from .catalog import Catalog
from .dynamics import military_strength
from .state import POWERS, REGIONS, WorldState

COMPONENTS = ("security", "influence", "economy", "prestige", "stability", "credibility")
DEFAULT_WEIGHTS = dict.fromkeys(COMPONENTS, 1.0)


def components(world: WorldState, power: str) -> dict[str, float]:
    state = world.powers[power]
    own = military_strength(world, power)
    rivals = [
        military_strength(world, p)
        for p in POWERS
        if p != power and world.powers[p].status == "active"
    ]
    strongest = max(rivals, default=0.0)
    security = own / (own + strongest) if own + strongest > 0 else 0.5
    total_importance = sum(world.regions[r].importance for r in REGIONS)
    influence = sum(
        world.regions[r].importance * world.regions[r].influence[power] for r in REGIONS
    )
    influence /= max(total_importance, 1e-9)
    economies = sum(
        world.powers[p].economy for p in POWERS if world.powers[p].status in ("active", "insurgent")
    )
    economy = state.economy / economies if economies > 0 else 0.0
    return {
        "security": security,
        "influence": influence,
        "economy": economy,
        "prestige": state.prestige,
        "stability": state.stability,
        "credibility": state.credibility,
    }


def score(world: WorldState, power: str, weights: Mapping[str, float]) -> float:
    if world.powers[power].status == "eliminated":
        return 0.0
    parts = components(world, power)
    return sum(weights.get(name, 1.0) * parts[name] for name in COMPONENTS)


def scores(world: WorldState, catalog: Catalog) -> dict[str, float]:
    return {p: score(world, p, catalog.powers[p].score_weights or DEFAULT_WEIGHTS) for p in POWERS}


def terminal_bonus(
    world: WorldState, catalog: Catalog, end_bonuses: bool = True
) -> dict[str, float]:
    """Terminal terms: the nuclear-war penalty, and the end bonuses of ``powers.yaml``
    when *end_bonuses* (the score reward only; under the goal reward the outcome's
    effects carry its meaning, E33)."""

    bonus = dict.fromkeys(POWERS, 0.0)
    if world.ended is None:
        return bonus
    penalty = float(catalog.rewards.get("nuclear_war_penalty", 20.0))
    for p in POWERS:
        if end_bonuses:
            bonus[p] += catalog.powers[p].end_bonus.get(world.ended, 0.0)
        if world.nuclear_war:
            bonus[p] -= penalty
    return bonus
