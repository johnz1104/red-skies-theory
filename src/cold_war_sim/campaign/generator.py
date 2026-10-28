"""The crisis generator: crises where the world's state makes them likely (``data/generator.yaml``).

GOALS.md section 12, decisions 16-21. Each quarter, for every region without a crisis of its
own, each template (``standoff``, ``proxy_war``, ``uprising``, ``alignment``, ``settlement``)
starts a crisis with hazard

    h = rate x group_factor x decade_factor x exp(sum_k slope_k (x_k - reference_k))

over the region's features (instability, how contested its influence is, tension, colonial
rule, a recent war). Its roles are cast from the world: the region's leading power (``$A``)
and its strongest hostile rival (``$B``). A generated crisis is a major event with the id
``gen:<template>:<region>:<A>-<B>``, compiled on first use by the catalog
(:meth:`.catalog.EventTable.__missing__`) from the template with derived payoffs and the
shared effects contract, so it carries no hand-set stakes.

Named crises take precedence: a region with a major event active, or one that started within
``precedence_quarters``, gets none. Draws are counter-based uniforms of the seed, template,
region and quarter, so they do not depend on the policies. The generator is off unless
``EngineConfig.generator`` is set, and never runs with historical chance (the replay has only
named crises).
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ..families import attrition, bargaining, crisis, intervention
from . import calendar
from .rng import counter_uniform
from .state import POWERS, REGIONS, WorldState

if TYPE_CHECKING:
    from .catalog import Catalog

PREFIX = "gen"
TEMPLATES = ("standoff", "proxy_war", "uprising", "alignment", "settlement")
FAMILY_OF = {
    "standoff": "crisis",
    "proxy_war": "attrition",
    "uprising": "intervention",
    "alignment": "alignment",
    "settlement": "bargaining",
}
TITLES = {
    "standoff": "A standoff",
    "proxy_war": "A proxy war",
    "uprising": "An uprising",
    "alignment": "A contest for alignment",
    "settlement": "Settlement talks",
}
MIN_INFLUENCE = 0.02  # a power below this share has no stake in the region
RECENT_WAR_QUARTERS = 8
WAR_LABELS = frozenset(
    {"limited_war", "partition", "stalemate", "a_victory", "b_victory", "a_withdrew", "b_withdrew"}
)


class GeneratorError(ValueError):
    pass


# ------------------------------------------------------------------ identities
def event_id(template: str, region: str, roles: tuple[str, str]) -> str:
    return f"{PREFIX}:{template}:{region}:{roles[0]}-{roles[1]}"


def is_generated(event: str) -> bool:
    return event.startswith(f"{PREFIX}:")


def parse_id(event: str) -> tuple[str, str, tuple[str, str]]:
    """``gen:<template>:<region>:<A>-<B>`` -> (template, region, (A, B))."""

    parts = event.split(":")
    if len(parts) != 4 or parts[0] != PREFIX:
        raise GeneratorError(f"not a generated event id: {event!r}")
    template, region, pair = parts[1], parts[2], parts[3].split("-")
    if template not in TEMPLATES or region not in REGIONS or len(pair) != 2:
        raise GeneratorError(f"not a generated event id: {event!r}")
    if not set(pair) <= set(POWERS) or pair[0] == pair[1]:
        raise GeneratorError(f"not a generated event id: {event!r}")
    return template, region, (pair[0], pair[1])


# ------------------------------------------------------------------ raw events
def _substitute(node: Any, names: Mapping[str, str]) -> Any:
    if isinstance(node, str):
        for key, value in names.items():
            node = node.replace(key, value)
        return node
    if isinstance(node, list):
        return [_substitute(item, names) for item in node]
    if isinstance(node, dict):
        return {_substitute(k, names): _substitute(v, names) for k, v in node.items()}
    return node


def labels_of(template: str, roles: tuple[str, str]) -> tuple[str, ...]:
    family = FAMILY_OF[template]
    if family == "alignment":
        return (f"aligned_{roles[0]}", f"aligned_{roles[1]}", "nonaligned")
    modules = {"crisis": crisis, "attrition": attrition, "intervention": intervention}
    return tuple(modules[family].OUTCOMES) if family in modules else tuple(bargaining.OUTCOMES)


def raw_event(doc: Mapping[str, Any], event: str) -> dict[str, Any]:
    """The raw major event a generated id stands for, in the catalog's language."""

    template, region, (a, b) = parse_id(event)
    labels = labels_of(template, (a, b))
    spec = doc["templates"][template]
    names = {"$A": a, "$B": b, "$R": region}
    family = FAMILY_OF[template]
    model: dict[str, Any] = {
        "family": family,
        "tier": 1,
        "payoffs": "derived",
        "effects": "contract",
        "region": region,
        "config": _substitute(copy.deepcopy(spec["config"]), names),
    }
    if "cadence" in spec:
        model["cadence"] = spec["cadence"]
    if family == "attrition":
        model["effort_costs"] = "derived"
    if spec.get("bindings"):
        model["bindings"] = _substitute(copy.deepcopy(spec["bindings"]), names)
    additions = _substitute(copy.deepcopy(spec.get("additions", {})), names)
    region_name = region.replace("_", " ")
    return {
        "id": event,
        "kind": "major",
        "once": False,
        "title": f"{TITLES[template]} in {region_name} ({a} and {b})",
        "narrative": f"{TITLES[template]} breaks out in {region_name}.",
        "model": model,
        "outcomes": {
            label: {
                "narrative": f"{label.replace('_', ' ')}",
                "additions": additions.get(label, []),
            }
            for label in labels
        },
    }


# ------------------------------------------------------------------ the world
def group_of(doc: Mapping[str, Any], region: str) -> str:
    for group, members in doc["groups"].items():
        if region in members:
            return str(group)
    raise GeneratorError(f"region {region!r} is in no group")


def decade_of(doc: Mapping[str, Any], quarter: int) -> int:
    year = calendar.year(quarter)
    decades = sorted(int(d) for d in doc["decades"])
    return max(d for d in decades if d <= max(year, decades[0]))


def cast(world: WorldState, region: str) -> tuple[str, str] | None:
    """The region's leading active power and its strongest hostile rival, or ``None``."""

    influence = world.regions[region].influence
    active = [p for p in POWERS if world.powers[p].status == "active"]
    ranked = sorted(active, key=lambda p: (-influence[p], p))
    if not ranked or influence[ranked[0]] < MIN_INFLUENCE:
        return None
    leader = ranked[0]
    hostile = [
        p
        for p in ranked[1:]
        if world.powers[leader].relations.get(p, 0.0) < 0.0 and influence[p] >= MIN_INFLUENCE
    ]
    if not hostile:
        return None
    return leader, hostile[0]


def _named_regions(catalog: Catalog) -> dict[str, str]:
    """Each named major event's region (the catalog's own events only)."""

    return {
        e.id: e.major.region
        for e in catalog.events.values()
        if e.major is not None and e.major.region is not None
    }


def _region_of(catalog: Catalog, named: Mapping[str, str], event: str) -> str | None:
    if is_generated(event):
        return parse_id(event)[1]
    return named.get(event)


def blocked(catalog: Catalog, world: WorldState, region: str, precedence: int) -> bool:
    """A major event is active in the region, or one started there recently."""

    named = _named_regions(catalog)
    for event in world.active:
        if _region_of(catalog, named, event) == region:
            return True
    for event, quarter in world.fired.items():
        if world.quarter - quarter < precedence and _region_of(catalog, named, event) == region:
            return True
    return False


def features(
    catalog: Catalog, world: WorldState, region: str, roles: tuple[str, str]
) -> dict[str, float]:
    state = world.regions[region]
    named = _named_regions(catalog)
    recent = any(
        world.quarter - quarter < RECENT_WAR_QUARTERS
        and _region_of(catalog, named, event) == region
        and world.outcomes.get(event) in WAR_LABELS
        for event, quarter in world.fired.items()
    )
    return {
        "instability": 1.0 - state.stability,
        "contest": state.influence[roles[1]],
        "tension": world.tension,
        "colonial": 1.0 if state.colonial_owner in POWERS else 0.0,
        "recent_war": 1.0 if recent else 0.0,
    }


def hazard(
    doc: Mapping[str, Any], template: str, region: str, quarter: int, x: Mapping[str, float]
) -> float:
    spec = doc["templates"][template]
    required = spec.get("requires")
    if required is not None and x[str(required)] <= 0.0:
        return 0.0
    rate = float(spec["rate"])
    rate *= float(spec["group_factor"][group_of(doc, region)])
    rate *= float(spec["decade_factor"][decade_of(doc, quarter)])
    exponent = 0.0
    for name, feature in doc["features"].items():
        exponent += float(feature.get(template, 0.0)) * (x[name] - float(feature["reference"]))
    return min(1.0, rate * math.exp(exponent))


def propose(catalog: Catalog, world: WorldState, seed: int) -> list[str]:
    """The generated crises that start this quarter: at most one per region."""

    doc = catalog.raw.generator if catalog.raw is not None else None
    if not doc:
        return []
    precedence = int(doc.get("precedence_quarters", 4))
    started: list[str] = []
    for region in REGIONS:
        if blocked(catalog, world, region, precedence):
            continue
        roles = cast(world, region)
        if roles is None:
            continue
        x = features(catalog, world, region, roles)
        for template in TEMPLATES:
            if template not in doc["templates"]:
                continue
            h = hazard(doc, template, region, world.quarter, x)
            if counter_uniform(seed, "generator", f"{template}:{region}", world.quarter) < h:
                started.append(event_id(template, region, roles))
                break
    return started


__all__ = [
    "FAMILY_OF",
    "PREFIX",
    "TEMPLATES",
    "GeneratorError",
    "blocked",
    "cast",
    "decade_of",
    "event_id",
    "features",
    "group_of",
    "hazard",
    "is_generated",
    "parse_id",
    "propose",
    "raw_event",
]
