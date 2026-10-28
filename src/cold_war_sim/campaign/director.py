"""Event director: decides which events fire each quarter.

Candidates are scanned in (priority desc, id) order. A candidate fires when
it has not fired before (unless ``once`` is false), no event it excludes has
fired, it is inside its window, its condition holds, and its hazard draw
passes. Cutscenes apply their effects immediately; because those effects (or
a newly started major event) can make other conditions true, the scan repeats
up to ``MAX_PASSES`` times.
Major events are returned for instantiation after the cascade, and decision
events for the engine to put to their power.

An event requested by a ``start_event`` effect fires at the next director
phase regardless of window, condition and hazard (``once`` and exclusions
still apply).
"""

from __future__ import annotations

from dataclasses import dataclass

from .catalog import Catalog, EventSpec
from .effects import EffectContext
from .rng import counter_uniform
from .state import WorldState

MAX_PASSES = 4


@dataclass
class DirectorResult:
    cutscenes: list[EventSpec]
    majors: list[EventSpec]
    requested: list[str]
    decisions: list[EventSpec]


def eligible(event: EventSpec, world: WorldState, seed: int, requested: set[str]) -> bool:
    if event.once and event.id in world.fired:
        return False
    if event.id in world.active:
        return False
    if any(other in world.fired for other in event.excludes):
        return False
    if event.id in requested:
        return True
    if not event.in_window(world.quarter):
        return False
    if not event.condition(world):
        return False
    if event.hazard >= 1.0:
        return True
    return counter_uniform(seed, "hazard", event.id, world.quarter) < event.hazard


def run_director(
    catalog: Catalog, world: WorldState, seed: int, requested: set[str]
) -> DirectorResult:
    result = DirectorResult([], [], [], [])
    ordered = catalog.ordered
    fired_now: set[str] = set()  # an event fires at most once a quarter, even if repeatable
    for _ in range(MAX_PASSES):
        changed = False
        for event in ordered:
            if event.id in fired_now or not eligible(event, world, seed, requested):
                continue
            fired_now.add(event.id)
            world.fired[event.id] = world.quarter
            changed = True  # "fired" conditions may now hold for other events
            if event.major is not None:
                result.majors.append(event)
                continue
            if event.decision is not None:
                result.decisions.append(event)
                continue
            ctx = EffectContext(leaders=catalog.leaders, seed=seed, source=event.id)
            event.effects.apply(world, ctx)
            result.requested.extend(ctx.requested_events)
            if event.narrative:
                world.log.append((world.quarter, event.id, event.narrative))
            result.cutscenes.append(event)
            if world.ended is not None:
                return result
        if not changed:
            break
    return result
