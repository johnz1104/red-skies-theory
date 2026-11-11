"""Event director: decides which events fire each quarter.

Candidates are scanned in (priority desc, id) order. Cutscenes apply their effects at
once; because those effects, or a newly started major event, can make other conditions
true, the scan repeats up to ``MAX_PASSES`` times. Major events are returned for
instantiation after the cascade, and decision events for the engine to put to their
power.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...records import Record
from ..catalog.effects import EffectContext
from ..engine.reproducibility import counter_uniform
from ..world.state import WorldState

if TYPE_CHECKING:
    from ..catalog.core import Catalog, EventSpec

MAX_PASSES = 4


class DirectorResult(Record):
    """What fired in one quarter: the cutscenes applied, the major and decision events to
    start, and the events their effects requested for the next quarter.
    """

    def __init__(
        self,
        cutscenes: list[EventSpec],
        majors: list[EventSpec],
        requested: list[str],
        decisions: list[EventSpec],
    ) -> None:
        self.cutscenes = cutscenes
        self.majors = majors
        self.requested = requested
        self.decisions = decisions


def eligible(event: EventSpec, world: WorldState, seed: int, requested: set[str]) -> bool:
    """Whether *event* fires now.

    It must not have fired before (unless it repeats) nor be running, and no event it
    excludes may have fired. A requested event then fires regardless; otherwise it must be
    inside its window, its condition must hold, and its hazard draw must pass (a
    counter-based uniform of the seed, the event and the quarter).
    """

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
    """Fire this quarter's events on *world*; each fires at most once a quarter."""

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
