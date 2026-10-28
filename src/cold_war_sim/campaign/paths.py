"""Validated dotted paths to numeric world-state values.

Accepted forms::

    tension | arms_control
    powers.<POWER>.<field>               e.g. powers.USSR.tech_nuclear
    powers.<POWER>.relations.<POWER>
    regions.<REGION>.<field>             e.g. regions.caribbean.stability
    regions.<REGION>.influence.<POWER>

Paths are compiled once (when the catalog loads) into getter and setter
closures; an unknown path is a load-time error, never a runtime one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .state import (
    GLOBAL_FIELDS,
    INFLUENCE_BOUNDS,
    POWER_FIELDS,
    POWERS,
    REGION_FIELDS,
    REGIONS,
    RELATION_BOUNDS,
    WorldState,
    normalize_influence,
)

Getter = Callable[[WorldState], float]
Setter = Callable[[WorldState, float], None]


class PathError(ValueError):
    """Raised for a malformed or unknown world-state path."""


@dataclass(frozen=True, slots=True)
class PathRef:
    text: str
    get: Getter
    set: Setter
    low: float
    high: float

    def add(self, world: WorldState, delta: float) -> None:
        self.set(world, self.get(world) + delta)


def _clamped(low: float, high: float, raw: Setter) -> Setter:
    def setter(world: WorldState, value: float) -> None:
        raw(world, min(high, max(low, float(value))))

    return setter


def compile_path(text: str) -> PathRef:
    parts = str(text).split(".")
    if len(parts) == 1 and parts[0] in GLOBAL_FIELDS:
        name = parts[0]
        low, high = GLOBAL_FIELDS[name]

        def get_global(world: WorldState) -> float:
            return float(getattr(world, name))

        def set_global(world: WorldState, value: float) -> None:
            setattr(world, name, value)

        return PathRef(text, get_global, _clamped(low, high, set_global), low, high)

    if len(parts) >= 3 and parts[0] == "powers":
        power = parts[1]
        if power not in POWERS:
            raise PathError(f"{text}: unknown power {power!r}")
        if len(parts) == 3 and parts[2] in POWER_FIELDS:
            name = parts[2]
            low, high = POWER_FIELDS[name]

            def get_power(world: WorldState) -> float:
                return float(getattr(world.powers[power], name))

            def set_power(world: WorldState, value: float) -> None:
                setattr(world.powers[power], name, value)

            return PathRef(text, get_power, _clamped(low, high, set_power), low, high)
        if len(parts) == 4 and parts[2] == "relations":
            other = parts[3]
            if other not in POWERS or other == power:
                raise PathError(f"{text}: invalid relation target {other!r}")
            low, high = RELATION_BOUNDS

            def get_relation(world: WorldState) -> float:
                return world.powers[power].relations[other]

            def set_relation(world: WorldState, value: float) -> None:
                # Relations are symmetric.
                world.powers[power].relations[other] = value
                world.powers[other].relations[power] = value

            return PathRef(text, get_relation, _clamped(low, high, set_relation), low, high)
        raise PathError(f"{text}: unknown power field")

    if len(parts) >= 3 and parts[0] == "regions":
        region = parts[1]
        if region not in REGIONS:
            raise PathError(f"{text}: unknown region {region!r}")
        if len(parts) == 3 and parts[2] in REGION_FIELDS:
            name = parts[2]
            low, high = REGION_FIELDS[name]

            def get_region(world: WorldState) -> float:
                return float(getattr(world.regions[region], name))

            def set_region(world: WorldState, value: float) -> None:
                setattr(world.regions[region], name, value)

            return PathRef(text, get_region, _clamped(low, high, set_region), low, high)
        if len(parts) == 4 and parts[2] == "influence":
            power = parts[3]
            if power not in POWERS:
                raise PathError(f"{text}: unknown power {power!r}")
            low, high = INFLUENCE_BOUNDS

            def get_influence(world: WorldState) -> float:
                return world.regions[region].influence[power]

            def set_influence(world: WorldState, value: float) -> None:
                state = world.regions[region]
                state.influence[power] = value
                normalize_influence(state)

            return PathRef(text, get_influence, _clamped(low, high, set_influence), low, high)
        raise PathError(f"{text}: unknown region field")

    raise PathError(f"unknown world path {text!r}")
