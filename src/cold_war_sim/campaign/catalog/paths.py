"""Validated dotted paths to numeric world-state values.

Accepted forms::

    tension | arms_control
    powers.<POWER>.<field>               e.g. powers.USSR.tech_nuclear
    powers.<POWER>.relations.<POWER>
    regions.<REGION>.<field>             e.g. regions.caribbean.stability
    regions.<REGION>.influence.<POWER>

A path is compiled once, when the catalog loads, into a :class:`PathRef`; an unknown path
is a load-time error.
"""

from __future__ import annotations

from collections.abc import Callable

from ...records import Record
from ..world.state import (
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


class PathRef(Record):
    """A compiled path: reads the value, and writes it clamped to [``low``, ``high``]."""

    def __init__(self, text: str, get: Getter, write: Setter, low: float, high: float) -> None:
        self.text = text
        self.get = get
        self._write = write
        self.low = low
        self.high = high

    def set(self, world: WorldState, value: float) -> None:
        self._write(world, min(self.high, max(self.low, float(value))))

    def add(self, world: WorldState, delta: float) -> None:
        self.set(world, self.get(world) + delta)


def compile_path(text: str) -> PathRef:
    """The reference that *text* names."""

    parts = str(text).split(".")
    if len(parts) == 1 and parts[0] in GLOBAL_FIELDS:
        return _global(text, parts[0])
    if len(parts) >= 3 and parts[0] in PATH_KINDS:
        return PATH_KINDS[parts[0]](text, parts[1], parts[2:])
    raise PathError(f"unknown world path {text!r}")


def _global(text: str, name: str) -> PathRef:
    """``tension`` or ``arms_control``."""

    def get(world: WorldState) -> float:
        return float(getattr(world, name))

    def write(world: WorldState, value: float) -> None:
        setattr(world, name, value)

    return PathRef(text, get, write, *GLOBAL_FIELDS[name])


def _power(text: str, power: str, rest: list[str]) -> PathRef:
    """``powers.<POWER>.<field>`` or ``powers.<POWER>.relations.<POWER>`` (symmetric)."""

    if power not in POWERS:
        raise PathError(f"{text}: unknown power {power!r}")
    if len(rest) == 1 and rest[0] in POWER_FIELDS:
        name = rest[0]

        def get(world: WorldState) -> float:
            return float(getattr(world.powers[power], name))

        def write(world: WorldState, value: float) -> None:
            setattr(world.powers[power], name, value)

        return PathRef(text, get, write, *POWER_FIELDS[name])
    if len(rest) == 2 and rest[0] == "relations":
        other = rest[1]
        if other not in POWERS or other == power:
            raise PathError(f"{text}: invalid relation target {other!r}")

        def get_relation(world: WorldState) -> float:
            return world.powers[power].relations[other]

        def write_relation(world: WorldState, value: float) -> None:
            world.powers[power].relations[other] = value
            world.powers[other].relations[power] = value

        return PathRef(text, get_relation, write_relation, *RELATION_BOUNDS)
    raise PathError(f"{text}: unknown power field")


def _region(text: str, region: str, rest: list[str]) -> PathRef:
    """``regions.<REGION>.<field>`` or ``regions.<REGION>.influence.<POWER>``.

    Writing an influence renormalizes the region's shares.
    """

    if region not in REGIONS:
        raise PathError(f"{text}: unknown region {region!r}")
    if len(rest) == 1 and rest[0] in REGION_FIELDS:
        name = rest[0]

        def get(world: WorldState) -> float:
            return float(getattr(world.regions[region], name))

        def write(world: WorldState, value: float) -> None:
            setattr(world.regions[region], name, value)

        return PathRef(text, get, write, *REGION_FIELDS[name])
    if len(rest) == 2 and rest[0] == "influence":
        power = rest[1]
        if power not in POWERS:
            raise PathError(f"{text}: unknown power {power!r}")

        def get_influence(world: WorldState) -> float:
            return world.regions[region].influence[power]

        def write_influence(world: WorldState, value: float) -> None:
            state = world.regions[region]
            state.influence[power] = value
            normalize_influence(state)

        return PathRef(text, get_influence, write_influence, *INFLUENCE_BOUNDS)
    raise PathError(f"{text}: unknown region field")


PATH_KINDS: dict[str, Callable[[str, str, list[str]], PathRef]] = {
    "powers": _power,
    "regions": _region,
}
