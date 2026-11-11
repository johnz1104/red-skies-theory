"""Trigger conditions compiled from catalog YAML into predicates on the world.

A condition is a mapping with one operator key, or a list (meaning ``all``). Each
operator's compiler below documents its form. Compilation validates every path, power,
region and operator, and records the events, flags and leaders a condition refers to.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Any

from ..world import calendar
from ..world.state import (
    DIPLOMACY_ACTIONS,
    FOCUS_MODES,
    NEUTRAL,
    POWERS,
    REGIMES,
    REGIONS,
    STATUSES,
    WorldState,
)
from .paths import compile_path

Predicate = Callable[[WorldState], bool]

OPERATORS: dict[str, Callable[[float, float], bool]] = {
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
    "eq": operator.eq,
    "ne": operator.ne,
}


class ConditionError(ValueError):
    """Raised for a malformed condition."""


class CompileContext:
    """The references that compiled conditions and effects make, for the catalog to check."""

    def __init__(self) -> None:
        self.event_refs: set[str] = set()
        self.flags_read: set[str] = set()
        self.flags_cleared: set[str] = set()  # flags some effect unsets
        self.leaders: set[str] = set()


def names(value: Any, what: str) -> list[str]:
    """A name or a list of names, as a list."""

    items = value if isinstance(value, list) else [value]
    if not items or not all(isinstance(v, str) and v for v in items):
        raise ConditionError(f"{what} must be a name or a list of names")
    return list(items)


def require_keys(
    name: str, spec: Any, required: set[str], optional: set[str] | None = None
) -> dict[str, Any]:
    """*spec*, which must be a mapping with the *required* keys and no others but the *optional*."""

    if not isinstance(spec, dict):
        raise ConditionError(f"{name} expects a mapping")
    optional = optional or set()
    missing = required - set(spec)
    unknown = set(spec) - required - optional
    if missing or unknown:
        raise ConditionError(f"{name}: missing {sorted(missing)} / unknown {sorted(unknown)}")
    return spec


def _power(value: Any) -> str:
    if value not in POWERS:
        raise ConditionError(f"unknown power {value!r}")
    return str(value)


def compile_condition(spec: Any, ctx: CompileContext) -> Predicate:
    """The predicate of *spec*: ``None`` or ``True`` always holds; a list is a conjunction."""

    if spec is None or spec is True:
        return lambda world: True
    if isinstance(spec, list):
        parts = [compile_condition(item, ctx) for item in spec]
        return lambda world: all(p(world) for p in parts)
    if not isinstance(spec, dict) or len(spec) != 1:
        raise ConditionError(f"a condition must be a single-key mapping or a list, got {spec!r}")
    ((key, value),) = spec.items()
    if key not in CONDITIONS:
        raise ConditionError(f"unknown condition operator {key!r}")
    return CONDITIONS[key](key, value, ctx)


# ------------------------------------------------------------------ the operators
def _always(key: str, value: Any, ctx: CompileContext) -> Predicate:
    """``always: true | false``"""

    result = bool(value)
    return lambda world: result


def _all_any(key: str, value: Any, ctx: CompileContext) -> Predicate:
    """``all: [conditions]`` and ``any: [conditions]``"""

    if not isinstance(value, list) or not value:
        raise ConditionError(f"{key} expects a non-empty list")
    parts = [compile_condition(item, ctx) for item in value]
    if key == "all":
        return lambda world: all(p(world) for p in parts)
    return lambda world: any(p(world) for p in parts)


def _not(key: str, value: Any, ctx: CompileContext) -> Predicate:
    """``not: condition``"""

    inner = compile_condition(value, ctx)
    return lambda world: not inner(world)


def _after(key: str, value: Any, ctx: CompileContext) -> Predicate:
    """``after: "1962-Q2"`` (inclusive)"""

    q = calendar.parse(value)
    return lambda world: world.quarter >= q


def _before(key: str, value: Any, ctx: CompileContext) -> Predicate:
    """``before: "1962-Q2"`` (inclusive)"""

    q = calendar.parse(value)
    return lambda world: world.quarter <= q


def _quarter_of_year(key: str, value: Any, ctx: CompileContext) -> Predicate:
    """``quarter_of_year: 1 | [1, 3]``: the calendar quarter, for yearly decisions"""

    wanted = value if isinstance(value, list) else [value]
    if not wanted or not all(isinstance(v, int) and 1 <= v <= 4 for v in wanted):
        raise ConditionError("quarter_of_year expects quarters 1 to 4")
    in_year = frozenset(wanted)
    return lambda world: calendar.quarter_of_year(world.quarter) in in_year


def _flags(key: str, value: Any, ctx: CompileContext) -> Predicate:
    """``flag: name | [names]`` (all set) and ``no_flag: name | [names]`` (none set)"""

    flags = names(value, key)
    ctx.flags_read.update(flags)
    if key == "flag":
        return lambda world: all(n in world.flags for n in flags)
    return lambda world: not any(n in world.flags for n in flags)


def _events(key: str, value: Any, ctx: CompileContext) -> Predicate:
    """``fired: event | [events]`` (all have started) and ``active: ...`` (all are running)"""

    events = names(value, key)
    ctx.event_refs.update(events)
    if key == "fired":
        return lambda world: all(n in world.fired for n in events)
    return lambda world: all(n in world.active for n in events)


def _active_for(key: str, value: Any, ctx: CompileContext) -> Predicate:
    """``active_for: {event, quarters}``: running for at least that many quarters"""

    spec = require_keys(key, value, {"event", "quarters"})
    event, quarters = spec["event"], int(spec["quarters"])
    ctx.event_refs.add(event)
    return lambda world: event in world.active and world.quarter - world.active[event] >= quarters


def _outcome(key: str, value: Any, ctx: CompileContext) -> Predicate:
    """``outcome: {event, in: [labels or tags]}``: the event ended, in one of those if given"""

    spec = require_keys(key, value, {"event"}, {"in"})
    event = spec["event"]
    ctx.event_refs.add(event)
    accepted = set(names(spec["in"], "outcome.in")) if "in" in spec else None

    def outcome_check(world: WorldState) -> bool:
        if event not in world.outcomes:
            return False
        if accepted is None:
            return True
        return world.outcomes[event] in accepted or bool(
            accepted & set(world.outcome_tags.get(event, ()))
        )

    return outcome_check


def _compare(key: str, value: Any, ctx: CompileContext) -> Predicate:
    """``compare: {path, op, value}`` or ``{path, op, path_rhs, scale}``: path op scale x rhs"""

    spec = require_keys(key, value, {"path", "op"}, {"value", "path_rhs", "scale"})
    lhs = compile_path(spec["path"])
    op_name = spec["op"]
    if op_name not in OPERATORS:
        raise ConditionError(f"compare.op must be one of {sorted(OPERATORS)}")
    op = OPERATORS[op_name]
    scale = float(spec.get("scale", 1.0))
    if ("value" in spec) == ("path_rhs" in spec):
        raise ConditionError("compare needs exactly one of value or path_rhs")
    if "value" in spec:
        rhs_value = float(spec["value"]) * scale
        return lambda world: op(lhs.get(world), rhs_value)
    rhs = compile_path(spec["path_rhs"])
    return lambda world: op(lhs.get(world), scale * rhs.get(world))


def _status(key: str, value: Any, ctx: CompileContext) -> Predicate:
    """``status: {power, in: [...]}``"""

    spec = require_keys(key, value, {"power", "in"})
    power = _power(spec["power"])
    statuses = set(names(spec["in"], "status.in"))
    if not statuses <= set(STATUSES):
        raise ConditionError(f"status.in must be among {STATUSES}")
    return lambda world: world.powers[power].status in statuses


def _regime(key: str, value: Any, ctx: CompileContext) -> Predicate:
    """``regime: {power, in: [...]}``"""

    spec = require_keys(key, value, {"power", "in"})
    power = _power(spec["power"])
    regimes = set(names(spec["in"], "regime.in"))
    if not regimes <= set(REGIMES):
        raise ConditionError(f"regime.in must be among {REGIMES}")
    return lambda world: world.powers[power].regime in regimes


def _controller(key: str, value: Any, ctx: CompileContext) -> Predicate:
    """``controller: {region, in: [powers or neutral]}``"""

    spec = require_keys(key, value, {"region", "in"})
    region = spec["region"]
    if region not in REGIONS:
        raise ConditionError(f"unknown region {region!r}")
    allowed = set(names(spec["in"], "controller.in"))
    if not allowed <= set(POWERS) | {NEUTRAL}:
        raise ConditionError("controller.in must name powers or 'neutral'")
    return lambda world: world.regions[region].controller in allowed


def _leader(key: str, value: Any, ctx: CompileContext) -> Predicate:
    """``leader: {power, in: [leader ids]}``"""

    spec = require_keys(key, value, {"power", "in"})
    power = _power(spec["power"])
    leaders = set(names(spec["in"], "leader.in"))
    ctx.leaders.update(leaders)
    return lambda world: world.powers[power].leader in leaders


def _focus(key: str, value: Any, ctx: CompileContext) -> Predicate:
    """``focus: {power, region?, mode?, within_years}``: a matching focus in recent years"""

    spec = require_keys(key, value, {"power", "within_years"}, {"region", "mode"})
    power = _power(spec["power"])
    region_name = spec.get("region")
    mode = spec.get("mode")
    if region_name is not None and region_name not in REGIONS:
        raise ConditionError(f"unknown region {region_name!r}")
    if mode is not None and mode not in FOCUS_MODES:
        raise ConditionError(f"focus.mode must be one of {FOCUS_MODES}")
    years = int(spec["within_years"])

    def focus_check(world: WorldState) -> bool:
        current = calendar.year(world.quarter)
        return any(
            current - y <= years
            and (region_name is None or r == region_name)
            and (mode is None or m == mode)
            for y, r, m in world.powers[power].focus_history
        )

    return focus_check


def _diplomacy(key: str, value: Any, ctx: CompileContext) -> Predicate:
    """``diplomacy: {power, action, target?, within_years}``: a matching stance in recent years"""

    spec = require_keys(key, value, {"power", "action", "within_years"}, {"target"})
    power = _power(spec["power"])
    action = spec["action"]
    if action not in DIPLOMACY_ACTIONS:
        raise ConditionError(f"diplomacy.action must be one of {DIPLOMACY_ACTIONS}")
    target = spec.get("target")
    if target is not None:
        _power(target)
    years = int(spec["within_years"])

    def diplomacy_check(world: WorldState) -> bool:
        current = calendar.year(world.quarter)
        return any(
            current - y <= years and a == action and (target is None or t == target)
            for y, a, t in world.powers[power].diplomacy_history
        )

    return diplomacy_check


Compiler = Callable[[str, Any, CompileContext], Predicate]
CONDITIONS: dict[str, Compiler] = {
    "always": _always,
    "all": _all_any,
    "any": _all_any,
    "not": _not,
    "after": _after,
    "before": _before,
    "quarter_of_year": _quarter_of_year,
    "flag": _flags,
    "no_flag": _flags,
    "fired": _events,
    "active": _events,
    "active_for": _active_for,
    "outcome": _outcome,
    "compare": _compare,
    "status": _status,
    "regime": _regime,
    "controller": _controller,
    "leader": _leader,
    "focus": _focus,
    "diplomacy": _diplomacy,
}
