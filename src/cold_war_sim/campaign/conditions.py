"""Trigger conditions compiled from catalog YAML.

A condition is a mapping with one operator key, or a list (meaning ``all``)::

    all / any: [conditions]          not: condition
    after / before: "1962-Q2"        (inclusive)
    quarter_of_year: 1 | [1, 3]      (the calendar quarter, for yearly decisions)
    flag: name | [names]             no_flag: name | [names]
    fired: event_id                  active: event_id
    active_for: {event, quarters}    outcome: {event, in: [labels or tags]}
    compare: {path, op, value} | {path, op, path_rhs, scale}
    status: {power, in: [...]}       controller: {region, in: [...]}
    regime: {power, in: [...]}
    leader: {power, in: [...]}       always: true
    focus: {power, region?, mode?, within_years}
    diplomacy: {power, action, target?, within_years}

Compilation validates every path, power, region and operator, and records the
event ids a condition refers to so the catalog can check them.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import calendar
from .paths import compile_path
from .state import (
    DIPLOMACY_ACTIONS,
    FOCUS_MODES,
    NEUTRAL,
    POWERS,
    REGIMES,
    REGIONS,
    STATUSES,
    WorldState,
)

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


@dataclass
class CompileContext:
    """Collects references made by compiled conditions and effects."""

    event_refs: set[str] = field(default_factory=set)
    flags_read: set[str] = field(default_factory=set)
    flags_cleared: set[str] = field(default_factory=set)  # flags some effect unsets
    leaders: set[str] = field(default_factory=set)


def _names(value: Any, what: str) -> list[str]:
    items = value if isinstance(value, list) else [value]
    if not items or not all(isinstance(v, str) and v for v in items):
        raise ConditionError(f"{what} must be a name or a list of names")
    return list(items)


def _power(value: Any) -> str:
    if value not in POWERS:
        raise ConditionError(f"unknown power {value!r}")
    return str(value)


def _require_keys(
    name: str, spec: Any, required: set[str], optional: set[str] | None = None
) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ConditionError(f"{name} expects a mapping")
    optional = optional or set()
    missing = required - set(spec)
    unknown = set(spec) - required - optional
    if missing or unknown:
        raise ConditionError(f"{name}: missing {sorted(missing)} / unknown {sorted(unknown)}")
    return spec


def compile_condition(spec: Any, ctx: CompileContext) -> Predicate:
    if spec is None or spec is True:
        return lambda world: True
    if isinstance(spec, list):
        parts = [compile_condition(item, ctx) for item in spec]
        return lambda world: all(p(world) for p in parts)
    if not isinstance(spec, dict) or len(spec) != 1:
        raise ConditionError(f"a condition must be a single-key mapping or a list, got {spec!r}")
    ((key, value),) = spec.items()

    if key == "always":
        result = bool(value)
        return lambda world: result
    if key in ("all", "any"):
        if not isinstance(value, list) or not value:
            raise ConditionError(f"{key} expects a non-empty list")
        parts = [compile_condition(item, ctx) for item in value]
        if key == "all":
            return lambda world: all(p(world) for p in parts)
        return lambda world: any(p(world) for p in parts)
    if key == "not":
        inner = compile_condition(value, ctx)
        return lambda world: not inner(world)
    if key == "after":
        q = calendar.parse(value)
        return lambda world: world.quarter >= q
    if key == "before":
        q = calendar.parse(value)
        return lambda world: world.quarter <= q
    if key == "quarter_of_year":
        wanted = value if isinstance(value, list) else [value]
        if not wanted or not all(isinstance(v, int) and 1 <= v <= 4 for v in wanted):
            raise ConditionError("quarter_of_year expects quarters 1 to 4")
        in_year = frozenset(wanted)
        return lambda world: calendar.quarter_of_year(world.quarter) in in_year
    if key in ("flag", "no_flag"):
        names = _names(value, key)
        ctx.flags_read.update(names)
        if key == "flag":
            return lambda world: all(n in world.flags for n in names)
        return lambda world: not any(n in world.flags for n in names)
    if key in ("fired", "active"):
        names = _names(value, key)
        ctx.event_refs.update(names)
        if key == "fired":
            return lambda world: all(n in world.fired for n in names)
        return lambda world: all(n in world.active for n in names)
    if key == "active_for":
        spec_map = _require_keys(key, value, {"event", "quarters"})
        event, quarters = spec_map["event"], int(spec_map["quarters"])
        ctx.event_refs.add(event)
        return lambda world: (
            event in world.active and world.quarter - world.active[event] >= quarters
        )
    if key == "outcome":
        spec_map = _require_keys(key, value, {"event"}, {"in"})
        event = spec_map["event"]
        ctx.event_refs.add(event)
        accepted = set(_names(spec_map["in"], "outcome.in")) if "in" in spec_map else None

        def outcome_check(world: WorldState) -> bool:
            if event not in world.outcomes:
                return False
            if accepted is None:
                return True
            return world.outcomes[event] in accepted or bool(
                accepted & set(world.outcome_tags.get(event, ()))
            )

        return outcome_check
    if key == "compare":
        spec_map = _require_keys(key, value, {"path", "op"}, {"value", "path_rhs", "scale"})
        lhs = compile_path(spec_map["path"])
        op_name = spec_map["op"]
        if op_name not in OPERATORS:
            raise ConditionError(f"compare.op must be one of {sorted(OPERATORS)}")
        op = OPERATORS[op_name]
        scale = float(spec_map.get("scale", 1.0))
        if ("value" in spec_map) == ("path_rhs" in spec_map):
            raise ConditionError("compare needs exactly one of value or path_rhs")
        if "value" in spec_map:
            rhs_value = float(spec_map["value"]) * scale
            return lambda world: op(lhs.get(world), rhs_value)
        rhs = compile_path(spec_map["path_rhs"])
        return lambda world: op(lhs.get(world), scale * rhs.get(world))
    if key == "status":
        spec_map = _require_keys(key, value, {"power", "in"})
        power = _power(spec_map["power"])
        statuses = set(_names(spec_map["in"], "status.in"))
        if not statuses <= set(STATUSES):
            raise ConditionError(f"status.in must be among {STATUSES}")
        return lambda world: world.powers[power].status in statuses
    if key == "regime":
        spec_map = _require_keys(key, value, {"power", "in"})
        power = _power(spec_map["power"])
        regimes = set(_names(spec_map["in"], "regime.in"))
        if not regimes <= set(REGIMES):
            raise ConditionError(f"regime.in must be among {REGIMES}")
        return lambda world: world.powers[power].regime in regimes
    if key == "controller":
        spec_map = _require_keys(key, value, {"region", "in"})
        region = spec_map["region"]
        if region not in REGIONS:
            raise ConditionError(f"unknown region {region!r}")
        allowed = set(_names(spec_map["in"], "controller.in"))
        if not allowed <= set(POWERS) | {NEUTRAL}:
            raise ConditionError("controller.in must name powers or 'neutral'")
        return lambda world: world.regions[region].controller in allowed
    if key == "leader":
        spec_map = _require_keys(key, value, {"power", "in"})
        power = _power(spec_map["power"])
        leaders = set(_names(spec_map["in"], "leader.in"))
        ctx.leaders.update(leaders)
        return lambda world: world.powers[power].leader in leaders
    if key == "focus":
        spec_map = _require_keys(key, value, {"power", "within_years"}, {"region", "mode"})
        power = _power(spec_map["power"])
        region_name = spec_map.get("region")
        mode = spec_map.get("mode")
        if region_name is not None and region_name not in REGIONS:
            raise ConditionError(f"unknown region {region_name!r}")
        if mode is not None and mode not in FOCUS_MODES:
            raise ConditionError(f"focus.mode must be one of {FOCUS_MODES}")
        years = int(spec_map["within_years"])

        def focus_check(world: WorldState) -> bool:
            current = calendar.year(world.quarter)
            return any(
                current - y <= years
                and (region_name is None or r == region_name)
                and (mode is None or m == mode)
                for y, r, m in world.powers[power].focus_history
            )

        return focus_check
    if key == "diplomacy":
        spec_map = _require_keys(key, value, {"power", "action", "within_years"}, {"target"})
        power = _power(spec_map["power"])
        action = spec_map["action"]
        if action not in DIPLOMACY_ACTIONS:
            raise ConditionError(f"diplomacy.action must be one of {DIPLOMACY_ACTIONS}")
        target = spec_map.get("target")
        if target is not None:
            _power(target)
        years = int(spec_map["within_years"])

        def diplomacy_check(world: WorldState) -> bool:
            current = calendar.year(world.quarter)
            return any(
                current - y <= years and a == action and (target is None or t == target)
                for y, a, t in world.powers[power].diplomacy_history
            )

        return diplomacy_check
    raise ConditionError(f"unknown condition operator {key!r}")
