"""World-state effects compiled from catalog YAML.

An effect list is a list of single-key mappings::

    flag: name | [names]            unflag: name | [names]
    add: {path, value, scale_by?}   set: {path, value}
    multiply: {path, factor}        (within the path's bounds)
    shift_influence: {region, toward, amount, scale_by?}
    set_controller: {region, to}    relations: {a, b, add, scale_by?}
    tension: delta                  leader: leader_id
    status: {power, to}             regime: {power, to}
    colonial_owner: {region, to}    (to: a power, or null for independence)
    start_event: event_id
    end_campaign: reason
    lottery: {chance, win: [effects], lose: [effects]}

``scale_by`` multiplies a numeric amount by an outcome metric (for example a
negotiated ``share``); ``{metric: share, invert: true}`` uses ``1 - share``.

A ``lottery`` applies ``win`` with probability ``chance`` (a number, or a metric
as in ``scale_by``) and ``lose`` otherwise, for indivisible results such as an
election. The draw is a counter-based uniform of the seed, the source event, the
quarter and the lottery's index, so it never depends on evaluation order.
Derived payoffs take the expectation over the branches (``EffectContext.forced``
fixes them). Lotteries do not nest.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .conditions import CompileContext, ConditionError
from .paths import compile_path
from .rng import counter_uniform
from .state import (
    NEUTRAL,
    POWERS,
    REGIMES,
    REGIONS,
    STATUSES,
    WorldState,
    normalize_influence,
    update_controller,
)

# `set_controller`: a region made neutral caps every power's influence at NEUTRAL_CAP; a
# region handed to a power lifts its influence to CONTROL_SHARE
NEUTRAL_CAP = 0.35
CONTROL_SHARE = 0.65


@dataclass(frozen=True, slots=True)
class LeaderSpec:
    id: str
    power: str
    name: str
    hawkishness: float
    risk_tolerance: float
    reformism: float


@dataclass
class EffectContext:
    """Runtime information available to effects."""

    leaders: Mapping[str, LeaderSpec]
    metrics: Mapping[str, float] = field(default_factory=dict)
    requested_events: list[str] = field(default_factory=list)
    seed: int = 0  # with source: the lottery draws
    source: str = ""
    forced: Mapping[int, bool] | None = None  # lottery index -> win, for expectations


Effect = Callable[[WorldState, EffectContext], None]


@dataclass(frozen=True)
class EffectList:
    effects: tuple[Effect, ...]
    flags_set: frozenset[str]
    lotteries: tuple[Callable[[EffectContext], float], ...] = ()  # chance of each, by index

    def apply(self, world: WorldState, ctx: EffectContext) -> None:
        for effect in self.effects:
            effect(world, ctx)
            if world.ended is not None and world.nuclear_war:
                return

    def __deepcopy__(self, memo: dict[int, object]) -> EffectList:
        return self


EMPTY_EFFECTS = EffectList((), frozenset())


def _scale(spec: Mapping[str, Any]) -> Callable[[EffectContext], float]:
    raw = spec.get("scale_by")
    if raw is None:
        return lambda ctx: 1.0
    if isinstance(raw, str):
        metric, invert = raw, False
    elif isinstance(raw, dict) and set(raw) <= {"metric", "invert"} and "metric" in raw:
        metric, invert = raw["metric"], bool(raw.get("invert", False))
    else:
        raise ConditionError(f"invalid scale_by {raw!r}")

    def factor(ctx: EffectContext) -> float:
        value = float(ctx.metrics.get(metric, 0.0))
        return 1.0 - value if invert else value

    return factor


def _keys(
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


def _names(value: Any, what: str) -> list[str]:
    items = value if isinstance(value, list) else [value]
    if not items or not all(isinstance(v, str) and v for v in items):
        raise ConditionError(f"{what} must be a name or list of names")
    return list(items)


def _chance(raw: Any) -> Callable[[EffectContext], float]:
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        if not 0.0 <= float(raw) <= 1.0:
            raise ConditionError(f"lottery chance {raw!r} is not a probability")
        fixed = float(raw)
        return lambda ectx: fixed
    factor = _scale({"scale_by": raw})
    return lambda ectx: min(1.0, max(0.0, factor(ectx)))


def _compile_one(
    spec: Any,
    ctx: CompileContext,
    flags: set[str],
    lotteries: list[Callable[[EffectContext], float]] | None = None,
) -> Effect:
    if not isinstance(spec, dict) or len(spec) != 1:
        raise ConditionError(f"an effect must be a single-key mapping, got {spec!r}")
    ((key, value),) = spec.items()

    if key in ("flag", "unflag"):
        names = _names(value, key)
        if key == "unflag":
            ctx.flags_cleared.update(names)
        if key == "flag":
            flags.update(names)

            def set_flags(world: WorldState, ectx: EffectContext) -> None:
                world.flags.update(names)

            return set_flags

        def clear_flags(world: WorldState, ectx: EffectContext) -> None:
            world.flags.difference_update(names)

        return clear_flags
    if key == "add":
        s = _keys(key, value, {"path", "value"}, {"scale_by"})
        ref, amount, factor = compile_path(s["path"]), float(s["value"]), _scale(s)

        def add(world: WorldState, ectx: EffectContext) -> None:
            ref.add(world, amount * factor(ectx))

        return add
    if key == "multiply":  # scale a quantity, within its bounds (the contract's war costs)
        s = _keys(key, value, {"path", "factor"})
        ref, by = compile_path(s["path"]), float(s["factor"])
        if by < 0.0:
            raise ConditionError(f"multiply: factor must be non-negative in {s}")

        def multiply(world: WorldState, ectx: EffectContext) -> None:
            ref.set(world, ref.get(world) * by)

        return multiply
    if key == "set":
        s = _keys(key, value, {"path", "value"})
        ref, amount = compile_path(s["path"]), float(s["value"])

        def assign(world: WorldState, ectx: EffectContext) -> None:
            ref.set(world, amount)

        return assign
    if key == "tension":
        delta = float(value)
        ref = compile_path("tension")

        def tension(world: WorldState, ectx: EffectContext) -> None:
            ref.add(world, delta)

        return tension
    if key == "shift_influence":
        s = _keys(key, value, {"region", "toward", "amount"}, {"scale_by"})
        region, toward = s["region"], s["toward"]
        if region not in REGIONS or toward not in POWERS:
            raise ConditionError(f"shift_influence: bad region or power in {s}")
        amount, factor = float(s["amount"]), _scale(s)

        def shift(world: WorldState, ectx: EffectContext) -> None:
            shift_influence(world, region, toward, amount * factor(ectx))

        return shift
    if key == "set_controller":
        s = _keys(key, value, {"region", "to"})
        region, target = s["region"], s["to"]
        if region not in REGIONS or target not in (*POWERS, NEUTRAL):
            raise ConditionError(f"set_controller: bad region or controller in {s}")

        def control(world: WorldState, ectx: EffectContext) -> None:
            state = world.regions[region]
            if target == NEUTRAL:
                for p in POWERS:
                    state.influence[p] = min(state.influence[p], NEUTRAL_CAP)
            else:
                shift_influence(
                    world, region, target, max(0.0, CONTROL_SHARE - state.influence[target])
                )
            update_controller(state)

        return control
    if key == "relations":
        s = _keys(key, value, {"a", "b", "add"}, {"scale_by"})
        a, b = s["a"], s["b"]
        if a not in POWERS or b not in POWERS or a == b:
            raise ConditionError(f"relations: bad powers in {s}")
        ref, amount, factor = compile_path(f"powers.{a}.relations.{b}"), float(s["add"]), _scale(s)

        def relate(world: WorldState, ectx: EffectContext) -> None:
            ref.add(world, amount * factor(ectx))

        return relate
    if key == "leader":
        leader_id = str(value)
        ctx.leaders.add(leader_id)

        def lead(world: WorldState, ectx: EffectContext) -> None:
            apply_leader(world, ectx.leaders[leader_id])

        return lead
    if key == "status":
        s = _keys(key, value, {"power", "to"})
        power, status = s["power"], s["to"]
        if power not in POWERS or status not in STATUSES:
            raise ConditionError(f"status: bad power or status in {s}")

        def set_status(world: WorldState, ectx: EffectContext) -> None:
            world.powers[power].status = status

        return set_status
    if key == "regime":
        s = _keys(key, value, {"power", "to"})
        power, regime = s["power"], s["to"]
        if power not in POWERS or regime not in REGIMES:
            raise ConditionError(f"regime: bad power or regime in {s}")

        def set_regime(world: WorldState, ectx: EffectContext) -> None:
            world.powers[power].regime = regime

        return set_regime
    if key == "colonial_owner":
        s = _keys(key, value, {"region", "to"})
        region, owner = s["region"], s["to"]
        if region not in REGIONS or (owner is not None and owner not in POWERS):
            raise ConditionError(f"colonial_owner: bad region or owner in {s}")

        def set_owner(world: WorldState, ectx: EffectContext) -> None:
            world.regions[region].colonial_owner = owner

        return set_owner
    if key == "start_event":
        event = str(value)
        ctx.event_refs.add(event)

        def request(world: WorldState, ectx: EffectContext) -> None:
            ectx.requested_events.append(event)

        return request
    if key == "lottery":
        s = _keys(key, value, {"chance", "win", "lose"})
        if lotteries is None:
            raise ConditionError("lotteries do not nest")
        chance = _chance(s["chance"])
        win, lose = compile_effects(s["win"], ctx), compile_effects(s["lose"], ctx)
        if win.lotteries or lose.lotteries:
            raise ConditionError("lotteries do not nest")
        flags.update(win.flags_set | lose.flags_set)
        index = len(lotteries)
        lotteries.append(chance)

        def draw(world: WorldState, ectx: EffectContext) -> None:
            if ectx.forced is not None and index in ectx.forced:
                won = ectx.forced[index]
            else:
                u = counter_uniform(ectx.seed, "lottery", ectx.source, world.quarter, index)
                won = u < chance(ectx)
            (win if won else lose).apply(world, ectx)

        return draw
    if key == "end_campaign":
        reason = str(value)

        def end(world: WorldState, ectx: EffectContext) -> None:
            if world.ended is None:
                world.ended = reason
            if reason == "nuclear_war":
                world.nuclear_war = True

        return end
    raise ConditionError(f"unknown effect {key!r}")


def compile_effects(spec: Any, ctx: CompileContext) -> EffectList:
    if spec is None:
        return EMPTY_EFFECTS
    if not isinstance(spec, list):
        raise ConditionError("effects must be a list")
    flags: set[str] = set()
    lotteries: list[Callable[[EffectContext], float]] = []
    effects = tuple(_compile_one(item, ctx, flags, lotteries) for item in spec)
    return EffectList(effects, frozenset(flags), tuple(lotteries))


def shift_influence(world: WorldState, region: str, toward: str, amount: float) -> None:
    """Move ``amount`` of influence in *region* to *toward*, taken from others.

    Negative amounts move influence away from *toward* into the neutral share.
    """

    state = world.regions[region]
    if amount >= 0:
        others = [p for p in POWERS if p != toward]
        free = max(0.0, 1.0 - sum(state.influence.values()))
        gain = min(amount, 1.0 - state.influence[toward])
        from_neutral = min(gain, free)
        remainder = gain - from_neutral
        other_total = sum(state.influence[p] for p in others)
        if remainder > 0 and other_total > 0:
            for p in others:
                state.influence[p] -= remainder * state.influence[p] / other_total
        state.influence[toward] += from_neutral + min(remainder, other_total)
    else:
        state.influence[toward] = max(0.0, state.influence[toward] + amount)
    normalize_influence(state)
    update_controller(state)


def apply_leader(world: WorldState, leader: LeaderSpec) -> None:
    power = world.powers[leader.power]
    power.leader = leader.id
    power.hawkishness = leader.hawkishness
    power.risk_tolerance = leader.risk_tolerance
    power.reformism = leader.reformism
