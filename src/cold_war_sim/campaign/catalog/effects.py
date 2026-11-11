"""World-state effects compiled from catalog YAML.

An effect list is a list of single-key mappings; each effect's compiler below documents
its form. ``scale_by`` multiplies a numeric amount by an outcome metric (for example a
negotiated ``share``); ``{metric: share, invert: true}`` uses ``1 - share``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ...records import Record
from ..engine.reproducibility import counter_uniform
from ..world.state import (
    NEUTRAL,
    POWERS,
    REGIMES,
    REGIONS,
    STATUSES,
    WorldState,
    normalize_influence,
    update_controller,
)
from .conditions import CompileContext, ConditionError, require_keys
from .paths import compile_path

# `set_controller`: a region made neutral caps every power's influence at NEUTRAL_CAP; a
# region handed to a power lifts its influence to CONTROL_SHARE
NEUTRAL_CAP = 0.35
CONTROL_SHARE = 0.65


class LeaderSpec(Record):
    """A leader and the traits it gives its power."""

    def __init__(
        self,
        id: str,
        power: str,
        name: str,
        hawkishness: float,
        risk_tolerance: float,
        reformism: float,
    ) -> None:
        self.id = id
        self.power = power
        self.name = name
        self.hawkishness = hawkishness
        self.risk_tolerance = risk_tolerance
        self.reformism = reformism


class EffectContext:
    """What effects may read and write besides the world when they are applied."""

    def __init__(
        self,
        leaders: Mapping[str, LeaderSpec],
        metrics: Mapping[str, float] | None = None,
        requested_events: list[str] | None = None,
        seed: int = 0,
        source: str = "",
        forced: Mapping[int, bool] | None = None,
    ) -> None:
        self.leaders = leaders
        self.metrics: Mapping[str, float] = {} if metrics is None else metrics  # the outcome's
        self.requested_events: list[str] = [] if requested_events is None else requested_events
        self.seed = seed  # with source: the lottery draws
        self.source = source
        self.forced = forced  # lottery index -> win, for expectations


Effect = Callable[[WorldState, EffectContext], None]
Chance = Callable[[EffectContext], float]


class EffectList(Record):
    """Compiled effects, applied in order; the campaign stops applying after a nuclear war.

    ``flags_set`` are the flags the effects may set; ``lotteries`` give each lottery's
    chance of winning, by index.
    """

    def __init__(
        self,
        effects: tuple[Effect, ...],
        flags_set: frozenset[str],
        lotteries: tuple[Chance, ...] = (),
    ) -> None:
        self.effects = effects
        self.flags_set = flags_set
        self.lotteries = lotteries

    def apply(self, world: WorldState, ctx: EffectContext) -> None:
        for effect in self.effects:
            effect(world, ctx)
            if world.ended is not None and world.nuclear_war:
                return

    def __deepcopy__(self, memo: dict[int, object]) -> EffectList:
        return self


EMPTY_EFFECTS = EffectList((), frozenset())


class _Compiling:
    """One effect list being compiled: the catalog's references, the flags it sets and
    its lotteries' chances."""

    def __init__(self, ctx: CompileContext, flags: set[str], lotteries: list[Chance]) -> None:
        self.ctx = ctx
        self.flags = flags
        self.lotteries = lotteries


def _scale(spec: Mapping[str, Any]) -> Chance:
    """The factor of ``scale_by``: a metric, or one minus it; 1 if absent."""

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


def _names(value: Any, what: str) -> list[str]:
    items = value if isinstance(value, list) else [value]
    if not items or not all(isinstance(v, str) and v for v in items):
        raise ConditionError(f"{what} must be a name or list of names")
    return list(items)


def _chance(raw: Any) -> Chance:
    """A lottery's chance: a probability, or a metric as in ``scale_by`` (clamped to [0, 1])."""

    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        if not 0.0 <= float(raw) <= 1.0:
            raise ConditionError(f"lottery chance {raw!r} is not a probability")
        fixed = float(raw)
        return lambda ectx: fixed
    factor = _scale({"scale_by": raw})
    return lambda ectx: min(1.0, max(0.0, factor(ectx)))


# -------------------------------------------------------------------- the effects
def _flag(key: str, value: Any, c: _Compiling) -> Effect:
    """``flag: name | [names]`` sets flags; ``unflag: ...`` clears them."""

    flags = _names(value, key)
    if key == "unflag":
        c.ctx.flags_cleared.update(flags)

        def clear_flags(world: WorldState, ectx: EffectContext) -> None:
            world.flags.difference_update(flags)

        return clear_flags
    c.flags.update(flags)

    def set_flags(world: WorldState, ectx: EffectContext) -> None:
        world.flags.update(flags)

    return set_flags


def _add(key: str, value: Any, c: _Compiling) -> Effect:
    """``add: {path, value, scale_by?}``"""

    s = require_keys(key, value, {"path", "value"}, {"scale_by"})
    ref, amount, factor = compile_path(s["path"]), float(s["value"]), _scale(s)

    def add(world: WorldState, ectx: EffectContext) -> None:
        ref.add(world, amount * factor(ectx))

    return add


def _multiply(key: str, value: Any, c: _Compiling) -> Effect:
    """``multiply: {path, factor}``: scale a quantity, within its bounds."""

    s = require_keys(key, value, {"path", "factor"})
    ref, by = compile_path(s["path"]), float(s["factor"])
    if by < 0.0:
        raise ConditionError(f"multiply: factor must be non-negative in {s}")

    def multiply(world: WorldState, ectx: EffectContext) -> None:
        ref.set(world, ref.get(world) * by)

    return multiply


def _set(key: str, value: Any, c: _Compiling) -> Effect:
    """``set: {path, value}``"""

    s = require_keys(key, value, {"path", "value"})
    ref, amount = compile_path(s["path"]), float(s["value"])

    def assign(world: WorldState, ectx: EffectContext) -> None:
        ref.set(world, amount)

    return assign


def _tension(key: str, value: Any, c: _Compiling) -> Effect:
    """``tension: delta``"""

    delta = float(value)
    ref = compile_path("tension")

    def tension(world: WorldState, ectx: EffectContext) -> None:
        ref.add(world, delta)

    return tension


def _shift_influence(key: str, value: Any, c: _Compiling) -> Effect:
    """``shift_influence: {region, toward, amount, scale_by?}``"""

    s = require_keys(key, value, {"region", "toward", "amount"}, {"scale_by"})
    region, toward = s["region"], s["toward"]
    if region not in REGIONS or toward not in POWERS:
        raise ConditionError(f"shift_influence: bad region or power in {s}")
    amount, factor = float(s["amount"]), _scale(s)

    def shift(world: WorldState, ectx: EffectContext) -> None:
        shift_influence(world, region, toward, amount * factor(ectx))

    return shift


def _set_controller(key: str, value: Any, c: _Compiling) -> Effect:
    """``set_controller: {region, to}``: to a power, or to ``neutral``."""

    s = require_keys(key, value, {"region", "to"})
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


def _relations(key: str, value: Any, c: _Compiling) -> Effect:
    """``relations: {a, b, add, scale_by?}``"""

    s = require_keys(key, value, {"a", "b", "add"}, {"scale_by"})
    a, b = s["a"], s["b"]
    if a not in POWERS or b not in POWERS or a == b:
        raise ConditionError(f"relations: bad powers in {s}")
    ref, amount, factor = compile_path(f"powers.{a}.relations.{b}"), float(s["add"]), _scale(s)

    def relate(world: WorldState, ectx: EffectContext) -> None:
        ref.add(world, amount * factor(ectx))

    return relate


def _leader(key: str, value: Any, c: _Compiling) -> Effect:
    """``leader: leader_id``: the leader takes office and its traits apply."""

    leader_id = str(value)
    c.ctx.leaders.add(leader_id)

    def lead(world: WorldState, ectx: EffectContext) -> None:
        apply_leader(world, ectx.leaders[leader_id])

    return lead


def _status(key: str, value: Any, c: _Compiling) -> Effect:
    """``status: {power, to}``"""

    s = require_keys(key, value, {"power", "to"})
    power, status = s["power"], s["to"]
    if power not in POWERS or status not in STATUSES:
        raise ConditionError(f"status: bad power or status in {s}")

    def set_status(world: WorldState, ectx: EffectContext) -> None:
        world.powers[power].status = status

    return set_status


def _regime(key: str, value: Any, c: _Compiling) -> Effect:
    """``regime: {power, to}``"""

    s = require_keys(key, value, {"power", "to"})
    power, regime = s["power"], s["to"]
    if power not in POWERS or regime not in REGIMES:
        raise ConditionError(f"regime: bad power or regime in {s}")

    def set_regime(world: WorldState, ectx: EffectContext) -> None:
        world.powers[power].regime = regime

    return set_regime


def _colonial_owner(key: str, value: Any, c: _Compiling) -> Effect:
    """``colonial_owner: {region, to}``: to a power, or ``null`` for independence."""

    s = require_keys(key, value, {"region", "to"})
    region, owner = s["region"], s["to"]
    if region not in REGIONS or (owner is not None and owner not in POWERS):
        raise ConditionError(f"colonial_owner: bad region or owner in {s}")

    def set_owner(world: WorldState, ectx: EffectContext) -> None:
        world.regions[region].colonial_owner = owner

    return set_owner


def _start_event(key: str, value: Any, c: _Compiling) -> Effect:
    """``start_event: event_id``: the event starts next quarter."""

    event = str(value)
    c.ctx.event_refs.add(event)

    def request(world: WorldState, ectx: EffectContext) -> None:
        ectx.requested_events.append(event)

    return request


def _lottery(key: str, value: Any, c: _Compiling) -> Effect:
    """``lottery: {chance, win: [effects], lose: [effects]}``: an indivisible result.

    ``win`` applies with probability ``chance`` and ``lose`` otherwise. The draw is a
    counter-based uniform of the seed, the source event, the quarter and the lottery's
    index, so it never depends on evaluation order. ``EffectContext.forced`` fixes the
    branch, for taking expectations. Lotteries do not nest.
    """

    s = require_keys(key, value, {"chance", "win", "lose"})
    chance = _chance(s["chance"])
    win, lose = compile_effects(s["win"], c.ctx), compile_effects(s["lose"], c.ctx)
    if win.lotteries or lose.lotteries:
        raise ConditionError("lotteries do not nest")
    c.flags.update(win.flags_set | lose.flags_set)
    index = len(c.lotteries)
    c.lotteries.append(chance)

    def draw(world: WorldState, ectx: EffectContext) -> None:
        if ectx.forced is not None and index in ectx.forced:
            won = ectx.forced[index]
        else:
            u = counter_uniform(ectx.seed, "lottery", ectx.source, world.quarter, index)
            won = u < chance(ectx)
        (win if won else lose).apply(world, ectx)

    return draw


def _end_campaign(key: str, value: Any, c: _Compiling) -> Effect:
    """``end_campaign: reason``"""

    reason = str(value)

    def end(world: WorldState, ectx: EffectContext) -> None:
        if world.ended is None:
            world.ended = reason
        if reason == "nuclear_war":
            world.nuclear_war = True

    return end


EFFECTS: dict[str, Callable[[str, Any, _Compiling], Effect]] = {
    "flag": _flag,
    "unflag": _flag,
    "add": _add,
    "multiply": _multiply,
    "set": _set,
    "tension": _tension,
    "shift_influence": _shift_influence,
    "set_controller": _set_controller,
    "relations": _relations,
    "leader": _leader,
    "status": _status,
    "regime": _regime,
    "colonial_owner": _colonial_owner,
    "start_event": _start_event,
    "lottery": _lottery,
    "end_campaign": _end_campaign,
}


def _compile_one(spec: Any, compiling: _Compiling) -> Effect:
    if not isinstance(spec, dict) or len(spec) != 1:
        raise ConditionError(f"an effect must be a single-key mapping, got {spec!r}")
    ((key, value),) = spec.items()
    if key not in EFFECTS:
        raise ConditionError(f"unknown effect {key!r}")
    return EFFECTS[key](key, value, compiling)


def compile_effects(spec: Any, ctx: CompileContext) -> EffectList:
    """The effect list of *spec* (``None`` is the empty list)."""

    if spec is None:
        return EMPTY_EFFECTS
    if not isinstance(spec, list):
        raise ConditionError("effects must be a list")
    compiling = _Compiling(ctx, set(), [])
    effects = tuple(_compile_one(item, compiling) for item in spec)
    return EffectList(effects, frozenset(compiling.flags), tuple(compiling.lotteries))


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
    """Put *leader* in office: its power takes the leader's traits."""

    power = world.powers[leader.power]
    power.leader = leader.id
    power.hawkishness = leader.hawkishness
    power.risk_tolerance = leader.risk_tolerance
    power.reformism = leader.reformism
