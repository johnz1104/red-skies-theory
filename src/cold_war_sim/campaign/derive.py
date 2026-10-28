"""Derived payoffs: power roles value outcomes by their goal scores (``docs/GOALS.md`` §7).

For an event starting in world ``W``, a power role's value for an outcome is::

    kappa * w . [c(W') - c(W)]  +  kappa * terminal terms

``c`` are the role's goal components, ``w`` its current weights, and ``W'`` a
copy of ``W`` with the outcome's effects applied plus its deterministic
consequences: cutscenes and leadership events the outcome newly enables that
fire for certain (hazard 1), cascading as the director does.

**Follow-ons** (delayed payoffs): every later cutscene adds
``(P(W') - P(W)) * kappa * w . [c(W' + its effects) - c(W')]``, where ``P`` is its
discounted chance of firing within the lookahead (``goals.yaml``), given its
window, hazard and whether its condition holds. An outcome that makes an event
more likely gains its value; one that blocks it loses it. The count is linear
(no interactions), conditions are read on the world just after the outcome, and
later major events are not counted (their outcomes are open). Terminal terms are
the nuclear-war penalty, the loss of the role's seat (elimination or a regime
change such as the Communists losing China: the goal score drops to zero, at
least ``loss_floor``) and end-of-campaign bonuses. Values are rounded to two
decimals so baseline caching stays useful.

**Action costs** (crisis): an option whose own tag (``<actor>:move=<id>``,
``:signal=``, ``:partner=``) has effects costs each power role minus the goal value of
those effects, whatever the outcome (the consequences of taking that action).

Only power roles are derived, and computer-run roles mapped to an identity (below);
other NPC roles keep hand-set values, as do costs of actions (escalation, signals,
effort, offers). Each family stores values in a
different place (:func:`slots`); bargaining uses the explicit ``agreement``,
``vetoed_value`` and consent values of the bargaining family, and attrition wars
with offers a ``settlement`` value by share, the line through the grid's end offers.

**Attrition sponsors' shares:** an outcome's ``side_share:<role>`` metric (a sponsor's share
of its side's effort, by which the shared effects contract splits influence) is valued at
equal shares within each side, the neutral expectation before the war is fought.

**Effort costs** (attrition events with ``effort_costs: derived``): a quarter of full
effort costs ``kappa * [w_economy * E + w_regime * S * persistence]``. ``E`` is the
permanent fall in the economy component from the dynamics' war drag (the war's
expenses), and ``S`` the stability lost in that quarter (casualties and the growth
slowdown). Stability recovers after the war, so its loss counts by its discounted
duration, ``persistence = (1 - d) / (1 - d + d * r)``, with ``d`` the lookahead
discount and ``r`` the quarterly reversion rate. The effort levels (1/3, 2/3, 1)
become the cost triple and this value each power sponsor's ``cost_scale``. Fighting with
its own troops (``direct``) costs a power more: the domestic cost of casualties,
``kappa * w_regime * r * casualty_drain * casualty_sensitivity * persistence``
(:func:`casualty_cost`), is added to its ``direct_cost``.

**Risk attitudes** (``goals.yaml`` ``risk``, E48): a power role in a crisis or an
intervention gets its leader's prospect-theory attitude (``risk``), which the family
applies to the role's total payoff, a gain or loss from the status quo.

**Identity roles** (attrition, ``model.identities``): a computer-run role that fights
for an identity not yet holding its seat (the Nationalists in China's civil war) values
outcomes by the share of the country it would govern. ``S`` is the identity's goal
score in the world after its side's victory (which must hand it the seat), with
consequences and follow-ons; it starts with share ``s0`` (side B: ``1 -
initial_control``). An outcome leaving it share ``s`` is worth ``kappa * S * (s - s0)``:
1 for its victory, its own share for a settlement, the stated ``stalemate_share`` for
a stalemate. Its defeat costs ``kappa * max(S * s0, loss_floor)``, the seat-loss rule.

**Decision events** (:func:`decision_values`) value each option the same way and
also count **major follow-ons**, one level deep: every later major event whose
chance the option changes adds ``(P(after) - P(before)) * E[value]``, where the
expectation is over that event's reference play (rollouts at the reference
temperature) at the world where it is likelier, with its own derived payoffs.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..families import alignment, attrition, bargaining, crisis, intervention
from . import goals
from .effects import EffectContext, EffectList
from .goals import GoalConfig
from .state import POWERS, WorldState, identity, seat

if TYPE_CHECKING:
    from .catalog import Catalog, EventSpec

DECIMALS = 2
MAX_PASSES = 4  # as the director


@dataclass
class Valuer:
    """Values effect sets for power roles at one world state."""

    catalog: Catalog
    event: EventSpec
    world: WorldState
    config: GoalConfig
    _cache: dict[Any, WorldState] = field(default_factory=dict)
    _applied: dict[tuple[int, str], WorldState] = field(default_factory=dict)

    @property
    def kappa(self) -> float:
        return self.config.kappa

    def after(
        self,
        effects: Iterable[EffectList],
        *,
        label: str | None = None,
        metrics: Mapping[str, float] | None = None,
        tags: tuple[str, ...] = (),
        forced: Mapping[int, bool] | None = None,
    ) -> WorldState:
        effect_lists = tuple(effects)
        key = (
            tuple(id(e) for e in effect_lists),
            label,
            tuple(sorted((metrics or {}).items())),
            tags,
            tuple(sorted((forced or {}).items())),
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        world = copy.deepcopy(self.world)
        ctx = EffectContext(
            leaders=self.catalog.leaders, metrics=dict(metrics or {}), forced=forced or {}
        )
        if label is not None:
            world.outcomes[self.event.id] = label
        if tags:
            world.outcome_tags[self.event.id] = tags
        for effect_list in effect_lists:
            effect_list.apply(world, ctx)
        if label is not None or tags:
            self._consequences(world)
        self._cache[key] = world
        return world

    def _consequences(self, world: WorldState) -> None:
        quarter = world.quarter + 1  # outcomes apply in RESOLVE; cutscenes fire next quarter
        applied: set[str] = set()
        for _ in range(MAX_PASSES):
            changed = False
            for event in self.catalog.ordered:
                if event.major is not None or event.id in applied or event.hazard < 1.0:
                    continue
                if event.once and event.id in world.fired:
                    continue
                if any(other in world.fired or other in applied for other in event.excludes):
                    continue
                if not event.in_window(quarter) or not event.condition(world):
                    continue
                if event.condition(self.world):
                    continue  # enabled already: not a consequence of this outcome
                event.effects.apply(world, EffectContext(leaders=self.catalog.leaders))
                applied.add(event.id)
                world.fired[event.id] = quarter
                changed = True
                if world.ended is not None:
                    return
            if not changed:
                return

    def value(self, after: WorldState, power: str) -> float:
        before = self.world
        rewards = self.catalog.rewards
        ident = identity(before, power)
        was_out = before.powers[power].status == "eliminated"
        lost = after.powers[power].status == "eliminated" and not was_out
        if lost or identity(after, power) != ident:
            # losing the seat: the goal score drops to zero (at least loss_floor)
            held = goals.score(before, ident, self.catalog, self.config)
            floor = float(rewards.get("loss_floor", 0.1))
            return round(-self.kappa * max(held, floor), DECIMALS)
        weights = goals.row(before, ident, self.catalog, self.config).weights
        start = goals.components(before, ident, self.catalog, self.config)
        end = goals.components(after, ident, self.catalog, self.config)
        total = goals.reward(start, end, weights) + self.follow_on(after, ident, weights, end)
        # end bonuses belong to the score reward only (E33); the nuclear-war penalty stays
        if after.nuclear_war and not before.nuclear_war:
            total -= float(rewards.get("nuclear_war_penalty", 20.0))
        return round(self.kappa * total, DECIMALS)

    # ------------------------------------------------------------ follow-ons
    def _span(self) -> tuple[int, int]:
        """First and last quarter of the lookahead: the event's maximum duration
        (capped at the lookahead) plus the lookahead."""

        major = self.event.major
        horizon = self.config.lookahead_quarters
        if major is None:  # a decision event: no duration of its own
            start = self.world.quarter + 1
            return start, start + horizon - 1
        config = major.config
        if major.family == "attrition":
            duration = int(config.get("max_rounds", 0))
        elif major.family == "bargaining" and major.cadence != "instant":
            duration = int(config.get("horizon", 0))
        else:
            duration = 0
        start = self.world.quarter + 1
        return start, start + min(duration, horizon) + horizon - 1

    def _chance(self, event: EventSpec, world: WorldState) -> float:
        """Discounted chance that *event* fires in the lookahead, read on *world*."""

        if event.once and event.id in world.fired:
            return 0.0
        if any(other in world.fired for other in event.excludes):
            return 0.0
        if not event.condition(world):
            return 0.0
        first, last = self._span()
        hazard = min(max(event.hazard, 0.0), 1.0)
        discount = self.config.lookahead_discount
        chance, survive = 0.0, 1.0
        for quarter in range(first, last + 1):
            if not event.in_window(quarter):
                continue
            chance += survive * hazard * discount ** (quarter - self.world.quarter)
            survive *= 1.0 - hazard
            if survive <= 1e-9:
                break
        return chance

    def follow_on(
        self, after: WorldState, ident: str, weights: Mapping[str, float], base: Mapping[str, float]
    ) -> float:
        """Goal value of the later cutscenes the outcome makes more or less likely."""

        if self.config.lookahead_quarters <= 0:
            return 0.0
        total = 0.0
        for event in self.catalog.ordered:
            if event.major is not None:
                continue
            change = self._chance(event, after) - self._chance(event, self.world)
            if abs(change) < 1e-9:
                continue
            key = (id(after), event.id)
            world = self._applied.get(key)
            if world is None:
                world = copy.deepcopy(after)
                event.effects.apply(world, EffectContext(leaders=self.catalog.leaders))
                self._applied[key] = world
            if world.ended is not None or identity(world, seat(ident)) != ident:
                continue  # campaign-ending or seat-changing follow-ons are left to the engine
            end = goals.components(world, ident, self.catalog, self.config)
            total += change * goals.reward(base, end, weights)
        return total

    def label_value(
        self, label: str, power: str, metrics: Mapping[str, float] | None = None
    ) -> float:
        major = self.event.major
        assert major is not None
        effects = major.outcomes[label].effects
        total = 0.0
        for forced, weight in _branches(effects, metrics):
            after = self.after([effects], label=label, metrics=metrics, forced=forced)
            value = self.value(after, power)
            if label == "nuclear_war" and not after.nuclear_war:
                # the label means nuclear war even if its effects forget to end the campaign
                penalty = float(self.catalog.rewards.get("nuclear_war_penalty", 20.0))
                value = round(value - self.kappa * penalty, DECIMALS)
            total += weight * value
        return round(total, DECIMALS)

    def tag_value(self, tag: str, power: str) -> float | None:
        major = self.event.major
        assert major is not None
        spec = major.tags.get(tag)
        if spec is None:
            return None
        return self.value(self.after([spec.effects], tags=(tag,)), power)


def _branches(
    effects: EffectList, metrics: Mapping[str, float] | None
) -> list[tuple[dict[int, bool], float]]:
    """Every combination of an effect list's lottery results, with its probability."""

    ctx = EffectContext(leaders={}, metrics=dict(metrics or {}))
    chances = [chance(ctx) for chance in effects.lotteries]
    result: list[tuple[dict[int, bool], float]] = [({}, 1.0)]
    for index, p in enumerate(chances):
        result = [
            ({**forced, index: won}, weight * (p if won else 1.0 - p))
            for forced, weight in result
            for won in (True, False)
        ]
    return [(forced, weight) for forced, weight in result if weight > 0.0]


# ------------------------------------------------------------------ per family
def _powers(ids: Iterable[str]) -> list[str]:
    return [i for i in ids if i in POWERS]


def _risk(v: Valuer, power: str) -> dict[str, float] | None:
    """The risk attitude of *power*'s leader (``goals.yaml`` ``risk``), or ``None`` when
    risk-neutral; the family applies it to the role's total payoff."""

    risk = v.config.risk_of(v.world.powers[power].leader)
    if risk == (1.0, 1.0, 1.0):
        return None
    return dict(zip(("alpha", "beta", "lambda"), risk, strict=True))


def _crisis(v: Valuer, config: dict[str, Any]) -> None:
    powers = _powers(config["roles"])
    for power in powers:
        role = config["roles"][power]
        role["values"] = {label: v.label_value(label, power) for label in crisis.OUTCOMES}
        risk = _risk(v, power)
        if risk is not None:
            role["risk"] = risk
        backed = v.tag_value(f"{power}:backed_down_after_commitment", power)
        if backed is not None:
            role["audience_cost"] = max(0.0, -backed)
    major = v.event.major
    assert major is not None
    for option, tag in _action_tags(config, major.tags):
        costs = dict(option.get("costs", {}))
        for power in powers:
            value = v.tag_value(tag, power)
            assert value is not None
            costs[power] = round(-value, DECIMALS)
        option["costs"] = costs


def _action_tags(
    config: Mapping[str, Any], tags: Iterable[str]
) -> list[tuple[dict[str, Any], str]]:
    """Crisis options whose own tag ("<actor>:move=<id>", ":signal=", ":partner=") has
    effects: their consequences, whoever the outcome, are the option's costs (E41)."""

    tag_set = set(tags)
    result = []
    for stage in config.get("script", []):
        kind = stage.get("stage")
        if kind not in ("move", "signal", "partner"):
            continue
        for option in stage["options"]:
            tag = f"{stage['actor']}:{kind}={option['id']}"
            if tag in tag_set:
                result.append((option, tag))
    return result


def effort_cost(v: Valuer, power: str) -> float:
    """Derived cost to *power* of one quarter of full war effort (module docstring)."""

    from .dynamics import DT, DynamicsParams

    params = DynamicsParams.from_catalog(v.catalog)
    ident = identity(v.world, power)
    weights = goals.row(v.world, ident, v.catalog, v.config).weights
    drag = params["war_growth_drag"] * DT
    economy = -math.log(1.0 - drag) / math.log(v.config.economy_scale)
    reversion = params["stability_reversion"] * DT
    stability = reversion * (
        4.0 * params["war_stability_drain"]
        + params["stability_growth_effect"] * params["war_growth_drag"]
    )
    d = v.config.lookahead_discount
    persistence = (1.0 - d) / (1.0 - d + d * reversion)
    total = weights.get("economy", 0.0) * economy
    total += weights.get("regime", 0.0) * stability * persistence
    return round(v.kappa * total, 4)


def casualty_cost(v: Valuer, power: str) -> float:
    """Derived domestic cost to *power* of one quarter fought with its own troops."""

    from .dynamics import DT, DynamicsParams

    params = DynamicsParams.from_catalog(v.catalog)
    ident = identity(v.world, power)
    weights = goals.row(v.world, ident, v.catalog, v.config).weights
    reversion = params["stability_reversion"] * DT
    drop = params["casualty_drain"] * params.casualty_sensitivity.get(power, 1.0)
    d = v.config.lookahead_discount
    persistence = (1.0 - d) / (1.0 - d + d * reversion)
    return round(v.kappa * weights.get("regime", 0.0) * reversion * drop * persistence, 4)


def _attrition(v: Valuer, config: dict[str, Any]) -> None:
    major = v.event.major
    assert major is not None
    labels = attrition.OUTCOMES + ((attrition.REFUSED,) if config.get("approval") else ())
    start = float(config.get("initial_control", 0.5))
    if major.effort_costs == "derived":
        config["costs"] = dict(zip(("low", "high", "direct"), (1 / 3, 2 / 3, 1.0), strict=True))

    # each sponsor's expected share of its side's effort (the outcome's side_share metric,
    # which the shared effects contract splits influence by): equal shares within a side
    sides: dict[str, list[str]] = {}
    for role, spec in config["sponsors"].items():
        sides.setdefault(str(spec.get("side")), []).append(role)
    expected = {f"side_share:{r}": 1.0 / len(rs) for rs in sides.values() for r in rs}

    def at_share(share: float, power: str) -> float:
        metrics = {"share": share, "share_change": share - start, **expected}
        return v.label_value("partition", power, metrics)

    def refused(power: str) -> float:
        """Refusing costs the refuser what its own "<role>:refused" tag does."""

        tag = major.tags.get(f"{power}:refused")
        if tag is None or power not in config.get("approval", ()):
            return v.label_value(attrition.REFUSED, power)
        after = v.after(
            [major.outcomes[attrition.REFUSED].effects, tag.effects],
            label=attrition.REFUSED,
            tags=(f"{power}:refused",),
        )
        return v.value(after, power)

    for power in _powers(config["sponsors"]):
        sponsor = config["sponsors"][power]
        sponsor["values"] = {
            label: (
                at_share(start, power)
                if label == "partition"
                else refused(power)
                if label == attrition.REFUSED
                else v.label_value(label, power, expected)
            )
            for label in labels
        }
        if major.effort_costs == "derived":
            sponsor["cost_scale"] = effort_cost(v, power)
            if sponsor.get("direct"):
                hand = float(sponsor.get("direct_cost", 0.0))
                sponsor["direct_cost"] = round(hand + casualty_cost(v, power), DECIMALS)
        if "offers" in config:
            # the line through the grid's end offers (clamped effects stay linear inside it)
            first, last = min(config["offers"]["grid"]), max(config["offers"]["grid"])
            low, high = at_share(first, power), at_share(last, power)
            slope = (high - low) / (last - first)  # per unit of A's share
            if sponsor["side"] == "a":  # A's own share is the share
                base, per_share = low - slope * first, slope
            else:
                base, per_share = high + slope * (1.0 - last), -slope
            sponsor["settlement"] = {
                "base": round(base, DECIMALS),
                "per_share": round(per_share, DECIMALS),
            }
        bluffed = v.tag_value(f"{power}:bluffed", power)
        if bluffed is not None and power in config.get("red_lines", {}):
            config["red_lines"][power]["audience_cost"] = max(0.0, -bluffed)
        conceded = v.tag_value(f"{power}:conceded", power)
        if conceded is not None and power in config.get("commitments", {}):
            config["commitments"][power]["audience_cost"] = max(0.0, -conceded)
        withdrew = v.tag_value(f"{power}:withdrew", power)  # abandoning an ally (E39)
        if withdrew is not None:
            sponsor["withdraw_cost"] = max(0.0, -withdrew)
    for role, (ident, stalemate_share) in major.identities.items():
        _identity_role(v, config, role, ident, stalemate_share, start)
    offers = config.get("offers")
    if offers is not None:
        for index, item in enumerate(offers.get("consent", [])):
            role = item if isinstance(item, str) else item["role"]
            vetoed = v.tag_value(f"{role}:vetoed", role) if role in POWERS else None
            if vetoed is not None:
                entry = {"role": role} if isinstance(item, str) else dict(item)
                entry["veto_cost"] = max(0.0, -vetoed)
                offers["consent"][index] = entry


def _identity_role(
    v: Valuer, config: dict[str, Any], role: str, ident: str, stalemate_share: float, start: float
) -> None:
    """Values for a computer-run *role* fighting for *ident* (module docstring)."""

    major = v.event.major
    assert major is not None
    sponsor = config["sponsors"][role]
    side = sponsor["side"]
    other = "b" if side == "a" else "a"
    win = f"{side}_victory"
    after = v.after([major.outcomes[win].effects], label=win)
    if identity(after, seat(ident)) != ident:
        raise ValueError(f"{v.event.id}: {win} does not hand {ident} its seat ({role})")
    weights = goals.row(after, ident, v.catalog, v.config).weights
    end = goals.components(after, ident, v.catalog, v.config)
    ruled = goals.combine(end, weights) + v.follow_on(after, ident, weights, end)
    held = start if side == "a" else 1.0 - start
    per_share = v.kappa * ruled

    def at(share: float) -> float:
        return round(per_share * (share - held), DECIMALS)

    floor = float(v.catalog.rewards.get("loss_floor", 0.1))
    lost = round(-v.kappa * max(ruled * held, floor), DECIMALS)
    values = {
        win: at(1.0),
        f"{other}_withdrew": at(1.0),
        f"{other}_victory": lost,
        f"{side}_withdrew": lost,
        "partition": at(held),
        "stalemate": at(stalemate_share),
    }
    if config.get("approval"):
        values[attrition.REFUSED] = at(held)
    sponsor["values"] = values
    if "offers" in config:  # by its own share
        sponsor["settlement"] = {
            "base": round(-per_share * held, DECIMALS),
            "per_share": round(per_share, DECIMALS),
        }


def _intervention(v: Valuer, config: dict[str, Any]) -> None:
    for key in ("patron", "rival"):
        role = config["roles"][key]
        if role in POWERS:
            config["values"][key] = {
                label: v.label_value(label, role) for label in intervention.OUTCOMES
            }
            risk = _risk(v, role)
            if risk is not None:
                config.setdefault("risk", {})[key] = risk


def _alignment(v: Valuer, config: dict[str, Any]) -> None:
    major = v.event.major
    assert major is not None
    patrons = list(config["patrons"])
    clients = list(config["clients"])

    def delta(client_index: int, choice: str, power: str) -> float:
        client = clients[client_index]
        if client_index == 0:
            return v.label_value(choice, power)
        value = v.tag_value(f"{client}:{choice}", power)
        return 0.0 if value is None else value

    for patron in _powers(patrons):
        rivals = [p for p in patrons if p != patron]
        gain, loss = {}, {}
        for index, client in enumerate(clients):
            base = delta(index, "nonaligned", patron)
            gain[client] = round(delta(index, f"aligned_{patron}", patron) - base, DECIMALS)
            if rivals:
                hurt = sum(delta(index, f"aligned_{r}", patron) - base for r in rivals) / len(
                    rivals
                )
                loss[client] = round(max(0.0, -hurt), DECIMALS)  # the family takes losses >= 0
        config["patrons"][patron]["gain"] = gain
        config["patrons"][patron]["loss"] = loss
        forbade = v.tag_value(f"{patron}:forbade", patron)
        if forbade is not None and config["patrons"][patron].get("control"):
            config["patrons"][patron]["forbid_cost"] = round(-forbade, DECIMALS)
        # what each offer is worth beyond the clients it wins, if the lead client takes it
        # up: its tag's effects and the follow-ons it makes likely (e.g. recovery)
        lead = f"aligned_{patron}"
        if lead in major.outcomes:
            worth = {}
            for level in alignment.LEVELS:
                offer = f"offer:{patron}={level}"
                spec = major.tags.get(offer)
                effects = [major.outcomes[lead].effects] + ([spec.effects] if spec else [])
                worth[level] = v.value(v.after(effects, label=lead, tags=(offer,)), patron)
            config["patrons"][patron]["offer_return"] = {
                level: round(worth[level] - worth["none"], DECIMALS) for level in alignment.LEVELS
            }
    for index, client in enumerate(clients):
        spec = config["clients"][client]
        if client in POWERS and spec.get("mode", "player") == "player":
            spec["affinity"] = {p: delta(index, f"aligned_{p}", client) for p in patrons}
            spec["nonaligned"] = delta(index, "nonaligned", client)


def _bargaining(v: Valuer, config: dict[str, Any]) -> None:
    major = v.event.major
    assert major is not None

    def at_share(share: float, power: str) -> float:
        return v.label_value("agreement", power, {"share": share})

    for key in bargaining.SIDE_ROLES:
        power = config["roles"][key]
        if power not in POWERS:
            continue
        side = config["sides"][key]
        low, high = at_share(0.0, power), at_share(1.0, power)
        if key == "A":  # A's own share is the share
            side["agreement"] = {"base": low, "per_share": round(high - low, DECIMALS)}
        else:  # B's own share is 1 - share
            side["agreement"] = {"base": high, "per_share": round(low - high, DECIMALS)}
        side["breakdown_value"] = v.label_value("breakdown", power)
        side["impasse_value"] = v.label_value("impasse", power)
        side["vetoed_value"] = v.label_value("vetoed", power)
        broke = v.tag_value(f"{power}:broke_commitment", power)
        if broke is not None and "commitment" in side:
            side["commitment_cost"] = max(0.0, -broke)
    for item in config.get("consent", []):
        power = item["role"]
        if power not in POWERS:
            continue
        low, high = at_share(0.0, power), at_share(1.0, power)
        item["base"] = low
        item["per_share"] = round(high - low, DECIMALS)
        # vetoing costs the vetoer what its own "<role>:vetoed" tag does (e.g. relations)
        veto_tag = major.tags.get(f"{power}:vetoed")
        if veto_tag is None:
            item["veto_value"] = v.label_value("vetoed", power)
        else:
            after = v.after([major.outcomes["vetoed"].effects, veto_tag.effects], label="vetoed")
            item["veto_value"] = v.value(after, power)
        item["breakdown_value"] = v.label_value("breakdown", power)
        item["impasse_value"] = v.label_value("impasse", power)


FAMILIES = {
    "crisis": _crisis,
    "attrition": _attrition,
    "intervention": _intervention,
    "alignment": _alignment,
    "bargaining": _bargaining,
}


def apply(
    catalog: Catalog,
    event: EventSpec,
    world: WorldState,
    config: dict[str, Any],
    goal_config: GoalConfig | None = None,
) -> dict[str, Any]:
    """Replace the power roles' values in a bound *config* with derived ones."""

    major = event.major
    assert major is not None
    valuer = Valuer(catalog, event, world, goal_config or catalog.goals)
    FAMILIES[major.family](valuer, config)
    return config


def slots(
    family: str,
    config: Mapping[str, Any],
    tags: Iterable[str],
    effort_costs: str = "hand",
    identities: Iterable[str] = (),
) -> list[tuple[Any, ...]]:
    """Config path prefixes that hold derived values (bindings may not target them)."""

    tag_set = set(tags)
    result: list[tuple[Any, ...]] = []
    if family == "crisis":
        for power in _powers(config["roles"]):
            result.append(("roles", power, "values"))
            result.append(("roles", power, "risk"))
            if f"{power}:backed_down_after_commitment" in tag_set:
                result.append(("roles", power, "audience_cost"))
        for index, stage in enumerate(config.get("script", [])):
            for position, option in enumerate(stage.get("options", [])):
                if f"{stage.get('actor')}:{stage.get('stage')}={option.get('id')}" in tag_set:
                    result.append(("script", index, "options", position, "costs"))
    elif family == "attrition":
        for power in _powers(config["sponsors"]):
            result.append(("sponsors", power, "values"))
            if effort_costs == "derived":
                result.append(("sponsors", power, "cost_scale"))
            if "offers" in config:
                result.append(("sponsors", power, "settlement"))
            if f"{power}:bluffed" in tag_set and power in config.get("red_lines", {}):
                result.append(("red_lines", power, "audience_cost"))
            if f"{power}:conceded" in tag_set and power in config.get("commitments", {}):
                result.append(("commitments", power, "audience_cost"))
            if f"{power}:withdrew" in tag_set:
                result.append(("sponsors", power, "withdraw_cost"))
        for index, item in enumerate(config.get("offers", {}).get("consent", [])):
            role = item if isinstance(item, str) else item.get("role")
            if role in POWERS and f"{role}:vetoed" in tag_set:
                result.append(("offers", "consent", index, "veto_cost"))
        for role in identities:
            result.append(("sponsors", role, "values"))
            if "offers" in config:
                result.append(("sponsors", role, "settlement"))
    elif family == "intervention":
        for key in ("patron", "rival"):
            if config["roles"][key] in POWERS:
                result.append(("values", key))
                result.append(("risk", key))
    elif family == "alignment":
        for patron in _powers(config["patrons"]):
            result.extend([("patrons", patron, "gain"), ("patrons", patron, "loss")])
            if f"{patron}:forbade" in tag_set and config["patrons"][patron].get("control"):
                result.append(("patrons", patron, "forbid_cost"))
            result.append(("patrons", patron, "offer_return"))
        for client in _powers(config["clients"]):
            result.extend([("clients", client, "affinity"), ("clients", client, "nonaligned")])
    elif family == "bargaining":
        for key in bargaining.SIDE_ROLES:
            power = config["roles"][key]
            if power in POWERS:
                for name in ("agreement", "breakdown_value", "impasse_value", "vetoed_value"):
                    result.append(("sides", key, name))
                if f"{power}:broke_commitment" in tag_set:
                    result.append(("sides", key, "commitment_cost"))
        for index, item in enumerate(config.get("consent", [])):
            if item["role"] in POWERS:
                for name in ("base", "per_share", "veto_value", "breakdown_value", "impasse_value"):
                    result.append(("consent", index, name))
    return result


# --------------------------------------------------------------- decisions
_EXPECTED: dict[str, float] = {}
_EXPECTED_LIMIT = 2048
FOLLOW_ON_ROLLOUTS = 200  # plain rollouts, for games that do not open with chance
STRATIFIED_ROLLOUTS = 72  # budget for the opening cells that remain open, by their mass (E39)
MIN_ROLLOUTS = 24
MAX_CELLS = 32


def _opening_cells(game: Any, policy: Any) -> list[tuple[Any, float]]:
    """The game's opening moves expanded exactly, chance by its odds and choices by the
    policy's probabilities, while the cells number at most MAX_CELLS: (state,
    probability) pairs. Cells that end the game (a refusal) need no rollouts."""

    from ..games import CHANCE

    cells: list[tuple[Any, float]] = [(game.new_initial_state(), 1.0)]
    while any(not s.is_terminal() for s, _ in cells):
        children: list[tuple[Any, float]] = []
        for state, weight in cells:
            if state.is_terminal():
                children.append((state, weight))
                continue
            if state.current_player() == CHANCE:
                moves = list(state.chance_outcomes())
            else:
                moves = list(policy.action_probabilities(state).items())
            for action, q in moves:
                if q <= 0.0:
                    continue
                child = state.clone()
                child.apply_action(action)
                children.append((child, weight * q))
        if len(children) > MAX_CELLS:
            break
        cells = children
    return cells


def _expected_value(
    catalog: Catalog, event: EventSpec, world: WorldState, power: str, goal_config: GoalConfig
) -> float:
    """Expected value to *power* of major *event* starting at *world*, under reference
    play (rollouts of the engine's baseline), with the event's own derived payoffs.

    The opening moves are expanded exactly (E39): cells that end the game count
    exactly, and a rollout budget in proportion to the probability left open (at
    least MIN_ROLLOUTS) is shared among the others by probability, at least one
    each. This removes the variance between cells (for a war, between the private
    types and between approval and refusal)."""

    import numpy as np

    from ..families import build_family
    from ..games import CHANCE, sample_action
    from .engine import EngineConfig, baseline_policy
    from .serialization import canonical_json

    major = event.major
    assert major is not None
    config = catalog.instantiate_config(event.id, world, goal_config)
    # the catalog's fingerprint: follow-on effects may differ between variants (E42)
    key = canonical_json(
        {"event": event.id, "config": config, "power": power, "catalog": catalog.fingerprint}
    )
    cached = _EXPECTED.get(key)
    if cached is not None:
        return cached
    defaults = EngineConfig()
    policy = baseline_policy(
        major.family, config, defaults.baseline_iterations, defaults.baseline_temperature
    )
    game = build_family(major.family, config)
    rng = np.random.default_rng(0)
    valuer = Valuer(catalog, event, world, goal_config)

    def worth(state: Any) -> float:
        if power in game.roles:
            return float(state.returns()[power])
        outcome = state.outcome()
        return valuer.label_value(outcome.label, power, outcome.metrics)

    cells = _opening_cells(game, policy)
    open_cells = [(s, w) for s, w in cells if not s.is_terminal()]
    mass = sum(w for _, w in open_cells)
    if len(cells) == 1:
        budget = FOLLOW_ON_ROLLOUTS
    else:
        budget = max(MIN_ROLLOUTS, round(STRATIFIED_ROLLOUTS * mass))
    value = sum(w * worth(s) for s, w in cells if s.is_terminal())
    for start, weight in open_cells:
        count = max(1, round(budget * weight / mass))
        total = 0.0
        for _ in range(count):
            state = start.clone()
            while not state.is_terminal():
                if state.current_player() == CHANCE:
                    state.apply_action(sample_action(dict(state.chance_outcomes()), rng))
                else:
                    state.apply_action(sample_action(policy.action_probabilities(state), rng))
            total += worth(state)
        value += weight * total / count
    if len(_EXPECTED) > _EXPECTED_LIMIT:
        _EXPECTED.clear()
    _EXPECTED[key] = value
    return value


def decision_values(
    catalog: Catalog, event: EventSpec, world: WorldState, goal_config: GoalConfig | None = None
) -> dict[str, float]:
    """Derived value of each option of decision *event* to its power at *world*:
    goal change, cutscene follow-ons and major follow-ons (one level deep)."""

    spec = event.decision
    assert spec is not None
    config = goal_config or catalog.goals
    valuer = Valuer(catalog, event, world, config)
    power = spec.power
    result = {}
    for option, outcome in spec.options.items():
        after = valuer.after([outcome.effects], label=option)
        value = valuer.value(after, power)
        if config.lookahead_quarters > 0:
            for other in catalog.ordered:
                if other.major is None or other.id == event.id:
                    continue
                change = valuer._chance(other, after) - valuer._chance(other, world)
                if abs(change) < 1e-9:
                    continue
                where = after if change > 0 else world
                value += change * _expected_value(catalog, other, where, power, config)
        result[option] = round(value, DECIMALS)
    return result


__all__ = ["DECIMALS", "Valuer", "apply", "decision_values", "slots"]
