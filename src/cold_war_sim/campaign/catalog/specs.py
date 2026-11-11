"""Catalog records and numeric addresses into the source documents."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from typing import Any

from ...families import build_family
from ...records import Record
from ..world.state import POWERS, WorldState
from .conditions import Predicate
from .effects import EffectList
from .paths import PathRef

CADENCES = ("instant", "quarterly", "annual")
KINDS = ("major", "cutscene", "leadership", "decision")


class CatalogError(ValueError):
    """Raised when campaign data fails validation."""


class Binding(Record):
    """A linear map from the world to one number of an event's config.

    The value at config path ``at`` becomes ``base + sum(coef x world value)``, clamped to
    ``clamp``; without a ``base`` the terms are added to the config's own value.
    """

    def __init__(
        self,
        at: tuple[str | int, ...],
        base: float | None,
        terms: tuple[tuple[PathRef, float], ...],
        clamp: tuple[float, float] | None,
    ) -> None:
        self.at = at
        self.base = base
        self.terms = terms
        self.clamp = clamp

    def value(self, world: WorldState, current: float) -> float:
        """The bound value in *world*, given the config's *current* one."""

        result = current if self.base is None else self.base
        for ref, coef in self.terms:
            result += coef * ref.get(world)
        if self.clamp is not None:
            result = min(self.clamp[1], max(self.clamp[0], result))
        return result


class OutcomeSpec(Record):
    """The effects of an outcome label, a tag or a decision's option, with its narrative."""

    def __init__(
        self,
        effects: EffectList,
        narrative: str,
    ) -> None:
        self.effects = effects
        self.narrative = narrative


class HistoricalPlan(Record):
    """Historical choices and chance draws for the instances of a major event that
    start before *before* (``None``: any later instance)."""

    def __init__(
        self,
        before: int | None,
        choices: Mapping[str, tuple[str, ...]],
        chance: tuple[str, ...],
    ) -> None:
        self.before = before
        self.choices = choices
        self.chance = chance


class DecisionSpec(Record):
    """A single power's choice among options with effects (``kind: decision``)."""

    def __init__(
        self,
        power: str,
        options: Mapping[str, OutcomeSpec],
        tier: int,
        historical: tuple[str, ...],
    ) -> None:
        self.power = power
        self.options = options
        self.tier = tier
        self.historical = historical  # one choice per firing; the last repeats

    def historical_choice(self, firing: int) -> str | None:
        """The option history took at the event's *firing*-th firing (the last one repeats)."""

        if not self.historical:
            return None
        return self.historical[min(firing, len(self.historical) - 1)]


class MajorSpec(Record):
    """A major event's sub-game: its family and config, the bindings that tie the config
    to the world, the effects of each outcome and tag, and the historical play.
    """

    def __init__(
        self,
        family: str,
        cadence: str,
        region: str | None,
        config: Mapping[str, Any],
        bindings: tuple[Binding, ...],
        outcomes: Mapping[str, OutcomeSpec],
        tags: Mapping[str, OutcomeSpec],
        historical_choices: Mapping[str, tuple[str, ...]],
        historical_chance: tuple[str, ...],
        historical_outcome: str | None,
        roles: tuple[str, ...],
        tier: int = 1,
        payoffs: str = "hand",
        holdout: bool = False,
        results: Mapping[str, Mapping[str, float]] | None = None,
        optional_roles: Mapping[str, Predicate] | None = None,
        plans: tuple[HistoricalPlan, ...] = (),
        effort_costs: str = "hand",
        identities: Mapping[str, tuple[str, float]] | None = None,
    ) -> None:
        self.family = family
        self.cadence = cadence
        self.region = region
        self.config = config
        self.bindings = bindings
        self.outcomes = outcomes
        self.tags = tags
        self.historical_choices = historical_choices
        self.historical_chance = historical_chance
        self.historical_outcome = historical_outcome
        self.roles = roles
        self.tier = (
            tier  # review priority; above EngineConfig.playable_tier the event runs on rails
        )
        self.payoffs = payoffs  # "derived": power roles' values come from their goals
        self.holdout = holdout  # historical choices never used when adjusting goals
        self.results: Mapping[str, Mapping[str, float]] = (
            {} if results is None else results
        )  # standing overrides
        # roles that take part only while their condition holds at the start (attrition)
        self.optional_roles: Mapping[str, Predicate] = (
            {} if optional_roles is None else optional_roles
        )
        self.plans = plans  # per-instance history for events that re-fire
        self.effort_costs = (
            effort_costs  # "derived": attrition effort costs from the dynamics and goals
        )
        # computer-run roles valued by an identity's goals: role -> (identity, stalemate share)
        self.identities: Mapping[str, tuple[str, float]] = {} if identities is None else identities

    def power_roles(self) -> tuple[str, ...]:
        """The roles played by powers (the rest are computer-run)."""

        return tuple(r for r in self.roles if r in POWERS)

    def plan(self, started: int) -> tuple[Mapping[str, tuple[str, ...]], tuple[str, ...]]:
        """Historical choices and chance for an instance started in quarter *started*."""

        for plan in self.plans:
            if plan.before is None or started < plan.before:
                return plan.choices, plan.chance
        return self.historical_choices, self.historical_chance

    def dropped_roles(self, world: WorldState) -> set[str]:
        """The optional roles whose condition fails in *world*, and so take no part."""

        return {role for role, condition in self.optional_roles.items() if not condition(world)}

    def bound_config(self, world: WorldState, drop: Iterable[str] | None = None) -> dict[str, Any]:
        """The config bound at *world*; optional roles whose condition fails (or the
        roles in *drop*, when given) are removed first."""

        config = copy.deepcopy(dict(self.config))
        dropped = set(self.dropped_roles(world) if drop is None else drop)
        if dropped:
            drop_roles(config, dropped)
        for binding in self.bindings:
            if dropped & {k for k in binding.at if isinstance(k, str)}:
                continue  # a binding on a role that is not taking part
            container: Any = config
            for key in binding.at[:-1]:
                container = container[key]
            last: Any = binding.at[-1]
            current = container[last] if isinstance(container, list) else container.get(last, 0.0)
            container[last] = binding.value(world, float(current))
        return config

    def build(self, world: WorldState) -> Any:
        """The game of *config*."""

        return build_family(self.family, self.bound_config(world))


def drop_roles(config: dict[str, Any], dropped: set[str]) -> None:
    """Remove attrition sponsors from a config; a dropped lead passes to the next
    sponsor on its side."""

    sponsors = config["sponsors"]
    side_of = {role: sponsors[role].get("side") for role in dropped if role in sponsors}
    for role in dropped:
        sponsors.pop(role, None)
    leads = config.get("leads", {})
    for side, lead in list(leads.items()):
        if lead in dropped:
            remaining = [r for r, spec in sponsors.items() if spec.get("side") == side]
            if not remaining:
                raise CatalogError(f"dropping {sorted(dropped)} leaves side {side} empty")
            leads[side] = remaining[0]
    if "approval" in config:
        config["approval"] = [r for r in config["approval"] if r not in dropped]
        if not config["approval"]:
            raise CatalogError(f"dropping {sorted(dropped)} leaves no approver")
    offers = config.get("offers")
    if offers is not None and "consent" in offers:
        gone = dropped | set(leads.values())  # a role that now leads needs no consent of its own
        offers["consent"] = [
            item
            for item in offers["consent"]
            if (item if isinstance(item, str) else item.get("role")) not in gone
        ]
    for role in dropped:
        config.get("red_lines", {}).pop(role, None)
    for role in dropped:  # a dropped role's own types, lead hand-over and lag go with it
        config.get("types", {}).pop(role, None)
        config.get("lead_on_entry", {}).pop(role, None)
        if isinstance(config.get("deployment"), dict):
            config["deployment"].pop(role, None)
    commitments = config.get("commitments", {})
    for role in [r for r in commitments if r in dropped]:  # passes to the side's new lead
        terms = commitments.pop(role)
        side = side_of.get(role)
        if side in leads:
            commitments.setdefault(leads[side], terms)


class EventSpec(Record):
    """An event of the catalog: when it may fire (window, condition, hazard), what it does
    (effects), and its sub-game or decision if it has one.
    """

    def __init__(
        self,
        id: str,
        kind: str,
        title: str,
        era: int,
        earliest: int | None,
        latest: int | None,
        condition: Predicate,
        hazard: float,
        once: bool,
        priority: int,
        excludes: tuple[str, ...],
        historical_date: int | None,
        narrative: str,
        provenance: str,
        effects: EffectList,
        major: MajorSpec | None,
        race: tuple[str, str] | None = None,
        decision: DecisionSpec | None = None,
    ) -> None:
        self.id = id
        self.kind = kind
        self.title = title
        self.era = era
        self.earliest = earliest
        self.latest = latest
        self.condition = condition
        self.hazard = hazard
        self.once = once
        self.priority = priority
        self.excludes = excludes
        self.historical_date = historical_date
        self.narrative = narrative
        self.provenance = provenance
        self.effects = effects
        self.major = major
        self.race = race  # (winner, loser) of a space-race first
        self.decision = decision

    def in_window(self, quarter: int) -> bool:
        """Whether *quarter* lies in the event's window."""

        if self.earliest is not None and quarter < self.earliest:
            return False
        return not (self.latest is not None and quarter > self.latest)

    def __deepcopy__(self, memo: dict[int, object]) -> EventSpec:
        return self


class PowerConfig(Record):
    """A power's opening state and constants."""

    def __init__(
        self,
        id: str,
        name: str,
        initial: Mapping[str, Any],
        tech_rates: Mapping[str, float],
        military_efficiency: float,
        score_weights: Mapping[str, float],
        casualty_sensitivity: float,
        end_bonus: Mapping[str, float],
    ) -> None:
        self.id = id
        self.name = name
        self.initial = initial
        self.tech_rates = tech_rates
        self.military_efficiency = military_efficiency
        self.score_weights = score_weights
        self.casualty_sensitivity = casualty_sensitivity
        self.end_bonus = end_bonus


class RegionConfig(Record):
    """A region's opening state and constants."""

    def __init__(
        self,
        id: str,
        name: str,
        importance: float,
        stability: float,
        base_stability: float,
        colonial_owner: str | None,
        influence: Mapping[str, float],
    ) -> None:
        self.id = id
        self.name = name
        self.importance = importance
        self.stability = stability
        self.base_stability = base_stability
        self.colonial_owner = colonial_owner
        self.influence = influence


DOCUMENTS = (
    "leaders",
    "powers",
    "regions",
    "initial",
    "strategy",
    "goals",
    "contract",
    "generator",
)
# documents the shared effects contract reads: a change recompiles every opted-in event
CONTRACT_INPUTS = ("contract", "regions", "powers")


class RawDocuments(Record):
    """The parsed source documents, kept so that a catalog can be recompiled with some
    numbers changed (:meth:`Catalog.variant`). Never mutated: a variant copies the
    documents it changes."""

    def __init__(
        self,
        leaders: Any,
        powers: Any,
        regions: Any,
        initial: Any,
        strategy: Any,
        goals: Any,
        events: tuple[tuple[int, dict[str, Any]], ...],
        era_titles: Mapping[int, str],
        contract: Any = None,
        generator: Any = None,
    ) -> None:
        self.leaders = leaders
        self.powers = powers
        self.regions = regions
        self.initial = initial
        self.strategy = strategy
        self.goals = goals
        self.events = events  # (era, raw event), in load order
        self.era_titles = era_titles
        self.contract = contract  # the shared effects contract (data/contract.yaml)
        self.generator = (
            generator  # the crisis generator's templates and rates (data/generator.yaml)
        )

    def document(self, name: str) -> Any:
        """The document named *name* (``event:<id>`` for one event's entry)."""

        if name.startswith("event:"):
            event_id = name.removeprefix("event:")
            for _, raw in self.events:
                if raw["id"] == event_id:
                    return raw
            raise CatalogError(f"unknown event {event_id!r}")
        if name not in DOCUMENTS:
            raise CatalogError(f"unknown document {name!r}")
        return getattr(self, name)

    def __deepcopy__(self, memo: dict[int, object]) -> RawDocuments:
        return self


def split_address(address: str) -> tuple[str, tuple[str, ...]]:
    """``"goals:kappa"`` -> ``("goals", ("kappa",))``; ``"event:x/a/0/b"`` ->
    ``("event:x", ("a", "0", "b"))``; ``"dynamics:key"`` names a dynamics coefficient,
    stored in the powers document's ``dynamics`` block. Paths use ``/`` because some keys
    (binding terms) contain dots."""

    doc, sep, path = address.partition(":")
    if not sep or not path:
        raise CatalogError(f"bad address {address!r}")
    if doc == "event":
        event_id, _, rest = path.partition("/")
        if not rest:
            raise CatalogError(f"bad address {address!r}")
        return f"event:{event_id}", tuple(rest.split("/"))
    if doc == "dynamics":
        return "powers", ("dynamics", path)
    return doc, tuple(path.split("/"))


def _step(container: Any, key: str) -> Any:
    """The child of *container* under *key* (a list index if the container is a list)."""

    if isinstance(container, list):
        return int(key)
    if isinstance(container, dict) and key not in container and key.lstrip("-").isdigit():
        return int(key)
    return key


def get_number(document: Any, keys: tuple[str, ...]) -> Any:
    """The value at *keys* in a raw document (``KeyError``/``IndexError`` if absent)."""

    value = document
    for key in keys:
        value = value[_step(value, key)]
    return value


def set_number(document: Any, keys: tuple[str, ...], value: float) -> None:
    """Set the value at *keys*; the last key may be new only in a mapping."""

    container = document
    for key in keys[:-1]:
        container = container[_step(container, key)]
    container[_step(container, keys[-1])] = value
