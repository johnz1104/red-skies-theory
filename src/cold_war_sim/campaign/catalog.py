"""Loading and validating the campaign data: powers, regions, leaders, events.

Everything that can be checked statically is checked when the catalog loads:
JSON-schema structure, world paths, event and leader references, family
configurations (each major event is built once against the opening world),
outcome coverage, and historical choices against each role's action set.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from importlib import resources
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from ..families import FAMILIES, build_family
from ..families.attrition import LEVELS as ATTRITION_LEVELS
from ..families.common import ConfigError
from ..games import CHANCE, build_tree
from ..games.traverse import DECISION
from . import calendar, contract, derive, generator
from .conditions import CompileContext, ConditionError, Predicate, compile_condition
from .dynamics import POSTURE_KEYS
from .effects import EffectList, LeaderSpec, apply_leader, compile_effects
from .goals import GoalConfig, GoalError, parse_goals
from .paths import PathRef, compile_path
from .serialization import stable_hash
from .state import (
    IDENTITIES,
    NEUTRAL,
    POSTURES,
    POWER_FIELDS,
    POWERS,
    REGIMES,
    REGIONS,
    STATUSES,
    PowerState,
    RegionState,
    WorldState,
    normalize_influence,
    update_controller,
)

CADENCES = ("instant", "quarterly", "annual")
KINDS = ("major", "cutscene", "leadership", "decision")


class CatalogError(ValueError):
    """Raised when campaign data fails validation."""


@dataclass(frozen=True)
class Binding:
    at: tuple[str | int, ...]
    base: float | None
    terms: tuple[tuple[PathRef, float], ...]
    clamp: tuple[float, float] | None

    def value(self, world: WorldState, current: float) -> float:
        result = current if self.base is None else self.base
        for ref, coef in self.terms:
            result += coef * ref.get(world)
        if self.clamp is not None:
            result = min(self.clamp[1], max(self.clamp[0], result))
        return result


@dataclass(frozen=True)
class OutcomeSpec:
    effects: EffectList
    narrative: str


@dataclass(frozen=True)
class HistoricalPlan:
    """Historical choices and chance draws for the instances of a major event that
    start before *before* (``None``: any later instance)."""

    before: int | None
    choices: Mapping[str, tuple[str, ...]]
    chance: tuple[str, ...]


@dataclass(frozen=True)
class DecisionSpec:
    """A single power's choice among options with effects (``kind: decision``)."""

    power: str
    options: Mapping[str, OutcomeSpec]
    tier: int
    historical: tuple[str, ...]  # one choice per firing; the last repeats

    def historical_choice(self, firing: int) -> str | None:
        if not self.historical:
            return None
        return self.historical[min(firing, len(self.historical) - 1)]


@dataclass(frozen=True)
class MajorSpec:
    family: str
    cadence: str
    region: str | None
    config: Mapping[str, Any]
    bindings: tuple[Binding, ...]
    outcomes: Mapping[str, OutcomeSpec]
    tags: Mapping[str, OutcomeSpec]
    historical_choices: Mapping[str, tuple[str, ...]]
    historical_chance: tuple[str, ...]
    historical_outcome: str | None
    roles: tuple[str, ...]
    tier: int = 1  # review priority; above EngineConfig.playable_tier the event runs on rails
    payoffs: str = "hand"  # "derived": power roles' values come from their goals
    holdout: bool = False  # historical choices never used when adjusting goals
    results: Mapping[str, Mapping[str, float]] = field(default_factory=dict)  # standing overrides
    # roles that take part only while their condition holds at the start (attrition)
    optional_roles: Mapping[str, Predicate] = field(default_factory=dict)
    plans: tuple[HistoricalPlan, ...] = ()  # per-instance history for events that re-fire
    effort_costs: str = "hand"  # "derived": attrition effort costs from the dynamics and goals
    # computer-run roles valued by an identity's goals: role -> (identity, stalemate share)
    identities: Mapping[str, tuple[str, float]] = field(default_factory=dict)

    def power_roles(self) -> tuple[str, ...]:
        return tuple(r for r in self.roles if r in POWERS)

    def plan(self, started: int) -> tuple[Mapping[str, tuple[str, ...]], tuple[str, ...]]:
        """Historical choices and chance for an instance started in quarter *started*."""

        for plan in self.plans:
            if plan.before is None or started < plan.before:
                return plan.choices, plan.chance
        return self.historical_choices, self.historical_chance

    def dropped_roles(self, world: WorldState) -> set[str]:
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


@dataclass(frozen=True)
class EventSpec:
    id: str
    kind: str
    title: str
    era: int
    earliest: int | None
    latest: int | None
    condition: Predicate
    hazard: float
    once: bool
    priority: int
    excludes: tuple[str, ...]
    historical_date: int | None
    narrative: str
    provenance: str
    effects: EffectList
    major: MajorSpec | None
    race: tuple[str, str] | None = None  # (winner, loser) of a space-race first
    decision: DecisionSpec | None = None

    def in_window(self, quarter: int) -> bool:
        if self.earliest is not None and quarter < self.earliest:
            return False
        return not (self.latest is not None and quarter > self.latest)

    def __deepcopy__(self, memo: dict[int, object]) -> EventSpec:
        return self


@dataclass(frozen=True)
class PowerConfig:
    id: str
    name: str
    initial: Mapping[str, Any]
    tech_rates: Mapping[str, float]
    military_efficiency: float
    score_weights: Mapping[str, float]
    casualty_sensitivity: float
    end_bonus: Mapping[str, float]


@dataclass(frozen=True)
class RegionConfig:
    id: str
    name: str
    importance: float
    stability: float
    base_stability: float
    colonial_owner: str | None
    influence: Mapping[str, float]


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


@dataclass(frozen=True)
class RawDocuments:
    """The parsed source documents, kept so that a catalog can be recompiled with some
    numbers changed (:meth:`Catalog.variant`). Never mutated: a variant copies the
    documents it changes."""

    leaders: Any
    powers: Any
    regions: Any
    initial: Any
    strategy: Any
    goals: Any
    events: tuple[tuple[int, dict[str, Any]], ...]  # (era, raw event), in load order
    era_titles: Mapping[int, str]
    contract: Any = None  # the shared effects contract (data/contract.yaml)
    generator: Any = None  # the crisis generator's templates and rates (data/generator.yaml)

    def document(self, name: str) -> Any:
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


class EventTable(dict[str, "EventSpec"]):
    """The catalog's events by id. Generated crises (``gen:...`` ids, :mod:`.generator`)
    are compiled on first use and kept apart, so iterating the table lists only the
    catalog's own events while any generated id still resolves (in any process)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.generated: dict[str, EventSpec] = {}
        self.owner: Catalog | None = None

    def __missing__(self, key: str) -> EventSpec:
        spec = self.generated.get(key)
        if spec is None:
            if self.owner is None or not generator.is_generated(key):
                raise KeyError(key)
            spec = _compile_generated(self.owner, key)
            self.generated[key] = spec
        return spec

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, CatalogError):
            return default


@dataclass
class Catalog:
    events: dict[str, EventSpec]
    leaders: dict[str, LeaderSpec]
    powers: dict[str, PowerConfig]
    regions: dict[str, RegionConfig]
    relations: dict[tuple[str, str], float]
    initial_flags: frozenset[str]
    initial_tension: float
    dynamics: Mapping[str, Any]
    rewards: Mapping[str, Any]
    historical_strategy: Mapping[str, tuple[Mapping[str, Any], ...]]
    goals: GoalConfig
    era_titles: dict[int, str] = field(default_factory=dict)
    cleared_flags: frozenset[str] = frozenset()  # flags some effect in the catalog unsets
    postures: Mapping[str, tuple[float, float, float, float]] = field(default_factory=dict)
    raw: RawDocuments | None = None  # the source documents, for variants
    fingerprint: str = ""  # "" for the catalog as loaded; a digest of a variant's changes

    def __deepcopy__(self, memo: dict[int, object]) -> Catalog:
        return self

    def variant(self, values: Mapping[str, float], fingerprint: str | None = None) -> Catalog:
        """A copy with the numbers at the given addresses (:func:`split_address`) changed.

        Only the documents that change are recompiled; the game-tree validation is not
        repeated, since changing a number never changes an event's structure (structural
        numbers are fixed in the assumptions register)."""

        if not values:
            return self
        raw = self.raw
        if raw is None:
            raise CatalogError("this catalog keeps no source documents")
        changes: dict[str, list[tuple[tuple[str, ...], float]]] = {}
        for address, value in values.items():
            doc, keys = split_address(address)
            changes.setdefault(doc, []).append((keys, float(value)))
        patched: dict[str, Any] = {}
        for doc, items in changes.items():
            document = copy.deepcopy(raw.document(doc))
            for keys, value in items:
                if doc == "powers" and keys[0] == "dynamics":
                    document.setdefault("dynamics", {})
                else:
                    try:
                        get_number(document, keys)
                    except (KeyError, IndexError, ValueError, TypeError) as exc:
                        raise CatalogError(f"no number at {doc}:{'/'.join(keys)}") from exc
                set_number(document, keys, value)
            patched[doc] = document
        table = EventTable(self.events)
        new = replace(self, events=table)
        table.owner = new
        if "leaders" in patched:
            new.leaders = _parse_leaders(patched["leaders"])
        if "powers" in patched:
            powers, relations, dynamics, rewards, postures = _parse_powers(
                patched["powers"], new.leaders
            )
            new.powers, new.relations, new.dynamics = powers, relations, dynamics
            new.rewards, new.postures = rewards, postures
        if "regions" in patched:
            new.regions = _parse_regions(patched["regions"])
        if "initial" in patched:
            new.initial_tension = float(patched["initial"].get("tension", 0.3))
        if "goals" in patched:
            new.goals = _parse_goals(patched["goals"], new.leaders)
        events = list(raw.events)
        inputs_changed = any(name in patched for name in CONTRACT_INPUTS)
        contract_doc = patched.get("contract", raw.contract)
        regions_doc = patched.get("regions", raw.regions)
        dynamics_in = contract.dynamics_of(patched.get("powers", raw.powers))
        sensitivity_in = contract.sensitivity_of(patched.get("powers", raw.powers))
        for index, (era, raw_event) in enumerate(raw.events):
            doc = f"event:{raw_event['id']}"
            opted = (raw_event.get("model") or {}).get("effects") == "contract"
            if doc not in patched and not (opted and inputs_changed):
                continue
            source = patched.get(doc, raw_event)
            events[index] = (era, source)
            base = self.events[raw_event["id"]]
            compiled = _expanded(source, contract_doc, regions_doc, dynamics_in, sensitivity_in)
            event = _parse_event(compiled, era, CompileContext())
            object.__setattr__(event, "excludes", base.excludes)
            if event.major is not None and base.major is not None:
                object.__setattr__(event.major, "roles", base.major.roles)
            new.events[event.id] = event
        new.raw = replace(
            raw,
            events=tuple(events),
            **{name: patched[name] for name in DOCUMENTS if name in patched},
        )
        new.fingerprint = fingerprint if fingerprint is not None else stable_hash(dict(values))
        return new

    @property
    def ordered(self) -> tuple[EventSpec, ...]:
        return tuple(sorted(self.events.values(), key=lambda e: (-e.priority, e.id)))

    @property
    def major_ids(self) -> tuple[str, ...]:
        return tuple(sorted(e.id for e in self.events.values() if e.kind == "major"))

    @property
    def observed_flags(self) -> tuple[str, ...]:
        flags = set(self.initial_flags)
        for event in self.events.values():
            flags |= event.effects.flags_set
            if event.major is not None:
                for outcome in (*event.major.outcomes.values(), *event.major.tags.values()):
                    flags |= outcome.effects.flags_set
            if event.decision is not None:
                for outcome in event.decision.options.values():
                    flags |= outcome.effects.flags_set
        return tuple(sorted(flags))

    def instantiate_config(
        self, event_id: str, world: WorldState, goal_config: GoalConfig | None = None
    ) -> dict[str, Any]:
        """The event's config bound at *world*, with derived payoffs if it uses them."""

        event = self.events[event_id]
        major = event.major
        assert major is not None
        config = major.bound_config(world)
        if major.payoffs == "derived":
            derive.apply(self, event, world, config, goal_config)
        return config

    def counts(self) -> dict[str, int]:
        counts = dict.fromkeys(KINDS, 0)
        for event in self.events.values():
            counts[event.kind] += 1
        return counts

    def initial_world(self) -> WorldState:
        start = calendar.START
        powers: dict[str, PowerState] = {}
        for power_id in POWERS:
            cfg = self.powers[power_id]
            state = PowerState(relations={q: 0.0 for q in POWERS if q != power_id})
            for key, value in cfg.initial.items():
                if key == "leader":
                    continue
                setattr(state, key, value)
            powers[power_id] = state
        for (a, b), value in self.relations.items():
            powers[a].relations[b] = value
            powers[b].relations[a] = value
        regions: dict[str, RegionState] = {}
        for region_id in REGIONS:
            cfg_r = self.regions[region_id]
            region = RegionState(
                influence={p: float(cfg_r.influence.get(p, 0.0)) for p in POWERS},
                stability=cfg_r.stability,
                importance=cfg_r.importance,
                base_stability=cfg_r.base_stability,
                colonial_owner=cfg_r.colonial_owner,
            )
            normalize_influence(region)
            update_controller(region)
            regions[region_id] = region
        world = WorldState(
            quarter=start,
            tension=self.initial_tension,
            flags=set(self.initial_flags),
            powers=powers,
            regions=regions,
        )
        for power_id in POWERS:
            leader_id = self.powers[power_id].initial.get("leader")
            if leader_id:
                apply_leader(world, self.leaders[leader_id])
        return world


# --------------------------------------------------------------------------- IO
CONTRACT_FILE = "contract.yaml"
GENERATOR_FILE = "generator.yaml"


def data_root() -> Path:
    return Path(str(resources.files("cold_war_sim.campaign") / "data"))


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _event_schema(root: Path) -> dict[str, Any]:
    import json

    with (root / "schema" / "event-catalog.schema.json").open("r", encoding="utf-8") as handle:
        schema: dict[str, Any] = json.load(handle)
    return schema


def load_catalog(root: Path | None = None, *, event_files: Iterable[Path] | None = None) -> Catalog:
    """Load and validate the campaign data under *root* (default: package data)."""

    return compile_catalog(read_documents(root, event_files=event_files))


def read_documents(
    root: Path | None = None, *, event_files: Iterable[Path] | None = None
) -> RawDocuments:
    """Read the source documents, checking the event files against the schema."""

    root = root or data_root()
    files = (
        sorted(event_files) if event_files is not None else sorted((root / "events").glob("*.yaml"))
    )
    schema = _event_schema(root)
    raw_events: list[tuple[int, dict[str, Any]]] = []
    era_titles: dict[int, str] = {}
    for path in files:
        doc = _load_yaml(path)
        try:
            jsonschema.validate(doc, schema)
        except jsonschema.ValidationError as exc:
            location = "/".join(str(p) for p in exc.absolute_path)
            raise CatalogError(f"{path.name}: {location}: {exc.message}") from exc
        era_titles[int(doc["era"])] = str(doc["title"])
        raw_events.extend((int(doc["era"]), event) for event in doc["events"])
    return RawDocuments(
        leaders=_load_yaml(root / "leaders.yaml"),
        powers=_load_yaml(root / "powers.yaml"),
        regions=_load_yaml(root / "regions.yaml"),
        initial=_load_yaml(root / "initial_1945.yaml") or {},
        strategy=_load_yaml(root / "historical_strategy.yaml") or {},
        goals=_load_yaml(root / "goals.yaml"),
        events=tuple(raw_events),
        era_titles=era_titles,
        contract=_load_yaml(root / CONTRACT_FILE) if (root / CONTRACT_FILE).exists() else None,
        generator=_load_yaml(root / GENERATOR_FILE) if (root / GENERATOR_FILE).exists() else None,
    )


def compile_catalog(raw: RawDocuments) -> Catalog:
    """Compile and validate the source documents."""

    leaders = _parse_leaders(raw.leaders)
    powers, relations, dynamics, rewards, postures = _parse_powers(raw.powers, leaders)
    regions = _parse_regions(raw.regions)
    strategy = _parse_strategy(raw.strategy)
    goals = _parse_goals(raw.goals, leaders)
    table = EventTable()
    catalog = Catalog(
        events=table,
        leaders=leaders,
        powers=powers,
        regions=regions,
        relations=relations,
        initial_flags=frozenset(raw.initial.get("flags", [])),
        initial_tension=float(raw.initial.get("tension", 0.3)),
        dynamics=dynamics,
        rewards=rewards,
        historical_strategy=strategy,
        goals=goals,
        era_titles=dict(raw.era_titles),
        postures=postures,
        raw=raw,
    )
    table.owner = catalog
    ctx = CompileContext()
    dynamics_in = contract.dynamics_of(raw.powers)
    sensitivity_in = contract.sensitivity_of(raw.powers)
    for era, raw_event in raw.events:
        expanded = _expanded(raw_event, raw.contract, raw.regions, dynamics_in, sensitivity_in)
        event = _parse_event(expanded, era, ctx)
        if event.id in catalog.events:
            raise CatalogError(f"duplicate event id {event.id!r}")
        catalog.events[event.id] = event
    _validate_references(catalog, ctx)
    _symmetric_excludes(catalog)
    catalog.cleared_flags = frozenset(ctx.flags_cleared)
    world = catalog.initial_world()
    for event in catalog.events.values():
        if event.major is not None:
            _validate_major(event, world)
    for event in catalog.events.values():
        if event.major is not None and event.major.payoffs == "derived":
            _validate_derived(catalog, event, world)
    return catalog


# ---------------------------------------------------------------------- parsing
def _parse_goals(doc: Any, leaders: Mapping[str, LeaderSpec]) -> GoalConfig:
    ctx = CompileContext()
    try:
        goals = parse_goals(doc, ctx)
    except (GoalError, ConditionError) as exc:
        raise CatalogError(f"goals.yaml: {exc}") from exc
    if ctx.event_refs:
        raise CatalogError(f"goals.yaml: conditions may not name events {sorted(ctx.event_refs)}")
    unknown = ctx.leaders - set(leaders)
    if unknown:
        raise CatalogError(f"goals.yaml: unknown leaders {sorted(unknown)}")
    return goals


def _parse_leaders(doc: Any) -> dict[str, LeaderSpec]:
    leaders: dict[str, LeaderSpec] = {}
    for leader_id, raw in (doc or {}).get("leaders", {}).items():
        if raw.get("power") not in POWERS:
            raise CatalogError(f"leader {leader_id}: unknown power {raw.get('power')!r}")
        traits = {}
        for trait in ("hawkishness", "risk_tolerance", "reformism"):
            value = float(raw.get(trait, 0.5))
            if not 0.0 <= value <= 1.0:
                raise CatalogError(f"leader {leader_id}: {trait} must lie in [0, 1]")
            traits[trait] = value
        leaders[leader_id] = LeaderSpec(
            id=leader_id, power=raw["power"], name=str(raw.get("name", leader_id)), **traits
        )
    return leaders


Postures = dict[str, tuple[float, float, float, float]]


def _parse_powers(
    doc: Any, leaders: Mapping[str, LeaderSpec]
) -> tuple[
    dict[str, PowerConfig], dict[tuple[str, str], float], dict[str, Any], dict[str, Any], Postures
]:
    raw_powers = doc.get("powers", {})
    if set(raw_powers) != set(POWERS):
        raise CatalogError(f"powers.yaml must define exactly {POWERS}")
    allowed_initial = set(POWER_FIELDS) | {"status", "leader", "posture", "regime"}
    powers: dict[str, PowerConfig] = {}
    for power_id in POWERS:
        raw = raw_powers[power_id]
        initial = dict(raw.get("initial", {}))
        unknown = set(initial) - allowed_initial
        if unknown:
            raise CatalogError(f"powers.{power_id}.initial: unknown fields {sorted(unknown)}")
        if initial.get("status", "active") not in STATUSES:
            raise CatalogError(f"powers.{power_id}.initial.status is invalid")
        if initial.get("posture", "balanced") not in POSTURES:
            raise CatalogError(f"powers.{power_id}.initial.posture is invalid")
        if initial.get("regime", REGIMES[0]) not in REGIMES:
            raise CatalogError(f"powers.{power_id}.initial.regime is invalid")
        leader = initial.get("leader")
        if leader is not None and (leader not in leaders or leaders[leader].power != power_id):
            raise CatalogError(
                f"powers.{power_id}.initial.leader {leader!r} is not a leader of {power_id}"
            )
        for key, (low, high) in POWER_FIELDS.items():
            if key in initial and not low <= float(initial[key]) <= high:
                raise CatalogError(f"powers.{power_id}.initial.{key} outside [{low}, {high}]")
        powers[power_id] = PowerConfig(
            id=power_id,
            name=str(raw.get("name", power_id)),
            initial=initial,
            tech_rates={k: float(v) for k, v in raw.get("tech_rates", {}).items()},
            military_efficiency=float(raw.get("military_efficiency", 1.0)),
            score_weights={k: float(v) for k, v in raw.get("score_weights", {}).items()},
            end_bonus={k: float(v) for k, v in raw.get("end_bonus", {}).items()},
            casualty_sensitivity=float(raw.get("casualty_sensitivity", 1.0)),
        )
    relations: dict[tuple[str, str], float] = {}
    for key, value in doc.get("relations", {}).items():
        a, _, b = str(key).partition("-")
        if a not in POWERS or b not in POWERS or a == b:
            raise CatalogError(f"relations key {key!r} must be POWER-POWER")
        relations[(a, b)] = float(value)
    dynamics, rewards = dict(doc.get("dynamics", {})), dict(doc.get("rewards", {}))
    return powers, relations, dynamics, rewards, _parse_postures(doc.get("postures", {}))


def _parse_postures(raw: Mapping[str, Any]) -> Postures:
    if set(raw) != set(POSTURES):
        raise CatalogError(f"powers.yaml postures must define exactly {sorted(POSTURES)}")
    postures: Postures = {}
    for posture in POSTURES:
        shares = raw[posture]
        if set(shares) != set(POSTURE_KEYS):
            raise CatalogError(f"postures.{posture} must give shares for {POSTURE_KEYS}")
        values = tuple(float(shares[key]) for key in POSTURE_KEYS)
        if min(values) < 0.0 or abs(sum(values) - 1.0) > 1e-6:
            raise CatalogError(f"postures.{posture}: shares must be non-negative and sum to 1")
        postures[posture] = (values[0], values[1], values[2], values[3])
    return postures


def _parse_regions(doc: Any) -> dict[str, RegionConfig]:
    raw_regions = doc.get("regions", {})
    if set(raw_regions) != set(REGIONS):
        missing = set(REGIONS) - set(raw_regions)
        extra = set(raw_regions) - set(REGIONS)
        raise CatalogError(
            f"regions.yaml mismatch: missing {sorted(missing)}, extra {sorted(extra)}"
        )
    regions = {}
    for region_id in REGIONS:
        raw = raw_regions[region_id]
        influence = {k: float(v) for k, v in raw.get("influence", {}).items()}
        if not set(influence) <= set(POWERS):
            raise CatalogError(f"regions.{region_id}.influence names unknown powers")
        if sum(influence.values()) > 1.0 + 1e-9:
            raise CatalogError(f"regions.{region_id}.influence sums above 1")
        owner = raw.get("colonial_owner")
        if owner is not None and owner not in POWERS:
            raise CatalogError(f"regions.{region_id}.colonial_owner is not a power")
        regions[region_id] = RegionConfig(
            id=region_id,
            name=str(raw.get("name", region_id)),
            importance=float(raw.get("importance", 1.0)),
            stability=float(raw.get("stability", 0.6)),
            base_stability=float(raw.get("base_stability", raw.get("stability", 0.6))),
            colonial_owner=owner,
            influence=influence,
        )
    return regions


def _parse_strategy(doc: Mapping[str, Any]) -> dict[str, tuple[Mapping[str, Any], ...]]:
    result: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for power_id, rows in doc.items():
        if power_id not in POWERS:
            raise CatalogError(f"historical_strategy: unknown power {power_id!r}")
        parsed = []
        for row in rows:
            unknown = set(row) - {"from", "to", "posture", "focus", "diplomacy"}
            if unknown:
                raise CatalogError(
                    f"historical_strategy.{power_id}: unknown keys {sorted(unknown)}"
                )
            if row.get("posture", "balanced") not in POSTURES:
                raise CatalogError(
                    f"historical_strategy.{power_id}: bad posture {row.get('posture')!r}"
                )
            focus = row.get("focus")
            if focus is not None and (
                len(focus) != 2 or focus[0] not in REGIONS or focus[1] not in ("aid", "covert")
            ):
                raise CatalogError(f"historical_strategy.{power_id}: bad focus {focus!r}")
            diplomacy = row.get("diplomacy")
            if diplomacy is not None and (
                len(diplomacy) != 2
                or diplomacy[0] not in ("engage", "pressure")
                or diplomacy[1] not in POWERS
            ):
                raise CatalogError(f"historical_strategy.{power_id}: bad diplomacy {diplomacy!r}")
            parsed.append(dict(row))
        result[power_id] = tuple(parsed)
    return result


def expand_sequence(items: Iterable[str]) -> tuple[str, ...]:
    """Expand ``"x*3"`` to three ``x`` and ``"a/b*2"`` to ``a, b, a, b``."""

    result: list[str] = []
    for item in items:
        token, star, count = str(item).partition("*")
        repeat = int(count) if star else 1
        if repeat < 1:
            raise CatalogError(f"invalid repetition in {item!r}")
        result.extend(token.split("/") * repeat)
    return tuple(result)


def _parse_path_keys(text: str) -> tuple[str | int, ...]:
    return tuple(int(part) if part.isdigit() else part for part in text.split("."))


def _compile_generated(catalog: Catalog, event_id: str) -> EventSpec:
    """The major event a generated id stands for (:func:`.generator.raw_event`), compiled
    and validated like the catalog's own."""

    raw = catalog.raw
    if raw is None or not raw.generator:
        raise CatalogError(f"{event_id}: this catalog has no crisis generator")
    try:
        raw_event = generator.raw_event(raw.generator, event_id)
    except generator.GeneratorError as exc:
        raise CatalogError(str(exc)) from exc
    expanded = _expanded(
        raw_event,
        raw.contract,
        raw.regions,
        contract.dynamics_of(raw.powers),
        contract.sensitivity_of(raw.powers),
    )
    event = _parse_event(expanded, 0, CompileContext())
    _validate_major(event, catalog.initial_world())
    return event


def _expanded(
    raw_event: dict[str, Any],
    contract_doc: Any,
    regions_doc: Any,
    dynamics: Mapping[str, float],
    sensitivity: Mapping[str, float],
) -> dict[str, Any]:
    """The event as compiled: with the shared effects contract written into its outcomes
    if it opts in (the kept raw document is unchanged)."""

    try:
        return dict(contract.expand(raw_event, contract_doc, regions_doc, dynamics, sensitivity))
    except (contract.ContractError, KeyError) as exc:
        raise CatalogError(f"event {raw_event.get('id')}: contract: {exc}") from exc


def _parse_event(raw: dict[str, Any], era: int, ctx: CompileContext) -> EventSpec:
    event_id = raw["id"]
    try:
        window = raw.get("window", {})
        earliest = calendar.parse(window["earliest"]) if "earliest" in window else None
        latest = calendar.parse(window["latest"]) if "latest" in window else None
        if earliest is not None and latest is not None and latest < earliest:
            raise CatalogError("window.latest precedes window.earliest")
        condition = compile_condition(raw.get("when"), ctx)
        effects = compile_effects(raw.get("effects"), ctx)
        major = _parse_major(raw, ctx) if raw["kind"] == "major" else None
        decision = _parse_decision(raw, ctx) if raw["kind"] == "decision" else None
    except (ConditionError, ValueError, KeyError, ConfigError) as exc:
        raise CatalogError(f"event {event_id}: {exc}") from exc
    historical = raw.get("historical_date")
    race = None
    if "race" in raw:
        winner, loser = raw["race"]["winner"], raw["race"]["loser"]
        if winner not in POWERS or loser not in POWERS or winner == loser:
            raise CatalogError(f"event {event_id}: race needs two different powers")
        race = (winner, loser)
    return EventSpec(
        id=event_id,
        kind=raw["kind"],
        title=raw["title"],
        era=era,
        earliest=earliest,
        latest=latest,
        condition=condition,
        hazard=float(raw.get("hazard", 1.0)),
        once=bool(raw.get("once", True)),
        priority=int(raw.get("priority", 0)),
        excludes=tuple(raw.get("excludes", ())),
        historical_date=calendar.parse(historical) if historical else None,
        narrative=str(raw.get("narrative", "")),
        provenance=str(raw.get("provenance", "illustrative")),
        effects=effects,
        major=major,
        race=race,
        decision=decision,
    )


def _parse_decision(raw: dict[str, Any], ctx: CompileContext) -> DecisionSpec:
    spec = raw["decision"]
    power = spec["power"]
    if power not in POWERS:
        raise CatalogError(f"decision power {power!r} is not a power")
    options = {
        option: OutcomeSpec(
            compile_effects(item.get("effects"), ctx), str(item.get("narrative", ""))
        )
        for option, item in raw["outcomes"].items()
    }
    if len(options) < 2:
        raise CatalogError("a decision needs at least two options")
    historical = expand_sequence(spec.get("historical", ()))
    unknown = set(historical) - set(options)
    if unknown:
        raise CatalogError(f"historical choices {sorted(unknown)} are not options")
    return DecisionSpec(
        power=power, options=options, tier=int(spec.get("tier", 1)), historical=historical
    )


def _parse_major(raw: dict[str, Any], ctx: CompileContext) -> MajorSpec:
    model = raw["model"]
    family = model["family"]
    if family not in FAMILIES:
        raise CatalogError(f"unknown family {family!r}")
    cadence = model.get("cadence", "instant")
    if cadence == "annual" and family != "attrition":
        raise CatalogError("only attrition events may use the annual cadence")
    if cadence != "instant" and family not in ("attrition", "bargaining"):
        raise CatalogError(f"{family} events must use the instant cadence")
    region = model.get("region")
    if region is not None and region not in REGIONS:
        raise CatalogError(f"unknown region {region!r}")
    bindings = []
    for item in model.get("bindings", []):
        clamp = item.get("clamp")
        bindings.append(
            Binding(
                at=_parse_path_keys(item["at"]),
                base=float(item["base"]) if "base" in item else None,
                terms=tuple(
                    (compile_path(path), float(coef))
                    for path, coef in item.get("terms", {}).items()
                ),
                clamp=(float(clamp[0]), float(clamp[1])) if clamp else None,
            )
        )
    outcomes = {
        label: OutcomeSpec(
            compile_effects(spec.get("effects"), ctx), str(spec.get("narrative", ""))
        )
        for label, spec in raw["outcomes"].items()
    }
    tags = {
        tag: OutcomeSpec(compile_effects(spec.get("effects"), ctx), str(spec.get("narrative", "")))
        for tag, spec in raw.get("tags", {}).items()
    }
    historical = model.get("historical", {})
    optional_roles = {
        role: compile_condition(condition, ctx)
        for role, condition in model.get("optional_roles", {}).items()
    }
    if optional_roles and family != "attrition":
        raise CatalogError("only attrition events may have optional roles")
    effort_costs = str(model.get("effort_costs", "hand"))
    if effort_costs == "derived" and (family != "attrition" or model.get("payoffs") != "derived"):
        raise CatalogError("derived effort costs need an attrition event with derived payoffs")
    identities: dict[str, tuple[str, float]] = {}
    for role, spec in model.get("identities", {}).items():
        if family != "attrition" or model.get("payoffs") != "derived":
            raise CatalogError("identity roles need an attrition event with derived payoffs")
        if role in POWERS or role not in model["config"].get("sponsors", {}):
            raise CatalogError(f"identities.{role}: must be a computer-run sponsor of the war")
        if spec["identity"] not in IDENTITIES:
            raise CatalogError(f"identities.{role}: unknown identity {spec['identity']!r}")
        identities[role] = (str(spec["identity"]), float(spec["stalemate_share"]))
    plans = tuple(
        HistoricalPlan(
            before=calendar.parse(item["before"]) if "before" in item else None,
            choices={role: expand_sequence(opts) for role, opts in item.get("choices", {}).items()},
            chance=expand_sequence(item.get("chance", ())),
        )
        for item in historical.get("instances", ())
    )
    if plans and ("choices" in historical or "chance" in historical):
        raise CatalogError("historical: give either instances or choices and chance")
    return MajorSpec(
        family=family,
        cadence=cadence,
        region=region,
        config=model["config"],
        bindings=tuple(bindings),
        outcomes=outcomes,
        tags=tags,
        historical_choices={
            role: expand_sequence(opts) for role, opts in historical.get("choices", {}).items()
        },
        historical_chance=expand_sequence(historical.get("chance", ())),
        historical_outcome=historical.get("outcome"),
        roles=(),
        tier=int(model.get("tier", 1)),
        payoffs=str(model.get("payoffs", "hand")),
        holdout=bool(model.get("holdout", False)),
        results={label: dict(scores) for label, scores in model.get("results", {}).items()},
        optional_roles=optional_roles,
        plans=plans,
        effort_costs=effort_costs,
        identities=identities,
    )


# ------------------------------------------------------------------- validation
def _validate_references(catalog: Catalog, ctx: CompileContext) -> None:
    missing_events = ctx.event_refs - set(catalog.events)
    if missing_events:
        raise CatalogError(f"references to unknown events: {sorted(missing_events)}")
    missing_leaders = ctx.leaders - set(catalog.leaders)
    if missing_leaders:
        raise CatalogError(f"references to unknown leaders: {sorted(missing_leaders)}")
    for event in catalog.events.values():
        unknown = set(event.excludes) - set(catalog.events)
        if unknown:
            raise CatalogError(f"event {event.id}: excludes unknown events {sorted(unknown)}")


def _symmetric_excludes(catalog: Catalog) -> None:
    extra: dict[str, set[str]] = {eid: set() for eid in catalog.events}
    for event in catalog.events.values():
        for other in event.excludes:
            extra[other].add(event.id)
    for event_id, others in extra.items():
        event = catalog.events[event_id]
        merged = tuple(sorted(set(event.excludes) | others))
        if merged != event.excludes:
            object.__setattr__(event, "excludes", merged)


def _role_actions(event: EventSpec, game: Any) -> dict[str, set[str]]:
    assert event.major is not None
    family = event.major.family
    if family == "attrition":
        params = game.params
        result = {role: set(ATTRITION_LEVELS) for role in game.roles}
        for role in params.approval:
            result[role] |= {"approve", "refuse"}
        for line in params.red_lines:
            result[line.role] |= {"warn", "silent"}
        for item in params.commitments:
            result[item.role] |= {"commit", "hold_off"}
        if params.offers is not None:
            leads = {params.lead_a, params.lead_b, *(role for role, _ in params.lead_on_entry)}
            for role in leads:
                result[role] |= {"no_offer", "accept", "reject", *params.offers.ids()}
            for item in params.offers.consent:
                result[item.role] |= {"ratify", "veto"}
        return result
    if family == "bargaining":
        offers = {game.params.offer_id(i) for i in range(len(game.params.grid))}
        vocab = offers | {"accept", "reject", "commit", "no_commitment"}
        result = {game.params.a_role: vocab, game.params.b_role: set(vocab)}
        for consent in game.params.consent:
            result[consent.id] = {"ratify", "veto"}
        return result
    tree = build_tree(game.new_initial_state(), max_nodes=200_000)
    actions: dict[str, set[str]] = {role: set() for role in game.roles}
    for node in tree.nodes:
        if node.kind == DECISION:
            actions[tree.roles[node.player]].update(node.actions)
    return actions


def _chance_vocabulary(event: EventSpec, game: Any) -> set[str]:
    assert event.major is not None
    family = event.major.family
    if family == "attrition":
        draws = {"a_gains", "b_gains"}
        if game.params.mobility is not None:
            draws.add("hold")
        for holder in game.params.holders:
            draws |= set(game.params.types_of(holder).kinds)
        return draws
    if family == "bargaining":
        talks = {"breakdown", "talks_continue"}
        if game.params.a_types is not None or game.params.b_types is not None:
            talks |= {"strong", "weak"}
        if game.params.contest is not None:
            talks |= {"a_gains", "b_gains"}
        return talks
    if family == "alignment":
        return {*game.outcome_labels}
    tree = build_tree(game.new_initial_state(), max_nodes=200_000)
    vocab: set[str] = set()
    for node in tree.nodes:
        if node.kind == 1:
            vocab.update(node.actions)
    return vocab


def _validate_major(event: EventSpec, world: WorldState) -> None:
    major = event.major
    assert major is not None
    try:
        game = build_family(major.family, major.bound_config(world, drop=()))
        if major.optional_roles:
            reduced = build_family(
                major.family, major.bound_config(world, drop=set(major.optional_roles))
            )
            if set(reduced.outcome_labels) != set(game.outcome_labels):
                raise CatalogError("dropping optional roles changes the outcome labels")
    except (ConfigError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise CatalogError(
            f"event {event.id}: invalid {major.family} configuration: {exc}"
        ) from exc
    unknown_optional = set(major.optional_roles) - set(game.roles)
    if unknown_optional:
        raise CatalogError(f"event {event.id}: optional roles {sorted(unknown_optional)} unknown")
    labels = set(game.outcome_labels)
    missing = labels - set(major.outcomes)
    extra = set(major.outcomes) - labels
    if missing or extra:
        raise CatalogError(
            f"event {event.id}: outcomes must cover exactly {sorted(labels)} "
            f"(missing {sorted(missing)}, unexpected {sorted(extra)})"
        )
    if major.historical_outcome is not None and major.historical_outcome not in labels:
        raise CatalogError(
            f"event {event.id}: historical outcome {major.historical_outcome!r} is not a label"
        )
    actions = _role_actions(event, game)
    vocabulary = _chance_vocabulary(event, game)
    plans = [(major.historical_choices, major.historical_chance)]
    plans += [(plan.choices, plan.chance) for plan in major.plans]
    for choices_by_role, chance in plans:
        for role, choices in choices_by_role.items():
            if role not in actions:
                raise CatalogError(
                    f"event {event.id}: historical choices for unknown role {role!r}"
                )
            unknown = set(choices) - actions[role]
            if unknown:
                raise CatalogError(
                    f"event {event.id}: role {role} has no actions {sorted(unknown)}"
                )
        unknown = set(chance) - vocabulary
        if unknown:
            raise CatalogError(
                f"event {event.id}: unknown historical chance outcomes {sorted(unknown)}"
            )
    for role in game.roles:
        if role in POWERS or role == NEUTRAL:
            continue
        if not role.isupper():
            raise CatalogError(f"event {event.id}: NPC role {role!r} must be upper-case")
    for label, scores in major.results.items():
        if label not in labels or not set(scores) <= set(game.roles) & set(POWERS):
            raise CatalogError(f"event {event.id}: results.{label} names unknown labels or roles")
    object.__setattr__(major, "roles", tuple(game.roles))
    if CHANCE in game.roles:
        raise CatalogError(f"event {event.id}: a role may not be named {CHANCE!r}")


def _validate_derived(catalog: Catalog, event: EventSpec, world: WorldState) -> None:
    major = event.major
    assert major is not None
    held = derive.slots(
        major.family, major.config, major.tags, major.effort_costs, tuple(major.identities)
    )
    for binding in major.bindings:
        for slot in held:
            if binding.at[: len(slot)] == slot:
                path = ".".join(str(k) for k in binding.at)
                raise CatalogError(f"event {event.id}: binding on derived value {path}")
    try:
        build_family(major.family, catalog.instantiate_config(event.id, world))
    except (ConfigError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise CatalogError(
            f"event {event.id}: derived payoffs give an invalid config: {exc}"
        ) from exc
