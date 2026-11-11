"""Load the campaign catalog and validate its events, choices and world references.

Loading checks the schema, paths, references and outcome coverage. It also
builds each major event against the opening world and checks historical choices
against the actions available to each role.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from importlib import resources
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from ...records import Record
from ..engine.reproducibility import stable_hash
from ..events import generator
from ..payoffs import derive
from ..payoffs.goals import GoalConfig
from ..world import calendar
from ..world.state import (
    POWERS,
    REGIONS,
    PowerState,
    RegionState,
    WorldState,
    normalize_influence,
    update_controller,
)
from . import contract
from .conditions import CompileContext
from .effects import LeaderSpec, apply_leader
from .parsing import (
    _expanded,
    _parse_event,
    _parse_goals,
    _parse_leaders,
    _parse_powers,
    _parse_regions,
    _parse_strategy,
    _symmetric_excludes,
    _validate_derived,
    _validate_major,
    _validate_references,
    expand_sequence,
)
from .specs import (
    CADENCES,
    CONTRACT_INPUTS,
    DOCUMENTS,
    KINDS,
    Binding,
    CatalogError,
    DecisionSpec,
    EventSpec,
    HistoricalPlan,
    MajorSpec,
    OutcomeSpec,
    PowerConfig,
    RawDocuments,
    RegionConfig,
    drop_roles,
    get_number,
    set_number,
    split_address,
)


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
        """The event *key*, or *default* if it is unknown or does not compile."""

        try:
            return self[key]
        except (KeyError, CatalogError):
            return default


class Catalog(Record):
    """The compiled campaign data.

    A catalog is immutable and shared: copies of an engine state hold the same object.
    """

    def __init__(
        self,
        events: dict[str, EventSpec],
        leaders: dict[str, LeaderSpec],
        powers: dict[str, PowerConfig],
        regions: dict[str, RegionConfig],
        relations: dict[tuple[str, str], float],
        initial_flags: frozenset[str],
        initial_tension: float,
        dynamics: Mapping[str, Any],
        rewards: Mapping[str, Any],
        historical_strategy: Mapping[str, tuple[Mapping[str, Any], ...]],
        goals: GoalConfig,
        era_titles: dict[int, str] | None = None,
        cleared_flags: frozenset[str] = frozenset(),
        postures: Mapping[str, tuple[float, float, float, float]] | None = None,
        raw: RawDocuments | None = None,
        fingerprint: str = "",
    ) -> None:
        self.events = events
        self.leaders = leaders
        self.powers = powers
        self.regions = regions
        self.relations = relations
        self.initial_flags = initial_flags
        self.initial_tension = initial_tension
        self.dynamics = dynamics
        self.rewards = rewards
        self.historical_strategy = historical_strategy
        self.goals = goals
        self.era_titles: dict[int, str] = {} if era_titles is None else era_titles
        self.cleared_flags = cleared_flags  # flags some effect in the catalog unsets
        self.postures: Mapping[str, tuple[float, float, float, float]] = (
            {} if postures is None else postures
        )
        self.raw = raw  # the source documents, for variants
        self.fingerprint = (
            fingerprint  # "" for the catalog as loaded; a digest of a variant's changes
        )
        self._digest: str | None = None  # the cached digest

    def __deepcopy__(self, memo: dict[int, object]) -> Catalog:
        return self

    def replace(self, **changes: Any) -> Catalog:
        """A copy with the named attributes changed; its digest is computed afresh."""

        new = super().replace(**changes)
        new._digest = None
        return new

    @property
    def digest(self) -> str:
        """A digest of the source documents: state captured under another catalog (a
        scenario entry, whose event was bound when it started) is stale under this one."""

        if self._digest is None:
            raw = self.raw
            if raw is None:
                self._digest = self.fingerprint
            else:
                docs = {k: v for k, v in raw.to_dict().items() if k != "era_titles"}
                self._digest = stable_hash(docs)
        return self._digest

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
        new = self.replace(events=table)
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
        new.raw = raw.replace(
            events=tuple(events),
            **{name: patched[name] for name in DOCUMENTS if name in patched},
        )
        new.fingerprint = fingerprint if fingerprint is not None else stable_hash(dict(values))
        return new

    @property
    def ordered(self) -> tuple[EventSpec, ...]:
        """The events in the order the director scans them: priority first, then id."""

        return tuple(sorted(self.events.values(), key=lambda e: (-e.priority, e.id)))

    @property
    def major_ids(self) -> tuple[str, ...]:
        """The ids of the major events, sorted."""

        return tuple(sorted(e.id for e in self.events.values() if e.kind == "major"))

    @property
    def observed_flags(self) -> tuple[str, ...]:
        """Every flag the catalog can set, sorted: the observation's flag block."""

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
        """The number of events of each kind."""

        counts = dict.fromkeys(KINDS, 0)
        for event in self.events.values():
            counts[event.kind] += 1
        return counts

    def initial_world(self) -> WorldState:
        """The world at the start of the campaign."""

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
    """The directory of the packaged campaign data."""

    return Path(str(resources.files("cold_war_sim.campaign") / "data"))


def _load_yaml(path: Path) -> Any:
    """The parsed YAML document at *path*."""

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _event_schema(root: Path) -> dict[str, Any]:
    """The JSON schema of the event files."""

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


__all__ = [
    "CADENCES",
    "CONTRACT_FILE",
    "CONTRACT_INPUTS",
    "DOCUMENTS",
    "GENERATOR_FILE",
    "KINDS",
    "Binding",
    "Catalog",
    "CatalogError",
    "DecisionSpec",
    "EventSpec",
    "EventTable",
    "HistoricalPlan",
    "MajorSpec",
    "OutcomeSpec",
    "PowerConfig",
    "RawDocuments",
    "RegionConfig",
    "compile_catalog",
    "data_root",
    "drop_roles",
    "expand_sequence",
    "get_number",
    "load_catalog",
    "read_documents",
    "set_number",
    "split_address",
]
