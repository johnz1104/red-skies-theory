"""Read catalog mappings and check that their events and choices agree."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any

from ...families import FAMILIES, build_family
from ...families.attrition import LEVELS as ATTRITION_LEVELS
from ...families.common import ConfigError
from ...games import CHANCE, build_tree
from ...games.tree import CHANCE_NODE, DECISION
from ..payoffs import derive
from ..payoffs.goals import GoalConfig, GoalError, parse_goals
from ..world import calendar
from ..world.dynamics import POSTURE_KEYS
from ..world.state import (
    IDENTITIES,
    NEUTRAL,
    POSTURES,
    POWER_FIELDS,
    POWERS,
    REGIMES,
    REGIONS,
    STATUSES,
    WorldState,
)
from . import contract
from .conditions import CompileContext, ConditionError, compile_condition
from .effects import LeaderSpec, compile_effects
from .paths import compile_path
from .specs import (
    Binding,
    CatalogError,
    DecisionSpec,
    EventSpec,
    HistoricalPlan,
    MajorSpec,
    OutcomeSpec,
    PowerConfig,
    RegionConfig,
)

if TYPE_CHECKING:
    from .core import Catalog


# ---------------------------------------------------------------------- parsing
def _parse_goals(doc: Any, leaders: Mapping[str, LeaderSpec]) -> GoalConfig:
    """The goal configuration, with its conditions compiled against the leaders."""

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
    """The leaders by id."""

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
    """The powers' opening states, with the dynamics and reward settings."""

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
    """Each posture's budget shares, which sum to 1."""

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
    """The regions' opening states."""

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
    """The historical strategy choices of each power, by year."""

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
    """A dotted config path as keys, with list indices as integers."""

    return tuple(int(part) if part.isdigit() else part for part in text.split("."))


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
    """One event of the catalog."""

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
    """A decision event's choice: its power, options and historical choices."""

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
    """A major event's sub-game.

    Only attrition and bargaining events may run over several quarters, and only
    attrition events may have optional roles, derived effort costs or identity roles.
    """

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
    """Every event, flag and leader that a condition or effect names must exist."""

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
    """Make ``excludes`` symmetric: if A excludes B, B excludes A."""

    extra: dict[str, set[str]] = {eid: set() for eid in catalog.events}
    for event in catalog.events.values():
        for other in event.excludes:
            extra[other].add(event.id)
    for event_id, others in extra.items():
        event = catalog.events[event_id]
        merged = tuple(sorted(set(event.excludes) | others))
        if merged != event.excludes:
            object.__setattr__(event, "excludes", merged)


def _attrition_actions(game: Any) -> dict[str, set[str]]:
    """Every action a sponsor may meet: effort levels, and the blocks' choices."""

    params = game.params
    result = {role: set(ATTRITION_LEVELS) for role in game.roles}
    for role in params.approval:
        result[role] |= {"approve", "refuse"}
    for line in params.red_lines:
        result[line.role] |= {"warn", "silent"}
    for item in params.commitments:
        result[item.role] |= {"commit", "hold_off"}
    if params.offers is not None:
        leads = {params.lead_a, params.lead_b, *params.lead_on_entry}
        for role in leads:
            result[role] |= {"no_offer", "accept", "reject", *params.offers.ids()}
        for item in params.offers.consent:
            result[item.role] |= {"ratify", "veto"}
    return result


def _bargaining_actions(game: Any) -> dict[str, set[str]]:
    """Offers, responses and commitments for the principals; ratification for consent roles."""

    offers = {game.params.offer_id(i) for i in range(len(game.params.grid))}
    vocab = offers | {"accept", "reject", "commit", "no_commitment"}
    result = {game.params.a_role: vocab, game.params.b_role: set(vocab)}
    for consent in game.params.consent:
        result[consent.id] = {"ratify", "veto"}
    return result


def _tree_actions(game: Any) -> dict[str, set[str]]:
    """The actions at each role's decision nodes, read off the game tree."""

    tree = build_tree(game.new_initial_state(), max_nodes=200_000)
    actions: dict[str, set[str]] = {role: set() for role in game.roles}
    for node in tree.nodes:
        if node.kind == DECISION:
            actions[tree.roles[node.player]].update(node.actions)
    return actions


ROLE_ACTIONS: dict[str, Callable[[Any], dict[str, set[str]]]] = {
    "attrition": _attrition_actions,
    "bargaining": _bargaining_actions,
}


def _role_actions(event: EventSpec, game: Any) -> dict[str, set[str]]:
    """Each role's action vocabulary in *game* (historical choices are checked against it)."""

    assert event.major is not None
    return ROLE_ACTIONS.get(event.major.family, _tree_actions)(game)


def _attrition_chance(game: Any) -> set[str]:
    """Battle results, and every type holder's kinds."""

    draws = {"a_gains", "b_gains"}
    if game.params.mobility is not None:
        draws.add("hold")
    for holder in game.params.holders:
        draws |= set(game.params.types_of(holder).kinds)
    return draws


def _bargaining_chance(game: Any) -> set[str]:
    """Breakdown draws, and type and contest draws where the game has them."""

    talks = {"breakdown", "talks_continue"}
    if game.params.a_types is not None or game.params.b_types is not None:
        talks |= {"strong", "weak"}
    if game.params.contest is not None:
        talks |= {"a_gains", "b_gains"}
    return talks


def _alignment_chance(game: Any) -> set[str]:
    """A chance client's choices."""

    return {*game.outcome_labels}


def _tree_chance(game: Any) -> set[str]:
    """The outcomes at the chance nodes of the game tree."""

    tree = build_tree(game.new_initial_state(), max_nodes=200_000)
    vocab: set[str] = set()
    for node in tree.nodes:
        if node.kind == CHANCE_NODE:
            vocab.update(node.actions)
    return vocab


CHANCE_VOCABULARY: dict[str, Callable[[Any], set[str]]] = {
    "attrition": _attrition_chance,
    "bargaining": _bargaining_chance,
    "alignment": _alignment_chance,
}


def _chance_vocabulary(event: EventSpec, game: Any) -> set[str]:
    """The chance outcomes of *game* (historical chance is checked against them)."""

    assert event.major is not None
    return CHANCE_VOCABULARY.get(event.major.family, _tree_chance)(game)


def _validate_major(event: EventSpec, world: WorldState) -> None:
    """Build the event's game at *world* and check its data against it: an effect for
    every outcome label, and historical choices and chance draws the game can produce.
    """

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
    """A derived-payoff event must derive at the opening world, and no binding may
    target a derived value.
    """

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
