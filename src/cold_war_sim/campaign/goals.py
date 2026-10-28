"""Goal functions: what each learning identity values (``docs/GOALS.md``).

A goal score is a weighted sum of seven components, each computed from the
world state alone:

``security``     own military strength against threat-weighted others, plus the
                 net hold of the identity's buffer regions (its approaches)
``influence``    influence where the identity cares, net of rivals' influence
``economy``      ln(GDP / GDP in 1945) / ln(scale)
``regime``       domestic stability of the ruling order; a power governing only part of
                 its country (an insurgency) counts the share it governs
``empire``       colonies still held, each by a hold value in [-1, 1]
``autonomy``     own tested bomb and freedom from a superpower's alliance
``credibility``  reputation for keeping commitments

The threat weight of another power is ``(1 - relation) / 2``. A friendly power
(relation > 0) also adds ``relation * band(x)``, where ``x`` is its strength over
yours and the band (GOALS.md §14) is about 0 below a floor and positive above it.
Where a relationship is documented as one in which the stronger partner resents a
rising junior (Stalin and Mao), the band also has a ceiling and a cost beyond it.
Weights come from
``data/goals.yaml`` and change only with leader, date or regime. The reward each
quarter is the *current* weights times the change in the components
(:func:`reward`), so a weight switch never creates a reward jump by itself.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .conditions import CompileContext, Predicate, compile_condition
from .state import IDENTITIES, POWERS, REGIONS, WorldState, seat

if TYPE_CHECKING:
    from .catalog import Catalog

COMPONENTS = ("security", "influence", "economy", "regime", "empire", "autonomy", "credibility")
NUCLEAR_MODELS = ("linear", "weibull")
BOMB_THRESHOLD = 1.0  # tech_nuclear at the first tested weapon


class GoalError(ValueError):
    """Raised when goals.yaml fails validation."""


@dataclass(frozen=True)
class NuclearModel:
    model: str
    weight: float = 0.6
    max: float = 0.0
    n_star: float = 1.0
    k: float = 1.0

    def value(self, nuclear: float) -> float:
        if self.model == "linear":
            return self.weight * nuclear
        return self.max * (1.0 - math.exp(-((max(0.0, nuclear) / self.n_star) ** self.k)))


@dataclass(frozen=True)
class Band:
    floor: float
    ceiling: float | None  # None: no ceiling


@dataclass(frozen=True)
class AlliesConfig:
    height: float
    cost: float
    smoothness: float
    default: Band
    bands: Mapping[str, Mapping[str, Band]]  # viewer identity -> other seat -> band

    def band(self, viewer: str, other: str) -> Band:
        return self.bands.get(viewer, {}).get(other, self.default)

    def value(self, viewer: str, other: str, ratio: float) -> float:
        band = self.band(viewer, other)
        value = _logistic(ratio - band.floor, self.smoothness * max(band.floor, 0.05))
        if band.ceiling is not None:
            width = self.smoothness * max(band.ceiling, 0.05)
            value -= (1.0 + self.cost) * _logistic(ratio - band.ceiling, width)
        return self.height * value


def _logistic(z: float, width: float) -> float:
    u = max(-60.0, min(60.0, z / width))
    return 1.0 / (1.0 + math.exp(-u))


@dataclass(frozen=True)
class WeightRow:
    label: str
    when: Predicate | None
    weights: Mapping[str, float]


@dataclass(frozen=True)
class GoalConfig:
    kappa: float
    economy_scale: float
    nuclear: NuclearModel
    interest: Mapping[str, Mapping[str, float]]
    rivalry: Mapping[str, float]
    subordinate: Mapping[str, Predicate]
    rows: Mapping[str, tuple[WeightRow, ...]]
    allies: AlliesConfig | None = None
    buffer_weight: float = 0.0  # security per unit of net hold of the buffer regions
    buffers: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    lookahead_quarters: int = 0  # derived payoffs: follow-on events counted this far ahead
    lookahead_discount: float = 1.0  # per quarter
    # power -> (home region, condition): while the condition holds the power governs only
    # part of its country, and its regime counts its influence share there (E40)
    partial_regime: Mapping[str, tuple[str, Predicate]] = field(default_factory=dict)
    standing_k: float = 32.0  # the standing scoreboard (never in a reward)
    standing_start: float = 1500.0
    # risk attitudes (E48): prospect theory's value function on derived event values,
    # (alpha, beta, lambda); per leader where documented. (1, 1, 1) is risk-neutral
    risk: tuple[float, float, float] = (1.0, 1.0, 1.0)
    risk_by_leader: Mapping[str, tuple[float, float, float]] = field(default_factory=dict)

    def risk_of(self, leader: str | None) -> tuple[float, float, float]:
        return self.risk_by_leader.get(leader or "", self.risk)

    def __deepcopy__(self, memo: dict[int, object]) -> GoalConfig:
        return self


# ------------------------------------------------------------------ components
def military_strength(world: WorldState, power: str, config: GoalConfig) -> float:
    state = world.powers[power]
    return state.conventional + config.nuclear.value(state.nuclear)


def threat_weight(world: WorldState, power: str, other: str) -> float:
    return (1.0 - world.powers[power].relations.get(other, 0.0)) / 2.0


def components(
    world: WorldState, identity_id: str, catalog: Catalog, config: GoalConfig | None = None
) -> dict[str, float]:
    """The seven components; *config* overrides ``catalog.goals`` (used by sweeps)."""

    config = config or catalog.goals
    power = seat(identity_id)
    state = world.powers[power]
    others = [q for q in POWERS if q != power]

    own = military_strength(world, power, config)
    allies = config.allies

    def friendly(q: str) -> float:
        """relation * band value of q for this identity (0 if not friendly)."""
        relation = world.powers[power].relations.get(q, 0.0)
        if allies is None or relation <= 0.0 or own <= 0.0:
            return 0.0
        return relation * allies.value(identity_id, q, military_strength(world, q, config) / own)

    friends = {q: friendly(q) for q in others if world.powers[q].status != "eliminated"}
    threats = sum(
        threat_weight(world, power, q) * military_strength(world, q, config)
        for q in others
        if world.powers[q].status == "active"
    )
    allied = sum(f * own for q, f in friends.items() if world.powers[q].status == "active")
    support, burden = max(allied, 0.0), max(-allied, 0.0)
    total = own + support + threats + burden
    security = (own + support) / total if total > 0 else 0.5
    # buffers: a power's approaches absorb threats, so holding them is security too
    # (GOALS.md §16); a rival's presence there counts in full, not scaled by rivalry
    buffer_regions = config.buffers.get(identity_id, ())
    if buffer_regions and config.buffer_weight:
        held = weights = 0.0
        for region_id in buffer_regions:
            region = world.regions[region_id]
            rivals = sum(threat_weight(world, power, q) * region.influence[q] for q in others)
            helped = sum(f * region.influence[q] for q, f in friends.items())
            held += region.importance * (region.influence[power] + helped - rivals)
            weights += region.importance
        security += config.buffer_weight * held / weights

    interest = config.interest.get(identity_id, {})
    rivalry = config.rivalry[identity_id]
    weight_sum = 0.0
    influence = 0.0
    for region_id in REGIONS:
        region = world.regions[region_id]
        weight = region.importance * interest.get(region_id, 1.0)
        rivals = sum(threat_weight(world, power, q) * region.influence[q] for q in others)
        helped = sum(f * region.influence[q] for q, f in friends.items())
        influence += weight * (region.influence[power] + helped - rivalry * rivals)
        weight_sum += weight
    influence = influence / weight_sum if weight_sum > 0 else 0.0

    base_economy = float(catalog.powers[power].initial.get("economy", 1.0))
    economy = math.log(max(state.economy, 1e-9) / base_economy) / math.log(config.economy_scale)

    held_1945 = sum(r.importance for r in catalog.regions.values() if r.colonial_owner == power)
    empire = 0.0
    if held_1945 > 0:
        for region in world.regions.values():
            if region.colonial_owner == power:
                hold = region.influence[power] - (1.0 - region.stability)
                empire += region.importance * hold
        empire /= held_1945

    subordinate = config.subordinate.get(identity_id)
    has_bomb = state.tech_nuclear >= BOMB_THRESHOLD
    independent = subordinate is None or not subordinate(world)
    autonomy = 0.5 * float(has_bomb) + 0.5 * float(independent)

    regime = state.stability
    partial = config.partial_regime.get(power)
    if partial is not None and partial[1](world):
        # a power governing only part of its country (an insurgency, or a regime holding
        # part of a partitioned country) counts the stability of the share it governs: it
        # has less to lose and the country to win (user decision 2026-10-14; E40)
        regime *= world.regions[partial[0]].influence[power]

    return {
        "security": security,
        "influence": influence,
        "economy": economy,
        "regime": regime,
        "empire": empire,
        "autonomy": autonomy,
        "credibility": state.credibility,
    }


def row(
    world: WorldState, identity_id: str, catalog: Catalog, config: GoalConfig | None = None
) -> WeightRow:
    for candidate in (config or catalog.goals).rows[identity_id]:
        if candidate.when is None or candidate.when(world):
            return candidate
    raise AssertionError("the last weight row has no condition")  # enforced at load


def combine(parts: Mapping[str, float], weights: Mapping[str, float]) -> float:
    return sum(weight * parts[name] for name, weight in weights.items())


def score(
    world: WorldState, identity_id: str, catalog: Catalog, config: GoalConfig | None = None
) -> float:
    parts = components(world, identity_id, catalog, config)
    return combine(parts, row(world, identity_id, catalog, config).weights)


def reward(
    previous: Mapping[str, float], current: Mapping[str, float], weights: Mapping[str, float]
) -> float:
    """Current weights times the change in components."""

    return sum(weight * (current[name] - previous[name]) for name, weight in weights.items())


# --------------------------------------------------------------------- parsing
def _mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise GoalError(f"{name} must be a mapping")
    return value


def _only(name: str, value: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise GoalError(f"{name}: unknown keys {sorted(unknown)}")


def _number(name: str, value: Any, *, low: float = 0.0, strict: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise GoalError(f"{name} must be a number") from exc
    if not math.isfinite(number) or number < low or (strict and number == low):
        raise GoalError(f"{name} must be {'above' if strict else 'at least'} {low}")
    return number


def _identities(name: str, value: Mapping[str, Any]) -> None:
    unknown = set(value) - set(IDENTITIES)
    if unknown:
        raise GoalError(f"{name}: unknown identities {sorted(unknown)}")


def _nuclear(raw: Mapping[str, Any]) -> NuclearModel:
    model = raw.get("model")
    if model == "linear":
        _only("security.nuclear", raw, {"model", "weight"})
        return NuclearModel("linear", weight=_number("nuclear.weight", raw.get("weight", 0.6)))
    if model == "weibull":
        _only("security.nuclear", raw, {"model", "max", "n_star", "k"})
        return NuclearModel(
            "weibull",
            max=_number("nuclear.max", raw.get("max")),
            n_star=_number("nuclear.n_star", raw.get("n_star"), strict=True),
            k=_number("nuclear.k", raw.get("k"), strict=True),
        )
    raise GoalError(f"security.nuclear.model must be one of {NUCLEAR_MODELS}")


def _rows(raw: Any, identity_id: str, ctx: CompileContext) -> tuple[WeightRow, ...]:
    if not isinstance(raw, list) or not raw:
        raise GoalError(f"weights.{identity_id} must be a non-empty list")
    rows = []
    for index, item in enumerate(raw):
        name = f"weights.{identity_id}[{index}]"
        item = _mapping(name, item)
        _only(name, item, {"label", "when", "weights"})
        is_last = index == len(raw) - 1
        if is_last and "when" in item:
            raise GoalError(f"{name}: the last row is the default and takes no 'when'")
        if not is_last and "when" not in item:
            raise GoalError(f"{name}: only the last row may omit 'when'")
        weights = _mapping(f"{name}.weights", item.get("weights"))
        unknown = set(weights) - set(COMPONENTS)
        if unknown:
            raise GoalError(f"{name}: unknown components {sorted(unknown)}")
        parsed = {k: _number(f"{name}.weights.{k}", v) for k, v in weights.items()}
        if abs(sum(parsed.values()) - 1.0) > 1e-6:
            raise GoalError(f"{name}: weights sum to {sum(parsed.values()):.4f}, not 1")
        when = compile_condition(item["when"], ctx) if "when" in item else None
        rows.append(WeightRow(str(item.get("label", f"row {index}")), when, parsed))
    return tuple(rows)


def _band(name: str, raw: Any) -> Band:
    raw = _mapping(name, raw)
    _only(name, raw, {"floor", "ceiling"})
    floor = _number(f"{name}.floor", raw.get("floor", 0.0))
    ceiling = raw.get("ceiling")
    if ceiling is None:
        return Band(floor, None)
    ceiling = _number(f"{name}.ceiling", ceiling)
    if ceiling <= floor:
        raise GoalError(f"{name}: the ceiling must lie above the floor")
    return Band(floor, ceiling)


def _allies(raw: Any) -> AlliesConfig:
    raw = _mapping("components.allies", raw)
    _only("components.allies", raw, {"height", "cost", "smoothness", "default", "bands"})
    bands_raw = _mapping("allies.bands", raw.get("bands", {}))
    _identities("allies.bands", bands_raw)
    bands: dict[str, dict[str, Band]] = {}
    for viewer, others in bands_raw.items():
        others = _mapping(f"allies.bands.{viewer}", others)
        unknown = set(others) - set(POWERS)
        if unknown or seat(viewer) in others:
            raise GoalError(
                f"allies.bands.{viewer}: bad powers {sorted(unknown) or [seat(viewer)]}"
            )
        bands[viewer] = {q: _band(f"allies.bands.{viewer}.{q}", b) for q, b in others.items()}
    return AlliesConfig(
        height=_number("allies.height", raw.get("height", 0.5)),
        cost=_number("allies.cost", raw.get("cost", 0.5)),
        smoothness=_number("allies.smoothness", raw.get("smoothness", 0.2), strict=True),
        default=_band("allies.default", raw.get("default", {"floor": 0.1})),
        bands=bands,
    )


def parse_goals(doc: Any, ctx: CompileContext) -> GoalConfig:
    """Parse goals.yaml. Conditions record their references in *ctx*."""

    doc = _mapping("goals.yaml", doc)
    _only("goals.yaml", doc, {"kappa", "lookahead", "components", "weights", "standing", "risk"})
    lookahead = _mapping("lookahead", doc.get("lookahead", {}))
    _only("lookahead", lookahead, {"quarters", "discount"})
    quarters = lookahead.get("quarters", 0)
    if not isinstance(quarters, int) or quarters < 0:
        raise GoalError("lookahead.quarters must be a non-negative integer")
    comps = _mapping("components", doc.get("components", {}))
    _only("components", comps, {"economy", "security", "influence", "autonomy", "allies", "regime"})
    economy = _mapping("components.economy", comps.get("economy", {}))
    _only("components.economy", economy, {"scale"})
    security = _mapping("components.security", comps.get("security", {}))
    _only("components.security", security, {"nuclear", "buffer"})
    buffer = _mapping("security.buffer", security.get("buffer", {}))
    _only("security.buffer", buffer, {"weight", "regions"})
    buffer_raw = _mapping("security.buffer.regions", buffer.get("regions", {}))
    _identities("security.buffer.regions", buffer_raw)
    buffers: dict[str, tuple[str, ...]] = {}
    for identity_id, regions in buffer_raw.items():
        if not isinstance(regions, list) or not set(regions) <= set(REGIONS):
            raise GoalError(f"security.buffer.regions.{identity_id}: a list of known regions")
        buffers[identity_id] = tuple(regions)
    influence = _mapping("components.influence", comps.get("influence", {}))
    _only("components.influence", influence, {"interest", "rivalry"})
    autonomy = _mapping("components.autonomy", comps.get("autonomy", {}))
    _only("components.autonomy", autonomy, {"subordinate"})
    regime_raw = _mapping("components.regime", comps.get("regime", {}))
    _only("components.regime", regime_raw, {"partial"})
    partial_regime: dict[str, tuple[str, Predicate]] = {}
    for power_id, spec in _mapping("regime.partial", regime_raw.get("partial", {})).items():
        spec = _mapping(f"regime.partial.{power_id}", spec)
        _only(f"regime.partial.{power_id}", spec, {"region", "when"})
        region_id = spec.get("region")
        if power_id not in POWERS or region_id not in REGIONS or "when" not in spec:
            raise GoalError(f"regime.partial.{power_id}: a power, a known region and a condition")
        partial_regime[power_id] = (str(region_id), compile_condition(spec["when"], ctx))

    interest_raw = _mapping("influence.interest", influence.get("interest", {}))
    _identities("influence.interest", interest_raw)
    interest: dict[str, dict[str, float]] = {}
    for identity_id, regions in interest_raw.items():
        regions = _mapping(f"influence.interest.{identity_id}", regions)
        unknown = set(regions) - set(REGIONS)
        if unknown:
            raise GoalError(f"influence.interest.{identity_id}: unknown regions {sorted(unknown)}")
        interest[identity_id] = {
            r: _number(f"interest.{identity_id}.{r}", m) for r, m in regions.items()
        }
    rivalry_raw = _mapping("influence.rivalry", influence.get("rivalry", {}))
    _identities("influence.rivalry", rivalry_raw)
    missing = set(IDENTITIES) - set(rivalry_raw)
    if missing:
        raise GoalError(f"influence.rivalry: missing {sorted(missing)}")
    rivalry = {i: _number(f"rivalry.{i}", v) for i, v in rivalry_raw.items()}

    allies = _allies(comps["allies"]) if "allies" in comps else None

    subordinate_raw = _mapping("autonomy.subordinate", autonomy.get("subordinate", {}))
    _identities("autonomy.subordinate", subordinate_raw)
    subordinate = {i: compile_condition(spec, ctx) for i, spec in subordinate_raw.items()}

    standing = _mapping("standing", doc.get("standing", {}))
    _only("standing", standing, {"k", "start"})
    risk_raw = _mapping("risk", doc.get("risk", {}))
    _only("risk", risk_raw, {"alpha", "beta", "lambda", "leaders"})

    def _risk(name: str, spec: Mapping[str, Any], default: tuple[float, float, float]) -> Any:
        values = tuple(
            _number(f"{name}.{key}", spec.get(key, d), strict=True)
            for key, d in zip(("alpha", "beta", "lambda"), default, strict=True)
        )
        if values[0] > 1.0 or values[1] > 1.0:
            raise GoalError(f"{name}: alpha and beta must lie in (0, 1]")
        return values

    risk = _risk("risk", risk_raw, (1.0, 1.0, 1.0))
    risk_by_leader = {
        str(leader): _risk(f"risk.leaders.{leader}", _mapping(f"risk.leaders.{leader}", spec), risk)
        for leader, spec in _mapping("risk.leaders", risk_raw.get("leaders", {})).items()
    }

    weights_raw = _mapping("weights", doc.get("weights"))
    if set(weights_raw) != set(IDENTITIES):
        raise GoalError(f"weights must define exactly {IDENTITIES}")
    rows = {i: _rows(weights_raw[i], i, ctx) for i in IDENTITIES}

    return GoalConfig(
        kappa=_number("kappa", doc.get("kappa", 1.0), strict=True),
        economy_scale=_number("economy.scale", economy.get("scale", 4.0), low=1.0, strict=True),
        nuclear=_nuclear(
            _mapping("security.nuclear", security.get("nuclear", {"model": "linear"}))
        ),
        interest=interest,
        rivalry=rivalry,
        subordinate=subordinate,
        rows=rows,
        allies=allies,
        buffer_weight=_number("security.buffer.weight", buffer.get("weight", 0.0)),
        buffers=buffers,
        partial_regime=partial_regime,
        lookahead_quarters=quarters,
        lookahead_discount=_number("lookahead.discount", lookahead.get("discount", 1.0)),
        standing_k=_number("standing.k", standing.get("k", 32.0)),
        standing_start=_number("standing.start", standing.get("start", 1500.0), low=-math.inf),
        risk=risk,
        risk_by_leader=risk_by_leader,
    )


__all__ = [
    "COMPONENTS",
    "AlliesConfig",
    "Band",
    "GoalConfig",
    "GoalError",
    "WeightRow",
    "combine",
    "components",
    "military_strength",
    "parse_goals",
    "reward",
    "row",
    "score",
    "threat_weight",
]
