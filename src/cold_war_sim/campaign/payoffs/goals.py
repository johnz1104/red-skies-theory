"""Goal functions: what each learning identity values.

A goal score is a weighted sum of seven components of the world state (``COMPONENTS``;
:func:`components` defines each). Weights come from ``data/goals.yaml`` and change only
with leader, date or regime. The reward each quarter is the current weights times the
change in the components (:func:`reward`), so a change of weights never creates a
reward jump by itself.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from ...records import Record
from ..catalog.conditions import CompileContext, Predicate, compile_condition
from ..world.state import IDENTITIES, POWERS, REGIONS, WorldState, seat

if TYPE_CHECKING:
    from ..catalog.core import Catalog

COMPONENTS = ("security", "influence", "economy", "regime", "empire", "autonomy", "credibility")
NUCLEAR_MODELS = ("linear", "weibull")
BOMB_THRESHOLD = 1.0  # tech_nuclear at the first tested weapon


class GoalError(ValueError):
    """Raised when goals.yaml fails validation."""


class NuclearModel(Record):
    """The contribution of a nuclear arsenal ``n`` to military strength.

    ``linear``: ``weight x n``. ``weibull``: ``max x (1 - exp(-(n / n_star)^k))``, which saturates.
    """

    def __init__(
        self,
        model: str,
        weight: float = 0.6,
        max: float = 0.0,
        n_star: float = 1.0,
        k: float = 1.0,
    ) -> None:
        self.model = model
        self.weight = weight
        self.max = max
        self.n_star = n_star
        self.k = k

    def value(self, nuclear: float) -> float:
        """The strength an arsenal of size *nuclear* adds."""

        if self.model == "linear":
            return self.weight * nuclear
        return self.max * (1.0 - math.exp(-((max(0.0, nuclear) / self.n_star) ** self.k)))


class Band(Record):
    """The range of a friendly power's relative strength in which it is an asset: above
    ``floor`` and, where one is set, below ``ceiling``.
    """

    def __init__(
        self,
        floor: float,
        ceiling: float | None,
    ) -> None:
        self.floor = floor
        self.ceiling = ceiling  # None: no ceiling


class AlliesConfig(Record):
    """How friendly powers count toward security and influence.

    A friend whose strength over the viewer's is ``x`` adds ``relation x value(x)``:
    about 0 below its band's floor, ``height`` above it and, past a ceiling, a net cost
    (the stronger partner resents a rising junior).
    """

    def __init__(
        self,
        height: float,
        cost: float,
        smoothness: float,
        default: Band,
        bands: Mapping[str, Mapping[str, Band]],
    ) -> None:
        self.height = height
        self.cost = cost
        self.smoothness = smoothness
        self.default = default
        self.bands = bands  # viewer identity -> other seat -> band

    def band(self, viewer: str, other: str) -> Band:
        """The band *viewer* applies to *other* (the default unless a pair is listed)."""

        return self.bands.get(viewer, {}).get(other, self.default)

    def value(self, viewer: str, other: str, ratio: float) -> float:
        """``height x [L(x - floor) - (1 + cost) L(x - ceiling)]`` at ratio ``x``, with ``L`` a
        logistic of width ``smoothness x`` the threshold.
        """

        band = self.band(viewer, other)
        value = _logistic(ratio - band.floor, self.smoothness * max(band.floor, 0.05))
        if band.ceiling is not None:
            width = self.smoothness * max(band.ceiling, 0.05)
            value -= (1.0 + self.cost) * _logistic(ratio - band.ceiling, width)
        return self.height * value


def _logistic(z: float, width: float) -> float:
    """The logistic function of ``z / width`` (with the argument clipped to +/-60)."""

    u = max(-60.0, min(60.0, z / width))
    return 1.0 / (1.0 + math.exp(-u))


class WeightRow(Record):
    """One row of an identity's weights, in force while its condition ``when`` holds
    (the last row has none and is the default).
    """

    def __init__(
        self,
        label: str,
        when: Predicate | None,
        weights: Mapping[str, float],
    ) -> None:
        self.label = label
        self.when = when
        self.weights = weights


class GoalConfig(Record):
    """The goal functions' parameters (``data/goals.yaml``): the components' settings,
    each identity's weight rows, the lookahead for derived payoffs, the standing
    scoreboard's constants and the risk attitudes.
    """

    def __init__(
        self,
        kappa: float,
        economy_scale: float,
        nuclear: NuclearModel,
        interest: Mapping[str, Mapping[str, float]],
        rivalry: Mapping[str, float],
        subordinate: Mapping[str, Predicate],
        rows: Mapping[str, tuple[WeightRow, ...]],
        allies: AlliesConfig | None = None,
        buffer_weight: float = 0.0,
        buffers: Mapping[str, tuple[str, ...]] | None = None,
        lookahead_quarters: int = 0,
        lookahead_discount: float = 1.0,
        partial_regime: Mapping[str, tuple[str, Predicate]] | None = None,
        standing_k: float = 32.0,
        standing_start: float = 1500.0,
        risk: tuple[float, float, float] = (1.0, 1.0, 1.0),
        risk_by_leader: Mapping[str, tuple[float, float, float]] | None = None,
    ) -> None:
        self.kappa = kappa
        self.economy_scale = economy_scale
        self.nuclear = nuclear
        self.interest = interest
        self.rivalry = rivalry
        self.subordinate = subordinate
        self.rows = rows
        self.allies = allies
        self.buffer_weight = buffer_weight  # security per unit of net hold of the buffer regions
        self.buffers: Mapping[str, tuple[str, ...]] = {} if buffers is None else buffers
        self.lookahead_quarters = (
            lookahead_quarters  # derived payoffs: follow-on events counted this far ahead
        )
        self.lookahead_discount = lookahead_discount  # per quarter
        # power -> (home region, condition): while the condition holds the power governs only
        # part of its country, and its regime counts its influence share there
        self.partial_regime: Mapping[str, tuple[str, Predicate]] = (
            {} if partial_regime is None else partial_regime
        )
        self.standing_k = standing_k  # the standing scoreboard (never in a reward)
        self.standing_start = standing_start
        # risk attitudes: prospect theory's value function on derived event values, (alpha, beta,
        # lambda); per leader where documented. (1, 1, 1) is risk-neutral
        self.risk = risk
        self.risk_by_leader: Mapping[str, tuple[float, float, float]] = (
            {} if risk_by_leader is None else risk_by_leader
        )

    def risk_of(self, leader: str | None) -> tuple[float, float, float]:
        """The prospect-theory attitude (alpha, beta, lambda) of *leader*."""

        return self.risk_by_leader.get(leader or "", self.risk)

    def __deepcopy__(self, memo: dict[int, object]) -> GoalConfig:
        return self


# ------------------------------------------------------------------ components
def military_strength(world: WorldState, power: str, config: GoalConfig) -> float:
    """Conventional forces plus the nuclear arsenal's contribution."""

    state = world.powers[power]
    return state.conventional + config.nuclear.value(state.nuclear)


def threat_weight(world: WorldState, power: str, other: str) -> float:
    """How much *other* threatens *power*: ``(1 - relation) / 2``."""

    return (1.0 - world.powers[power].relations.get(other, 0.0)) / 2.0


class _Viewpoint:
    """One identity's view of the world, shared by its components.

    ``friends[q]`` is ``relation x band(strength of q / own strength)`` for each friendly
    power still in the game (0 for the rest); the threat weight of a power is
    ``(1 - relation) / 2``.
    """

    def __init__(
        self, world: WorldState, identity_id: str, catalog: Catalog, config: GoalConfig
    ) -> None:
        self.world = world
        self.identity = identity_id
        self.catalog = catalog
        self.config = config
        self.power = seat(identity_id)
        self.state = world.powers[self.power]
        self.others = [q for q in POWERS if q != self.power]
        self.own = military_strength(world, self.power, config)
        self.friends = {
            q: self._friendly(q) for q in self.others if world.powers[q].status != "eliminated"
        }

    def _friendly(self, q: str) -> float:
        relation = self.world.powers[self.power].relations.get(q, 0.0)
        allies = self.config.allies
        if allies is None or relation <= 0.0 or self.own <= 0.0:
            return 0.0
        ratio = military_strength(self.world, q, self.config) / self.own
        return relation * allies.value(self.identity, q, ratio)

    def net_hold(self, region_id: str, rivalry: float) -> float:
        """Own influence in a region, plus friends' (by how much each helps), less rivals'
        (by threat weight, times *rivalry*)."""

        region = self.world.regions[region_id]
        rivals = sum(
            threat_weight(self.world, self.power, q) * region.influence[q] for q in self.others
        )
        helped = sum(f * region.influence[q] for q, f in self.friends.items())
        return region.influence[self.power] + helped - rivalry * rivals


def _security(v: _Viewpoint) -> float:
    """``(own + allied support) / (own + support + weighted threats + allied burden)``, plus
    ``buffer_weight`` times the importance-weighted net hold of the identity's buffer
    regions (its approaches absorb threats, so holding them is security too)."""

    world, config = v.world, v.config
    threats = sum(
        threat_weight(world, v.power, q) * military_strength(world, q, config)
        for q in v.others
        if world.powers[q].status == "active"
    )
    allied = sum(f * v.own for q, f in v.friends.items() if world.powers[q].status == "active")
    support, burden = max(allied, 0.0), max(-allied, 0.0)
    total = v.own + support + threats + burden
    security = (v.own + support) / total if total > 0 else 0.5
    buffer_regions = config.buffers.get(v.identity, ())
    if buffer_regions and config.buffer_weight:
        held = weights = 0.0
        for region_id in buffer_regions:
            importance = world.regions[region_id].importance
            held += importance * v.net_hold(region_id, 1.0)  # a rival there counts in full
            weights += importance
        security += config.buffer_weight * held / weights
    return security


def _influence(v: _Viewpoint) -> float:
    """The mean over regions, weighted by importance times the identity's interest, of its
    net hold (rivals' influence scaled by its ``rivalry``)."""

    interest = v.config.interest.get(v.identity, {})
    rivalry = v.config.rivalry[v.identity]
    weight_sum = 0.0
    influence = 0.0
    for region_id in REGIONS:
        weight = v.world.regions[region_id].importance * interest.get(region_id, 1.0)
        influence += weight * v.net_hold(region_id, rivalry)
        weight_sum += weight
    return influence / weight_sum if weight_sum > 0 else 0.0


def _economy(v: _Viewpoint) -> float:
    """``ln(GDP / GDP in 1945) / ln(economy_scale)``."""

    base_economy = float(v.catalog.powers[v.power].initial.get("economy", 1.0))
    return math.log(max(v.state.economy, 1e-9) / base_economy) / math.log(v.config.economy_scale)


def _regime(v: _Viewpoint) -> float:
    """The ruling order's stability; a power governing only part of its country (an
    insurgency, or one side of a partition) counts the share it governs."""

    regime = v.state.stability
    partial = v.config.partial_regime.get(v.power)
    if partial is not None and partial[1](v.world):
        regime *= v.world.regions[partial[0]].influence[v.power]
    return regime


def _empire(v: _Viewpoint) -> float:
    """The colonies still held, each by ``importance x (influence - instability)``, over
    the importance of those held in 1945."""

    power = v.power
    held_1945 = sum(r.importance for r in v.catalog.regions.values() if r.colonial_owner == power)
    empire = 0.0
    if held_1945 > 0:
        for region in v.world.regions.values():
            if region.colonial_owner == power:
                hold = region.influence[power] - (1.0 - region.stability)
                empire += region.importance * hold
        empire /= held_1945
    return empire


def _autonomy(v: _Viewpoint) -> float:
    """Half for a tested bomb, half for freedom from a superpower's alliance."""

    subordinate = v.config.subordinate.get(v.identity)
    has_bomb = v.state.tech_nuclear >= BOMB_THRESHOLD
    independent = subordinate is None or not subordinate(v.world)
    return 0.5 * float(has_bomb) + 0.5 * float(independent)


def _credibility(v: _Viewpoint) -> float:
    """The power's reputation for keeping commitments."""

    return v.state.credibility


COMPONENT_VALUES: dict[str, Callable[[_Viewpoint], float]] = {
    "security": _security,
    "influence": _influence,
    "economy": _economy,
    "regime": _regime,
    "empire": _empire,
    "autonomy": _autonomy,
    "credibility": _credibility,
}


def components(
    world: WorldState, identity_id: str, catalog: Catalog, config: GoalConfig | None = None
) -> dict[str, float]:
    """The seven components; *config* overrides ``catalog.goals`` (used by sweeps)."""

    view = _Viewpoint(world, identity_id, catalog, config or catalog.goals)
    return {name: value(view) for name, value in COMPONENT_VALUES.items()}


def row(
    world: WorldState, identity_id: str, catalog: Catalog, config: GoalConfig | None = None
) -> WeightRow:
    """The identity's weight row in force in *world*: the first whose condition holds."""

    for candidate in (config or catalog.goals).rows[identity_id]:
        if candidate.when is None or candidate.when(world):
            return candidate
    raise AssertionError("the last weight row has no condition")  # enforced at load


def combine(parts: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """The weighted sum of the components *parts*."""

    return sum(weight * parts[name] for name, weight in weights.items())


def score(
    world: WorldState, identity_id: str, catalog: Catalog, config: GoalConfig | None = None
) -> float:
    """The identity's goal score in *world*."""

    parts = components(world, identity_id, catalog, config)
    return combine(parts, row(world, identity_id, catalog, config).weights)


def reward(
    previous: Mapping[str, float], current: Mapping[str, float], weights: Mapping[str, float]
) -> float:
    """Current weights times the change in components."""

    return sum(weight * (current[name] - previous[name]) for name, weight in weights.items())


# --------------------------------------------------------------------- parsing
def _mapping(name: str, value: Any) -> Mapping[str, Any]:
    """*value*, which must be a mapping."""

    if not isinstance(value, dict):
        raise GoalError(f"{name} must be a mapping")
    return value


def _only(name: str, value: Mapping[str, Any], allowed: set[str]) -> None:
    """Raise if *value* has a key outside *allowed*."""

    unknown = set(value) - allowed
    if unknown:
        raise GoalError(f"{name}: unknown keys {sorted(unknown)}")


def _number(name: str, value: Any, *, low: float = 0.0, strict: bool = False) -> float:
    """*value* as a finite number of at least *low* (above it if *strict*)."""

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise GoalError(f"{name} must be a number") from exc
    if not math.isfinite(number) or number < low or (strict and number == low):
        raise GoalError(f"{name} must be {'above' if strict else 'at least'} {low}")
    return number


def _identities(name: str, value: Mapping[str, Any]) -> None:
    """Raise if *value* has a key that is not an identity."""

    unknown = set(value) - set(IDENTITIES)
    if unknown:
        raise GoalError(f"{name}: unknown identities {sorted(unknown)}")


def _nuclear(raw: Mapping[str, Any]) -> NuclearModel:
    """The nuclear model of ``components.security.nuclear``."""

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
    """An identity's weight rows: weights over the components summing to 1, every row but
    the last with a condition.
    """

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
    """A band ``{floor, ceiling?}``; a ceiling must lie above the floor."""

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
    """The allies settings of ``components.allies``."""

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


def _buffers(security: Mapping[str, Any]) -> tuple[Mapping[str, Any], dict[str, tuple[str, ...]]]:
    """``security.buffer``: the mapping, and each identity's buffer regions."""

    buffer = _mapping("security.buffer", security.get("buffer", {}))
    _only("security.buffer", buffer, {"weight", "regions"})
    buffer_raw = _mapping("security.buffer.regions", buffer.get("regions", {}))
    _identities("security.buffer.regions", buffer_raw)
    buffers: dict[str, tuple[str, ...]] = {}
    for identity_id, regions in buffer_raw.items():
        if not isinstance(regions, list) or not set(regions) <= set(REGIONS):
            raise GoalError(f"security.buffer.regions.{identity_id}: a list of known regions")
        buffers[identity_id] = tuple(regions)
    return buffer, buffers


def _partial_regime(
    regime_raw: Mapping[str, Any], ctx: CompileContext
) -> dict[str, tuple[str, Predicate]]:
    """``regime.partial``: each power's home region and the condition under which it
    governs only part of it."""

    partial_regime: dict[str, tuple[str, Predicate]] = {}
    for power_id, spec in _mapping("regime.partial", regime_raw.get("partial", {})).items():
        spec = _mapping(f"regime.partial.{power_id}", spec)
        _only(f"regime.partial.{power_id}", spec, {"region", "when"})
        region_id = spec.get("region")
        if power_id not in POWERS or region_id not in REGIONS or "when" not in spec:
            raise GoalError(f"regime.partial.{power_id}: a power, a known region and a condition")
        partial_regime[power_id] = (str(region_id), compile_condition(spec["when"], ctx))
    return partial_regime


def _interest(influence: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """``influence.interest``: each identity's interest multiplier by region."""

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
    return interest


def _rivalry(influence: Mapping[str, Any]) -> dict[str, float]:
    """``influence.rivalry``: how much each identity counts rivals' influence against its own."""

    rivalry_raw = _mapping("influence.rivalry", influence.get("rivalry", {}))
    _identities("influence.rivalry", rivalry_raw)
    missing = set(IDENTITIES) - set(rivalry_raw)
    if missing:
        raise GoalError(f"influence.rivalry: missing {sorted(missing)}")
    return {i: _number(f"rivalry.{i}", v) for i, v in rivalry_raw.items()}


Attitude = tuple[float, float, float]  # prospect theory's (alpha, beta, lambda)


def _attitude(name: str, spec: Mapping[str, Any], default: Attitude) -> Attitude:
    """A prospect-theory attitude; alpha and beta lie in (0, 1], lambda is positive."""

    alpha, beta, lam = (
        _number(f"{name}.{key}", spec.get(key, d), strict=True)
        for key, d in zip(("alpha", "beta", "lambda"), default, strict=True)
    )
    if alpha > 1.0 or beta > 1.0:
        raise GoalError(f"{name}: alpha and beta must lie in (0, 1]")
    return (alpha, beta, lam)


def _risks(risk_raw: Mapping[str, Any]) -> tuple[Attitude, dict[str, Attitude]]:
    """``risk``: the default attitude, and each listed leader's (defaulting to it)."""

    risk = _attitude("risk", risk_raw, (1.0, 1.0, 1.0))
    by_leader = {
        str(leader): _attitude(
            f"risk.leaders.{leader}", _mapping(f"risk.leaders.{leader}", spec), risk
        )
        for leader, spec in _mapping("risk.leaders", risk_raw.get("leaders", {})).items()
    }
    return risk, by_leader


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
    buffer, buffers = _buffers(security)
    influence = _mapping("components.influence", comps.get("influence", {}))
    _only("components.influence", influence, {"interest", "rivalry"})
    autonomy = _mapping("components.autonomy", comps.get("autonomy", {}))
    _only("components.autonomy", autonomy, {"subordinate"})
    regime_raw = _mapping("components.regime", comps.get("regime", {}))
    _only("components.regime", regime_raw, {"partial"})
    partial_regime = _partial_regime(regime_raw, ctx)
    interest = _interest(influence)
    rivalry = _rivalry(influence)
    allies = _allies(comps["allies"]) if "allies" in comps else None
    subordinate_raw = _mapping("autonomy.subordinate", autonomy.get("subordinate", {}))
    _identities("autonomy.subordinate", subordinate_raw)
    subordinate = {i: compile_condition(spec, ctx) for i, spec in subordinate_raw.items()}
    standing = _mapping("standing", doc.get("standing", {}))
    _only("standing", standing, {"k", "start"})
    risk_raw = _mapping("risk", doc.get("risk", {}))
    _only("risk", risk_raw, {"alpha", "beta", "lambda", "leaders"})
    risk, risk_by_leader = _risks(risk_raw)
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
