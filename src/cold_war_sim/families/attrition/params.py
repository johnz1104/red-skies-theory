"""Parameters and configuration parsing for wars of attrition."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, Any

from ...games.protocol import ImmutableParams
from ...records import Record
from ..common import (
    ConfigError,
    Section,
    ValueTable,
    as_mapping,
    finite,
    identifier,
    nonnegative,
    probability,
    reject_unknown,
    require,
    value_table,
)

if TYPE_CHECKING:
    from .game import AttritionGame


NAME = "attrition"
OUTCOMES = ("a_victory", "b_victory", "partition", "a_withdrew", "b_withdrew", "stalemate")
REFUSED = "refused"
LEVELS = ("withdraw", "low", "high", "direct", "negotiate")
EFFORT = {"withdraw": 0.0, "low": 1 / 3, "high": 2 / 3, "direct": 1.0, "negotiate": 1 / 3}
STRONG, WEAK = "strong", "weak"
KINDS = (STRONG, WEAK)
STAGES = ("types", "approval", "warn", "commit", "offer", "respond", "ratify", "effort", "battle")
PROPOSERS = ("alternate", "a", "b")
EPS = 1e-12


class Sponsor(ImmutableParams, Record):
    """A sponsor of one side: its capacity, its costs and what it values.

    ``side``           ``a`` or ``b``.
    ``capacity``       its contribution to the side's effort per unit of effort level [1].
    ``direct``         whether it may fight with its own forces.
    ``cost_scale``     multiplies the per-round effort costs [1].
    ``values``         its payoff at each outcome label.
    ``settlement``     ``{base, per_share}``: a partition at its own share ``x`` is worth
                       ``base + per_share x`` (else its ``partition`` value).
    ``proxy_gain``     earned each round it stays in while another sponsor on its side
                       fights direct (an ally carries the fighting).
    ``withdraw_cost``  paid once if it ends the war withdrawn (abandoning an ally).
    ``direct_cost``    paid each round it fights direct, on top of the effort cost.
    """

    def __init__(
        self,
        id: str,
        side: str,
        capacity: float,
        direct: bool,
        cost_scale: float,
        values: ValueTable,
        settlement: tuple[float, float] | None = None,
        proxy_gain: float = 0.0,
        withdraw_cost: float = 0.0,
        direct_cost: float = 0.0,
    ) -> None:
        self.id = id
        self.side = side
        self.capacity = capacity
        self.direct = direct
        self.cost_scale = cost_scale
        self.values = values
        self.settlement = settlement  # (base, per_share) over its own share
        self.proxy_gain = proxy_gain  # per round in which another sponsor on its side is direct
        self.withdraw_cost = withdraw_cost  # paid once on ending the war withdrawn
        self.direct_cost = (
            direct_cost  # paid in each round it fights direct, on top of effort costs
        )

    def partition_value(self, share: float | None) -> float:
        """Its value of a partition at *share* (side A's), by its own share when it values
        settlements by share.
        """

        if self.settlement is None or share is None:
            return self.values["partition"]
        own = share if self.side == "a" else 1.0 - share
        return self.settlement[0] + self.settlement[1] * own


class SideTypes(ImmutableParams, Record):
    """The private type of a side or of a sponsor: a finite set of kinds, strongest first.

    Each kind multiplies the holder's contest effort (``strength``) and its effort costs
    (``cost``); a sponsor's kind may also scale its proxy gain (``proxy``). Nature draws the
    kind with ``probabilities``. Those who do not observe it hold the prior ``belief`` (the
    other side; for a ``private`` sponsor type, everyone else), which may differ from
    nature's. An ``unseen`` type is observed by no one: the holder's side holds
    ``own_belief`` and everyone learns from the battles.
    """

    def __init__(
        self,
        kinds: tuple[str, ...],
        probabilities: tuple[float, ...],
        strength: tuple[float, ...],
        cost: tuple[float, ...],
        belief: tuple[float, ...] | None = None,
        proxy: tuple[float, ...] = (),
        private: bool = False,
        unseen: bool = False,
        own_belief: tuple[float, ...] | None = None,
    ) -> None:
        self.kinds = kinds  # strongest first
        self.probabilities = probabilities  # nature's
        self.strength = strength  # multipliers on the side's contest effort
        self.cost = cost  # multipliers on its sponsors' effort costs
        self.belief = belief  # the other side's prior, if not nature's
        self.proxy = proxy  # multipliers on a sponsor's proxy gain (1 if empty)
        self.private = private  # seen by the holder alone, not by its side
        self.unseen = unseen  # seen by no one, not even its holder's side
        self.own_belief = own_belief  # the holder side's prior, if unseen

    def strength_of(self, kind: str) -> float:
        """The contest-effort multiplier of *kind*."""

        return self.strength[self.kinds.index(kind)]

    def proxy_of(self, kind: str) -> float:
        """The proxy-gain multiplier of *kind* (1 if none is set)."""

        return self.proxy[self.kinds.index(kind)] if self.proxy else 1.0

    def cost_of(self, kind: str) -> float:
        """The effort-cost multiplier of *kind*."""

        return self.cost[self.kinds.index(kind)]

    def rank(self, kind: str) -> int:
        """0 for the strongest kind."""

        return self.kinds.index(kind)

    def distribution(self) -> dict[str, float]:
        """Nature's distribution (the kinds that can be drawn)."""

        return {k: q for k, q in zip(self.kinds, self.probabilities, strict=True) if q > 0.0}

    def believed(self) -> dict[str, float]:
        """The other side's prior over these kinds (nature's unless a belief is set)."""

        weights = self.probabilities if self.belief is None else self.belief
        return {k: q for k, q in zip(self.kinds, weights, strict=True) if q > 0.0}

    def own_believed(self) -> dict[str, float]:
        """The holder side's own prior over an unseen type (nature's unless set)."""

        weights = self.probabilities if self.own_belief is None else self.own_belief
        return {k: q for k, q in zip(self.kinds, weights, strict=True) if q > 0.0}


class Consent(ImmutableParams, Record):
    """A sponsor that ratifies or vetoes a settlement; vetoing costs it ``veto_cost``."""

    def __init__(
        self,
        role: str,
        veto_cost: float,
    ) -> None:
        self.role = role
        self.veto_cost = veto_cost


class OfferParams(ImmutableParams, Record):
    """Offers: bargaining while fighting.

    From round ``opens`` on, before the fighting, one lead may offer a settlement from
    ``grid`` (side A's share of the contested ground); the other lead accepts or rejects,
    and the ``consent`` roles ratify or veto. An agreement ends the war as ``partition``
    with metric ``share``. ``proposer`` is ``alternate`` (A first), ``a`` or ``b``.
    """

    def __init__(
        self,
        grid: tuple[float, ...],
        proposer: str,
        consent: tuple[Consent, ...],
        opens: int,
    ) -> None:
        self.grid = grid  # side A's share of the contested ground
        self.proposer = proposer  # "alternate" (A first), "a" or "b"
        self.consent = consent
        self.opens = opens  # first round with offers

    def offer_id(self, index: int) -> str:
        """The action id of the grid's offer *index*: ``offer_050`` for a share of 0.5."""

        return f"offer_{round(self.grid[index] * 100):03d}"

    def ids(self) -> tuple[str, ...]:
        return tuple(self.offer_id(i) for i in range(len(self.grid)))

    def proposer_side(self, round_: int) -> str:
        """The side whose lead may offer in round *round_*."""

        if self.proposer == "alternate":
            return "a" if (round_ - self.opens) % 2 == 0 else "b"
        return self.proposer

    def consent_of(self, role: str) -> Consent:
        for item in self.consent:
            if item.role == role:
                return item
        raise KeyError(role)


class Mobility(ImmutableParams, Record):
    """The force-to-space ratio (Mearsheimer 1983; Biddle 2004).

    A battle moves the front with probability ``1 / (1 + (E / space)^sharpness)``, where
    ``E`` is the total effort committed; otherwise the front holds.
    """

    def __init__(
        self,
        space: float,
        sharpness: float,
    ) -> None:
        self.space = space
        self.sharpness = sharpness

    def moves(self, total: float) -> float:
        """P(the front moves) given the *total* effort committed."""

        if total <= 0.0:
            return 1.0
        return float(1.0 / (1.0 + (total / self.space) ** self.sharpness))


class RedLine(ImmutableParams, Record):
    """A sponsor's red line on the front.

    The line is crossed when control rises above it (``above``) or falls below it. The
    sponsor may warn, once, when the front is within a step of the line. If the line is
    then crossed and it does not go ``direct`` in the next round, it pays ``audience_cost``.
    """

    def __init__(
        self,
        role: str,
        line: float,
        above: bool,
        audience_cost: float,
    ) -> None:
        self.role = role
        self.line = line
        self.above = above  # crossed when control (side A's share) rises above the line
        self.audience_cost = audience_cost

    def crossed(self, control: float) -> bool:
        """Whether the front at *control* is past the line."""

        return control > self.line + EPS if self.above else control < self.line - EPS

    def near(self, control: float, step: float) -> bool:
        """Whether the front is within one *step* of the line without having crossed it."""

        if self.crossed(control):
            return False
        gap = self.line - control if self.above else control - self.line
        return gap <= step + EPS

    def beyond(self, step: float) -> float:
        """The control one step past the line."""

        return min(1.0, self.line + step) if self.above else max(0.0, self.line - step)


class Commitment(ImmutableParams, Record):
    """A lead's public commitment to minimum terms (Fearon 1994).

    Once talks are open the lead may commit, once, to a share of at least ``at_least``.
    Settling for less costs it ``audience_cost x scale``.
    """

    def __init__(
        self,
        role: str,
        side: str,
        at_least: float,
        audience_cost: float,
        scale: float = 1.0,
    ) -> None:
        self.role = role  # a side lead
        self.side = side
        self.at_least = at_least  # minimum own share
        self.audience_cost = audience_cost
        self.scale = scale  # multiplier on the audience cost (it may follow the world)

    @property
    def cost(self) -> float:
        """The audience cost of conceding."""

        return self.audience_cost * self.scale

    def breached(self, share: float) -> bool:
        """Whether a settlement at *share* (side A's) gives the lead less than it committed to."""

        own = share if self.side == "a" else 1.0 - share
        return own < self.at_least - EPS


class AttritionParams(ImmutableParams, Record):
    """The game form and payoffs of a war. Config keys (defaults in brackets):

    ``sponsors``           ``{role: {...}}`` (:class:`Sponsor`), at least one on each side.
    ``leads``              ``{a, b}``: each side's lead [its first sponsor].
    ``local_strength``     ``{a, b}``: each side's effort before its sponsors' [1].
    ``initial_control``    side A's starting share of the ground [0.5].
    ``step``               the share a battle moves the front [0.1].
    ``victory_threshold``  control at which A wins; B wins at its complement [0.85].
    ``max_rounds``         rounds before stalemate [12].
    ``gamma``              the contest's decisiveness [1]:
                           P(A wins a battle) = E_a^gamma / (E_a^gamma + E_b^gamma).
    ``effort``, ``costs``, ``weariness``   ``{low, high, direct}``: the effort a level
                           adds per unit of capacity, its cost, and the weariness it builds.
    ``weariness_limit``    weariness at which a sponsor is forced out [1].
    ``approval``           sponsors that must approve the war first; a refusal ends the
                           game as ``refused``.
    ``offers``             :class:`OfferParams`.
    ``types``              ``{a, b, <role>}``: :class:`SideTypes`.
    ``mobility``           :class:`Mobility`.
    ``red_lines``          ``{role: {...}}``: :class:`RedLine`.
    ``deployment``         the share of a sponsor's direct effort that arrives in its first
                           direct round (one number, or ``{role: share}``) [1].
    ``momentum``           each net step gained moves this share of the two local strengths
                           from the loser to the winner; a side keeps a tenth of its own [0].
    ``commitments``        ``{lead: {...}}``: :class:`Commitment`; needs offers.
    ``lead_on_entry``      ``{role: side}``: a sponsor that first goes direct becomes its
                           side's lead.
    """

    def __init__(
        self,
        sponsors: tuple[Sponsor, ...],
        lead_a: str,
        lead_b: str,
        local_a: float,
        local_b: float,
        initial_control: float,
        step: float,
        victory_threshold: float,
        max_rounds: int,
        gamma: float,
        effort: tuple[float, float, float],
        costs: tuple[float, float, float],
        weariness: tuple[float, float, float],
        weariness_limit: float,
        approval: tuple[str, ...] = (),
        offers: OfferParams | None = None,
        a_types: SideTypes | None = None,
        b_types: SideTypes | None = None,
        mobility: Mobility | None = None,
        red_lines: tuple[RedLine, ...] = (),
        deployment: float = 1.0,
        deployment_roles: dict[str, float] | None = None,
        momentum: float = 0.0,
        commitments: tuple[Commitment, ...] = (),
        sponsor_types: dict[str, SideTypes] | None = None,
        lead_on_entry: dict[str, str] | None = None,
    ) -> None:
        self.sponsors = sponsors
        self.lead_a = lead_a
        self.lead_b = lead_b
        self.local_a = local_a
        self.local_b = local_b
        self.initial_control = initial_control
        self.step = step
        self.victory_threshold = victory_threshold
        self.max_rounds = max_rounds
        self.gamma = gamma
        self.effort = effort  # low, high, direct
        self.costs = costs
        self.weariness = weariness
        self.weariness_limit = weariness_limit
        self.approval = approval
        self.offers = offers
        self.a_types = a_types
        self.b_types = b_types
        self.mobility = mobility
        self.red_lines = red_lines
        self.deployment = (
            deployment  # share of direct effort that arrives in a sponsor's first direct round
        )
        self.deployment_roles = deployment_roles or {}  # role -> its share, if given by role
        self.momentum = momentum  # share of the local strengths that moves per net step gained
        self.commitments = commitments
        self.sponsor_types = sponsor_types or {}  # role -> its own levels, in config order
        self.lead_on_entry = lead_on_entry or {}  # role -> the side it leads once it goes direct
        self._sponsor = {s.id: s for s in sponsors}
        self._red_line = {item.role: item for item in self.red_lines}
        self._commitment = {item.role: item for item in self.commitments}

    def sponsor(self, role: str) -> Sponsor:
        return self._sponsor[role]

    def level_index(self, level: str) -> int:
        """The index of an effort *level* in the ``effort``, ``costs`` and ``weariness`` triples
        (-1 for a withdrawal).
        """

        return {"low": 0, "negotiate": 0, "high": 1, "direct": 2}.get(level, -1)

    def local_strength(self, side: str, control: float) -> float:
        """The side's local strength with the front at *control* (it shifts with
        ``momentum``; a function of the front alone)."""

        base = self.local_a if side == "a" else self.local_b
        if self.momentum == 0.0:
            return base
        steps = (control - self.initial_control) / self.step
        shift = self.momentum * steps * (self.local_a + self.local_b)
        return max(0.1 * base, base + shift if side == "a" else base - shift)

    def deployment_of(self, role: str) -> float:
        """Share of the role's direct effort that arrives in its first direct round."""

        return self.deployment_roles.get(role, self.deployment)

    @property
    def labels(self) -> tuple[str, ...]:
        """The outcome labels: ``refused`` is one only when the war needs approval."""

        return (*OUTCOMES, REFUSED) if self.approval else OUTCOMES

    @property
    def extended(self) -> bool:
        """Whether any bargaining-while-fighting block is configured."""

        return bool(
            self.approval
            or self.offers is not None
            or self.a_types is not None
            or self.b_types is not None
            or self.mobility is not None
            or self.red_lines
            or self.deployment < 1.0
            or any(share < 1.0 for share in self.deployment_roles.values())
            or self.momentum > 0.0
            or self.commitments
            or self.sponsor_types
            or self.lead_on_entry
        )

    @property
    def holders(self) -> tuple[str, ...]:
        """The type holders: the sides (``a``, ``b``), then the sponsors with their own types."""

        sides = tuple(side for side in ("a", "b") if self.types_of(side) is not None)
        return sides + tuple(self.sponsor_types)

    def types_of(self, holder: str) -> SideTypes | None:
        """The type of *holder* (a side or a sponsor), if it has one."""

        if holder == "a":
            return self.a_types
        if holder == "b":
            return self.b_types
        return self.sponsor_types.get(holder)

    def holder_side(self, holder: str) -> str:
        """The side a type *holder* belongs to."""

        return holder if holder in ("a", "b") else self.sponsor(holder).side

    def lead(self, side: str) -> str:
        return self.lead_a if side == "a" else self.lead_b

    def side_roles(self, side: str) -> tuple[str, ...]:
        """The sponsors of *side*, in config order."""

        return tuple(s.id for s in self.sponsors if s.side == side)

    def red_line(self, role: str) -> RedLine | None:
        return self._red_line.get(role)

    def commitment(self, role: str) -> Commitment | None:
        return self._commitment.get(role)


def other_side(side: str) -> str:
    """The side facing *side*."""

    return "b" if side == "a" else "a"


Triple = tuple[float, float, float]  # by effort level: low, high, direct


def build(config: Mapping[str, Any]) -> AttritionGame:
    """The war of attrition of *config* (keys: :class:`AttritionParams`)."""

    from .game import AttritionGame

    with Section("attrition", config) as s:
        approval_raw = s.get("approval", [])
        labels = (*OUTCOMES, REFUSED) if approval_raw else OUTCOMES
        sponsor_blocks = as_mapping(s.at("sponsors"), s.get("sponsors"))
        sponsors = [_sponsor(role, raw, labels) for role, raw in sponsor_blocks.items()]
        side_of = {sp.id: sp.side for sp in sponsors}
        if set(side_of.values()) != {"a", "b"}:
            raise ConfigError("attrition needs at least one sponsor on each side")
        lead_a, lead_b = _leads(s, sponsors)
        with s.section("local_strength", {}) as local:
            local_a = local.nonnegative("a", 1.0)
            local_b = local.nonnegative("b", 1.0)
        threshold = s.probability("victory_threshold", 0.85)
        if threshold <= 0.5:
            raise ConfigError("attrition.victory_threshold must exceed 0.5")
        initial = s.probability("initial_control", 0.5)
        if not 1.0 - threshold < initial < threshold:
            raise ConfigError("attrition.initial_control must lie strictly between the thresholds")
        gamma = _positive(s, "gamma", 1.0)
        step = s.probability("step", 0.1)
        if step <= 0.0:
            raise ConfigError("attrition.step must be positive")
        weariness_limit = _positive(s, "weariness_limit", 1.0)
        approval = _roles(s.at("approval"), approval_raw, side_of)
        offers = (
            _offers(s.section("offers"), side_of, (lead_a, lead_b)) if s.has("offers") else None
        )
        types = as_mapping(s.at("types"), s.get("types", {}))
        reject_unknown(s.at("types"), types, ("a", "b", *side_of))
        lead_on_entry = _lead_on_entry(s, sponsors)
        deployment, deployment_roles = _deployment(s.get("deployment", 1.0), sponsors)
        params = AttritionParams(
            sponsors=tuple(sponsors),
            lead_a=lead_a,
            lead_b=lead_b,
            local_a=local_a,
            local_b=local_b,
            initial_control=initial,
            step=step,
            victory_threshold=threshold,
            max_rounds=s.positive_int("max_rounds", 12),
            gamma=gamma,
            effort=_triple(s, "effort", (1.0, 2.0, 4.0)),
            costs=_triple(s, "costs", (0.02, 0.06, 0.15)),
            weariness=_triple(s, "weariness", (0.0, 0.02, 0.08)),
            weariness_limit=weariness_limit,
            approval=approval,
            offers=offers,
            a_types=_side_types("a", types["a"]) if "a" in types else None,
            b_types=_side_types("b", types["b"]) if "b" in types else None,
            sponsor_types={
                role: _side_types(role, spec, sponsor=True)
                for role, spec in types.items()
                if role in side_of
            },
            lead_on_entry=lead_on_entry,
            mobility=_mobility(s.section("mobility")) if s.has("mobility") else None,
            red_lines=_red_lines(s.get("red_lines", {}), sponsors),
            deployment=deployment,
            deployment_roles=deployment_roles,
            momentum=s.nonnegative("momentum", 0.0),
            commitments=_commitments(
                s.get("commitments", {}), side_of, {lead_a, lead_b, *lead_on_entry}, offers
            ),
        )
    return AttritionGame(params)


def _positive(s: Section, key: str, default: float) -> float:
    """A number above zero."""

    value = s.number(key, default)
    if value <= 0.0:
        raise ConfigError(f"{s.at(key)} must be positive")
    return value


def _triple(s: Section, key: str, default: Triple) -> Triple:
    """A non-negative number for each effort level under *key*."""

    with s.section(key, {}) as levels:
        return (
            levels.nonnegative("low", default[0]),
            levels.nonnegative("high", default[1]),
            levels.nonnegative("direct", default[2]),
        )


def _sponsor(role_id: str, data: Any, labels: tuple[str, ...]) -> Sponsor:
    """Read one sponsor, including outcome values and optional settlement terms."""

    with Section(f"attrition.sponsors.{role_id}", data) as s:
        side = s.get("side")
        if side not in ("a", "b"):
            raise ConfigError(f"{s.at('side')} must be 'a' or 'b'")
        settlement = None
        if s.has("settlement"):
            with s.section("settlement") as terms:
                settlement = (terms.number("base"), terms.number("per_share"))
        return Sponsor(
            id=identifier(s.name, role_id),
            side=side,
            capacity=s.nonnegative("capacity", 1.0),
            direct=s.flag("direct"),
            cost_scale=s.nonnegative("cost_scale", 1.0),
            values=value_table(s.at("values"), s.get("values"), labels),
            settlement=settlement,
            proxy_gain=s.nonnegative("proxy_gain", 0.0),
            withdraw_cost=s.nonnegative("withdraw_cost", 0.0),
            direct_cost=s.nonnegative("direct_cost", 0.0),
        )


def _leads(s: Section, sponsors: list[Sponsor]) -> tuple[str, str]:
    """Each side's lead: the one named, or the side's first sponsor."""

    side_of = {sp.id: sp.side for sp in sponsors}
    with s.section("leads", {}) as leads:
        lead_a = leads.get("a", next(sp.id for sp in sponsors if sp.side == "a"))
        lead_b = leads.get("b", next(sp.id for sp in sponsors if sp.side == "b"))
    if side_of.get(lead_a) != "a" or side_of.get(lead_b) != "b":
        raise ConfigError("attrition leads must be sponsors on their own side")
    return lead_a, lead_b


def _roles(name: str, raw: Any, known: Mapping[str, str]) -> tuple[str, ...]:
    """A list of distinct sponsors."""

    if not isinstance(raw, (list, tuple)):
        raise ConfigError(f"{name} must be a list")
    roles = tuple(identifier(name, r) for r in raw)
    if len(set(roles)) != len(roles):
        raise ConfigError(f"{name} lists a role twice")
    for role in roles:
        if role not in known:
            raise ConfigError(f"{name}: {role!r} is not a sponsor")
    return roles


def _offers(s: Section, side_of: Mapping[str, str], leads: tuple[str, str]) -> OfferParams:
    """Read the increasing offer shares and the sponsors whose consent is required."""

    with s:
        grid = tuple(probability(s.at("grid"), x) for x in s.sequence("grid"))
        if not grid or any(not 0.0 < g < 1.0 for g in grid) or list(grid) != sorted(set(grid)):
            raise ConfigError(f"{s.at('grid')} must be increasing shares strictly inside (0, 1)")
        proposer = s.choice("proposer", PROPOSERS, "alternate")
        consent = []
        for i, item in enumerate(s.sequence("consent", [])):
            with Section(
                f"{s.at('consent')}.{i}", {"role": item} if isinstance(item, str) else item
            ) as c:
                role = identifier(c.at("role"), c.get("role"))
                if role not in side_of:
                    raise ConfigError(f"{c.name}: {role!r} is not a sponsor")
                if role in leads:
                    raise ConfigError(f"{c.name}: a side lead cannot be a consent role")
                consent.append(Consent(role, c.nonnegative("veto_cost", 0.0)))
        if len({c.role for c in consent}) != len(consent):
            raise ConfigError(f"{s.at('consent')} lists a role twice")
        opens = s.get("opens", 0)
        if isinstance(opens, bool) or not isinstance(opens, int) or opens < 0:
            raise ConfigError(f"{s.at('opens')} must be a non-negative integer")
        return OfferParams(grid, proposer, tuple(consent), opens)


def _side_types(holder: str, raw: Any, sponsor: bool = False) -> SideTypes:
    """Two kinds (``prior``, ``strong``, ``weak``) or ``levels``: {name: {weight,
    belief?, strength, cost, proxy?}}, strongest first, weights normalized. ``belief``
    is the prior of those who do not see the type; levels without one get none once
    any level has one. A sponsor's type may be ``private`` and its levels may scale
    its proxy gain (``proxy``)."""

    name = f"attrition.types.{holder}"
    data = as_mapping(name, raw)
    belief: tuple[float, ...] | None = None
    own_belief: tuple[float, ...] | None = None
    extra: tuple[str, ...] = ("private", "unseen") if sponsor else ("unseen",)
    private = bool(data.get("private", False)) if sponsor else False
    unseen = bool(data.get("unseen", False))
    if private and unseen:
        raise ConfigError(f"{name}: a type cannot be both private and unseen")
    if "levels" in data:
        reject_unknown(name, data, ("levels", *extra))
        levels = as_mapping(f"{name}.levels", data["levels"])
        if len(levels) < 2:
            raise ConfigError(f"{name}.levels needs at least two levels")
        specs = {identifier(f"{name}.levels", k): v for k, v in levels.items()}
        weights = {}
        for kind, spec in specs.items():
            entry = as_mapping(f"{name}.levels.{kind}", spec)
            weights[kind] = nonnegative(
                f"{name}.levels.{kind}.weight", require(name, entry, "weight")
            )
        total = sum(weights.values())
        if total <= 0.0:
            raise ConfigError(f"{name}.levels needs a positive weight")
        allowed: tuple[str, ...] = ("weight", "belief", "strength", "cost")
        if sponsor:
            allowed += ("proxy",)
        if unseen:
            allowed += ("own_belief",)
        probabilities = {k: w / total for k, w in weights.items()}
        if any("belief" in as_mapping(name, spec) for spec in specs.values()):
            believed = [
                nonnegative(f"{name}.levels.{k}.belief", as_mapping(name, v).get("belief", 0.0))
                for k, v in specs.items()
            ]
            if sum(believed) <= 0.0:
                raise ConfigError(f"{name}.levels needs a positive belief")
            belief = tuple(b / sum(believed) for b in believed)
        if unseen and any("own_belief" in as_mapping(name, spec) for spec in specs.values()):
            owned = [
                nonnegative(
                    f"{name}.levels.{k}.own_belief", as_mapping(name, v).get("own_belief", 0.0)
                )
                for k, v in specs.items()
            ]
            if sum(owned) <= 0.0:
                raise ConfigError(f"{name}.levels needs a positive own_belief")
            own_belief = tuple(b / sum(owned) for b in owned)
    else:
        reject_unknown(name, data, ("prior", "strong", "weak", *extra))
        prior = probability(f"{name}.prior", require(name, data, "prior"))
        specs = {kind: data.get(kind, {}) for kind in KINDS}
        allowed = ("strength", "cost")
        probabilities = {STRONG: prior, WEAK: 1.0 - prior}
    strength, cost, proxy = [], [], []
    for kind, spec in specs.items():
        entry = as_mapping(f"{name}.{kind}", spec)
        reject_unknown(f"{name}.{kind}", entry, allowed)
        value = finite(f"{name}.{kind}.strength", entry.get("strength", 1.0))
        if value <= 0.0:
            raise ConfigError(f"{name}.{kind}.strength must be positive")
        strength.append(value)
        cost.append(nonnegative(f"{name}.{kind}.cost", entry.get("cost", 1.0)))
        proxy.append(nonnegative(f"{name}.{kind}.proxy", entry.get("proxy", 1.0)))
    kinds = tuple(specs)
    return SideTypes(
        kinds,
        tuple(probabilities[k] for k in kinds),
        tuple(strength),
        tuple(cost),
        belief,
        tuple(proxy) if any(x != 1.0 for x in proxy) else (),
        private,
        unseen,
        own_belief,
    )


def _mobility(s: Section) -> Mobility:
    with s:
        space, sharpness = s.number("space"), s.number("sharpness")
        if space <= 0.0 or sharpness <= 0.0:
            raise ConfigError(f"{s.at('space')} and sharpness must be positive")
        return Mobility(space, sharpness)


def _lead_on_entry(s: Section, sponsors: list[Sponsor]) -> dict[str, str]:
    """``{role: side}``: each role must be a direct sponsor of the side it would lead."""

    by_id = {sp.id: sp for sp in sponsors}
    listed = as_mapping(s.at("lead_on_entry"), s.get("lead_on_entry", {}))
    for role, side in listed.items():
        sponsor = by_id.get(role)
        if sponsor is None or not sponsor.direct or side != sponsor.side:
            raise ConfigError(
                f"{s.at('lead_on_entry')}.{role}: needs a direct sponsor of side {side!r}"
            )
    return dict(listed)


def _deployment(raw: Any, sponsors: list[Sponsor]) -> tuple[float, dict[str, float]]:
    """One share for every sponsor, or shares by role for direct sponsors (unlisted
    sponsors arrive in full)."""

    def share(name: str, value: Any) -> float:
        result = probability(name, value)
        if result <= 0.0:
            raise ConfigError(f"{name} must lie in (0, 1]")
        return result

    if not isinstance(raw, Mapping):
        return share("attrition.deployment", raw), {}
    direct = {sp.id for sp in sponsors if sp.direct}
    roles: dict[str, float] = {}
    for role, value in raw.items():
        if role not in direct:
            raise ConfigError(f"attrition.deployment: {role!r} is not a direct sponsor")
        roles[role] = share(f"attrition.deployment.{role}", value)
    return 1.0, roles


def _red_lines(raw: Any, sponsors: list[Sponsor]) -> tuple[RedLine, ...]:
    """Read intervention thresholds, requiring each named sponsor to have direct forces."""

    by_id = {sp.id: sp for sp in sponsors}
    result = []
    for role, spec in as_mapping("attrition.red_lines", raw).items():
        name = f"attrition.red_lines.{role}"
        if role not in by_id:
            raise ConfigError(f"{name}: {role!r} is not a sponsor")
        if not by_id[role].direct:
            raise ConfigError(f"{name}: the role must be able to go direct")
        with Section(name, spec) as s:
            if s.has("above") == s.has("below"):
                raise ConfigError(f"{name} needs exactly one of 'above' and 'below'")
            above = s.has("above")
            line = probability(s.at("line"), s.get("above" if above else "below"))
            result.append(RedLine(role, line, above, s.nonnegative("audience_cost", 0.0)))
    return tuple(result)


def _commitments(
    raw: Any, side_of: Mapping[str, str], leads: Collection[str], offers: OfferParams | None
) -> tuple[Commitment, ...]:
    """Read leaders' minimum settlement shares and the costs of breaking those promises."""

    data = as_mapping("attrition.commitments", raw)
    if data and offers is None:
        raise ConfigError("attrition.commitments need offers")
    result = []
    for role, spec in data.items():
        name = f"attrition.commitments.{role}"
        if role not in leads:
            raise ConfigError(f"{name}: only a side lead (now or on entry) can commit")
        with Section(name, spec) as s:
            at_least = s.probability("at_least")
            if not 0.0 < at_least < 1.0:
                raise ConfigError(f"{s.at('at_least')} must lie strictly inside (0, 1)")
            cost = s.nonnegative("audience_cost", 0.0)
            result.append(
                Commitment(role, side_of[role], at_least, cost, s.nonnegative("scale", 1.0))
            )
    return tuple(result)
