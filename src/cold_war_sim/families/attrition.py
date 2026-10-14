"""Proxy war as a stochastic war of attrition between sponsor coalitions.

Two local sides fight over *control* (side A's share, in [0, 1]). Each round
every active sponsor role secretly chooses an effort level; choices are
revealed together, then one battle is resolved by a contest-success function

    P(A gains) = E_A^gamma / (E_A^gamma + E_B^gamma),
    E_side = local strength + sum(capacity x effort level of its sponsors).

Control moves one step toward the winner. Effort costs accumulate every round,
and high or direct effort builds war-weariness; a sponsor whose weariness
reaches the limit is forced out (domestic support collapses). A sponsor may
``withdraw`` permanently; in a war without offers, the two side leads may both
choose ``negotiate`` to end the war in partition at the line of contact.

Outcome labels: ``a_victory``, ``b_victory``, ``partition``, ``a_withdrew``
(every A sponsor left), ``b_withdrew``, ``stalemate`` (round limit). Long
horizons make the tree too large for exact methods, so the baseline is
:class:`AttritionLookaheadPolicy` (:class:`AttritionHeuristicPolicy`, a
one-round rule, is kept for comparison).

Optional blocks turn the war into bargaining while fighting (Wagner 2000;
Filson and Werner 2002; Slantchev 2003; Powell 2004):

* ``approval``: before the war each listed sponsor approves or refuses it, in
  turn; a refusal ends the game as ``refused`` (a label only these wars have).
* ``offers``: each round, before the fighting, one side's lead may offer a
  settlement (side A's share of the contested ground); the other lead accepts
  or rejects, and listed consent roles ratify or veto. An agreement ends the
  war as ``partition`` with metric ``share``; sponsors may value it by share
  (``settlement``). Settlements then come only through offers: the leads fight
  while they bargain and cannot ``negotiate`` a settlement at the line.
* ``types``: each side's strength (a multiplier on its contest effort) and
  resolve (a multiplier on its sponsors' effort costs) may be private, drawn by
  chance at the start and seen only by that side's sponsors: two kinds
  (``prior``, ``strong``, ``weak``) or any number of ``levels``, strongest first,
  so that beliefs can shift a little with each round a side holds out. A
  sponsor may have levels of its own (``types.<ROLE>``: its stakes, as a cost
  multiplier, and an optional strength multiplier), seen by its side, or with
  ``private: true`` by the sponsor alone; a sponsor's level may also scale its
  ``proxy_gain`` (``proxy``). Each level may carry a ``belief`` weight: the prior
  of those who do not see it (the other side, and for a private type everyone
  else), which may differ from nature's (a leader's own, possibly mistaken, view).
  A type may also be ``unseen``: no one sees it, not even its holder's side, which
  holds its own prior (``own_belief``); everyone learns it from the battles.
* ``mobility``: the chance that a battle moves the front at all falls with the
  total force committed, the force-to-space ratio (Mearsheimer 1983; Biddle
  2004); otherwise the front holds.
* ``red_lines``: a sponsor may warn, once, when the front comes within a step of
  a line. If the line is crossed and it does not go ``direct`` in the next
  round, it pays an audience cost (tag ``<role>:bluffed``).
* ``deployment``: direct intervention takes time to arrive. A sponsor's direct
  effort counts at this fraction in the first round it goes direct, and in full
  from its next round. One number for every sponsor, or ``{role: fraction}``
  (unlisted sponsors arrive in full): a power that massed its forces beforehand
  strikes at full strength.
* ``momentum``: the local balance shifts with the front. Each net step a side has
  gained from the starting line moves this share of the two local strengths' sum
  from the loser's local strength to the winner's (ground won brings over part of
  the enemy's forces, as defections do); each side keeps at least a tenth of its
  starting strength.
* ``lead_on_entry``: a sponsor that first goes ``direct`` becomes its side's lead
  (a belligerent that takes over the war it joins).
* ``commitments`` (with ``offers``): once talks are open, a side lead may commit
  publicly, once, to minimum terms (its own share). A settlement that gives it
  less costs it an audience cost (tag ``<role>:conceded``), times an optional
  ``scale`` (the weight of the pledge may follow the world, such as a patron's
  stance). Two leads committed to incompatible terms fight a war of
  attrition over which one backs down (Fearon 1994 on audience costs).

A sponsor may also value the war going on: ``proxy_gain`` is its gain in each round
in which another sponsor on its side fights ``direct`` while it stays in the war (an
ally carries the fighting while the enemy pays for it). With a consent role, it
makes that role hold out for better terms or veto a settlement outright. A sponsor
that ends the war withdrawn, by choice or forced out, pays its ``withdraw_cost``
(abandoning an ally or a client), and one that fights ``direct`` pays its
``direct_cost`` each such round on top of its effort cost (the political price of
sending its own forces).

Their reference policy is :class:`~.attrition_beliefs.AttritionBeliefPolicy`.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

from ..games.protocol import (
    CHANCE,
    TERMINAL,
    GameState,
    ImmutableParams,
    LoggedState,
    Outcome,
    check_legal,
)
from .common import (
    ConfigError,
    ValueTable,
    as_list,
    as_mapping,
    finite,
    identifier,
    nonnegative,
    positive_int,
    probability,
    reject_unknown,
    require,
    value_table,
)

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


@dataclass(frozen=True, slots=True)
class Sponsor(ImmutableParams):
    id: str
    side: str
    capacity: float
    direct: bool
    cost_scale: float
    values: ValueTable
    settlement: tuple[float, float] | None = None  # (base, per_share) over its own share
    proxy_gain: float = 0.0  # per round in which another sponsor on its side is direct
    withdraw_cost: float = 0.0  # paid once on ending the war withdrawn
    direct_cost: float = 0.0  # paid in each round it fights direct, on top of effort costs

    def partition_value(self, share: float | None) -> float:
        """Value of a partition; by share when the sponsor values settlements by share."""

        if self.settlement is None or share is None:
            return self.values["partition"]
        own = share if self.side == "a" else 1.0 - share
        return self.settlement[0] + self.settlement[1] * own


@dataclass(frozen=True, slots=True)
class SideTypes(ImmutableParams):
    kinds: tuple[str, ...]  # strongest first
    probabilities: tuple[float, ...]  # nature's
    strength: tuple[float, ...]  # multipliers on the side's contest effort
    cost: tuple[float, ...]  # multipliers on its sponsors' effort costs
    belief: tuple[float, ...] | None = None  # the other side's prior, if not nature's
    proxy: tuple[float, ...] = ()  # multipliers on a sponsor's proxy gain (1 if empty)
    private: bool = False  # seen by the holder alone, not by its side
    unseen: bool = False  # seen by no one, not even its holder's side
    own_belief: tuple[float, ...] | None = None  # the holder side's prior, if unseen

    def strength_of(self, kind: str) -> float:
        return self.strength[self.kinds.index(kind)]

    def proxy_of(self, kind: str) -> float:
        return self.proxy[self.kinds.index(kind)] if self.proxy else 1.0

    def cost_of(self, kind: str) -> float:
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


@dataclass(frozen=True, slots=True)
class Consent(ImmutableParams):
    role: str
    veto_cost: float


@dataclass(frozen=True, slots=True)
class OfferParams(ImmutableParams):
    grid: tuple[float, ...]  # side A's share of the contested ground
    proposer: str  # "alternate" (A first), "a" or "b"
    consent: tuple[Consent, ...]
    opens: int  # first round with offers

    def offer_id(self, index: int) -> str:
        return f"offer_{round(self.grid[index] * 100):03d}"

    def ids(self) -> tuple[str, ...]:
        return tuple(self.offer_id(i) for i in range(len(self.grid)))

    def proposer_side(self, round_: int) -> str:
        if self.proposer == "alternate":
            return "a" if (round_ - self.opens) % 2 == 0 else "b"
        return self.proposer

    def consent_of(self, role: str) -> Consent:
        for item in self.consent:
            if item.role == role:
                return item
        raise KeyError(role)


@dataclass(frozen=True, slots=True)
class Mobility(ImmutableParams):
    space: float
    sharpness: float

    def moves(self, total: float) -> float:
        """Chance that a battle moves the front, given the total force committed."""

        if total <= 0.0:
            return 1.0
        return float(1.0 / (1.0 + (total / self.space) ** self.sharpness))


@dataclass(frozen=True, slots=True)
class RedLine(ImmutableParams):
    role: str
    line: float
    above: bool  # crossed when control (side A's share) rises above the line
    audience_cost: float

    def crossed(self, control: float) -> bool:
        return control > self.line + EPS if self.above else control < self.line - EPS

    def near(self, control: float, step: float) -> bool:
        if self.crossed(control):
            return False
        gap = self.line - control if self.above else control - self.line
        return gap <= step + EPS

    def beyond(self, step: float) -> float:
        """The control one step past the line."""

        return min(1.0, self.line + step) if self.above else max(0.0, self.line - step)


@dataclass(frozen=True, slots=True)
class Commitment(ImmutableParams):
    role: str  # a side lead
    side: str
    at_least: float  # minimum own share
    audience_cost: float
    scale: float = 1.0  # multiplier on the audience cost (it may follow the world)

    @property
    def cost(self) -> float:
        """The audience cost charged for conceding."""

        return self.audience_cost * self.scale

    def breached(self, share: float) -> bool:
        own = share if self.side == "a" else 1.0 - share
        return own < self.at_least - EPS


@dataclass(frozen=True, slots=True)
class AttritionParams(ImmutableParams):
    sponsors: tuple[Sponsor, ...]
    lead_a: str
    lead_b: str
    local_a: float
    local_b: float
    initial_control: float
    step: float
    victory_threshold: float
    max_rounds: int
    gamma: float
    effort: tuple[float, float, float]  # low, high, direct
    costs: tuple[float, float, float]
    weariness: tuple[float, float, float]
    weariness_limit: float
    approval: tuple[str, ...] = ()
    offers: OfferParams | None = None
    a_types: SideTypes | None = None
    b_types: SideTypes | None = None
    mobility: Mobility | None = None
    red_lines: tuple[RedLine, ...] = ()
    deployment: float = 1.0  # share of direct effort that arrives in a sponsor's first direct round
    deployment_roles: tuple[tuple[str, float], ...] = ()  # per-role shares, if given by role
    momentum: float = 0.0  # share of the local strengths that moves per net step gained
    commitments: tuple[Commitment, ...] = ()
    sponsor_types: tuple[tuple[str, SideTypes], ...] = ()  # (role, levels) in config order
    lead_on_entry: tuple[tuple[str, str], ...] = ()  # (role, side): leads once it goes direct

    def sponsor(self, role: str) -> Sponsor:
        for sponsor in self.sponsors:
            if sponsor.id == role:
                return sponsor
        raise KeyError(role)

    def level_index(self, level: str) -> int:
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

        return dict(self.deployment_roles).get(role, self.deployment)

    @property
    def labels(self) -> tuple[str, ...]:
        return (*OUTCOMES, REFUSED) if self.approval else OUTCOMES

    @property
    def extended(self) -> bool:
        """True when any bargaining-while-fighting block is configured."""

        return bool(
            self.approval
            or self.offers is not None
            or self.a_types is not None
            or self.b_types is not None
            or self.mobility is not None
            or self.red_lines
            or self.deployment < 1.0
            or any(share < 1.0 for _, share in self.deployment_roles)
            or self.momentum > 0.0
            or self.commitments
            or self.sponsor_types
            or self.lead_on_entry
        )

    @property
    def holders(self) -> tuple[str, ...]:
        """Who has private types: the sides (``a``, ``b``), then sponsors."""

        sides = tuple(side for side in ("a", "b") if self.types_of(side) is not None)
        return sides + tuple(role for role, _ in self.sponsor_types)

    def types_of(self, holder: str) -> SideTypes | None:
        if holder == "a":
            return self.a_types
        if holder == "b":
            return self.b_types
        return dict(self.sponsor_types).get(holder)

    def holder_side(self, holder: str) -> str:
        return holder if holder in ("a", "b") else self.sponsor(holder).side

    def lead(self, side: str) -> str:
        return self.lead_a if side == "a" else self.lead_b

    def side_roles(self, side: str) -> tuple[str, ...]:
        return tuple(s.id for s in self.sponsors if s.side == side)

    def red_line(self, role: str) -> RedLine | None:
        for item in self.red_lines:
            if item.role == role:
                return item
        return None

    def commitment(self, role: str) -> Commitment | None:
        for item in self.commitments:
            if item.role == role:
                return item
        return None


def other_side(side: str) -> str:
    return "b" if side == "a" else "a"


class AttritionGame:
    name = NAME

    def __init__(self, params: AttritionParams) -> None:
        self.params = params

    @property
    def outcome_labels(self) -> tuple[str, ...]:
        return self.params.labels

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(s.id for s in self.params.sponsors)

    def new_initial_state(self) -> AttritionState:
        return AttritionState(self.params)


class AttritionState(LoggedState):
    def __init__(self, params: AttritionParams) -> None:
        super().__init__(s.id for s in params.sponsors)
        self.params = params
        self.control = params.initial_control
        self.round = 0
        self.withdrawn: set[str] = set()
        self.choices: dict[str, str] = {}
        self.last_choices: dict[str, str] = {}
        self.pending: list[str] = []
        self.cost: dict[str, float] = {s.id: 0.0 for s in params.sponsors}
        self.weariness: dict[str, float] = {s.id: 0.0 for s in params.sponsors}
        self.effort: dict[str, float] = {s.id: 0.0 for s in params.sponsors}
        self.ever_direct: set[str] = set()
        self.deployed: set[str] = set()  # went direct in an earlier round: fully deployed
        self.label: str | None = None
        # bargaining while fighting
        self.stage = "effort"
        self.types: dict[str, str] = {}  # holder -> kind
        self.priors: dict[str, dict[str, float]] = {}  # nature's distribution, by holder
        for holder in params.holders:
            holder_types = params.types_of(holder)
            assert holder_types is not None
            self.priors[holder] = holder_types.distribution()
        self.type_queue = list(params.holders)
        self.leads = {"a": params.lead_a, "b": params.lead_b}
        self.approvals = list(params.approval)
        self.warn_queue: list[str] = []
        self.commit_queue: list[str] = []
        self.ratify_queue: list[str] = []
        self.offer: int | None = None
        self.share: float | None = None
        self.agreed = False
        self.warned: set[str] = set()
        self.bluffed: set[str] = set()
        self.honored: set[str] = set()
        self.vetoed: set[str] = set()
        self.committed: set[str] = set()
        self.conceded: set[str] = set()
        self.refused_by: str | None = None
        self.last_battle: str | None = None
        self.peak = self.trough = self.control
        self._next_type()

    def clone(self) -> AttritionState:
        """An independent copy, much faster than a deep copy: every container holds
        strings and numbers (the per-role view logs and priors, one level more), so
        copying each one level deep suffices; the immutable parameters are shared."""

        new = object.__new__(type(self))
        for name, value in self.__dict__.items():
            if isinstance(value, (list, set)):
                value = value.copy()
            elif isinstance(value, dict):
                value = {
                    k: v.copy() if isinstance(v, (list, dict, set)) else v for k, v in value.items()
                }
            new.__dict__[name] = value
        return new

    # ------------------------------------------------------------------ flow
    @property
    def battle(self) -> bool:
        return self.stage == "battle"

    def _next_type(self) -> None:
        while self.type_queue:
            outcomes = self.type_outcomes(self.type_queue[0])
            if len(outcomes) > 1:
                self.stage = "types"
                return
            self._set_type(outcomes[0][0])
        if self.approvals:
            self.stage = "approval"
            return
        self._start_round()

    def type_outcomes(self, holder: str) -> tuple[tuple[str, float], ...]:
        return tuple(self.priors[holder].items())

    def _set_type(self, kind: str) -> None:
        holder = self.type_queue.pop(0)
        self.types[holder] = kind
        holder_types = self.params.types_of(holder)
        assert holder_types is not None
        side = self.params.holder_side(holder)
        seen_by: tuple[str, ...] = (
            ()
            if holder_types.unseen
            else (holder,)
            if holder_types.private
            else self.params.side_roles(side)
        )
        self.log(f"type:{holder}={kind}", visible_to=seen_by)

    def lead(self, side: str) -> str:
        """The side's current lead (it may have passed to a sponsor that went direct)."""

        return self.leads[side]

    def _start_round(self) -> None:
        p = self.params
        self.choices = {}
        self.offer = None
        self.warn_queue = [
            rl.role
            for rl in p.red_lines
            if rl.role not in self.withdrawn
            and rl.role not in self.warned
            and rl.near(self.control, p.step)
        ]
        if self.warn_queue:
            self.stage = "warn"
            return
        self._open_commitments()

    def _open_commitments(self) -> None:
        offers = self.params.offers
        self.commit_queue = [
            c.role
            for c in self.params.commitments
            if offers is not None
            and self.round >= offers.opens
            and c.role == self.lead(c.side)
            and c.role not in self.withdrawn
            and c.role not in self.committed
        ]
        if self.commit_queue:
            self.stage = "commit"
            return
        self._open_offers()

    def _open_offers(self) -> None:
        proposer, responder = self.proposer(), self.responder()
        if (
            proposer is not None
            and responder is not None
            and proposer not in self.withdrawn
            and responder not in self.withdrawn
        ):
            self.stage = "offer"
            return
        self._open_efforts()

    def _open_efforts(self) -> None:
        self.offer = None
        self.ratify_queue = []
        self.pending = [s.id for s in self.params.sponsors if s.id not in self.withdrawn]
        self.stage = "effort"

    def proposer(self) -> str | None:
        """The lead who may make an offer this round, if offers are open."""

        offers = self.params.offers
        if offers is None or self.round < offers.opens:
            return None
        return self.lead(offers.proposer_side(self.round))

    def responder(self) -> str | None:
        offers = self.params.offers
        if offers is None or self.round < offers.opens:
            return None
        return self.lead(other_side(offers.proposer_side(self.round)))

    def current_player(self) -> str:
        if self.label is not None:
            return TERMINAL
        stage = self.stage
        if stage in ("types", "battle"):
            return CHANCE
        if stage == "approval":
            return self.approvals[0]
        if stage == "warn":
            return self.warn_queue[0]
        if stage == "commit":
            return self.commit_queue[0]
        if stage == "offer":
            role = self.proposer()
            assert role is not None
            return role
        if stage == "respond":
            role = self.responder()
            assert role is not None
            return role
        if stage == "ratify":
            return self.ratify_queue[0]
        return self.pending[0]

    def is_terminal(self) -> bool:
        return self.label is not None

    def legal_actions(self) -> tuple[str, ...]:
        if self.label is not None or self.stage in ("types", "battle"):
            return ()
        stage = self.stage
        if stage == "approval":
            return ("approve", "refuse")
        if stage == "warn":
            return ("silent", "warn")
        if stage == "commit":
            return ("hold_off", "commit")
        if stage == "offer":
            assert self.params.offers is not None
            return ("no_offer", *self.params.offers.ids())
        if stage == "respond":
            return ("accept", "reject")
        if stage == "ratify":
            return ("ratify", "veto")
        role = self.pending[0]
        sponsor = self.params.sponsor(role)
        options = ["withdraw", "low", "high"]
        if sponsor.direct:
            options.append("direct")
        if role in self.leads.values() and self.params.offers is None:
            options.append("negotiate")  # with offers, settlements come through offers
        return tuple(options)

    # ------------------------------------------------------------- the contest
    def side_effort(
        self,
        side: str,
        choices: Mapping[str, str],
        kinds: Mapping[str, str] | None = None,
        deployed: Collection[str] | None = None,
        control: float | None = None,
    ) -> float:
        """Side effort under *choices*; *kinds* overrides the sides' (private) types,
        *deployed* the sponsors whose direct effort has fully arrived, and *control*
        the front (which sets the local strengths under ``momentum``)."""

        p = self.params
        arrived = self.deployed if deployed is None else deployed
        total = p.local_strength(side, self.control if control is None else control)
        for sponsor in p.sponsors:
            if sponsor.side != side or sponsor.id in self.withdrawn:
                continue
            choice = choices.get(sponsor.id, "withdraw")
            level = p.level_index(choice)
            if level >= 0:
                share = 1.0
                if choice == "direct" and sponsor.id not in arrived:
                    share = p.deployment_of(sponsor.id)
                share *= self._sponsor_strength(sponsor.id, kinds)
                total += sponsor.capacity * p.effort[level] * share
        side_types = p.types_of(side)
        if side_types is not None:
            kind = (kinds or {}).get(side, self.types.get(side))
            if kind is not None:
                total *= side_types.strength_of(kind)
        return total

    def win_probability(
        self,
        choices: Mapping[str, str],
        kinds: Mapping[str, str] | None = None,
        deployed: Collection[str] | None = None,
        control: float | None = None,
    ) -> float:
        g = self.params.gamma
        e_a = float(self.side_effort("a", choices, kinds, deployed, control) ** g)
        e_b = float(self.side_effort("b", choices, kinds, deployed, control) ** g)
        if e_a + e_b <= 0.0:
            return 0.5
        return e_a / (e_a + e_b)

    def move_probability(
        self,
        choices: Mapping[str, str],
        kinds: Mapping[str, str] | None = None,
        deployed: Collection[str] | None = None,
        control: float | None = None,
    ) -> float:
        """Chance that a battle moves the front (1 without ``mobility``)."""

        mobility = self.params.mobility
        if mobility is None:
            return 1.0
        total = self.side_effort("a", choices, kinds, deployed, control)
        total += self.side_effort("b", choices, kinds, deployed, control)
        return mobility.moves(total)

    def battle_outcomes(
        self,
        choices: Mapping[str, str],
        kinds: Mapping[str, str] | None = None,
        deployed: Collection[str] | None = None,
    ) -> tuple[tuple[str, float], ...]:
        p_a = self.win_probability(choices, kinds, deployed)
        move = self.move_probability(choices, kinds, deployed)
        raw = (("a_gains", p_a * move), ("b_gains", (1.0 - p_a) * move), ("hold", 1.0 - move))
        return tuple((a, q) for a, q in raw if q > 0.0)

    def _sponsor_strength(self, role: str, kinds: Mapping[str, str] | None) -> float:
        holder_types = dict(self.params.sponsor_types).get(role)
        if holder_types is None:
            return 1.0
        kind = (kinds or {}).get(role, self.types.get(role))
        return holder_types.strength_of(kind) if kind is not None else 1.0

    def cost_multiplier(self, side: str, kind: str | None = None) -> float:
        side_types = self.params.types_of(side)
        if side_types is None:
            return 1.0
        kind = kind if kind is not None else self.types.get(side)
        return side_types.cost_of(kind) if kind is not None else 1.0

    def role_cost_multiplier(self, role: str, kinds: Mapping[str, str] | None = None) -> float:
        """The side's cost multiplier times the sponsor's own (its stakes), under *kinds*
        (the actual types where *kinds* is silent)."""

        side = self.params.sponsor(role).side
        total = self.cost_multiplier(side, (kinds or {}).get(side))
        holder_types = dict(self.params.sponsor_types).get(role)
        if holder_types is not None:
            kind = (kinds or {}).get(role, self.types.get(role))
            if kind is not None:
                total *= holder_types.cost_of(kind)
        return total

    def proxy_multiplier(self, role: str, kinds: Mapping[str, str] | None = None) -> float:
        """The sponsor's own level's multiplier on its proxy gain, under *kinds* (the
        actual type where *kinds* is silent)."""

        holder_types = dict(self.params.sponsor_types).get(role)
        if holder_types is None:
            return 1.0
        kind = (kinds or {}).get(role, self.types.get(role))
        return holder_types.proxy_of(kind) if kind is not None else 1.0

    def proxy_gain(
        self, role: str, choices: Mapping[str, str], kinds: Mapping[str, str] | None = None
    ) -> float:
        """The sponsor's proxy gain for a round under *choices*: earned while it stays
        in the war and another sponsor on its side fights direct."""

        sponsor = self.params.sponsor(role)
        if sponsor.proxy_gain == 0.0 or self.params.level_index(choices.get(role, "")) < 0:
            return 0.0
        for other in self.params.sponsors:
            if (
                other.id != role
                and other.side == sponsor.side
                and other.id not in self.withdrawn
                and choices.get(other.id) == "direct"
            ):
                return sponsor.proxy_gain * self.proxy_multiplier(role, kinds)
        return 0.0

    def chance_outcomes(self) -> tuple[tuple[str, float], ...]:
        if self.label is not None:
            return ()
        if self.stage == "types":
            return self.type_outcomes(self.type_queue[0])
        if self.stage != "battle":
            return ()
        return self.battle_outcomes(self.choices)

    # ---------------------------------------------------------------- actions
    def apply_action(self, action: str) -> None:
        player = self.current_player()
        if player == TERMINAL:
            raise ConfigError("cannot act in a finished war")
        if player == CHANCE:
            check_legal(action, tuple(a for a, _ in self.chance_outcomes()))
            if self.stage == "types":
                self._set_type(action)
                self._next_type()
            else:
                self._resolve_battle(action)
        else:
            check_legal(action, self.legal_actions())
            stage = self.stage
            if stage == "approval":
                self._on_approval(player, action)
            elif stage == "warn":
                self._on_warn(player, action)
            elif stage == "commit":
                self._on_commit(player, action)
            elif stage == "offer":
                self._on_offer(player, action)
            elif stage == "respond":
                self._on_respond(player, action)
            elif stage == "ratify":
                self._on_ratify(player, action)
            else:
                self.choices[player] = action
                self.log(f"r{self.round}:self={action}", visible_to=[player])
                self.pending.pop(0)
                if not self.pending:
                    self._reveal()
        self._history.append(action)

    def _on_approval(self, role: str, action: str) -> None:
        self.log(f"approval:{role}={action}")
        self.approvals.pop(0)
        if action == "refuse":
            self.refused_by = role
            self.label = REFUSED
        elif not self.approvals:
            self._start_round()

    def _on_warn(self, role: str, action: str) -> None:
        self.log(f"r{self.round}:{role}={action}")
        self.warn_queue.pop(0)
        if action == "warn":
            self.warned.add(role)
        if not self.warn_queue:
            self._open_commitments()

    def _on_commit(self, role: str, action: str) -> None:
        self.log(f"r{self.round}:{role}={action}")
        self.commit_queue.pop(0)
        if action == "commit":
            self.committed.add(role)
        if not self.commit_queue:
            self._open_offers()

    def _on_offer(self, role: str, action: str) -> None:
        self.log(f"r{self.round}:{role}={action}")
        if action == "no_offer":
            self._open_efforts()
            return
        assert self.params.offers is not None
        self.offer = self.params.offers.ids().index(action)
        self.stage = "respond"

    def _on_respond(self, role: str, action: str) -> None:
        self.log(f"r{self.round}:{role}={action}")
        if action == "reject":
            self._open_efforts()
            return
        assert self.params.offers is not None
        self.ratify_queue = [
            c.role for c in self.params.offers.consent if c.role not in self.withdrawn
        ]
        if self.ratify_queue:
            self.stage = "ratify"
        else:
            self._agree()

    @property
    def terms(self) -> float | None:
        """The share under ratification: the offer accepted."""

        offers = self.params.offers
        if offers is None or self.offer is None:
            return None
        return offers.grid[self.offer]

    def _on_ratify(self, role: str, action: str) -> None:
        self.log(f"r{self.round}:{role}={action}")
        self.ratify_queue.pop(0)
        if action == "veto":
            assert self.params.offers is not None
            self.vetoed.add(role)
            self.cost[role] += self.params.offers.consent_of(role).veto_cost
            self._open_efforts()
        elif not self.ratify_queue:
            self._agree()

    def _agree(self) -> None:
        assert self.params.offers is not None and self.offer is not None
        self.share = self.params.offers.grid[self.offer]
        self.agreed = True
        self.label = "partition"
        self._charge_commitments()

    def _settle_at_line(self) -> None:
        self.label = "partition"
        if self.params.extended:
            self.share = self.control  # the line of contact
            self._charge_commitments()

    def _charge_commitments(self) -> None:
        """A committed lead settling for less than its terms pays its audience cost."""

        assert self.share is not None
        for item in self.params.commitments:
            if item.role in self.committed and item.breached(self.share):
                self.conceded.add(item.role)
                self.cost[item.role] += item.cost

    def _reveal(self) -> None:
        p = self.params
        for role, choice in self.choices.items():
            self.log(f"r{self.round}:{role}={choice}")
            level = p.level_index(choice)
            if level >= 0:
                sponsor = p.sponsor(role)
                self.cost[role] += (
                    sponsor.cost_scale * p.costs[level] * self.role_cost_multiplier(role)
                )
                self.weariness[role] += p.weariness[level]
                self.effort[role] += EFFORT[choice]
                if choice == "direct":
                    self.ever_direct.add(role)
                    self.cost[role] += sponsor.direct_cost
        for role in self.choices:
            self.cost[role] -= self.proxy_gain(role, self.choices)
        self._check_red_lines()
        for role, side in p.lead_on_entry:  # a belligerent that goes direct takes the lead
            if self.choices.get(role) == "direct" and self.leads[side] != role:
                self.leads[side] = role
                self.log(f"r{self.round}:lead:{side}={role}")
        self.last_choices = dict(self.choices)
        for role, choice in self.choices.items():
            if choice == "withdraw":
                self._withdraw(role)
        if self._side_gone("a") and self._side_gone("b"):
            self.label = "stalemate"
        elif self._side_gone("a"):
            self.label = "a_withdrew"
        elif self._side_gone("b"):
            self.label = "b_withdrew"
        elif all(self.choices.get(lead) == "negotiate" for lead in self.leads.values()):
            self._settle_at_line()
        else:
            self._to_battle()

    def _to_battle(self) -> None:
        self.stage = "battle"
        outcomes = self.chance_outcomes()
        if len(outcomes) == 1:
            self._resolve_battle(outcomes[0][0])

    def _check_red_lines(self) -> None:
        """A warner must go direct in the first round that starts with its line crossed."""

        for item in self.params.red_lines:
            role = item.role
            if role not in self.warned or role in self.bluffed or role in self.honored:
                continue
            if not item.crossed(self.control):
                continue
            if self.choices.get(role) == "direct":
                self.honored.add(role)
            else:
                self.bluffed.add(role)
                self.cost[role] += item.audience_cost

    def _withdraw(self, role: str) -> None:
        self.withdrawn.add(role)
        self.cost[role] += self.params.sponsor(role).withdraw_cost

    def _side_gone(self, side: str) -> bool:
        return all(s.id in self.withdrawn for s in self.params.sponsors if s.side == side)

    def _resolve_battle(self, action: str) -> None:
        p = self.params
        self.log(f"r{self.round}:battle={action}")
        self.last_battle = action
        self.deployed |= {role for role, choice in self.choices.items() if choice == "direct"}
        if action != "hold":
            delta = p.step if action == "a_gains" else -p.step
            self.control = min(1.0, max(0.0, self.control + delta))
        self.peak = max(self.peak, self.control)
        self.trough = min(self.trough, self.control)
        self.stage = "effort"
        if self.control >= p.victory_threshold - 1e-12:
            self.label = "a_victory"
            return
        if self.control <= 1.0 - p.victory_threshold + 1e-12:
            self.label = "b_victory"
            return
        self.round += 1
        if self.round >= p.max_rounds:
            self.label = "stalemate"
            return
        for sponsor in p.sponsors:
            if (
                sponsor.id not in self.withdrawn
                and self.weariness[sponsor.id] >= p.weariness_limit - 1e-12
            ):
                self._withdraw(sponsor.id)
                self.log(f"r{self.round}:{sponsor.id}=forced_out")
        if self._side_gone("a"):
            self.label = "a_withdrew"
        elif self._side_gone("b"):
            self.label = "b_withdrew"
        else:
            self._start_round()

    # --------------------------------------------------------------- payoffs
    def returns(self) -> dict[str, float]:
        if self.label is None:
            raise ConfigError("returns requested before the war ended")
        result = {}
        for s in self.params.sponsors:
            value = (
                s.partition_value(self.share) if self.label == "partition" else s.values[self.label]
            )
            result[s.id] = value - self.cost[s.id]
        return result

    def outcome(self) -> Outcome:
        if self.label is None:
            raise ConfigError("outcome requested before the war ended")
        p = self.params
        tags = [f"{role}:direct" for role in sorted(self.ever_direct)]
        tags += [f"{role}:withdrew" for role in sorted(self.withdrawn)]
        if self.agreed:
            tags.append("agreed")
        tags += [f"{role}:vetoed" for role in sorted(self.vetoed)]
        if self.refused_by is not None:
            tags.append(f"{self.refused_by}:refused")
        tags += [f"{role}:warned" for role in sorted(self.warned)]
        tags += [f"{role}:bluffed" for role in sorted(self.bluffed)]
        tags += [f"{role}:honored" for role in sorted(self.honored)]
        tags += [f"{role}:committed" for role in sorted(self.committed)]
        tags += [f"{role}:conceded" for role in sorted(self.conceded)]
        metrics: dict[str, float] = {"control": self.control, "rounds": float(self.round)}
        for sponsor in p.sponsors:
            metrics[f"cost:{sponsor.id}"] = self.cost[sponsor.id]
            metrics[f"weariness:{sponsor.id}"] = self.weariness[sponsor.id]
            # commitment: effort over the war relative to full effort for max_rounds
            metrics[f"commit:{sponsor.id}"] = min(1.0, self.effort[sponsor.id] / p.max_rounds)
            metrics[f"effort:{sponsor.id}"] = self.effort[sponsor.id] / max(1, self.round + 1)
        if p.extended:
            metrics["peak"] = self.peak
            metrics["trough"] = self.trough
            if self.share is not None:
                metrics["share"] = self.share
                metrics["share_change"] = self.share - p.initial_control
            for holder, kind in self.types.items():
                holder_types = p.types_of(holder)
                assert holder_types is not None
                metrics[f"strong:{holder}"] = 1.0 if holder_types.rank(kind) == 0 else 0.0
                metrics[f"rank:{holder}"] = float(holder_types.rank(kind))
        return Outcome(self.label, tuple(tags), metrics)

    def _strength_feature(self, side: str, kind: str | None) -> float:
        """1 for the strongest kind, 0 for the weakest, 0.5 when unknown."""

        side_types = self.params.types_of(side)
        if kind is None or side_types is None:
            return 0.5
        return 1.0 - side_types.rank(kind) / max(1, len(side_types.kinds) - 1)

    def observation(self, role: str) -> tuple[float, ...]:
        p = self.params
        sponsor = p.sponsor(role)
        own_control = self.control if sponsor.side == "a" else 1.0 - self.control
        last = self.last_choices
        e_a = self.side_effort("a", last) if last else 0.0
        e_b = self.side_effort("b", last) if last else 0.0
        own_e, other_e = (e_a, e_b) if sponsor.side == "a" else (e_b, e_a)
        other_lead = self.lead(other_side(sponsor.side))
        kind = self.types.get(sponsor.side)
        side_types = p.types_of(sponsor.side)
        if side_types is not None and side_types.unseen:
            kind = None  # its own side does not know it
        own_types = p.types_of(role) if role in dict(p.sponsor_types) else None
        own_kind = None if own_types is not None and own_types.unseen else self.types.get(role)
        offer = 0.0
        if self.offer is not None and p.offers is not None:
            share = p.offers.grid[self.offer]
            offer = share if sponsor.side == "a" else 1.0 - share
        other_warned = any(
            item.role in self.warned and p.sponsor(item.role).side != sponsor.side
            for item in p.red_lines
        )
        own_committed = any(
            item.role in self.committed and item.side == sponsor.side for item in p.commitments
        )
        other_committed = any(
            item.role in self.committed and item.side != sponsor.side for item in p.commitments
        )
        return (
            own_control,
            self.round / p.max_rounds,
            self.weariness[role] / max(p.weariness_limit, 1e-9),
            min(1.0, self.cost[role]),
            (p.level_index(last.get(role, "withdraw")) + 1) / 3.0,
            own_e / (own_e + other_e) if own_e + other_e > 0 else 0.5,
            1.0 if last.get(other_lead) == "negotiate" else 0.0,
            1.0 if role in self.leads.values() else 0.0,
            1.0 if sponsor.side == "a" else 0.0,
            len(self.withdrawn) / len(p.sponsors),
            self._strength_feature(sponsor.side, kind),
            STAGES.index(self.stage) / (len(STAGES) - 1),
            offer,
            1.0 if self.proposer() == role else 0.0,
            1.0 if self.last_battle == "hold" else 0.0,
            1.0 if other_warned else 0.0,
            1.0 if own_committed else 0.0,
            1.0 if other_committed else 0.0,
            self._strength_feature(role, own_kind),  # own sponsor level
        )


def _plain_only(params: AttritionParams, name: str) -> None:
    if params.extended:
        raise ConfigError(
            f"{name} plays plain wars only; use AttritionBeliefPolicy for wars with "
            "approval, offers, types, mobility, red lines or commitments"
        )


class AttritionHeuristicPolicy:
    """Myopic baseline: best one-round response to the others' last choices.

    The value of control is interpolated linearly between the two victory
    labels; the role negotiates or withdraws when that settles for more than
    the interpolated value of continuing.
    """

    def __init__(self, game: AttritionGame) -> None:
        _plain_only(game.params, "AttritionHeuristicPolicy")
        self.params = game.params

    def _value_of_control(self, sponsor: Sponsor, control: float) -> float:
        p = self.params
        low, high = 1.0 - p.victory_threshold, p.victory_threshold
        frac = (control - low) / max(high - low, 1e-9)
        v_a, v_b = sponsor.values["a_victory"], sponsor.values["b_victory"]
        return v_b + (v_a - v_b) * min(1.0, max(0.0, frac))

    def action_probabilities(self, state: GameState) -> dict[str, float]:
        if not isinstance(state, AttritionState):
            raise TypeError("AttritionHeuristicPolicy needs an AttritionState")
        p = self.params
        role = state.current_player()
        sponsor = p.sponsor(role)
        legal = state.legal_actions()
        assumed = {s.id: state.last_choices.get(s.id, "low") for s in p.sponsors}
        current_value = self._value_of_control(sponsor, state.control)
        scores: dict[str, float] = {}
        for action in legal:
            if action == "withdraw":
                label = "a_withdrew" if sponsor.side == "a" else "b_withdrew"
                scores[action] = sponsor.values[label] - sponsor.withdraw_cost
                continue
            if action == "negotiate":
                scores[action] = sponsor.values["partition"]
                continue
            trial = dict(assumed)
            trial[role] = action
            p_a = state.win_probability(trial)
            up = self._value_of_control(sponsor, state.control + p.step)
            down = self._value_of_control(sponsor, state.control - p.step)
            expected = p_a * up + (1.0 - p_a) * down
            level = p.level_index(action)
            scores[action] = (
                expected - sponsor.cost_scale * p.costs[level] + state.proxy_gain(role, trial)
            )
            if action == "direct":
                scores[action] -= sponsor.direct_cost
            if state.weariness[role] + p.weariness[level] >= p.weariness_limit:
                scores[action] -= abs(current_value)  # avoid being forced out
        best = max(legal, key=lambda a: (scores[a], -legal.index(a)))
        return {action: (1.0 if action == best else 0.0) for action in legal}


class AttritionLookaheadPolicy:
    """Baseline that looks ahead to the end of the war.

    For each effort level, with the other sponsors assumed to repeat their last
    choices, the value of continuing is computed exactly for the random walk of
    control: victory or defeat at the thresholds, stalemate at the round limit
    (or the withdrawal value, if that level's weariness would force the sponsor
    out first), minus the per-round cost of the effort. The role negotiates or
    withdraws when that settles for more. Unlike the one-round heuristic, a
    weak starting position does not by itself make settling look better.

    With ``temperature > 0`` the choice is a logit (quantal) response:
    ``P(action)`` is proportional to ``exp(score / temperature)``.
    """

    def __init__(self, game: AttritionGame, temperature: float = 0.0) -> None:
        if temperature < 0.0:
            raise ConfigError("temperature must be non-negative")
        _plain_only(game.params, "AttritionLookaheadPolicy")
        self.params = game.params
        self.temperature = temperature

    def continuation_value(self, state: AttritionState, role: str, action: str) -> float:
        """Value of fighting on at *action* while the others repeat their last choices."""

        p = self.params
        sponsor = p.sponsor(role)
        level = p.level_index(action)
        trial = {s.id: state.last_choices.get(s.id, "low") for s in p.sponsors}
        trial[role] = action
        p_a = state.win_probability(trial)
        cost = sponsor.cost_scale * p.costs[level] - state.proxy_gain(role, trial)
        if action == "direct":
            cost += sponsor.direct_cost
        rounds = max(0, p.max_rounds - state.round)
        end_value = sponsor.values["stalemate"]
        per_round = p.weariness[level]
        if per_round > 0.0:
            room = p.weariness_limit - state.weariness[role]
            until_out = max(1, math.ceil(room / per_round - 1e-12))
            if until_out < rounds:
                rounds = until_out
                label = "a_withdrew" if sponsor.side == "a" else "b_withdrew"
                end_value = sponsor.values[label] - sponsor.withdraw_cost
        low, high = 1.0 - p.victory_threshold, p.victory_threshold
        cells = max(2, round((high - low) / p.step))
        values = [end_value] * (cells + 1)
        values[0], values[cells] = sponsor.values["b_victory"], sponsor.values["a_victory"]
        for _ in range(rounds):
            values = [
                values[0],
                *(
                    p_a * values[i + 1] + (1.0 - p_a) * values[i - 1] - cost
                    for i in range(1, cells)
                ),
                values[cells],
            ]
        position = min(cells - 1, max(1, round((state.control - low) / p.step)))
        return values[position]

    def action_probabilities(self, state: GameState) -> dict[str, float]:
        if not isinstance(state, AttritionState):
            raise TypeError("AttritionLookaheadPolicy needs an AttritionState")
        p = self.params
        role = state.current_player()
        sponsor = p.sponsor(role)
        legal = state.legal_actions()
        scores: dict[str, float] = {}
        for action in legal:
            if action == "withdraw":
                label = "a_withdrew" if sponsor.side == "a" else "b_withdrew"
                scores[action] = sponsor.values[label] - sponsor.withdraw_cost
            elif action == "negotiate":
                scores[action] = sponsor.values["partition"]
            else:
                scores[action] = self.continuation_value(state, role, action)
        if self.temperature > 0.0:
            top = max(scores.values())
            weights = {a: math.exp((scores[a] - top) / self.temperature) for a in legal}
            total = sum(weights.values())
            return {a: w / total for a, w in weights.items()}
        best = max(legal, key=lambda a: (scores[a], -legal.index(a)))
        return {action: (1.0 if action == best else 0.0) for action in legal}


# ---------------------------------------------------------------------- build
def _roles(name: str, raw: Any, known: Mapping[str, str]) -> tuple[str, ...]:
    roles = tuple(identifier(name, r) for r in as_list(name, raw))
    if len(set(roles)) != len(roles):
        raise ConfigError(f"{name} lists a role twice")
    for role in roles:
        if role not in known:
            raise ConfigError(f"{name}: {role!r} is not a sponsor")
    return roles


def _offers(raw: Any, side_of: Mapping[str, str], leads: tuple[str, str]) -> OfferParams:
    data = as_mapping("attrition.offers", raw)
    reject_unknown("attrition.offers", data, ("grid", "proposer", "consent", "opens"))
    grid = tuple(
        probability("attrition.offers.grid", x)
        for x in as_list("attrition.offers.grid", require("attrition.offers", data, "grid"))
    )
    if not grid or any(not 0.0 < g < 1.0 for g in grid) or list(grid) != sorted(set(grid)):
        raise ConfigError("attrition.offers.grid must be increasing shares strictly inside (0, 1)")
    proposer = data.get("proposer", "alternate")
    if proposer not in PROPOSERS:
        raise ConfigError(f"attrition.offers.proposer must be one of {PROPOSERS}")
    consent = []
    for i, item in enumerate(as_list("attrition.offers.consent", data.get("consent", []))):
        name = f"attrition.offers.consent.{i}"
        if isinstance(item, str):
            item = {"role": item}
        item = as_mapping(name, item)
        reject_unknown(name, item, ("role", "veto_cost"))
        role = identifier(f"{name}.role", require(name, item, "role"))
        if role not in side_of:
            raise ConfigError(f"{name}: {role!r} is not a sponsor")
        if role in leads:
            raise ConfigError(f"{name}: a side lead cannot be a consent role")
        consent.append(Consent(role, nonnegative(f"{name}.veto_cost", item.get("veto_cost", 0.0))))
    if len({c.role for c in consent}) != len(consent):
        raise ConfigError("attrition.offers.consent lists a role twice")
    opens = data.get("opens", 0)
    if isinstance(opens, bool) or not isinstance(opens, int) or opens < 0:
        raise ConfigError("attrition.offers.opens must be a non-negative integer")
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


def _deployment(raw: Any, sponsors: list[Sponsor]) -> tuple[float, tuple[tuple[str, float], ...]]:
    """One share for every sponsor, or shares by role for direct sponsors (unlisted
    sponsors arrive in full)."""

    def share(name: str, value: Any) -> float:
        result = probability(name, value)
        if result <= 0.0:
            raise ConfigError(f"{name} must lie in (0, 1]")
        return result

    if not isinstance(raw, Mapping):
        return share("attrition.deployment", raw), ()
    direct = {s.id for s in sponsors if s.direct}
    roles = []
    for role, value in as_mapping("attrition.deployment", raw).items():
        if role not in direct:
            raise ConfigError(f"attrition.deployment: {role!r} is not a direct sponsor")
        roles.append((role, share(f"attrition.deployment.{role}", value)))
    return 1.0, tuple(roles)


def _red_lines(raw: Any, sponsors: list[Sponsor]) -> tuple[RedLine, ...]:
    by_id = {s.id: s for s in sponsors}
    result = []
    for role, spec in as_mapping("attrition.red_lines", raw).items():
        name = f"attrition.red_lines.{role}"
        if role not in by_id:
            raise ConfigError(f"{name}: {role!r} is not a sponsor")
        if not by_id[role].direct:
            raise ConfigError(f"{name}: the role must be able to go direct")
        data = as_mapping(name, spec)
        reject_unknown(name, data, ("above", "below", "audience_cost"))
        if ("above" in data) == ("below" in data):
            raise ConfigError(f"{name} needs exactly one of 'above' and 'below'")
        above = "above" in data
        line = probability(f"{name}.line", data["above" if above else "below"])
        cost = nonnegative(f"{name}.audience_cost", data.get("audience_cost", 0.0))
        result.append(RedLine(role, line, above, cost))
    return tuple(result)


def _commitments(
    raw: Any, side_of: Mapping[str, str], leads: Collection[str], offers: OfferParams | None
) -> tuple[Commitment, ...]:
    data = as_mapping("attrition.commitments", raw)
    if data and offers is None:
        raise ConfigError("attrition.commitments need offers")
    result = []
    for role, spec in data.items():
        name = f"attrition.commitments.{role}"
        if role not in leads:
            raise ConfigError(f"{name}: only a side lead (now or on entry) can commit")
        item = as_mapping(name, spec)
        reject_unknown(name, item, ("at_least", "audience_cost", "scale"))
        at_least = probability(f"{name}.at_least", require(name, item, "at_least"))
        if not 0.0 < at_least < 1.0:
            raise ConfigError(f"{name}.at_least must lie strictly inside (0, 1)")
        cost = nonnegative(f"{name}.audience_cost", item.get("audience_cost", 0.0))
        scale = nonnegative(f"{name}.scale", item.get("scale", 1.0))
        result.append(Commitment(role, side_of[role], at_least, cost, scale))
    return tuple(result)


def build(config: Mapping[str, Any]) -> AttritionGame:
    keys = (
        "sponsors",
        "leads",
        "local_strength",
        "initial_control",
        "step",
        "victory_threshold",
        "max_rounds",
        "gamma",
        "effort",
        "costs",
        "weariness",
        "weariness_limit",
        "approval",
        "offers",
        "types",
        "mobility",
        "red_lines",
        "deployment",
        "momentum",
        "commitments",
        "lead_on_entry",
    )
    reject_unknown("attrition", config, keys)
    approval_raw = config.get("approval", [])
    labels = (*OUTCOMES, REFUSED) if approval_raw else OUTCOMES
    sponsors = []
    for role_id, raw in as_mapping(
        "attrition.sponsors", require("attrition", config, "sponsors")
    ).items():
        name = f"attrition.sponsors.{role_id}"
        item = as_mapping(name, raw)
        reject_unknown(
            name,
            item,
            (
                "side",
                "capacity",
                "direct",
                "cost_scale",
                "values",
                "settlement",
                "proxy_gain",
                "withdraw_cost",
                "direct_cost",
            ),
        )
        side = require(name, item, "side")
        if side not in ("a", "b"):
            raise ConfigError(f"{name}.side must be 'a' or 'b'")
        settlement = None
        if "settlement" in item:
            spec = as_mapping(f"{name}.settlement", item["settlement"])
            reject_unknown(f"{name}.settlement", spec, ("base", "per_share"))
            settlement = (
                finite(f"{name}.settlement.base", require(name, spec, "base")),
                finite(f"{name}.settlement.per_share", require(name, spec, "per_share")),
            )
        sponsors.append(
            Sponsor(
                id=identifier(name, role_id),
                side=side,
                capacity=nonnegative(f"{name}.capacity", item.get("capacity", 1.0)),
                direct=bool(item.get("direct", False)),
                cost_scale=nonnegative(f"{name}.cost_scale", item.get("cost_scale", 1.0)),
                values=value_table(f"{name}.values", require(name, item, "values"), labels),
                settlement=settlement,
                proxy_gain=nonnegative(f"{name}.proxy_gain", item.get("proxy_gain", 0.0)),
                withdraw_cost=nonnegative(f"{name}.withdraw_cost", item.get("withdraw_cost", 0.0)),
                direct_cost=nonnegative(f"{name}.direct_cost", item.get("direct_cost", 0.0)),
            )
        )
    if not any(s.side == "a" for s in sponsors) or not any(s.side == "b" for s in sponsors):
        raise ConfigError("attrition needs at least one sponsor on each side")
    leads = as_mapping("attrition.leads", config.get("leads", {}))
    reject_unknown("attrition.leads", leads, ("a", "b"))
    side_of = {s.id: s.side for s in sponsors}
    lead_a = leads.get("a", next(s.id for s in sponsors if s.side == "a"))
    lead_b = leads.get("b", next(s.id for s in sponsors if s.side == "b"))
    if side_of.get(lead_a) != "a" or side_of.get(lead_b) != "b":
        raise ConfigError("attrition leads must be sponsors on their own side")
    local = as_mapping("attrition.local_strength", config.get("local_strength", {}))
    reject_unknown("attrition.local_strength", local, ("a", "b"))

    def triple(key: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
        data = as_mapping(f"attrition.{key}", config.get(key, {}))
        reject_unknown(f"attrition.{key}", data, ("low", "high", "direct"))
        return (
            nonnegative(f"attrition.{key}.low", data.get("low", default[0])),
            nonnegative(f"attrition.{key}.high", data.get("high", default[1])),
            nonnegative(f"attrition.{key}.direct", data.get("direct", default[2])),
        )

    threshold = probability("attrition.victory_threshold", config.get("victory_threshold", 0.85))
    if threshold <= 0.5:
        raise ConfigError("attrition.victory_threshold must exceed 0.5")
    initial = probability("attrition.initial_control", config.get("initial_control", 0.5))
    if not 1.0 - threshold < initial < threshold:
        raise ConfigError("attrition.initial_control must lie strictly between the thresholds")
    gamma = finite("attrition.gamma", config.get("gamma", 1.0))
    if gamma <= 0.0:
        raise ConfigError("attrition.gamma must be positive")
    step = probability("attrition.step", config.get("step", 0.1))
    if step <= 0.0:
        raise ConfigError("attrition.step must be positive")
    weariness_limit = nonnegative("attrition.weariness_limit", config.get("weariness_limit", 1.0))
    if weariness_limit <= 0.0:
        raise ConfigError("attrition.weariness_limit must be positive")
    approval = _roles("attrition.approval", approval_raw, side_of)
    offers = _offers(config["offers"], side_of, (lead_a, lead_b)) if "offers" in config else None
    types = as_mapping("attrition.types", config.get("types", {}))
    reject_unknown("attrition.types", types, ("a", "b", *side_of))
    sponsor_types = tuple(
        (role, _side_types(role, spec, sponsor=True))
        for role, spec in types.items()
        if role in side_of
    )
    lead_on_entry = []
    for role, side in as_mapping(
        "attrition.lead_on_entry", config.get("lead_on_entry", {})
    ).items():
        name = f"attrition.lead_on_entry.{role}"
        sponsor = next((sp for sp in sponsors if sp.id == role), None)
        if sponsor is None or not sponsor.direct or side != sponsor.side:
            raise ConfigError(f"{name}: needs a direct sponsor of side {side!r}")
        lead_on_entry.append((role, side))
    mobility = None
    if "mobility" in config:
        data = as_mapping("attrition.mobility", config["mobility"])
        reject_unknown("attrition.mobility", data, ("space", "sharpness"))
        space = finite("attrition.mobility.space", require("attrition.mobility", data, "space"))
        sharp = finite(
            "attrition.mobility.sharpness", require("attrition.mobility", data, "sharpness")
        )
        if space <= 0.0 or sharp <= 0.0:
            raise ConfigError("attrition.mobility.space and sharpness must be positive")
        mobility = Mobility(space, sharp)
    deployment, deployment_roles = _deployment(config.get("deployment", 1.0), sponsors)
    momentum = nonnegative("attrition.momentum", config.get("momentum", 0.0))
    params = AttritionParams(
        sponsors=tuple(sponsors),
        lead_a=lead_a,
        lead_b=lead_b,
        local_a=nonnegative("local_strength.a", local.get("a", 1.0)),
        local_b=nonnegative("local_strength.b", local.get("b", 1.0)),
        initial_control=initial,
        step=step,
        victory_threshold=threshold,
        max_rounds=positive_int("attrition.max_rounds", config.get("max_rounds", 12)),
        gamma=gamma,
        effort=triple("effort", (1.0, 2.0, 4.0)),
        costs=triple("costs", (0.02, 0.06, 0.15)),
        weariness=triple("weariness", (0.0, 0.02, 0.08)),
        weariness_limit=weariness_limit,
        approval=approval,
        offers=offers,
        a_types=_side_types("a", types["a"]) if "a" in types else None,
        b_types=_side_types("b", types["b"]) if "b" in types else None,
        sponsor_types=sponsor_types,
        lead_on_entry=tuple(lead_on_entry),
        mobility=mobility,
        red_lines=_red_lines(config.get("red_lines", {}), sponsors),
        deployment=deployment,
        deployment_roles=deployment_roles,
        momentum=momentum,
        commitments=_commitments(
            config.get("commitments", {}),
            side_of,
            {lead_a, lead_b, *(role for role, _ in lead_on_entry)},
            offers,
        ),
    )
    return AttritionGame(params)
