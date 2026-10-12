"""Alternating-offers bargaining under a risk of breakdown.

Two sides divide a stake. ``share`` always denotes side A's share. Each
period the proposer offers a share from a grid and the responder accepts or
rejects; every rejection risks a breakdown whose probability grows with the
period. If neither agreement nor breakdown occurs within the horizon, the
talks end in impasse (the status quo).

Optional extensions:

* **commitments** (Schelling): before talks each side may publicly commit to
  a position (A: a minimum share; B: a maximum A share). Agreeing beyond one's
  own commitment costs an audience cost, which makes the commitment credible.
* **consent roles**: third parties that ratify or veto an agreement.
* **explicit values**: a side may value an agreement as ``base + per_share * own
  share`` instead of its share of a common ``surplus``, value a vetoed deal
  apart from an impasse, and consent roles may value breakdown and impasse.
* **a contest between periods** (bargaining while fighting): after each
  rejected offer a public contest moves a ``position`` one ``step`` toward A
  (probability ``win_probability``) or toward B. Reaching 1 means B's position
  collapses and the game ends as an agreement at share 1 (tag
  ``<B>:collapsed``); reaching 0 ends it at share 0 (``<A>:collapsed``).
* **hidden resolve** (``types``): a side's per-period cost of carrying on is
  ``strong`` (low) or ``weak`` (high), drawn by chance at the start and seen
  only by that side. With types one side makes every offer (``proposers``).
* **a fixed proposer** (``proposers: A`` or ``B``) instead of alternating.

Outcome labels: ``agreement`` (metric ``share``), ``breakdown``, ``impasse``,
``vetoed``. Without a contest or types the game has perfect information, and
:func:`solve` computes its subgame-perfect equilibrium by dynamic programming over
(commitments, period). With either, :mod:`.bargaining_beliefs` computes a perfect
Bayesian equilibrium over beliefs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
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
    as_list,
    as_mapping,
    clamp,
    finite,
    identifier,
    nonnegative,
    positive_int,
    probability,
    reject_unknown,
    require,
)

NAME = "bargaining"
OUTCOMES = ("agreement", "breakdown", "impasse", "vetoed")
SIDE_ROLES = ("A", "B")
TOLERANCE = 1e-9
STRONG, WEAK = "strong", "weak"
TYPES = (STRONG, WEAK)
SELECTIONS = ("proposer_best", "pooling_first")


@dataclass(frozen=True, slots=True)
class SideParams(ImmutableParams):
    discount: float
    delay_cost: float
    breakdown_value: float
    impasse_value: float
    commitment: float | None  # A: minimum share; B: maximum A share
    commitment_cost: float
    agreement: tuple[float, float] | None  # (base, per_share); None: share of the surplus
    vetoed_value: float | None  # None: the impasse value


@dataclass(frozen=True, slots=True)
class ConsentParams(ImmutableParams):
    id: str
    base: float
    per_share: float
    veto_value: float
    breakdown_value: float
    impasse_value: float


@dataclass(frozen=True, slots=True)
class ContestParams(ImmutableParams):
    initial: float
    step: float
    win_probability: float  # P(the position moves toward A) in one period
    steps_up: int  # steps from the initial position to 1 (B collapses)
    steps_down: int  # steps from the initial position to 0 (A collapses)


@dataclass(frozen=True, slots=True)
class TypeParams(ImmutableParams):
    prior: float  # P(strong)
    strong_cost: float  # per-period delay cost of the strong (resolute) type
    weak_cost: float

    def cost(self, kind: str) -> float:
        return self.strong_cost if kind == STRONG else self.weak_cost


@dataclass(frozen=True, slots=True)
class BargainingParams(ImmutableParams):
    a_role: str
    b_role: str
    grid: tuple[float, ...]
    horizon: int
    first_proposer: str  # "A" or "B"
    surplus: float
    a: SideParams
    b: SideParams
    breakdown_base: float
    breakdown_growth: float
    breakdown_max: float
    consent: tuple[ConsentParams, ...]
    proposers: str = "alternate"  # "alternate", "A" or "B"
    contest: ContestParams | None = None
    a_types: TypeParams | None = None
    b_types: TypeParams | None = None
    selection: str = "proposer_best"  # equilibrium selection when a proposer is typed

    @property
    def extended(self) -> bool:
        """True when the game needs the belief-based solver."""

        return self.contest is not None or self.a_types is not None or self.b_types is not None

    def breakdown_risk(self, period: int) -> float:
        return clamp(self.breakdown_base + self.breakdown_growth * period, 0.0, self.breakdown_max)

    def proposer(self, period: int) -> str:
        if self.proposers != "alternate":
            return self.proposers
        first = self.first_proposer
        return first if period % 2 == 0 else ("B" if first == "A" else "A")

    def types_of(self, key: str) -> TypeParams | None:
        return self.a_types if key == "A" else self.b_types

    def delay_cost(self, key: str, kind: str | None) -> float:
        types = self.types_of(key)
        if types is None or kind is None:
            return self.side(key).delay_cost
        return types.cost(kind)

    def side(self, key: str) -> SideParams:
        return self.a if key == "A" else self.b

    def role_of(self, key: str) -> str:
        return self.a_role if key == "A" else self.b_role

    def offer_id(self, index: int) -> str:
        return f"offer_{round(self.grid[index] * 100):03d}"


class BargainingGame:
    name = NAME
    outcome_labels = OUTCOMES

    def __init__(self, params: BargainingParams) -> None:
        self.params = params

    @property
    def roles(self) -> tuple[str, ...]:
        p = self.params
        return (p.a_role, p.b_role, *(c.id for c in p.consent))

    def new_initial_state(self) -> BargainingState:
        return BargainingState(self.params)


def side_payoff(
    p: BargainingParams,
    key: str,
    label: str,
    *,
    share: float | None,
    period: int,
    agreed_period: int | None,
    committed: bool,
    cost: float,
) -> float:
    """One principal's payoff for a terminal *label*, with per-period delay *cost*."""

    side = p.side(key)
    if label == "agreement":
        assert share is not None and agreed_period is not None
        own_share = share if key == "A" else 1.0 - share
        if side.agreement is None:
            worth = p.surplus * own_share
        else:
            worth = side.agreement[0] + side.agreement[1] * own_share
        value = worth * side.discount**agreed_period
        value -= cost * agreed_period
        if committed and side.commitment is not None:
            violated = (
                share < side.commitment - TOLERANCE
                if key == "A"
                else share > side.commitment + TOLERANCE
            )
            if violated:
                value -= side.commitment_cost
        return value
    if label == "breakdown":
        return side.breakdown_value - cost * period
    if label == "impasse":
        return side.impasse_value - cost * period
    vetoed = side.impasse_value if side.vetoed_value is None else side.vetoed_value
    return vetoed - cost * (agreed_period or 0)


class BargainingState(LoggedState):
    def __init__(self, params: BargainingParams) -> None:
        super().__init__((params.a_role, params.b_role, *(c.id for c in params.consent)))
        self.params = params
        self.phase = "type_a"
        self.committed = {"A": False, "B": False}
        self.types: dict[str, str | None] = {"A": None, "B": None}
        self.period = 0
        self.position = 0  # contest steps from the initial position (+ toward A)
        self.last_contest = 0  # +1 A gained, -1 B gained, 0 none yet
        self.offer: int | None = None
        self.consent_index = 0
        self.label: str | None = None
        self.share: float | None = None
        self.agreed_period: int | None = None
        self.tags: list[str] = []
        # public record for belief tracking: (kind, ...) in order
        self.events: list[tuple[Any, ...]] = []
        self._advance_setup()

    def _advance_setup(self) -> None:
        p = self.params
        if self.phase == "type_a" and p.a_types is None:
            self.phase = "type_b"
        if self.phase == "type_b" and p.b_types is None:
            self.phase = "commit_a"
        if self.phase == "commit_a" and p.a.commitment is None:
            self.phase = "commit_b"
        if self.phase == "commit_b" and p.b.commitment is None:
            self.phase = "propose"

    def current_player(self) -> str:
        p = self.params
        if self.label is not None:
            return TERMINAL
        if self.phase in ("type_a", "type_b", "breakdown", "contest"):
            return CHANCE
        if self.phase == "commit_a":
            return p.a_role
        if self.phase == "commit_b":
            return p.b_role
        if self.phase == "propose":
            return p.role_of(p.proposer(self.period))
        if self.phase == "respond":
            responder = "B" if p.proposer(self.period) == "A" else "A"
            return p.role_of(responder)
        return p.consent[self.consent_index].id

    def is_terminal(self) -> bool:
        return self.label is not None

    def legal_actions(self) -> tuple[str, ...]:
        if self.label is not None or self.current_player() == CHANCE:
            return ()
        if self.phase in ("commit_a", "commit_b"):
            return ("no_commitment", "commit")
        if self.phase == "propose":
            return tuple(self.params.offer_id(i) for i in range(len(self.params.grid)))
        if self.phase == "respond":
            return ("accept", "reject")
        return ("ratify", "veto")

    def chance_outcomes(self) -> tuple[tuple[str, float], ...]:
        if self.label is not None:
            return ()
        p = self.params
        if self.phase in ("type_a", "type_b"):
            types = p.a_types if self.phase == "type_a" else p.b_types
            assert types is not None
            pairs = ((STRONG, types.prior), (WEAK, 1.0 - types.prior))
        elif self.phase == "breakdown":
            risk = p.breakdown_risk(self.period)
            pairs = (("breakdown", risk), ("talks_continue", 1.0 - risk))
        elif self.phase == "contest":
            assert p.contest is not None
            q = p.contest.win_probability
            pairs = (("a_gains", q), ("b_gains", 1.0 - q))
        else:
            return ()
        return tuple((a, prob) for a, prob in pairs if prob > 0.0)

    def apply_action(self, action: str) -> None:
        p = self.params
        player = self.current_player()
        legal = (
            tuple(a for a, _ in self.chance_outcomes())
            if player == CHANCE
            else self.legal_actions()
        )
        check_legal(action, legal)
        if self.phase in ("type_a", "type_b"):
            key = "A" if self.phase == "type_a" else "B"
            self.types[key] = action
            self.log(f"type:{key}={action}", visible_to=(p.role_of(key),))
            self.phase = "type_b" if key == "A" else "commit_a"
            self._advance_setup()
        elif self.phase in ("commit_a", "commit_b"):
            key = "A" if self.phase == "commit_a" else "B"
            self.committed[key] = action == "commit"
            self.log(f"commit:{key}={action}")
            self.events.append(("commit", key, action == "commit"))
            if action == "commit":
                self.tags.append(f"{p.role_of(key)}:committed")
            self.phase = "commit_b" if self.phase == "commit_a" else "propose"
            self._advance_setup()
        elif self.phase == "propose":
            self.offer = self.legal_actions().index(action)
            self.log(f"t{self.period}:offer={action}")
            self.events.append(("offer", self.period, self.offer))
            self.phase = "respond"
        elif self.phase == "respond":
            self.log(f"t{self.period}:{action}")
            self.events.append(("respond", self.period, action == "accept"))
            if action == "accept":
                assert self.offer is not None
                self.share = p.grid[self.offer]
                self.agreed_period = self.period
                self._after_agreement()
            else:
                self._after_rejection()
        elif self.phase == "breakdown":
            self.log(f"t{self.period}:{action}")
            if action == "breakdown":
                self.label = "breakdown"
            else:
                self._after_talks_continue()
        elif self.phase == "contest":
            self.log(f"t{self.period}:{action}")
            self._move(1 if action == "a_gains" else -1)
        else:
            consent = p.consent[self.consent_index]
            self.log(f"consent:{consent.id}={action}")
            if action == "veto":
                self.label = "vetoed"
                self.tags.append(f"{consent.id}:vetoed")
            else:
                self.consent_index += 1
                if self.consent_index >= len(p.consent):
                    self.label = "agreement"
        self._history.append(action)

    def _after_agreement(self) -> None:
        if self.params.consent:
            self.phase = "consent"
            self.consent_index = 0
        else:
            self.label = "agreement"

    def _after_rejection(self) -> None:
        risk = self.params.breakdown_risk(self.period)
        if risk >= 1.0:
            self.label = "breakdown"
        elif risk <= 0.0:
            self._after_talks_continue()
        else:
            self.phase = "breakdown"

    def _after_talks_continue(self) -> None:
        contest = self.params.contest
        if contest is None:
            self._next_period()
        elif contest.win_probability >= 1.0:
            self._move(1)
        elif contest.win_probability <= 0.0:
            self._move(-1)
        else:
            self.phase = "contest"

    def _move(self, direction: int) -> None:
        """Resolve one contest; the losing side's position may collapse."""

        contest = self.params.contest
        assert contest is not None
        self.position += direction
        self.last_contest = direction
        self.events.append(("contest", self.period, direction))
        if self.position >= contest.steps_up or self.position <= -contest.steps_down:
            winner_a = self.position >= contest.steps_up
            self.period += 1
            self.share = 1.0 if winner_a else 0.0
            self.agreed_period = self.period
            loser = self.params.b_role if winner_a else self.params.a_role
            self.tags.append(f"{loser}:collapsed")
            self.label = "agreement"
        else:
            self._next_period()

    def _next_period(self) -> None:
        self.period += 1
        self.offer = None
        if self.period >= self.params.horizon:
            self.label = "impasse"
        else:
            self.phase = "propose"

    # ----------------------------------------------------------------- payoffs
    def _side_payoff(self, key: str) -> float:
        assert self.label is not None
        return side_payoff(
            self.params,
            key,
            self.label,
            share=self.share,
            period=self.period,
            agreed_period=self.agreed_period,
            committed=self.committed[key],
            cost=self.params.delay_cost(key, self.types[key]),
        )

    def returns(self) -> dict[str, float]:
        if self.label is None:
            raise ConfigError("returns requested before bargaining ended")
        p = self.params
        result = {p.a_role: self._side_payoff("A"), p.b_role: self._side_payoff("B")}
        for consent in p.consent:
            if self.label == "agreement":
                assert self.share is not None
                result[consent.id] = consent.base + consent.per_share * self.share
            elif self.label == "vetoed":
                result[consent.id] = consent.veto_value
            elif self.label == "breakdown":
                result[consent.id] = consent.breakdown_value
            else:
                result[consent.id] = consent.impasse_value
        return result

    def outcome(self) -> Outcome:
        if self.label is None:
            raise ConfigError("outcome requested before bargaining ended")
        p = self.params
        tags = list(self.tags)
        metrics = {"period": float(self.period), "horizon": float(p.horizon)}
        if self.share is not None:
            metrics["share"] = self.share
        metrics[f"commit:{p.a_role}"] = 1.0  # both principals stake their position
        metrics[f"commit:{p.b_role}"] = 1.0
        if p.contest is not None:
            metrics["position"] = p.contest.initial + p.contest.step * self.position
        for key in ("A", "B"):
            kind = self.types[key]
            if kind is not None:  # revealed once the game is over
                metrics[f"strong:{p.role_of(key)}"] = 1.0 if kind == STRONG else 0.0
            side = p.side(key)
            if self.label == "agreement" and self.committed[key] and side.commitment is not None:
                assert self.share is not None
                violated = (
                    self.share < side.commitment - TOLERANCE
                    if key == "A"
                    else self.share > side.commitment + TOLERANCE
                )
                if violated:
                    tags.append(f"{p.role_of(key)}:broke_commitment")
        return Outcome(self.label, tuple(tags), metrics)

    def observation(self, role: str) -> tuple[float, ...]:
        p = self.params
        key = "A" if role == p.a_role else "B" if role == p.b_role else ""
        offer_share = p.grid[self.offer] if self.offer is not None else 0.0
        own_offer = offer_share if key != "B" else 1.0 - offer_share
        kind = self.types.get(key) if key else None
        position = 0.5
        if p.contest is not None:
            position = p.contest.initial + p.contest.step * self.position
        return (
            self.period / p.horizon,
            p.breakdown_risk(self.period),
            own_offer,
            1.0 if self.phase == "respond" else 0.0,
            1.0 if key and p.role_of(p.proposer(self.period)) == role else 0.0,
            1.0 if key and self.committed.get(key, False) else 0.0,
            1.0 if key and self.committed.get("B" if key == "A" else "A", False) else 0.0,
            1.0 if self.phase == "consent" else 0.0,
            (self.share or 0.0),
            1.0 if key == "A" else 0.0,
            position if key != "B" else 1.0 - position,
            float(self.last_contest) if key != "B" else -float(self.last_contest),
            0.5 if kind is None else (1.0 if kind == STRONG else 0.0),
        )


# ---------------------------------------------------------------- equilibrium
@dataclass(frozen=True, slots=True)
class BargainingSolution(ImmutableParams):
    """Subgame-perfect choices for every (commitments, period) and offer."""

    commit: dict[tuple[str, tuple[bool, ...]], str]
    offer: dict[tuple[bool, bool, int], int]
    accept: dict[tuple[bool, bool, int, int], bool]
    ratify: dict[tuple[int, int], bool]
    values: dict[tuple[bool, bool], tuple[float, float]]


def _agreement_values(
    p: BargainingParams, committed: tuple[bool, bool], period: int, offer: int
) -> tuple[float, float, bool]:
    """Side payoffs if *offer* is accepted at *period*, and whether consent ratifies."""

    state = BargainingState(p)
    state.committed = {"A": committed[0], "B": committed[1]}
    state.share = p.grid[offer]
    state.agreed_period = period
    state.period = period
    ratified = all(consent_ratifies(c, p.grid[offer]) for c in p.consent)
    state.label = "agreement" if ratified else "vetoed"
    return state._side_payoff("A"), state._side_payoff("B"), ratified


def consent_ratifies(consent: ConsentParams, share: float) -> bool:
    return consent.base + consent.per_share * share >= consent.veto_value - TOLERANCE


def solve(game: BargainingGame) -> BargainingSolution:
    """Backward induction over (commitments, period); ties favor accepting and the
    proposer's earliest grid offer."""

    p = game.params
    if p.extended:
        raise ValueError("bargaining with types or a contest: use bargaining_beliefs")
    offer_choice: dict[tuple[bool, bool, int], int] = {}
    accept_choice: dict[tuple[bool, bool, int, int], bool] = {}
    values: dict[tuple[bool, bool], tuple[float, float]] = {}
    combos = [(ca, cb) for ca in (False, True) for cb in (False, True)]
    for ca, cb in combos:
        committed = (ca, cb)
        dummy = BargainingState(p)
        dummy.committed = {"A": ca, "B": cb}
        dummy.label = "impasse"
        dummy.period = p.horizon
        continuation = (dummy._side_payoff("A"), dummy._side_payoff("B"))
        for period in range(p.horizon - 1, -1, -1):
            risk = p.breakdown_risk(period)
            dummy.label, dummy.period = "breakdown", period
            breakdown = (dummy._side_payoff("A"), dummy._side_payoff("B"))
            reject_value = tuple(
                risk * breakdown[i] + (1.0 - risk) * continuation[i] for i in range(2)
            )
            proposer = p.proposer(period)
            prop_i = 0 if proposer == "A" else 1
            resp_i = 1 - prop_i
            best_offer, best_value = 0, None
            best_pair: tuple[float, float] = (0.0, 0.0)
            for offer in range(len(p.grid)):
                va, vb, _ = _agreement_values(p, committed, period, offer)
                accept_pair = (va, vb)
                accepts = accept_pair[resp_i] >= reject_value[resp_i] - TOLERANCE
                accept_choice[(ca, cb, period, offer)] = accepts
                pair = accept_pair if accepts else (reject_value[0], reject_value[1])
                if best_value is None or pair[prop_i] > best_value + TOLERANCE:
                    best_offer, best_value, best_pair = offer, pair[prop_i], pair
            offer_choice[(ca, cb, period)] = best_offer
            continuation = best_pair
        values[committed] = continuation
    commit_choice: dict[tuple[str, tuple[bool, ...]], str] = {}
    if p.b.commitment is not None:
        for ca in (False, True):
            commit_choice[("B", (ca,))] = (
                "commit"
                if values[(ca, True)][1] > values[(ca, False)][1] + TOLERANCE
                else "no_commitment"
            )

    def b_choice(ca: bool) -> bool:
        if p.b.commitment is None:
            return False
        return commit_choice[("B", (ca,))] == "commit"

    if p.a.commitment is not None:
        a_commit = (
            values[(True, b_choice(True))][0] > values[(False, b_choice(False))][0] + TOLERANCE
        )
        commit_choice[("A", ())] = "commit" if a_commit else "no_commitment"
    ratify = {
        (i, offer): consent_ratifies(c, p.grid[offer])
        for i, c in enumerate(p.consent)
        for offer in range(len(p.grid))
    }
    return BargainingSolution(commit_choice, offer_choice, accept_choice, ratify, values)


@dataclass(frozen=True, slots=True)
class LogitSolution(ImmutableParams):
    """Logit (quantal response) choice probabilities by (commitments, period)."""

    commit: dict[tuple[str, tuple[bool, ...]], float]  # P(commit)
    offer: dict[tuple[bool, bool, int], tuple[float, ...]]  # P(each offer)
    accept: dict[tuple[bool, bool, int, int], float]
    ratify: dict[tuple[int, int], float]
    values: dict[tuple[bool, bool], tuple[float, float]]


def _logistic(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def solve_logit(game: BargainingGame, temperature: float) -> LogitSolution:
    """Agent-form logit QRE of a perfect-information bargaining game: backward
    induction in which every choice is a logit response at *temperature*."""

    p = game.params
    if p.extended:
        raise ValueError("bargaining with types or a contest: use bargaining_beliefs")
    tau = temperature
    ratify = {
        (i, x): _logistic((c.base + c.per_share * p.grid[x] - c.veto_value) / tau)
        for i, c in enumerate(p.consent)
        for x in range(len(p.grid))
    }
    offer_probs: dict[tuple[bool, bool, int], tuple[float, ...]] = {}
    accept_probs: dict[tuple[bool, bool, int, int], float] = {}
    values: dict[tuple[bool, bool], tuple[float, float]] = {}
    for ca in (False, True):
        for cb in (False, True):
            dummy = BargainingState(p)
            dummy.committed = {"A": ca, "B": cb}
            dummy.label, dummy.period = "impasse", p.horizon
            continuation = (dummy._side_payoff("A"), dummy._side_payoff("B"))
            for period in range(p.horizon - 1, -1, -1):
                risk = p.breakdown_risk(period)
                dummy.label, dummy.period = "breakdown", period
                breakdown = (dummy._side_payoff("A"), dummy._side_payoff("B"))
                reject = tuple(risk * breakdown[i] + (1 - risk) * continuation[i] for i in range(2))
                prop_i = 0 if p.proposer(period) == "A" else 1
                pairs = []
                for x in range(len(p.grid)):
                    state = BargainingState(p)
                    state.committed = {"A": ca, "B": cb}
                    state.share, state.agreed_period, state.period = p.grid[x], period, period
                    state.label = "agreement"
                    agreed = (state._side_payoff("A"), state._side_payoff("B"))
                    state.label = "vetoed"
                    vetoed = (state._side_payoff("A"), state._side_payoff("B"))
                    ratified = 1.0
                    for i in range(len(p.consent)):
                        ratified *= ratify[(i, x)]
                    accepted = tuple(
                        ratified * agreed[i] + (1 - ratified) * vetoed[i] for i in range(2)
                    )
                    a = _logistic((accepted[1 - prop_i] - reject[1 - prop_i]) / tau)
                    accept_probs[(ca, cb, period, x)] = a
                    pairs.append(tuple(a * accepted[i] + (1 - a) * reject[i] for i in range(2)))
                top = max(pair[prop_i] for pair in pairs)
                weights = [math.exp((pair[prop_i] - top) / tau) for pair in pairs]
                total = sum(weights)
                probs = tuple(w / total for w in weights)
                offer_probs[(ca, cb, period)] = probs
                continuation = (
                    sum(q * pair[0] for q, pair in zip(probs, pairs, strict=True)),
                    sum(q * pair[1] for q, pair in zip(probs, pairs, strict=True)),
                )
            values[(ca, cb)] = continuation
    commit: dict[tuple[str, tuple[bool, ...]], float] = {}

    def b_commits(ca: bool) -> float:
        if p.b.commitment is None:
            return 0.0
        return commit[("B", (ca,))]

    if p.b.commitment is not None:
        for ca in (False, True):
            commit[("B", (ca,))] = _logistic((values[(ca, True)][1] - values[(ca, False)][1]) / tau)

    def a_value(ca: bool) -> float:
        q = b_commits(ca)
        return q * values[(ca, True)][0] + (1 - q) * values[(ca, False)][0]

    if p.a.commitment is not None:
        commit[("A", ())] = _logistic((a_value(True) - a_value(False)) / tau)
    return LogitSolution(commit, offer_probs, accept_probs, ratify, values)


class BargainingEquilibriumPolicy:
    """Plays the equilibrium in live states: the subgame-perfect solution from
    :func:`solve` (or its logit version when ``temperature > 0``), or, with types
    or a contest, the perfect Bayesian one from :mod:`.bargaining_beliefs`
    (which ignores the temperature)."""

    def __init__(self, game: BargainingGame, temperature: float = 0.0) -> None:
        if temperature < 0.0:
            raise ConfigError("temperature must be non-negative")
        self.beliefs: Any = None
        self.logit: LogitSolution | None = None
        if game.params.extended:
            from .bargaining_beliefs import BeliefPolicy  # deferred: circular import

            self.beliefs = BeliefPolicy(game.params)
        elif temperature > 0.0:
            self.logit = solve_logit(game, temperature)
        else:
            self.solution = solve(game)

    def _logit_probabilities(self, state: BargainingState) -> dict[str, float]:
        s = self.logit
        assert s is not None
        ca, cb = state.committed["A"], state.committed["B"]
        if state.phase in ("commit_a", "commit_b"):
            key = ("A", ()) if state.phase == "commit_a" else ("B", (ca,))
            q = s.commit.get(key, 0.0)
            return {"no_commitment": 1.0 - q, "commit": q}
        if state.phase == "propose":
            probs = s.offer[(ca, cb, state.period)]
            return dict(zip(state.legal_actions(), probs, strict=True))
        assert state.offer is not None
        if state.phase == "respond":
            a = s.accept[(ca, cb, state.period, state.offer)]
            return {"accept": a, "reject": 1.0 - a}
        r = s.ratify[(state.consent_index, state.offer)]
        return {"ratify": r, "veto": 1.0 - r}

    def action_probabilities(self, state: GameState) -> dict[str, float]:
        if not isinstance(state, BargainingState):
            raise TypeError("BargainingEquilibriumPolicy needs a BargainingState")
        if self.beliefs is not None:
            return dict(self.beliefs.action_probabilities(state))
        if self.logit is not None:
            return self._logit_probabilities(state)
        s = self.solution
        ca, cb = state.committed["A"], state.committed["B"]
        if state.phase == "commit_a":
            choice = s.commit.get(("A", ()), "no_commitment")
        elif state.phase == "commit_b":
            choice = s.commit.get(("B", (ca,)), "no_commitment")
        elif state.phase == "propose":
            choice = state.legal_actions()[s.offer[(ca, cb, state.period)]]
        elif state.phase == "respond":
            assert state.offer is not None
            choice = "accept" if s.accept[(ca, cb, state.period, state.offer)] else "reject"
        else:
            assert state.offer is not None
            choice = "ratify" if s.ratify[(state.consent_index, state.offer)] else "veto"
        return {action: (1.0 if action == choice else 0.0) for action in state.legal_actions()}


# ------------------------------------------------------------------ configuration
def _side(name: str, data: Any, key: str) -> SideParams:
    mapping = as_mapping(name, data)
    reject_unknown(
        name,
        mapping,
        (
            "discount",
            "delay_cost",
            "breakdown_value",
            "impasse_value",
            "commitment",
            "commitment_cost",
            "agreement",
            "vetoed_value",
        ),
    )
    discount = finite(f"{name}.discount", mapping.get("discount", 1.0))
    if not 0.0 < discount <= 1.0:
        raise ConfigError(f"{name}.discount must lie in (0, 1]")
    commitment = mapping.get("commitment")
    agreement = None
    if "agreement" in mapping:
        raw = as_mapping(f"{name}.agreement", mapping["agreement"])
        reject_unknown(f"{name}.agreement", raw, ("base", "per_share"))
        agreement = (
            finite(f"{name}.agreement.base", raw.get("base", 0.0)),
            finite(f"{name}.agreement.per_share", raw.get("per_share", 0.0)),
        )
    vetoed = mapping.get("vetoed_value")
    return SideParams(
        discount=discount,
        delay_cost=nonnegative(f"{name}.delay_cost", mapping.get("delay_cost", 0.0)),
        breakdown_value=finite(
            f"{name}.breakdown_value", require(name, mapping, "breakdown_value")
        ),
        impasse_value=finite(f"{name}.impasse_value", require(name, mapping, "impasse_value")),
        commitment=None if commitment is None else probability(f"{name}.commitment", commitment),
        commitment_cost=nonnegative(f"{name}.commitment_cost", mapping.get("commitment_cost", 0.0)),
        agreement=agreement,
        vetoed_value=None if vetoed is None else finite(f"{name}.vetoed_value", vetoed),
    )


def _contest(data: Any) -> ContestParams:
    name = "bargaining.contest"
    mapping = as_mapping(name, data)
    reject_unknown(name, mapping, ("initial", "step", "win_probability"))
    initial = probability(f"{name}.initial", mapping.get("initial", 0.5))
    step = probability(f"{name}.step", require(name, mapping, "step"))
    if not 0.0 < initial < 1.0 or step <= 0.0:
        raise ConfigError(f"{name}: initial must lie in (0, 1) and step must be positive")
    up, down = (1.0 - initial) / step, initial / step
    if abs(up - round(up)) > 1e-6 or abs(down - round(down)) > 1e-6:
        raise ConfigError(f"{name}: 0 and 1 must be whole steps from the initial position")
    return ContestParams(
        initial=initial,
        step=step,
        win_probability=probability(
            f"{name}.win_probability", require(name, mapping, "win_probability")
        ),
        steps_up=round(up),
        steps_down=round(down),
    )


def _types(name: str, data: Any) -> TypeParams:
    mapping = as_mapping(name, data)
    reject_unknown(name, mapping, ("prior", "delay_cost"))
    prior = as_mapping(f"{name}.prior", require(name, mapping, "prior"))
    reject_unknown(f"{name}.prior", prior, TYPES)
    strong = probability(f"{name}.prior.strong", require(f"{name}.prior", prior, STRONG))
    weak = probability(f"{name}.prior.weak", prior.get(WEAK, 1.0 - strong))
    if abs(strong + weak - 1.0) > 1e-9:
        raise ConfigError(f"{name}.prior must sum to 1")
    costs = as_mapping(f"{name}.delay_cost", require(name, mapping, "delay_cost"))
    reject_unknown(f"{name}.delay_cost", costs, TYPES)
    return TypeParams(
        prior=strong,
        strong_cost=nonnegative(f"{name}.delay_cost.strong", require(name, costs, STRONG)),
        weak_cost=nonnegative(f"{name}.delay_cost.weak", require(name, costs, WEAK)),
    )


def build(config: Mapping[str, Any]) -> BargainingGame:
    reject_unknown(
        "bargaining",
        config,
        (
            "roles",
            "grid",
            "horizon",
            "first_proposer",
            "proposers",
            "surplus",
            "sides",
            "breakdown",
            "consent",
            "contest",
            "types",
            "selection",
        ),
    )
    roles = as_mapping("bargaining.roles", require("bargaining", config, "roles"))
    reject_unknown("bargaining.roles", roles, SIDE_ROLES)
    a_role = identifier("bargaining.roles.A", require("bargaining.roles", roles, "A"))
    b_role = identifier("bargaining.roles.B", require("bargaining.roles", roles, "B"))
    grid = tuple(
        probability("bargaining.grid", x)
        for x in as_list("bargaining.grid", config.get("grid", [0.0, 0.25, 0.5, 0.75, 1.0]))
    )
    if len(grid) < 2 or any(b <= a for a, b in pairwise(grid)):
        raise ConfigError("bargaining.grid must be strictly increasing with at least 2 points")
    if len(grid) > 11:
        raise ConfigError("bargaining.grid may have at most 11 points")
    first = config.get("first_proposer", "A")
    if first not in SIDE_ROLES:
        raise ConfigError("bargaining.first_proposer must be 'A' or 'B'")
    sides = as_mapping("bargaining.sides", require("bargaining", config, "sides"))
    reject_unknown("bargaining.sides", sides, SIDE_ROLES)
    breakdown = as_mapping("bargaining.breakdown", config.get("breakdown", {}))
    reject_unknown("bargaining.breakdown", breakdown, ("base", "growth", "max"))
    consent = []
    for i, raw in enumerate(as_list("bargaining.consent", config.get("consent", []))):
        item = as_mapping(f"bargaining.consent[{i}]", raw)
        reject_unknown(
            f"bargaining.consent[{i}]",
            item,
            ("role", "base", "per_share", "veto_value", "breakdown_value", "impasse_value"),
        )
        consent.append(
            ConsentParams(
                id=identifier(f"bargaining.consent[{i}].role", require("consent", item, "role")),
                base=finite("consent.base", item.get("base", 0.0)),
                per_share=finite("consent.per_share", item.get("per_share", 0.0)),
                veto_value=finite("consent.veto_value", item.get("veto_value", 0.0)),
                breakdown_value=finite("consent.breakdown_value", item.get("breakdown_value", 0.0)),
                impasse_value=finite("consent.impasse_value", item.get("impasse_value", 0.0)),
            )
        )
    ids = [a_role, b_role, *(c.id for c in consent)]
    if len(set(ids)) != len(ids):
        raise ConfigError("bargaining role ids must be unique")
    proposers = config.get("proposers", "alternate")
    if proposers not in ("alternate", *SIDE_ROLES):
        raise ConfigError("bargaining.proposers must be 'alternate', 'A' or 'B'")
    types = as_mapping("bargaining.types", config.get("types", {}))
    reject_unknown("bargaining.types", types, SIDE_ROLES)
    a_types = _types("bargaining.types.A", types["A"]) if "A" in types else None
    b_types = _types("bargaining.types.B", types["B"]) if "B" in types else None
    if (a_types or b_types) and proposers == "alternate":
        raise ConfigError("bargaining.types needs a fixed proposer (proposers: A or B)")
    selection = config.get("selection", "proposer_best")
    if selection not in SELECTIONS:
        raise ConfigError(f"bargaining.selection must be one of {SELECTIONS}")
    contest = _contest(config["contest"]) if "contest" in config else None
    params = BargainingParams(
        a_role=a_role,
        b_role=b_role,
        grid=grid,
        horizon=positive_int("bargaining.horizon", config.get("horizon", 4)),
        first_proposer=first,
        surplus=nonnegative("bargaining.surplus", config.get("surplus", 1.0)),
        a=_side("bargaining.sides.A", require("bargaining.sides", sides, "A"), "A"),
        b=_side("bargaining.sides.B", require("bargaining.sides", sides, "B"), "B"),
        breakdown_base=probability("bargaining.breakdown.base", breakdown.get("base", 0.0)),
        breakdown_growth=nonnegative("bargaining.breakdown.growth", breakdown.get("growth", 0.0)),
        breakdown_max=probability("bargaining.breakdown.max", breakdown.get("max", 1.0)),
        consent=tuple(consent),
        proposers=proposers,
        contest=contest,
        a_types=a_types,
        b_types=b_types,
        selection=selection,
    )
    return BargainingGame(params)
