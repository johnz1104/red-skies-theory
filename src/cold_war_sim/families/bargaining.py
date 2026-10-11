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

Outcome labels: ``agreement`` (metric ``share``), ``breakdown``, ``impasse``,
``vetoed``. The game has perfect information, and :func:`solve` computes its
subgame-perfect equilibrium by dynamic programming over (commitments, period).
"""

from __future__ import annotations

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

    def breakdown_risk(self, period: int) -> float:
        return clamp(self.breakdown_base + self.breakdown_growth * period, 0.0, self.breakdown_max)

    def proposer(self, period: int) -> str:
        first = self.first_proposer
        return first if period % 2 == 0 else ("B" if first == "A" else "A")

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


class BargainingState(LoggedState):
    def __init__(self, params: BargainingParams) -> None:
        super().__init__((params.a_role, params.b_role, *(c.id for c in params.consent)))
        self.params = params
        self.phase = "commit_a"
        self.committed = {"A": False, "B": False}
        self.period = 0
        self.offer: int | None = None
        self.consent_index = 0
        self.label: str | None = None
        self.share: float | None = None
        self.agreed_period: int | None = None
        self.tags: list[str] = []
        self._skip_commitments()

    def _skip_commitments(self) -> None:
        if self.phase == "commit_a" and self.params.a.commitment is None:
            self.phase = "commit_b"
        if self.phase == "commit_b" and self.params.b.commitment is None:
            self.phase = "propose"

    def current_player(self) -> str:
        p = self.params
        if self.label is not None:
            return TERMINAL
        if self.phase == "commit_a":
            return p.a_role
        if self.phase == "commit_b":
            return p.b_role
        if self.phase == "propose":
            return p.role_of(p.proposer(self.period))
        if self.phase == "respond":
            responder = "B" if p.proposer(self.period) == "A" else "A"
            return p.role_of(responder)
        if self.phase == "breakdown":
            return CHANCE
        return p.consent[self.consent_index].id

    def is_terminal(self) -> bool:
        return self.label is not None

    def legal_actions(self) -> tuple[str, ...]:
        if self.label is not None or self.phase == "breakdown":
            return ()
        if self.phase in ("commit_a", "commit_b"):
            return ("no_commitment", "commit")
        if self.phase == "propose":
            return tuple(self.params.offer_id(i) for i in range(len(self.params.grid)))
        if self.phase == "respond":
            return ("accept", "reject")
        return ("ratify", "veto")

    def chance_outcomes(self) -> tuple[tuple[str, float], ...]:
        if self.phase != "breakdown" or self.label is not None:
            return ()
        risk = self.params.breakdown_risk(self.period)
        pairs = (("breakdown", risk), ("talks_continue", 1.0 - risk))
        return tuple((a, p) for a, p in pairs if p > 0.0)

    def apply_action(self, action: str) -> None:
        p = self.params
        player = self.current_player()
        legal = (
            tuple(a for a, _ in self.chance_outcomes())
            if player == CHANCE
            else self.legal_actions()
        )
        check_legal(action, legal)
        if self.phase in ("commit_a", "commit_b"):
            key = "A" if self.phase == "commit_a" else "B"
            self.committed[key] = action == "commit"
            self.log(f"commit:{key}={action}")
            if action == "commit":
                self.tags.append(f"{p.role_of(key)}:committed")
            self.phase = "commit_b" if self.phase == "commit_a" else "propose"
            self._skip_commitments()
        elif self.phase == "propose":
            self.offer = self.legal_actions().index(action)
            self.log(f"t{self.period}:offer={action}")
            self.phase = "respond"
        elif self.phase == "respond":
            self.log(f"t{self.period}:{action}")
            if action == "accept":
                assert self.offer is not None
                self.share = p.grid[self.offer]
                self.agreed_period = self.period
                self._after_agreement()
            else:
                self.phase = "breakdown"
                self._advance_after_rejection_if_certain()
        elif self.phase == "breakdown":
            self.log(f"t{self.period}:{action}")
            if action == "breakdown":
                self.label = "breakdown"
            else:
                self._next_period()
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

    def _advance_after_rejection_if_certain(self) -> None:
        if self.params.breakdown_risk(self.period) <= 0.0:
            self._next_period()
        elif self.params.breakdown_risk(self.period) >= 1.0:
            self.label = "breakdown"

    def _next_period(self) -> None:
        self.period += 1
        self.offer = None
        if self.period >= self.params.horizon:
            self.label = "impasse"
        else:
            self.phase = "propose"

    # ----------------------------------------------------------------- payoffs
    def _side_payoff(self, key: str) -> float:
        p = self.params
        side = p.side(key)
        label = self.label
        if label == "agreement":
            assert self.share is not None and self.agreed_period is not None
            own_share = self.share if key == "A" else 1.0 - self.share
            if side.agreement is None:
                worth = p.surplus * own_share
            else:
                worth = side.agreement[0] + side.agreement[1] * own_share
            value = worth * side.discount**self.agreed_period
            value -= side.delay_cost * self.agreed_period
            if self.committed[key] and side.commitment is not None:
                violated = (
                    self.share < side.commitment - TOLERANCE
                    if key == "A"
                    else self.share > side.commitment + TOLERANCE
                )
                if violated:
                    value -= side.commitment_cost
            return value
        if label == "breakdown":
            return side.breakdown_value - side.delay_cost * self.period
        if label == "impasse":
            return side.impasse_value - side.delay_cost * self.period
        vetoed = side.impasse_value if side.vetoed_value is None else side.vetoed_value
        return vetoed - side.delay_cost * (self.agreed_period or 0)

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
        tags = list(self.tags)
        metrics = {"period": float(self.period), "horizon": float(self.params.horizon)}
        if self.share is not None:
            metrics["share"] = self.share
        metrics[f"commit:{self.params.a_role}"] = 1.0  # both principals stake their position
        metrics[f"commit:{self.params.b_role}"] = 1.0
        for key in ("A", "B"):
            side = self.params.side(key)
            if self.label == "agreement" and self.committed[key] and side.commitment is not None:
                assert self.share is not None
                violated = (
                    self.share < side.commitment - TOLERANCE
                    if key == "A"
                    else self.share > side.commitment + TOLERANCE
                )
                if violated:
                    tags.append(f"{self.params.role_of(key)}:broke_commitment")
        return Outcome(self.label, tuple(tags), metrics)

    def observation(self, role: str) -> tuple[float, ...]:
        p = self.params
        key = "A" if role == p.a_role else "B" if role == p.b_role else ""
        offer_share = p.grid[self.offer] if self.offer is not None else 0.0
        own_offer = offer_share if key != "B" else 1.0 - offer_share
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
    ratified = all(_consent_ratifies(c, p.grid[offer]) for c in p.consent)
    state.label = "agreement" if ratified else "vetoed"
    return state._side_payoff("A"), state._side_payoff("B"), ratified


def _consent_ratifies(consent: ConsentParams, share: float) -> bool:
    return consent.base + consent.per_share * share >= consent.veto_value - TOLERANCE


def solve(game: BargainingGame) -> BargainingSolution:
    """Backward induction over (commitments, period); ties favor accepting and the
    proposer's earliest grid offer."""

    p = game.params
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
        (i, offer): _consent_ratifies(c, p.grid[offer])
        for i, c in enumerate(p.consent)
        for offer in range(len(p.grid))
    }
    return BargainingSolution(commit_choice, offer_choice, accept_choice, ratify, values)


class BargainingEquilibriumPolicy:
    """Plays the subgame-perfect solution from :func:`solve` in live states."""

    def __init__(self, game: BargainingGame) -> None:
        self.solution = solve(game)

    def action_probabilities(self, state: GameState) -> dict[str, float]:
        if not isinstance(state, BargainingState):
            raise TypeError("BargainingEquilibriumPolicy needs a BargainingState")
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


def build(config: Mapping[str, Any]) -> BargainingGame:
    reject_unknown(
        "bargaining",
        config,
        ("roles", "grid", "horizon", "first_proposer", "surplus", "sides", "breakdown", "consent"),
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
    )
    return BargainingGame(params)
