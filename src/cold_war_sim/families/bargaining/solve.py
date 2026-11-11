"""Equilibrium of perfect-information bargaining by backward induction.

Without types or a contest the game has perfect information: :func:`solve` gives its
subgame-perfect equilibrium and :func:`solve_logit` its agent-form logit QRE, both by
dynamic programming over (commitments, period).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ...games.protocol import GameState, ImmutableParams
from ...records import Record
from ..common import ConfigError, logistic
from .game import (
    TOLERANCE,
    BargainingGame,
    BargainingParams,
    BargainingState,
    ConsentParams,
    side_payoff,
)

Pair = tuple[float, float]  # (side A's value, side B's value)
COMMITMENTS = [(ca, cb) for ca in (False, True) for cb in (False, True)]


class BargainingSolution(ImmutableParams, Record):
    """The subgame-perfect choice at every decision, and the value of each subgame.

    Keys: ``commit`` by (side, commitments already made); ``offer`` by (A committed,
    B committed, period); ``accept`` by those and the offer; ``ratify`` by (consent
    index, offer); ``values`` by (A committed, B committed).
    """

    def __init__(
        self,
        commit: dict[tuple[str, tuple[bool, ...]], str],
        offer: dict[tuple[bool, bool, int], int],
        accept: dict[tuple[bool, bool, int, int], bool],
        ratify: dict[tuple[int, int], bool],
        values: dict[tuple[bool, bool], Pair],
    ) -> None:
        self.commit = commit
        self.offer = offer
        self.accept = accept
        self.ratify = ratify
        self.values = values


class LogitSolution(ImmutableParams, Record):
    """The logit choice probabilities at every decision, keyed as :class:`BargainingSolution`."""

    def __init__(
        self,
        commit: dict[tuple[str, tuple[bool, ...]], float],
        offer: dict[tuple[bool, bool, int], tuple[float, ...]],
        accept: dict[tuple[bool, bool, int, int], float],
        ratify: dict[tuple[int, int], float],
        values: dict[tuple[bool, bool], Pair],
    ) -> None:
        self.commit = commit  # P(commit)
        self.offer = offer  # P(each offer)
        self.accept = accept  # P(accept)
        self.ratify = ratify  # P(ratify)
        self.values = values


def _payoffs(
    p: BargainingParams,
    committed: tuple[bool, bool],
    label: str,
    *,
    period: int,
    share: float | None = None,
) -> Pair:
    """Both principals' payoffs at a terminal *label* reached in *period*."""

    agreed = period if share is not None else None
    a, b = (
        side_payoff(
            p,
            key,
            label,
            share=share,
            period=period,
            agreed_period=agreed,
            committed=committed[i],
            cost=p.delay_cost(key, None),
        )
        for i, key in enumerate(("A", "B"))
    )
    return a, b


def _mix(q: float, first: Pair, second: Pair) -> Pair:
    """The pair ``q x first + (1 - q) x second``."""

    return (q * first[0] + (1 - q) * second[0], q * first[1] + (1 - q) * second[1])


def ratify_choices(consent: Sequence[ConsentParams], share: float) -> tuple[bool, ...]:
    """Each ratifier's subgame-perfect choice at *share*, in the order they are asked,
    solved backward: once a later ratifier will veto, an earlier one's approval
    changes nothing, so it ratifies rather than pay its own veto cost (with no cost it is
    indifferent and keeps its own comparison)."""

    choices: list[bool] = []
    later = True  # every later ratifier approves
    for c in reversed(consent):
        agree = c.base + c.per_share * share
        own_veto = c.veto_value - c.veto_cost
        ratifies = True if not later and c.veto_cost > 0.0 else agree >= own_veto - TOLERANCE
        choices.append(ratifies)
        later = later and ratifies
    return tuple(reversed(choices))


def ratify_probabilities(
    consent: Sequence[ConsentParams], share: float, temperature: float
) -> tuple[float, ...]:
    """Each ratifier's logit choice at *share* when asked, solved backward: its
    approval yields agreement only if every later ratifier approves too, and a vetoed
    ending otherwise."""

    probs: list[float] = []
    later = 1.0  # probability that every later ratifier approves
    for c in reversed(consent):
        agree = c.base + c.per_share * share
        ratify = later * agree + (1.0 - later) * c.veto_value
        q = logistic((ratify - (c.veto_value - c.veto_cost)) / temperature)
        probs.append(q)
        later *= q
    return tuple(reversed(probs))


def solve(game: BargainingGame) -> BargainingSolution:
    """The subgame-perfect equilibrium, by backward induction over (commitments, period).

    A responder accepts when indifferent, and a proposer makes its earliest best offer.
    """

    p = game.params
    if p.extended:
        raise ValueError("bargaining with types or a contest: use bargaining.beliefs")
    offer_choice: dict[tuple[bool, bool, int], int] = {}
    accept_choice: dict[tuple[bool, bool, int, int], bool] = {}
    values: dict[tuple[bool, bool], Pair] = {}
    ratifies = [ratify_choices(p.consent, share) for share in p.grid]
    for committed in COMMITMENTS:
        ca, cb = committed
        continuation = _payoffs(p, committed, "impasse", period=p.horizon)
        for period in range(p.horizon - 1, -1, -1):
            risk = p.breakdown_risk(period)
            breakdown = _payoffs(p, committed, "breakdown", period=period)
            reject_value = (
                risk * breakdown[0] + (1.0 - risk) * continuation[0],
                risk * breakdown[1] + (1.0 - risk) * continuation[1],
            )
            prop_i = 0 if p.proposer(period) == "A" else 1
            resp_i = 1 - prop_i
            best_offer, best_value = 0, None
            best_pair: Pair = (0.0, 0.0)
            for offer, share in enumerate(p.grid):
                label = "agreement" if all(ratifies[offer]) else "vetoed"
                accept_pair = _payoffs(p, committed, label, period=period, share=share)
                accepts = accept_pair[resp_i] >= reject_value[resp_i] - TOLERANCE
                accept_choice[(ca, cb, period, offer)] = accepts
                pair = accept_pair if accepts else reject_value
                if best_value is None or pair[prop_i] > best_value + TOLERANCE:
                    best_offer, best_value, best_pair = offer, pair[prop_i], pair
            offer_choice[(ca, cb, period)] = best_offer
            continuation = best_pair
        values[committed] = continuation
    ratify = {
        (i, offer): choice
        for offer in range(len(p.grid))
        for i, choice in enumerate(ratifies[offer])
    }
    commit_choice = _commitments(p, values)
    return BargainingSolution(commit_choice, offer_choice, accept_choice, ratify, values)


def _commitments(
    p: BargainingParams, values: dict[tuple[bool, bool], Pair]
) -> dict[tuple[str, tuple[bool, ...]], str]:
    """The commitment choices: B best responds to A's choice, and A anticipates it."""

    commit: dict[tuple[str, tuple[bool, ...]], str] = {}

    def b_commits(ca: bool) -> bool:
        if p.b.commitment is None:
            return False
        return values[(ca, True)][1] > values[(ca, False)][1] + TOLERANCE

    if p.b.commitment is not None:
        for ca in (False, True):
            commit[("B", (ca,))] = "commit" if b_commits(ca) else "no_commitment"
    if p.a.commitment is not None:
        a_commits = (
            values[(True, b_commits(True))][0] > values[(False, b_commits(False))][0] + TOLERANCE
        )
        commit[("A", ())] = "commit" if a_commits else "no_commitment"
    return commit


def solve_logit(game: BargainingGame, temperature: float) -> LogitSolution:
    """The agent-form logit QRE: backward induction in which every choice is a logit
    response at *temperature*."""

    p = game.params
    if p.extended:
        raise ValueError("bargaining with types or a contest: use bargaining.beliefs")
    tau = temperature
    ratify = {
        (i, x): q
        for x in range(len(p.grid))
        for i, q in enumerate(ratify_probabilities(p.consent, p.grid[x], tau))
    }
    offer_probs: dict[tuple[bool, bool, int], tuple[float, ...]] = {}
    accept_probs: dict[tuple[bool, bool, int, int], float] = {}
    values: dict[tuple[bool, bool], Pair] = {}
    for committed in COMMITMENTS:
        ca, cb = committed
        continuation = _payoffs(p, committed, "impasse", period=p.horizon)
        for period in range(p.horizon - 1, -1, -1):
            risk = p.breakdown_risk(period)
            breakdown = _payoffs(p, committed, "breakdown", period=period)
            reject = _mix(risk, breakdown, continuation)
            prop_i = 0 if p.proposer(period) == "A" else 1
            pairs = []
            for x, share in enumerate(p.grid):
                agreed = _payoffs(p, committed, "agreement", period=period, share=share)
                vetoed = _payoffs(p, committed, "vetoed", period=period, share=share)
                ratified = 1.0
                for i in range(len(p.consent)):
                    ratified *= ratify[(i, x)]
                accepted = _mix(ratified, agreed, vetoed)
                a = logistic((accepted[1 - prop_i] - reject[1 - prop_i]) / tau)
                accept_probs[(ca, cb, period, x)] = a
                pairs.append(_mix(a, accepted, reject))
            top = max(pair[prop_i] for pair in pairs)
            weights = [math.exp((pair[prop_i] - top) / tau) for pair in pairs]
            total = sum(weights)
            probs = tuple(w / total for w in weights)
            offer_probs[(ca, cb, period)] = probs
            continuation = (
                sum(q * pair[0] for q, pair in zip(probs, pairs, strict=True)),
                sum(q * pair[1] for q, pair in zip(probs, pairs, strict=True)),
            )
        values[committed] = continuation
    commit: dict[tuple[str, tuple[bool, ...]], float] = {}
    if p.b.commitment is not None:
        for ca in (False, True):
            commit[("B", (ca,))] = logistic((values[(ca, True)][1] - values[(ca, False)][1]) / tau)

    def a_value(ca: bool) -> float:
        q = commit[("B", (ca,))] if p.b.commitment is not None else 0.0
        return q * values[(ca, True)][0] + (1 - q) * values[(ca, False)][0]

    if p.a.commitment is not None:
        commit[("A", ())] = logistic((a_value(True) - a_value(False)) / tau)
    return LogitSolution(commit, offer_probs, accept_probs, ratify, values)


class BargainingEquilibriumPolicy:
    """The equilibrium as a policy.

    With perfect information it plays the subgame-perfect solution (:func:`solve`), or
    its logit version when ``temperature > 0``. With types or a contest it plays the
    perfect Bayesian equilibrium of :mod:`.beliefs`, which ignores the temperature.
    """

    def __init__(self, game: BargainingGame, temperature: float = 0.0) -> None:
        if temperature < 0.0:
            raise ConfigError("temperature must be non-negative")
        self.beliefs: Any = None
        self.logit: LogitSolution | None = None
        if game.params.extended:
            from .beliefs import BeliefPolicy

            self.beliefs = BeliefPolicy(game.params)
        elif temperature > 0.0:
            self.logit = solve_logit(game, temperature)
        else:
            self.solution = solve(game)

    def action_probabilities(self, state: GameState) -> dict[str, float]:
        if not isinstance(state, BargainingState):
            raise TypeError("BargainingEquilibriumPolicy needs a BargainingState")
        if self.beliefs is not None:
            return dict(self.beliefs.action_probabilities(state))
        if self.logit is not None:
            return self._logit_probabilities(state, self.logit)
        choice = self._choice(state)
        return {action: (1.0 if action == choice else 0.0) for action in state.legal_actions()}

    def _choice(self, state: BargainingState) -> str:
        """The subgame-perfect action at *state*."""

        s = self.solution
        ca, cb = state.committed["A"], state.committed["B"]
        if state.phase == "commit_a":
            return s.commit.get(("A", ()), "no_commitment")
        if state.phase == "commit_b":
            return s.commit.get(("B", (ca,)), "no_commitment")
        if state.phase == "propose":
            return state.legal_actions()[s.offer[(ca, cb, state.period)]]
        assert state.offer is not None
        if state.phase == "respond":
            return "accept" if s.accept[(ca, cb, state.period, state.offer)] else "reject"
        return "ratify" if s.ratify[(state.consent_index, state.offer)] else "veto"

    @staticmethod
    def _logit_probabilities(state: BargainingState, s: LogitSolution) -> dict[str, float]:
        """The logit choice probabilities at *state*."""

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
