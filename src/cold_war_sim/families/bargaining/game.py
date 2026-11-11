"""Bargaining parameters, choices and payoffs, including breakdown and consent."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise
from typing import Any

from ...games.protocol import CHANCE, TERMINAL, ImmutableParams, LoggedState, Outcome, check_legal
from ...records import Record
from ..common import ConfigError, Section, clamp, identifier, probability

NAME = "bargaining"
OUTCOMES = ("agreement", "breakdown", "impasse", "vetoed")
SIDE_ROLES = ("A", "B")
TOLERANCE = 1e-9
STRONG, WEAK = "strong", "weak"
TYPES = (STRONG, WEAK)
SELECTIONS = ("proposer_best", "pooling_first")


def other(key: str) -> str:
    """The side facing side *key*."""

    return "B" if key == "A" else "A"


class SideParams(ImmutableParams, Record):
    """One side's payoff parameters. ``share`` always means side A's share.

    ``discount``         per-period discount factor on an agreement's worth, in (0, 1] [1].
    ``delay_cost``       cost per period of talks [0].
    ``breakdown_value``  its payoff if talks break down.
    ``impasse_value``    its payoff if the horizon is reached (the status quo).
    ``vetoed_value``     its payoff if a consent role vetoes [the impasse value].
    ``agreement``        ``{base, per_share}``: an agreement is worth
                         ``base + per_share x own share`` [own share of ``surplus``].
    ``commitment``       the position it may commit to before talks (A: a minimum share;
                         B: a maximum share for A) (Schelling 1960).
    ``commitment_cost``  the audience cost of agreeing beyond its commitment [0].
    """

    def __init__(
        self,
        discount: float,
        delay_cost: float,
        breakdown_value: float,
        impasse_value: float,
        commitment: float | None,
        commitment_cost: float,
        agreement: tuple[float, float] | None,
        vetoed_value: float | None,
    ) -> None:
        self.discount = discount
        self.delay_cost = delay_cost
        self.breakdown_value = breakdown_value
        self.impasse_value = impasse_value
        self.commitment = commitment
        self.commitment_cost = commitment_cost
        self.agreement = agreement  # (base, per_share); None: its share of the surplus
        self.vetoed_value = vetoed_value  # None: the impasse value

    @classmethod
    def from_config(cls, s: Section) -> SideParams:
        """Read one side's agreement value, delay costs and possible commitment."""

        with s:
            discount = s.number("discount", 1.0)
            if not 0.0 < discount <= 1.0:
                raise ConfigError(f"{s.at('discount')} must lie in (0, 1]")
            agreement = None
            if s.has("agreement"):
                with s.section("agreement") as a:
                    agreement = (a.number("base", 0.0), a.number("per_share", 0.0))
            return cls(
                discount=discount,
                delay_cost=s.nonnegative("delay_cost", 0.0),
                breakdown_value=s.number("breakdown_value"),
                impasse_value=s.number("impasse_value"),
                commitment=None
                if s.get("commitment", None) is None
                else s.probability("commitment"),
                commitment_cost=s.nonnegative("commitment_cost", 0.0),
                agreement=agreement,
                vetoed_value=None
                if s.get("vetoed_value", None) is None
                else s.number("vetoed_value"),
            )


class ConsentParams(ImmutableParams, Record):
    """A third party that ratifies or vetoes an agreement.

    ``role``             its id.
    ``base``, ``per_share``   an agreement at share ``x`` is worth ``base + per_share x``.
    ``veto_value``       its payoff from a vetoed ending, whoever vetoed.
    ``veto_cost``        paid only by the role that casts the veto.
    ``breakdown_value``, ``impasse_value``   its payoffs from those endings.
    """

    def __init__(
        self,
        id: str,
        base: float,
        per_share: float,
        veto_value: float,
        breakdown_value: float,
        impasse_value: float,
        veto_cost: float = 0.0,
    ) -> None:
        self.id = id
        self.base = base
        self.per_share = per_share
        self.veto_value = veto_value
        self.breakdown_value = breakdown_value
        self.impasse_value = impasse_value
        self.veto_cost = veto_cost

    @classmethod
    def from_config(cls, s: Section) -> ConsentParams:
        """Read a third party's agreement values, veto payoff and veto cost."""

        with s:
            return cls(
                id=identifier(s.at("role"), s.get("role")),
                base=s.number("base", 0.0),
                per_share=s.number("per_share", 0.0),
                veto_value=s.number("veto_value", 0.0),
                breakdown_value=s.number("breakdown_value", 0.0),
                impasse_value=s.number("impasse_value", 0.0),
                veto_cost=s.nonnegative("veto_cost", 0.0),
            )


class ContestParams(ImmutableParams, Record):
    """A contest between periods (bargaining while fighting).

    After each rejected offer a public ``position`` moves one ``step`` toward A with
    probability ``win_probability``, else toward B, from ``initial``. At 1 side B has
    collapsed and the game ends as an agreement at share 1; at 0 side A has, at share 0.
    """

    def __init__(
        self, initial: float, step: float, win_probability: float, steps_up: int, steps_down: int
    ) -> None:
        self.initial = initial
        self.step = step
        self.win_probability = win_probability
        self.steps_up = steps_up  # steps from the initial position to 1
        self.steps_down = steps_down  # steps from the initial position to 0

    @classmethod
    def from_config(cls, s: Section) -> ContestParams:
        """Read the starting front and the probability and size of each movement."""

        with s:
            initial = s.probability("initial", 0.5)
            step = s.probability("step")
            if not 0.0 < initial < 1.0 or step <= 0.0:
                raise ConfigError(f"{s.name}: initial must lie in (0, 1) and step must be positive")
            up, down = (1.0 - initial) / step, initial / step
            if abs(up - round(up)) > 1e-6 or abs(down - round(down)) > 1e-6:
                raise ConfigError(
                    f"{s.name}: 0 and 1 must be whole steps from the initial position"
                )
            return cls(
                initial=initial,
                step=step,
                win_probability=s.probability("win_probability"),
                steps_up=round(up),
                steps_down=round(down),
            )


class TypeParams(ImmutableParams, Record):
    """A side's hidden resolve: its per-period delay cost is private information.

    ``prior``       ``{strong, weak}``: P(strong) and P(weak).
    ``delay_cost``  ``{strong, weak}``: the strong (resolute) type's cost is the lower.
    """

    def __init__(self, prior: float, strong_cost: float, weak_cost: float) -> None:
        self.prior = prior  # P(strong)
        self.strong_cost = strong_cost
        self.weak_cost = weak_cost

    def cost(self, kind: str) -> float:
        """The per-period delay cost of type *kind*."""

        return self.strong_cost if kind == STRONG else self.weak_cost

    @classmethod
    def from_config(cls, s: Section) -> TypeParams:
        """Read type probabilities and delay costs, with lower costs for the strong type."""

        with s:
            with s.section("prior") as prior:
                strong = prior.probability(STRONG)
                weak = prior.probability(WEAK, 1.0 - strong)
            if abs(strong + weak - 1.0) > 1e-9:
                raise ConfigError(f"{s.at('prior')} must sum to 1")
            with s.section("delay_cost") as costs:
                return cls(
                    prior=strong,
                    strong_cost=costs.nonnegative(STRONG),
                    weak_cost=costs.nonnegative(WEAK),
                )


class BargainingParams(ImmutableParams, Record):
    """The game form and payoffs. Config keys (defaults in brackets):

    ``roles``           ``{A, B}``: the two principals.
    ``grid``            the offers: shares for A, strictly increasing [0, .25, .5, .75, 1].
    ``horizon``         periods before impasse [4].
    ``first_proposer``  ``A`` or ``B`` [A]; proposers alternate from there, unless
    ``proposers``       fixes one side as the proposer in every period [alternate].
    ``surplus``         the stake divided by a share [1].
    ``sides``           ``{A, B}``: :class:`SideParams`.
    ``breakdown``       ``{base, growth, max}``: a rejection in period ``t`` ends the talks
                        with probability ``clamp(base + growth t, 0, max)``.
    ``consent``         a list of :class:`ConsentParams`, asked in order.
    ``contest``         :class:`ContestParams`.
    ``types``           ``{A, B}``: :class:`TypeParams`; needs a fixed proposer.
    ``selection``       the equilibrium selected when a typed side proposes
                        [proposer_best].
    """

    def __init__(
        self,
        a_role: str,
        b_role: str,
        grid: tuple[float, ...],
        horizon: int,
        first_proposer: str,
        surplus: float,
        a: SideParams,
        b: SideParams,
        breakdown_base: float,
        breakdown_growth: float,
        breakdown_max: float,
        consent: tuple[ConsentParams, ...],
        proposers: str = "alternate",
        contest: ContestParams | None = None,
        a_types: TypeParams | None = None,
        b_types: TypeParams | None = None,
        selection: str = "proposer_best",
    ) -> None:
        self.a_role = a_role
        self.b_role = b_role
        self.grid = grid
        self.horizon = horizon
        self.first_proposer = first_proposer
        self.surplus = surplus
        self.a = a
        self.b = b
        self.breakdown_base = breakdown_base
        self.breakdown_growth = breakdown_growth
        self.breakdown_max = breakdown_max
        self.consent = consent
        self.proposers = proposers
        self.contest = contest
        self.a_types = a_types
        self.b_types = b_types
        self.selection = selection

    @property
    def extended(self) -> bool:
        """Whether there is hidden information or a contest, which need the belief solver."""

        return self.contest is not None or self.a_types is not None or self.b_types is not None

    def breakdown_risk(self, period: int) -> float:
        """P(breakdown | a rejection in *period*)."""

        return clamp(self.breakdown_base + self.breakdown_growth * period, 0.0, self.breakdown_max)

    def proposer(self, period: int) -> str:
        """The side that proposes in *period*."""

        if self.proposers != "alternate":
            return self.proposers
        first = self.first_proposer
        return first if period % 2 == 0 else other(first)

    def types_of(self, key: str) -> TypeParams | None:
        return self.a_types if key == "A" else self.b_types

    def delay_cost(self, key: str, kind: str | None) -> float:
        """Side *key*'s per-period delay cost, by its type *kind* when it has types."""

        types = self.types_of(key)
        if types is None or kind is None:
            return self.side(key).delay_cost
        return types.cost(kind)

    def side(self, key: str) -> SideParams:
        return self.a if key == "A" else self.b

    def role_of(self, key: str) -> str:
        return self.a_role if key == "A" else self.b_role

    def offer_id(self, index: int) -> str:
        """The action id of the grid's offer *index*: ``offer_050`` for a share of 0.5."""

        return f"offer_{round(self.grid[index] * 100):03d}"

    def offer_ids(self) -> tuple[str, ...]:
        return tuple(self.offer_id(i) for i in range(len(self.grid)))

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> BargainingParams:
        """Read the bargaining rules and validate offers, roles and optional private types."""

        with Section("bargaining", config) as s:
            with s.section("roles") as roles:
                a_role = identifier(roles.at("A"), roles.get("A"))
                b_role = identifier(roles.at("B"), roles.get("B"))
            grid = tuple(
                probability(s.at("grid"), x)
                for x in s.sequence("grid", [0.0, 0.25, 0.5, 0.75, 1.0])
            )
            if len(grid) < 2 or any(b <= a for a, b in pairwise(grid)):
                raise ConfigError(
                    "bargaining.grid must be strictly increasing with at least 2 points"
                )
            if len(grid) > 11:
                raise ConfigError("bargaining.grid may have at most 11 points")
            first = s.get("first_proposer", "A")
            if first not in SIDE_ROLES:
                raise ConfigError("bargaining.first_proposer must be 'A' or 'B'")
            sides = s.section("sides")
            consent = tuple(
                ConsentParams.from_config(Section(f"bargaining.consent[{i}]", raw))
                for i, raw in enumerate(s.sequence("consent", []))
            )
            ids = [a_role, b_role, *(c.id for c in consent)]
            if len(set(ids)) != len(ids):
                raise ConfigError("bargaining role ids must be unique")
            proposers = s.get("proposers", "alternate")
            if proposers not in ("alternate", *SIDE_ROLES):
                raise ConfigError("bargaining.proposers must be 'alternate', 'A' or 'B'")
            with s.section("types", {}) as types:
                a_types = TypeParams.from_config(types.section("A")) if types.has("A") else None
                b_types = TypeParams.from_config(types.section("B")) if types.has("B") else None
            if (a_types or b_types) and proposers == "alternate":
                raise ConfigError("bargaining.types needs a fixed proposer (proposers: A or B)")
            selection = s.choice("selection", SELECTIONS, "proposer_best")
            contest = ContestParams.from_config(s.section("contest")) if s.has("contest") else None
            horizon = s.positive_int("horizon", 4)
            surplus = s.nonnegative("surplus", 1.0)
            with sides:
                a = SideParams.from_config(sides.section("A"))
                b = SideParams.from_config(sides.section("B"))
            with s.section("breakdown", {}) as breakdown:
                return cls(
                    a_role=a_role,
                    b_role=b_role,
                    grid=grid,
                    horizon=horizon,
                    first_proposer=first,
                    surplus=surplus,
                    a=a,
                    b=b,
                    breakdown_base=breakdown.probability("base", 0.0),
                    breakdown_growth=breakdown.nonnegative("growth", 0.0),
                    breakdown_max=breakdown.probability("max", 1.0),
                    consent=consent,
                    proposers=proposers,
                    contest=contest,
                    a_types=a_types,
                    b_types=b_types,
                    selection=selection,
                )


Draws = tuple[tuple[str, float], ...]  # (chance outcome, probability)


class BargainingGame:
    """Alternating-offers bargaining: the factory of its initial states."""

    name = NAME
    outcome_labels = OUTCOMES

    def __init__(self, params: BargainingParams) -> None:
        self.params = params

    @property
    def roles(self) -> tuple[str, ...]:
        """The two principals, then the consent roles."""

        p = self.params
        return (p.a_role, p.b_role, *(c.id for c in p.consent))

    def new_initial_state(self) -> BargainingState:
        return BargainingState(self.params)


def breaks_commitment(side: SideParams, key: str, share: float) -> bool:
    """Whether an agreement at *share* goes beyond side *key*'s committed position."""

    assert side.commitment is not None
    if key == "A":
        return share < side.commitment - TOLERANCE
    return share > side.commitment + TOLERANCE


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
    """Principal *key*'s payoff at a terminal *label*, with per-period delay *cost*.

    An agreement in period ``t`` is worth ``worth x discount^t - cost x t``, less the
    commitment cost if it breaks the side's commitment. Breakdown and impasse pay their
    values less ``cost x period``; a veto pays the vetoed value less ``cost x t``.
    """

    side = p.side(key)
    if label == "agreement":
        assert share is not None and agreed_period is not None
        own_share = share if key == "A" else 1.0 - share
        if side.agreement is None:
            worth = p.surplus * own_share
        else:
            base, per_share = side.agreement
            worth = base + per_share * own_share
        value = worth * side.discount**agreed_period
        value -= cost * agreed_period
        if committed and side.commitment is not None and breaks_commitment(side, key, share):
            value -= side.commitment_cost
        return value
    if label == "breakdown":
        return side.breakdown_value - cost * period
    if label == "impasse":
        return side.impasse_value - cost * period
    vetoed = side.impasse_value if side.vetoed_value is None else side.vetoed_value
    return vetoed - cost * (agreed_period or 0)


class Phase:
    """A node of the game: who moves, with what actions or chances, and what follows."""

    def player(self, state: BargainingState) -> str:
        """The mover (chance by default)."""

        return CHANCE

    def legal_actions(self, state: BargainingState) -> tuple[str, ...]:
        """The mover's action set (empty at a chance node)."""

        return ()

    def draws(self, state: BargainingState) -> Draws:
        """The chance distribution, zero-probability outcomes included."""

        return ()

    def apply(self, state: BargainingState, action: str) -> None:
        """The transition on *action* (or on a chance outcome)."""

        raise NotImplementedError


class TypeDraw(Phase):
    """Nature draws a side's resolve, strong with its prior; only that side observes it."""

    def __init__(self, key: str) -> None:
        self.key = key

    def draws(self, state: BargainingState) -> Draws:
        types = state.params.types_of(self.key)
        assert types is not None
        return ((STRONG, types.prior), (WEAK, 1.0 - types.prior))

    def apply(self, state: BargainingState, action: str) -> None:
        key = self.key
        state.types[key] = action
        state.log(f"type:{key}={action}", visible_to=(state.params.role_of(key),))
        state.phase = "type_b" if key == "A" else "commit_a"
        state.skip_absent_setup()


class Commit(Phase):
    """A side publicly commits to its position, or does not (A first, then B)."""

    def __init__(self, key: str) -> None:
        self.key = key

    def player(self, state: BargainingState) -> str:
        return state.params.role_of(self.key)

    def legal_actions(self, state: BargainingState) -> tuple[str, ...]:
        return ("no_commitment", "commit")

    def apply(self, state: BargainingState, action: str) -> None:
        key, committed = self.key, action == "commit"
        state.committed[key] = committed
        state.log(f"commit:{key}={action}")
        state.events.append(("commit", key, committed))
        if committed:
            state.tags.append(f"{state.params.role_of(key)}:committed")
        state.phase = "commit_b" if key == "A" else "propose"
        state.skip_absent_setup()


class Propose(Phase):
    """The period's proposer offers a share from the grid."""

    def player(self, state: BargainingState) -> str:
        p = state.params
        return p.role_of(p.proposer(state.period))

    def legal_actions(self, state: BargainingState) -> tuple[str, ...]:
        return state.params.offer_ids()

    def apply(self, state: BargainingState, action: str) -> None:
        state.offer = self.legal_actions(state).index(action)
        state.log(f"t{state.period}:offer={action}")
        state.events.append(("offer", state.period, state.offer))
        state.phase = "respond"


class Respond(Phase):
    """The other side accepts the offer or rejects it."""

    def player(self, state: BargainingState) -> str:
        p = state.params
        return p.role_of(other(p.proposer(state.period)))

    def legal_actions(self, state: BargainingState) -> tuple[str, ...]:
        return ("accept", "reject")

    def apply(self, state: BargainingState, action: str) -> None:
        state.log(f"t{state.period}:{action}")
        state.events.append(("respond", state.period, action == "accept"))
        if action == "accept":
            assert state.offer is not None
            state.share = state.params.grid[state.offer]
            state.agreed_period = state.period
            state.after_agreement()
        else:
            state.after_rejection()


class Breakdown(Phase):
    """After a rejection, talks break down with the period's breakdown risk."""

    def draws(self, state: BargainingState) -> Draws:
        risk = state.params.breakdown_risk(state.period)
        return (("breakdown", risk), ("talks_continue", 1.0 - risk))

    def apply(self, state: BargainingState, action: str) -> None:
        state.log(f"t{state.period}:{action}")
        if action == "breakdown":
            state.label = "breakdown"
        else:
            state.after_talks_continue()


class Contest(Phase):
    """The contest between periods: the position moves toward A with ``win_probability``."""

    def draws(self, state: BargainingState) -> Draws:
        contest = state.params.contest
        assert contest is not None
        q = contest.win_probability
        return (("a_gains", q), ("b_gains", 1.0 - q))

    def apply(self, state: BargainingState, action: str) -> None:
        state.log(f"t{state.period}:{action}")
        state.move(1 if action == "a_gains" else -1)


class Consent(Phase):
    """Each consent role in turn ratifies the agreement or vetoes it."""

    def player(self, state: BargainingState) -> str:
        return state.params.consent[state.consent_index].id

    def legal_actions(self, state: BargainingState) -> tuple[str, ...]:
        return ("ratify", "veto")

    def apply(self, state: BargainingState, action: str) -> None:
        consent = state.params.consent[state.consent_index]
        state.log(f"consent:{consent.id}={action}")
        if action == "veto":
            state.label = "vetoed"
            state.tags.append(f"{consent.id}:vetoed")
        else:
            state.consent_index += 1
            if state.consent_index >= len(state.params.consent):
                state.label = "agreement"


PHASES: dict[str, Phase] = {
    "type_a": TypeDraw("A"),
    "type_b": TypeDraw("B"),
    "commit_a": Commit("A"),
    "commit_b": Commit("B"),
    "propose": Propose(),
    "respond": Respond(),
    "breakdown": Breakdown(),
    "contest": Contest(),
    "consent": Consent(),
}


class BargainingState(LoggedState):
    """A history of the talks: the phase, the period and what has been settled so far."""

    def __init__(self, params: BargainingParams) -> None:
        super().__init__((params.a_role, params.b_role, *(c.id for c in params.consent)))
        self.params = params
        self.phase = "type_a"  # a key of PHASES
        self.committed = {"A": False, "B": False}
        self.types: dict[str, str | None] = {"A": None, "B": None}
        self.period = 0
        self.position = 0  # contest steps from the initial position (+ toward A)
        self.last_contest = 0  # +1 A gained, -1 B gained, 0 none yet
        self.offer: int | None = None  # index in the grid of the offer on the table
        self.consent_index = 0
        self.label: str | None = None
        self.share: float | None = None
        self.agreed_period: int | None = None
        self.tags: list[str] = []
        # the public record that belief tracking replays: (kind, ...) in order
        self.events: list[tuple[Any, ...]] = []
        self.skip_absent_setup()

    def skip_absent_setup(self) -> None:
        """Pass over the type draws and commitment choices this game does not have."""

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
        if self.label is not None:
            return TERMINAL
        return PHASES[self.phase].player(self)

    def is_terminal(self) -> bool:
        return self.label is not None

    def legal_actions(self) -> tuple[str, ...]:
        if self.label is not None:
            return ()
        return PHASES[self.phase].legal_actions(self)

    def chance_outcomes(self) -> Draws:
        if self.label is not None:
            return ()
        return tuple((a, prob) for a, prob in PHASES[self.phase].draws(self) if prob > 0.0)

    def apply_action(self, action: str) -> None:
        player = self.current_player()
        legal = (
            tuple(a for a, _ in self.chance_outcomes())
            if player == CHANCE
            else self.legal_actions()
        )
        check_legal(action, legal)
        PHASES[self.phase].apply(self, action)
        self._history.append(action)

    # ------------------------------------------------- what follows a response
    def after_agreement(self) -> None:
        """An accepted offer goes to the consent roles, if any."""

        if self.params.consent:
            self.phase = "consent"
            self.consent_index = 0
        else:
            self.label = "agreement"

    def after_rejection(self) -> None:
        """A rejection risks breakdown (a chance node unless the risk is 0 or 1)."""

        risk = self.params.breakdown_risk(self.period)
        if risk >= 1.0:
            self.label = "breakdown"
        elif risk <= 0.0:
            self.after_talks_continue()
        else:
            self.phase = "breakdown"

    def after_talks_continue(self) -> None:
        """Surviving talks go to the contest, if any (a chance node unless it is certain)."""

        contest = self.params.contest
        if contest is None:
            self._next_period()
        elif contest.win_probability >= 1.0:
            self.move(1)
        elif contest.win_probability <= 0.0:
            self.move(-1)
        else:
            self.phase = "contest"

    def move(self, direction: int) -> None:
        """Move the contest's position one step; a side whose position reaches the end
        has collapsed, and the game ends as an agreement on the winner's terms."""

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
        """Evaluate this ending using the side's private delay cost and public commitment."""

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
        """The principals' payoffs (:func:`side_payoff`) and each consent role's value of
        the ending; a role that cast the veto also pays its veto cost."""

        if self.label is None:
            raise ConfigError("returns requested before bargaining ended")
        p = self.params
        result = {p.a_role: self._side_payoff("A"), p.b_role: self._side_payoff("B")}
        for consent in p.consent:
            if self.label == "agreement":
                assert self.share is not None
                result[consent.id] = consent.base + consent.per_share * self.share
            elif self.label == "vetoed":
                own = f"{consent.id}:vetoed" in self.tags
                result[consent.id] = consent.veto_value - (consent.veto_cost if own else 0.0)
            elif self.label == "breakdown":
                result[consent.id] = consent.breakdown_value
            else:
                result[consent.id] = consent.impasse_value
        return result

    def outcome(self) -> Outcome:
        """The ending with its tags and metrics: the period, the agreed ``share``, the
        contest's ``position``, each typed side's revealed type, and a
        ``<role>:broke_commitment`` tag for an agreement beyond a commitment."""

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
                if breaks_commitment(side, key, self.share):
                    tags.append(f"{p.role_of(key)}:broke_commitment")
        return Outcome(self.label, tuple(tags), metrics)

    def observation(self, role: str) -> tuple[float, ...]:
        """Thirteen features, from *role*'s side: the period, the breakdown risk, the
        offer on the table as its own share, whether a response or consent is due,
        whether it proposes, both sides' commitments, the agreed share, its side, the
        contest's position and last move, and its own type."""

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
            1.0 if key and self.committed.get(other(key), False) else 0.0,
            1.0 if self.phase == "consent" else 0.0,
            (self.share or 0.0),
            1.0 if key == "A" else 0.0,
            position if key != "B" else 1.0 - position,
            float(self.last_contest) if key != "B" else -float(self.last_contest),
            0.5 if kind is None else (1.0 if kind == STRONG else 0.0),
        )
