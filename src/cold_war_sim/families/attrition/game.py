"""The stages and state of a war of attrition, from entry to settlement or victory."""

from __future__ import annotations

from collections.abc import Collection, Mapping

from ...games.protocol import CHANCE, TERMINAL, LoggedState, Outcome, check_legal
from ..common import ConfigError
from .params import EFFORT, NAME, REFUSED, STAGES, AttritionParams, other_side

Draws = tuple[tuple[str, float], ...]  # (chance outcome, probability)


class Stage:
    """A node of the war: who moves, with what actions or chances, and what follows."""

    def player(self, state: AttritionState) -> str:
        """The mover (chance by default)."""

        return CHANCE

    def legal_actions(self, state: AttritionState) -> tuple[str, ...]:
        """The mover's action set (empty at a chance node)."""

        return ()

    def draws(self, state: AttritionState) -> Draws:
        """The chance distribution (empty at a decision)."""

        return ()

    def apply(self, state: AttritionState, player: str, action: str) -> None:
        """The transition on *action* (or on a chance outcome)."""

        raise NotImplementedError


class Types(Stage):
    """Nature draws each type holder's kind, in turn, before the war."""

    def draws(self, state: AttritionState) -> Draws:
        return state.type_outcomes(state.type_queue[0])

    def apply(self, state: AttritionState, player: str, action: str) -> None:
        state.set_type(action)
        state.next_type()


class Approval(Stage):
    """Each approving sponsor in turn approves the war or refuses it.

    A refusal ends the game as ``refused``.
    """

    def player(self, state: AttritionState) -> str:
        return state.approvals[0]

    def legal_actions(self, state: AttritionState) -> tuple[str, ...]:
        return ("approve", "refuse")

    def apply(self, state: AttritionState, player: str, action: str) -> None:
        state.log(f"approval:{player}={action}")
        state.approvals.pop(0)
        if action == "refuse":
            state.refused_by = player
            state.label = REFUSED
        elif not state.approvals:
            state.start_round()


class Warn(Stage):
    """A sponsor whose red line is one step away warns, once, or stays silent."""

    def player(self, state: AttritionState) -> str:
        return state.warn_queue[0]

    def legal_actions(self, state: AttritionState) -> tuple[str, ...]:
        return ("silent", "warn")

    def apply(self, state: AttritionState, player: str, action: str) -> None:
        state.log(f"r{state.round}:{player}={action}")
        state.warn_queue.pop(0)
        if action == "warn":
            state.warned.add(player)
        if not state.warn_queue:
            state.open_commitments()


class Commit(Stage):
    """A side lead commits publicly, once, to its minimum terms, or holds off."""

    def player(self, state: AttritionState) -> str:
        return state.commit_queue[0]

    def legal_actions(self, state: AttritionState) -> tuple[str, ...]:
        return ("hold_off", "commit")

    def apply(self, state: AttritionState, player: str, action: str) -> None:
        state.log(f"r{state.round}:{player}={action}")
        state.commit_queue.pop(0)
        if action == "commit":
            state.committed.add(player)
        if not state.commit_queue:
            state.open_offers()


class Offer(Stage):
    """The round's proposing lead offers a settlement (side A's share) or makes none."""

    def player(self, state: AttritionState) -> str:
        role = state.proposer()
        assert role is not None
        return role

    def legal_actions(self, state: AttritionState) -> tuple[str, ...]:
        assert state.params.offers is not None
        return ("no_offer", *state.params.offers.ids())

    def apply(self, state: AttritionState, player: str, action: str) -> None:
        state.log(f"r{state.round}:{player}={action}")
        if action == "no_offer":
            state.open_efforts()
            return
        assert state.params.offers is not None
        state.offer = state.params.offers.ids().index(action)
        state.stage = "respond"


class Respond(Stage):
    """The other lead accepts the offer or rejects it; an accepted offer goes to the
    consent roles still in the war, if any."""

    def player(self, state: AttritionState) -> str:
        role = state.responder()
        assert role is not None
        return role

    def legal_actions(self, state: AttritionState) -> tuple[str, ...]:
        return ("accept", "reject")

    def apply(self, state: AttritionState, player: str, action: str) -> None:
        state.log(f"r{state.round}:{player}={action}")
        if action == "reject":
            state.open_efforts()
            return
        assert state.params.offers is not None
        state.ratify_queue = [
            c.role for c in state.params.offers.consent if c.role not in state.withdrawn
        ]
        if state.ratify_queue:
            state.stage = "ratify"
        else:
            state.agree()


class Ratify(Stage):
    """Each consent role in turn ratifies the accepted offer or vetoes it at its veto cost."""

    def player(self, state: AttritionState) -> str:
        return state.ratify_queue[0]

    def legal_actions(self, state: AttritionState) -> tuple[str, ...]:
        return ("ratify", "veto")

    def apply(self, state: AttritionState, player: str, action: str) -> None:
        state.log(f"r{state.round}:{player}={action}")
        state.ratify_queue.pop(0)
        if action == "veto":
            assert state.params.offers is not None
            state.vetoed.add(player)
            state.cost[player] += state.params.offers.consent_of(player).veto_cost
            state.open_efforts()
        elif not state.ratify_queue:
            state.agree()


class Effort(Stage):
    """Every sponsor still in the war chooses its effort; the choices are simultaneous.

    A sponsor sees only its own choice until all are in. Without offers the leads may
    ``negotiate`` a partition at the line of contact.
    """

    def player(self, state: AttritionState) -> str:
        return state.pending[0]

    def legal_actions(self, state: AttritionState) -> tuple[str, ...]:
        role = state.pending[0]
        options = ["withdraw", "low", "high"]
        if state.params.sponsor(role).direct:
            options.append("direct")
        if role in state.leads.values() and state.params.offers is None:
            options.append("negotiate")  # with offers, settlements come through offers
        return tuple(options)

    def apply(self, state: AttritionState, player: str, action: str) -> None:
        state.choices[player] = action
        state.log(f"r{state.round}:self={action}", visible_to=[player])
        state.pending.pop(0)
        if not state.pending:
            state.reveal()


class Battle(Stage):
    """Nature resolves the round's battle: the front moves a step toward its winner, or
    holds."""

    def draws(self, state: AttritionState) -> Draws:
        return state.battle_outcomes(state.choices)

    def apply(self, state: AttritionState, player: str, action: str) -> None:
        state.resolve_battle(action)


STAGE: dict[str, Stage] = {
    "types": Types(),
    "approval": Approval(),
    "warn": Warn(),
    "commit": Commit(),
    "offer": Offer(),
    "respond": Respond(),
    "ratify": Ratify(),
    "effort": Effort(),
    "battle": Battle(),
}


class AttritionGame:
    """A war of attrition: the factory of its initial states."""

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
    """A history of the war: the front, the round, each sponsor's costs and weariness, and
    the stage of the current round.
    """

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
        self.next_type()

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
        """Whether the round's battle is about to be resolved."""

        return self.stage == "battle"

    def next_type(self) -> None:
        """Draw the next holder's type (a single possible kind is set without a chance node);
        then go to approval, or start the first round.
        """

        while self.type_queue:
            outcomes = self.type_outcomes(self.type_queue[0])
            if len(outcomes) > 1:
                self.stage = "types"
                return
            self.set_type(outcomes[0][0])
        if self.approvals:
            self.stage = "approval"
            return
        self.start_round()

    def type_outcomes(self, holder: str) -> tuple[tuple[str, float], ...]:
        """Nature's distribution over *holder*'s kinds."""

        return tuple(self.priors[holder].items())

    def set_type(self, kind: str) -> None:
        """Give the next holder its *kind* and log it to those who observe it: no one
        (``unseen``), the holder alone (``private``), or the holder's side.
        """

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

    def start_round(self) -> None:
        """Begin a round: sponsors whose red line is one step away may warn first."""

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
        self.open_commitments()

    def open_commitments(self) -> None:
        """Let each lead that has not yet committed do so, once talks are open."""

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
        self.open_offers()

    def open_offers(self) -> None:
        """Let the round's proposer make an offer, if both leads are still in the war."""

        proposer, responder = self.proposer(), self.responder()
        if (
            proposer is not None
            and responder is not None
            and proposer not in self.withdrawn
            and responder not in self.withdrawn
        ):
            self.stage = "offer"
            return
        self.open_efforts()

    def open_efforts(self) -> None:
        """Go to the effort choices of every sponsor still in the war."""

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
        """The lead who answers this round's offer, if offers are open."""

        offers = self.params.offers
        if offers is None or self.round < offers.opens:
            return None
        return self.lead(other_side(offers.proposer_side(self.round)))

    def current_player(self) -> str:
        if self.label is not None:
            return TERMINAL
        return STAGE[self.stage].player(self)

    def is_terminal(self) -> bool:
        return self.label is not None

    def legal_actions(self) -> tuple[str, ...]:
        if self.label is not None:
            return ()
        return STAGE[self.stage].legal_actions(self)

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
        """P(side A wins the battle) under *choices*: E_a^gamma / (E_a^gamma + E_b^gamma),
        or 1/2 when no one commits any effort.
        """

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
        """The battle's distribution over ``a_gains``, ``b_gains`` and ``hold``."""

        p_a = self.win_probability(choices, kinds, deployed)
        move = self.move_probability(choices, kinds, deployed)
        raw = (("a_gains", p_a * move), ("b_gains", (1.0 - p_a) * move), ("hold", 1.0 - move))
        return tuple((a, q) for a, q in raw if q > 0.0)

    def _sponsor_strength(self, role: str, kinds: Mapping[str, str] | None) -> float:
        """The effort multiplier of *role*'s own type under *kinds* (1 without one)."""

        holder_types = self.params.sponsor_types.get(role)
        if holder_types is None:
            return 1.0
        kind = (kinds or {}).get(role, self.types.get(role))
        return holder_types.strength_of(kind) if kind is not None else 1.0

    def cost_multiplier(self, side: str, kind: str | None = None) -> float:
        """The effort-cost multiplier of *side*'s type (*kind*, or its actual kind)."""

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
        holder_types = self.params.sponsor_types.get(role)
        if holder_types is not None:
            kind = (kinds or {}).get(role, self.types.get(role))
            if kind is not None:
                total *= holder_types.cost_of(kind)
        return total

    def proxy_multiplier(self, role: str, kinds: Mapping[str, str] | None = None) -> float:
        """The sponsor's own level's multiplier on its proxy gain, under *kinds* (the
        actual type where *kinds* is silent)."""

        holder_types = self.params.sponsor_types.get(role)
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
        return STAGE[self.stage].draws(self)

    # ---------------------------------------------------------------- actions
    def apply_action(self, action: str) -> None:
        player = self.current_player()
        if player == TERMINAL:
            raise ConfigError("cannot act in a finished war")
        if player == CHANCE:
            check_legal(action, tuple(a for a, _ in self.chance_outcomes()))
        else:
            check_legal(action, self.legal_actions())
        STAGE[self.stage].apply(self, player, action)
        self._history.append(action)

    @property
    def terms(self) -> float | None:
        """The share under ratification: the offer accepted."""

        offers = self.params.offers
        if offers is None or self.offer is None:
            return None
        return offers.grid[self.offer]

    def agree(self) -> None:
        """End the war in a partition at the accepted offer."""

        assert self.params.offers is not None and self.offer is not None
        self.share = self.params.offers.grid[self.offer]
        self.agreed = True
        self.label = "partition"
        self._charge_commitments()

    def _settle_at_line(self) -> None:
        """End the war in a partition at the line of contact."""

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

    def reveal(self) -> None:
        """Reveal the round's efforts and charge for them.

        Each sponsor pays its effort cost (scaled by its types), builds weariness and collects
        any proxy gain; red lines are checked and a sponsor going direct may take its side's
        lead. Withdrawals take effect: a side with no sponsor left has withdrawn, both leads
        negotiating settle at the line, and otherwise the battle follows.
        """

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
        for role, side in p.lead_on_entry.items():  # a belligerent that goes direct takes the lead
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
        """Go to the battle; a certain result is resolved without a chance node."""

        self.stage = "battle"
        outcomes = self.chance_outcomes()
        if len(outcomes) == 1:
            self.resolve_battle(outcomes[0][0])

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
        """Take *role* out of the war at its withdrawal cost."""

        self.withdrawn.add(role)
        self.cost[role] += self.params.sponsor(role).withdraw_cost

    def _side_gone(self, side: str) -> bool:
        """Whether every sponsor of *side* has withdrawn."""

        return all(s.id in self.withdrawn for s in self.params.sponsors if s.side == side)

    def resolve_battle(self, action: str) -> None:
        """Move the front by the battle's result and close the round.

        Control at a victory threshold ends the war; so does the round limit (stalemate).
        Sponsors at their weariness limit are forced out before the next round.
        """

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
            self.start_round()

    # --------------------------------------------------------------- payoffs
    def returns(self) -> dict[str, float]:
        """Each sponsor's value of the outcome (a partition by its share where it values
        settlements so) less its accumulated costs.
        """

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
        """The ending with its tags and metrics.

        Per sponsor: ``cost``, ``weariness``, ``effort`` (mean per round), ``commit`` (effort
        relative to full effort for ``max_rounds``) and ``side_share`` (its share of its side's
        effort). With bargaining blocks: the front's ``peak`` and ``trough``, the settled
        ``share`` and each holder's type.
        """

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
        for side in ("a", "b"):  # each sponsor's share of its side's effort (even if none)
            members = [s.id for s in p.sponsors if s.side == side]
            total = sum(self.effort[m] for m in members)
            for member in members:
                share = self.effort[member] / total if total > 0 else 1.0 / len(members)
                metrics[f"side_share:{member}"] = share
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
        """Nineteen features from *role*'s side: its control, the round, its weariness and
        cost, its last effort, its side's share of the last efforts, the stage, the offer on
        the table as its own share, warnings and commitments, and what it knows of its side's
        type and of its own.
        """

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
        own_types = p.types_of(role) if role in p.sponsor_types else None
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
