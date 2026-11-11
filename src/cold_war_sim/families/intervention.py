"""A patron responds to an uprising in a client state, with a rival watching.

A noisy report informs the patron's choice to accommodate, suppress, or
intervene. The rival may provide aid or confront an intervention. The final
outcome depends on these choices and the balance of forces.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..games.protocol import (
    CHANCE,
    PUBLIC,
    TERMINAL,
    ImmutableParams,
    LoggedState,
    Outcome,
    check_legal,
)
from ..records import Record
from .common import (
    ConfigError,
    Risk,
    Section,
    ValueTable,
    clamp,
    identifier,
    prospect,
    risk_attitude,
    value_table,
)

NAME = "intervention"
OUTCOMES = ("reformed", "crushed", "defected", "quagmire", "clash")
PATRON_ACTIONS = ("accommodate", "crackdown", "intervene")
RIVAL_ACTIONS = ("abstain", "condemn", "covert_aid", "confront")
# an action's commitment: own forces 1, local forces or covert aid 1/3, otherwise 0
COMMITMENT = {"intervene": 1.0, "confront": 1.0, "crackdown": 1 / 3, "covert_aid": 1 / 3}


class InterventionParams(ImmutableParams, Record):
    """The game form and payoffs. Config keys (defaults in brackets):

    ``roles``                  ``{patron, rival}``: the two players.
    ``p_strong``               P(the movement is strong) [0.5].
    ``report_accuracy``        P(the report is right about the strength) [0.7].
    ``rival_observes_report``  whether the rival sees the report too [no].
    ``movement``               ``{weak, strong}``: the movement's strength [0.5, 1.5].
    ``force``                  ``{crackdown, intervene}``: the patron's force [1, 3].
    ``rival_boost``            ``{covert_aid, confront}``: added to the movement [0.3, 1].
    ``gamma``                  the contest's decisiveness [1]:
                               P(hold) = F^gamma / (F^gamma + M^gamma).
    ``accommodate_defect``     ``{weak, strong}``: P(defection | accommodation) [0.05, 0.4].
    ``aid_defect_shift``       added to that probability when the rival aids [0.1].
    ``quagmire_prob``          P(quagmire | held) after intervening against a strong
                               movement [0].
    ``clash_risk``             P(clash) when a confronted patron stands firm [0.1].
    ``values``                 ``{patron, rival}``: each player's payoff at each outcome.
    ``costs``                  the cost of ``crackdown``, ``intervene`` and ``back_down``
                               to the patron, and of ``condemn``, ``covert_aid`` and
                               ``confront`` to the rival [0].
    ``risk``                   ``{patron, rival}``: optional prospect-theory attitudes.
    """

    def __init__(
        self,
        patron: str,
        rival: str,
        p_strong: float,
        report_accuracy: float,
        rival_observes_report: bool,
        movement: dict[str, float],
        force: dict[str, float],
        boost: dict[str, float],
        gamma: float,
        accommodate_defect: dict[str, float],
        aid_defect_shift: float,
        quagmire_prob: float,
        clash_risk: float,
        patron_values: ValueTable,
        rival_values: ValueTable,
        costs: dict[str, float],
        patron_risk: Risk | None = None,
        rival_risk: Risk | None = None,
    ) -> None:
        self.patron = patron
        self.rival = rival
        self.p_strong = p_strong
        self.report_accuracy = report_accuracy
        self.rival_observes_report = rival_observes_report
        self.movement = movement  # strength -> the movement's strength in the contest
        self.force = force  # crackdown | intervene -> the patron's force
        self.boost = boost  # covert_aid | confront -> added to the movement
        self.gamma = gamma
        self.accommodate_defect = accommodate_defect  # strength -> P(defected | accommodate)
        self.aid_defect_shift = aid_defect_shift
        self.quagmire_prob = quagmire_prob
        self.clash_risk = clash_risk
        self.patron_values = patron_values
        self.rival_values = rival_values
        self.costs = costs  # action -> its cost to the player taking it
        self.patron_risk = patron_risk
        self.rival_risk = rival_risk

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> InterventionParams:
        """Read the actors, forces, report accuracy and payoffs for an intervention."""

        with Section("intervention", config) as s:
            with s.section("risk", {}) as risk:
                patron_risk = (
                    risk_attitude(risk.at("patron"), risk.get("patron"))
                    if risk.has("patron")
                    else None
                )
                rival_risk = (
                    risk_attitude(risk.at("rival"), risk.get("rival"))
                    if risk.has("rival")
                    else None
                )
            with s.section("roles") as roles:
                patron = identifier(roles.at("patron"), roles.get("patron"))
                rival = identifier(roles.at("rival"), roles.get("rival"))
            if patron == rival:
                raise ConfigError("intervention patron and rival must differ")
            with s.section("movement", {}) as m:
                movement = {
                    "weak": m.nonnegative("weak", 0.5),
                    "strong": m.nonnegative("strong", 1.5),
                }
            with s.section("force", {}) as f:
                force = {
                    "crackdown": f.nonnegative("crackdown", 1.0),
                    "intervene": f.nonnegative("intervene", 3.0),
                }
            with s.section("rival_boost", {}) as b:
                boost = {
                    "covert_aid": b.nonnegative("covert_aid", 0.3),
                    "confront": b.nonnegative("confront", 1.0),
                }
            with s.section("accommodate_defect", {}) as d:
                accommodate_defect = {
                    "weak": d.probability("weak", 0.05),
                    "strong": d.probability("strong", 0.4),
                }
            with s.section("values") as values:
                patron_values = value_table(values.at("patron"), values.get("patron"), OUTCOMES)
                rival_values = value_table(values.at("rival"), values.get("rival"), OUTCOMES)
            with s.section("costs", {}) as c:
                costs = {
                    action: c.nonnegative(action, 0.0)
                    for action in ("crackdown", "intervene", "back_down", *RIVAL_ACTIONS[1:])
                }
            gamma = s.number("gamma", 1.0)
            if gamma <= 0.0:
                raise ConfigError("intervention.gamma must be positive")
            return cls(
                patron=patron,
                rival=rival,
                p_strong=s.probability("p_strong", 0.5),
                report_accuracy=s.probability("report_accuracy", 0.7),
                rival_observes_report=s.flag("rival_observes_report"),
                movement=movement,
                force=force,
                boost=boost,
                gamma=gamma,
                accommodate_defect=accommodate_defect,
                aid_defect_shift=s.nonnegative("aid_defect_shift", 0.1),
                quagmire_prob=s.probability("quagmire_prob", 0.0),
                clash_risk=s.probability("clash_risk", 0.1),
                patron_values=patron_values,
                rival_values=rival_values,
                costs=costs,
                patron_risk=patron_risk,
                rival_risk=rival_risk,
            )


Draws = tuple[tuple[str, float], ...]  # (chance outcome, probability)


class Phase:
    """A node of the game: who moves, with what actions or chances, and what follows."""

    def player(self, state: InterventionState) -> str:
        """The mover (chance by default)."""

        return CHANCE

    def legal_actions(self, state: InterventionState) -> tuple[str, ...]:
        """The mover's action set (empty at a chance node)."""

        return ()

    def draws(self, state: InterventionState) -> Draws:
        """The chance distribution, zero-probability outcomes included."""

        return ()

    def apply(self, state: InterventionState, action: str) -> None:
        """The transition on *action* (or on a chance outcome)."""

        raise NotImplementedError


class Strength(Phase):
    """Nature draws the movement's strength: strong with probability ``p_strong``."""

    def draws(self, state: InterventionState) -> Draws:
        p = state.params.p_strong
        return (("strong", p), ("weak", 1.0 - p))

    def apply(self, state: InterventionState, action: str) -> None:
        state.strength = action
        state.phase = "report"


class Report(Phase):
    """A binary signal of the strength, right with probability ``report_accuracy``.

    The patron sees it; the rival does too under ``rival_observes_report``.
    """

    def draws(self, state: InterventionState) -> Draws:
        accuracy = state.params.report_accuracy
        alarm = accuracy if state.strength == "strong" else 1.0 - accuracy
        return (("alarming", alarm), ("calm", 1.0 - alarm))

    def apply(self, state: InterventionState, action: str) -> None:
        p = state.params
        state.report = action
        observers = (p.patron, p.rival) if p.rival_observes_report else (p.patron,)
        state.log(f"report={action}", visible_to=observers)
        state.phase = "patron"


class PatronMove(Phase):
    """The patron accommodates, cracks down (local forces) or intervenes (its own)."""

    def player(self, state: InterventionState) -> str:
        return state.params.patron

    def legal_actions(self, state: InterventionState) -> tuple[str, ...]:
        return PATRON_ACTIONS

    def apply(self, state: InterventionState, action: str) -> None:
        state.patron_action = action
        state.log(f"patron={action}", visible_to=PUBLIC)
        state.phase = "rival"


class RivalMove(Phase):
    """The rival abstains, condemns or aids covertly; it may confront an intervention."""

    def player(self, state: InterventionState) -> str:
        return state.params.rival

    def legal_actions(self, state: InterventionState) -> tuple[str, ...]:
        return RIVAL_ACTIONS if state.patron_action == "intervene" else RIVAL_ACTIONS[:3]

    def apply(self, state: InterventionState, action: str) -> None:
        state.rival_action = action
        state.log(f"rival={action}", visible_to=PUBLIC)
        state.phase = "reply" if action == "confront" else "resolve"


class Reply(Phase):
    """A confronted patron backs down (the client defects) or stands firm."""

    def player(self, state: InterventionState) -> str:
        return state.params.patron

    def legal_actions(self, state: InterventionState) -> tuple[str, ...]:
        return ("back_down", "stand_firm")

    def apply(self, state: InterventionState, action: str) -> None:
        state.log(f"reply={action}", visible_to=PUBLIC)
        if action == "back_down":
            state.backed_down = True
            state.label = "defected"
        else:
            state.stood_firm = True
            state.phase = "clash"


class Clash(Phase):
    """Standing firm ends in a direct clash with probability ``clash_risk``."""

    def draws(self, state: InterventionState) -> Draws:
        risk = state.params.clash_risk
        return (("clash", risk), ("no_clash", 1.0 - risk))

    def apply(self, state: InterventionState, action: str) -> None:
        state.log(f"clash={action}", visible_to=PUBLIC)
        if action == "clash":
            state.label = "clash"
        else:
            state.phase = "resolve"


class Resolve(Phase):
    """Resolve the uprising from the forces committed to it.

    After accommodation, the client defects with probability
    ``accommodate_defect[strength]``, increased by ``aid_defect_shift`` if the
    rival aids it. Otherwise it reforms.

    After suppression or intervention, force F faces movement strength M,
    including the rival's support:

        P(hold) = F^gamma / (F^gamma + M^gamma)

    A successful hold crushes the uprising, except that intervention against a
    strong movement becomes a quagmire with probability ``quagmire_prob``.
    Failure means the client defects.
    """

    def draws(self, state: InterventionState) -> Draws:
        p = state.params
        strength = "strong" if state.strength == "strong" else "weak"
        if state.patron_action == "accommodate":
            aided = state.rival_action in p.boost
            shift = p.aid_defect_shift if aided else 0.0
            defect = clamp(p.accommodate_defect[strength] + shift, 0.0, 1.0)
            return (("defected", defect), ("reformed", 1.0 - defect))
        intervened = state.patron_action == "intervene"
        force = p.force["intervene" if intervened else "crackdown"]
        movement = p.movement[strength] + p.boost.get(state.rival_action or "", 0.0)
        hold = force**p.gamma / (force**p.gamma + movement**p.gamma) if force > 0 else 0.0
        quagmire = p.quagmire_prob if intervened and strength == "strong" else 0.0
        return (
            ("crushed", hold * (1.0 - quagmire)),
            ("quagmire", hold * quagmire),
            ("defected", 1.0 - hold),
        )

    def apply(self, state: InterventionState, action: str) -> None:
        state.label = action


PHASES: dict[str, Phase] = {
    "strength": Strength(),
    "report": Report(),
    "patron": PatronMove(),
    "rival": RivalMove(),
    "reply": Reply(),
    "clash": Clash(),
    "resolve": Resolve(),
}


class InterventionGame:
    """An intervention game with fixed parameters and fresh states for play."""

    name = NAME
    outcome_labels = OUTCOMES

    def __init__(self, params: InterventionParams) -> None:
        self.params = params

    @property
    def roles(self) -> tuple[str, ...]:
        return (self.params.patron, self.params.rival)

    def new_initial_state(self) -> InterventionState:
        return InterventionState(self.params)


class InterventionState(LoggedState):
    """A history of the intervention: the current phase and the moves made so far."""

    def __init__(self, params: InterventionParams) -> None:
        super().__init__((params.patron, params.rival))
        self.params = params
        self.phase = "strength"  # a key of PHASES
        self.strength: str | None = None
        self.report: str | None = None
        self.patron_action: str | None = None
        self.rival_action: str | None = None
        self.backed_down = False
        self.stood_firm = False
        self.label: str | None = None
        self._skip_certain()

    def current_player(self) -> str:
        if self.label is not None:
            return TERMINAL
        return PHASES[self.phase].player(self)

    def is_terminal(self) -> bool:
        return self.label is not None

    def legal_actions(self) -> tuple[str, ...]:
        return PHASES[self.phase].legal_actions(self)

    def chance_outcomes(self) -> tuple[tuple[str, float], ...]:
        if self.current_player() != CHANCE:
            return ()
        return tuple((a, p) for a, p in PHASES[self.phase].draws(self) if p > 0.0)

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
        self._skip_certain()

    def _skip_certain(self) -> None:
        """Resolve chance phases with a single possible outcome without a node."""

        while self.label is None and self.current_player() == CHANCE:
            outcomes = self.chance_outcomes()
            if len(outcomes) != 1:
                return
            PHASES[self.phase].apply(self, outcomes[0][0])

    def returns(self) -> dict[str, float]:
        """Each player's value of the outcome less the costs of its own actions (backing
        down has its own cost), through its risk attitude."""

        if self.label is None:
            raise ConfigError("returns requested before the intervention ended")
        p = self.params
        patron = p.patron_values[self.label]
        patron -= p.costs.get(self.patron_action or "", 0.0)
        if self.backed_down:
            patron -= p.costs["back_down"]
        rival = p.rival_values[self.label]
        rival -= p.costs.get(self.rival_action or "", 0.0)
        return {p.patron: prospect(patron, p.patron_risk), p.rival: prospect(rival, p.rival_risk)}

    def outcome(self) -> Outcome:
        """The outcome, tagged with both players' actions and the movement's strength;
        ``commit:<role>`` is the commitment of its action (``COMMITMENT``)."""

        if self.label is None:
            raise ConfigError("outcome requested before the intervention ended")
        tags = [f"patron:{self.patron_action}", f"rival:{self.rival_action}"]
        if self.strength:
            tags.append(f"strength:{self.strength}")
        if self.backed_down:
            tags.append("patron:backed_down")
        p = self.params
        metrics = {
            f"commit:{p.patron}": COMMITMENT.get(self.patron_action or "", 0.0),
            f"commit:{p.rival}": COMMITMENT.get(self.rival_action or "", 0.0),
        }
        return Outcome(self.label, tuple(tags), metrics)

    def observation(self, role: str) -> tuple[float, ...]:
        """Ten features: whether *role* is the patron, the report if it sees it, one-hot
        patron and rival actions, and whether the patron is replying to a confrontation."""

        is_patron = role == self.params.patron
        sees_report = is_patron or self.params.rival_observes_report
        report = {"alarming": 1.0, "calm": 0.0}.get(self.report or "", 0.5) if sees_report else 0.5
        return (
            1.0 if is_patron else 0.0,
            report,
            *(1.0 if self.patron_action == a else 0.0 for a in PATRON_ACTIONS),
            *(1.0 if self.rival_action == a else 0.0 for a in RIVAL_ACTIONS),
            1.0 if self.phase == "reply" else 0.0,
        )


def build(config: Mapping[str, Any]) -> InterventionGame:
    """The intervention game of *config* (keys: :class:`InterventionParams`)."""

    return InterventionGame(InterventionParams.from_config(config))


__all__ = [
    "COMMITMENT",
    "NAME",
    "OUTCOMES",
    "PATRON_ACTIONS",
    "RIVAL_ACTIONS",
    "InterventionGame",
    "InterventionParams",
    "InterventionState",
    "build",
]
