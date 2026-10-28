"""Intervention in a client state's upheaval, with a rival who may interfere.

Sequence:

1. Chance draws the movement's strength (weak or strong).
2. Chance draws an intelligence report on that strength, seen by the patron
   (and by the rival if ``rival_observes_report``).
3. The patron chooses ``accommodate``, ``crackdown`` (local forces) or
   ``intervene`` (its own forces).
4. The rival chooses ``abstain``, ``condemn`` or ``covert_aid``; if the patron
   intervened it may also ``confront``.
5. After a confrontation the patron chooses ``back_down`` or ``stand_firm``;
   standing firm risks a direct ``clash``.
6. Chance resolves the upheaval with a contest-success function: force against
   movement strength (boosted by rival aid). Accommodation instead resolves
   to reform or defection with strength-dependent probability.

Outcome labels: ``reformed``, ``crushed``, ``defected``, ``quagmire``,
``clash``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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
from .common import (
    ConfigError,
    Risk,
    ValueTable,
    as_mapping,
    clamp,
    finite,
    identifier,
    nonnegative,
    probability,
    prospect,
    reject_unknown,
    require,
    risk_attitude,
    value_table,
)

NAME = "intervention"
OUTCOMES = ("reformed", "crushed", "defected", "quagmire", "clash")
# Commitment of each action: own forces 1, local forces or covert aid 1/3, otherwise 0.
COMMITMENT = {"intervene": 1.0, "confront": 1.0, "crackdown": 1 / 3, "covert_aid": 1 / 3}
PATRON_ACTIONS = ("accommodate", "crackdown", "intervene")
RIVAL_ACTIONS = ("abstain", "condemn", "covert_aid", "confront")


@dataclass(frozen=True, slots=True)
class InterventionParams(ImmutableParams):
    patron: str
    rival: str
    p_strong: float
    report_accuracy: float
    rival_observes_report: bool
    movement_weak: float
    movement_strong: float
    crackdown_force: float
    intervention_force: float
    covert_boost: float
    confront_boost: float
    gamma: float
    accommodate_defect_weak: float
    accommodate_defect_strong: float
    aid_defect_shift: float
    quagmire_prob: float
    clash_risk: float
    patron_values: ValueTable
    rival_values: ValueTable
    patron_costs: tuple[float, float, float]  # crackdown, intervene, back_down
    rival_costs: tuple[float, float, float]  # condemn, covert_aid, confront
    patron_risk: Risk | None = None  # prospect theory on the total payoff (optional)
    rival_risk: Risk | None = None


class InterventionGame:
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
    def __init__(self, params: InterventionParams) -> None:
        super().__init__((params.patron, params.rival))
        self.params = params
        self.phase = "strength"
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
        if self.phase in ("strength", "report", "clash", "resolve"):
            return CHANCE
        if self.phase in ("patron", "reply"):
            return self.params.patron
        return self.params.rival

    def is_terminal(self) -> bool:
        return self.label is not None

    def legal_actions(self) -> tuple[str, ...]:
        if self.phase == "patron":
            return PATRON_ACTIONS
        if self.phase == "rival":
            return RIVAL_ACTIONS if self.patron_action == "intervene" else RIVAL_ACTIONS[:3]
        if self.phase == "reply":
            return ("back_down", "stand_firm")
        return ()

    def _pairs(self) -> tuple[tuple[str, float], ...]:
        p = self.params
        if self.phase == "strength":
            return (("strong", p.p_strong), ("weak", 1.0 - p.p_strong))
        if self.phase == "report":
            alarm = p.report_accuracy if self.strength == "strong" else 1.0 - p.report_accuracy
            return (("alarming", alarm), ("calm", 1.0 - alarm))
        if self.phase == "clash":
            return (("clash", p.clash_risk), ("no_clash", 1.0 - p.clash_risk))
        aided = self.rival_action in ("covert_aid", "confront")
        if self.patron_action == "accommodate":
            base = (
                p.accommodate_defect_strong
                if self.strength == "strong"
                else p.accommodate_defect_weak
            )
            defect = clamp(base + (p.aid_defect_shift if aided else 0.0), 0.0, 1.0)
            return (("defected", defect), ("reformed", 1.0 - defect))
        force = p.intervention_force if self.patron_action == "intervene" else p.crackdown_force
        movement = p.movement_strong if self.strength == "strong" else p.movement_weak
        if self.rival_action == "covert_aid":
            movement += p.covert_boost
        elif self.rival_action == "confront":
            movement += p.confront_boost
        hold = force**p.gamma / (force**p.gamma + movement**p.gamma) if force > 0 else 0.0
        quagmire = (
            p.quagmire_prob
            if self.patron_action == "intervene" and self.strength == "strong"
            else 0.0
        )
        return (
            ("crushed", hold * (1.0 - quagmire)),
            ("quagmire", hold * quagmire),
            ("defected", 1.0 - hold),
        )

    def chance_outcomes(self) -> tuple[tuple[str, float], ...]:
        if self.current_player() != CHANCE:
            return ()
        return tuple((a, p) for a, p in self._pairs() if p > 0.0)

    def apply_action(self, action: str) -> None:
        player = self.current_player()
        legal = (
            tuple(a for a, _ in self.chance_outcomes())
            if player == CHANCE
            else self.legal_actions()
        )
        check_legal(action, legal)
        self._apply(action)
        self._history.append(action)
        self._skip_certain()

    def _apply(self, action: str) -> None:
        p = self.params
        if self.phase == "strength":
            self.strength = action
            self.phase = "report"
        elif self.phase == "report":
            self.report = action
            observers = (p.patron, p.rival) if p.rival_observes_report else (p.patron,)
            self.log(f"report={action}", visible_to=observers)
            self.phase = "patron"
        elif self.phase == "patron":
            self.patron_action = action
            self.log(f"patron={action}", visible_to=PUBLIC)
            self.phase = "rival"
        elif self.phase == "rival":
            self.rival_action = action
            self.log(f"rival={action}", visible_to=PUBLIC)
            self.phase = "reply" if action == "confront" else "resolve"
        elif self.phase == "reply":
            self.log(f"reply={action}", visible_to=PUBLIC)
            if action == "back_down":
                self.backed_down = True
                self.label = "defected"
            else:
                self.stood_firm = True
                self.phase = "clash"
        elif self.phase == "clash":
            self.log(f"clash={action}", visible_to=PUBLIC)
            if action == "clash":
                self.label = "clash"
            else:
                self.phase = "resolve"
        else:
            self.label = action

    def _skip_certain(self) -> None:
        # Chance phases with a single possible outcome resolve without a node.
        while self.label is None and self.current_player() == CHANCE:
            outcomes = self.chance_outcomes()
            if len(outcomes) != 1:
                return
            self._apply(outcomes[0][0])

    def returns(self) -> dict[str, float]:
        if self.label is None:
            raise ConfigError("returns requested before the intervention ended")
        p = self.params
        patron = p.patron_values[self.label]
        if self.patron_action == "crackdown":
            patron -= p.patron_costs[0]
        elif self.patron_action == "intervene":
            patron -= p.patron_costs[1]
        if self.backed_down:
            patron -= p.patron_costs[2]
        rival = p.rival_values[self.label]
        if self.rival_action in RIVAL_ACTIONS[1:]:
            rival -= p.rival_costs[RIVAL_ACTIONS.index(self.rival_action) - 1]
        return {p.patron: prospect(patron, p.patron_risk), p.rival: prospect(rival, p.rival_risk)}

    def outcome(self) -> Outcome:
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
    keys = (
        "roles",
        "p_strong",
        "report_accuracy",
        "rival_observes_report",
        "movement",
        "force",
        "rival_boost",
        "gamma",
        "accommodate_defect",
        "aid_defect_shift",
        "quagmire_prob",
        "clash_risk",
        "values",
        "costs",
        "risk",
    )
    reject_unknown("intervention", config, keys)
    risk = as_mapping("intervention.risk", config.get("risk", {}))
    reject_unknown("intervention.risk", risk, ("patron", "rival"))
    roles = as_mapping("intervention.roles", require("intervention", config, "roles"))
    reject_unknown("intervention.roles", roles, ("patron", "rival"))
    patron = identifier("intervention.roles.patron", require("roles", roles, "patron"))
    rival = identifier("intervention.roles.rival", require("roles", roles, "rival"))
    if patron == rival:
        raise ConfigError("intervention patron and rival must differ")
    movement = as_mapping("intervention.movement", config.get("movement", {}))
    reject_unknown("intervention.movement", movement, ("weak", "strong"))
    force = as_mapping("intervention.force", config.get("force", {}))
    reject_unknown("intervention.force", force, ("crackdown", "intervene"))
    boost = as_mapping("intervention.rival_boost", config.get("rival_boost", {}))
    reject_unknown("intervention.rival_boost", boost, ("covert_aid", "confront"))
    defect = as_mapping("intervention.accommodate_defect", config.get("accommodate_defect", {}))
    reject_unknown("intervention.accommodate_defect", defect, ("weak", "strong"))
    values = as_mapping("intervention.values", require("intervention", config, "values"))
    reject_unknown("intervention.values", values, ("patron", "rival"))
    costs = as_mapping("intervention.costs", config.get("costs", {}))
    reject_unknown(
        "intervention.costs",
        costs,
        ("crackdown", "intervene", "back_down", "condemn", "covert_aid", "confront"),
    )
    gamma = finite("intervention.gamma", config.get("gamma", 1.0))
    if gamma <= 0.0:
        raise ConfigError("intervention.gamma must be positive")
    params = InterventionParams(
        patron=patron,
        rival=rival,
        p_strong=probability("intervention.p_strong", config.get("p_strong", 0.5)),
        report_accuracy=probability(
            "intervention.report_accuracy", config.get("report_accuracy", 0.7)
        ),
        rival_observes_report=bool(config.get("rival_observes_report", False)),
        movement_weak=nonnegative("movement.weak", movement.get("weak", 0.5)),
        movement_strong=nonnegative("movement.strong", movement.get("strong", 1.5)),
        crackdown_force=nonnegative("force.crackdown", force.get("crackdown", 1.0)),
        intervention_force=nonnegative("force.intervene", force.get("intervene", 3.0)),
        covert_boost=nonnegative("rival_boost.covert_aid", boost.get("covert_aid", 0.3)),
        confront_boost=nonnegative("rival_boost.confront", boost.get("confront", 1.0)),
        gamma=gamma,
        accommodate_defect_weak=probability("accommodate_defect.weak", defect.get("weak", 0.05)),
        accommodate_defect_strong=probability(
            "accommodate_defect.strong", defect.get("strong", 0.4)
        ),
        aid_defect_shift=nonnegative(
            "intervention.aid_defect_shift", config.get("aid_defect_shift", 0.1)
        ),
        quagmire_prob=probability("intervention.quagmire_prob", config.get("quagmire_prob", 0.0)),
        clash_risk=probability("intervention.clash_risk", config.get("clash_risk", 0.1)),
        patron_values=value_table(
            "intervention.values.patron", require("values", values, "patron"), OUTCOMES
        ),
        rival_values=value_table(
            "intervention.values.rival", require("values", values, "rival"), OUTCOMES
        ),
        patron_costs=(
            nonnegative("costs.crackdown", costs.get("crackdown", 0.0)),
            nonnegative("costs.intervene", costs.get("intervene", 0.0)),
            nonnegative("costs.back_down", costs.get("back_down", 0.0)),
        ),
        rival_costs=(
            nonnegative("costs.condemn", costs.get("condemn", 0.0)),
            nonnegative("costs.covert_aid", costs.get("covert_aid", 0.0)),
            nonnegative("costs.confront", costs.get("confront", 0.0)),
        ),
        patron_risk=risk_attitude("risk.patron", risk["patron"]) if "patron" in risk else None,
        rival_risk=risk_attitude("risk.rival", risk["rival"]) if "rival" in risk else None,
    )
    return InterventionGame(params)
