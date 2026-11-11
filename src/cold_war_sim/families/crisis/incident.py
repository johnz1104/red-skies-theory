"""Delegated risk: an incident hands the decision to fire to an officer, then a leader.

One ``incident`` stage expands to up to four nodes: nature draws the incident and the
officer's type; the officer restrains or fires; a leader waits or launches; nature
decides whether a launched limited war turns nuclear.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING

from ...games.protocol import ImmutableParams
from ...records import Record
from ..common import ConfigError, Section, as_mapping, clamp, distribution, identifier
from .params import OUTCOMES, SIDES
from .stages import ChanceStage, Draws, Stage

if TYPE_CHECKING:
    from .game import CrisisState


class Temperament(ImmutableParams, Record):
    """An officer type, drawn with probability ``weight`` and seen only by the officer.

    Restraining costs the type ``hold_cost`` (its belief that war has begun, times its
    duty); firing costs it ``fire_cost`` (its belief that the alarm is false, times the
    cost of a false shot). A steady type has ``fire_cost > hold_cost``; a by-the-book
    type's ``hold_cost`` exceeds its loss from war.
    """

    def __init__(self, weight: float, hold_cost: float, fire_cost: float) -> None:
        self.weight = weight
        self.hold_cost = hold_cost
        self.fire_cost = fire_cost


class LeaderChoice(ImmutableParams, Record):
    """The leader's decision node after an officer fires: ``launch`` or ``wait``.

    Keys of an incident's ``fire`` mapping:

    ``leader``           the deciding role; not the officer.
    ``launch``           the outcome of launching (default ``nuclear_war``).
    ``wait``             the outcome of waiting, or ``continue`` (the default).
    ``hold_cost``        the leader's cost of waiting.
    ``launch_nuclear``   P(nuclear | a launched ``limited_war``) when some role's
                         readiness is high; otherwise the war stays limited.
    """

    def __init__(
        self,
        leader: str,
        launch: str = "nuclear_war",
        wait: str = "continue",
        hold_cost: float = 0.0,
        launch_nuclear: float = 0.0,
    ) -> None:
        self.leader = leader
        self.launch = launch
        self.wait = wait
        self.hold_cost = hold_cost
        self.launch_nuclear = launch_nuclear

    def launch_may_escalate(self, state: CrisisState) -> bool:
        """Whether a launch leads to the chance node between limited and nuclear war."""

        return self.launch == "limited_war" and self.launch_nuclear > 0.0 and state.any_ready()

    @classmethod
    def from_config(cls, f: Section, officer: str, role_ids: Collection[str]) -> LeaderChoice:
        """Read the leader's choices after an officer fires, checking roles and endings."""

        leader = f.get("leader")
        if leader not in role_ids or leader == officer:
            raise ConfigError(f"{f.at('leader')} must be a known role other than the officer")
        launch = f.get("launch", "nuclear_war")
        wait = f.get("wait", "continue")
        if launch not in OUTCOMES or (wait != "continue" and wait not in OUTCOMES):
            raise ConfigError(f"{f.name}: launch and wait must be outcome labels (or continue)")
        return cls(
            leader=str(leader),
            launch=str(launch),
            wait=str(wait),
            hold_cost=f.nonnegative("hold_cost", 0.0),
            launch_nuclear=f.probability("launch_nuclear", 0.0),
        )


class Incident(ImmutableParams, Record):
    """The parameters of an ``incident`` stage.

    With ``r`` the rung, ``m`` the partners' risk multiplier and ``k`` the
    ``readiness_multiplier`` when some role's readiness is high (else 1)::

        P(incident) = clamp(min(cap, base + per_rung r) k m, 0, 1)

    and 0 under ``needs_readiness`` when no role's readiness is high.

    ``officer``        the role that restrains or fires.
    ``command``        the side whose control orders bind the officer (default
                       ``neutral``): a move's ``control`` share ``c`` scales the
                       officer's hold cost by ``1 - c``.
    ``temperaments``   ``{name: {weight, hold_cost, fire_cost}}``: the officer's type
                       distribution (:class:`Temperament`; by default one costless type).
    ``hold_per_rung``  added to the hold cost per rung.
    ``fire``           an outcome label (firing ends the game in it), or a mapping that
                       gives a leader the next move (:class:`LeaderChoice`).
    """

    def __init__(
        self,
        officer: str,
        command: str,
        base: float,
        per_rung: float,
        cap: float,
        readiness_multiplier: float,
        needs_readiness: bool,
        temperaments: dict[str, Temperament],
        hold_per_rung: float,
        fire_outcome: str | None,
        leader_choice: LeaderChoice | None,
    ) -> None:
        self.officer = officer
        self.command = command
        self.base = base
        self.per_rung = per_rung
        self.cap = cap
        self.readiness_multiplier = readiness_multiplier
        self.needs_readiness = needs_readiness
        self.temperaments = temperaments
        self.hold_per_rung = hold_per_rung
        self.fire_outcome = fire_outcome  # the outcome of firing; None when a leader decides
        self.leader_choice = leader_choice

    def chance(self, state: CrisisState) -> float:
        """P(incident) in *state*."""

        p = min(self.cap, self.base + self.per_rung * state.rung)
        ready = state.any_ready()
        if self.needs_readiness and not ready:
            return 0.0
        if ready:
            p *= self.readiness_multiplier
        return clamp(p * state.risk_multiplier, 0.0, 1.0)

    @classmethod
    def from_config(cls, s: Section, role_ids: Collection[str]) -> Incident:
        """Read incident odds, officer types and the consequences of firing."""

        officer = s.role("officer", role_ids)
        command = s.choice("command", SIDES, "neutral")
        temperaments = _temperaments(s)
        fire_outcome, leader_choice = None, None
        if isinstance(s.get("fire"), str):
            fire_outcome = str(s.get("fire"))
            if fire_outcome not in OUTCOMES:
                raise ConfigError(f"{s.at('fire')} must be an outcome label or a leader's alarm")
        else:
            with s.section("fire") as f:
                leader_choice = LeaderChoice.from_config(f, officer, role_ids)
        return cls(
            officer=officer,
            command=command,
            base=s.probability("base", 0.0),
            per_rung=s.nonnegative("per_rung", 0.0),
            cap=s.probability("cap", 1.0),
            readiness_multiplier=s.nonnegative("readiness_multiplier", 1.0),
            needs_readiness=s.flag("needs_readiness"),
            temperaments=temperaments,
            hold_per_rung=s.nonnegative("hold_per_rung", 0.0),
            fire_outcome=fire_outcome,
            leader_choice=leader_choice,
        )


def _temperaments(s: Section) -> dict[str, Temperament]:
    """The officer's type distribution; the weights sum to 1."""

    name = s.at("temperaments")
    raw = as_mapping(name, s.get("temperaments", {"steady": {}}))
    if not raw:
        raise ConfigError(f"{name} needs at least one level")
    weights: dict[str, object] = {}
    costs: dict[str, tuple[float, float]] = {}
    for level, spec in raw.items():
        with Section(f"{name}.{level}", spec) as t:
            weights[identifier(name, level)] = t.get("weight", 1.0)
            costs[level] = (t.nonnegative("hold_cost", 0.0), t.nonnegative("fire_cost", 0.0))
    nature = distribution(f"{name} (weights)", weights, tuple(weights))
    return {level: Temperament(weight, *costs[level]) for level, weight in nature.items()}


class IncidentDraw(ChanceStage):
    """Nature draws whether an incident occurs and, if so, the officer's type.

    The incident is public; the type is the officer's private information.
    """

    kind = "incident"

    def __init__(self, incident: Incident) -> None:
        self.incident = incident
        self.actor = incident.officer

    def applies(self, state: CrisisState) -> bool:
        return self.incident.chance(state) > 0.0

    def draws(self, state: CrisisState) -> Draws:
        p = self.incident.chance(state)
        return (
            ("calm", 1.0 - p),
            *(
                (f"incident:{name}", p * temperament.weight)
                for name, temperament in self.incident.temperaments.items()
            ),
        )

    def apply(self, state: CrisisState, action: str) -> None:
        if action == "calm":
            state.log("incident=calm")
            return
        officer = self.incident.officer
        state.temperament[officer] = action.partition(":")[2]
        state.alarm = "officer"
        state.log("incident")
        state.log(f"temperament={state.temperament[officer]}", visible_to=[officer])
        state.tags.append(f"incident:{officer}")


class OfficerAlarm(Stage):
    """The officer's decision: ``restrain`` or ``fire``.

    Restraining costs it ``(hold_cost + hold_per_rung r)(1 - c)``, with ``c`` its
    command's control share; firing costs it ``fire_cost`` and ends the game in the
    incident's outcome, or gives the leader the next move.
    """

    kind = "alarm"

    def __init__(self, incident: Incident) -> None:
        self.incident = incident
        self.actor = incident.officer

    def applies(self, state: CrisisState) -> bool:
        return state.alarm == "officer"

    def legal_actions(self, state: CrisisState) -> tuple[str, ...]:
        return ("restrain", "fire")

    def apply(self, state: CrisisState, action: str) -> None:
        incident, officer = self.incident, self.actor
        temperament = incident.temperaments[state.temperament[officer]]
        state.log(f"alarm:{officer}:{action}")
        if action == "restrain":
            hold = temperament.hold_cost + incident.hold_per_rung * state.rung
            state.charge(officer, hold * (1.0 - state.control[incident.command]))
            state.alarm = None
        else:
            state.charge(officer, temperament.fire_cost)
            if incident.fire_outcome is None:
                state.alarm = "leader"
            else:
                state.label, state.alarm = incident.fire_outcome, None
        state.tags.append(f"{officer}:alarm={action}")


class LeaderAlarm(Stage):
    """The leader's decision after an officer fires: ``wait`` (at its hold cost) or ``launch``."""

    kind = "alarm"

    def __init__(self, choice: LeaderChoice) -> None:
        self.choice = choice
        self.actor = choice.leader

    def applies(self, state: CrisisState) -> bool:
        return state.alarm == "leader"

    def legal_actions(self, state: CrisisState) -> tuple[str, ...]:
        return ("wait", "launch")

    def apply(self, state: CrisisState, action: str) -> None:
        choice, leader = self.choice, self.actor
        state.log(f"alarm:{leader}:{action}")
        if action == "launch" and choice.launch_may_escalate(state):
            state.alarm = "escalation"  # the answer may turn nuclear
        elif action == "launch":
            state.label, state.alarm = choice.launch, None
        else:
            state.charge(leader, choice.hold_cost)
            if choice.wait != "continue":
                state.label = choice.wait
            state.alarm = None
        state.tags.append(f"{leader}:alarm={action}")


class Escalation(ChanceStage):
    """Nature turns a launched limited war nuclear with probability ``launch_nuclear``."""

    kind = "escalation"

    def __init__(self, choice: LeaderChoice) -> None:
        self.choice = choice
        self.actor = choice.leader

    def applies(self, state: CrisisState) -> bool:
        return state.alarm == "escalation"

    def draws(self, state: CrisisState) -> Draws:
        x = self.choice.launch_nuclear
        return (("limited_war", 1.0 - x), ("nuclear_war", x))

    def apply(self, state: CrisisState, action: str) -> None:
        state.log(f"escalation={action}")
        state.label, state.alarm = action, None


def incident_stages(s: Section, role_ids: Collection[str]) -> list[Stage]:
    """The nodes of an ``incident`` stage: the draw, the officer's decision and, where
    configured, the leader's decision and the escalation draw."""

    incident = Incident.from_config(s, role_ids)
    stages: list[Stage] = [IncidentDraw(incident), OfficerAlarm(incident)]
    choice = incident.leader_choice
    if choice is not None:
        stages.append(LeaderAlarm(choice))
        if choice.launch_nuclear > 0.0:
            stages.append(Escalation(choice))
    return stages
