"""Stages of a crisis script: the chance and decision nodes of its extensive form."""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, Any

from ...games.protocol import CHANCE, PUBLIC, ImmutableParams
from ...records import Record
from ..common import (
    ConfigError,
    Section,
    as_mapping,
    clamp,
    distribution,
    finite,
    identifier,
    probability,
    reject_unknown,
)
from .params import (
    HIGH_LOW_TRAITS,
    OUTCOMES,
    MoveOption,
    Option,
    PartnerOption,
    SignalOption,
    read_options,
)

if TYPE_CHECKING:
    from .game import CrisisState

Draws = tuple[tuple[str, float], ...]  # (chance outcome, probability)


class ScriptSoFar:
    """The prefix of a script, as far as later stages' validity depends on it."""

    def __init__(self, side_of: dict[str, str]) -> None:
        self.side_of = side_of  # role -> side
        self.drawn: set[tuple[str, str]] = set()  # (role, trait) already drawn
        self.audiences: dict[str, tuple[str, ...] | None] = {}  # role -> observers of its reports


class Stage(ImmutableParams, Record):
    """A node of the script: a chance move, or a decision by ``actor``.

    Stages are visited in order; one that does not apply in the current state is passed
    over. ``kind`` is the stage's token in information states and outcome tags.
    """

    kind = ""
    actor = ""

    def applies(self, state: CrisisState) -> bool:
        """Whether the node is reached in *state*."""

        return True

    def pass_over(self, state: CrisisState) -> None:
        """The effect of a node that is not reached (none by default)."""

    def player(self, state: CrisisState) -> str:
        """The mover: ``actor``, or ``CHANCE``."""

        return self.actor

    def legal_actions(self, state: CrisisState) -> tuple[str, ...]:
        """The mover's action set (empty at a chance node)."""

        return ()

    def draws(self, state: CrisisState) -> Draws:
        """The chance distribution, zero-probability outcomes included."""

        return ()

    def apply(self, state: CrisisState, action: str) -> None:
        """The transition on *action* (or on a chance outcome)."""

        raise NotImplementedError

    def check(self, script: ScriptSoFar) -> None:
        """Raise if the stage is invalid after the stages before it."""


class ChanceStage(Stage):
    """A chance node."""

    def player(self, state: CrisisState) -> str:
        return CHANCE


class RecordedDraw(ChanceStage):
    """A chance node whose outcome is recorded even when it is degenerate.

    With a single outcome of positive probability there is no node, but the outcome is
    still stored and logged to its observers.
    """

    def applies(self, state: CrisisState) -> bool:
        return sum(1 for _, p in self.draws(state) if p > 0.0) > 1

    def pass_over(self, state: CrisisState) -> None:
        self.apply(state, next(value for value, p in self.draws(state) if p > 0.0))


# ----------------------------------------------------------------- private types
class TypeDraw(RecordedDraw):
    """Nature draws one component of a role's private type; only that role observes it.

    A ``types`` stage lists ``roles: {role: {...}}``, each with any of:

    ``p_high_resolve``     P(resolve = high); a high-resolve role's war payoff is scaled
                           by its ``high_resolve_war_factor``.
    ``p_high_readiness``   P(readiness = high); high readiness raises accident risk.
    ``stakes``             ``{level: {weight, offsets}}``: a level's ``offsets`` are added
                           to the role's own outcome values.

    **Heterogeneous priors.** ``resolve_beliefs`` and ``readiness_beliefs``
    (``{role: p}``) and ``stakes_beliefs`` (``{role: {level: weight}}``) give another
    role its own prior over the component, which may differ from nature's and may put
    weight on a level nature never draws.
    """

    kind = "type"

    def __init__(
        self,
        actor: str,
        trait: str,
        chances: dict[str, float],
        beliefs: dict[str, dict[str, float]],
        offsets: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.actor = actor
        self.trait = trait  # resolve, readiness or stakes
        self.chances = chances  # level -> probability (nature's; a role's prior in its view)
        self.beliefs = beliefs  # role -> its prior over the levels
        self.offsets = offsets or {}  # stakes level -> outcome label -> payoff shift

    def draws(self, state: CrisisState) -> Draws:
        return tuple(self.chances.items())

    def apply(self, state: CrisisState, action: str) -> None:
        state.types[self.actor][self.trait] = action
        state.log(f"type:{self.actor}:{self.trait}={action}", visible_to=[self.actor])

    def as_believed_by(self, role: str) -> TypeDraw:
        """The draw under *role*'s prior, where it holds one."""

        prior = self.beliefs.get(role)
        return self if prior is None else self.replace(chances=prior)

    def check(self, script: ScriptSoFar) -> None:
        """A role's stakes are drawn at most once."""

        if self.trait == "stakes" and (self.actor, "stakes") in script.drawn:
            raise ConfigError(f"{self.actor}'s stakes are drawn twice")
        script.drawn.add((self.actor, self.trait))

    @classmethod
    def from_config(cls, s: Section, role_ids: Collection[str]) -> list[Stage]:
        """A ``types`` stage as one draw per listed trait, role by role."""

        stages: list[Stage] = []
        name = s.at("roles")
        for role_id, data in as_mapping(name, s.get("roles")).items():
            if role_id not in role_ids:
                raise ConfigError(f"{name} refers to unknown role {role_id!r}")
            with Section(f"{name}.{role_id}", data) as traits:
                for trait in HIGH_LOW_TRAITS:
                    stages += cls._high_or_low(traits, role_id, trait, role_ids)
                stages += cls._stakes(traits, role_id, role_ids)
        return stages

    @classmethod
    def _high_or_low(
        cls, traits: Section, holder: str, trait: str, role_ids: Collection[str]
    ) -> list[Stage]:
        """The draw of a binary trait, if ``p_high_<trait>`` is given."""

        chance, believers = f"p_high_{trait}", f"{trait}_beliefs"
        if not traits.has(chance):
            if traits.has(believers):
                raise ConfigError(f"{traits.at(believers)} needs {chance}")
            return []
        beliefs = {
            role: _high_low(probability(f"{traits.at(believers)}.{role}", p))
            for role, p in _believers(traits, believers, holder, role_ids).items()
        }
        return [cls(holder, trait, _high_low(traits.probability(chance)), beliefs)]

    @classmethod
    def _stakes(cls, traits: Section, holder: str, role_ids: Collection[str]) -> list[Stage]:
        """The draw of the role's stakes level, if ``stakes`` is given."""

        if not traits.has("stakes"):
            if traits.has("stakes_beliefs"):
                raise ConfigError(f"{traits.at('stakes_beliefs')} needs stakes")
            return []
        name = traits.at("stakes")
        raw = as_mapping(name, traits.get("stakes"))
        if len(raw) < 2:
            raise ConfigError(f"{name} needs at least two levels")
        levels = tuple(identifier(name, level) for level in raw)
        weights: dict[str, Any] = {}
        offsets: dict[str, dict[str, float]] = {}
        for level in levels:
            with Section(f"{name}.{level}", raw[level]) as s:
                weights[level] = s.get("weight")
                shifts = as_mapping(s.at("offsets"), s.get("offsets", {}))
                reject_unknown(s.at("offsets"), shifts, OUTCOMES)
                offsets[level] = {
                    label: finite(f"{s.at('offsets')}.{label}", x) for label, x in shifts.items()
                }
        nature = distribution(f"{name} (weights)", weights, levels)
        beliefs = {
            role: distribution(f"{traits.at('stakes_beliefs')}.{role}", prior, levels)
            for role, prior in _believers(traits, "stakes_beliefs", holder, role_ids).items()
        }
        return [cls(holder, "stakes", nature, beliefs, offsets)]


def _high_low(p_high: float) -> dict[str, float]:
    """The Bernoulli distribution over (high, low) with P(high) = *p_high*."""

    return {"high": p_high, "low": 1.0 - p_high}


def _believers(traits: Section, key: str, holder: str, role_ids: Collection[str]) -> dict[str, Any]:
    """The roles holding their own prior under *key*: known roles other than the holder."""

    listed = as_mapping(traits.at(key), traits.get(key, {}))
    for role in listed:
        if role not in role_ids or role == holder:
            raise ConfigError(
                f"{traits.at(key)}: {role!r} must be a known role other than the holder"
            )
    return {str(role): prior for role, prior in listed.items()}


class Intel(RecordedDraw):
    """A binary signal about a role's trait, correct with probability ``accuracy``.

    P(alarming | high) = P(reassuring | low) = ``accuracy``. Config keys: ``about``,
    ``dimension`` (``resolve``, the default, or ``readiness``), ``accuracy``, and
    ``observers`` (the roles that see the signal; public if omitted).
    """

    kind = "intel"

    def __init__(
        self, actor: str, trait: str, accuracy: float, observers: tuple[str, ...] | None
    ) -> None:
        self.actor = actor  # the role the signal is about
        self.trait = trait  # resolve or readiness (config key ``dimension``)
        self.accuracy = accuracy
        self.observers = observers  # None: public

    def draws(self, state: CrisisState) -> Draws:
        level = state.types[self.actor].get(self.trait)
        p_alarm = self.accuracy if level == "high" else 1.0 - self.accuracy
        return (("alarming", p_alarm), ("reassuring", 1.0 - p_alarm))

    def apply(self, state: CrisisState, action: str) -> None:
        state.reports[(self.actor, self.trait)] = action
        visible = PUBLIC if self.observers is None else self.observers
        state.log(f"intel:{self.actor}={action}", visible_to=visible)
        state.tags.append(f"intel:{self.actor}={action}")

    def shown_to(self, role: str) -> bool:
        """Whether *role* observes the signal."""

        return self.observers is None or role in self.observers

    def check(self, script: ScriptSoFar) -> None:
        """Signals about a role share one audience and follow the draw they are about.

        One audience makes the observation a function of the information set.
        """

        about = self.actor
        if script.audiences.setdefault(about, self.observers) != self.observers:
            raise ConfigError(f"intel about {about} has two different audiences")
        if (about, self.trait) not in script.drawn:
            raise ConfigError(f"intel about {about}.{self.trait} precedes its type draw")

    @classmethod
    def from_config(cls, s: Section, role_ids: Collection[str]) -> list[Stage]:
        observers = None
        if s.get("observers", None) is not None:
            observers = tuple(s.sequence("observers"))
            if not set(observers) <= set(role_ids):
                raise ConfigError(f"{s.at('observers')} refers to unknown roles")
        trait = s.choice("dimension", HIGH_LOW_TRAITS, "resolve")
        return [cls(s.role("about", role_ids), trait, s.probability("accuracy"), observers)]


# ---------------------------------------------------------------------- decisions
class OptionStage(Stage):
    """A decision node over listed options. Config keys: ``actor``, ``options``.

    Choosing an option is public, charges the option's listed costs, applies its effect
    (:meth:`take`) and leaves the tag ``<actor>:<kind>=<id>``.
    """

    def __init__(self, actor: str, options: dict[str, Any]) -> None:
        self.actor = actor
        self.options = options  # id -> option, in config order

    def legal_actions(self, state: CrisisState) -> tuple[str, ...]:
        return tuple(self.options)

    def apply(self, state: CrisisState, action: str) -> None:
        option: Option = self.options[action]
        state.log(f"{self.kind}:{self.actor}:{action}")
        for role, cost in option.costs.items():
            state.charge(role, cost)
        self.take(state, option)
        state.tags.append(f"{self.actor}:{self.kind}={action}")

    def take(self, state: CrisisState, option: Any) -> None:
        """The option's effect on the state."""

        raise NotImplementedError


class Signal(OptionStage):
    """A public message that sets the sender's stance (:class:`SignalOption`).

    A hard stance is a commitment: conceding after it costs the sender its audience
    cost (Fearon 1994).
    """

    kind = "signal"

    def take(self, state: CrisisState, option: SignalOption) -> None:
        actor = self.actor
        if option.stance in ("hard", "soft"):
            state.stance[actor] = option.stance
        elif option.stance == "flip" and actor in state.stance:
            state.stance[actor] = "soft" if state.stance[actor] == "hard" else "hard"
        state.signal_cost[actor] = state.signal_cost.get(actor, 0.0) + option.cost

    @classmethod
    def from_config(cls, s: Section, role_ids: Collection[str]) -> list[Stage]:
        actor = s.role("actor", role_ids)
        return [cls(actor, read_options(s, SignalOption.from_config, role_ids))]


class Move(OptionStage):
    """A move on the escalation ladder (:class:`MoveOption`).

    Conceding ends the game for the other side; escalating raises the rung, capped at
    ``max_rung``; a deal is an offer the other side may accept; fighting ends the game in
    limited war. A neutral role can neither concede nor offer a deal.
    """

    kind = "move"

    def take(self, state: CrisisState, option: MoveOption) -> None:
        side = state.side_of(self.actor)
        effect = _MOVE_EFFECTS.get(option.kind)  # holding changes nothing
        if effect is not None:
            effect(state, self.actor, side, option)
        if option.control > 0.0:  # a control order binds the side's officers
            state.control[side] = max(state.control[side], option.control)

    def check(self, script: ScriptSoFar) -> None:
        bargains = any(o.kind in ("concede", "deal") for o in self.options.values())
        if script.side_of[self.actor] == "neutral" and bargains:
            raise ConfigError(f"neutral role {self.actor!r} cannot concede or propose deals")

    @classmethod
    def from_config(cls, s: Section, role_ids: Collection[str]) -> list[Stage]:
        actor = s.role("actor", role_ids)
        return [cls(actor, read_options(s, MoveOption.from_config, role_ids))]


def _concede(state: CrisisState, actor: str, side: str, option: MoveOption) -> None:
    """The other side prevails; backing down after a hard stance is tagged (audience cost)."""

    state.conceded_side = side
    state.label = "b_prevails" if side == "a" else "a_prevails"
    if state.stance.get(actor) == "hard":
        state.tags.append(f"{actor}:backed_down_after_commitment")


def _escalate(state: CrisisState, actor: str, side: str, option: MoveOption) -> None:
    """rung' = min(max_rung, rung + rungs); the side is charged the rungs it added."""

    new_rung = min(state.params.max_rung, state.rung + option.rungs)
    state.added[side] += new_rung - state.rung
    state.rung = new_rung


def _deal(state: CrisisState, actor: str, side: str, option: MoveOption) -> None:
    """The side puts a deal on the table for the other to accept."""

    state.deal_by = side


def _fight(state: CrisisState, actor: str, side: str, option: MoveOption) -> None:
    """The crisis ends in limited war."""

    state.label = "limited_war"


_MOVE_EFFECTS = {"concede": _concede, "escalate": _escalate, "deal": _deal, "fight": _fight}


class Partner(OptionStage):
    """A third party backs or pressures a side (:class:`PartnerOption`).

    Pressure is a cost the target side bears unless it concedes; the option may also
    scale the risk of accidents and incidents.
    """

    kind = "partner"

    def take(self, state: CrisisState, option: PartnerOption) -> None:
        if option.continue_cost > 0.0:
            state.pressure.append((option.target, option.continue_cost))
        state.risk_multiplier *= option.risk_multiplier

    @classmethod
    def from_config(cls, s: Section, role_ids: Collection[str]) -> list[Stage]:
        actor = s.role("actor", role_ids)
        return [cls(actor, read_options(s, PartnerOption.from_config, role_ids))]


class Respond(Stage):
    """The response to a pending deal: accept (compromise) or reject. Config key: ``actor``.

    Reached only when the other side has an offer on the table. A neutral role cannot
    respond.
    """

    kind = "respond"

    def __init__(self, actor: str) -> None:
        self.actor = actor

    def applies(self, state: CrisisState) -> bool:
        return state.deal_by is not None and state.deal_by != state.side_of(self.actor)

    def legal_actions(self, state: CrisisState) -> tuple[str, ...]:
        return ("accept", "reject")

    def apply(self, state: CrisisState, action: str) -> None:
        state.log(f"respond:{self.actor}:{action}")
        if action == "accept":
            state.label = "compromise"
            state.tags.append("deal")
        else:
            state.deal_by = None

    def check(self, script: ScriptSoFar) -> None:
        if script.side_of[self.actor] == "neutral":
            raise ConfigError(f"neutral role {self.actor!r} cannot respond to deals")

    @classmethod
    def from_config(cls, s: Section, role_ids: Collection[str]) -> list[Stage]:
        return [cls(s.role("actor", role_ids))]


# ------------------------------------------------------------------ risk by chance
class Accident(ChanceStage):
    """Inadvertent war: the threat that leaves something to chance (Schelling 1960).

    With ``r`` the rung, ``m`` the partners' risk multiplier and ``k`` the
    ``readiness_multiplier`` when some role's readiness is high (else 1)::

        P(war)           = clamp(min(cap, base + per_rung r) k m, 0, 1)
        P(nuclear | war) = clamp(nuclear_base + nuclear_per_rung r, 0, 1)
    """

    kind = "accident"

    def __init__(
        self,
        base: float,
        per_rung: float,
        cap: float,
        nuclear_base: float,
        nuclear_per_rung: float,
        readiness_multiplier: float,
    ) -> None:
        self.base = base
        self.per_rung = per_rung
        self.cap = cap
        self.nuclear_base = nuclear_base
        self.nuclear_per_rung = nuclear_per_rung
        self.readiness_multiplier = readiness_multiplier

    def war_chances(self, state: CrisisState) -> tuple[float, float]:
        """(P(war), P(nuclear war)) in *state*."""

        p_war = min(self.cap, self.base + self.per_rung * state.rung)
        if state.any_ready():
            p_war *= self.readiness_multiplier
        p_war = clamp(p_war * state.risk_multiplier, 0.0, 1.0)
        nuclear_share = clamp(self.nuclear_base + self.nuclear_per_rung * state.rung, 0.0, 1.0)
        return p_war, p_war * nuclear_share

    def applies(self, state: CrisisState) -> bool:
        return self.war_chances(state)[0] > 0.0

    def draws(self, state: CrisisState) -> Draws:
        p_war, p_nuclear = self.war_chances(state)
        return (
            ("calm", 1.0 - p_war),
            ("limited_war", p_war - p_nuclear),
            ("nuclear_war", p_nuclear),
        )

    def apply(self, state: CrisisState, action: str) -> None:
        state.log(f"accident={action}")
        if action != "calm":
            state.label = action
            state.tags.append("accident")

    @classmethod
    def from_config(cls, s: Section, role_ids: Collection[str]) -> list[Stage]:
        return [
            cls(
                base=s.probability("base", 0.0),
                per_rung=s.nonnegative("per_rung", 0.0),
                cap=s.probability("cap", 1.0),
                nuclear_base=s.probability("nuclear_base", 0.0),
                nuclear_per_rung=s.nonnegative("nuclear_per_rung", 0.0),
                readiness_multiplier=s.nonnegative("readiness_multiplier", 1.0),
            )
        ]
