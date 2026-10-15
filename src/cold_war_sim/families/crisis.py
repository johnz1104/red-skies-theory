"""Coercive crisis: an escalation ladder with risk left to chance.

Two sides (``a`` and ``b``) contest a stake; neutral roles may pressure either
side. The interaction is a *stage script* assembled from a small library:

``types``     chance draws each listed role's private resolve (and optionally
              readiness); a high-resolve role finds war less costly. A role may
              also draw private ``stakes``: named levels whose offsets shift its
              own outcome values (what it would gain from each outcome).
``intel``     a noisy report about one role's type, seen by chosen observers.
``signal``    a public message that sets the sender's stance (hard or soft).
              Conceding after a hard stance carries an audience cost.
``move``      concede, hold, escalate by k rungs, propose a deal, or fight.
``respond``   the other side accepts or rejects a pending deal.
``partner``   a role backs or pressures a side; pressure is a cost the side
              pays unless it backs down, and may scale accident risk.
``accident``  chance of inadvertent war, rising with the escalation rung and
              with high readiness; a share of wars is nuclear.

Outcome labels: ``a_prevails``, ``b_prevails``, ``compromise``,
``limited_war``, ``nuclear_war``. Each role values each label; payoffs also
subtract own escalation, signal, audience and pressure costs, and the costs an
option lists for named roles (``costs: {role: x}`` on a move, signal or partner
option, charged when it is taken; negative for a gain).

**Own beliefs.** A type draw may list roles that hold their own prior about it
(``resolve_beliefs``, ``stakes_beliefs``), which may differ from nature's: a
leader's mistaken view of another's resolve or stakes. :meth:`CrisisGame.view`
gives the game as one role sees it, and :class:`~.crisis_beliefs.CrisisBeliefPolicy`
lets each role play its part of the solution of its own view. A level that
nature never draws (weight 0) can still be believed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
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
    ValueTable,
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
    value_table,
)

NAME = "crisis"
OUTCOMES = ("a_prevails", "b_prevails", "compromise", "limited_war", "nuclear_war")
WAR_OUTCOMES = frozenset({"limited_war", "nuclear_war"})
SIDES = ("a", "b", "neutral")
STANCES = ("hard", "soft", "keep", "flip")
MOVE_KINDS = ("concede", "hold", "escalate", "deal", "fight")


@dataclass(frozen=True, slots=True)
class CrisisRole(ImmutableParams):
    id: str
    side: str
    values: ValueTable
    escalation_cost: float = 0.0
    audience_cost: float = 0.0
    high_resolve_war_factor: float = 1.0


@dataclass(frozen=True, slots=True)
class Step(ImmutableParams):
    """One atomic step of an expanded stage script."""

    kind: str  # type | intel | signal | move | respond | partner | accident
    actor: str = ""
    dimension: str = ""
    probability: float = 0.0
    observers: tuple[str, ...] | None = None
    options: tuple[tuple[str, str, float, float], ...] = ()  # (id, kind/stance/side, a, b)
    accident: tuple[float, float, float, float, float, float] = (0.0,) * 6
    levels: tuple[tuple[str, float], ...] = ()  # a stakes draw: (level, nature's weight)
    beliefs: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = ()  # role -> its prior
    offsets: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = ()  # level -> (label, x)
    option_costs: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = ()  # id -> (role, x)

    def draws(self) -> tuple[tuple[str, float], ...]:
        """A type draw's outcomes with nature's probabilities."""

        if self.levels:
            return self.levels
        return (("high", self.probability), ("low", 1.0 - self.probability))

    def believed_by(self, role: str) -> tuple[tuple[str, float], ...] | None:
        return next((prior for holder, prior in self.beliefs if holder == role), None)


@dataclass(frozen=True, slots=True)
class CrisisParams(ImmutableParams):
    roles: tuple[CrisisRole, ...]
    steps: tuple[Step, ...]
    max_rung: int
    final_outcome: str

    def offset(self, role_id: str, level: str | None, label: str) -> float:
        """The holder's offset to *label* at its stakes *level*."""

        if level is None:
            return 0.0
        for step in self.steps:
            if step.kind == "type" and step.dimension == "stakes" and step.actor == role_id:
                for name, table in step.offsets:
                    if name == level:
                        return dict(table).get(label, 0.0)
        return 0.0

    def role(self, role_id: str) -> CrisisRole:
        for role in self.roles:
            if role.id == role_id:
                return role
        raise KeyError(role_id)


class CrisisGame:
    name = NAME
    outcome_labels = OUTCOMES

    def __init__(self, params: CrisisParams) -> None:
        self.params = params

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(role.id for role in self.params.roles)

    @property
    def has_beliefs(self) -> bool:
        return any(step.beliefs for step in self.params.steps)

    def view(self, role: str) -> CrisisGame:
        """The game as *role* sees it: its own priors at the type draws it holds
        beliefs about, nature's elsewhere."""

        steps = []
        for step in self.params.steps:
            prior = step.believed_by(role) if step.kind == "type" else None
            if prior is None:
                steps.append(step)
            elif step.levels:
                steps.append(replace(step, levels=prior))
            else:
                steps.append(replace(step, probability=dict(prior)["high"]))
        return CrisisGame(replace(self.params, steps=tuple(steps)))

    def views(self) -> dict[str, CrisisGame]:
        """Each role's view; roles with the same priors share one game."""

        shared: dict[tuple[object, ...], CrisisGame] = {}
        result = {}
        for role in self.roles:
            signature = tuple(
                step.believed_by(role) if step.kind == "type" else None
                for step in self.params.steps
            )
            if signature not in shared:
                shared[signature] = self.view(role)
            result[role] = shared[signature]
        return result

    def new_initial_state(self) -> CrisisState:
        return CrisisState(self.params)


class CrisisState(LoggedState):
    def __init__(self, params: CrisisParams) -> None:
        super().__init__(role.id for role in params.roles)
        self.params = params
        self.index = 0
        self.types: dict[str, dict[str, str]] = {role.id: {} for role in params.roles}
        self.reports: dict[str, str] = {}
        self.rung = 0
        self.added: dict[str, int] = {"a": 0, "b": 0, "neutral": 0}
        self.stance: dict[str, str] = {}
        self.signal_cost: dict[str, float] = {}
        self.deal_by: str | None = None
        self.pressure: list[tuple[str, float]] = []
        self.action_cost: dict[str, float] = {}
        self.risk_multiplier = 1.0
        self.label: str | None = None
        self.conceded_side: str | None = None
        self.tags: list[str] = []
        self._skip()

    # ----------------------------------------------------------------- queries
    def _step(self) -> Step:
        return self.params.steps[self.index]

    def _side(self, role_id: str) -> str:
        return self.params.role(role_id).side

    def current_player(self) -> str:
        if self.label is not None:
            return TERMINAL
        step = self._step()
        if step.kind in ("type", "intel", "accident"):
            return CHANCE
        return step.actor

    def is_terminal(self) -> bool:
        return self.label is not None

    def legal_actions(self) -> tuple[str, ...]:
        if self.label is not None:
            return ()
        step = self._step()
        if step.kind in ("signal", "move", "partner"):
            return tuple(option[0] for option in step.options)
        if step.kind == "respond":
            return ("accept", "reject")
        return ()

    def _accident_probabilities(self, step: Step) -> tuple[float, float]:
        base, per_rung, cap, nuclear_base, nuclear_per_rung, readiness = step.accident
        p_war = min(cap, base + per_rung * self.rung)
        if any(t.get("readiness") == "high" for t in self.types.values()):
            p_war *= readiness
        p_war = clamp(p_war * self.risk_multiplier, 0.0, 1.0)
        p_nuclear = p_war * clamp(nuclear_base + nuclear_per_rung * self.rung, 0.0, 1.0)
        return p_war, p_nuclear

    def _chance_pairs(self, step: Step) -> tuple[tuple[str, float], ...]:
        if step.kind == "type":
            return step.draws()
        if step.kind == "intel":
            about = self.types[step.actor].get(step.dimension)
            p_alarm = step.probability if about == "high" else 1.0 - step.probability
            return (("alarming", p_alarm), ("reassuring", 1.0 - p_alarm))
        p_war, p_nuclear = self._accident_probabilities(step)
        return (
            ("calm", 1.0 - p_war),
            ("limited_war", p_war - p_nuclear),
            ("nuclear_war", p_nuclear),
        )

    def chance_outcomes(self) -> tuple[tuple[str, float], ...]:
        if self.current_player() != CHANCE:
            return ()
        return tuple((a, p) for a, p in self._chance_pairs(self._step()) if p > 0.0)

    # -------------------------------------------------------------- transition
    def apply_action(self, action: str) -> None:
        player = self.current_player()
        if player == TERMINAL:
            raise ConfigError("cannot act in a terminal crisis state")
        step = self._step()
        if player == CHANCE:
            check_legal(action, tuple(a for a, _ in self.chance_outcomes()))
            self._apply_chance(step, action)
        else:
            check_legal(action, self.legal_actions())
            self._apply_decision(step, action)
        self._history.append(action)
        self.index += 1
        self._skip()

    def _apply_chance(self, step: Step, action: str) -> None:
        if step.kind == "type":
            self.types[step.actor][step.dimension] = action
            self.log(f"type:{step.actor}:{step.dimension}={action}", visible_to=[step.actor])
        elif step.kind == "intel":
            self.reports[step.actor] = action
            visible = PUBLIC if step.observers is None else step.observers
            self.log(f"intel:{step.actor}={action}", visible_to=visible)
            self.tags.append(f"intel:{step.actor}={action}")
        else:
            self.log(f"accident={action}")
            if action != "calm":
                self.label = action
                self.tags.append("accident")

    def _apply_decision(self, step: Step, action: str) -> None:
        actor = step.actor
        option = next((o for o in step.options if o[0] == action), (action, "", 0.0, 0.0))
        self.log(f"{step.kind}:{actor}:{action}")
        for role, cost in dict(step.option_costs).get(action, ()):
            self.action_cost[role] = self.action_cost.get(role, 0.0) + cost
        side = self._side(actor)
        if step.kind == "signal":
            stance = option[1]
            if stance in ("hard", "soft"):
                self.stance[actor] = stance
            elif stance == "flip" and actor in self.stance:
                self.stance[actor] = "soft" if self.stance[actor] == "hard" else "hard"
            self.signal_cost[actor] = self.signal_cost.get(actor, 0.0) + option[2]
            self.tags.append(f"{actor}:signal={action}")
        elif step.kind == "move":
            kind = option[1]
            if kind == "concede":
                self.conceded_side = side
                self.label = "b_prevails" if side == "a" else "a_prevails"
                if self.stance.get(actor) == "hard":
                    self.tags.append(f"{actor}:backed_down_after_commitment")
            elif kind == "escalate":
                new_rung = min(self.params.max_rung, self.rung + int(option[2]))
                self.added[side] += new_rung - self.rung
                self.rung = new_rung
            elif kind == "deal":
                self.deal_by = side
            elif kind == "fight":
                self.label = "limited_war"
            self.tags.append(f"{actor}:move={action}")
        elif step.kind == "respond":
            if action == "accept":
                self.label = "compromise"
                self.tags.append("deal")
            else:
                self.deal_by = None
        elif step.kind == "partner":
            target_side = option[1]
            if option[2] > 0.0:
                self.pressure.append((target_side, option[2]))
            self.risk_multiplier *= option[3]
            self.tags.append(f"{actor}:partner={action}")

    def _applicable(self, step: Step) -> bool:
        if step.kind == "respond":
            return self.deal_by is not None and self.deal_by != self._side(step.actor)
        if step.kind == "accident":
            return self._accident_probabilities(step)[0] > 0.0
        if step.kind in ("type", "intel"):
            return sum(1 for _, p in self._chance_pairs(step) if p > 0.0) > 1
        return True

    def _skip(self) -> None:
        steps = self.params.steps
        while self.label is None and self.index < len(steps):
            step = steps[self.index]
            if self._applicable(step):
                return
            # A type or report with a single possible value is resolved
            # without a chance node, but is still logged to its observers.
            if step.kind in ("type", "intel"):
                certain = [a for a, p in self._chance_pairs(step) if p > 0.0]
                self._apply_chance(step, certain[0])
            self.index += 1
        if self.label is None:
            self.label = self.params.final_outcome

    # ----------------------------------------------------------------- payoffs
    def returns(self) -> dict[str, float]:
        if self.label is None:
            raise ConfigError("returns requested before the crisis ended")
        result: dict[str, float] = {}
        for role in self.params.roles:
            value = role.values[self.label]
            if self.label in WAR_OUTCOMES and self.types[role.id].get("resolve") == "high":
                value *= role.high_resolve_war_factor
            value += self.params.offset(role.id, self.types[role.id].get("stakes"), self.label)
            if role.side in ("a", "b"):
                value -= role.escalation_cost * self.added[role.side]
            value -= self.signal_cost.get(role.id, 0.0)
            value -= self.action_cost.get(role.id, 0.0)
            if self.conceded_side == role.side and self.stance.get(role.id) == "hard":
                value -= role.audience_cost
            for target_side, cost in self.pressure:
                if role.side == target_side and self.conceded_side != target_side:
                    value -= cost
            result[role.id] = value
        return result

    def outcome(self) -> Outcome:
        if self.label is None:
            raise ConfigError("outcome requested before the crisis ended")
        metrics = {
            "rung": float(self.rung),
            "max_rung": float(self.params.max_rung),
            "escalation_a": float(self.added["a"]),
            "escalation_b": float(self.added["b"]),
        }
        top = self.params.max_rung
        for role in self.params.roles:  # commitment: a stake for taking part, more per rung
            added = self.added[role.side] if role.side in ("a", "b") else 0
            metrics[f"commit:{role.id}"] = (1.0 + added) / (1.0 + top)
        return Outcome(self.label, tuple(self.tags), metrics)

    def observation(self, role: str) -> tuple[float, ...]:
        side = self._side(role)
        own = self.types[role]
        others_hard = any(
            stance == "hard" and self._side(r) not in (side, "neutral")
            for r, stance in self.stance.items()
        )
        report = next(iter(self.reports.values()), None)
        return (
            self.rung / max(1, self.params.max_rung),
            {"high": 1.0, "low": 0.0}.get(own.get("resolve", ""), 0.5),
            {"high": 1.0, "low": 0.0}.get(own.get("readiness", ""), 0.5),
            1.0 if self.stance.get(role) == "hard" else 0.0,
            1.0 if others_hard else 0.0,
            1.0 if self.deal_by is not None and self.deal_by != side else 0.0,
            1.0 if self.deal_by == side else 0.0,
            1.0 if any(target == side for target, _ in self.pressure) else 0.0,
            {"alarming": 1.0, "reassuring": 0.0}.get(report or "", 0.5),
            self.index / max(1, len(self.params.steps)),
            min(1.0, self.risk_multiplier - 1.0) if self.risk_multiplier > 1.0 else 0.0,
            1.0 if side == "a" else 0.0,
        )


# ------------------------------------------------------------------ configuration
def _parse_role(role_id: str, data: Any) -> CrisisRole:
    name = f"crisis.roles.{role_id}"
    mapping = as_mapping(name, data)
    reject_unknown(
        name,
        mapping,
        ("side", "values", "escalation_cost", "audience_cost", "high_resolve_war_factor"),
    )
    side = require(name, mapping, "side")
    if side not in SIDES:
        raise ConfigError(f"{name}.side must be one of {SIDES}")
    return CrisisRole(
        id=identifier(name, role_id),
        side=side,
        values=value_table(f"{name}.values", require(name, mapping, "values"), OUTCOMES),
        escalation_cost=nonnegative(f"{name}.escalation_cost", mapping.get("escalation_cost", 0.0)),
        audience_cost=nonnegative(f"{name}.audience_cost", mapping.get("audience_cost", 0.0)),
        high_resolve_war_factor=nonnegative(
            f"{name}.high_resolve_war_factor", mapping.get("high_resolve_war_factor", 1.0)
        ),
    )


def _parse_options(
    name: str, data: Any, kinds: tuple[str, ...], key: str, role_ids: set[str]
) -> dict[str, Any]:
    """Step fields for a stage's options: the options and any per-role costs."""

    options = []
    costs = []
    for i, raw in enumerate(as_list(name, data)):
        item = dict(as_mapping(f"{name}[{i}]", raw))
        option_id = identifier(f"{name}[{i}].id", require(name, item, "id"))
        listed = as_mapping(f"{name}[{i}].costs", item.pop("costs", {}))
        for role in listed:
            if role not in role_ids:
                raise ConfigError(f"{name}[{i}].costs refers to unknown role {role!r}")
        if listed:
            costs.append(
                (
                    option_id,
                    tuple(
                        (str(role), finite(f"{name}[{i}].costs.{role}", x))
                        for role, x in listed.items()
                    ),
                )
            )
        if key == "kind":
            reject_unknown(f"{name}[{i}]", item, ("id", "kind", "rungs"))
            kind = require(name, item, "kind")
            if kind not in kinds:
                raise ConfigError(f"{name}[{i}].kind must be one of {kinds}")
            rungs = positive_int(f"{name}[{i}].rungs", item.get("rungs", 1))
            options.append((option_id, kind, float(rungs), 0.0))
        elif key == "stance":
            reject_unknown(f"{name}[{i}]", item, ("id", "stance", "cost"))
            stance = item.get("stance", "keep")
            if stance not in kinds:
                raise ConfigError(f"{name}[{i}].stance must be one of {kinds}")
            options.append(
                (option_id, stance, nonnegative(f"{name}[{i}].cost", item.get("cost", 0.0)), 0.0)
            )
        else:
            reject_unknown(
                f"{name}[{i}]", item, ("id", "target", "continue_cost", "risk_multiplier")
            )
            target = item.get("target", "neutral")
            if target not in kinds:
                raise ConfigError(f"{name}[{i}].target must be one of {kinds}")
            options.append(
                (
                    option_id,
                    target,
                    nonnegative(f"{name}[{i}].continue_cost", item.get("continue_cost", 0.0)),
                    nonnegative(f"{name}[{i}].risk_multiplier", item.get("risk_multiplier", 1.0)),
                )
            )
    ids = [o[0] for o in options]
    if not options or len(set(ids)) != len(ids):
        raise ConfigError(f"{name} needs at least one option and unique ids")
    return {"options": tuple(options), "option_costs": tuple(costs)}


def _beliefs(
    name: str, data: Any, holder: str, role_ids: set[str]
) -> tuple[tuple[str, float], ...]:
    """Roles that hold their own probability of a high draw."""

    result = []
    for role, p in as_mapping(name, data).items():
        if role not in role_ids or role == holder:
            raise ConfigError(f"{name}: {role!r} must be a known role other than the holder")
        result.append((str(role), probability(f"{name}.{role}", p)))
    return tuple(result)


def _distribution(name: str, data: Any, levels: tuple[str, ...]) -> tuple[tuple[str, float], ...]:
    mapping = as_mapping(name, data)
    reject_unknown(name, mapping, levels)
    weights = tuple(
        (level, probability(f"{name}.{level}", mapping.get(level, 0.0))) for level in levels
    )
    if abs(sum(w for _, w in weights) - 1.0) > 1e-9:
        raise ConfigError(f"{name} must sum to 1")
    return weights


def _stakes_step(name: str, dims: Mapping[str, Any], holder: str, role_ids: set[str]) -> Step:
    """A private draw of the holder's stakes: named levels with offsets to its values."""

    raw = as_mapping(f"{name}.stakes", dims["stakes"])
    if len(raw) < 2:
        raise ConfigError(f"{name}.stakes needs at least two levels")
    levels = tuple(identifier(f"{name}.stakes", level) for level in raw)
    weights, offsets = {}, []
    for level in levels:
        item = as_mapping(f"{name}.stakes.{level}", raw[level])
        reject_unknown(f"{name}.stakes.{level}", item, ("weight", "offsets"))
        weights[level] = require(f"{name}.stakes.{level}", item, "weight")
        table = as_mapping(f"{name}.stakes.{level}.offsets", item.get("offsets", {}))
        reject_unknown(f"{name}.stakes.{level}.offsets", table, OUTCOMES)
        offsets.append(
            (
                level,
                tuple((label, finite(f"{name}.offsets.{label}", x)) for label, x in table.items()),
            )
        )
    nature = _distribution(f"{name}.stakes (weights)", weights, levels)
    beliefs = []
    for role, prior in as_mapping(f"{name}.stakes_beliefs", dims.get("stakes_beliefs", {})).items():
        if role not in role_ids or role == holder:
            raise ConfigError(
                f"{name}.stakes_beliefs: {role!r} must be a known role other than the holder"
            )
        beliefs.append((str(role), _distribution(f"{name}.stakes_beliefs.{role}", prior, levels)))
    return Step(
        "type", holder, "stakes", levels=nature, beliefs=tuple(beliefs), offsets=tuple(offsets)
    )


def _expand_stage(index: int, data: Any, role_ids: set[str]) -> list[Step]:
    name = f"crisis.script[{index}]"
    stage = as_mapping(name, data)
    kind = require(name, stage, "stage")

    def actor(key: str = "actor") -> str:
        value = require(name, stage, key)
        if value not in role_ids:
            raise ConfigError(f"{name}.{key} refers to unknown role {value!r}")
        return str(value)

    if kind == "types":
        reject_unknown(name, stage, ("stage", "roles"))
        steps = []
        for role_id, dims in as_mapping(f"{name}.roles", require(name, stage, "roles")).items():
            if role_id not in role_ids:
                raise ConfigError(f"{name}.roles refers to unknown role {role_id!r}")
            where = f"{name}.roles.{role_id}"
            dims_map = as_mapping(where, dims)
            reject_unknown(
                where,
                dims_map,
                (
                    "p_high_resolve",
                    "p_high_readiness",
                    "resolve_beliefs",
                    "stakes",
                    "stakes_beliefs",
                ),
            )
            if "p_high_resolve" in dims_map:
                beliefs = _beliefs(
                    f"{where}.resolve_beliefs",
                    dims_map.get("resolve_beliefs", {}),
                    role_id,
                    role_ids,
                )
                steps.append(
                    Step(
                        "type",
                        role_id,
                        "resolve",
                        probability(f"{name}.p_high_resolve", dims_map["p_high_resolve"]),
                        beliefs=tuple(
                            (holder, (("high", p), ("low", 1.0 - p))) for holder, p in beliefs
                        ),
                    )
                )
            elif "resolve_beliefs" in dims_map:
                raise ConfigError(f"{where}.resolve_beliefs needs p_high_resolve")
            if "p_high_readiness" in dims_map:
                steps.append(
                    Step(
                        "type",
                        role_id,
                        "readiness",
                        probability(f"{name}.p_high_readiness", dims_map["p_high_readiness"]),
                    )
                )
            if "stakes" in dims_map:
                steps.append(_stakes_step(where, dims_map, role_id, role_ids))
            elif "stakes_beliefs" in dims_map:
                raise ConfigError(f"{where}.stakes_beliefs needs stakes")
        return steps
    if kind == "intel":
        reject_unknown(name, stage, ("stage", "about", "dimension", "observers", "accuracy"))
        observers_raw = stage.get("observers")
        observers = None
        if observers_raw is not None:
            observers = tuple(as_list(f"{name}.observers", observers_raw))
            if not set(observers) <= role_ids:
                raise ConfigError(f"{name}.observers refers to unknown roles")
        dimension = stage.get("dimension", "resolve")
        if dimension not in ("resolve", "readiness"):
            raise ConfigError(f"{name}.dimension must be resolve or readiness")
        return [
            Step(
                "intel",
                actor("about"),
                dimension,
                probability(f"{name}.accuracy", require(name, stage, "accuracy")),
                observers,
            )
        ]
    if kind == "signal":
        reject_unknown(name, stage, ("stage", "actor", "options"))
        return [
            Step(
                "signal",
                actor(),
                **_parse_options(
                    f"{name}.options",
                    require(name, stage, "options"),
                    STANCES,
                    "stance",
                    role_ids,
                ),
            )
        ]
    if kind == "move":
        reject_unknown(name, stage, ("stage", "actor", "options"))
        return [
            Step(
                "move",
                actor(),
                **_parse_options(
                    f"{name}.options",
                    require(name, stage, "options"),
                    MOVE_KINDS,
                    "kind",
                    role_ids,
                ),
            )
        ]
    if kind == "respond":
        reject_unknown(name, stage, ("stage", "actor"))
        return [Step("respond", actor())]
    if kind == "partner":
        reject_unknown(name, stage, ("stage", "actor", "options"))
        return [
            Step(
                "partner",
                actor(),
                **_parse_options(
                    f"{name}.options",
                    require(name, stage, "options"),
                    SIDES,
                    "target",
                    role_ids,
                ),
            )
        ]
    if kind == "accident":
        reject_unknown(
            name,
            stage,
            (
                "stage",
                "base",
                "per_rung",
                "cap",
                "nuclear_base",
                "nuclear_per_rung",
                "readiness_multiplier",
            ),
        )
        values = (
            probability(f"{name}.base", stage.get("base", 0.0)),
            nonnegative(f"{name}.per_rung", stage.get("per_rung", 0.0)),
            probability(f"{name}.cap", stage.get("cap", 1.0)),
            probability(f"{name}.nuclear_base", stage.get("nuclear_base", 0.0)),
            nonnegative(f"{name}.nuclear_per_rung", stage.get("nuclear_per_rung", 0.0)),
            nonnegative(f"{name}.readiness_multiplier", stage.get("readiness_multiplier", 1.0)),
        )
        return [Step("accident", accident=values)]
    raise ConfigError(f"{name}.stage {kind!r} is not a crisis stage")


def build(config: Mapping[str, Any]) -> CrisisGame:
    """Build a crisis game from a mapping with ``roles``, ``script`` and options."""

    reject_unknown("crisis", config, ("roles", "script", "max_rung", "final_outcome"))
    roles = tuple(
        _parse_role(role_id, data)
        for role_id, data in as_mapping("crisis.roles", require("crisis", config, "roles")).items()
    )
    sides = {role.side for role in roles}
    if not {"a", "b"} <= sides:
        raise ConfigError("crisis needs at least one role on side a and one on side b")
    role_ids = {role.id for role in roles}
    steps: list[Step] = []
    for i, stage in enumerate(as_list("crisis.script", require("crisis", config, "script"))):
        steps.extend(_expand_stage(i, stage, role_ids))
    if not steps:
        raise ConfigError("crisis.script is empty")
    side_of = {role.id: role.side for role in roles}
    drawn: set[tuple[str, str]] = set()
    for step in steps:
        if step.kind == "type":
            if step.dimension == "stakes" and (step.actor, "stakes") in drawn:
                raise ConfigError(f"{step.actor}'s stakes are drawn twice")
            drawn.add((step.actor, step.dimension))
        if step.kind == "intel" and (step.actor, step.dimension) not in drawn:
            raise ConfigError(f"intel about {step.actor}.{step.dimension} precedes its type draw")
        if (
            step.kind == "move"
            and side_of[step.actor] == "neutral"
            and any(option[1] in ("concede", "deal") for option in step.options)
        ):
            raise ConfigError(f"neutral role {step.actor!r} cannot concede or propose deals")
        if step.kind == "respond" and side_of[step.actor] == "neutral":
            raise ConfigError(f"neutral role {step.actor!r} cannot respond to deals")
    final_outcome = config.get("final_outcome", "compromise")
    if final_outcome not in OUTCOMES:
        raise ConfigError(f"crisis.final_outcome must be one of {OUTCOMES}")
    max_rung = positive_int("crisis.max_rung", config.get("max_rung", 3))
    return CrisisGame(CrisisParams(roles, tuple(steps), max_rung, final_outcome))
