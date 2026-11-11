"""Build and play a crisis, including each player's view of hidden information."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from typing import Any, cast

from ...games.policy import TabularPolicy, tabular_from_tree
from ...games.protocol import (
    CHANCE,
    TERMINAL,
    GameState,
    ImmutableParams,
    LoggedState,
    Outcome,
    check_legal,
)
from ...games.tree import GameTree, TreePolicy, build_tree
from ...records import Record
from ..common import ConfigError, Section, as_mapping, prospect
from .incident import incident_stages
from .params import NAME, OUTCOMES, WAR_OUTCOMES, CrisisRole
from .stages import Accident, Intel, Move, Partner, Respond, ScriptSoFar, Signal, Stage, TypeDraw


class CrisisParams(ImmutableParams, Record):
    """The game form: roles, script, the height of the ladder (``max_rung``) and the outcome
    when the script ends with no one conceding (``final_outcome``).
    """

    def __init__(
        self,
        roles: tuple[CrisisRole, ...],
        stages: tuple[Stage, ...],
        max_rung: int,
        final_outcome: str,
    ) -> None:
        self.roles = roles
        self.stages = stages
        self.max_rung = max_rung
        self.final_outcome = final_outcome
        self._role = {role.id: role for role in roles}
        self._stakes = {
            s.actor: s for s in stages if isinstance(s, TypeDraw) and s.trait == "stakes"
        }
        self._reports: dict[str, Intel] = {}
        for stage in stages:
            if isinstance(stage, Intel):
                self._reports.setdefault(stage.actor, stage)

    def role(self, role_id: str) -> CrisisRole:
        """The role named *role_id*."""

        return self._role[role_id]

    def offset(self, role_id: str, level: str | None, label: str) -> float:
        """The shift that the holder's stakes *level* adds to its value of outcome *label*."""

        if level is None:
            return 0.0
        return self._stakes[role_id].offsets[level].get(label, 0.0)

    def report_shown_to(self, about: str, role: str) -> bool:
        """Whether *role* observes signals about the role *about* (each has one audience)."""

        return self._reports[about].shown_to(role)

    def as_believed_by(self, role: str) -> CrisisParams:
        """The game form under *role*'s own priors over others' types."""

        stages = tuple(
            stage.as_believed_by(role) if isinstance(stage, TypeDraw) else stage
            for stage in self.stages
        )
        return CrisisParams(self.roles, stages, self.max_rung, self.final_outcome)


class CrisisGame:
    """A crisis: the factory of its initial states and of each role's subjective view."""

    name = NAME
    outcome_labels = OUTCOMES

    def __init__(self, params: CrisisParams) -> None:
        self.params = params

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(role.id for role in self.params.roles)

    @property
    def has_beliefs(self) -> bool:
        """Whether some role holds its own prior over another's type."""

        return any(isinstance(s, TypeDraw) and s.beliefs for s in self.params.stages)

    def view(self, role: str) -> CrisisGame:
        """The game *role* believes it is playing: its own priors where it holds them, nature's
        elsewhere.
        """

        return CrisisGame(self.params.as_believed_by(role))

    def views(self) -> dict[str, CrisisGame]:
        """Each role's view; roles with the same priors share one game."""

        shared: dict[tuple[object, ...], CrisisGame] = {}
        result = {}
        for role in self.roles:
            priors = tuple(
                _hashable(s.beliefs.get(role)) if isinstance(s, TypeDraw) else None
                for s in self.params.stages
            )
            if priors not in shared:
                shared[priors] = self.view(role)
            result[role] = shared[priors]
        return result

    def new_initial_state(self) -> CrisisState:
        return CrisisState(self.params)


def _hashable(prior: dict[str, float] | None) -> tuple[tuple[str, float], ...] | None:
    return None if prior is None else tuple(prior.items())


class CrisisState(LoggedState):
    """A history of the crisis: the position in the script and the variables set so far."""

    def __init__(self, params: CrisisParams) -> None:
        super().__init__(role.id for role in params.roles)
        self.params = params
        self.index = 0  # the stage of the script the crisis is at
        self.label: str | None = None  # the outcome, once the crisis has ended
        self.tags: list[str] = []
        # private information
        self.types: dict[str, dict[str, str]] = {role.id: {} for role in params.roles}
        # (the role a report is about, its trait) -> the report; a second report about a
        # role is kept beside the first, never in its place
        self.reports: dict[tuple[str, str], str] = {}
        self.temperament: dict[str, str] = {}  # officer -> its temperament
        # the ladder and the bargaining
        self.rung = 0
        self.added: dict[str, int] = {"a": 0, "b": 0, "neutral": 0}  # rungs added, by side
        self.stance: dict[str, str] = {}  # role -> hard or soft
        self.deal_by: str | None = None  # the side with a deal on the table
        self.conceded_side: str | None = None
        self.pressure: list[tuple[str, float]] = []  # (target side, cost unless it backs down)
        # risk
        self.risk_multiplier = 1.0
        self.control: dict[str, float] = {"a": 0.0, "b": 0.0, "neutral": 0.0}
        self.alarm: str | None = None  # who an incident has alarmed: officer, leader, escalation
        # costs run up along the way
        self.signal_cost: dict[str, float] = {}
        self.action_cost: dict[str, float] = {}
        self._settle()

    # ----------------------------------------------------------------- queries
    def _stage(self) -> Stage:
        return self.params.stages[self.index]

    def side_of(self, role_id: str) -> str:
        """The side of the role *role_id*."""

        return self.params.role(role_id).side

    def any_ready(self) -> bool:
        """Whether some role's readiness is high."""

        return any(traits.get("readiness") == "high" for traits in self.types.values())

    def current_player(self) -> str:
        if self.label is not None:
            return TERMINAL
        return self._stage().player(self)

    def is_terminal(self) -> bool:
        return self.label is not None

    def legal_actions(self) -> tuple[str, ...]:
        if self.label is not None:
            return ()
        return self._stage().legal_actions(self)

    def chance_outcomes(self) -> tuple[tuple[str, float], ...]:
        if self.current_player() != CHANCE:
            return ()
        return tuple((draw, p) for draw, p in self._stage().draws(self) if p > 0.0)

    # -------------------------------------------------------------- transition
    def apply_action(self, action: str) -> None:
        player = self.current_player()
        if player == TERMINAL:
            raise ConfigError("cannot act in a terminal crisis state")
        if player == CHANCE:
            check_legal(action, tuple(draw for draw, _ in self.chance_outcomes()))
        else:
            check_legal(action, self.legal_actions())
        self._stage().apply(self, action)
        self._history.append(action)
        self.index += 1
        self._settle()

    def charge(self, role: str, cost: float) -> None:
        """Add *cost* to *role*'s accumulated action costs (a negative cost is a gain)."""

        self.action_cost[role] = self.action_cost.get(role, 0.0) + cost

    def _settle(self) -> None:
        """Advance to the next stage that applies; an exhausted script ends the game in
        ``final_outcome``.
        """

        stages = self.params.stages
        while self.label is None and self.index < len(stages):
            stage = stages[self.index]
            if stage.applies(self):
                return
            stage.pass_over(self)
            self.index += 1
        if self.label is None:
            self.label = self.params.final_outcome

    # ----------------------------------------------------------------- payoffs
    def returns(self) -> dict[str, float]:
        if self.label is None:
            raise ConfigError("returns requested before the crisis ended")
        return {role.id: self._payoff(role, self.label) for role in self.params.roles}

    def _payoff(self, role: CrisisRole, label: str) -> float:
        """*role*'s payoff at outcome *label*.

        Its value of the outcome (scaled by its war factor if it is a war and its resolve is
        high, shifted by its stakes), minus ``escalation_cost`` per rung its side added, its
        signal and action costs, its audience cost if its side conceded after its hard
        stance, and any pressure on its side that did not concede; the total passes through
        its prospect-theory value function.
        """

        own = self.types[role.id]
        value = role.values[label]
        if label in WAR_OUTCOMES and own.get("resolve") == "high":
            value *= role.high_resolve_war_factor
        value += self.params.offset(role.id, own.get("stakes"), label)
        if role.side in ("a", "b"):
            value -= role.escalation_cost * self.added[role.side]
        value -= self.signal_cost.get(role.id, 0.0)
        value -= self.action_cost.get(role.id, 0.0)
        if self.conceded_side == role.side and self.stance.get(role.id) == "hard":
            value -= role.audience_cost
        for target_side, cost in self.pressure:
            if role.side == target_side and self.conceded_side != target_side:
                value -= cost
        return prospect(value, role.risk)

    def outcome(self) -> Outcome:
        """The outcome with its tags and metrics.

        ``commit:<role>`` = (1 + rungs its side added) / (1 + ``max_rung``).
        """

        if self.label is None:
            raise ConfigError("outcome requested before the crisis ended")
        top = self.params.max_rung
        metrics = {
            "rung": float(self.rung),
            "max_rung": float(top),
            "escalation_a": float(self.added["a"]),
            "escalation_b": float(self.added["b"]),
        }
        for role in self.params.roles:
            added = self.added[role.side] if role.side in ("a", "b") else 0
            metrics[f"commit:{role.id}"] = (1.0 + added) / (1.0 + top)
        return Outcome(self.label, tuple(self.tags), metrics)

    def observation(self, role: str) -> tuple[float, ...]:
        """Twelve features of *role*'s information: the rung, its resolve and readiness, its
        stance, whether an opponent stands hard, a deal offered to it, a deal it offered,
        pressure on its side, the first signal it observed, progress through the script,
        added risk, and whether it is on side ``a``.
        """

        side = self.side_of(role)
        own = self.types[role]
        others_hard = any(
            stance == "hard" and self.side_of(other) not in (side, "neutral")
            for other, stance in self.stance.items()
        )
        report = next(  # the first report this role may see (never another's intel)
            (
                report
                for (about, _), report in self.reports.items()
                if self.params.report_shown_to(about, role)
            ),
            None,
        )
        high_low = {"high": 1.0, "low": 0.0}
        return (
            self.rung / max(1, self.params.max_rung),
            high_low.get(own.get("resolve", ""), 0.5),
            high_low.get(own.get("readiness", ""), 0.5),
            1.0 if self.stance.get(role) == "hard" else 0.0,
            1.0 if others_hard else 0.0,
            1.0 if self.deal_by is not None and self.deal_by != side else 0.0,
            1.0 if self.deal_by == side else 0.0,
            1.0 if any(target == side for target, _ in self.pressure) else 0.0,
            {"alarming": 1.0, "reassuring": 0.0}.get(report or "", 0.5),
            self.index / max(1, len(self.params.stages)),
            min(1.0, self.risk_multiplier - 1.0) if self.risk_multiplier > 1.0 else 0.0,
            1.0 if side == "a" else 0.0,
        )


# the reader of each kind of stage a script may list, by its ``stage`` name
STAGES: dict[str, Callable[[Section, Collection[str]], list[Stage]]] = {
    "types": TypeDraw.from_config,
    "intel": Intel.from_config,
    "signal": Signal.from_config,
    "move": Move.from_config,
    "respond": Respond.from_config,
    "partner": Partner.from_config,
    "accident": Accident.from_config,
    "incident": incident_stages,
}


def build(config: Mapping[str, Any]) -> CrisisGame:
    """The crisis game of *config*.

    ``roles``           ``{role: {...}}`` (:class:`CrisisRole`), with at least one role on
                        each of sides ``a`` and ``b``.
    ``script``          the list of stages, each named by its ``stage`` key (``STAGES``).
    ``max_rung``        the height of the escalation ladder (default 3).
    ``final_outcome``   the outcome if the script ends with no one conceding (default
                        ``compromise``).
    """

    with Section("crisis", config) as s:
        role_blocks = as_mapping("crisis.roles", s.get("roles"))
        roles = tuple(
            CrisisRole.from_config(role_id, data) for role_id, data in role_blocks.items()
        )
        if not {"a", "b"} <= {role.side for role in roles}:
            raise ConfigError("crisis needs at least one role on side a and one on side b")
        role_ids = {role.id for role in roles}
        stages: list[Stage] = []
        for index, data in enumerate(s.sequence("script")):
            stages.extend(_read_stage(f"crisis.script[{index}]", data, role_ids))
        if not stages:
            raise ConfigError("crisis.script is empty")
        _check_script(stages, roles)
        final_outcome = s.choice("final_outcome", OUTCOMES, "compromise")
        max_rung = s.positive_int("max_rung", 3)
    return CrisisGame(CrisisParams(roles, tuple(stages), max_rung, final_outcome))


def _read_stage(name: str, data: Any, role_ids: Collection[str]) -> list[Stage]:
    """The nodes of one script entry."""

    with Section(name, data) as s:
        kind = s.get("stage")
        if not isinstance(kind, str) or kind not in STAGES:
            raise ConfigError(f"{name}.stage {kind!r} is not a crisis stage")
        return STAGES[kind](s, role_ids)


def _check_script(stages: list[Stage], roles: tuple[CrisisRole, ...]) -> None:
    """Validate each stage against the stages before it."""

    script = ScriptSoFar({role.id: role.side for role in roles})
    for stage in stages:
        stage.check(script)


class CrisisBeliefPolicy:
    """Let each player act on its own beliefs about hidden types.

    Solve each distinct view once, then use that player's part of the solution.
    A view changes the prior, while keeping the same observations and actions.
    The resulting play need not be an equilibrium of the true game: players may
    disagree about its probabilities. When every prior agrees with nature's,
    all players use the solution of the true game.
    """

    def __init__(self, game: CrisisGame, solve: Callable[[GameTree], TreePolicy]) -> None:
        solved: dict[int, TabularPolicy] = {}
        self._policies: dict[str, TabularPolicy] = {}
        for role, view in game.views().items():
            key = id(view)  # roles with the same priors share one view object
            if key not in solved:
                tree = build_tree(cast(GameState, view.new_initial_state()))
                solved[key] = tabular_from_tree(tree, solve(tree))
            self._policies[role] = solved[key]

    def policy_of(self, role: str) -> TabularPolicy:
        """The tabular solution of *role*'s view."""

        return self._policies[role]

    def action_probabilities(self, state: GameState) -> dict[str, float]:
        return self._policies[str(state.current_player())].action_probabilities(state)
