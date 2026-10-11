"""Proxy war as a stochastic war of attrition between sponsor coalitions.

Two local sides fight over *control* (side A's share, in [0, 1]). Each round
every active sponsor role secretly chooses an effort level; choices are
revealed together, then one battle is resolved by a contest-success function

    P(A gains) = E_A^gamma / (E_A^gamma + E_B^gamma),
    E_side = local strength + sum(capacity x effort level of its sponsors).

Control moves one step toward the winner. Effort costs accumulate every round,
and high or direct effort builds war-weariness; a sponsor whose weariness
reaches the limit is forced out (domestic support collapses). A sponsor may
``withdraw`` permanently; the two side leads may both choose ``negotiate`` to
end the war in partition.

Outcome labels: ``a_victory``, ``b_victory``, ``partition``, ``a_withdrew``
(every A sponsor left), ``b_withdrew``, ``stalemate`` (round limit). Long
horizons make the tree too large for exact methods, so the baseline is
:class:`AttritionHeuristicPolicy`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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
    ValueTable,
    as_mapping,
    finite,
    identifier,
    nonnegative,
    positive_int,
    probability,
    reject_unknown,
    require,
    value_table,
)

NAME = "attrition"
OUTCOMES = ("a_victory", "b_victory", "partition", "a_withdrew", "b_withdrew", "stalemate")
LEVELS = ("withdraw", "low", "high", "direct", "negotiate")
EFFORT = {"withdraw": 0.0, "low": 1 / 3, "high": 2 / 3, "direct": 1.0, "negotiate": 1 / 3}


@dataclass(frozen=True, slots=True)
class Sponsor(ImmutableParams):
    id: str
    side: str
    capacity: float
    direct: bool
    cost_scale: float
    values: ValueTable


@dataclass(frozen=True, slots=True)
class AttritionParams(ImmutableParams):
    sponsors: tuple[Sponsor, ...]
    lead_a: str
    lead_b: str
    local_a: float
    local_b: float
    initial_control: float
    step: float
    victory_threshold: float
    max_rounds: int
    gamma: float
    effort: tuple[float, float, float]  # low, high, direct
    costs: tuple[float, float, float]
    weariness: tuple[float, float, float]
    weariness_limit: float

    def sponsor(self, role: str) -> Sponsor:
        for sponsor in self.sponsors:
            if sponsor.id == role:
                return sponsor
        raise KeyError(role)

    def level_index(self, level: str) -> int:
        return {"low": 0, "negotiate": 0, "high": 1, "direct": 2}.get(level, -1)


class AttritionGame:
    name = NAME
    outcome_labels = OUTCOMES

    def __init__(self, params: AttritionParams) -> None:
        self.params = params

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(s.id for s in self.params.sponsors)

    def new_initial_state(self) -> AttritionState:
        return AttritionState(self.params)


class AttritionState(LoggedState):
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
        self.battle = False
        self.label: str | None = None
        self._start_round()

    # ------------------------------------------------------------------ flow
    def _start_round(self) -> None:
        self.choices = {}
        self.pending = [s.id for s in self.params.sponsors if s.id not in self.withdrawn]
        self.battle = False

    def current_player(self) -> str:
        if self.label is not None:
            return TERMINAL
        if self.battle:
            return CHANCE
        return self.pending[0]

    def is_terminal(self) -> bool:
        return self.label is not None

    def legal_actions(self) -> tuple[str, ...]:
        if self.label is not None or self.battle:
            return ()
        role = self.pending[0]
        sponsor = self.params.sponsor(role)
        options = ["withdraw", "low", "high"]
        if sponsor.direct:
            options.append("direct")
        if role in (self.params.lead_a, self.params.lead_b):
            options.append("negotiate")
        return tuple(options)

    def side_effort(self, side: str, choices: Mapping[str, str]) -> float:
        p = self.params
        total = p.local_a if side == "a" else p.local_b
        for sponsor in p.sponsors:
            if sponsor.side != side or sponsor.id in self.withdrawn:
                continue
            level = p.level_index(choices.get(sponsor.id, "withdraw"))
            if level >= 0:
                total += sponsor.capacity * p.effort[level]
        return total

    def win_probability(self, choices: Mapping[str, str]) -> float:
        g = self.params.gamma
        e_a = float(self.side_effort("a", choices) ** g)
        e_b = float(self.side_effort("b", choices) ** g)
        if e_a + e_b <= 0.0:
            return 0.5
        return e_a / (e_a + e_b)

    def chance_outcomes(self) -> tuple[tuple[str, float], ...]:
        if not self.battle or self.label is not None:
            return ()
        p_a = self.win_probability(self.choices)
        return tuple((a, p) for a, p in (("a_gains", p_a), ("b_gains", 1.0 - p_a)) if p > 0.0)

    def apply_action(self, action: str) -> None:
        player = self.current_player()
        if player == TERMINAL:
            raise ConfigError("cannot act in a finished war")
        if player == CHANCE:
            check_legal(action, tuple(a for a, _ in self.chance_outcomes()))
            self._resolve_battle(action)
        else:
            check_legal(action, self.legal_actions())
            self.choices[player] = action
            self.log(f"r{self.round}:self={action}", visible_to=[player])
            self.pending.pop(0)
            if not self.pending:
                self._reveal()
        self._history.append(action)

    def _reveal(self) -> None:
        p = self.params
        for role, choice in self.choices.items():
            self.log(f"r{self.round}:{role}={choice}")
            level = p.level_index(choice)
            if level >= 0:
                sponsor = p.sponsor(role)
                self.cost[role] += sponsor.cost_scale * p.costs[level]
                self.weariness[role] += p.weariness[level]
                self.effort[role] += EFFORT[choice]
                if choice == "direct":
                    self.ever_direct.add(role)
        self.last_choices = dict(self.choices)
        for role, choice in self.choices.items():
            if choice == "withdraw":
                self.withdrawn.add(role)
        if self._side_gone("a") and self._side_gone("b"):
            self.label = "stalemate"
        elif self._side_gone("a"):
            self.label = "a_withdrew"
        elif self._side_gone("b"):
            self.label = "b_withdrew"
        elif (
            self.choices.get(p.lead_a) == "negotiate" and self.choices.get(p.lead_b) == "negotiate"
        ):
            self.label = "partition"
        else:
            self.battle = True
            outcomes = self.chance_outcomes()
            if len(outcomes) == 1:
                self._resolve_battle(outcomes[0][0])

    def _side_gone(self, side: str) -> bool:
        return all(s.id in self.withdrawn for s in self.params.sponsors if s.side == side)

    def _resolve_battle(self, action: str) -> None:
        p = self.params
        self.log(f"r{self.round}:battle={action}")
        delta = p.step if action == "a_gains" else -p.step
        self.control = min(1.0, max(0.0, self.control + delta))
        self.battle = False
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
                self.withdrawn.add(sponsor.id)
                self.log(f"r{self.round}:{sponsor.id}=forced_out")
        if self._side_gone("a"):
            self.label = "a_withdrew"
        elif self._side_gone("b"):
            self.label = "b_withdrew"
        else:
            self._start_round()

    # --------------------------------------------------------------- payoffs
    def returns(self) -> dict[str, float]:
        if self.label is None:
            raise ConfigError("returns requested before the war ended")
        return {s.id: s.values[self.label] - self.cost[s.id] for s in self.params.sponsors}

    def outcome(self) -> Outcome:
        if self.label is None:
            raise ConfigError("outcome requested before the war ended")
        tags = [f"{role}:direct" for role in sorted(self.ever_direct)]
        tags += [f"{role}:withdrew" for role in sorted(self.withdrawn)]
        metrics: dict[str, float] = {"control": self.control, "rounds": float(self.round)}
        for sponsor in self.params.sponsors:
            metrics[f"cost:{sponsor.id}"] = self.cost[sponsor.id]
            metrics[f"weariness:{sponsor.id}"] = self.weariness[sponsor.id]
            # commitment: effort over the war relative to full effort for max_rounds
            metrics[f"commit:{sponsor.id}"] = min(
                1.0, self.effort[sponsor.id] / self.params.max_rounds
            )
            metrics[f"effort:{sponsor.id}"] = self.effort[sponsor.id] / max(1, self.round + 1)
        return Outcome(self.label, tuple(tags), metrics)

    def observation(self, role: str) -> tuple[float, ...]:
        p = self.params
        sponsor = p.sponsor(role)
        own_control = self.control if sponsor.side == "a" else 1.0 - self.control
        last = self.last_choices
        e_a = self.side_effort("a", last) if last else 0.0
        e_b = self.side_effort("b", last) if last else 0.0
        own_e, other_e = (e_a, e_b) if sponsor.side == "a" else (e_b, e_a)
        other_lead = p.lead_b if sponsor.side == "a" else p.lead_a
        return (
            own_control,
            self.round / p.max_rounds,
            self.weariness[role] / max(p.weariness_limit, 1e-9),
            min(1.0, self.cost[role]),
            (p.level_index(last.get(role, "withdraw")) + 1) / 3.0,
            own_e / (own_e + other_e) if own_e + other_e > 0 else 0.5,
            1.0 if last.get(other_lead) == "negotiate" else 0.0,
            1.0 if role in (p.lead_a, p.lead_b) else 0.0,
            1.0 if sponsor.side == "a" else 0.0,
            len(self.withdrawn) / len(p.sponsors),
        )


class AttritionHeuristicPolicy:
    """Myopic baseline: best one-round response to the others' last choices.

    The value of control is interpolated linearly between the two victory
    labels; the role negotiates or withdraws when that settles for more than
    the interpolated value of continuing.
    """

    def __init__(self, game: AttritionGame) -> None:
        self.params = game.params

    def _value_of_control(self, sponsor: Sponsor, control: float) -> float:
        p = self.params
        low, high = 1.0 - p.victory_threshold, p.victory_threshold
        frac = (control - low) / max(high - low, 1e-9)
        v_a, v_b = sponsor.values["a_victory"], sponsor.values["b_victory"]
        return v_b + (v_a - v_b) * min(1.0, max(0.0, frac))

    def action_probabilities(self, state: GameState) -> dict[str, float]:
        if not isinstance(state, AttritionState):
            raise TypeError("AttritionHeuristicPolicy needs an AttritionState")
        p = self.params
        role = state.current_player()
        sponsor = p.sponsor(role)
        legal = state.legal_actions()
        assumed = {s.id: state.last_choices.get(s.id, "low") for s in p.sponsors}
        current_value = self._value_of_control(sponsor, state.control)
        scores: dict[str, float] = {}
        for action in legal:
            if action == "withdraw":
                label = "a_withdrew" if sponsor.side == "a" else "b_withdrew"
                scores[action] = sponsor.values[label]
                continue
            if action == "negotiate":
                scores[action] = sponsor.values["partition"]
                continue
            trial = dict(assumed)
            trial[role] = action
            p_a = state.win_probability(trial)
            up = self._value_of_control(sponsor, state.control + p.step)
            down = self._value_of_control(sponsor, state.control - p.step)
            expected = p_a * up + (1.0 - p_a) * down
            level = p.level_index(action)
            scores[action] = expected - sponsor.cost_scale * p.costs[level]
            if state.weariness[role] + p.weariness[level] >= p.weariness_limit:
                scores[action] -= abs(current_value)  # avoid being forced out
        best = max(legal, key=lambda a: (scores[a], -legal.index(a)))
        return {action: (1.0 if action == best else 0.0) for action in legal}


def build(config: Mapping[str, Any]) -> AttritionGame:
    keys = (
        "sponsors",
        "leads",
        "local_strength",
        "initial_control",
        "step",
        "victory_threshold",
        "max_rounds",
        "gamma",
        "effort",
        "costs",
        "weariness",
        "weariness_limit",
    )
    reject_unknown("attrition", config, keys)
    sponsors = []
    for role_id, raw in as_mapping(
        "attrition.sponsors", require("attrition", config, "sponsors")
    ).items():
        name = f"attrition.sponsors.{role_id}"
        item = as_mapping(name, raw)
        reject_unknown(name, item, ("side", "capacity", "direct", "cost_scale", "values"))
        side = require(name, item, "side")
        if side not in ("a", "b"):
            raise ConfigError(f"{name}.side must be 'a' or 'b'")
        sponsors.append(
            Sponsor(
                id=identifier(name, role_id),
                side=side,
                capacity=nonnegative(f"{name}.capacity", item.get("capacity", 1.0)),
                direct=bool(item.get("direct", False)),
                cost_scale=nonnegative(f"{name}.cost_scale", item.get("cost_scale", 1.0)),
                values=value_table(f"{name}.values", require(name, item, "values"), OUTCOMES),
            )
        )
    if not any(s.side == "a" for s in sponsors) or not any(s.side == "b" for s in sponsors):
        raise ConfigError("attrition needs at least one sponsor on each side")
    leads = as_mapping("attrition.leads", config.get("leads", {}))
    reject_unknown("attrition.leads", leads, ("a", "b"))
    side_of = {s.id: s.side for s in sponsors}
    lead_a = leads.get("a", next(s.id for s in sponsors if s.side == "a"))
    lead_b = leads.get("b", next(s.id for s in sponsors if s.side == "b"))
    if side_of.get(lead_a) != "a" or side_of.get(lead_b) != "b":
        raise ConfigError("attrition leads must be sponsors on their own side")
    local = as_mapping("attrition.local_strength", config.get("local_strength", {}))
    reject_unknown("attrition.local_strength", local, ("a", "b"))

    def triple(key: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
        data = as_mapping(f"attrition.{key}", config.get(key, {}))
        reject_unknown(f"attrition.{key}", data, ("low", "high", "direct"))
        return (
            nonnegative(f"attrition.{key}.low", data.get("low", default[0])),
            nonnegative(f"attrition.{key}.high", data.get("high", default[1])),
            nonnegative(f"attrition.{key}.direct", data.get("direct", default[2])),
        )

    threshold = probability("attrition.victory_threshold", config.get("victory_threshold", 0.85))
    if threshold <= 0.5:
        raise ConfigError("attrition.victory_threshold must exceed 0.5")
    initial = probability("attrition.initial_control", config.get("initial_control", 0.5))
    if not 1.0 - threshold < initial < threshold:
        raise ConfigError("attrition.initial_control must lie strictly between the thresholds")
    gamma = finite("attrition.gamma", config.get("gamma", 1.0))
    if gamma <= 0.0:
        raise ConfigError("attrition.gamma must be positive")
    step = probability("attrition.step", config.get("step", 0.1))
    if step <= 0.0:
        raise ConfigError("attrition.step must be positive")
    weariness_limit = nonnegative("attrition.weariness_limit", config.get("weariness_limit", 1.0))
    if weariness_limit <= 0.0:
        raise ConfigError("attrition.weariness_limit must be positive")
    params = AttritionParams(
        sponsors=tuple(sponsors),
        lead_a=lead_a,
        lead_b=lead_b,
        local_a=nonnegative("local_strength.a", local.get("a", 1.0)),
        local_b=nonnegative("local_strength.b", local.get("b", 1.0)),
        initial_control=initial,
        step=step,
        victory_threshold=threshold,
        max_rounds=positive_int("attrition.max_rounds", config.get("max_rounds", 12)),
        gamma=gamma,
        effort=triple("effort", (1.0, 2.0, 4.0)),
        costs=triple("costs", (0.02, 0.06, 0.15)),
        weariness=triple("weariness", (0.0, 0.02, 0.08)),
        weariness_limit=weariness_limit,
    )
    return AttritionGame(params)
