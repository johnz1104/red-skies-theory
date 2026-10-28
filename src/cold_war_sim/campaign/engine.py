"""The campaign engine: an explicit, deep-copyable quarter-by-quarter state machine.

Quarter phases::

    BEGIN -> DIRECTOR -> STRATEGIC (Q1 only) -> SUBGAMES -> RESOLVE
          -> DYNAMICS -> REWARDS -> END

``advance`` runs phases until the next decision belongs to an agent power
(NPC powers, NPC roles and single-option requests resolve inline) or the
campaign ends. ``act`` answers the pending decision and advances again.
Major events above ``EngineConfig.playable_tier`` run on rails: every role
plays its historical choices and chance follows the historical draws, so the
event still fires under its condition but never asks an agent. Decision events
(one power's choice among options) are put to their power in the strategic
phase and take effect in RESOLVE. When a leader changes, running attrition wars
whose roles include that power are re-bound at the current world.
All mutable data lives in :class:`EngineState`, so a snapshot is one
``deepcopy``; the catalog and cached baselines are shared, immutable inputs.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from ..families import attrition, bargaining, build_family
from ..families.attrition_beliefs import AttritionBeliefPolicy, choose
from ..families.crisis_beliefs import CrisisBeliefPolicy
from ..games import CHANCE, Policy, TabularPolicy, build_tree, sample_action, tabular_from_tree
from ..games.algorithms import backward_induction, cfr_plus, logit_qre
from ..games.traverse import GameTree, TreePolicy
from . import (
    assumptions,
    calendar,
    derive,
    generator,
    goals,
    observation,
    rewards,
    standing,
    strategic,
)
from .catalog import Catalog, MajorSpec, load_catalog
from .decisions import DecisionRequest, Option
from .director import run_director
from .dynamics import DynamicsParams, step
from .effects import EffectContext
from .policies import CampaignPolicy, HistoricalPolicy, make_policy
from .rng import stream
from .serialization import canonical_json, stable_hash
from .state import IDENTITIES, POWERS, WorldState, identity, seat

CHANCE_MODES = ("sample", "historical")
REWARD_MODES = ("goals", "score")  # goals: docs/GOALS.md; score: the pre-goals world score
EFFORT = {"withdraw": 0.0, "low": 1 / 3, "negotiate": 1 / 3, "high": 2 / 3, "direct": 1.0}


class Phase(StrEnum):
    BEGIN = "begin"
    DIRECTOR = "director"
    STRATEGIC = "strategic"
    SUBGAMES = "subgames"
    RESOLVE = "resolve"
    DYNAMICS = "dynamics"
    REWARDS = "rewards"
    END = "end"
    DONE = "done"


@dataclass(frozen=True)
class EngineConfig:
    agents: tuple[str, ...] = POWERS
    chance_mode: str = "sample"
    npc_policy: str = "historical"
    baseline_iterations: int = 300
    # computer-run reference play is a logit (quantal) response at this temperature, in
    # payoff units: near-ties become mixtures (user decision 2026-10-11); 0 = exact
    baseline_temperature: float = 0.1
    end_quarter: int = calendar.END
    playable_tier: int | None = None  # None: every major event is playable
    reward: str = "goals"
    # draw the catalog's numbers within the assumptions register's ranges at each reset
    # (user decision 7, 2026-10-14); None: the catalog's own values
    sampling: assumptions.Sampling | None = None
    # single-event scenarios (user, 2026-10-20; campaign/scenario.py): snapshot the state
    # when a listed event starts (``CampaignEngine.captured``), and with a scope ask agents
    # only inside that event and stop once it resolves
    capture: frozenset[str] = frozenset()
    scope: str | None = None
    # the crisis generator (data/generator.yaml, E49): crises where the world makes them
    # likely; never with historical chance, whose replay has only named crises
    generator: bool = False
    # how events above playable_tier run: "historical" (their documented choices and
    # chance) or "reference" (computer-run reference play and sampled chance, E49), so a
    # world that has left history is not pulled back to it (WRITEUP 6.5, hindsight 2)
    rails: str = "historical"

    def __post_init__(self) -> None:
        if not set(self.agents) <= set(POWERS):
            raise ValueError(f"agents must be drawn from {POWERS}")
        if self.chance_mode not in CHANCE_MODES:
            raise ValueError(f"chance_mode must be one of {CHANCE_MODES}")
        if not calendar.START <= self.end_quarter <= calendar.END:
            raise ValueError("end_quarter must lie within the campaign")
        if self.playable_tier is not None and self.playable_tier < 1:
            raise ValueError("playable_tier must be at least 1")
        if self.reward not in REWARD_MODES:
            raise ValueError(f"reward must be one of {REWARD_MODES}")
        if self.baseline_temperature < 0.0:
            raise ValueError("baseline_temperature must be non-negative")
        if self.rails not in ("historical", "reference"):
            raise ValueError("rails must be historical or reference")


@dataclass(slots=True)
class ActiveEvent:
    event_id: str
    seq: int
    started: int
    config: dict[str, Any]
    state: Any
    rng: np.random.Generator
    decisions: dict[str, int] = field(default_factory=dict)
    chance_count: int = 0
    round_marker: int = 0
    annual_choices: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class PendingDecision:
    request: DecisionRequest
    target: tuple[str, ...]  # ("strategy", power, stage) or ("subgame", event_id)


@dataclass(slots=True)
class EngineState:
    seed: int
    world: WorldState
    phase: Phase = Phase.DIRECTOR
    strategy_queue: list[tuple[str, str]] = field(default_factory=list)
    commits: dict[str, dict[str, str]] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    cursor: int = 0
    active: dict[str, ActiveEvent] = field(default_factory=dict)
    finished: list[tuple[int, str, Any, dict[str, float]]] = field(default_factory=list)
    requested: set[str] = field(default_factory=set)
    war_effort: dict[str, float] = field(default_factory=dict)
    crisis_requests: list[str] = field(default_factory=list)  # agents' challenges (E49)
    direct_effort: dict[str, float] = field(default_factory=dict)  # fought with own troops
    pending: PendingDecision | None = None
    scores: dict[str, float] = field(default_factory=dict)
    components: dict[str, dict[str, float]] = field(default_factory=dict)  # by identity
    reward_buffer: dict[str, float] = field(default_factory=dict)
    eliminated: set[str] = field(default_factory=set)
    holders: dict[str, str] = field(default_factory=dict)  # seat -> identity holding it
    retired: set[str] = field(default_factory=set)  # identities that lost their seat
    standing: dict[str, float] = field(default_factory=dict)  # scoreboard; never a reward
    standing_log: list[tuple[int, str, dict[str, float]]] = field(default_factory=list)
    seq: int = 0
    instances: dict[str, int] = field(default_factory=dict)
    decision_queue: list[str] = field(default_factory=list)  # decision events to put
    decided: list[tuple[str, str]] = field(default_factory=list)  # (event, option) to apply
    npc_rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    truncated: bool = False
    trace: list[dict[str, Any]] = field(default_factory=list)
    decisions: int = 0
    last_returns: dict[str, dict[str, float]] = field(default_factory=dict)
    assumptions: dict[str, float] = field(default_factory=dict)  # this episode's draw
    variant: Catalog | None = None  # the catalog with the draw applied (None: no sampling)
    scope_done: bool = False  # the scoped event has resolved (EngineConfig.scope)


# Baselines are pure functions of (family, bound config): cache them across
# engines and snapshots.
_BASELINES: OrderedDict[str, Policy] = OrderedDict()
_BASELINE_LIMIT = 512
# Attrition belief policies are lazy and their memos grow large, so only the latest few
# are kept; a running war uses its policy every quarter and stays among them (E39)
_BELIEF_POLICIES: OrderedDict[str, Policy] = OrderedDict()
_BELIEF_LIMIT = 4


def _solve_tree(tree: GameTree, iterations: int, temperature: float) -> TreePolicy:
    """Logit QRE at *temperature* > 0; otherwise exact equilibrium play."""

    if temperature > 0.0:
        return logit_qre(tree, temperature=temperature).policy
    if tree.is_perfect_information:
        return backward_induction(tree).policy
    return cfr_plus(tree, iterations=iterations, trembles=0.01).average_policy


def reference_tree_policy(
    game: Any, tree: GameTree, iterations: int, temperature: float = 0.0
) -> TreePolicy:
    """Reference play on *tree*, the true game's tree: a crisis whose roles hold their
    own beliefs plays each role's part of its own view (E41); otherwise the solver's
    solution."""

    if not getattr(game, "has_beliefs", False):
        return _solve_tree(tree, iterations, temperature)
    belief = CrisisBeliefPolicy(game, lambda view: _solve_tree(view, iterations, temperature))
    result = []
    for info in tree.infosets:
        dist = belief.policy_of(tree.roles[info.role]).table.get(info.key)
        uniform = 1.0 / len(info.actions)
        result.append([dist.get(a, 0.0) if dist else uniform for a in info.actions])
    return result


def baseline_policy(
    family: str, config: Mapping[str, Any], iterations: int, temperature: float = 0.0
) -> Policy:
    """The computer-run reference for a sub-game: exact equilibrium play, or with
    ``temperature > 0`` a logit quantal response equilibrium (the belief-based
    bargaining solver stays exact)."""

    key = canonical_json(
        {"family": family, "config": config, "iterations": iterations, "temperature": temperature}
    )
    cache, limit = (
        (_BELIEF_POLICIES, _BELIEF_LIMIT)
        if family == "attrition"
        else (_BASELINES, _BASELINE_LIMIT)
    )
    cached = cache.get(key)
    if cached is not None:
        cache.move_to_end(key)
        return cached
    game = build_family(family, config)
    policy: Policy
    if family == "bargaining":
        policy = bargaining.BargainingEquilibriumPolicy(game, temperature)
    elif family == "attrition":
        policy = AttritionBeliefPolicy(game, temperature)
    elif family == "crisis" and game.has_beliefs:  # each role solves its own view (E41)
        policy = CrisisBeliefPolicy(game, lambda tree: _solve_tree(tree, iterations, temperature))
    else:
        root = game.solver_state() if hasattr(game, "solver_state") else game.new_initial_state()
        tree = build_tree(root)
        policy = tabular_from_tree(tree, _solve_tree(tree, iterations, temperature))
    cache[key] = policy
    if len(cache) > limit:
        cache.popitem(last=False)
    return policy


class CampaignEngine:
    """Engine-native API: ``reset``, ``pending``, ``act``, ``observe``, ``pop_rewards``."""

    def __init__(self, catalog: Catalog | None = None, config: EngineConfig | None = None) -> None:
        self.base_catalog = self.catalog = catalog or load_catalog()
        self.config = config or EngineConfig()
        self.dynamics = DynamicsParams.from_catalog(self.catalog)
        self.observed_flags = self.catalog.observed_flags
        self.feature_names = observation.feature_names(self.observed_flags)
        self.npc_policy: CampaignPolicy = make_policy(self.config.npc_policy)
        self.rails_policy: CampaignPolicy = HistoricalPolicy()
        self.s = EngineState(seed=0, world=self.catalog.initial_world())
        # entry states of the events in EngineConfig.capture, taken as each starts
        self.captured: dict[str, EngineState] = {}

    # ------------------------------------------------------------------ public
    def reset(self, seed: int = 0) -> None:
        self.captured = {}
        drawn, variant = self._sample(int(seed))
        world = self.catalog.initial_world()
        self.s = EngineState(seed=int(seed), world=world, npc_rng=stream(int(seed), "__npc__"))
        if variant is not None:
            self.s.assumptions, self.s.variant = drawn, variant
            self.s.trace.append({"assumptions": variant.fingerprint})
        self.s.scores = rewards.scores(world, self.catalog)
        self.s.components = {
            identity(world, p): goals.components(world, identity(world, p), self.catalog)
            for p in POWERS
        }
        self.s.reward_buffer = dict.fromkeys(IDENTITIES, 0.0)
        self.s.holders = {p: identity(world, p) for p in POWERS}
        self.s.standing = dict.fromkeys(IDENTITIES, self.catalog.goals.standing_start)
        self._advance()

    @property
    def world(self) -> WorldState:
        return self.s.world

    @property
    def done(self) -> bool:
        return self.s.phase == Phase.DONE

    @property
    def truncated(self) -> bool:
        return self.s.truncated

    def pending(self) -> DecisionRequest | None:
        return self.s.pending.request if self.s.pending is not None else None

    def act(self, index: int) -> None:
        pending = self.s.pending
        if pending is None:
            raise RuntimeError("no decision is pending")
        request = pending.request
        if not 0 <= index < len(request.options):
            raise ValueError(
                f"option index {index} out of range for {len(request.options)} options"
            )
        self.s.pending = None
        if pending.target[0] == "strategy":
            self._commit_strategy(request, index, agent=True)
        elif pending.target[0] == "decision":
            self._commit_decision(request, index, agent=True)
        else:
            self._apply_subgame_choice(self.s.active[pending.target[1]], request, index, agent=True)
        self._advance()

    def agent_of(self, power: str) -> str:
        """The identity holding *power*'s seat (the China seat may change hands)."""

        return identity(self.world, power)

    def observe(self, agent: str) -> np.ndarray:
        return observation.encode(self, seat(agent))

    def observe_subgame(self, event_id: str, state: Any, role: str) -> np.ndarray:
        """The observation *role*'s agent would get deciding at *state* of the running
        event *event_id* (any state of its game, such as a node of its game tree): for
        scoring learned play on the event's tree. The engine's own state is unchanged."""

        active = self.s.active[event_id]
        saved = active.state, self.s.pending
        try:
            active.state = state
            request = self._subgame_request(active, role, tuple(state.legal_actions()))
            self.s.pending = PendingDecision(request, ("subgame", event_id))
            return self.observe(self.agent_of(role))
        finally:
            active.state, self.s.pending = saved

    def describe(self, agent: str) -> dict[str, Any]:
        return observation.describe(self, seat(agent))

    def action_mask(self, agent: str) -> np.ndarray:
        from .decisions import MAX_OPTIONS

        mask = np.zeros(MAX_OPTIONS, dtype=np.int8)
        request = self.pending()
        if (
            request is not None
            and request.power == seat(agent)
            and self.agent_of(request.power) == agent
        ):
            mask[: len(request.options)] = 1
        return mask

    def pop_rewards(self) -> dict[str, float]:
        """Rewards since the last call, keyed by identity."""

        result = dict(self.s.reward_buffer)
        self.s.reward_buffer = dict.fromkeys(IDENTITIES, 0.0)
        return result

    def snapshot(self) -> EngineState:
        return copy.deepcopy(self.s)

    def restore(self, snapshot: EngineState) -> None:
        self.s = copy.deepcopy(snapshot)
        catalog = self.s.variant or self.base_catalog
        if catalog is not self.catalog:
            self.catalog = catalog
            self.dynamics = DynamicsParams.from_catalog(catalog)

    def resume(self) -> None:
        """Advance a restored state to the next agent decision (or the end)."""

        self._advance()

    def _sample(self, seed: int) -> tuple[dict[str, float], Catalog | None]:
        """This episode's draw and catalog (the base catalog when sampling is off)."""

        sampling = self.config.sampling
        if sampling is None:
            if self.catalog is not self.base_catalog:
                self.catalog = self.base_catalog
                self.dynamics = DynamicsParams.from_catalog(self.catalog)
            return {}, None
        current = assumptions.plan_for(self.base_catalog, self.config.playable_tier)
        draw = seed if sampling.draw is None else sampling.draw
        values = current.draw(draw, sampling)
        variant = self.base_catalog.variant(values, fingerprint=stable_hash(values))
        self.catalog = variant
        self.dynamics = DynamicsParams.from_catalog(variant)
        return values, variant

    def trace_hash(self) -> str:
        return stable_hash(self.s.trace)

    def subgame_observation(self, event_id: str, role: str) -> tuple[float, ...]:
        active = self.s.active.get(event_id)
        if active is None:
            return ()
        return tuple(active.state.observation(role))

    def subgame_information(self, event_id: str, role: str) -> tuple[str, ...]:
        """The log tokens of *role*'s information state in the running event."""

        active = self.s.active.get(event_id)
        if active is None:
            return ()
        info = active.state.information_state(role)
        tokens = info[1] if isinstance(info, tuple) and len(info) == 2 else ()
        return tuple(str(t) for t in tokens)

    def decision_distribution(self, request: DecisionRequest) -> dict[str, float]:
        """Reference play for a decision event: a logit over its derived option values."""

        event = self.catalog.events[request.source]
        values = derive.decision_values(self.catalog, event, self.world)
        legal = tuple(o.id for o in request.options)
        return choose(values, legal, self.config.baseline_temperature)

    def baseline_distribution(self, request: DecisionRequest) -> dict[str, float]:
        if request.kind == "event_decision":
            return self.decision_distribution(request)
        active = self.s.active.get(request.source)
        if active is None or request.role is None:
            return {}
        major = self.catalog.events[active.event_id].major
        assert major is not None
        policy = baseline_policy(
            major.family,
            active.config,
            self.config.baseline_iterations,
            self.config.baseline_temperature,
        )
        dist = policy.action_probabilities(active.state)
        legal = {o.id for o in request.options}
        filtered = {a: p for a, p in dist.items() if a in legal and p > 0.0}
        total = sum(filtered.values())
        return {a: p / total for a, p in filtered.items()} if total > 0 else {}

    # ----------------------------------------------------------------- phases
    def _advance(self) -> None:
        s = self.s
        while s.pending is None and s.phase != Phase.DONE:
            if s.phase == Phase.BEGIN:
                s.world.quarter += 1
                s.phase = Phase.DIRECTOR
            elif s.phase == Phase.DIRECTOR:
                self._director()
                if s.world.ended is not None:
                    s.phase = Phase.RESOLVE
                    continue
                if calendar.quarter_of_year(s.world.quarter) == 1:
                    s.strategy_queue = [
                        (p, stage) for p in s.world.active_powers() for stage in strategic.STAGES
                    ]
                    s.commits = {p: {} for p in s.world.active_powers()}
                s.phase = Phase.STRATEGIC
                if self.config.capture:
                    self._capture()
            elif s.phase == Phase.STRATEGIC:
                if self._run_decisions() and self._run_strategy():
                    s.order = sorted(s.active, key=lambda eid: s.active[eid].seq)
                    s.cursor = 0
                    s.war_effort = dict.fromkeys(POWERS, 0.0)
                    s.direct_effort = dict.fromkeys(POWERS, 0.0)
                    s.phase = Phase.SUBGAMES
            elif s.phase == Phase.SUBGAMES:
                if self._run_subgames():
                    s.phase = Phase.RESOLVE
            elif s.phase == Phase.RESOLVE:
                self._resolve()
                s.phase = Phase.DYNAMICS
            elif s.phase == Phase.DYNAMICS:
                if s.world.ended is None:
                    step(s.world, self.dynamics, s.war_effort, s.direct_effort)
                s.phase = Phase.REWARDS
            elif s.phase == Phase.REWARDS:
                self._rewards()
                s.phase = Phase.END
            elif s.phase == Phase.END:
                if s.scope_done:
                    s.phase = Phase.DONE
                elif s.world.ended is not None or s.world.quarter >= self.config.end_quarter:
                    if s.world.ended is None and self.config.end_quarter < calendar.END:
                        s.truncated = True
                    s.phase = Phase.DONE
                else:
                    s.phase = Phase.BEGIN

    def _director(self) -> None:
        s = self.s
        requested = set(s.requested)
        s.requested.clear()
        leaders = {p: s.world.powers[p].leader for p in POWERS}
        result = run_director(self.catalog, s.world, s.seed, requested)
        s.requested.update(result.requested)
        changed = {p for p in POWERS if s.world.powers[p].leader != leaders[p]}
        if changed:
            self._rebind(changed)
        s.decision_queue.extend(event.id for event in result.decisions)
        for event in result.cutscenes:
            s.trace.append({"q": calendar.fmt(s.world.quarter), "event": event.id})
            if event.race is not None:
                self._record_standing(event.id, standing.race(self.catalog, event, s.world))
        for event in result.majors:
            self._instantiate(event.id)
        if self.config.generator and self.config.chance_mode == "sample" and s.world.ended is None:
            asked, s.crisis_requests = s.crisis_requests, []
            doc = self.catalog.raw.generator if self.catalog.raw is not None else None
            precedence = int((doc or {}).get("precedence_quarters", 4))
            for event_id in asked:  # agents' challenges, unless the region is taken
                region = generator.parse_id(event_id)[1]
                if not generator.blocked(self.catalog, s.world, region, precedence):
                    s.world.fired[event_id] = s.world.quarter
                    self._instantiate(event_id)
            for event_id in generator.propose(self.catalog, s.world, s.seed):
                s.world.fired[event_id] = s.world.quarter
                self._instantiate(event_id)

    def _capture(self) -> None:
        """Snapshot the state for each listed event that started this quarter: the entry
        state of a single-event scenario (its first instance only)."""

        s = self.s
        for event_id in sorted(self.config.capture):
            active = s.active.get(event_id)
            if active is None or event_id in self.captured or active.started != s.world.quarter:
                continue
            self.captured[event_id] = copy.deepcopy(s)

    def _rebind(self, powers: set[str]) -> None:
        """Re-bind running attrition wars whose roles include a power whose leader
        changed: bindings and derived values are recomputed at the current world and
        swapped into the running game, if its roles are unchanged."""

        s = self.s
        for event_id in sorted(s.active, key=lambda eid: s.active[eid].seq):
            active = s.active[event_id]
            major = self.catalog.events[event_id].major
            assert major is not None
            state = active.state
            if not isinstance(state, attrition.AttritionState):
                continue
            if not powers & set(state.roles):
                continue
            config = self.catalog.instantiate_config(event_id, s.world)
            game = build_family(major.family, config)
            if tuple(game.roles) != tuple(state.roles):
                continue
            state.params = game.params
            active.config = config
            s.trace.append({"q": calendar.fmt(s.world.quarter), "rebind": event_id})

    def _instantiate(self, event_id: str) -> None:
        s = self.s
        major = self.catalog.events[event_id].major
        assert major is not None
        config = self.catalog.instantiate_config(event_id, s.world)
        game = build_family(major.family, config)
        instance = s.instances.get(event_id, 0)
        s.instances[event_id] = instance + 1
        state = game.new_initial_state()
        s.seq += 1
        s.active[event_id] = ActiveEvent(
            event_id=event_id,
            seq=s.seq,
            started=s.world.quarter,
            config=config,
            state=state,
            rng=stream(s.seed, event_id, instance),
            round_marker=self._round_marker(state),
        )
        s.world.active[event_id] = s.world.quarter
        if major.region is not None and major.family == "attrition":
            s.world.regions[major.region].conflict = event_id
        s.trace.append({"q": calendar.fmt(s.world.quarter), "start": event_id})

    # --------------------------------------------------------------- strategy
    def _run_strategy(self) -> bool:
        s = self.s
        while s.strategy_queue:
            power, stage = s.strategy_queue[0]
            if s.world.powers[power].status != "active":
                s.strategy_queue.pop(0)
                continue
            request = strategic.request_for(
                s.world,
                power,
                stage,
                s.commits[power],
                self.catalog.historical_strategy,
                crises=self.config.generator and self.config.chance_mode == "sample",
            )
            if request is None:
                s.strategy_queue.pop(0)
                continue
            if self._asks(power) and len(request.options) > 1:
                s.pending = PendingDecision(request, ("strategy", power, stage))
                return False
            index = (
                0 if len(request.options) == 1 else self.npc_policy.choose(self, request, s.npc_rng)
            )
            self._commit_strategy(request, index, agent=False)
        return True

    def _run_decisions(self) -> bool:
        s = self.s
        while s.decision_queue:
            event = self.catalog.events[s.decision_queue[0]]
            spec = event.decision
            assert spec is not None
            power = spec.power
            if s.world.powers[power].status not in ("active", "insurgent"):
                s.decision_queue.pop(0)
                continue
            firing = s.instances.get(event.id, 0)
            target = spec.historical_choice(firing)
            options = tuple(Option(o, o.replace("_", " "), o == target) for o in spec.options)
            request = DecisionRequest(power, "event_decision", event.id, options)
            limit = self.config.playable_tier
            if limit is not None and spec.tier > limit:
                index = self.rails_policy.choose(self, request, s.npc_rng)
            elif self._asks(power):
                s.pending = PendingDecision(request, ("decision", event.id))
                return False
            else:
                index = self.npc_policy.choose(self, request, s.npc_rng)
            self._commit_decision(request, index, agent=False)
        return True

    def _commit_decision(self, request: DecisionRequest, index: int, *, agent: bool) -> None:
        s = self.s
        event_id = s.decision_queue.pop(0)
        option = request.options[index].id
        s.instances[event_id] = s.instances.get(event_id, 0) + 1
        s.decided.append((event_id, option))
        if agent:
            s.decisions += 1
        s.trace.append({"q": calendar.fmt(s.world.quarter), "decision": event_id, "c": option})

    def _commit_strategy(self, request: DecisionRequest, index: int, *, agent: bool) -> None:
        s = self.s
        power, stage = s.strategy_queue.pop(0)
        s.commits[power][stage] = request.options[index].id
        if agent:
            s.decisions += 1
            s.trace.append(
                {
                    "q": calendar.fmt(s.world.quarter),
                    "p": power,
                    "d": stage,
                    "c": request.options[index].id,
                }
            )

    def _asks(self, power: str, event_id: str | None = None) -> bool:
        """Whether *power*'s decision goes to its agent: an agent power, and with a scope
        only inside the scoped event."""

        if power not in self.config.agents:
            return False
        return self.config.scope is None or event_id == self.config.scope

    # --------------------------------------------------------------- subgames
    def on_rails(self, major: MajorSpec) -> bool:
        limit = self.config.playable_tier
        return limit is not None and major.tier > limit

    @staticmethod
    def _round_marker(state: Any) -> int:
        if isinstance(state, attrition.AttritionState):
            return state.round
        if isinstance(state, bargaining.BargainingState):
            return state.period
        return 0

    def _run_subgames(self) -> bool:
        s = self.s
        while s.cursor < len(s.order):
            active = s.active.get(s.order[s.cursor])
            if active is not None and not self._run_one(active):
                return False
            s.cursor += 1
        return True

    def _run_one(self, active: ActiveEvent) -> bool:
        """Advance one active event through this quarter's work.

        Returns True when the event needs nothing more this quarter, False when
        it is waiting for an agent decision.
        """

        s = self.s
        event = self.catalog.events[active.event_id]
        major = event.major
        assert major is not None
        state = active.state
        if any(f[1] == active.event_id for f in s.finished):
            return True
        while True:
            if state.is_terminal():
                s.finished.append((active.seq, active.event_id, state.outcome(), state.returns()))
                return True
            if major.cadence != "instant":
                marker = self._round_marker(state)
                if marker != active.round_marker:
                    active.round_marker = marker
                    self._record_war_effort(active)
                    return True
            player = state.current_player()
            if player == CHANCE:
                state.apply_action(self._chance(active))
                active.chance_count += 1
                continue
            role = player
            legal = state.legal_actions()
            if (
                major.cadence == "annual"
                and calendar.quarter_of_year(s.world.quarter) != 1
                and active.annual_choices.get(role) in legal
            ):
                state.apply_action(active.annual_choices[role])
                continue
            request = self._subgame_request(active, role, legal)
            if self.on_rails(major):
                rails = self.rails_policy if self.config.rails == "historical" else self.npc_policy
                index = rails.choose(self, request, s.npc_rng)
                self._apply_subgame_choice(active, request, index, agent=False)
                continue
            power = role if role in POWERS else None
            is_agent = (
                power is not None
                and self._asks(power, active.event_id)
                and s.world.powers[power].status in ("active", "insurgent")
            )
            if is_agent and len(request.options) > 1:
                s.pending = PendingDecision(request, ("subgame", active.event_id))
                return False
            index = (
                0 if len(request.options) == 1 else self.npc_policy.choose(self, request, s.npc_rng)
            )
            self._apply_subgame_choice(active, request, index, agent=False)

    def _subgame_request(
        self, active: ActiveEvent, role: str, legal: tuple[str, ...]
    ) -> DecisionRequest:
        major = self.catalog.events[active.event_id].major
        assert major is not None
        history = major.plan(active.started)[0].get(role, ())
        count = active.decisions.get(role, 0)
        target = history[min(count, len(history) - 1)] if history else None
        options = tuple(Option(a, a.replace("_", " "), a == target) for a in legal)
        return DecisionRequest(
            power=role,
            kind=f"subgame_{major.family}",
            source=active.event_id,
            options=options,
            role=role,
            family=major.family,
            round=self._round_marker(active.state),
        )

    def _apply_subgame_choice(
        self, active: ActiveEvent, request: DecisionRequest, index: int, *, agent: bool
    ) -> None:
        s = self.s
        role = request.role
        assert role is not None
        action = request.options[index].id
        active.state.apply_action(action)
        active.decisions[role] = active.decisions.get(role, 0) + 1
        major = self.catalog.events[active.event_id].major
        assert major is not None
        if major.cadence == "annual":
            active.annual_choices[role] = action
        if agent:
            s.decisions += 1
        s.trace.append(
            {"q": calendar.fmt(s.world.quarter), "e": active.event_id, "r": role, "c": action}
        )

    def _chance(self, active: ActiveEvent) -> str:
        outcomes = active.state.chance_outcomes()
        major = self.catalog.events[active.event_id].major
        assert major is not None
        on_rails = self.on_rails(major) and self.config.rails == "historical"
        if self.config.chance_mode == "historical" or on_rails:
            history = major.plan(active.started)[1]
            if history:
                target = history[min(active.chance_count, len(history) - 1)]
                if any(a == target for a, _ in outcomes):
                    return target
            best = max(range(len(outcomes)), key=lambda i: (outcomes[i][1], -i))
            return str(outcomes[best][0])
        return sample_action(dict(outcomes), active.rng)

    def _record_war_effort(self, active: ActiveEvent) -> None:
        state = active.state
        if not isinstance(state, attrition.AttritionState):
            return
        for role, choice in state.last_choices.items():
            if role in POWERS:
                self.s.war_effort[role] = self.s.war_effort.get(role, 0.0) + EFFORT.get(choice, 0.0)
                if choice == "direct":
                    self.s.direct_effort[role] = self.s.direct_effort.get(role, 0.0) + 1.0

    def _started_crisis(self, power: str, commits: Mapping[str, str]) -> str | None:
        """The generated crisis an agent's challenge or insurgency starts (E49)."""

        mode, region = commits.get("mode"), commits.get("region", "none")
        if mode not in ("challenge", "insurgency") or region == "none":
            return None
        leader = strategic.rival_leader(self.s.world, power, region)
        if leader is None:
            return None
        template = "standoff" if mode == "challenge" else "proxy_war"
        return generator.event_id(template, region, (leader, power))

    # ---------------------------------------------------------------- resolve
    def _resolve(self) -> None:
        s = self.s
        world = s.world
        for power, commits in s.commits.items():
            if commits:
                started = self._started_crisis(power, commits)
                strategic.apply_commits(world, power, commits)
                if started is not None:
                    s.crisis_requests.append(started)
        s.commits = {}
        for event_id, option in s.decided:
            decision = self.catalog.events[event_id].decision
            assert decision is not None
            ctx = EffectContext(leaders=self.catalog.leaders, seed=s.seed, source=event_id)
            world.outcomes[event_id] = option
            world.outcome_tags[event_id] = ()
            decision.options[option].effects.apply(world, ctx)
            s.requested.update(ctx.requested_events)
            narrative = decision.options[option].narrative
            world.log.append((world.quarter, event_id, narrative or f"{event_id}: {option}"))
        s.decided = []
        for _, event_id, outcome, returns in sorted(s.finished, key=lambda f: f[0]):
            active = s.active.pop(event_id)
            self._record_war_effort(active)
            event = self.catalog.events[event_id]
            major = event.major
            assert major is not None
            world.active.pop(event_id, None)
            world.outcomes[event_id] = outcome.label
            world.outcome_tags[event_id] = tuple(outcome.tags)
            if major.region is not None and world.regions[major.region].conflict == event_id:
                world.regions[major.region].conflict = None
            s.last_returns[event_id] = returns
            if event_id == self.config.scope:
                s.scope_done = True
            self._record_standing(
                event_id, standing.contest(self.catalog, event, active.config, outcome, world)
            )
            ctx = EffectContext(
                leaders=self.catalog.leaders,
                metrics=dict(outcome.metrics),
                seed=s.seed,
                source=event_id,
            )
            spec = major.outcomes[outcome.label]
            spec.effects.apply(world, ctx)
            narrative = spec.narrative
            for tag in outcome.tags:
                tag_spec = major.tags.get(tag)
                if tag_spec is not None:
                    tag_spec.effects.apply(world, ctx)
                    if tag_spec.narrative:
                        narrative = f"{narrative} {tag_spec.narrative}".strip()
            s.requested.update(ctx.requested_events)
            world.log.append(
                (world.quarter, event_id, narrative or f"{event.title}: {outcome.label}")
            )
            s.trace.append(
                {"q": calendar.fmt(world.quarter), "end": event_id, "outcome": outcome.label}
            )
        s.finished = []

    def _record_standing(self, event_id: str, changes: dict[str, float]) -> None:
        if not changes:
            return
        for ident, change in changes.items():
            self.s.standing[ident] = self.s.standing.get(ident, 0.0) + change
        self.s.standing_log.append((self.s.world.quarter, event_id, changes))

    def _loss_penalty(self, ident: str) -> float:
        """Losing a seat. Goal reward: the goal score drops to zero, so the penalty is
        the score held last quarter, at least ``loss_floor``. Score reward: the fixed
        ``elimination_penalty``."""

        settings = self.catalog.rewards
        if self.config.reward == "score":
            return float(settings.get("elimination_penalty", 5.0))
        parts = self.s.components.get(ident)
        held = (
            0.0
            if parts is None
            else goals.combine(parts, goals.row(self.world, ident, self.catalog).weights)
        )
        return max(held, float(settings.get("loss_floor", 0.1)))

    def _rewards(self) -> None:
        s = self.s
        world = s.world
        new_scores = rewards.scores(world, self.catalog)
        score_mode = self.config.reward == "score"
        for p in POWERS:
            holder = identity(world, p)
            previous = s.holders.get(p, holder)
            if holder != previous:
                # The seat changed hands (the Communists lost China): the outgoing
                # identity's game ends with the loss penalty, and the incoming one
                # starts from the current state with no reward for the jump.
                s.reward_buffer[previous] -= self._loss_penalty(previous)
                s.retired.add(previous)
                s.holders[p] = holder
                s.components[holder] = goals.components(world, holder, self.catalog)
            if world.powers[p].status == "eliminated" and p not in s.eliminated:
                s.eliminated.add(p)
                s.reward_buffer[holder] -= self._loss_penalty(holder)
            if score_mode:
                s.reward_buffer[holder] += new_scores[p] - s.scores.get(p, 0.0)
            elif world.powers[p].status != "eliminated":
                current = goals.components(world, holder, self.catalog)
                previous_parts = s.components.get(holder, current)
                weights = goals.row(world, holder, self.catalog).weights
                s.reward_buffer[holder] += goals.reward(previous_parts, current, weights)
                s.components[holder] = current
        s.scores = new_scores
        if world.ended is not None:
            terminal = rewards.terminal_bonus(world, self.catalog, end_bonuses=score_mode)
            for p, bonus in terminal.items():
                s.reward_buffer[identity(world, p)] += bonus


@dataclass
class EpisodeResult:
    seed: int
    end_reason: str
    final_quarter: str
    decisions: int
    returns: dict[str, float]
    timeline: list[tuple[str, str, str]]
    outcomes: dict[str, str]
    trace_hash: str
    standing: dict[str, float] = field(default_factory=dict)
    assumptions: dict[str, float] = field(default_factory=dict)  # the episode's draw, if any


def run_episode(
    engine: CampaignEngine,
    policies: Mapping[str, CampaignPolicy] | CampaignPolicy,
    seed: int,
    *,
    policy_seed: int | None = None,
    on_decision: Callable[[CampaignEngine, DecisionRequest], None] | None = None,
) -> EpisodeResult:
    """Play one campaign with the given policies (one for all agents, or per power)."""

    engine.reset(seed)
    rng = np.random.default_rng(seed if policy_seed is None else policy_seed)
    totals = dict.fromkeys(IDENTITIES, 0.0)
    while not engine.done:
        request = engine.pending()
        assert request is not None
        if on_decision is not None:
            on_decision(engine, request)
        policy = policies[request.power] if isinstance(policies, Mapping) else policies
        engine.act(policy.choose(engine, request, rng))
        for p, r in engine.pop_rewards().items():
            totals[p] += r
    for p, r in engine.pop_rewards().items():
        totals[p] += r
    world = engine.world
    timeline = [(calendar.fmt(q), eid, text) for q, eid, text in world.log]
    reason = world.ended or ("truncated" if engine.truncated else "campaign_end")
    return EpisodeResult(
        seed=seed,
        end_reason=reason,
        final_quarter=calendar.fmt(world.quarter),
        decisions=engine.s.decisions,
        returns=totals,
        timeline=timeline,
        outcomes=dict(world.outcomes),
        trace_hash=engine.trace_hash(),
        standing=dict(engine.s.standing),
        assumptions=dict(engine.s.assumptions),
    )


__all__ = [
    "CampaignEngine",
    "EngineConfig",
    "EpisodeResult",
    "Phase",
    "TabularPolicy",
    "baseline_policy",
    "run_episode",
]
