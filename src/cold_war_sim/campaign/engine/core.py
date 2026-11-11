"""Advance the campaign through quarterly decisions, event play and world changes."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

import numpy as np

from ...families import attrition, bargaining, build_family
from ...families.attrition import EFFORT, choose
from ...games import CHANCE, TabularPolicy, sample_action
from .. import assumptions
from ..catalog.core import Catalog, MajorSpec, load_catalog
from ..catalog.effects import EffectContext
from ..decisions import strategic
from ..decisions.policies import CampaignPolicy, HistoricalPolicy, make_policy
from ..decisions.reference_play import QRE_TOLERANCE, baseline_policy, reference_tree_policy
from ..decisions.requests import DecisionRequest, Option
from ..environments import observation
from ..events import generator
from ..events.director import run_director
from ..payoffs import derive, goals, rewards, standing
from ..world import calendar
from ..world.dynamics import DynamicsParams, step
from ..world.state import IDENTITIES, POWERS, WorldState, identity, seat
from .reproducibility import stable_hash, stream
from .state import (
    CHANCE_MODES,
    REWARD_MODES,
    ActiveEvent,
    EngineConfig,
    EngineState,
    EpisodeResult,
    ExpectedReturns,
    PendingDecision,
    Phase,
)

NATURE_NODE_CAP = 50_000  # sample larger stretches of chance one step at a time


class CampaignEngine:
    """Run the world until an agent needs to choose an action.

    Each quarter advances through BEGIN, DIRECTOR, STRATEGIC (Q1 only),
    SUBGAMES, RESOLVE, DYNAMICS, REWARDS and END. Computer-run choices are
    resolved along the way. ``pending`` describes the next agent choice;
    ``act`` applies it and continues.

    Mutable campaign data lives in EngineState. ``snapshot`` copies that
    state for later play, while the catalog and cached strategies are shared.
    """

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
        """Start a campaign from *seed* (drawing the catalog's numbers first if sampling is
        on) and run to the first agent decision.
        """

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
        """The current world."""

        return self.s.world

    @property
    def done(self) -> bool:
        """Whether the campaign is over."""

        return self.s.phase == Phase.DONE

    @property
    def truncated(self) -> bool:
        """Whether the campaign stopped at ``end_quarter`` before the calendar's end."""

        return self.s.truncated

    def pending(self) -> DecisionRequest | None:
        """The request awaiting an agent's answer, if any."""

        return self.s.pending.request if self.s.pending is not None else None

    def act(self, index: int) -> None:
        """Answer the pending request with option *index* and run to the next one."""

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
        """The observation vector of the identity *agent*."""

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
        """A readable observation of the identity *agent*."""

        return observation.describe(self, seat(agent))

    def action_mask(self, agent: str) -> np.ndarray:
        """1 for each option open to *agent* now, 0 elsewhere (all 0 when it is not its turn)."""

        from ..decisions.requests import MAX_OPTIONS

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
        """A deep copy of the engine's state."""

        return copy.deepcopy(self.s)

    def restore(self, snapshot: EngineState) -> None:
        """Continue from a *snapshot* (under the catalog variant it was taken with)."""

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
        """A hash of the campaign's trace: equal exactly when two campaigns played out alike."""

        return stable_hash(self.s.trace)

    def subgame_observation(self, event_id: str, role: str) -> tuple[float, ...]:
        """The family's own observation of the running *event_id* for *role*."""

        active = self.s.active.get(event_id)
        if active is None:
            return ()
        return tuple(active.state.observation(role))

    def pending_commits(self, power: str) -> dict[str, str]:
        """*power*'s own strategy choices so far this year, by stage (they apply once its
        turn is complete; other powers' stay secret until then)."""

        return dict(self.s.commits.get(power, {}))

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
        """Reference play's choice probabilities for *request* (a sub-game move or a decision event)."""

        if request.kind == "event_decision":
            return self.decision_distribution(request)
        active = self.s.active.get(request.source)
        if active is None or request.role is None:
            return {}
        return self._reference_distribution(active, active.state, [o.id for o in request.options])

    def _reference_distribution(
        self, active: ActiveEvent, state: Any, legal: Sequence[str]
    ) -> dict[str, float]:
        """Reference play's probabilities at *state*, any state of *active*'s game."""

        major = self.catalog.events[active.event_id].major
        assert major is not None
        policy = baseline_policy(
            major.family,
            active.config,
            self.config.baseline_iterations,
            self.config.baseline_temperature,
        )
        dist = policy.action_probabilities(state)
        allowed = set(legal)
        filtered = {a: p for a, p in dist.items() if a in allowed and p > 0.0}
        total = sum(filtered.values())
        return {a: p / total for a, p in filtered.items()} if total > 0 else {}

    def pop_expected(self) -> dict[str, float]:
        """The scoped event's chance-expected payoffs accrued since the last call, by role
        (``EngineConfig.chance_expected``; empty when off)."""

        expected = self.s.expected
        if expected is None:
            return {}
        values, expected.values = expected.values, {}
        return values

    # ----------------------------------------------------------------- phases
    def _advance(self) -> None:
        """Run phases until an agent must decide or the campaign is over."""

        s = self.s
        while s.pending is None and s.phase != Phase.DONE:
            self._PHASES[s.phase](self)

    def _phase_begin(self) -> None:
        """A new quarter starts."""

        self.s.world.quarter += 1
        self.s.phase = Phase.DIRECTOR

    def _phase_director(self) -> None:
        """Events fire; in the first quarter of a year the strategy turns are queued."""

        s = self.s
        self._director()
        if s.world.ended is not None:
            s.phase = Phase.RESOLVE
            return
        if calendar.quarter_of_year(s.world.quarter) == 1:
            s.strategy_queue = [
                (p, stage) for p in s.world.active_powers() for stage in strategic.STAGES
            ]
            s.commits = {p: {} for p in s.world.active_powers()}
        s.phase = Phase.STRATEGIC
        if self.config.capture:
            self._capture()

    def _phase_strategic(self) -> None:
        """Decision events and strategy turns are put; the phase holds while one is pending."""

        s = self.s
        if self._run_decisions() and self._run_strategy():
            s.order = sorted(s.active, key=lambda eid: s.active[eid].seq)
            s.cursor = 0
            s.war_effort = dict.fromkeys(POWERS, 0.0)
            s.direct_effort = dict.fromkeys(POWERS, 0.0)
            s.phase = Phase.SUBGAMES

    def _phase_subgames(self) -> None:
        """Running events play this quarter's moves, in the order they started."""

        if self._run_subgames():
            self.s.phase = Phase.RESOLVE

    def _phase_resolve(self) -> None:
        """Finished events' outcomes, the year's strategy and decisions take effect."""

        self._resolve()
        self.s.phase = Phase.DYNAMICS

    def _phase_dynamics(self) -> None:
        """The world moves one quarter under the dynamics."""

        s = self.s
        if s.world.ended is None:
            step(s.world, self.dynamics, s.war_effort, s.direct_effort)
        s.phase = Phase.REWARDS

    def _phase_rewards(self) -> None:
        """Each identity is paid the change in its goal score."""

        self._rewards()
        self.s.phase = Phase.END

    def _phase_end(self) -> None:
        """The campaign ends with its scope, the world or the calendar; else the next quarter."""

        s = self.s
        if s.scope_done:
            s.phase = Phase.DONE
        elif s.world.ended is not None or s.world.quarter >= self.config.end_quarter:
            if s.world.ended is None and self.config.end_quarter < calendar.END:
                s.truncated = True
            s.phase = Phase.DONE
        else:
            s.phase = Phase.BEGIN

    _PHASES: ClassVar[dict[Phase, Callable[[CampaignEngine], None]]] = {
        Phase.BEGIN: _phase_begin,
        Phase.DIRECTOR: _phase_director,
        Phase.STRATEGIC: _phase_strategic,
        Phase.SUBGAMES: _phase_subgames,
        Phase.RESOLVE: _phase_resolve,
        Phase.DYNAMICS: _phase_dynamics,
        Phase.REWARDS: _phase_rewards,
        Phase.END: _phase_end,
    }

    def _director(self) -> None:
        """Run the director: start the events that fire, queue decision events, and re-bind
        running wars whose powers changed leader.
        """

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
        """Start major event *event_id*: bind its config at the current world, build its
        game, and drop the optional roles whose condition fails.
        """

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
        """Put the year's strategy turns in order; returns False while one awaits an agent."""

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
        """Put the queued decision events to their powers; returns False while one awaits an agent."""

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
        """Record the option chosen for a decision event; it takes effect in RESOLVE."""

        s = self.s
        event_id = s.decision_queue.pop(0)
        option = request.options[index].id
        s.instances[event_id] = s.instances.get(event_id, 0) + 1
        s.decided.append((event_id, option))
        if agent:
            s.decisions += 1
        s.trace.append({"q": calendar.fmt(s.world.quarter), "decision": event_id, "c": option})

    def _commit_strategy(self, request: DecisionRequest, index: int, *, agent: bool) -> None:
        """Record a strategy choice; the year's choices take effect together in RESOLVE."""

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
        """Return whether this choice belongs to an agent.

        In a single-event scenario, agents choose only within that event.
        """

        if power not in self.config.agents:
            return False
        return self.config.scope is None or event_id == self.config.scope

    # --------------------------------------------------------------- subgames

    def on_rails(self, major: MajorSpec) -> bool:
        """Whether *major* is above the playable tier and so is played without asking agents."""

        limit = self.config.playable_tier
        return limit is not None and major.tier > limit

    @staticmethod
    def _round_marker(state: Any) -> int:
        """The round of a game that runs over several quarters (0 for an instant one)."""

        if isinstance(state, attrition.AttritionState):
            return state.round
        if isinstance(state, bargaining.BargainingState):
            return state.period
        return 0

    def _run_subgames(self) -> bool:
        """Advance each running event through this quarter; returns False while one awaits an agent."""

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
            if (
                self._expects(active, major)
                and (player == CHANCE or not self._agent_decides(active, player, state))
                and self._follow_expected(active)
            ):
                continue
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
            if self._agent_decides(active, role, state):
                s.pending = PendingDecision(request, ("subgame", active.event_id))
                return False
            index = (
                0 if len(request.options) == 1 else self.npc_policy.choose(self, request, s.npc_rng)
            )
            self._apply_subgame_choice(active, request, index, agent=False)

    def _agent_decides(self, active: ActiveEvent, role: str, state: Any) -> bool:
        """Whether *role*'s choice at *state* of *active*'s game goes to an agent."""

        return (
            role in POWERS
            and self._asks(role, active.event_id)
            and self.s.world.powers[role].status in ("active", "insurgent")
            and len(state.legal_actions()) > 1
        )

    def expects_scope(self) -> bool:
        """Return whether the scenario averages over chance outcomes.

        This requires ``chance_expected`` and an event that ends within one quarter.
        Wars and talks spanning several quarters keep their sampled outcomes.
        """

        scope = self.config.scope
        active = self.s.active.get(scope) if scope is not None else None
        if active is None:
            return False
        major = self.catalog.events[active.event_id].major
        return major is not None and self._expects(active, major)

    def _expects(self, active: ActiveEvent, major: MajorSpec) -> bool:
        """Whether *active* is the scoped instant event whose chance is taken in expectation."""

        return (
            self.config.chance_expected
            and active.event_id == self.config.scope
            and major.cadence == "instant"
            and not self.on_rails(major)
        )

    def _follow_expected(self, active: ActiveEvent) -> bool:
        """Average the outcomes before the next agent choice, then sample a path onward.

        Explore chance and computer-run choices until each path ends or reaches an
        agent. Add ending payoffs weighted by their probabilities. If any paths reach
        an agent, choose among them in proportion to their probabilities and multiply
        the survival weight by their total probability. Otherwise, sample an ending.

        Return False without changing the event if the search exceeds the node limit
        or a computer-run policy does not provide probabilities; ordinary sampling
        then handles the next step.
        """

        values: dict[str, float] = {}
        outcomes: dict[str, float] = {}
        ends: list[tuple[float, list[str]]] = []  # agent decisions
        terminals: list[tuple[float, list[str]]] = []
        stack: list[tuple[Any, float, list[str]]] = [(active.state.clone(), 1.0, [])]
        nodes = 0
        while stack:
            state, prob, path = stack.pop()
            nodes += 1
            if nodes > NATURE_NODE_CAP:
                return False
            if state.is_terminal():
                for role, value in state.returns().items():
                    values[role] = values.get(role, 0.0) + prob * value
                label = state.outcome().label
                outcomes[label] = outcomes.get(label, 0.0) + prob
                terminals.append((prob, path))
                continue
            player = state.current_player()
            if player == CHANCE:
                branches = [(a, p) for a, p in state.chance_outcomes() if p > 0.0]
            elif self._agent_decides(active, player, state):
                ends.append((prob, path))
                continue
            else:
                dist = self._role_distribution(active, state)
                if dist is None:
                    return False
                branches = list(dist.items())
            for action, p in branches:
                child = state.clone()
                child.apply_action(action)
                stack.append((child, prob * p, [*path, action]))
        expected = self.s.expected
        if expected is None:
            expected = self.s.expected = ExpectedReturns()
        expected.add(values, outcomes)
        mass = sum(p for p, _ in ends)
        pool = ends if mass > 0.0 else terminals
        weights = np.asarray([p for p, _ in pool])
        chosen = pool[int(active.rng.choice(len(pool), p=weights / weights.sum()))][1]
        expected.survival = expected.survival * mass if mass > 0.0 else 0.0
        for action in chosen:  # through the usual bookkeeping: trace, counts
            state = active.state
            player = state.current_player()
            if player == CHANCE:
                state.apply_action(action)
                active.chance_count += 1
            else:
                request = self._subgame_request(active, player, state.legal_actions())
                self._apply_subgame_choice(active, request, request.index_of(action), agent=False)
        return True

    def _role_distribution(self, active: ActiveEvent, state: Any) -> dict[str, float] | None:
        """A computer-run role's choice probabilities at *state*, as the computer-run
        policy draws them (None for a policy that gives none)."""

        legal = list(state.legal_actions())
        if len(legal) == 1:
            return {legal[0]: 1.0}
        if self.config.npc_policy == "random":
            return {a: 1.0 / len(legal) for a in legal}
        if self.config.npc_policy != "equilibrium":
            return None
        return self._reference_distribution(active, state, legal) or None

    def _subgame_request(
        self, active: ActiveEvent, role: str, legal: tuple[str, ...]
    ) -> DecisionRequest:
        """The request for *role*'s move in a running event, with the historical option marked."""

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
        """Apply the chosen option to the event's game and record it in the trace."""

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
        """The chance outcome at the event's chance node: the historical draw in historical
        mode or on historical rails, else a draw from the event's own stream.
        """

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
        """Add each power's effort in this round of a war to the quarter's war effort."""

        state = active.state
        if not isinstance(state, attrition.AttritionState):
            return
        for role, choice in state.last_choices.items():
            if role in POWERS:
                self.s.war_effort[role] = self.s.war_effort.get(role, 0.0) + EFFORT.get(choice, 0.0)
                if choice == "direct":
                    self.s.direct_effort[role] = self.s.direct_effort.get(role, 0.0) + 1.0

    def _started_crisis(self, power: str, commits: Mapping[str, str]) -> str | None:
        """The generated crisis an agent's challenge or insurgency starts."""

        mode, region = commits.get("mode"), commits.get("region", "none")
        if mode not in ("challenge", "insurgency") or region == "none":
            return None
        leader = strategic.rival_leader(self.s.world, power, region)
        if leader is None:
            return None
        template = "standoff" if mode == "challenge" else "proxy_war"
        return generator.event_id(template, region, (leader, power))

    def _resolve(self) -> None:
        """Apply what the quarter decided: strategy choices, decision events' options, and
        the outcome effects and standing changes of the events that finished.
        """

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
            world.resolved[event_id] = world.quarter
            if major.region is not None and world.regions[major.region].conflict == event_id:
                world.regions[major.region].conflict = None
            s.last_returns[event_id] = returns
            if event_id == self.config.scope:
                s.scope_done = True
                if self._expects(active, major):  # the realized end, at the weight left
                    if s.expected is None:
                        s.expected = ExpectedReturns()
                    s.expected.close(returns, outcome.label)
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
        """Add *changes* to the standing scoreboard and its log."""

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
            else goals.combine(parts, goals.row(self.s.world, ident, self.catalog).weights)
        )
        return max(held, float(settings.get("loss_floor", 0.1)))

    def _rewards(self) -> None:
        """Pay each identity the change in its goal score (or in the world score), with the
        terminal terms, and hand a seat that changed regime to its new identity.
        """

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
    "CHANCE_MODES",
    "EFFORT",
    "NATURE_NODE_CAP",
    "QRE_TOLERANCE",
    "REWARD_MODES",
    "ActiveEvent",
    "CampaignEngine",
    "EngineConfig",
    "EngineState",
    "EpisodeResult",
    "ExpectedReturns",
    "PendingDecision",
    "Phase",
    "TabularPolicy",
    "baseline_policy",
    "reference_tree_policy",
    "run_episode",
]
