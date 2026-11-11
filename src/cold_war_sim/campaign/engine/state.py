"""Campaign settings and the state needed to pause, resume and score an episode."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

import numpy as np

from ...records import Record
from .. import assumptions
from ..catalog.core import Catalog
from ..decisions.requests import DecisionRequest
from ..world import calendar
from ..world.state import POWERS, WorldState

CHANCE_MODES = ("sample", "historical")
REWARD_MODES = ("goals", "score")  # goal changes, or the original world-score reward


class Phase(StrEnum):
    """The phases of a quarter, and ``DONE`` once the campaign is over."""

    BEGIN = "begin"
    DIRECTOR = "director"
    STRATEGIC = "strategic"
    SUBGAMES = "subgames"
    RESOLVE = "resolve"
    DYNAMICS = "dynamics"
    REWARDS = "rewards"
    END = "end"
    DONE = "done"


class EngineConfig(Record):
    """Settings for agents, chance, rewards, and playable events.

    ``scope`` limits agent choices to one event. ``capture`` names events whose
    starting states should be saved for later scenarios.
    """

    def __init__(
        self,
        agents: tuple[str, ...] = POWERS,
        chance_mode: str = "sample",
        npc_policy: str = "historical",
        baseline_iterations: int = 300,
        baseline_temperature: float = 0.1,
        end_quarter: int = calendar.END,
        playable_tier: int | None = None,
        reward: str = "goals",
        sampling: assumptions.Sampling | None = None,
        capture: frozenset[str] = frozenset(),
        scope: str | None = None,
        generator: bool = False,
        rails: str = "historical",
        chance_expected: bool = False,
    ) -> None:
        self.agents = agents
        self.chance_mode = chance_mode
        self.npc_policy = npc_policy
        self.baseline_iterations = baseline_iterations
        # computer-run reference play is a logit (quantal) response at this temperature, in
        # payoff units: near-ties become mixtures; 0 = exact
        self.baseline_temperature = baseline_temperature
        self.end_quarter = end_quarter
        self.playable_tier = playable_tier  # None: every major event is playable
        self.reward = reward
        # draw the catalog's numbers within the assumptions register's ranges at each reset;
        # None: the catalog's own values
        self.sampling = sampling
        # single-event scenarios (campaign/scenario.py): snapshot the state when a listed event
        # starts (``CampaignEngine.captured``), and with a scope ask agents only inside that
        # event and stop once it resolves
        self.capture = capture
        self.scope = scope
        # the crisis generator (data/generator.yaml): crises where the world makes them likely;
        # never with historical chance, whose replay has only named crises
        self.generator = generator
        # Events above playable_tier use either their historical choices and chance,
        # or reference play with sampled chance that responds to the current world.
        self.rails = rails
        # For an instant scenario with sampled chance, average over chance and
        # computer-run choices before the next agent decision. ExpectedReturns
        # tracks the probability of continuing, reducing noise from rare catastrophes.
        self.chance_expected = chance_expected
        self._check()

    def _check(self) -> None:
        """Raise if a setting is out of range or two settings conflict."""

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
        if self.chance_expected and self.chance_mode != "sample":
            raise ValueError("chance_expected needs sampled chance")


class ActiveEvent(Record):
    """A running major event: its game as bound when it started, the game's state, its
    own random stream, and counts of the decisions and chance draws made in it.
    """

    def __init__(
        self,
        event_id: str,
        seq: int,
        started: int,
        config: dict[str, Any],
        state: Any,
        rng: np.random.Generator,
        decisions: dict[str, int] | None = None,
        chance_count: int = 0,
        round_marker: int = 0,
        annual_choices: dict[str, str] | None = None,
    ) -> None:
        self.event_id = event_id
        self.seq = seq
        self.started = started
        self.config = config
        self.state = state
        self.rng = rng
        self.decisions: dict[str, int] = {} if decisions is None else decisions
        self.chance_count = chance_count
        self.round_marker = round_marker
        self.annual_choices: dict[str, str] = {} if annual_choices is None else annual_choices


class ExpectedReturns(Record):
    """Track expected payoffs along a sampled path through a scenario.

    Before each agent choice, add the payoffs of possible endings, weighted by
    their probabilities. Play continues along a sampled path that reaches an
    agent. ``survival`` is the probability of reaching these choices so far.

    For a stretch with ending probabilities p_i and payoffs u_i, add
    ``survival * sum_i p_i u_i``. Multiply survival by the total probability
    of reaching another agent. A later realized ending counts at the remaining
    weight. This keeps the return estimate unbiased, including rare endings.
    """

    def __init__(
        self,
        survival: float = 1.0,
        values: dict[str, float] | None = None,
        outcomes: dict[str, float] | None = None,
    ) -> None:
        self.survival = survival
        self.values: dict[str, float] = (
            {} if values is None else values
        )  # role -> weighted, not yet popped
        self.outcomes: dict[str, float] = (
            {} if outcomes is None else outcomes
        )  # label -> probability mass

    def add(self, values: Mapping[str, float], outcomes: Mapping[str, float]) -> None:
        """Count *values* and outcome masses reached with the path's current weight."""

        for role, value in values.items():
            self.values[role] = self.values.get(role, 0.0) + self.survival * value
        for label, mass in outcomes.items():
            self.outcomes[label] = self.outcomes.get(label, 0.0) + self.survival * mass

    def close(self, returns: Mapping[str, float], label: str) -> None:
        """The sampled path ended: count its realized returns and leave no weight."""

        self.add(returns, {label: 1.0})
        self.survival = 0.0


class PendingDecision(Record):
    """The request an agent must answer, and what it is for: a strategy turn, a move in
    a sub-game, or a decision event.
    """

    def __init__(
        self,
        request: DecisionRequest,
        target: tuple[str, ...],
    ) -> None:
        self.request = request
        self.target = target  # ("strategy", power, stage) or ("subgame", event_id)


class EngineState(Record):
    """Everything that changes during a campaign: the world, the phase, the running
    events, the queues of the current quarter, rewards, standing and the trace.
    """

    def __init__(
        self,
        seed: int,
        world: WorldState,
        npc_rng: np.random.Generator | None = None,
    ) -> None:
        self.seed = seed
        self.world = world
        self.phase: Phase = Phase.DIRECTOR
        self.strategy_queue: list[tuple[str, str]] = []
        self.commits: dict[str, dict[str, str]] = {}
        self.order: list[str] = []
        self.cursor: int = 0
        self.active: dict[str, ActiveEvent] = {}
        self.finished: list[tuple[int, str, Any, dict[str, float]]] = []
        self.requested: set[str] = set()
        self.war_effort: dict[str, float] = {}
        self.crisis_requests: list[str] = []  # agents' challenges
        self.direct_effort: dict[str, float] = {}  # fought with own troops
        self.pending: PendingDecision | None = None
        self.scores: dict[str, float] = {}
        self.components: dict[str, dict[str, float]] = {}  # by identity
        self.reward_buffer: dict[str, float] = {}
        self.eliminated: set[str] = set()
        self.holders: dict[str, str] = {}  # seat -> identity holding it
        self.retired: set[str] = set()  # identities that lost their seat
        self.standing: dict[str, float] = {}  # scoreboard; never a reward
        self.standing_log: list[tuple[int, str, dict[str, float]]] = []
        self.seq: int = 0
        self.instances: dict[str, int] = {}
        self.decision_queue: list[str] = []  # decision events to put
        self.decided: list[tuple[str, str]] = []  # (event, option) to apply
        self.npc_rng: np.random.Generator = np.random.default_rng(0) if npc_rng is None else npc_rng
        self.truncated: bool = False
        self.trace: list[dict[str, Any]] = []
        self.decisions: int = 0
        self.last_returns: dict[str, dict[str, float]] = {}
        self.assumptions: dict[str, float] = {}  # this episode's draw
        self.variant: Catalog | None = None  # the catalog with the draw applied (None: no sampling)
        self.scope_done: bool = False  # the scoped event has resolved (EngineConfig.scope)
        self.expected: ExpectedReturns | None = None  # EngineConfig.chance_expected


# Baselines are pure functions of (family, bound config): cache them across


class EpisodeResult(Record):
    """A finished campaign: how it ended, each identity's return and standing, the
    timeline, the events' outcomes and the trace hash.
    """

    def __init__(
        self,
        seed: int,
        end_reason: str,
        final_quarter: str,
        decisions: int,
        returns: dict[str, float],
        timeline: list[tuple[str, str, str]],
        outcomes: dict[str, str],
        trace_hash: str,
        standing: dict[str, float] | None = None,
        assumptions: dict[str, float] | None = None,
    ) -> None:
        self.seed = seed
        self.end_reason = end_reason
        self.final_quarter = final_quarter
        self.decisions = decisions
        self.returns = returns
        self.timeline = timeline
        self.outcomes = outcomes
        self.trace_hash = trace_hash
        self.standing: dict[str, float] = {} if standing is None else standing
        self.assumptions: dict[str, float] = (
            {} if assumptions is None else assumptions
        )  # the episode's draw, if any
