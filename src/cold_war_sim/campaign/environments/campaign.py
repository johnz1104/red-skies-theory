"""An AEC-shaped multi-agent facade over :class:`CampaignEngine`.

The interface mirrors PettingZoo's agent-environment-cycle API without the
dependency, so an adapter is a thin wrapper later:

* ``possible_agents`` lists the learning identities of the agent seats: USA, USSR,
  PRC (the Communists), ROC (the Nationalists), UK, FRA. ``agents`` lists the live
  ones. If the Communists lose the civil war, the PRC agent is terminated (with
  the regime-loss penalty) and the ROC agent joins to play the China seat.
* ``agent_selection`` is the agent that must act next.
* ``last()`` returns that agent's observation, its reward accumulated since it
  last acted, and its termination/truncation flags.
* ``step(action)`` applies an option index; once an agent is terminated or
  truncated it must be stepped with ``None`` to remove it.

Rewards for every power accrue each quarter, whether or not it acted.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from ..catalog.core import Catalog
from ..decisions.requests import MAX_OPTIONS
from ..engine.core import CampaignEngine, EngineConfig
from ..world import calendar
from ..world.state import IDENTITIES, POWERS, identity, seat


class AgentCycle:
    """An agent-environment cycle over a campaign engine (PettingZoo's AEC shape).

    One agent is selected at a time: the identity whose decision the engine awaits, or an
    agent whose game is over and that must be stepped once with ``None`` to leave.
    """

    engine: CampaignEngine
    agents: list[str]
    rewards: dict[str, float]
    _cumulative_rewards: dict[str, float]
    terminations: dict[str, bool]
    truncations: dict[str, bool]
    infos: dict[str, dict[str, Any]]
    agent_selection: str

    @property
    def observation_size(self) -> int:
        """The length of the observation vector."""

        return len(self.engine.feature_names)

    @property
    def action_size(self) -> int:
        """The largest number of options a request can have."""

        return MAX_OPTIONS

    def observe(self, agent: str) -> np.ndarray:
        """The observation of *agent*."""

        return self.engine.observe(agent)

    def action_mask(self, agent: str) -> np.ndarray:
        """The mask of the options open to *agent*."""

        return self.engine.action_mask(agent)

    def last(self) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """The selected agent's observation, cumulative reward, termination, truncation and info."""

        agent = self.agent_selection
        return (
            self.observe(agent),
            self._cumulative_rewards[agent],
            self.terminations[agent],
            self.truncations[agent],
            self.infos[agent],
        )

    def step(self, action: int | None) -> None:
        """Answer the selected agent's request with *action* (``None`` for an agent that is done)."""

        agent = self.agent_selection
        if self.terminations.get(agent) or self.truncations.get(agent):
            if action is not None:
                raise ValueError("terminated or truncated agents must be stepped with None")
            self._remove(agent)
            self.rewards = dict.fromkeys(self.agents, 0.0)  # a removal step earns nothing
            self._select_next()
            return
        if action is None:
            raise ValueError("a live agent needs an action")
        request = self.engine.pending()
        if request is None or self.engine.agent_of(request.power) != agent:
            raise RuntimeError("environment is out of sync with the engine")
        self._cumulative_rewards[agent] = 0.0
        self.engine.act(int(action))
        self._collect()

    def _select_next(self) -> None:
        """Select an agent that is done and not yet removed, else the one whose decision is pending."""

        for agent in self.agents:
            if self.terminations[agent] or self.truncations[agent]:
                self.agent_selection = agent
                return
        request = self.engine.pending()
        self.agent_selection = self.engine.agent_of(request.power) if request is not None else ""

    def _remove(self, agent: str) -> None:
        """Drop an agent that is done from the tables."""

        self.agents.remove(agent)
        for table in (
            self.rewards,
            self._cumulative_rewards,
            self.terminations,
            self.truncations,
            self.infos,
        ):
            table.pop(agent, None)

    def _collect(self) -> None:
        """Pay the rewards that accrued since the last step and select the next agent."""

        raise NotImplementedError


class ColdWarEnv(AgentCycle):
    """The campaign as an agent-environment-cycle environment (PettingZoo's AEC shape,
    without depending on it). Agents are the identities holding a seat; the China seat's
    identity may change during a campaign.
    """

    metadata: ClassVar[dict[str, Any]] = {
        "name": "cold_war_campaign_v0",
        "is_parallelizable": False,
    }

    def __init__(self, catalog: Catalog | None = None, config: EngineConfig | None = None) -> None:
        config = config or EngineConfig()
        self.engine = CampaignEngine(catalog, config)
        self.possible_agents: tuple[str, ...] = tuple(
            i for i in IDENTITIES if seat(i) in config.agents
        )
        self._seen: set[str] = set()
        self.agents: list[str] = []
        self.rewards: dict[str, float] = {}
        self._cumulative_rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict[str, Any]] = {}
        self.agent_selection: str = ""

    def reset(self, seed: int = 0) -> None:
        """Start a campaign from *seed* with the identities that hold a seat as agents."""

        self.engine.reset(seed)
        world = self.engine.world
        holders = {identity(world, p) for p in self.engine.config.agents}
        self.agents = [
            a
            for a in self.possible_agents
            if a in holders and world.powers[seat(a)].status != "eliminated"
        ]
        self._seen = set(self.agents)
        self.rewards = dict.fromkeys(self.agents, 0.0)
        self._cumulative_rewards = dict.fromkeys(self.agents, 0.0)
        self.terminations = dict.fromkeys(self.agents, False)
        self.truncations = dict.fromkeys(self.agents, False)
        self.infos = {a: {} for a in self.agents}
        self._collect()

    # ---------------------------------------------------------------- helpers
    def _collect(self) -> None:
        """Pay the rewards that accrued, mark agents whose game is over, and let an identity
        that took over a seat join.
        """

        step_rewards = self.engine.pop_rewards()
        self.rewards = {a: step_rewards.get(a, 0.0) for a in self.agents}
        for agent in self.agents:
            self._cumulative_rewards[agent] += self.rewards[agent]
            self.infos[agent] = self._info()
        world = self.engine.world
        for agent in self.agents:
            power = seat(agent)
            if world.powers[power].status == "eliminated" or identity(world, power) != agent:
                self.terminations[agent] = True
        for power in self.engine.config.agents:
            holder = identity(world, power)
            if holder not in self._seen and world.powers[power].status != "eliminated":
                self._join(holder, step_rewards.get(holder, 0.0))
        if self.engine.done:
            for agent in self.agents:
                if self.engine.truncated:
                    self.truncations[agent] = True
                else:
                    self.terminations[agent] = True
        self._select_next()

    def _join(self, agent: str, reward: float) -> None:
        """Add *agent* mid-campaign with its first *reward*."""

        self._seen.add(agent)
        self.agents.append(agent)
        self.rewards[agent] = reward
        self._cumulative_rewards[agent] = reward
        self.terminations[agent] = False
        self.truncations[agent] = False
        self.infos[agent] = self._info()

    def _info(self) -> dict[str, Any]:
        """The date, and the assumptions' digest when numbers are being sampled."""

        info: dict[str, Any] = {
            "quarter": calendar.fmt(self.engine.world.quarter),
            "quarters_elapsed": self.engine.world.quarter - calendar.START,
        }
        if self.engine.s.variant is not None:  # sampling: the draw is never observed
            info["assumptions"] = self.engine.s.variant.fingerprint
        return info


__all__ = ["IDENTITIES", "POWERS", "ColdWarEnv"]
