"""PettingZoo AEC adapters for the campaign's environments.

Both environments already follow the agent-environment-cycle protocol; the adapter adds
the spaces PettingZoo expects. An observation is ``{"observation": float32 vector,
"action_mask": int8 vector}``; an action is an option index (``Discrete(MAX_OPTIONS)``),
legal where the mask is 1. Rewards that accrue before the first decision are kept apart
in ``reset_rewards`` (no action can change them).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

from ..decisions import MAX_OPTIONS
from ..env import ColdWarEnv
from ..scenario import ScenarioEnv


class CampaignAEC(AECEnv):
    """A PettingZoo ``AECEnv`` over a :class:`ScenarioEnv` or :class:`ColdWarEnv`."""

    def __init__(self, env: ScenarioEnv | ColdWarEnv) -> None:
        super().__init__()
        self.env = env
        self.metadata = {**env.metadata, "render_modes": []}
        self.possible_agents = list(env.possible_agents)
        size = env.observation_size
        self.observation_spaces = {
            a: spaces.Dict(
                {
                    "observation": spaces.Box(-np.inf, np.inf, (size,), np.float32),
                    "action_mask": spaces.MultiBinary(MAX_OPTIONS),
                }
            )
            for a in self.possible_agents
        }
        self.action_spaces = {a: spaces.Discrete(MAX_OPTIONS) for a in self.possible_agents}
        self.reset_rewards: dict[str, float] = {}
        self._sync()

    def observation_space(self, agent: str) -> spaces.Space[Any]:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space[Any]:
        return self.action_spaces[agent]

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        """Reset; a scenario takes ``options`` ``event`` and ``entry`` (others are ignored)."""

        seed = 0 if seed is None else int(seed)
        env = self.env
        if isinstance(env, ScenarioEnv):
            chosen = {k: v for k, v in (options or {}).items() if k in ("event", "entry")}
            env.reset(seed, **chosen)
        else:
            env.reset(seed)
        # PettingZoo starts every agent at zero: rewards that accrued before the first
        # decision (the campaign's quarters before its first strategy turn) are set aside
        self.reset_rewards = {a: env._cumulative_rewards.get(a, 0.0) for a in env.agents}
        for agent in env.agents:
            env._cumulative_rewards[agent] = 0.0
            env.rewards[agent] = 0.0
        self._sync()

    def observe(self, agent: str) -> dict[str, np.ndarray]:
        return {
            "observation": np.asarray(self.env.observe(agent), dtype=np.float32),
            "action_mask": np.asarray(self.env.action_mask(agent), dtype=np.int8),
        }

    def step(self, action: int | None) -> None:
        self.env.step(None if action is None else int(action))
        self._sync()

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None

    def _sync(self) -> None:
        env = self.env
        self.agents = list(env.agents)
        self.rewards = dict(env.rewards)
        self._cumulative_rewards = env._cumulative_rewards
        self.terminations = dict(env.terminations)
        self.truncations = dict(env.truncations)
        self.infos = dict(env.infos)
        self.agent_selection = env.agent_selection


__all__ = ["CampaignAEC"]
