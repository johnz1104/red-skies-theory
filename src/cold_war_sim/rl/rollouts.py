"""Play training episodes and collect the decisions used to update the policies."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ..campaign.catalog.core import Catalog, load_catalog
from ..campaign.engine.core import EngineConfig
from ..campaign.environments.campaign import ColdWarEnv
from ..campaign.environments.scenario import ScenarioEnv, load_entries
from ..records import Record

if TYPE_CHECKING:
    from .ppo import PPOConfig
from .networks import ActorCritic, LearnedPolicy, RunningStats


class Episode(Record):
    """One played episode: its event, entry and outcome, each identity's return, and each
    identity's recorded decisions.
    """

    def __init__(
        self,
        event_id: str,
        entry: str,
        outcome: str | None,
        returns: dict[str, float],
        steps: dict[str, dict[str, list[Any]]],
        masses: dict[str, float] | None = None,
    ) -> None:
        self.event_id = event_id
        self.entry = entry
        self.outcome = outcome
        self.returns = returns
        # identity -> obs, mask, action, logp, value, reward, survival
        self.steps = steps
        self.masses: dict[str, float] = (
            {} if masses is None else masses
        )  # outcome -> probability mass


def play_episode(
    env: ScenarioEnv | CampaignTrainingEnv,
    policy: LearnedPolicy,
    seed: int,
    rng: np.random.Generator,
    event: str | None = None,
    record: Iterable[str] | None = None,
) -> Episode:
    """One episode; only identities in *record* (default: all) keep their steps."""

    keep = None if record is None else set(record)
    env.reset(seed, event=event)
    steps: dict[str, dict[str, list[Any]]] = {}
    while env.agents:
        agent = env.agent_selection
        obs, reward, term, trunc, info = env.last()
        record = steps.get(agent)
        if record is not None and record["reward"]:
            record["reward"][-1] += reward
        if term or trunc:
            env.step(None)
            continue
        mask = env.action_mask(agent)
        probs, value = policy.forward(agent, obs, mask)
        p = probs[0].astype(np.float64)
        p = p / p.sum()
        action = int(rng.choice(len(p), p=p))
        if keep is not None and agent not in keep:
            env.step(action)
            continue
        record = steps.setdefault(
            agent,
            {k: [] for k in ("obs", "mask", "action", "logp", "value", "reward", "survival")},
        )
        record["obs"].append(np.asarray(obs, dtype=np.float32))
        record["mask"].append(np.asarray(mask, dtype=np.int8))
        record["action"].append(action)
        record["logp"].append(math.log(max(p[action], 1e-12)))
        record["value"].append(float(value[0]))
        record["reward"].append(0.0)
        record["survival"].append(float(info.get("survival", 1.0)))
        env.step(action)
    masses: dict[str, float] = {}
    if isinstance(env, ScenarioEnv) and env.expected:
        masses = env.outcome_masses()
    return Episode(
        env.event_id,
        env.entry.label if env.entry else "",
        env.outcome(),
        env.returns(),
        steps,
        masses,
    )


_WORKER: dict[str, Any] = {}


def _init_worker(config: PPOConfig) -> None:
    """Give a worker process its environment (one thread for torch)."""

    torch.set_num_threads(1)
    catalog = load_catalog()
    _WORKER["env"] = make_env(config, catalog)
    _WORKER["models"] = {}


class CampaignTrainingEnv(ColdWarEnv):
    """The whole campaign (freestyle) with :class:`ScenarioEnv`'s training interface:
    rewards are the goal function's per quarter, and an episode's return is their sum."""

    event_id = "campaign"
    entry = None

    def reset(self, seed: int = 0, event: str | None = None, entry: int | None = None) -> None:
        """Start a campaign and reset the return totals."""

        self._totals: dict[str, float] = defaultdict(float)
        super().reset(seed)

    def _collect(self) -> None:
        """Pay the rewards and add them to each identity's return."""

        super()._collect()
        for agent, reward in self.rewards.items():
            self._totals[agent] += reward

    def returns(self) -> dict[str, float]:
        """Each identity's return so far."""

        return dict(self._totals)

    def outcome(self) -> str | None:
        """How the campaign ended."""

        return self.engine.world.ended or "end_of_game"


def make_env(config: PPOConfig, catalog: Catalog) -> ScenarioEnv | CampaignTrainingEnv:
    """The training environment of *config*: scenarios from its entries, or the campaign."""

    if config.mode == "campaign":
        return CampaignTrainingEnv(
            catalog,
            EngineConfig(
                npc_policy=config.npc_policy,
                playable_tier=config.playable_tier,
                agents=config.agents,
                generator=config.generator,
                rails=config.rails,
            ),
        )
    entries = load_entries(Path(config.entries))
    if config.events:
        missing = sorted(set(config.events) - set(entries))
        if missing:
            raise ValueError(f"no entries for {missing}")
        entries = {e: entries[e] for e in config.events}
    engine_config = EngineConfig(
        npc_policy=config.npc_policy,
        playable_tier=config.playable_tier,
        agents=config.agents,
        chance_expected=config.chance_expected,
    )
    return ScenarioEnv(entries, catalog, engine_config)


Weights = dict[str, dict[str, np.ndarray]]
Norms = dict[str, dict[str, Any]]  # identity -> RunningStats.state()
Snapshot = tuple[Weights, Norms]  # a league entry: networks with their own normalizers


def _models(weights: Weights, obs_size: int, hidden: int, cache: dict[str, Any]) -> dict[str, Any]:
    """Networks with *weights*, reusing the cached ones of a worker."""

    models = {}
    for ident, state in weights.items():
        model = cache.get(ident)
        if model is None:
            model = cache[ident] = ActorCritic(obs_size, hidden)
        model.load_state_dict({k: torch.as_tensor(v) for k, v in state.items()})
        model.eval()
        models[ident] = model
    return models


def _stats(norms: Norms) -> dict[str, RunningStats]:
    """Normalizers from their saved states."""

    return {k: RunningStats.from_state(v) for k, v in norms.items()}


def _rollout(
    task: tuple[Weights, Norms, list[int], int, int, list[Snapshot], float],
) -> list[Episode]:
    """Play one worker's episodes, each identity with the current networks or, with
    probability *share*, a league snapshot (whose steps are not recorded).
    """

    weights, norms, seeds, obs_size, hidden, pool, share = task
    env = _WORKER["env"]
    current = _models(weights, obs_size, hidden, _WORKER["models"])
    stats = _stats(norms)
    past = [
        (_models(w, obs_size, hidden, _WORKER.setdefault(f"pool{n}", {})), _stats(old))
        for n, (w, old) in enumerate(pool)
    ]
    episodes = []
    for seed in seeds:
        rng = np.random.default_rng(seed + 7919)
        models, used, record = dict(current), dict(stats), set(current)
        if past:
            for ident in list(current):  # each identity may play a past snapshot instead
                if rng.random() < share:
                    snapshot, snapshot_stats = past[int(rng.integers(len(past)))]
                    models[ident] = snapshot[ident]
                    used[ident] = snapshot_stats[ident]  # with the normalizer it learned on
                    record.discard(ident)
        policy = LearnedPolicy(models, used)
        episodes.append(play_episode(env, policy, seed, rng, record=record))
    return episodes
