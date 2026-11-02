"""Masked-PPO self-play on single-event scenarios.

* **Agents:** one actor-critic per learning identity (USA, USSR, PRC, ROC, UK, FRA); an
  identity's network plays every role its power takes in any event. All agents learn at
  once (simultaneous self-play); computer-run roles keep reference play.
* **Inputs:** the campaign observation. Its numeric part is normalized by running
  statistics per identity; its last ``TOKEN_SLOTS`` entries are hashed log-token ids
  (the role's information state, newest first), embedded and concatenated, so the input
  identifies the information set exactly (E44).
* **Policy:** masked logits over ``MAX_OPTIONS`` (illegal options get -1e9).
* **Credit:** an agent's reward (the scenario's return, at the end) is credited to its last
  decision; GAE over each agent's decisions in an episode (gamma 1, lambda 0.95).
* **Update:** clipped PPO. Two objectives (E46):

  - ``qre`` (the default): the target is the logit agent QRE at ``temperature`` (in payoff
    units, the reference play's tau). At each information set, maximizing
    sum_a pi(a) Q(a) + tau H(pi) has the fixed point pi ~ exp(Q / tau), the QRE that
    ``games.algorithms.qre`` computes on trees, provided advantages stay in payoff units:
    rewards are the event's payoffs (the scenario return times kappa), advantages are not
    normalized, the entropy weight is tau, and the value loss is Huber. Every action keeps
    probability e^(-dQ/tau), so rare branches keep being explored (E45: Suez's committed
    Moscow never learned to intervene under plain PPO).
  - ``ppo``: E45's plain PPO, rewards scaled by the running standard deviation of returns,
    advantages normalized per minibatch, entropy weight ``entropy`` (not a temperature).
* **Rollouts:** worker processes, each with its own :class:`ScenarioEnv`; the learner
  sends weights and normalizer statistics each iteration. Episode seeds are fixed by the
  run seed and iteration, so a run is reproducible for a given worker count.
* **Freestyle** (``mode="campaign"``, E49): the whole campaign from 1945 instead of one
  event (:class:`CampaignTrainingEnv`), rewarded by the goal function each quarter, in
  tier-1 mode with tier-2 events on reference play (``rails``) and optionally the crisis
  generator.
* **League** (``league=K``, E49): every K iterations each identity's weights join a pool;
  in each episode each identity plays a past snapshot with probability ``league_share``
  (its steps then train nothing), so agents meet older opponents and conventions that
  beat only the current ones lose their edge (fictitious self-play, as in AlphaStar's
  league).
* **Best responses** (``trainable``): only the listed identities learn; the rest keep
  their weights. ``br.best_response`` trains one against a frozen checkpoint and
  measures its gain (approximate exploitability, for events without a tree).
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..catalog import Catalog, load_catalog
from ..decisions import MAX_OPTIONS
from ..engine import EngineConfig
from ..env import ColdWarEnv
from ..observation import TOKEN_SLOTS, TOKEN_SPACE
from ..scenario import Entry, ScenarioEnv, load_entries
from ..state import IDENTITIES, POWERS

EMBED = 16  # size of a token embedding
OBJECTIVES = ("qre", "ppo")
HUBER_DELTA = 10.0  # payoff units: quadratic for crisis stakes, linear for catastrophes


@dataclass(frozen=True)
class PPOConfig:
    entries: str  # a pickle of scenario entries (scenario.save_entries)
    events: tuple[str, ...] = ()  # () = every event in the entries
    iterations: int = 50
    episodes: int = 256  # per iteration, across workers
    workers: int = 4
    hidden: int = 256
    lr: float = 3e-4
    epochs: int = 4
    minibatch: int = 256
    clip: float = 0.2
    gamma: float = 1.0
    lam: float = 0.95
    entropy: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 0
    playable_tier: int | None = 1
    npc_policy: str = "equilibrium"
    agents: tuple[str, ...] = POWERS  # powers whose roles learn
    checkpoint_every: int = 10
    eval_every: int = 10  # exploitability on tree events; 0 = never
    objective: str = "qre"  # "qre": the logit QRE at `temperature`; "ppo": E45's plain PPO
    temperature: float = 0.1  # payoff units; the reference play's tau (E26)
    # the QRE path (qre objective): start hot and cool geometrically to `temperature` over
    # the first `anneal` share of iterations, as games.algorithms.qre does on trees (E46)
    temperature_start: float | None = None
    anneal: float = 0.5
    # after cooling, the learning rate decays linearly to lr_final (None: constant): near a
    # steep QRE, sampling noise otherwise moves play around it (E46)
    lr_final: float | None = None
    mode: str = "scenario"  # "scenario": single events from entries; "campaign": freestyle
    generator: bool = False  # campaign mode: the crisis generator
    rails: str = "reference"  # campaign mode: tier-2 events on reference play
    trainable: tuple[str, ...] = ()  # identities that learn; () = all
    league: int = 0  # snapshot every K iterations into the league pool; 0 = off
    league_share: float = 0.5  # chance an identity plays a past snapshot in an episode
    # scenario mode: nature's stretches taken in expectation, the sampled path weighted
    # (EngineConfig.chance_expected, E50); advantages and losses use the weights
    chance_expected: bool = False

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.temperature_start is not None and self.temperature_start < self.temperature:
            raise ValueError("temperature_start must be at least temperature")
        if not 0.0 < self.anneal <= 1.0:
            raise ValueError("anneal must lie in (0, 1]")
        if self.mode not in ("scenario", "campaign"):
            raise ValueError("mode must be scenario or campaign")
        if not set(self.trainable) <= set(IDENTITIES):
            raise ValueError(f"trainable identities must be drawn from {IDENTITIES}")
        if not 0.0 <= self.league_share <= 1.0:
            raise ValueError("league_share must lie in [0, 1]")
        if self.chance_expected and self.mode != "scenario":
            raise ValueError("chance_expected needs scenario mode (the event's own returns)")

    def lr_at(self, iteration: int) -> float:
        """The learning rate at *iteration*: constant while cooling, then (with ``lr_final``)
        linear decay to ``lr_final`` at the last iteration."""

        if self.lr_final is None:
            return self.lr
        start = self.anneal * self.iterations if self.temperature_start is not None else 0.0
        span = max(1.0, self.iterations - 1 - start)
        progress = min(1.0, max(0.0, (iteration - start) / span))
        return float(self.lr + (self.lr_final - self.lr) * progress)

    def temperature_at(self, iteration: int) -> float:
        """The qre objective's temperature at *iteration*: geometric cooling from
        ``temperature_start`` to ``temperature`` over the first ``anneal`` share."""

        if self.temperature_start is None:
            return self.temperature
        span = max(1.0, self.anneal * self.iterations)
        progress = min(1.0, iteration / span)
        return float(
            self.temperature_start * (self.temperature / self.temperature_start) ** progress
        )


# ------------------------------------------------------------------ statistics
@dataclass
class RunningStats:
    """Running mean and variance (Chan et al.'s parallel update)."""

    count: float = 1e-4
    mean: np.ndarray = field(default_factory=lambda: np.zeros(1))
    var: np.ndarray = field(default_factory=lambda: np.ones(1))

    @classmethod
    def of_size(cls, size: int) -> RunningStats:
        return cls(1e-4, np.zeros(size), np.ones(size))

    def update(self, batch: np.ndarray) -> None:
        if len(batch) == 0:
            return
        b_mean, b_var, b_count = batch.mean(axis=0), batch.var(axis=0), float(len(batch))
        delta = b_mean - self.mean
        total = self.count + b_count
        self.mean = self.mean + delta * b_count / total
        m2 = self.var * self.count + b_var * b_count + delta**2 * self.count * b_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        result: np.ndarray = np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10.0, 10.0)
        return result

    def state(self) -> dict[str, Any]:
        return {"count": self.count, "mean": self.mean.tolist(), "var": self.var.tolist()}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> RunningStats:
        return cls(float(state["count"]), np.asarray(state["mean"]), np.asarray(state["var"]))


# --------------------------------------------------------------------- network
class ActorCritic(nn.Module):
    """Numeric features plus embedded log tokens -> masked logits and a value."""

    def __init__(self, obs_size: int, hidden: int = 256) -> None:
        super().__init__()
        self.numeric = obs_size - TOKEN_SLOTS
        self.embed = nn.Embedding(TOKEN_SPACE, EMBED, padding_idx=0)
        self.body = nn.Sequential(
            nn.Linear(self.numeric + TOKEN_SLOTS * EMBED, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, MAX_OPTIONS)
        self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.zeros_(self.pi.bias)

    def forward(
        self, numeric: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embed(tokens).flatten(1)
        h = self.body(torch.cat([numeric, embedded], dim=1))
        logits = self.pi(h).masked_fill(mask == 0, -1e9)
        return logits, self.v(h).squeeze(-1)


def split(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Numeric features and token ids of observations (rows)."""

    obs = np.atleast_2d(obs)
    return obs[:, :-TOKEN_SLOTS], obs[:, -TOKEN_SLOTS:].astype(np.int64)


class LearnedPolicy:
    """Networks and normalizers as a policy over options (``evaluate.OptionPolicy``)."""

    def __init__(
        self, models: Mapping[str, ActorCritic], norms: Mapping[str, RunningStats]
    ) -> None:
        self.models = dict(models)
        self.norms = dict(norms)

    def forward(
        self, ident: str, obs: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        numeric, tokens = split(obs)
        x = torch.as_tensor(self.norms[ident].normalize(numeric), dtype=torch.float32)
        with torch.no_grad():
            logits, value = self.models[ident](
                x, torch.as_tensor(tokens), torch.as_tensor(np.atleast_2d(mask))
            )
            probs = torch.softmax(logits, dim=-1)
        return probs.numpy(), value.numpy()

    def probabilities(self, ident: str, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        probs, _ = self.forward(ident, obs, mask)
        result: np.ndarray = probs[0]
        return result


# --------------------------------------------------------------------- rollouts
@dataclass
class Episode:
    event_id: str
    entry: str
    outcome: str | None
    returns: dict[str, float]
    # identity -> obs, mask, action, logp, value, reward, survival
    steps: dict[str, dict[str, list[Any]]]
    masses: dict[str, float] = field(default_factory=dict)  # outcome -> probability mass


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
    if isinstance(env, ScenarioEnv) and env.config.chance_expected:
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
        self._totals: dict[str, float] = defaultdict(float)
        super().reset(seed)

    def _collect(self) -> None:
        super()._collect()
        for agent, reward in self.rewards.items():
            self._totals[agent] += reward

    def returns(self) -> dict[str, float]:
        return dict(self._totals)

    def outcome(self) -> str | None:
        return self.engine.world.ended or "end_of_game"


def make_env(config: PPOConfig, catalog: Catalog) -> ScenarioEnv | CampaignTrainingEnv:
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


def _models(weights: Weights, obs_size: int, hidden: int, cache: dict[str, Any]) -> dict[str, Any]:
    models = {}
    for ident, state in weights.items():
        model = cache.get(ident)
        if model is None:
            model = cache[ident] = ActorCritic(obs_size, hidden)
        model.load_state_dict({k: torch.as_tensor(v) for k, v in state.items()})
        model.eval()
        models[ident] = model
    return models


def _rollout(
    task: tuple[Weights, dict[str, dict[str, Any]], list[int], int, int, list[Weights], float],
) -> list[Episode]:
    weights, norms, seeds, obs_size, hidden, pool, share = task
    env = _WORKER["env"]
    current = _models(weights, obs_size, hidden, _WORKER["models"])
    stats = {k: RunningStats.from_state(v) for k, v in norms.items()}
    past = [
        _models(w, obs_size, hidden, _WORKER.setdefault(f"pool{n}", {})) for n, w in enumerate(pool)
    ]
    episodes = []
    for seed in seeds:
        rng = np.random.default_rng(seed + 7919)
        models, record = dict(current), set(current)
        if past:
            for ident in list(current):  # each identity may play a past snapshot instead
                if rng.random() < share:
                    models[ident] = past[int(rng.integers(len(past)))][ident]
                    record.discard(ident)
        policy = LearnedPolicy(models, stats)
        episodes.append(play_episode(env, policy, seed, rng, record=record))
    return episodes


# ---------------------------------------------------------------------- learner
def gae(
    rewards: Sequence[float],
    values: Sequence[float],
    gamma: float,
    lam: float,
    survival: Sequence[float] | None = None,
) -> np.ndarray:
    """Advantages of one agent's decisions in one episode (the last one is terminal).

    With *survival* (chance-expected returns, E50) the rewards are weighted from the
    episode's start, and step t's return is taken relative to its own weight s_t: the
    reward is r_t / s_t, and the step to the next decision is discounted by s_{t+1} / s_t.
    With every weight 1 this is the usual GAE."""

    n = len(rewards)
    advantages = np.zeros(n)
    running = 0.0
    for t in range(n - 1, -1, -1):
        following = values[t + 1] if t + 1 < n else 0.0
        reward, discount = rewards[t], gamma
        if survival is not None:
            reward = reward / survival[t]
            if t + 1 < n:
                discount = gamma * survival[t + 1] / survival[t]
        delta = reward + discount * following - values[t]
        running = delta + discount * lam * running
        advantages[t] = running
    return advantages


@dataclass
class Batch:
    obs: np.ndarray
    mask: np.ndarray
    action: np.ndarray
    logp: np.ndarray
    advantage: np.ndarray
    target: np.ndarray
    weight: np.ndarray | None = None  # each decision's survival weight (E50)


def batch_for(
    episodes: Iterable[Episode],
    ident: str,
    scale: float,
    gamma: float,
    lam: float,
    weighted: bool = False,
) -> Batch | None:
    """One identity's decisions; *weighted* uses the survival weights of chance-expected
    returns in the advantages and as the decisions' loss weights."""

    obs: list[np.ndarray] = []
    mask: list[np.ndarray] = []
    action: list[int] = []
    logp: list[float] = []
    adv: list[float] = []
    target: list[float] = []
    weight: list[float] = []
    for episode in episodes:
        record = episode.steps.get(ident)
        if not record:
            continue
        rewards = [r / scale for r in record["reward"]]
        values = record["value"]
        survival = record["survival"] if weighted else None
        a = gae(rewards, values, gamma, lam, survival)
        weight.extend(survival or ())
        obs.extend(record["obs"])
        mask.extend(record["mask"])
        action.extend(record["action"])
        logp.extend(record["logp"])
        adv.extend(a)
        target.extend(a + np.asarray(values))
    if not obs:
        return None
    return Batch(
        np.stack(obs),
        np.stack(mask),
        np.asarray(action),
        np.asarray(logp),
        np.asarray(adv),
        np.asarray(target),
        np.asarray(weight) if weighted else None,
    )


def _mean(x: torch.Tensor, w: torch.Tensor | None) -> torch.Tensor:
    return x.mean() if w is None else (w * x).mean()


def update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    norm: RunningStats,
    config: PPOConfig,
    generator: torch.Generator,
    temperature: float | None = None,
) -> dict[str, float]:
    """One PPO update; *temperature* is the qre objective's current tau (default: the
    config's)."""

    numeric, tokens = split(batch.obs)
    x = torch.as_tensor(norm.normalize(numeric), dtype=torch.float32)
    tok = torch.as_tensor(tokens)
    mask = torch.as_tensor(batch.mask)
    action = torch.as_tensor(batch.action)
    old_logp = torch.as_tensor(batch.logp, dtype=torch.float32)
    advantage = torch.as_tensor(batch.advantage, dtype=torch.float32)
    target = torch.as_tensor(batch.target, dtype=torch.float32)
    weights = None if batch.weight is None else torch.as_tensor(batch.weight, dtype=torch.float32)
    n = len(action)
    qre = config.objective == "qre"
    stats: dict[str, list[float]] = defaultdict(list)
    model.train()
    for _ in range(config.epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, config.minibatch):
            idx = order[start : start + config.minibatch]
            logits, value = model(x[idx], tok[idx], mask[idx])
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(action[idx])
            ratio = torch.exp(logp - old_logp[idx])
            adv = advantage[idx]
            # survival weights keep each information set's beliefs at the true posterior
            w = None if weights is None else weights[idx] / weights[idx].mean()
            if qre:
                # payoff units, so the entropy weight is a temperature
                if w is None:
                    value_loss = nn.functional.huber_loss(value, target[idx], delta=HUBER_DELTA)
                else:
                    huber = nn.functional.huber_loss(
                        value, target[idx], delta=HUBER_DELTA, reduction="none"
                    )
                    value_loss = _mean(huber, w)
                weight = config.temperature if temperature is None else temperature
            else:
                if len(idx) > 1 and w is None:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                elif len(idx) > 1:  # weighted like the losses
                    centre = _mean(adv, w)
                    adv = (adv - centre) / (_mean((adv - centre) ** 2, w).sqrt() + 1e-8)
                value_loss = 0.5 * _mean((value - target[idx]) ** 2, w)
                weight = config.entropy
            clipped = torch.clamp(ratio, 1 - config.clip, 1 + config.clip)
            policy_loss = -_mean(torch.min(ratio * adv, clipped * adv), w)
            entropy = _mean(dist.entropy(), w)
            loss = policy_loss + config.value_coef * value_loss - weight * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                stats["policy_loss"].append(float(policy_loss))
                stats["value_loss"].append(float(value_loss))
                stats["entropy"].append(float(entropy))
                stats["approx_kl"].append(float((old_logp[idx] - logp).mean()))
                stats["clipfrac"].append(float(((ratio - 1).abs() > config.clip).float().mean()))
    return {k: float(np.mean(v)) for k, v in stats.items()}


# ------------------------------------------------------------------- training
@dataclass
class Learner:
    config: PPOConfig
    obs_size: int
    models: dict[str, ActorCritic]
    optimizers: dict[str, torch.optim.Optimizer]
    obs_norms: dict[str, RunningStats]
    return_norms: dict[str, RunningStats]
    iteration: int = 0
    kappa: float = 100.0  # goal units -> payoff units (the qre objective learns on payoffs)

    @classmethod
    def create(cls, config: PPOConfig, obs_size: int, kappa: float = 100.0) -> Learner:
        torch.manual_seed(config.seed)
        models = {i: ActorCritic(obs_size, config.hidden) for i in IDENTITIES}
        optimizers: dict[str, torch.optim.Optimizer] = {
            i: torch.optim.Adam(m.parameters(), lr=config.lr, eps=1e-5) for i, m in models.items()
        }
        numeric = obs_size - TOKEN_SLOTS
        return cls(
            config,
            obs_size,
            models,
            optimizers,
            {i: RunningStats.of_size(numeric) for i in IDENTITIES},
            {i: RunningStats.of_size(1) for i in IDENTITIES},
            kappa=kappa,
        )

    def policy(self) -> LearnedPolicy:
        return LearnedPolicy(self.models, self.obs_norms)

    def weights(self) -> dict[str, dict[str, np.ndarray]]:
        return {
            i: {k: v.detach().cpu().numpy() for k, v in m.state_dict().items()}
            for i, m in self.models.items()
        }

    def step(self, episodes: Sequence[Episode]) -> dict[str, dict[str, float]]:
        generator = torch.Generator().manual_seed(self.config.seed * 1000 + self.iteration)
        stats: dict[str, dict[str, float]] = {}
        if self.config.lr_final is not None:
            for optimizer in self.optimizers.values():
                for group in optimizer.param_groups:
                    group["lr"] = self.config.lr_at(self.iteration)
        for ident in IDENTITIES:
            if self.config.trainable and ident not in self.config.trainable:
                continue
            if self.config.objective == "qre":
                scale = 1.0 / self.kappa  # rewards in payoff units
            else:
                scale = max(float(math.sqrt(self.return_norms[ident].var[0]) or 1.0), 1e-3)
            batch = batch_for(
                episodes,
                ident,
                scale,
                self.config.gamma,
                self.config.lam,
                weighted=self.config.chance_expected,
            )
            if batch is None:
                continue
            stats[ident] = update(
                self.models[ident],
                self.optimizers[ident],
                batch,
                self.obs_norms[ident],
                self.config,
                generator,
                self.config.temperature_at(self.iteration),
            )
            stats[ident]["decisions"] = float(len(batch.action))
        for ident in IDENTITIES:  # normalizers move after the update (the batch used the old)
            raw = [o for e in episodes for o in e.steps.get(ident, {}).get("obs", [])]
            if raw:
                self.obs_norms[ident].update(split(np.stack(raw))[0])
            returns = [e.returns[ident] for e in episodes if ident in e.returns]
            if returns:
                self.return_norms[ident].update(np.asarray(returns)[:, None])
        self.iteration += 1
        return stats

    def save(self, path: Path) -> None:
        torch.save(
            {
                "config": asdict(self.config),
                "obs_size": self.obs_size,
                "iteration": self.iteration,
                "kappa": self.kappa,
                "models": {i: m.state_dict() for i, m in self.models.items()},
                "optimizers": {i: o.state_dict() for i, o in self.optimizers.items()},
                "obs_norms": {i: n.state() for i, n in self.obs_norms.items()},
                "return_norms": {i: n.state() for i, n in self.return_norms.items()},
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> Learner:
        data = torch.load(path, weights_only=True)  # tensors and plain values only
        raw = dict(data["config"])
        raw["events"] = tuple(raw["events"])
        raw["agents"] = tuple(raw["agents"])
        raw.setdefault("objective", "ppo")  # checkpoints from before E46 used plain PPO
        raw.setdefault("temperature_start", None)
        raw.setdefault("anneal", 0.5)
        raw.setdefault("lr_final", None)
        raw["trainable"] = tuple(raw.get("trainable", ()))
        learner = cls.create(
            PPOConfig(**raw), int(data["obs_size"]), float(data.get("kappa", 100.0))
        )
        for ident, state in data["models"].items():
            learner.models[ident].load_state_dict(state)
        for ident, state in data["optimizers"].items():
            learner.optimizers[ident].load_state_dict(state)
        learner.obs_norms = {i: RunningStats.from_state(s) for i, s in data["obs_norms"].items()}
        learner.return_norms = {
            i: RunningStats.from_state(s) for i, s in data["return_norms"].items()
        }
        learner.iteration = int(data["iteration"])
        return learner


def summarize(episodes: Sequence[Episode]) -> dict[str, Any]:
    outcomes: dict[str, dict[str, float]] = defaultdict(dict)
    returns: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for e in episodes:
        tally = outcomes[e.event_id]
        # chance-expected returns: the mass of every terminal counted, not one outcome
        for label, n in (e.masses or {str(e.outcome): 1}).items():
            tally[label] = tally.get(label, 0) + n
        for ident, value in e.returns.items():
            returns[e.event_id][ident].append(value)
    return {
        "outcomes": {
            k: {label: round(n, 4) for label, n in v.items()} for k, v in sorted(outcomes.items())
        },
        "returns": {
            ev: {i: round(float(np.mean(v)), 5) for i, v in sorted(r.items())}
            for ev, r in sorted(returns.items())
        },
    }


def train(
    config: PPOConfig,
    out: Path,
    *,
    resume: Path | None = None,
    progress: Callable[[str], None] = print,
    evaluate: Callable[[LearnedPolicy], dict[str, Any]] | None = None,
) -> Learner:
    """Train for ``config.iterations`` iterations, logging to ``out/log.jsonl``."""

    out.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog()
    probe = make_env(config, catalog)
    learner = (
        Learner.load(resume)
        if resume
        else Learner.create(config, probe.observation_size, catalog.goals.kappa)
    )
    learner.config = config  # a resumed run may change what learns (best responses)
    (out / "config.json").write_text(json.dumps(asdict(config), indent=1))
    log = (out / "log.jsonl").open("a", encoding="utf-8")
    workers = max(1, config.workers)
    league: list[Weights] = []  # past snapshots (league play); not saved in checkpoints
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(config,)) as pool:
        while learner.iteration < config.iterations:
            it = learner.iteration
            started = time.perf_counter()
            base = (config.seed * 1_000_003 + it) * config.episodes
            seeds = [base + k for k in range(config.episodes)]
            chunks = [seeds[w::workers] for w in range(workers)]
            weights = learner.weights()
            norms = {i: n.state() for i, n in learner.obs_norms.items()}
            if config.league and it and it % config.league == 0:
                league.append(weights)
            tasks = [
                (weights, norms, c, learner.obs_size, config.hidden, league, config.league_share)
                for c in chunks
                if c
            ]
            episodes = [e for chunk in pool.map(_rollout, tasks) for e in chunk]
            collected = time.perf_counter()
            stats = learner.step(episodes)
            record: dict[str, Any] = {
                "iteration": it,
                "temperature": round(config.temperature_at(it), 5),
                "lr": config.lr_at(it),
                "episodes": len(episodes),
                "collect_s": round(collected - started, 2),
                "update_s": round(time.perf_counter() - collected, 2),
                "learn": {i: {k: round(v, 5) for k, v in s.items()} for i, s in stats.items()},
                **summarize(episodes),
            }
            done = learner.iteration
            if (
                evaluate is not None
                and config.eval_every
                and (done % config.eval_every == 0 or done == config.iterations)
            ):
                record["eval"] = evaluate(learner.policy())
            log.write(json.dumps(record) + "\n")
            log.flush()
            progress(
                f"iteration {done}/{config.iterations}: {len(episodes)} episodes, "
                f"collect {record['collect_s']} s, update {record['update_s']} s"
            )
            if done % config.checkpoint_every == 0 or done == config.iterations:
                learner.save(out / "checkpoint.pt")
    log.close()
    return learner


def scenario_entries(config: PPOConfig) -> dict[str, list[Entry]]:
    entries = load_entries(Path(config.entries))
    return {e: entries[e] for e in config.events} if config.events else entries


__all__ = [
    "ActorCritic",
    "Episode",
    "LearnedPolicy",
    "Learner",
    "PPOConfig",
    "RunningStats",
    "gae",
    "make_env",
    "play_episode",
    "scenario_entries",
    "train",
]
