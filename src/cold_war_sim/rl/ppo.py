"""Train campaign policies with PPO or an entropy-regularized payoff objective."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..campaign.catalog.core import load_catalog
from ..campaign.environments.observation import TOKEN_SLOTS
from ..campaign.environments.scenario import Entry, load_entries
from ..campaign.world.state import IDENTITIES, POWERS
from ..records import Record
from .networks import EMBED, ActorCritic, LearnedPolicy, RunningStats, split
from .rollouts import (
    _WORKER,
    CampaignTrainingEnv,
    Episode,
    Norms,
    Snapshot,
    Weights,
    _init_worker,
    _rollout,
    make_env,
    play_episode,
)

OBJECTIVES = ("qre", "ppo")
HUBER_DELTA = 10.0  # payoff units: quadratic for crisis stakes, linear for catastrophes


class PPOConfig(Record):
    """The training configuration (saved in checkpoints and ``config.json`` by field name).

    **Objectives**. ``qre``, the default, targets the logit agent QRE at
    ``temperature`` (payoff units): at each information set, maximizing
    ``sum_a pi(a) Q(a) + tau H(pi)`` has the fixed point ``pi ~ exp(Q / tau)``, provided
    advantages stay in payoff units. So rewards are the event's payoffs (the scenario's
    return times kappa), advantages are not normalized, the entropy weight is tau, and the
    value loss is Huber. ``ppo`` is plain PPO: returns scaled by their running deviation,
    advantages normalized per minibatch, entropy weight ``entropy``.

    **Annealing.** With ``temperature_start``, tau cools geometrically to ``temperature``
    over the first ``anneal`` share of iterations; with ``lr_final``, the learning rate then
    decays linearly to it.

    **Modes.** ``scenario`` plays single events from ``entries``; ``campaign`` plays the whole
    campaign from 1945, rewarded by the goal function each quarter (freestyle).

    **League** (``league=K``). Every K iterations each identity's weights join a pool, and in
    each episode each identity plays a past snapshot with probability ``league_share``
    (fictitious self-play). **Best responses:** only the ``trainable`` identities learn.

    **Chance-expected returns** (``chance_expected``): see
    ``EngineConfig.chance_expected``; advantages and losses use the survival weights.
    """

    def __init__(
        self,
        entries: str,
        events: tuple[str, ...] = (),
        iterations: int = 50,
        episodes: int = 256,
        workers: int = 4,
        hidden: int = 256,
        lr: float = 3e-4,
        epochs: int = 4,
        minibatch: int = 256,
        clip: float = 0.2,
        gamma: float = 1.0,
        lam: float = 0.95,
        entropy: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        seed: int = 0,
        playable_tier: int | None = 1,
        npc_policy: str = "equilibrium",
        agents: tuple[str, ...] = POWERS,
        checkpoint_every: int = 10,
        eval_every: int = 10,
        objective: str = "qre",
        temperature: float = 0.1,
        temperature_start: float | None = None,
        anneal: float = 0.5,
        lr_final: float | None = None,
        mode: str = "scenario",
        generator: bool = False,
        rails: str = "reference",
        trainable: tuple[str, ...] = (),
        league: int = 0,
        league_share: float = 0.5,
        chance_expected: bool = False,
    ) -> None:
        self.entries = entries  # a pickle of scenario entries (scenario.save_entries)
        self.events = events  # () = every event in the entries
        self.iterations = iterations
        self.episodes = episodes  # per iteration, across workers
        self.workers = workers
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.minibatch = minibatch
        self.clip = clip
        self.gamma = gamma
        self.lam = lam
        self.entropy = entropy
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        self.playable_tier = playable_tier
        self.npc_policy = npc_policy
        self.agents = agents  # powers whose roles learn
        self.checkpoint_every = checkpoint_every
        self.eval_every = eval_every  # exploitability on tree events; 0 = never
        self.objective = objective  # "qre": the logit QRE at `temperature`; "ppo": plain PPO
        self.temperature = temperature  # payoff units; the reference play's tau
        # the QRE path (qre objective): start hot and cool geometrically to `temperature` over
        # the first `anneal` share of iterations, as games.algorithms.qre does on trees
        self.temperature_start = temperature_start
        self.anneal = anneal
        # after cooling, the learning rate decays linearly to lr_final (None: constant): near a
        # steep QRE, sampling noise otherwise moves play around it
        self.lr_final = lr_final
        self.mode = mode  # "scenario": single events from entries; "campaign": freestyle
        self.generator = generator  # campaign mode: the crisis generator
        self.rails = rails  # campaign mode: tier-2 events on reference play
        self.trainable = trainable  # identities that learn; () = all
        self.league = league  # snapshot every K iterations into the league pool; 0 = off
        self.league_share = league_share  # chance an identity plays a past snapshot in an episode
        # scenario mode: nature's stretches taken in expectation, the sampled path weighted
        # (EngineConfig.chance_expected); advantages and losses use the weights
        self.chance_expected = chance_expected
        self._check()

    def _check(self) -> None:
        """Raise if a setting is out of range or two settings conflict."""

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


def gae(
    rewards: Sequence[float],
    values: Sequence[float],
    gamma: float,
    lam: float,
    survival: Sequence[float] | None = None,
) -> np.ndarray:
    """Estimate advantages along one agent's episode using GAE.

    With survival weights s_t, rewards r_t already include the probability of
    reaching each choice. Divide by s_t to value the choice conditional on
    reaching it, and discount the next value by d_t = gamma * s_(t+1) / s_t:

        delta_t = r_t / s_t + d_t * V_(t+1) - V_t
        A_t = delta_t + d_t * lam * A_(t+1)

    At the last choice the next value and advantage are zero. Without survival
    weights, use s_t = 1 throughout, giving ordinary GAE.
    """

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


class Batch(Record):
    """One identity's decisions from an iteration's episodes: observations, masks,
    actions, old log-probabilities, advantages, value targets and survival weights.
    """

    def __init__(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        action: np.ndarray,
        logp: np.ndarray,
        advantage: np.ndarray,
        target: np.ndarray,
        weight: np.ndarray | None = None,
    ) -> None:
        self.obs = obs
        self.mask = mask
        self.action = action
        self.logp = logp
        self.advantage = advantage
        self.target = target
        self.weight = weight  # each decision's survival weight


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
    """The mean of *x*, weighted by *w* if given."""

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


class Learner(Record):
    """The learner: each identity's network, optimizer and normalizers, and the iteration."""

    def __init__(
        self,
        config: PPOConfig,
        obs_size: int,
        models: dict[str, ActorCritic],
        optimizers: dict[str, torch.optim.Optimizer],
        obs_norms: dict[str, RunningStats],
        return_norms: dict[str, RunningStats],
        iteration: int = 0,
        kappa: float = 100.0,
    ) -> None:
        self.config = config
        self.obs_size = obs_size
        self.models = models
        self.optimizers = optimizers
        self.obs_norms = obs_norms
        self.return_norms = return_norms
        self.iteration = iteration
        self.kappa = kappa  # goal units -> payoff units (the qre objective learns on payoffs)

    @classmethod
    def create(cls, config: PPOConfig, obs_size: int, kappa: float = 100.0) -> Learner:
        """A fresh learner for observations of *obs_size*, seeded by the config."""

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
        """The current networks as a policy."""

        return LearnedPolicy(self.models, self.obs_norms)

    def weights(self) -> dict[str, dict[str, np.ndarray]]:
        """Copy the network parameters so saved opponents stay fixed during later training."""

        return {
            i: {k: v.detach().cpu().numpy().copy() for k, v in m.state_dict().items()}
            for i, m in self.models.items()
        }

    def step(self, episodes: Sequence[Episode]) -> dict[str, dict[str, float]]:
        """One PPO update of every trainable identity on *episodes*."""

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
            if self.config.trainable and ident not in self.config.trainable:
                continue  # a frozen identity keeps its preprocessing too (best responses)
            raw = [o for e in episodes for o in e.steps.get(ident, {}).get("obs", [])]
            if raw:
                self.obs_norms[ident].update(split(np.stack(raw))[0])
            returns = [e.returns[ident] for e in episodes if ident in e.returns]
            if returns:
                self.return_norms[ident].update(np.asarray(returns)[:, None])
        self.iteration += 1
        return stats

    def save(self, path: Path) -> None:
        """Save the learner (tensors and plain values only)."""

        torch.save(
            {
                "config": self.config.to_dict(),
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
        """A learner saved by :meth:`save`."""

        data = torch.load(path, weights_only=True)  # tensors and plain values only
        raw = dict(data["config"])
        raw["events"] = tuple(raw["events"])
        raw["agents"] = tuple(raw["agents"])
        raw.setdefault("objective", "ppo")  # older checkpoints without this field used PPO
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
    """Each event's outcome shares and mean returns over *episodes*."""

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
    """Train the policies and write progress to ``out/log.jsonl``.

    Each learning identity has one actor-critic shared across its event roles.
    Identities learn together; computer-run roles retain reference strategies.
    Episode seeds depend on the run seed and iteration, making episode
    collection reproducible for a fixed worker count.
    """

    out.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog()
    probe = make_env(config, catalog)
    learner = (
        Learner.load(resume)
        if resume
        else Learner.create(config, probe.observation_size, catalog.goals.kappa)
    )
    learner.config = config  # a resumed run may change what learns (best responses)
    (out / "config.json").write_text(json.dumps(config.to_dict(), indent=1))
    log = (out / "log.jsonl").open("a", encoding="utf-8")
    workers = max(1, config.workers)
    league: list[Snapshot] = []  # past snapshots (league play); not saved in checkpoints
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
                league.append((weights, norms))
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
    """The entries of the config's events."""

    entries = load_entries(Path(config.entries))
    return {e: entries[e] for e in config.events} if config.events else entries


__all__ = [
    "EMBED",
    "HUBER_DELTA",
    "IDENTITIES",
    "OBJECTIVES",
    "_WORKER",
    "ActorCritic",
    "Batch",
    "CampaignTrainingEnv",
    "Episode",
    "LearnedPolicy",
    "Learner",
    "Norms",
    "PPOConfig",
    "RunningStats",
    "Snapshot",
    "Weights",
    "_init_worker",
    "_rollout",
    "batch_for",
    "gae",
    "make_env",
    "play_episode",
    "scenario_entries",
    "split",
    "summarize",
    "train",
    "update",
]
