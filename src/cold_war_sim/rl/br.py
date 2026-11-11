"""Approximate best responses: exploitability for events without a tree.

Wars, talks and the freestyle campaign have no game tree to compute exact best responses
on (``evaluate`` covers instant tree events). Here one identity's best response is
learned: a PPO run continues from a frozen checkpoint with only that identity learning
(plain PPO, low entropy: a best response, not a QRE), every other network fixed. Its
**gain** is its mean return as the best responder less its mean return under the
checkpoint's own policy, the others unchanged, over the same episode seeds (common random
numbers), with a paired standard error. Summed over identities, the gains estimate the
checkpoint's exploitability from below (a learned response is at most a best response).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .ppo import Learner, Snapshot, Weights, _init_worker, _rollout, train

EVAL_SEED = 10_000_019  # evaluation episodes use seeds training never draws


def _returns(
    config: Any,
    weights: Weights,
    norms: dict[str, dict[str, Any]],
    obs_size: int,
    seeds: list[int],
    identity: str,
    workers: int,
) -> dict[int, float]:
    """The identity's return in each episode it takes part in, by seed."""

    chunks = [seeds[w::workers] for w in range(workers)]
    empty: list[Snapshot] = []
    tasks = [(weights, norms, c, obs_size, config.hidden, empty, 0.0) for c in chunks if c]
    out: dict[int, float] = {}
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(config,)) as pool:
        for chunk, episodes in zip(
            (c for c in chunks if c), pool.map(_rollout, tasks), strict=True
        ):
            for seed, episode in zip(chunk, episodes, strict=True):
                if identity in episode.returns:
                    out[seed] = episode.returns[identity]
    return out


def best_response(
    checkpoint: Path,
    identity: str,
    out: Path,
    *,
    iterations: int,
    episodes: int,
    eval_episodes: int,
    workers: int,
    seed: int = 0,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Train *identity* alone against a frozen checkpoint and measure its gain: an
    approximate exploitability, for events without a tree.
    """

    base = Learner.load(checkpoint)
    config = base.config.replace(
        trainable=(identity,),
        objective="ppo",
        entropy=0.001,
        temperature_start=None,
        lr_final=None,
        league=0,
        eval_every=0,
        iterations=base.iteration + iterations,
        checkpoint_every=max(1, iterations),
        episodes=episodes,
        workers=workers,
        seed=base.config.seed + 1 + seed,
    )
    train(config, out, resume=checkpoint, progress=progress)
    responder = Learner.load(out / "checkpoint.pt")
    seeds = [EVAL_SEED + seed * eval_episodes + k for k in range(eval_episodes)]
    norms = {i: n.state() for i, n in base.obs_norms.items()}
    learned = _returns(config, base.weights(), norms, base.obs_size, seeds, identity, workers)
    weights = base.weights()
    weights[identity] = responder.weights()[identity]
    norms[identity] = responder.obs_norms[identity].state()
    responded = _returns(config, weights, norms, base.obs_size, seeds, identity, workers)
    common = sorted(set(learned) & set(responded))
    diff = np.asarray([responded[s] - learned[s] for s in common])
    n = len(common)
    gain = float(diff.mean()) if n else 0.0
    se = float(diff.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return {
        "identity": identity,
        "episodes": n,
        "learned": float(np.mean([learned[s] for s in common])) if n else 0.0,
        "responder": float(np.mean([responded[s] for s in common])) if n else 0.0,
        "gain": gain,
        "se": se,
        "gain_payoff": gain * base.kappa,
    }


__all__ = ["best_response"]
