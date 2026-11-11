"""Neural policies and observation normalization for campaign decisions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn

from ..campaign.decisions.requests import MAX_OPTIONS
from ..campaign.environments.observation import TOKEN_SLOTS, TOKEN_SPACE
from ..records import Record

EMBED = 16  # size of a token embedding


class RunningStats(Record):
    """Running mean and variance (Chan et al.'s parallel update)."""

    def __init__(
        self,
        count: float = 1e-4,
        mean: np.ndarray | None = None,
        var: np.ndarray | None = None,
    ) -> None:
        self.count = count
        self.mean: np.ndarray = np.zeros(1) if mean is None else mean
        self.var: np.ndarray = np.ones(1) if var is None else var

    @classmethod
    def of_size(cls, size: int) -> RunningStats:
        """Statistics of a vector of *size* (unit variance to start)."""

        return cls(1e-4, np.zeros(size), np.ones(size))

    def update(self, batch: np.ndarray) -> None:
        """Merge a batch into the mean and variance (Chan's parallel algorithm)."""

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
        """*x* standardized by the statistics, clipped to [-10, 10]."""

        result: np.ndarray = np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10.0, 10.0)
        return result

    def state(self) -> dict[str, Any]:
        """The statistics as plain values, for checkpoints and workers."""

        return {"count": self.count, "mean": self.mean.tolist(), "var": self.var.tolist()}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> RunningStats:
        """Statistics from :meth:`state`'s values."""

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
        """Masked logits over the options (illegal ones at -1e9) and the state value."""

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
        """Option probabilities and the value for one observation of *ident*."""

        numeric, tokens = split(obs)
        x = torch.as_tensor(self.norms[ident].normalize(numeric), dtype=torch.float32)
        with torch.no_grad():
            logits, value = self.models[ident](
                x, torch.as_tensor(tokens), torch.as_tensor(np.atleast_2d(mask))
            )
            probs = torch.softmax(logits, dim=-1)
        return probs.numpy(), value.numpy()

    def probabilities(self, ident: str, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Option probabilities for one observation of *ident*."""

        probs, _ = self.forward(ident, obs, mask)
        result: np.ndarray = probs[0]
        return result
