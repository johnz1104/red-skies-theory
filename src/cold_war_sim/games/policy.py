"""Policies over live game states, and conversion from tree policies."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Protocol, runtime_checkable

import numpy as np

from .protocol import CHANCE, TERMINAL, GameState
from .traverse import GameTree, TreePolicy


@runtime_checkable
class Policy(Protocol):
    """Distribution over the current player's legal actions."""

    def action_probabilities(self, state: GameState) -> dict[str, float]: ...


class UniformRandomPolicy:
    def action_probabilities(self, state: GameState) -> dict[str, float]:
        legal = state.legal_actions()
        return {action: 1.0 / len(legal) for action in legal}


class TabularPolicy:
    """Policy keyed by information state; unknown states fall back to uniform."""

    def __init__(self, table: Mapping[Hashable, Mapping[str, float]]) -> None:
        self._table = {key: dict(dist) for key, dist in table.items()}

    @property
    def table(self) -> dict[Hashable, dict[str, float]]:
        return self._table

    def action_probabilities(self, state: GameState) -> dict[str, float]:
        player = state.current_player()
        if player in (CHANCE, TERMINAL):
            raise ValueError("policies act only at decision states")
        legal = state.legal_actions()
        dist = self._table.get(state.information_state(player))
        if dist is None:
            return {action: 1.0 / len(legal) for action in legal}
        return {action: dist.get(action, 0.0) for action in legal}


def tabular_from_tree(tree: GameTree, policy: TreePolicy) -> TabularPolicy:
    table = {
        info.key: dict(zip(info.actions, policy[idx], strict=True))
        for idx, info in enumerate(tree.infosets)
    }
    return TabularPolicy(table)


def sample_action(distribution: Mapping[str, float], rng: np.random.Generator) -> str:
    """Sample one action; the final action absorbs floating-point residue."""

    u = float(rng.random())
    cumulative = 0.0
    last_positive: str | None = None
    for action, prob in distribution.items():
        if prob <= 0.0:
            continue
        last_positive = action
        cumulative += prob
        if u < cumulative:
            return action
    if last_positive is None:
        raise ValueError("distribution has no positive mass")
    return last_positive
