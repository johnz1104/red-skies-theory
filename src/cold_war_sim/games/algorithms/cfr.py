"""CFR+ with alternating updates and linearly weighted policy averaging.

In two-player zero-sum games the average policy converges to a Nash
equilibrium. In general-sum or N-player games CFR has no such guarantee, so
callers should report :func:`nash_conv` of the returned policy rather than
assume it is an equilibrium.

With ``trembles > 0`` every role plays an epsilon-perturbed strategy during
traversal, so every information set is reached with positive probability and
off-path behaviour is optimized too (a perturbed-game approximation of
sequential rationality). The returned average policy is unperturbed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from ..traverse import CHANCE_NODE, TERMINAL_NODE, GameTree, TreePolicy
from .best_response import nash_conv


@dataclass(frozen=True, slots=True)
class CFRResult:
    average_policy: TreePolicy
    iterations: int
    nash_conv: float


def _regret_matching(regrets: list[float]) -> list[float]:
    positive = [max(r, 0.0) for r in regrets]
    total = sum(positive)
    if total <= 0.0:
        return [1.0 / len(regrets)] * len(regrets)
    return [p / total for p in positive]


def cfr_plus(
    tree: GameTree, *, iterations: int = 1000, delay: int = 0, trembles: float = 0.0
) -> CFRResult:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not 0.0 <= trembles < 1.0:
        raise ValueError("trembles must lie in [0, 1)")
    regrets = [[0.0] * len(info.actions) for info in tree.infosets]
    strategy_sums = [[0.0] * len(info.actions) for info in tree.infosets]
    # The current strategy is frozen for the duration of one traversal; regrets
    # updated at one history of an information set must not change the
    # strategy used at another history of the same set in the same pass.
    current: list[list[float]] = [_regret_matching(row) for row in regrets]

    def perturb(sigma: list[float]) -> list[float]:
        if trembles == 0.0:
            return sigma
        floor = trembles / len(sigma)
        return [(1.0 - trembles) * p + floor for p in sigma]

    played: list[list[float]] = [perturb(row) for row in current]
    n_roles = len(tree.roles)
    nodes = tree.nodes

    def walk(idx: int, role: int, reach_self: float, reach_other: float, weight: float) -> float:
        node = nodes[idx]
        if node.kind == TERMINAL_NODE:
            return node.returns[role]
        if node.kind == CHANCE_NODE:
            total = 0.0
            for prob, child in zip(node.probs, node.children, strict=True):
                if prob:
                    total += prob * walk(child, role, reach_self, reach_other * prob, weight)
            return total
        sigma = played[node.infoset]
        if node.player == role:
            child_values = [
                walk(child, role, reach_self * p, reach_other, weight)
                for p, child in zip(sigma, node.children, strict=True)
            ]
            value = sum(p * v for p, v in zip(sigma, child_values, strict=True))
            row = regrets[node.infoset]
            sums = strategy_sums[node.infoset]
            own = current[node.infoset]
            for slot, child_value in enumerate(child_values):
                row[slot] += reach_other * (child_value - value)
                sums[slot] += weight * reach_self * own[slot]
            return value
        total = 0.0
        for p, child in zip(sigma, node.children, strict=True):
            # Zero-probability branches are still visited so the updating role
            # accumulates a policy at information sets a deviator could reach.
            total += p * walk(child, role, reach_self, reach_other * p, weight)
        return total

    previous_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(previous_limit, 10_000))
    try:
        for t in range(1, iterations + 1):
            weight = float(max(t - delay, 0))
            for role in range(n_roles):
                walk(0, role, 1.0, 1.0, weight)
                for info_idx, info in enumerate(tree.infosets):
                    if info.role == role:
                        regrets[info_idx] = [max(r, 0.0) for r in regrets[info_idx]]
                        current[info_idx] = _regret_matching(regrets[info_idx])
                        played[info_idx] = perturb(current[info_idx])
    finally:
        sys.setrecursionlimit(previous_limit)

    average: TreePolicy = []
    for info_idx, info in enumerate(tree.infosets):
        total = sum(strategy_sums[info_idx])
        if total <= 0.0:
            average.append([1.0 / len(info.actions)] * len(info.actions))
        else:
            average.append([s / total for s in strategy_sums[info_idx]])
    return CFRResult(average, iterations, nash_conv(tree, average).nash_conv)
