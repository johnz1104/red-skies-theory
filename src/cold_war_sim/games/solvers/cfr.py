"""CFR+ with alternating updates and linearly weighted policy averaging.

In two-player zero-sum games the average policy converges to a Nash equilibrium. In
general-sum or N-player games there is no such guarantee, so judge the result by its
NashConv.
"""

from __future__ import annotations

import sys

from ...records import Record
from ..tree import CHANCE_NODE, TERMINAL_NODE, GameTree, TreePolicy
from .best_response import nash_conv


class CFRResult(Record):
    """The average policy after ``iterations`` of CFR+, with its NashConv."""

    def __init__(self, average_policy: TreePolicy, iterations: int, nash_conv: float) -> None:
        self.average_policy = average_policy
        self.iterations = iterations
        self.nash_conv = nash_conv


def _regret_matching(regrets: list[float]) -> list[float]:
    """Probabilities proportional to positive regret (uniform when there is none)."""

    positive = [max(r, 0.0) for r in regrets]
    total = sum(positive)
    if total <= 0.0:
        return [1.0 / len(regrets)] * len(regrets)
    return [p / total for p in positive]


class _CFRPlus:
    """The tables of one CFR+ run: regrets, strategy sums and the current strategies.

    ``current`` is frozen for one traversal: regrets updated at one history of an
    information set must not change the strategy used at another in the same pass.
    ``played`` is ``current`` mixed with a uniform tremble.
    """

    def __init__(self, tree: GameTree, trembles: float) -> None:
        self.tree = tree
        self.trembles = trembles
        self.regrets = [[0.0] * len(info.actions) for info in tree.infosets]
        self.strategy_sums = [[0.0] * len(info.actions) for info in tree.infosets]
        self.current = [_regret_matching(row) for row in self.regrets]
        self.played = [self.perturb(row) for row in self.current]

    def perturb(self, sigma: list[float]) -> list[float]:
        """*sigma* with a share ``trembles`` of uniform play mixed in."""

        if self.trembles == 0.0:
            return sigma
        floor = self.trembles / len(sigma)
        return [(1.0 - self.trembles) * p + floor for p in sigma]

    def walk(
        self, idx: int, role: int, reach_self: float, reach_other: float, weight: float
    ) -> float:
        """*role*'s value at node *idx* under the played strategies; at its own nodes,
        add counterfactual regret and the weighted strategy to the tables."""

        node = self.tree.nodes[idx]
        if node.kind == TERMINAL_NODE:
            return node.returns[role]
        walk = self.walk
        if node.kind == CHANCE_NODE:
            total = 0.0
            for prob, child in zip(node.probs, node.children, strict=True):
                if prob:
                    total += prob * walk(child, role, reach_self, reach_other * prob, weight)
            return total
        sigma = self.played[node.infoset]
        if node.player == role:
            child_values = [
                walk(child, role, reach_self * p, reach_other, weight)
                for p, child in zip(sigma, node.children, strict=True)
            ]
            value = sum(p * v for p, v in zip(sigma, child_values, strict=True))
            row = self.regrets[node.infoset]
            sums = self.strategy_sums[node.infoset]
            own = self.current[node.infoset]
            for slot, child_value in enumerate(child_values):
                row[slot] += reach_other * (child_value - value)
                sums[slot] += weight * reach_self * own[slot]
            return value
        total = 0.0
        for p, child in zip(sigma, node.children, strict=True):
            # zero-probability branches are still visited, so the updating role
            # accumulates a policy at information sets a deviator could reach
            total += p * walk(child, role, reach_self, reach_other * p, weight)
        return total

    def update(self, role: int) -> None:
        """CFR+'s step for *role*: floor its regrets at zero and rematch its strategies."""

        for info_idx, info in enumerate(self.tree.infosets):
            if info.role == role:
                self.regrets[info_idx] = [max(r, 0.0) for r in self.regrets[info_idx]]
                self.current[info_idx] = _regret_matching(self.regrets[info_idx])
                self.played[info_idx] = self.perturb(self.current[info_idx])

    def average(self) -> TreePolicy:
        """The weighted average strategy (uniform where nothing was accumulated)."""

        average: TreePolicy = []
        for sums in self.strategy_sums:
            total = sum(sums)
            if total <= 0.0:
                average.append([1.0 / len(sums)] * len(sums))
            else:
                average.append([s / total for s in sums])
        return average


def cfr_plus(
    tree: GameTree, *, iterations: int = 1000, delay: int = 0, trembles: float = 0.0
) -> CFRResult:
    """Run CFR+ on *tree* and return the average policy.

    Iteration ``t`` enters the average with weight ``max(t - delay, 0)``. With
    ``trembles > 0`` every role plays an epsilon-perturbed strategy during traversal, so
    every information set is reached and off-path play is optimized too (a
    perturbed-game approximation of sequential rationality); the returned average is
    unperturbed.
    """

    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not 0.0 <= trembles < 1.0:
        raise ValueError("trembles must lie in [0, 1)")
    run = _CFRPlus(tree, trembles)
    previous_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(previous_limit, 10_000))
    try:
        for t in range(1, iterations + 1):
            weight = float(max(t - delay, 0))
            for role in range(len(tree.roles)):
                run.walk(0, role, 1.0, 1.0, weight)
                run.update(role)
    finally:
        sys.setrecursionlimit(previous_limit)
    average = run.average()
    return CFRResult(average, iterations, nash_conv(tree, average).nash_conv)
