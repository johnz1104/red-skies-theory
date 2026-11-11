"""Exact best responses and NashConv on finite trees with imperfect information."""

from __future__ import annotations

import sys

from ...records import Record
from ..tree import (
    CHANCE_NODE,
    DECISION,
    TERMINAL_NODE,
    GameTree,
    TreePolicy,
    counterfactual_reach,
    node_values,
)

DEFAULT_TOLERANCE = 1e-12


class BestResponse(Record):
    """A role's best response to a profile: its value and its action at each information set."""

    def __init__(self, role: str, value: float, choices: dict[int, int]) -> None:
        self.role = role
        self.value = value
        self.choices = choices  # infoset index -> action index


class NashConvResult(Record):
    """NashConv of a profile: the sum over roles of the gain from a best response.

    It is zero exactly at a Nash equilibrium.
    """

    def __init__(
        self,
        nash_conv: float,
        gains: dict[str, float],
        on_policy: dict[str, float],
        best_response: dict[str, float],
    ) -> None:
        self.nash_conv = nash_conv
        self.gains = gains  # role -> best-response value minus on-policy value
        self.on_policy = on_policy
        self.best_response = best_response


class _BestResponder:
    """The best response of role ``r`` to ``policy``, by memoized recursion.

    At an information set the role maximizes the sum over its histories of
    counterfactual reach times the action's value. Values below the role's own later
    decisions use its best response there, which is well defined under perfect recall.
    """

    def __init__(self, tree: GameTree, policy: TreePolicy, r: int, tolerance: float) -> None:
        self.tree = tree
        self.policy = policy
        self.r = r
        self.tolerance = tolerance
        self.reach = counterfactual_reach(tree, policy, r)
        self.values: dict[int, float] = {}
        self.choices: dict[int, int] = {}

    def value(self, idx: int) -> float:
        """The role's value at node *idx* when it best responds from there on."""

        cached = self.values.get(idx)
        if cached is not None:
            return cached
        node = self.tree.nodes[idx]
        value = self.value
        if node.kind == TERMINAL_NODE:
            result = node.returns[self.r]
        elif node.kind == CHANCE_NODE:
            result = sum(p * value(c) for p, c in zip(node.probs, node.children, strict=True) if p)
        elif node.player != self.r:
            weights = self.policy[node.infoset]
            result = sum(p * value(c) for p, c in zip(weights, node.children, strict=True) if p)
        else:
            result = value(node.children[self.choose(node.infoset)])
        self.values[idx] = result
        return result

    def choose(self, info_idx: int) -> int:
        """The role's best action at information set *info_idx* (earliest on ties)."""

        cached = self.choices.get(info_idx)
        if cached is not None:
            return cached
        info = self.tree.infosets[info_idx]
        totals = [0.0] * len(info.actions)
        for node_idx in info.nodes:
            weight = self.reach[node_idx]
            if weight == 0.0:
                continue
            for slot, child in enumerate(self.tree.nodes[node_idx].children):
                totals[slot] += weight * self.value(child)
        best = 0
        for slot in range(1, len(totals)):
            if totals[slot] > totals[best] + self.tolerance:
                best = slot
        self.choices[info_idx] = best
        return best


def best_response(
    tree: GameTree, policy: TreePolicy, role: str, *, tolerance: float = DEFAULT_TOLERANCE
) -> BestResponse:
    """*role*'s exact best response to *policy*."""

    r = tree.role_index(role)
    responder = _BestResponder(tree, policy, r, tolerance)
    previous_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(previous_limit, 10_000))
    try:
        root_value = responder.value(0)
        for info_idx, info in enumerate(tree.infosets):
            if info.role == r and tree.nodes[info.nodes[0]].kind == DECISION:
                responder.choose(info_idx)
    finally:
        sys.setrecursionlimit(previous_limit)
    return BestResponse(role=role, value=root_value, choices=dict(responder.choices))


def nash_conv(tree: GameTree, policy: TreePolicy) -> NashConvResult:
    """NashConv of *policy*: the sum over roles of the gain from deviating to a best response."""

    on_policy = dict(zip(tree.roles, node_values(tree, policy)[0], strict=True))
    br_values = {role: best_response(tree, policy, role).value for role in tree.roles}
    gains = {role: max(0.0, br_values[role] - on_policy[role]) for role in tree.roles}
    return NashConvResult(
        nash_conv=sum(gains.values()),
        gains=gains,
        on_policy=on_policy,
        best_response=br_values,
    )
