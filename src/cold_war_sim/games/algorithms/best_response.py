"""Exact best responses and NashConv on finite trees with imperfect information.

For role *i* and a fixed policy of everyone else, the best response at an
information set maximizes the counterfactual-reach-weighted value over the
histories in that set. Values below a role's own later decisions use that
role's best response there too, which is well defined under perfect recall.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from ..traverse import CHANCE_NODE, DECISION, TERMINAL_NODE, GameTree, TreePolicy, node_values

DEFAULT_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class BestResponse:
    role: str
    value: float
    choices: dict[int, int]  # infoset index -> action index


@dataclass(frozen=True, slots=True)
class NashConvResult:
    nash_conv: float
    gains: dict[str, float]
    on_policy: dict[str, float]
    best_response: dict[str, float]


def _counterfactual_reach(tree: GameTree, policy: TreePolicy, role: int) -> list[float]:
    reach = [0.0] * len(tree.nodes)
    reach[0] = 1.0
    for idx, node in enumerate(tree.nodes):
        if node.kind == TERMINAL_NODE or reach[idx] == 0.0:
            continue
        if node.kind == CHANCE_NODE:
            weights: tuple[float, ...] | list[float] = node.probs
        elif node.player == role:
            weights = [1.0] * len(node.children)
        else:
            weights = policy[node.infoset]
        for weight, child in zip(weights, node.children, strict=True):
            reach[child] = reach[idx] * weight
    return reach


def best_response(
    tree: GameTree, policy: TreePolicy, role: str, *, tolerance: float = DEFAULT_TOLERANCE
) -> BestResponse:
    r = tree.role_index(role)
    reach = _counterfactual_reach(tree, policy, r)
    value_memo: dict[int, float] = {}
    choice_memo: dict[int, int] = {}

    previous_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(previous_limit, 10_000))

    def value(idx: int) -> float:
        cached = value_memo.get(idx)
        if cached is not None:
            return cached
        node = tree.nodes[idx]
        if node.kind == TERMINAL_NODE:
            result = node.returns[r]
        elif node.kind == CHANCE_NODE:
            result = sum(p * value(c) for p, c in zip(node.probs, node.children, strict=True) if p)
        elif node.player != r:
            result = sum(
                p * value(c) for p, c in zip(policy[node.infoset], node.children, strict=True) if p
            )
        else:
            result = value(node.children[choose(node.infoset)])
        value_memo[idx] = result
        return result

    def choose(info_idx: int) -> int:
        cached = choice_memo.get(info_idx)
        if cached is not None:
            return cached
        info = tree.infosets[info_idx]
        totals = [0.0] * len(info.actions)
        for node_idx in info.nodes:
            weight = reach[node_idx]
            if weight == 0.0:
                continue
            children = tree.nodes[node_idx].children
            for slot, child in enumerate(children):
                totals[slot] += weight * value(child)
        best = 0
        for slot in range(1, len(totals)):
            if totals[slot] > totals[best] + tolerance:
                best = slot
        choice_memo[info_idx] = best
        return best

    try:
        root_value = value(0)
        for info_idx, info in enumerate(tree.infosets):
            if info.role == r and tree.nodes[info.nodes[0]].kind == DECISION:
                choose(info_idx)
    finally:
        sys.setrecursionlimit(previous_limit)
    return BestResponse(role=role, value=root_value, choices=dict(choice_memo))


def nash_conv(tree: GameTree, policy: TreePolicy) -> NashConvResult:
    """Sum over roles of the gain from deviating to a best response."""

    on_policy = dict(zip(tree.roles, node_values(tree, policy)[0], strict=True))
    br_values = {role: best_response(tree, policy, role).value for role in tree.roles}
    gains = {role: max(0.0, br_values[role] - on_policy[role]) for role in tree.roles}
    return NashConvResult(
        nash_conv=sum(gains.values()),
        gains=gains,
        on_policy=on_policy,
        best_response=br_values,
    )
