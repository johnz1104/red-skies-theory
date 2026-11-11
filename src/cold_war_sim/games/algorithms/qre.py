"""Logit quantal response equilibrium (agent form) on explicit game trees.

Players choose noisily: at every information set the probability of an action
is proportional to ``exp(value / temperature)``, where the value is the
action's expected return to the acting player given everyone's (noisy) play.
Small temperatures approach best responses; large ones approach uniform play.
Near-ties become genuine mixtures instead of all-or-nothing choices
(McKelvey and Palfrey 1995, 1998).

With perfect information one bottom-up pass is exact (logit backward
induction). Otherwise the fixed point is followed along the QRE path from a hot
start (near-uniform play) down to the target temperature, each step by damped
iteration: the action values at an information set are its nodes' child values
weighted by the counterfactual reach of each node (the probability of getting
there if the acting player had steered toward it).

Convergence is judged by the fixed-point **residual**, the largest gap between the
policy and the logit response to it, never by the size of a damped step: with the
damping at its floor a step is 200 times smaller than the gap (E52: Kuhn poker at
0.1 stopped with steps of 0.001 and a gap of 0.2).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ..traverse import CHANCE_NODE, DECISION, TERMINAL_NODE, GameTree, TreePolicy, node_values


@dataclass(frozen=True, slots=True)
class QREResult:
    policy: TreePolicy
    iterations: int
    residual: float  # the policy's largest gap to its own logit response (0 when exact)


def softmax(values: Sequence[float], temperature: float) -> list[float]:
    top = max(values)
    weights = [math.exp((v - top) / temperature) for v in values]
    total = sum(weights)
    return [w / total for w in weights]


def _logit_backward(tree: GameTree, temperature: float) -> TreePolicy:
    policy: TreePolicy = [[0.0] * len(info.actions) for info in tree.infosets]
    n_roles = len(tree.roles)
    values: list[tuple[float, ...]] = [()] * len(tree.nodes)
    for idx in range(len(tree.nodes) - 1, -1, -1):
        node = tree.nodes[idx]
        if node.kind == TERMINAL_NODE:
            values[idx] = node.returns
            continue
        if node.kind == CHANCE_NODE:
            weights: Sequence[float] = node.probs
        else:
            weights = softmax([values[c][node.player] for c in node.children], temperature)
            policy[node.infoset] = list(weights)
        acc = [0.0] * n_roles
        for weight, child in zip(weights, node.children, strict=True):
            for r in range(n_roles):
                acc[r] += weight * values[child][r]
        values[idx] = tuple(acc)
    return policy


def _counterfactual_reach(tree: GameTree, policy: TreePolicy, role: int) -> list[float]:
    reach = [0.0] * len(tree.nodes)
    reach[0] = 1.0
    for idx, node in enumerate(tree.nodes):
        if node.kind == TERMINAL_NODE or reach[idx] == 0.0:
            continue
        if node.kind == CHANCE_NODE:
            weights: Sequence[float] = node.probs
        elif node.player == role:
            weights = [1.0] * len(node.children)
        else:
            weights = policy[node.infoset]
        for weight, child in zip(weights, node.children, strict=True):
            reach[child] = reach[idx] * weight
    return reach


def _response(tree: GameTree, policy: TreePolicy, temperature: float) -> tuple[TreePolicy, float]:
    """The logit response to *policy* at every information set, and the residual: the
    largest gap between the two."""

    values = node_values(tree, policy)
    reaches = [_counterfactual_reach(tree, policy, r) for r in range(len(tree.roles))]
    residual = 0.0
    response: TreePolicy = []
    for index, info in enumerate(tree.infosets):
        reach = reaches[info.role]
        n_actions = len(info.actions)
        totals = [0.0] * n_actions
        weight = 0.0
        for node_idx in info.nodes:
            node = tree.nodes[node_idx]
            assert node.kind == DECISION
            w = reach[node_idx]
            weight += w
            for slot, child in enumerate(node.children):
                totals[slot] += w * values[child][info.role]
        if weight <= 0.0:  # unreachable even by own steering: weigh nodes equally
            totals = [0.0] * n_actions
            for node_idx in info.nodes:
                for slot, child in enumerate(tree.nodes[node_idx].children):
                    totals[slot] += values[child][info.role]
            weight = float(len(info.nodes))
        target = softmax([total / weight for total in totals], temperature)
        residual = max(
            residual, max(abs(t - o) for t, o in zip(target, policy[index], strict=True))
        )
        response.append(target)
    return response, residual


def fixed_point_residual(tree: GameTree, policy: TreePolicy, temperature: float) -> float:
    """How far *policy* is from a logit QRE at *temperature*: the largest gap between its
    probabilities and the logit response to it (0 at the equilibrium)."""

    return _response(tree, policy, temperature)[1]


def _iterate(
    tree: GameTree,
    policy: TreePolicy,
    temperature: float,
    iterations: int,
    damping: float,
    tolerance: float,
) -> tuple[TreePolicy, float, int]:
    """Damped fixed-point iteration at one temperature, until the residual falls below
    *tolerance* or *iterations* steps are used; the damping halves whenever the residual
    grows (an oscillation). Returns the policy with its own residual."""

    previous, done = math.inf, 0
    while True:
        response, residual = _response(tree, policy, temperature)
        if residual < tolerance or done >= iterations:
            return policy, residual, done
        if residual > previous:
            damping = max(damping * 0.5, 0.005)
        previous = residual
        policy = [
            [(1.0 - damping) * o + damping * n for o, n in zip(old, new, strict=True)]
            for old, new in zip(policy, response, strict=True)
        ]
        done += 1


def logit_qre(
    tree: GameTree,
    *,
    temperature: float,
    iterations: int = 300,
    damping: float = 0.5,
    tolerance: float = 1e-6,
    steps: int = 12,
    final_iterations: int = 5000,
    path_tolerance: float = 1e-3,
) -> QREResult:
    """Agent-form logit QRE of *tree* at *temperature* (in payoff units).

    Imperfect information: follow the QRE path from a hot start (near-uniform
    play, at the payoff range) down to *temperature* in *steps* geometric steps,
    each solved by damped iteration from the last: the path's steps only warm-start
    the next (at most *iterations* each, to *path_tolerance*), and the target
    temperature is solved to *tolerance* in at most *final_iterations*. The result's
    residual is the returned policy's own; callers judge convergence by it.
    """

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must lie in (0, 1]")
    if tree.is_perfect_information:
        return QREResult(_logit_backward(tree, temperature), 1, 0.0)
    payoffs = [r for n in tree.nodes if n.kind == TERMINAL_NODE for r in n.returns]
    span = max(payoffs) - min(payoffs) if payoffs else 1.0
    hot = max(temperature, span)
    count = max(1, steps) if hot > temperature else 1
    schedule = [hot * (temperature / hot) ** (k / max(1, count - 1)) for k in range(count)]
    schedule[-1] = temperature
    policy: TreePolicy = [[1.0 / len(info.actions)] * len(info.actions) for info in tree.infosets]
    residual, total = math.inf, 0
    for k, step_temperature in enumerate(schedule):
        final = k == len(schedule) - 1
        policy, residual, used = _iterate(
            tree,
            policy,
            step_temperature,
            final_iterations if final else iterations,
            damping,
            tolerance if final else max(tolerance, path_tolerance),
        )
        total += used
    return QREResult(policy, total, residual)


__all__ = ["QREResult", "fixed_point_residual", "logit_qre", "softmax"]
