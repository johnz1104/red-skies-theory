"""Logit quantal response equilibrium on explicit game trees.

At each information set, actions have probabilities proportional to
``exp(value / temperature)``. Value is the expected payoff given everyone's
current play. Low temperatures favor the best actions; high temperatures
make choices more uniform (McKelvey and Palfrey 1995, 1998).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ...records import Record
from ..tree import (
    DECISION,
    TERMINAL_NODE,
    GameTree,
    TreePolicy,
    counterfactual_reach,
    expectation,
    node_values,
    uniform_policy,
)


class QREResult(Record):
    """A profile with its distance from the logit fixed point.

    ``residual`` is the largest gap between the profile and the logit response to it (0
    when the solution is exact).
    """

    def __init__(self, policy: TreePolicy, iterations: int, residual: float) -> None:
        self.policy = policy
        self.iterations = iterations
        self.residual = residual


def softmax(values: Sequence[float], temperature: float) -> list[float]:
    """The logit choice probabilities exp(v / temperature), normalized."""

    top = max(values)
    weights = [math.exp((v - top) / temperature) for v in values]
    total = sum(weights)
    return [w / total for w in weights]


def _logit_backward(tree: GameTree, temperature: float) -> TreePolicy:
    """The exact QRE of a perfect-information tree: logit backward induction."""

    policy: TreePolicy = [[0.0] * len(info.actions) for info in tree.infosets]
    values: list[tuple[float, ...]] = [()] * len(tree.nodes)
    for idx in range(len(tree.nodes) - 1, -1, -1):
        node = tree.nodes[idx]
        if node.kind == TERMINAL_NODE:
            values[idx] = node.returns
            continue
        weights: Sequence[float] = node.probs
        if node.kind == DECISION:
            weights = softmax([values[c][node.player] for c in node.children], temperature)
            policy[node.infoset] = list(weights)
        values[idx] = expectation(weights, node.children, values)
    return policy


def _response(tree: GameTree, policy: TreePolicy, temperature: float) -> tuple[TreePolicy, float]:
    """The logit response to *policy* at every information set, and the residual.

    An action's value at an information set is its nodes' child values weighted by each
    node's counterfactual reach. The residual is the largest gap between the response
    and *policy*.
    """

    values = node_values(tree, policy)
    reaches = [counterfactual_reach(tree, policy, r) for r in range(len(tree.roles))]
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
    """Iterate toward a logit equilibrium at one temperature.

    Stop when ``max |pi - logit_response(pi)| < tolerance`` or the iteration
    limit is reached. Here pi is the policy and the response is its implied
    logit choice probabilities. Measure this gap before damping: a small update
    can occur while the policy remains far from equilibrium. Halve the damping
    when the gap grows, down to 0.005. Return the policy and its own residual.
    """

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


def _cooling_schedule(tree: GameTree, temperature: float, steps: int) -> list[float]:
    """Temperatures from the payoff range (near-uniform play) down to *temperature*,
    geometrically spaced."""

    payoffs = [r for n in tree.nodes if n.kind == TERMINAL_NODE for r in n.returns]
    span = max(payoffs) - min(payoffs) if payoffs else 1.0
    hot = max(temperature, span)
    count = max(1, steps) if hot > temperature else 1
    schedule = [hot * (temperature / hot) ** (k / max(1, count - 1)) for k in range(count)]
    schedule[-1] = temperature
    return schedule


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

    With perfect information one bottom-up pass is exact. Otherwise the fixed point is
    followed along the QRE path, cooling in *steps* geometric steps: each step warm-starts
    the next (at most *iterations* each, to *path_tolerance*), and the target temperature
    is solved to *tolerance* in at most *final_iterations*. The result's residual is the
    returned policy's own; callers judge convergence by it.
    """

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must lie in (0, 1]")
    if tree.is_perfect_information:
        return QREResult(_logit_backward(tree, temperature), 1, 0.0)
    schedule = _cooling_schedule(tree, temperature, steps)
    policy = uniform_policy(tree)
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
