"""Explicit game trees built from a state machine, and exact evaluation on them.

Tree algorithms (backward induction, best responses, CFR+) run on a compact
:class:`GameTree` rather than on live states. The tree is built once by depth-
first expansion; node indices are in preorder, so every child has a larger
index than its parent and bottom-up passes can iterate in reverse.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field

from .protocol import CHANCE, TERMINAL, GameState, Outcome

DECISION = 0
CHANCE_NODE = 1
TERMINAL_NODE = 2

PROBABILITY_TOLERANCE = 1e-9

# One probability vector per information set, aligned with ``Infoset.actions``.
TreePolicy = list[list[float]]


class TreeTooLargeError(RuntimeError):
    """Raised when expansion exceeds the configured node budget."""


@dataclass(slots=True)
class Node:
    kind: int
    player: int
    infoset: int
    actions: tuple[str, ...]
    children: tuple[int, ...]
    probs: tuple[float, ...]
    returns: tuple[float, ...]
    outcome: Outcome | None


@dataclass(slots=True)
class Infoset:
    role: int
    key: Hashable
    actions: tuple[str, ...]
    nodes: list[int] = field(default_factory=list)


@dataclass(slots=True)
class GameTree:
    roles: tuple[str, ...]
    nodes: list[Node]
    infosets: list[Infoset]
    infoset_index: dict[Hashable, int]

    @property
    def is_perfect_information(self) -> bool:
        return all(len(infoset.nodes) == 1 for infoset in self.infosets)

    def role_index(self, role: str) -> int:
        return self.roles.index(role)


def build_tree(root: GameState, *, max_nodes: int = 500_000) -> GameTree:
    """Expand *root* into a :class:`GameTree` (the root state is not modified)."""

    roles = tuple(root.roles)
    nodes: list[Node] = []
    infosets: list[Infoset] = []
    infoset_index: dict[Hashable, int] = {}

    # Iterative preorder expansion. Each stack entry is (state, parent, slot).
    placeholder: list[list[int]] = []
    stack: list[tuple[GameState, int, int]] = [(root.clone(), -1, -1)]
    while stack:
        state, parent, slot = stack.pop()
        index = len(nodes)
        if index >= max_nodes:
            raise TreeTooLargeError(f"game tree exceeds {max_nodes} nodes")
        if parent >= 0:
            placeholder[parent][slot] = index
        player = state.current_player()
        if player == TERMINAL:
            returns = state.returns()
            nodes.append(
                Node(
                    TERMINAL_NODE,
                    -1,
                    -1,
                    (),
                    (),
                    (),
                    tuple(float(returns[r]) for r in roles),
                    state.outcome(),
                )
            )
            placeholder.append([])
            continue
        if player == CHANCE:
            outcomes = state.chance_outcomes()
            if not outcomes:
                raise ValueError("chance node without outcomes")
            total = sum(p for _, p in outcomes)
            if abs(total - 1.0) > PROBABILITY_TOLERANCE or any(p < 0 for _, p in outcomes):
                raise ValueError(f"chance probabilities must be a distribution, got {outcomes}")
            actions = tuple(a for a, _ in outcomes)
            probs = tuple(float(p) for _, p in outcomes)
            nodes.append(Node(CHANCE_NODE, -1, -1, actions, (), probs, (), None))
        else:
            actions = tuple(state.legal_actions())
            if not actions:
                raise ValueError(f"decision node for {player!r} has no legal actions")
            key = state.information_state(player)
            role_idx = roles.index(player)
            info_idx = infoset_index.get(key)
            if info_idx is None:
                info_idx = len(infosets)
                infoset_index[key] = info_idx
                infosets.append(Infoset(role_idx, key, actions))
            else:
                infoset = infosets[info_idx]
                if infoset.role != role_idx or infoset.actions != actions:
                    raise ValueError(f"information set {key!r} is inconsistent across histories")
            infosets[info_idx].nodes.append(index)
            nodes.append(Node(DECISION, role_idx, info_idx, actions, (), (), (), None))
        placeholder.append([-1] * len(actions))
        # Push children in reverse so the first action is expanded first.
        for slot_idx in range(len(actions) - 1, -1, -1):
            child = state.clone()
            child.apply_action(actions[slot_idx])
            stack.append((child, index, slot_idx))

    for idx, node in enumerate(nodes):
        node.children = tuple(placeholder[idx])
    return GameTree(roles, nodes, infosets, infoset_index)


def uniform_policy(tree: GameTree) -> TreePolicy:
    return [[1.0 / len(info.actions)] * len(info.actions) for info in tree.infosets]


def node_values(tree: GameTree, policy: TreePolicy) -> list[tuple[float, ...]]:
    """Expected return vector at every node under *policy* (bottom-up)."""

    n_roles = len(tree.roles)
    values: list[tuple[float, ...]] = [()] * len(tree.nodes)
    for idx in range(len(tree.nodes) - 1, -1, -1):
        node = tree.nodes[idx]
        if node.kind == TERMINAL_NODE:
            values[idx] = node.returns
            continue
        weights = node.probs if node.kind == CHANCE_NODE else policy[node.infoset]
        acc = [0.0] * n_roles
        for weight, child in zip(weights, node.children, strict=True):
            if weight == 0.0:
                continue
            child_value = values[child]
            for r in range(n_roles):
                acc[r] += weight * child_value[r]
        values[idx] = tuple(acc)
    return values


def expected_returns(tree: GameTree, policy: TreePolicy) -> dict[str, float]:
    root = node_values(tree, policy)[0]
    return dict(zip(tree.roles, root, strict=True))


def reach_probabilities(tree: GameTree, policy: TreePolicy) -> list[float]:
    """Probability of reaching each node under *policy* and chance."""

    reach = [0.0] * len(tree.nodes)
    reach[0] = 1.0
    for idx, node in enumerate(tree.nodes):
        if node.kind == TERMINAL_NODE or reach[idx] == 0.0:
            continue
        weights = node.probs if node.kind == CHANCE_NODE else policy[node.infoset]
        for weight, child in zip(weights, node.children, strict=True):
            reach[child] = reach[idx] * weight
    return reach


def outcome_distribution(tree: GameTree, policy: TreePolicy) -> dict[str, float]:
    """Probability of each terminal outcome label under *policy*."""

    reach = reach_probabilities(tree, policy)
    dist: dict[str, float] = {}
    for idx, node in enumerate(tree.nodes):
        if node.kind == TERMINAL_NODE and reach[idx] > 0.0 and node.outcome is not None:
            dist[node.outcome.label] = dist.get(node.outcome.label, 0.0) + reach[idx]
    return dict(sorted(dist.items()))
