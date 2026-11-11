"""Explicit game trees, and exact evaluation of policies on them.

A :class:`GameTree` is built once by depth-first expansion of a state. Node indices are
in preorder, so every child has a larger index than its parent: a pass in reverse order
is bottom-up, a pass in order is top-down.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

from .protocol import CHANCE, TERMINAL, GameState, Outcome

DECISION = 0
CHANCE_NODE = 1
TERMINAL_NODE = 2

PROBABILITY_TOLERANCE = 1e-9

# A behaviour strategy profile: one distribution per information set, over its actions.
TreePolicy = list[list[float]]


class TreeTooLargeError(RuntimeError):
    """Raised when expansion exceeds the configured node budget."""


class Node:
    """A history of the game.

    A decision node has a ``player`` (a role index) and an ``infoset``; a chance node has
    ``probs`` over its ``actions``; a terminal node has ``returns`` (one per role) and an
    ``outcome``. ``children`` are node indices, aligned with ``actions``.
    """

    def __init__(
        self,
        kind: int,
        *,
        player: int = -1,
        infoset: int = -1,
        actions: tuple[str, ...] = (),
        probs: tuple[float, ...] = (),
        returns: tuple[float, ...] = (),
        outcome: Outcome | None = None,
    ) -> None:
        self.kind = kind
        self.player = player
        self.infoset = infoset
        self.actions = actions
        self.children: tuple[int, ...] = ()
        self.probs = probs
        self.returns = returns
        self.outcome = outcome


class Infoset:
    """An information set of one role: the histories (``nodes``) it cannot tell apart."""

    def __init__(self, role: int, key: Hashable, actions: tuple[str, ...]) -> None:
        self.role = role
        self.key = key
        self.actions = actions
        self.nodes: list[int] = []


class GameTree:
    """A finite extensive-form game: its nodes in preorder and its information sets."""

    def __init__(
        self,
        roles: tuple[str, ...],
        nodes: list[Node],
        infosets: list[Infoset],
        infoset_index: dict[Hashable, int],
    ) -> None:
        self.roles = roles
        self.nodes = nodes
        self.infosets = infosets
        self.infoset_index = infoset_index  # information-state key -> index in infosets

    @property
    def is_perfect_information(self) -> bool:
        """Whether every information set is a single history."""

        return all(len(infoset.nodes) == 1 for infoset in self.infosets)

    def role_index(self, role: str) -> int:
        """The index of *role* in return vectors."""

        return self.roles.index(role)


def build_tree(root: GameState, *, max_nodes: int = 500_000) -> GameTree:
    """The game tree below *root* (which is left unchanged)."""

    roles = tuple(root.roles)
    nodes: list[Node] = []
    infosets: list[Infoset] = []
    infoset_index: dict[Hashable, int] = {}
    children: list[list[int]] = []  # by node, filled in as its children are numbered

    # iterative preorder expansion; a stack entry is (state, parent, slot in the parent)
    stack: list[tuple[GameState, int, int]] = [(root.clone(), -1, -1)]
    while stack:
        state, parent, slot = stack.pop()
        index = len(nodes)
        if index >= max_nodes:
            raise TreeTooLargeError(f"game tree exceeds {max_nodes} nodes")
        if parent >= 0:
            children[parent][slot] = index
        player = state.current_player()
        if player == TERMINAL:
            nodes.append(_terminal_node(state, roles))
            children.append([])
            continue
        if player == CHANCE:
            node = _chance_node(state)
        else:
            node = _decision_node(state, player, index, roles, infosets, infoset_index)
        nodes.append(node)
        children.append([-1] * len(node.actions))
        # push children in reverse so the first action is expanded first
        for slot_idx in range(len(node.actions) - 1, -1, -1):
            child = state.clone()
            child.apply_action(node.actions[slot_idx])
            stack.append((child, index, slot_idx))

    for node, node_children in zip(nodes, children, strict=True):
        node.children = tuple(node_children)
    return GameTree(roles, nodes, infosets, infoset_index)


def _terminal_node(state: GameState, roles: tuple[str, ...]) -> Node:
    returns = state.returns()
    return Node(
        TERMINAL_NODE,
        returns=tuple(float(returns[r]) for r in roles),
        outcome=state.outcome(),
    )


def _chance_node(state: GameState) -> Node:
    """A chance node; its outcomes must form a probability distribution."""

    outcomes = state.chance_outcomes()
    if not outcomes:
        raise ValueError("chance node without outcomes")
    total = sum(p for _, p in outcomes)
    if abs(total - 1.0) > PROBABILITY_TOLERANCE or any(p < 0 for _, p in outcomes):
        raise ValueError(f"chance probabilities must be a distribution, got {outcomes}")
    return Node(
        CHANCE_NODE,
        actions=tuple(a for a, _ in outcomes),
        probs=tuple(float(p) for _, p in outcomes),
    )


def _decision_node(
    state: GameState,
    player: str,
    index: int,
    roles: tuple[str, ...],
    infosets: list[Infoset],
    infoset_index: dict[Hashable, int],
) -> Node:
    """A decision node, filed under its information set (created on first sight).

    Every history in an information set must have the same mover and action set.
    """

    actions = tuple(state.legal_actions())
    if not actions:
        raise ValueError(f"decision node for {player!r} has no legal actions")
    key = state.information_state(player)
    role = roles.index(player)
    info_idx = infoset_index.get(key)
    if info_idx is None:
        info_idx = infoset_index[key] = len(infosets)
        infosets.append(Infoset(role, key, actions))
    else:
        infoset = infosets[info_idx]
        if infoset.role != role or infoset.actions != actions:
            raise ValueError(f"information set {key!r} is inconsistent across histories")
    infosets[info_idx].nodes.append(index)
    return Node(DECISION, player=role, infoset=info_idx, actions=actions)


def uniform_policy(tree: GameTree) -> TreePolicy:
    """The profile that mixes uniformly at every information set."""

    return [[1.0 / len(info.actions)] * len(info.actions) for info in tree.infosets]


def expectation(
    weights: Sequence[float], children: Sequence[int], values: Sequence[tuple[float, ...]]
) -> tuple[float, ...]:
    """The return vector sum_k weights[k] * values[children[k]]."""

    acc = [0.0] * len(values[children[0]])
    for weight, child in zip(weights, children, strict=True):
        for r, value in enumerate(values[child]):
            acc[r] += weight * value
    return tuple(acc)


def node_values(tree: GameTree, policy: TreePolicy) -> list[tuple[float, ...]]:
    """The expected return vector at every node under *policy* (one bottom-up pass)."""

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
    """Each role's expected return at the root under *policy*."""

    root = node_values(tree, policy)[0]
    return dict(zip(tree.roles, root, strict=True))


def reach_probabilities(tree: GameTree, policy: TreePolicy) -> list[float]:
    """The probability of reaching each node under *policy* and chance."""

    return _reach(tree, policy, role=None)


def counterfactual_reach(tree: GameTree, policy: TreePolicy, role: int) -> list[float]:
    """The probability of reaching each node if *role* steers toward it.

    The product of chance's and the other roles' probabilities along the path; *role*'s
    own choices count as 1.
    """

    return _reach(tree, policy, role)


def _reach(tree: GameTree, policy: TreePolicy, role: int | None) -> list[float]:
    """Reach probabilities top-down, with *role*'s own choices (if given) taken as certain."""

    reach = [0.0] * len(tree.nodes)
    reach[0] = 1.0
    for idx, node in enumerate(tree.nodes):
        if node.kind == TERMINAL_NODE or reach[idx] == 0.0:
            continue
        weights: Sequence[float]
        if node.kind == CHANCE_NODE:
            weights = node.probs
        elif node.player == role:
            weights = [1.0] * len(node.children)
        else:
            weights = policy[node.infoset]
        for weight, child in zip(weights, node.children, strict=True):
            reach[child] = reach[idx] * weight
    return reach


def outcome_distribution(tree: GameTree, policy: TreePolicy) -> dict[str, float]:
    """The probability of each terminal outcome label under *policy*."""

    reach = reach_probabilities(tree, policy)
    dist: dict[str, float] = {}
    for idx, node in enumerate(tree.nodes):
        if node.kind == TERMINAL_NODE and reach[idx] > 0.0 and node.outcome is not None:
            dist[node.outcome.label] = dist.get(node.outcome.label, 0.0) + reach[idx]
    return dict(sorted(dist.items()))
