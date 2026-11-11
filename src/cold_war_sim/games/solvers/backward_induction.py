"""Backward induction for finite perfect-information games with N players."""

from __future__ import annotations

from ...records import Record
from ..tree import CHANCE_NODE, TERMINAL_NODE, GameTree, TreePolicy, expectation

DEFAULT_TOLERANCE = 1e-10


class NotPerfectInformationError(ValueError):
    """Raised when an information set contains more than one history."""


class BackwardInductionResult(Record):
    """A subgame-perfect pure profile with the value vector at every node.

    Ties go to the earliest action in declared order, so the selected equilibrium is
    deterministic; ``ties`` counts the comparisons that tied.
    """

    def __init__(
        self,
        policy: TreePolicy,
        values: dict[str, float],
        node_values: tuple[tuple[float, ...], ...],
        ties: int,
    ) -> None:
        self.policy = policy
        self.values = values  # each role's value at the root
        self.node_values = node_values
        self.ties = ties


def backward_induction(
    tree: GameTree, *, tolerance: float = DEFAULT_TOLERANCE
) -> BackwardInductionResult:
    """The subgame-perfect equilibrium of *tree*: each mover takes its best child, bottom-up."""

    if not tree.is_perfect_information:
        raise NotPerfectInformationError("backward induction requires perfect information")
    values: list[tuple[float, ...]] = [()] * len(tree.nodes)
    policy: TreePolicy = [[0.0] * len(info.actions) for info in tree.infosets]
    ties = 0
    for idx in range(len(tree.nodes) - 1, -1, -1):
        node = tree.nodes[idx]
        if node.kind == TERMINAL_NODE:
            values[idx] = node.returns
            continue
        if node.kind == CHANCE_NODE:
            values[idx] = expectation(node.probs, node.children, values)
            continue
        best_slot = 0
        best_value = values[node.children[0]][node.player]
        for slot in range(1, len(node.children)):
            candidate = values[node.children[slot]][node.player]
            if candidate > best_value + tolerance:
                best_slot, best_value = slot, candidate
            elif abs(candidate - best_value) <= tolerance:
                ties += 1
        policy[node.infoset][best_slot] = 1.0
        values[idx] = values[node.children[best_slot]]
    return BackwardInductionResult(
        policy=policy,
        values=dict(zip(tree.roles, values[0], strict=True)),
        node_values=tuple(values),
        ties=ties,
    )
