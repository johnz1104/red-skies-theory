"""Backward induction for finite perfect-information games with N players."""

from __future__ import annotations

from dataclasses import dataclass

from ..traverse import CHANCE_NODE, TERMINAL_NODE, GameTree, TreePolicy

DEFAULT_TOLERANCE = 1e-10


class NotPerfectInformationError(ValueError):
    """Raised when an information set contains more than one history."""


@dataclass(frozen=True, slots=True)
class BackwardInductionResult:
    """A subgame-perfect pure policy and the value vector at each node.

    Ties are broken toward the earliest action in declared order, which makes
    the selected equilibrium deterministic and reproducible.
    """

    policy: TreePolicy
    values: dict[str, float]
    node_values: tuple[tuple[float, ...], ...]
    ties: int


def backward_induction(
    tree: GameTree, *, tolerance: float = DEFAULT_TOLERANCE
) -> BackwardInductionResult:
    if not tree.is_perfect_information:
        raise NotPerfectInformationError("backward induction requires perfect information")
    n_roles = len(tree.roles)
    values: list[tuple[float, ...]] = [()] * len(tree.nodes)
    policy: TreePolicy = [[0.0] * len(info.actions) for info in tree.infosets]
    ties = 0
    for idx in range(len(tree.nodes) - 1, -1, -1):
        node = tree.nodes[idx]
        if node.kind == TERMINAL_NODE:
            values[idx] = node.returns
            continue
        if node.kind == CHANCE_NODE:
            acc = [0.0] * n_roles
            for prob, child in zip(node.probs, node.children, strict=True):
                for r in range(n_roles):
                    acc[r] += prob * values[child][r]
            values[idx] = tuple(acc)
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
