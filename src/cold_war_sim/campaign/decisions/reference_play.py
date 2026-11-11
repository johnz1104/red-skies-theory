"""Reference strategies for event games, with cached solutions for repeated play."""

from __future__ import annotations

import warnings
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

from ...families import bargaining, build_family
from ...families.attrition import AttritionBeliefPolicy
from ...families.crisis import CrisisBeliefPolicy
from ...games import Policy, build_tree, tabular_from_tree
from ...games.solvers import backward_induction, cfr_plus, logit_qre
from ...games.tree import GameTree, TreePolicy
from ..engine.reproducibility import canonical_json

# engines and snapshots.
_BASELINES: OrderedDict[str, Policy] = OrderedDict()
_BASELINE_LIMIT = 512
# Attrition belief policies are lazy and their memos grow large, so only the latest few are kept;
# a running war uses its policy every quarter and stays among them
_BELIEF_POLICIES: OrderedDict[str, Policy] = OrderedDict()
_BELIEF_LIMIT = 4


QRE_TOLERANCE = 1e-3  # a reference QRE stopped further from its fixed point is warned about


def _solve_tree(tree: GameTree, iterations: int, temperature: float) -> TreePolicy:
    """Logit QRE at *temperature* > 0; otherwise backward induction with perfect
    information, or CFR+'s average strategy (an equilibrium only in two-player zero-sum
    games; in general-sum games it approaches a coarse correlated equilibrium)."""

    if temperature > 0.0:
        result = logit_qre(tree, temperature=temperature)
        if result.residual > QRE_TOLERANCE:  # reported, not fatal mid-campaign
            warnings.warn(
                f"reference QRE at temperature {temperature} stopped {result.residual:.1e} "
                f"from its fixed point ({len(tree.nodes)} nodes)",
                RuntimeWarning,
                stacklevel=2,
            )
        return result.policy
    if tree.is_perfect_information:
        return backward_induction(tree).policy
    return cfr_plus(tree, iterations=iterations, trembles=0.01).average_policy


def reference_tree_policy(
    game: Any, tree: GameTree, iterations: int, temperature: float = 0.0
) -> TreePolicy:
    """Reference play on *tree*, the true game's tree: a crisis whose roles hold their
    own beliefs plays each role's part of its own view; otherwise the solver's
    solution."""

    if not getattr(game, "has_beliefs", False):
        return _solve_tree(tree, iterations, temperature)
    belief = CrisisBeliefPolicy(game, lambda view: _solve_tree(view, iterations, temperature))
    result = []
    for info in tree.infosets:
        dist = belief.policy_of(tree.roles[info.role]).table.get(info.key)
        uniform = 1.0 / len(info.actions)
        result.append([dist.get(a, 0.0) if dist else uniform for a in info.actions])
    return result


def baseline_policy(
    family: str, config: Mapping[str, Any], iterations: int, temperature: float = 0.0
) -> Policy:
    """The computer-run reference for a sub-game: equilibrium play (see
    :func:`_solve_tree`), or with ``temperature > 0`` a logit quantal response
    equilibrium (the belief-based bargaining solver stays exact)."""

    key = canonical_json(
        {"family": family, "config": config, "iterations": iterations, "temperature": temperature}
    )
    cache, limit = (
        (_BELIEF_POLICIES, _BELIEF_LIMIT)
        if family == "attrition"
        else (_BASELINES, _BASELINE_LIMIT)
    )
    cached = cache.get(key)
    if cached is not None:
        cache.move_to_end(key)
        return cached
    game = build_family(family, config)
    policy: Policy
    if family == "bargaining":
        policy = bargaining.BargainingEquilibriumPolicy(game, temperature)
    elif family == "attrition":
        policy = AttritionBeliefPolicy(game, temperature)
    elif family == "crisis" and game.has_beliefs:  # each role solves its own view
        policy = CrisisBeliefPolicy(game, lambda tree: _solve_tree(tree, iterations, temperature))
    else:
        root = game.solver_state() if hasattr(game, "solver_state") else game.new_initial_state()
        tree = build_tree(root)
        policy = tabular_from_tree(tree, _solve_tree(tree, iterations, temperature))
    cache[key] = policy
    if len(cache) > limit:
        cache.popitem(last=False)
    return policy
