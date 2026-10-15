"""Reference play for crises in which roles hold their own beliefs about types.

A role that holds its own prior about another's resolve or stakes plays the
game it believes it is in: :meth:`CrisisGame.view` replaces the chance draws it
holds beliefs about with its own priors. Each distinct view is solved once with
the given tree solver, and each role acts on its own view's solution, looked up
by its information state. Others' types never enter a role's information state,
so the keys a role meets in the true game are keys of its view.

The result is a subjective equilibrium, not an equilibrium of the true game:
each role best responds (or quantal responds) to how it expects the others to
play under its own beliefs. When every belief equals nature's prior, every view
is the true game and the policy is the solver's own solution.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from ..games.policy import TabularPolicy, tabular_from_tree
from ..games.protocol import GameState
from ..games.traverse import GameTree, TreePolicy, build_tree
from .crisis import CrisisGame


class CrisisBeliefPolicy:
    """Each role plays its part of the solution of its own view of the crisis."""

    def __init__(self, game: CrisisGame, solve: Callable[[GameTree], TreePolicy]) -> None:
        solved: dict[int, TabularPolicy] = {}
        self._policies: dict[str, TabularPolicy] = {}
        for role, view in game.views().items():
            key = id(view)  # roles with the same priors share one view object
            if key not in solved:
                tree = build_tree(cast(GameState, view.new_initial_state()))
                solved[key] = tabular_from_tree(tree, solve(tree))
            self._policies[role] = solved[key]

    def policy_of(self, role: str) -> TabularPolicy:
        """The tabular solution of *role*'s view."""

        return self._policies[role]

    def action_probabilities(self, state: GameState) -> dict[str, float]:
        return self._policies[str(state.current_player())].action_probabilities(state)
