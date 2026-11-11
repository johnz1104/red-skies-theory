"""Logit (quantal response) reference play in the attrition and bargaining families."""

from __future__ import annotations

from typing import Any

import pytest

from cold_war_sim.families import ConfigError, attrition, bargaining
from cold_war_sim.games import CHANCE, build_tree
from cold_war_sim.games.algorithms import logit_qre

from . import configs


def _decision_states(game: Any, limit: int = 40) -> list[Any]:
    """Decision states reached by following the first legal action or chance outcome."""

    states, state = [], game.new_initial_state()
    while not state.is_terminal() and len(states) < limit:
        if state.current_player() == CHANCE:
            state.apply_action(state.chance_outcomes()[0][0])
            continue
        states.append(state.clone())
        state.apply_action(state.legal_actions()[0])
    return states


def test_attrition_temperature_zero_is_the_argmax() -> None:
    game = attrition.build(configs.attrition_config())
    exact = attrition.AttritionLookaheadPolicy(game)
    noisy = attrition.AttritionLookaheadPolicy(game, temperature=0.1)
    for state in _decision_states(game):
        dist = exact.action_probabilities(state)
        assert sorted(dist.values())[-1] == 1.0
        soft = noisy.action_probabilities(state)
        assert sum(soft.values()) == pytest.approx(1.0)
        assert max(soft, key=lambda a: soft[a]) == max(dist, key=lambda a: dist[a])
    with pytest.raises(ConfigError):
        attrition.AttritionLookaheadPolicy(game, temperature=-1.0)


@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_bargaining_logit_approaches_the_exact_solution(horizon: int) -> None:
    game = bargaining.build(configs.bargaining_config(horizon=horizon))
    exact = bargaining.BargainingEquilibriumPolicy(game)
    cold = bargaining.BargainingEquilibriumPolicy(game, temperature=1e-4)
    noisy = bargaining.BargainingEquilibriumPolicy(game, temperature=0.3)
    for state in _decision_states(game):
        dist = exact.action_probabilities(state)
        chosen = max(dist, key=lambda a: dist[a])
        assert cold.action_probabilities(state)[chosen] == pytest.approx(1.0, abs=1e-3)
        soft = noisy.action_probabilities(state)
        assert sum(soft.values()) == pytest.approx(1.0)
        assert min(soft.values()) > 0.0  # every choice keeps some probability


def test_bargaining_belief_solver_ignores_the_temperature() -> None:
    config = configs.bargaining_config(horizon=2, commitments=False, consent=False)
    config["proposers"] = "A"
    config["types"] = {"B": {"prior": {"strong": 0.5}, "delay_cost": {"strong": 0.05, "weak": 0.5}}}
    game = bargaining.build(config)
    policy = bargaining.BargainingEquilibriumPolicy(game, temperature=0.3)
    assert policy.beliefs is not None and policy.logit is None


# ratifiers that disagree about the share: the first's approval counts only if the later
# ones approve too, which they often do not (E52); the third pays for its own veto
RATIFIERS = [
    {"role": "C1", "base": -0.2, "per_share": 1.0, "veto_value": 0.0},
    {"role": "C2", "base": 0.3, "per_share": -0.6, "veto_value": 0.0},
    {"role": "C3", "base": 0.1, "per_share": 0.0, "veto_value": 0.2, "veto_cost": 0.4},
]


@pytest.mark.parametrize("ratifiers", [1, 2, 3])
def test_bargaining_logit_matches_the_generic_tree_qre(ratifiers: int) -> None:
    config = configs.bargaining_config(horizon=2)
    config["consent"] = RATIFIERS[:ratifiers]
    game = bargaining.build(config)
    tau = 0.3
    tree = build_tree(game.new_initial_state())
    assert tree.is_perfect_information
    generic = logit_qre(tree, temperature=tau).policy  # logit backward induction
    special = bargaining.BargainingEquilibriumPolicy(game, temperature=tau)
    stack, checked = [game.new_initial_state()], 0
    while stack:
        state = stack.pop()
        player = state.current_player()
        if state.is_terminal():
            continue
        if player == CHANCE:
            actions = [a for a, p in state.chance_outcomes() if p > 0.0]
        else:
            actions = list(state.legal_actions())
            index = tree.infoset_index[state.information_state(player)]
            dist = special.action_probabilities(state)
            expected = dict(zip(tree.infosets[index].actions, generic[index], strict=True))
            assert dist == pytest.approx(expected, abs=1e-9)
            checked += 1
        for action in actions:
            child = state.clone()
            child.apply_action(action)
            stack.append(child)
    assert checked == len(tree.infosets)
