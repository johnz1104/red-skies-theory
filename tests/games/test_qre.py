from __future__ import annotations

import math

import pytest

from cold_war_sim.games import build_tree
from cold_war_sim.games.algorithms import logit_qre, nash_conv
from cold_war_sim.games.algorithms.qre import _counterfactual_reach, softmax
from cold_war_sim.games.traverse import node_values

from .fixtures import KuhnState, MatchingPenniesState, UltimatumState


def test_perfect_information_is_logit_backward_induction() -> None:
    tree = build_tree(UltimatumState())
    tau = 0.5
    result = logit_qre(tree, temperature=tau)
    assert result.residual == 0.0
    accept = {k: 1.0 / (1.0 + math.exp(-k / tau)) for k in range(4)}  # accept k vs reject 0
    expected = softmax([accept[k] * (3 - k) for k in range(4)], tau)
    root = tree.nodes[0]
    assert result.policy[root.infoset] == pytest.approx(expected)
    for slot, child in enumerate(root.children):
        responder = tree.nodes[child]
        assert result.policy[responder.infoset][0] == pytest.approx(accept[slot])


def test_matching_pennies_quantal_equilibrium_is_uniform() -> None:
    result = logit_qre(build_tree(MatchingPenniesState()), temperature=0.3)
    for probs in result.policy:
        assert probs == pytest.approx([0.5, 0.5], abs=1e-6)


def test_kuhn_quantal_equilibrium_is_a_logit_fixed_point() -> None:
    tree = build_tree(KuhnState())
    tau = 0.2
    result = logit_qre(tree, temperature=tau, iterations=2000, tolerance=1e-9)
    assert result.residual < 1e-7
    values = node_values(tree, result.policy)
    for index, info in enumerate(tree.infosets):
        reach = _counterfactual_reach(tree, result.policy, info.role)
        totals = [0.0] * len(info.actions)
        weight = sum(reach[n] for n in info.nodes)
        for n in info.nodes:
            for slot, child in enumerate(tree.nodes[n].children):
                totals[slot] += reach[n] * values[child][info.role]
        target = softmax([t / weight for t in totals], tau)
        assert result.policy[index] == pytest.approx(target, abs=1e-5)


def test_temperature_moves_between_best_response_and_uniform() -> None:
    tree = build_tree(KuhnState())
    cold = logit_qre(tree, temperature=0.01, iterations=3000)
    assert nash_conv(tree, cold.policy).nash_conv < 0.02
    hot = logit_qre(tree, temperature=100.0)
    for probs in hot.policy:
        assert probs == pytest.approx([1.0 / len(probs)] * len(probs), abs=0.01)
    for probs in cold.policy + hot.policy:
        assert sum(probs) == pytest.approx(1.0)


def test_temperature_must_be_positive() -> None:
    with pytest.raises(ValueError):
        logit_qre(build_tree(UltimatumState()), temperature=0.0)
