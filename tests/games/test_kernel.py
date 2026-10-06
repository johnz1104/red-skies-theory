from __future__ import annotations

import pickle

import numpy as np
import pytest

from cold_war_sim.games import (
    IllegalActionError,
    TabularPolicy,
    build_tree,
    expected_returns,
    outcome_distribution,
    sample_action,
    tabular_from_tree,
)
from cold_war_sim.games.algorithms import (
    NotPerfectInformationError,
    backward_induction,
    best_response,
    cfr_plus,
    nash_conv,
)
from cold_war_sim.games.traverse import TreeTooLargeError, uniform_policy

from .fixtures import KuhnState, MatchingPenniesState, UltimatumState


def test_kuhn_tree_shape() -> None:
    tree = build_tree(KuhnState())
    terminals = [n for n in tree.nodes if n.kind == 2]
    assert len(terminals) == 30  # 6 deals x 5 betting sequences
    assert len(tree.infosets) == 12  # 3 cards x 4 decision points
    assert not tree.is_perfect_information


def test_kuhn_cfr_plus_converges_to_known_value() -> None:
    tree = build_tree(KuhnState())
    result = cfr_plus(tree, iterations=1500)
    values = expected_returns(tree, result.average_policy)
    assert values["p0"] == pytest.approx(-1.0 / 18.0, abs=1e-3)
    assert result.nash_conv < 1e-3


def test_uniform_kuhn_is_exploitable() -> None:
    tree = build_tree(KuhnState())
    assert nash_conv(tree, uniform_policy(tree)).nash_conv > 0.5


def test_matching_pennies_equilibrium_is_uniform() -> None:
    tree = build_tree(MatchingPenniesState())
    assert len(tree.infosets) == 2  # p1 cannot see p0's coin
    result = cfr_plus(tree, iterations=500)
    for vector in result.average_policy:
        assert vector == pytest.approx([0.5, 0.5], abs=1e-2)
    assert nash_conv(tree, uniform_policy(tree)).nash_conv == pytest.approx(0.0, abs=1e-12)


def test_ultimatum_backward_induction_breaks_ties_in_declared_order() -> None:
    tree = build_tree(UltimatumState())
    assert tree.is_perfect_information
    result = backward_induction(tree)
    assert result.values == {"proposer": 3.0, "responder": 0.0}
    # Responder is indifferent at offer0 and accepts (declared first).
    assert outcome_distribution(tree, result.policy) == {"agreement": 1.0}
    assert nash_conv(tree, result.policy).nash_conv == pytest.approx(0.0)


def test_backward_induction_rejects_imperfect_information() -> None:
    with pytest.raises(NotPerfectInformationError):
        backward_induction(build_tree(MatchingPenniesState()))


def test_best_response_against_always_bet() -> None:
    tree = build_tree(KuhnState())
    policy = uniform_policy(tree)
    br = best_response(tree, policy, "p1")
    assert br.value > expected_returns(tree, policy)["p1"]


def test_states_clone_and_pickle() -> None:
    state = KuhnState()
    state.apply_action("K")
    clone = state.clone()
    clone.apply_action("J")
    assert state.history() == ("K",)
    restored = pickle.loads(pickle.dumps(clone))
    assert restored.history() == ("K", "J")
    assert restored.information_state("p0") == clone.information_state("p0")


def test_illegal_action_raises() -> None:
    state = UltimatumState()
    with pytest.raises(IllegalActionError):
        state.apply_action("accept")


def test_tabular_round_trip_and_sampling() -> None:
    tree = build_tree(UltimatumState())
    tabular = tabular_from_tree(tree, backward_induction(tree).policy)
    assert isinstance(tabular, TabularPolicy)
    state = UltimatumState()
    dist = tabular.action_probabilities(state)
    assert dist["offer0"] == 1.0
    rng = np.random.default_rng(0)
    assert sample_action(dist, rng) == "offer0"


def test_tree_budget() -> None:
    with pytest.raises(TreeTooLargeError):
        build_tree(KuhnState(), max_nodes=10)
