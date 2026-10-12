"""Bargaining with a contest and hidden resolve, solved by the belief-based solver."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from cold_war_sim.families import ConfigError, alignment, bargaining
from cold_war_sim.families.bargaining_beliefs import BeliefPolicy
from cold_war_sim.games import CHANCE, TERMINAL, build_tree, expected_returns
from cold_war_sim.games.algorithms import cfr_plus, nash_conv
from cold_war_sim.games.policy import tabular_from_tree
from cold_war_sim.games.traverse import reach_probabilities

from . import configs


def expected_values(game: Any, policy: Any) -> dict[str, float]:
    """Exact expected returns of *policy* (mixed actions included) by recursion."""

    def value(state: Any) -> dict[str, float]:
        if state.is_terminal():
            return state.returns()
        if state.current_player() == CHANCE:
            branches = list(state.chance_outcomes())
        else:
            branches = [(a, p) for a, p in policy.action_probabilities(state).items() if p > 0]
        total: dict[str, float] = {}
        for action, prob in branches:
            child = state.clone()
            child.apply_action(action)
            for role, v in value(child).items():
                total[role] = total.get(role, 0.0) + prob * v
        return total

    return value(game.new_initial_state())


def tree_policy(game: Any, policy: Any) -> tuple[Any, list[list[float]]]:
    """The explicit tree and *policy* as a tree policy (checked to respect infosets)."""

    tree = build_tree(game.new_initial_state())
    table: dict[Any, dict[str, float]] = {}
    stack = [game.new_initial_state()]
    while stack:
        state = stack.pop()
        player = state.current_player()
        if player == TERMINAL:
            continue
        if player == CHANCE:
            actions = [a for a, _ in state.chance_outcomes()]
        else:
            actions = list(state.legal_actions())
            key = state.information_state(player)
            dist = policy.action_probabilities(state)
            if key in table:
                assert all(table[key][a] == pytest.approx(dist[a]) for a in actions)
            table[key] = dist
        for action in actions:
            child = state.clone()
            child.apply_action(action)
            stack.append(child)
    return tree, [[table[info.key][a] for a in info.actions] for info in tree.infosets]


def screening_config(prior: float, horizon: int = 1) -> dict[str, Any]:
    """A proposes; B's cost of waiting is low (strong) or high (weak)."""

    return {
        "roles": {"A": "P", "B": "R"},
        "grid": [0.0, 0.5, 1.0],
        "horizon": horizon,
        "proposers": "A",
        "sides": {
            "A": {"breakdown_value": 0.0, "impasse_value": 0.0, "agreement": {"per_share": 1.0}},
            "B": {"breakdown_value": 0.0, "impasse_value": 0.6, "agreement": {"per_share": 1.0}},
        },
        "types": {"B": {"prior": {"strong": prior}, "delay_cost": {"strong": 0.1, "weak": 0.6}}},
    }


# a two-sided case with signalling, screening and a jump in the responder's behaviour
TWO_SIDED = {
    "roles": {"A": "W", "B": "S"},
    "grid": [0, 0.5, 1.0],
    "horizon": 3,
    "proposers": "A",
    "sides": {
        "A": {
            "breakdown_value": -2.2130599786744516,
            "impasse_value": 0.625547008945079,
            "agreement": {"base": -1.07938686302094, "per_share": 1.9703819488632701},
        },
        "B": {
            "breakdown_value": -0.057788400596284095,
            "impasse_value": 0.94248579049568,
            "agreement": {"base": -0.41281508883969953, "per_share": 2.0824537110948684},
        },
    },
    "breakdown": {"base": 0.027689120404537082, "growth": 0.016065200877512686},
    "types": {
        "A": {
            "prior": {"strong": 0.5096411513287273},
            "delay_cost": {"strong": 0.023173122494154066, "weak": 0.7364428288762502},
        },
        "B": {
            "prior": {"strong": 0.6660098686053788},
            "delay_cost": {"strong": 0.1226006602106081, "weak": 0.9421083933536318},
        },
    },
    "contest": {"initial": 0.5, "step": 0.25, "win_probability": 0.22375572599852173},
}

# one-sided: the weak responder mixes after the first offer
MIXING = {
    "roles": {"A": "W", "B": "S"},
    "grid": [0, 0.5, 1.0],
    "horizon": 2,
    "proposers": "A",
    "sides": {
        "A": {
            "breakdown_value": -2.9587864241505644,
            "impasse_value": 0.1508368365248971,
            "agreement": {"base": -0.3567042422111706, "per_share": 2.228314547485776},
            "commitment": 0.5,
            "commitment_cost": 0.15229496284881394,
        },
        "B": {
            "breakdown_value": -2.027560872577872,
            "impasse_value": 0.5758614089432399,
            "agreement": {"base": -0.77322805033519, "per_share": 2.9990027045140537},
        },
    },
    "breakdown": {"base": 0.07760316524447806, "growth": 0.08306314412717591},
    "types": {
        "B": {
            "prior": {"strong": 0.31958234546420006},
            "delay_cost": {"strong": 0.08645299286925322, "weak": 0.6585043837346303},
        }
    },
    "contest": {"initial": 0.5, "step": 0.25, "win_probability": 0.3167656086393407},
}


@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_plain_configs_solve_alike_with_the_belief_solver(horizon: int) -> None:
    game = bargaining.build(configs.bargaining_config(horizon=horizon))
    assert not game.params.extended
    old = expected_values(game, bargaining.BargainingEquilibriumPolicy(game))
    new = expected_values(game, BeliefPolicy(game.params))
    assert new == pytest.approx(old)


@pytest.mark.parametrize(("prior", "kind"), [(1.0, "strong"), (0.0, "weak")])
def test_certain_types_match_complete_information(prior: float, kind: str) -> None:
    typed = copy.deepcopy(MIXING)
    typed["types"]["B"]["prior"] = {"strong": prior}
    known = copy.deepcopy(MIXING)
    del known["types"]
    known["sides"]["B"]["delay_cost"] = MIXING["types"]["B"]["delay_cost"][kind]
    known["proposers"] = "A"
    game_typed, game_known = bargaining.build(typed), bargaining.build(known)
    values_typed = expected_values(game_typed, bargaining.BargainingEquilibriumPolicy(game_typed))
    values_known = expected_values(game_known, bargaining.BargainingEquilibriumPolicy(game_known))
    assert values_typed == pytest.approx(values_known, abs=1e-9)


@pytest.mark.parametrize(("prior", "offer"), [(0.3, "offer_100"), (0.7, "offer_050")])
def test_one_period_screening_switches_at_the_cutoff(prior: float, offer: str) -> None:
    # The strong responder rejects a full demand (it keeps 0.6 - 0.1 = 0.5 by waiting), the weak
    # one accepts anything (0.6 - 0.6 = 0). Demanding 1 earns 1 - mu, offering 0.5 earns 0.5:
    # the proposer screens when mu < 0.5.
    game = bargaining.build(screening_config(prior))
    policy = bargaining.BargainingEquilibriumPolicy(game)
    state = game.new_initial_state()
    state.apply_action("strong")
    dist = policy.action_probabilities(state)
    assert dist[offer] == pytest.approx(1.0)


def test_screening_delays_agreement_with_positive_probability() -> None:
    game = bargaining.build(screening_config(0.3, horizon=3))
    tree, policy = tree_policy(game, bargaining.BargainingEquilibriumPolicy(game))
    rejected = 0.0
    reach = reach_probabilities(tree, policy)
    for index, node in enumerate(tree.nodes):
        if node.kind == 0 and "reject" in node.actions:
            rejected += reach[index] * policy[node.infoset][node.actions.index("reject")]
    assert rejected > 0.1


@pytest.mark.parametrize("config", [TWO_SIDED, MIXING], ids=["two_sided", "mixing"])
def test_belief_solution_is_an_equilibrium(config: dict[str, Any]) -> None:
    game = bargaining.build(config)
    tree, policy = tree_policy(game, bargaining.BargainingEquilibriumPolicy(game))
    assert nash_conv(tree, policy).nash_conv < 1e-6


def test_beliefs_follow_bayes_rule_after_a_mixed_rejection() -> None:
    game = bargaining.build(MIXING)
    policy = BeliefPolicy(game.params)
    state = game.new_initial_state()
    for action in ("weak", "no_commitment", "offer_050"):
        state.apply_action(action)
    accept_weak = policy.action_probabilities(state)["accept"]
    assert 0.0 < accept_weak < 1.0
    state.apply_action("reject")
    *_, mu_b, _ = policy.track(state)
    prior = MIXING["types"]["B"]["prior"]["strong"]
    assert mu_b == pytest.approx(prior / (prior + (1 - prior) * (1 - accept_weak)))


def test_types_are_private_and_revealed_only_at_the_end() -> None:
    game = bargaining.build(TWO_SIDED)
    state = game.new_initial_state()
    state.apply_action("strong")
    state.apply_action("weak")
    assert any("type:A" in token for token in state.information_state("W")[1])
    assert not any("type:A" in token for token in state.information_state("S")[1])
    assert not any("type:B" in token for token in state.information_state("W")[1])
    assert state.observation("W")[-1] == 1.0 and state.observation("S")[-1] == 0.0


def test_a_contest_can_end_the_standoff() -> None:
    config = screening_config(0.5, horizon=3)
    config["contest"] = {"initial": 0.5, "step": 0.5, "win_probability": 1.0}
    game = bargaining.build(config)
    state = game.new_initial_state()
    for action in ("weak", "offer_100", "reject"):
        state.apply_action(action)
    outcome = state.outcome()
    assert outcome.label == "agreement" and outcome.metrics["share"] == 1.0
    assert "R:collapsed" in outcome.tags and outcome.metrics["period"] == 1.0
    assert outcome.metrics["strong:R"] == 0.0


def test_types_need_a_fixed_proposer() -> None:
    config = screening_config(0.5)
    config["proposers"] = "alternate"
    with pytest.raises(ConfigError):
        bargaining.build(config)
    with pytest.raises(ValueError):
        bargaining.solve(bargaining.build(screening_config(0.5)))


def test_alignment_solver_state_folds_trailing_chance_clients() -> None:
    config = configs.alignment_config()
    config["clients"]["c4"] = {"mode": "chance", "affinity": {"p1": 0.5}, "temperature": 0.4}
    config["patrons"]["p1"]["gain"]["c4"] = 0.7
    game = alignment.build(config)
    full = build_tree(game.new_initial_state())
    folded = build_tree(game.solver_state())
    assert len(folded.nodes) < len(full.nodes)
    full_policy = cfr_plus(full, iterations=300).average_policy
    folded_policy = cfr_plus(folded, iterations=300).average_policy
    table = tabular_from_tree(folded, folded_policy).table
    for index, info in enumerate(full.infosets):
        assert [table[info.key][a] for a in info.actions] == pytest.approx(full_policy[index])
    assert expected_returns(folded, folded_policy) == pytest.approx(
        expected_returns(full, full_policy)
    )


def forbid_config() -> dict[str, Any]:
    config = configs.alignment_config()
    config["patrons"]["p2"]["control"] = {"c3": 2.0}
    config["patrons"]["p2"]["forbid_cost"] = 0.3
    return config


def test_a_patron_can_forbid_its_clients_after_the_offers() -> None:
    game = alignment.build(forbid_config())
    state = game.new_initial_state()
    state.apply_action("aid")  # p1
    state.apply_action("none")  # p2
    assert state.current_player() == "p2" and state.legal_actions() == ("allow", "forbid")
    free = state.clone()
    free.apply_action("allow")
    state.apply_action("forbid")
    client = game.params.clients[2]
    assert state.client_utility(client, "aligned_p1") == pytest.approx(
        free.client_utility(client, "aligned_p1") - 2.0
    )
    # the controlled chance client almost never defies
    for s in (state, free):
        s.apply_action("aligned_p1")  # c1
        s.apply_action("aligned_p1")  # c2
    defy = dict(state.chance_outcomes())["aligned_p1"]
    assert defy < dict(free.chance_outcomes())["aligned_p1"] / 10
    state.apply_action("aligned_p2")
    free.apply_action("aligned_p2")
    assert state.returns()["p2"] == pytest.approx(free.returns()["p2"] - 0.3)
    outcome = state.outcome()
    assert "p2:forbade" in outcome.tags and outcome.metrics["forbade:p2"] == 1.0


def test_folding_keeps_the_forbid_decision() -> None:
    game = alignment.build(forbid_config())
    full, folded = build_tree(game.new_initial_state()), build_tree(game.solver_state())
    full_policy = cfr_plus(full, iterations=300).average_policy
    folded_policy = cfr_plus(folded, iterations=300).average_policy
    table = tabular_from_tree(folded, folded_policy).table
    for index, info in enumerate(full.infosets):
        assert [table[info.key][a] for a in info.actions] == pytest.approx(full_policy[index])
