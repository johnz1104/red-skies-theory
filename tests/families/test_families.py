from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import pytest

from cold_war_sim.families import (
    FAMILIES,
    ConfigError,
    alignment,
    attrition,
    bargaining,
    build_family,
    crisis,
    intervention,
)
from cold_war_sim.games import (
    CHANCE,
    TERMINAL,
    UniformRandomPolicy,
    build_tree,
    expected_returns,
    outcome_distribution,
    sample_action,
)
from cold_war_sim.games.algorithms import backward_induction, cfr_plus, nash_conv
from cold_war_sim.games.traverse import TERMINAL_NODE

from . import configs

SMALL_CONFIGS: list[tuple[str, Any]] = [
    ("crisis", configs.signaling_crisis),
    ("crisis", configs.warning_crisis),
    ("crisis", configs.coalition_crisis),
    ("bargaining", lambda: configs.bargaining_config(horizon=2)),
    ("intervention", configs.intervention_config),
    ("alignment", configs.alignment_config),
    ("attrition", lambda: configs.attrition_config(max_rounds=2)),
]


@pytest.mark.parametrize(("family", "make"), SMALL_CONFIGS)
def test_outcomes_report_each_principals_commitment(family: str, make: Any) -> None:
    game = build_family(family, make())
    principals = {
        "crisis": lambda g: [r.id for r in g.params.roles],
        "bargaining": lambda g: [g.params.a_role, g.params.b_role],
        "intervention": lambda g: [g.params.patron, g.params.rival],
        "alignment": lambda g: [p.id for p in g.params.patrons],
        "attrition": lambda g: [s.id for s in g.params.sponsors],
    }[family](game)
    rng = np.random.default_rng(0)
    for _ in range(20):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.current_player() == CHANCE:
                state.apply_action(sample_action(dict(state.chance_outcomes()), rng))
            else:
                state.apply_action(str(rng.choice(state.legal_actions())))
        metrics = state.outcome().metrics
        for role in principals:
            assert 0.0 <= metrics[f"commit:{role}"] <= 1.0


def _perturb_numbers(value: Any, rng: np.random.Generator) -> Any:
    if isinstance(value, dict):
        return {k: _perturb_numbers(v, rng) for k, v in value.items()}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) + float(rng.normal(0.0, 0.5))
    return value


def _randomize(config: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for key in ("roles", "sponsors", "clients", "patrons"):
        block = result.get(key)
        if isinstance(block, dict):
            for role in block.values():
                if isinstance(role, dict):
                    for field in ("values", "gain", "affinity"):
                        if field in role:
                            role[field] = _perturb_numbers(role[field], rng)
    if "values" in result and isinstance(result["values"], dict):
        result["values"] = _perturb_numbers(result["values"], rng)
    return result


@pytest.mark.parametrize(("family", "factory"), SMALL_CONFIGS)
def test_tree_invariants_under_random_payoffs(family: str, factory: Any) -> None:
    rng = np.random.default_rng(7)
    for _ in range(5):
        game = build_family(family, _randomize(factory(), rng))
        tree = build_tree(game.new_initial_state())  # validates chance distributions
        labels = set(game.outcome_labels)
        for node in tree.nodes:
            if node.kind == TERMINAL_NODE:
                assert node.outcome is not None and node.outcome.label in labels
                assert len(node.returns) == len(game.roles)
                assert all(math.isfinite(v) for v in node.returns)
            else:
                assert node.actions
        assert set(tree.roles) == set(game.roles)


@pytest.mark.parametrize(("family", "factory"), SMALL_CONFIGS)
def test_random_rollouts_terminate_and_clone_independently(family: str, factory: Any) -> None:
    game = build_family(family, factory())
    rng = np.random.default_rng(3)
    policy = UniformRandomPolicy()
    for _ in range(20):
        state = game.new_initial_state()
        steps = 0
        while not state.is_terminal():
            player = state.current_player()
            if player == CHANCE:
                action = sample_action(dict(state.chance_outcomes()), rng)
            else:
                snapshot = state.clone()
                action = sample_action(policy.action_probabilities(state), rng)
                assert snapshot.history() == state.history()
            state.apply_action(action)
            steps += 1
            assert steps < 500
        assert state.current_player() == TERMINAL
        assert set(state.returns()) == set(game.roles)


def test_declared_action_order_is_preserved() -> None:
    state = crisis.build(configs.warning_crisis()).new_initial_state()
    state.apply_action("high")
    assert state.legal_actions() == ("quiet", "warn")
    state.apply_action("warn")
    assert state.legal_actions() == ("restraint", "limited_advance", "full_advance")


def test_complete_information_crisis_by_hand() -> None:
    v = configs.crisis_values
    config = {
        "roles": {
            "A": {"side": "a", "values": v(3, -1, 0, -5, -50), "escalation_cost": 0.5},
            "B": {"side": "b", "values": v(-2, 2, 0, -4, -50)},
        },
        "script": [
            {
                "stage": "move",
                "actor": "A",
                "options": [
                    {"id": "back_off", "kind": "concede"},
                    {"id": "push", "kind": "escalate"},
                ],
            },
            {
                "stage": "move",
                "actor": "B",
                "options": [{"id": "yield", "kind": "concede"}, {"id": "resist", "kind": "fight"}],
            },
        ],
    }
    tree = build_tree(crisis.build(config).new_initial_state())
    result = backward_induction(tree)
    # B yields (-2 > -4); A pushes (3 - 0.5 > -1).
    assert result.values == {"A": pytest.approx(2.5), "B": pytest.approx(-2.0)}
    assert outcome_distribution(tree, result.policy) == {"a_prevails": 1.0}


@pytest.mark.parametrize(
    "factory", [configs.signaling_crisis, configs.warning_crisis, configs.coalition_crisis]
)
def test_crisis_instances_solve_with_low_nash_conv(factory: Any) -> None:
    tree = build_tree(crisis.build(factory()).new_initial_state())
    result = cfr_plus(tree, iterations=600, trembles=0.01)
    assert result.nash_conv < 0.05


def test_trembling_baseline_is_sequentially_rational() -> None:
    tree = build_tree(crisis.build(configs.warning_crisis()).new_initial_state())
    result = cfr_plus(tree, iterations=1500, trembles=0.01)
    for idx, info in enumerate(tree.infosets):
        role, view = info.key  # type: ignore[misc]
        if (
            role == "B"
            and "type:B:resolve=low" in view
            and any(t.startswith("move:A:") for t in view)
        ):
            # A low-resolve defender stays out after an advance: -2 (or -3 after a warning) beats -5.
            assert result.average_policy[idx][0] > 0.95


def test_audience_cost_and_pressure_payoffs() -> None:
    state = crisis.build(configs.coalition_crisis()).new_initial_state()
    for action in ("invade", "join", "threaten", "pressure_side_a", "continue"):
        state.apply_action(action)
    while state.current_player() == CHANCE:
        state.apply_action("calm")
    assert state.outcome().label == "a_prevails"
    returns = state.returns()
    assert returns["A1"] == pytest.approx(1.5 - 1.5)  # pressure cost, side a did not back down
    assert returns["B2"] == pytest.approx(-1 - 0.1)  # signal cost; side b added no rungs


def test_one_period_bargaining_is_the_ultimatum_game() -> None:
    config = {
        "roles": {"A": "P", "B": "R"},
        "grid": [0.0, 0.5, 1.0],
        "horizon": 1,
        "sides": {
            "A": {"breakdown_value": 0, "impasse_value": 0},
            "B": {"breakdown_value": 0, "impasse_value": 0},
        },
    }
    game = bargaining.build(config)
    tree = build_tree(game.new_initial_state())
    assert backward_induction(tree).values == {"P": 1.0, "R": 0.0}
    solution = bargaining.solve(game)
    assert solution.offer[(False, False, 0)] == 2


@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_bargaining_dynamic_program_matches_backward_induction(horizon: int) -> None:
    game = bargaining.build(configs.bargaining_config(horizon=horizon))
    tree = build_tree(game.new_initial_state())
    bi = backward_induction(tree)
    policy = bargaining.BargainingEquilibriumPolicy(game)

    def value(state: Any) -> dict[str, float]:
        if state.is_terminal():
            return state.returns()
        if state.current_player() == CHANCE:
            total: dict[str, float] = {}
            for action, prob in state.chance_outcomes():
                child = state.clone()
                child.apply_action(action)
                for role, v in value(child).items():
                    total[role] = total.get(role, 0.0) + prob * v
            return total
        dist = policy.action_probabilities(state)
        child = state.clone()
        child.apply_action(max(dist, key=lambda a: dist[a]))
        return value(child)

    dp_values = value(game.new_initial_state())
    for role, v in bi.values.items():
        assert dp_values[role] == pytest.approx(v)


def test_bargaining_breakdown_and_commitment_tags() -> None:
    game = bargaining.build(configs.bargaining_config(horizon=2, consent=False))
    state = game.new_initial_state()
    state.apply_action("commit")  # A commits to >= 0.75
    state.apply_action("no_commitment")
    state.apply_action("offer_050")
    state.apply_action("accept")
    outcome = state.outcome()
    assert outcome.label == "agreement" and outcome.metrics["share"] == 0.5
    assert "P1:broke_commitment" in outcome.tags
    assert state.returns()["P1"] == pytest.approx(2.0 * 0.5 - 1.0)


def test_explicit_bargaining_values_reproduce_the_surplus_split() -> None:
    plain = configs.bargaining_config(horizon=3)
    explicit = configs.bargaining_config(horizon=3)
    for side in explicit["sides"].values():
        side["agreement"] = {"base": 0.0, "per_share": explicit["surplus"]}
    explicit["surplus"] = 0.0
    a = bargaining.solve(bargaining.build(plain))
    b = bargaining.solve(bargaining.build(explicit))
    assert a.offer == b.offer and a.accept == b.accept and a.commit == b.commit
    assert a.values == pytest.approx(b.values)


def test_bargaining_vetoed_and_consent_values() -> None:
    config = configs.bargaining_config(horizon=2, commitments=False)
    config["sides"]["A"]["vetoed_value"] = -0.7
    config["consent"][0] |= {"veto_value": 5.0, "breakdown_value": -1.5, "impasse_value": 0.4}
    game = bargaining.build(config)
    vetoed = game.new_initial_state()
    for action in ("offer_050", "accept", "veto"):
        vetoed.apply_action(action)
    assert vetoed.outcome().label == "vetoed"
    assert vetoed.returns()["P1"] == pytest.approx(-0.7)
    assert vetoed.returns()["P2"] == pytest.approx(0.3)  # falls back to the impasse value
    broken = game.new_initial_state()
    for action in ("offer_050", "reject", "breakdown"):
        broken.apply_action(action)
    assert broken.outcome().label == "breakdown"
    assert broken.returns()["C"] == pytest.approx(-1.5)


def test_intervention_solves_and_reports_tags() -> None:
    game = intervention.build(configs.intervention_config())
    tree = build_tree(game.new_initial_state())
    result = cfr_plus(tree, iterations=600, trembles=0.01)
    assert result.nash_conv < 0.01
    state = game.new_initial_state()
    for action in ("strong", "alarming", "intervene", "confront", "back_down"):
        state.apply_action(action)
    assert state.outcome().label == "defected"
    assert "patron:backed_down" in state.outcome().tags


def test_alignment_logit_client_and_player_clients() -> None:
    game = alignment.build(configs.alignment_config())
    assert game.roles == ("p1", "p2", "c1", "c2")
    state = game.new_initial_state()
    state.apply_action("aid")  # p1
    state.apply_action("none")  # p2
    state.apply_action("aligned_p1")  # c1
    state.apply_action("nonaligned")  # c2
    probs = dict(state.chance_outcomes())
    utilities = {"aligned_p1": 0.2 + 0.5, "aligned_p2": 0.6 + 0.0, "nonaligned": 0.0}
    weights = {k: math.exp(u / 0.5) for k, u in utilities.items()}
    total = sum(weights.values())
    for key, weight in weights.items():
        assert probs[key] == pytest.approx(weight / total)
    state.apply_action("aligned_p2")
    outcome = state.outcome()
    assert outcome.label == "aligned_p1"
    assert "c3:aligned_p2" in outcome.tags
    returns = state.returns()
    assert returns["p1"] == pytest.approx(1.0 - 0.2 - 0.5)  # c1 gain - aid cost - c3 loss
    tree = build_tree(game.new_initial_state())
    assert nash_conv(tree, cfr_plus(tree, iterations=300).average_policy).nash_conv < 0.05


def test_attrition_partition_withdrawal_and_heuristic() -> None:
    game = attrition.build(configs.attrition_config())
    state = game.new_initial_state()
    state.apply_action("negotiate")  # a1 (lead a)
    state.apply_action("low")  # b1
    state.apply_action("negotiate")  # b2 (lead b)
    assert state.outcome().label == "partition"

    state = game.new_initial_state()
    state.apply_action("withdraw")
    state.apply_action("low")
    state.apply_action("low")
    assert state.outcome().label == "a_withdrew"

    heuristic = attrition.AttritionHeuristicPolicy(game)
    rng = np.random.default_rng(11)
    labels = set()
    for _ in range(30):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.current_player() == CHANCE:
                state.apply_action(sample_action(dict(state.chance_outcomes()), rng))
            else:
                state.apply_action(sample_action(heuristic.action_probabilities(state), rng))
        assert state.outcome().metrics["rounds"] <= game.params.max_rounds
        labels.add(state.outcome().label)
    assert labels <= set(attrition.OUTCOMES)


def _uphill_war(win_value: float, rounds: int = 28) -> dict[str, Any]:
    """A winnable war from a weak start: side a starts at 0.35 and needs 10 net wins."""

    return {
        "sponsors": {
            "a1": {
                "side": "a",
                "capacity": 1.2,
                "direct": True,
                "values": configs.attrition_values(win_value, -5, 1.0, -5, win_value, 0.5),
            },
            "b1": {
                "side": "b",
                "capacity": 0.8,
                "values": configs.attrition_values(-2, 2, 0.3, 2, -2, 0),
            },
        },
        "leads": {"a": "a1", "b": "b1"},
        "local_strength": {"a": 1.0, "b": 1.6},
        "initial_control": 0.35,
        "step": 0.05,
        "max_rounds": rounds,
        "weariness": {"high": 0.0, "direct": 0.0},
    }


def test_lookahead_matches_the_gamblers_ruin_formula() -> None:
    config = _uphill_war(5.0, rounds=2000)
    config["costs"] = {"low": 0.0, "high": 0.0, "direct": 0.0}
    game = attrition.build(config)
    state = game.new_initial_state()
    policy = attrition.AttritionLookaheadPolicy(game)
    value = policy.continuation_value(state, "a1", "direct")
    p_win = state.win_probability({"a1": "direct", "b1": "low"})
    r, start, cells = (1 - p_win) / p_win, 4, 14  # control 0.35 is 4 steps above 0.15
    reach = (1 - r**start) / (1 - r**cells)
    assert value == pytest.approx(reach * 5.0 + (1 - reach) * -5.0, abs=1e-6)


def test_lookahead_fights_a_winnable_war_from_a_weak_start() -> None:
    game = attrition.build(_uphill_war(10.0))
    state = game.new_initial_state()
    myopic = attrition.AttritionHeuristicPolicy(game).action_probabilities(state)
    lookahead = attrition.AttritionLookaheadPolicy(game).action_probabilities(state)
    assert max(myopic, key=myopic.get) == "negotiate"  # the straight-line value of 0.35 is low
    assert max(lookahead, key=lookahead.get) == "direct"


def test_lookahead_avoids_levels_that_would_force_it_out() -> None:
    config = _uphill_war(10.0)
    config["weariness"] = {"high": 0.05, "direct": 0.2}
    config["weariness_limit"] = 0.3
    game = attrition.build(config)
    state = game.new_initial_state()
    state.weariness["a1"] = 0.25  # one more round of direct effort forces a1 out
    choice = attrition.AttritionLookaheadPolicy(game).action_probabilities(state)
    assert max(choice, key=choice.get) != "direct"


def test_attrition_weariness_forces_withdrawal() -> None:
    config = configs.attrition_config()
    config["weariness"] = {"high": 0.5, "direct": 1.0}
    config["initial_control"] = 0.5
    config["step"] = 0.01
    game = attrition.build(config)
    state = game.new_initial_state()
    state.apply_action("direct")
    state.apply_action("low")
    state.apply_action("low")
    while state.current_player() == CHANCE:
        state.apply_action(state.chance_outcomes()[0][0])
    assert state.outcome().label == "a_withdrew"


def test_expected_returns_are_consistent_with_rollouts() -> None:
    game = intervention.build(configs.intervention_config())
    tree = build_tree(game.new_initial_state())
    from cold_war_sim.games.traverse import uniform_policy

    exact = expected_returns(tree, uniform_policy(tree))
    rng = np.random.default_rng(0)
    totals = dict.fromkeys(game.roles, 0.0)
    n = 4000
    policy = UniformRandomPolicy()
    for _ in range(n):
        state = game.new_initial_state()
        while not state.is_terminal():
            dist = (
                dict(state.chance_outcomes())
                if state.current_player() == CHANCE
                else policy.action_probabilities(state)
            )
            state.apply_action(sample_action(dist, rng))
        for role, v in state.returns().items():
            totals[role] += v / n
    for role in game.roles:
        assert totals[role] == pytest.approx(exact[role], abs=0.15)


@pytest.mark.parametrize(
    ("family", "mutation"),
    [
        ("crisis", lambda c: c["roles"]["A"].update(side="x")),
        ("crisis", lambda c: c["script"].append({"stage": "nonsense"})),
        (
            "crisis",
            lambda c: c["script"].insert(0, {"stage": "intel", "about": "A", "accuracy": 0.7}),
        ),
        ("bargaining", lambda c: c.update(grid=[0.5, 0.2])),
        ("attrition", lambda c: c.update(initial_control=0.95)),
        ("alignment", lambda c: c["clients"]["c1"].update(mode="robot")),
        ("intervention", lambda c: c.update(unexpected=1)),
    ],
)
def test_invalid_configs_are_rejected(family: str, mutation: Any) -> None:
    factory = {
        "crisis": configs.signaling_crisis,
        "bargaining": configs.bargaining_config,
        "attrition": configs.attrition_config,
        "alignment": configs.alignment_config,
        "intervention": configs.intervention_config,
    }[family]
    config = factory()
    mutation(config)
    with pytest.raises(ConfigError):
        build_family(family, config)


def test_registry_lists_all_families() -> None:
    assert set(FAMILIES) == {"crisis", "bargaining", "intervention", "attrition", "alignment"}
