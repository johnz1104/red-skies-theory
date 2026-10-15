"""Crises with private stakes and roles' own beliefs about others' types."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from cold_war_sim.families import ConfigError, crisis
from cold_war_sim.families.crisis_beliefs import CrisisBeliefPolicy
from cold_war_sim.games import CHANCE, build_tree
from cold_war_sim.games.algorithms.qre import logit_qre
from cold_war_sim.games.policy import tabular_from_tree
from cold_war_sim.games.traverse import GameTree, TreePolicy

from .configs import crisis_values


def _solve(tree: GameTree) -> TreePolicy:
    return logit_qre(tree, temperature=0.1).policy


def beliefs_crisis(
    resolve_belief: float | None = 0.6,
    stakes_belief: float | None = 0.8,
    acquiescent_weight: float = 0.0,
) -> dict[str, Any]:
    """A moves first; a neutral partner P may pressure it; B may intervene late.

    P is opposed to A in fact (by default), and A may believe it acquiescent; B
    rarely has the resolve to intervene, and A may believe it likelier."""

    p_types: dict[str, Any] = {
        "stakes": {
            "opposed": {"weight": 1.0 - acquiescent_weight},
            "acquiescent": {"weight": acquiescent_weight, "offsets": {"a_prevails": 3.0}},
        }
    }
    if stakes_belief is not None:
        p_types["stakes_beliefs"] = {
            "A": {"opposed": 1.0 - stakes_belief, "acquiescent": stakes_belief}
        }
    b_types: dict[str, Any] = {"p_high_resolve": 0.1}
    if resolve_belief is not None:
        b_types["resolve_beliefs"] = {"A": resolve_belief}
    return {
        "roles": {
            "A": {"side": "a", "values": crisis_values(2, -1, 0, -4, -40), "escalation_cost": 0.5},
            "B": {
                "side": "b",
                "values": crisis_values(-1, 1, 0, -3, -40),
                "high_resolve_war_factor": 0.1,
            },
            "P": {"side": "neutral", "values": crisis_values(-1, 0.5, 0, -2, -40)},
        },
        "script": [
            {"stage": "types", "roles": {"B": b_types, "P": p_types}},
            {
                "stage": "move",
                "actor": "A",
                "options": [
                    {"id": "refrain", "kind": "concede"},
                    {"id": "invade", "kind": "escalate"},
                ],
            },
            {
                "stage": "partner",
                "actor": "P",
                "options": [
                    {"id": "pressure", "target": "a", "continue_cost": 4.0},
                    {"id": "stand_aside", "target": "a"},
                ],
            },
            {
                "stage": "move",
                "actor": "A",
                "options": [
                    {"id": "withdraw", "kind": "concede"},
                    {"id": "press_on", "kind": "hold"},
                ],
            },
            {
                "stage": "move",
                "actor": "B",
                "options": [
                    {"id": "stand_aside", "kind": "hold"},
                    {"id": "intervene", "kind": "fight"},
                ],
            },
            {"stage": "accident", "base": 0.01},
        ],
        "final_outcome": "a_prevails",
    }


def _first_decision(game: crisis.CrisisGame, actor: str) -> Any:
    state = game.new_initial_state()
    while state.current_player() != actor:
        if state.current_player() == CHANCE:
            state.apply_action(state.chance_outcomes()[0][0])
        else:
            state.apply_action(state.legal_actions()[-1])
    return state


def test_beliefs_equal_to_nature_give_the_plain_solution() -> None:
    config = beliefs_crisis(resolve_belief=0.1, stakes_belief=0.5, acquiescent_weight=0.5)
    game = crisis.build(config)
    plain = tabular_from_tree(tree := build_tree(game.new_initial_state()), _solve(tree))
    policy = CrisisBeliefPolicy(game, _solve)
    for role in game.roles:
        assert policy.policy_of(role).table == plain.table


def test_a_role_acts_on_its_own_belief_about_a_partners_stakes() -> None:
    believing = crisis.build(beliefs_crisis(resolve_belief=None))
    informed = crisis.build(beliefs_crisis(resolve_belief=None, stakes_belief=None))
    assert believing.has_beliefs and not informed.has_beliefs
    state = _first_decision(believing, "A")
    invade = CrisisBeliefPolicy(believing, _solve).action_probabilities(state)["invade"]
    tree = build_tree(informed.new_initial_state())
    plain = tabular_from_tree(tree, _solve(tree))
    assert plain.action_probabilities(_first_decision(informed, "A"))["invade"] < 0.2
    assert invade > 0.8  # A expects P to stand aside, as it believes P acquiescent


def test_a_believed_level_nature_never_draws_exists_only_in_the_believers_view() -> None:
    game = crisis.build(beliefs_crisis())
    state = game.new_initial_state()
    assert state.current_player() == CHANCE
    assert {a for a, _ in state.chance_outcomes()} == {"high", "low"}  # B's resolve only
    state.apply_action("low")
    assert state.types["P"]["stakes"] == "opposed"  # drawn for certain, logged to P only
    assert any("stakes=opposed" in token for token in state.information_state("P")[1])
    assert not any("stakes" in token for token in state.information_state("A")[1])
    view = game.views()["A"]
    root = view.new_initial_state()
    assert dict(root.chance_outcomes())["high"] == pytest.approx(0.6)
    root.apply_action("low")
    assert dict(root.chance_outcomes()) == pytest.approx({"opposed": 0.2, "acquiescent": 0.8})
    views = game.views()
    assert views["B"] is views["P"]  # same priors, one shared view


def test_every_information_state_a_role_meets_is_in_its_view() -> None:
    game = crisis.build(beliefs_crisis())
    policy = CrisisBeliefPolicy(game, _solve)
    tree = build_tree(game.new_initial_state())
    for info in tree.infosets:
        assert info.key in policy.policy_of(info.key[0]).table


def test_stakes_offsets_enter_only_the_holders_returns() -> None:
    game = crisis.build(beliefs_crisis(acquiescent_weight=0.5))

    def play(level: str) -> dict[str, float]:
        state = game.new_initial_state()
        moves = {"A": ["invade", "press_on"], "P": ["stand_aside"], "B": ["stand_aside"]}
        while not state.is_terminal():
            player = state.current_player()
            if player == CHANCE:
                outcomes = dict(state.chance_outcomes())
                state.apply_action(
                    level if level in outcomes else "low" if "low" in outcomes else "calm"
                )
            else:
                state.apply_action(moves[player].pop(0))
        assert state.outcome().label == "a_prevails"
        return state.returns()

    opposed, acquiescent = play("opposed"), play("acquiescent")
    assert acquiescent["P"] == pytest.approx(opposed["P"] + 3.0)
    assert acquiescent["A"] == opposed["A"] and acquiescent["B"] == opposed["B"]


def test_a_resolute_role_follows_through_and_a_bluffing_one_does_not() -> None:
    game = crisis.build(beliefs_crisis(resolve_belief=None, stakes_belief=None))
    tree = build_tree(game.new_initial_state())
    plain = tabular_from_tree(tree, _solve(tree))
    for resolve, intervenes in (("high", True), ("low", False)):
        state = game.new_initial_state()
        state.apply_action(resolve)
        for action in ("invade", "stand_aside", "press_on"):
            state.apply_action(action)
        assert state.current_player() == "B"
        assert (plain.action_probabilities(state)["intervene"] > 0.5) is intervenes


def test_a_role_that_believes_a_threat_backs_down() -> None:
    def press_on(resolve_belief: float | None) -> float:
        game = crisis.build(beliefs_crisis(resolve_belief=resolve_belief, stakes_belief=None))
        state = game.new_initial_state()
        state.apply_action("low")  # B is bluffing
        state.apply_action("invade")
        state.apply_action("stand_aside")
        policy = CrisisBeliefPolicy(game, _solve)
        return policy.action_probabilities(state)["press_on"]

    assert press_on(None) > 0.9  # nature: B is rarely resolute
    assert press_on(0.6) < 0.1  # A believes B likely to intervene


def test_an_options_costs_fall_on_the_listed_roles_when_it_is_taken() -> None:
    config = beliefs_crisis(resolve_belief=None, stakes_belief=None)
    config["script"][2]["options"][0]["costs"] = {"P": 0.7, "A": 0.4}  # P's pressure
    game = crisis.build(config)

    def play(partner: str) -> dict[str, float]:
        state = game.new_initial_state()
        for action in ("low", "invade", partner, "withdraw"):
            state.apply_action(action)
        return state.returns()

    plain = crisis.build(beliefs_crisis(resolve_belief=None, stakes_belief=None))
    base = plain.new_initial_state()
    for action in ("low", "invade", "pressure", "withdraw"):
        base.apply_action(action)
    pressured = play("pressure")
    assert pressured["P"] == pytest.approx(base.returns()["P"] - 0.7)
    assert pressured["A"] == pytest.approx(base.returns()["A"] - 0.4)
    assert pressured["B"] == pytest.approx(base.returns()["B"])
    assert play("stand_aside")["P"] == pytest.approx(0.5)  # b_prevails, no cost when not taken


def test_a_costly_option_is_chosen_less() -> None:
    def pressure(cost: float) -> float:
        config = beliefs_crisis(resolve_belief=None, stakes_belief=None)
        config["script"][2]["options"][0]["costs"] = {"P": cost}
        game = crisis.build(config)
        tree = build_tree(game.new_initial_state())
        policy = tabular_from_tree(tree, _solve(tree))
        state = game.new_initial_state()
        for action in ("low", "invade"):
            state.apply_action(action)
        return policy.action_probabilities(state)["pressure"]

    assert pressure(0.0) > 0.9
    assert pressure(3.0) < 0.1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["script"][0]["roles"]["B"].update(resolve_beliefs={"B": 0.5}),
        lambda c: c["script"][0]["roles"]["B"].update(resolve_beliefs={"Z": 0.5}),
        lambda c: c["script"][0]["roles"]["B"].pop("p_high_resolve"),
        lambda c: c["script"][0]["roles"]["P"]["stakes_beliefs"]["A"].update(opposed=0.5),
        lambda c: c["script"][0]["roles"]["P"]["stakes"]["opposed"].update(weight=0.5),
        lambda c: c["script"][0]["roles"]["P"]["stakes"]["acquiescent"].update(offsets={"win": 1}),
        lambda c: c["script"][0]["roles"]["P"]["stakes"].pop("acquiescent"),
        lambda c: c["script"][0]["roles"]["P"].pop("stakes"),
        lambda c: c["script"][1]["options"][0].update(costs={"Z": 1.0}),
        lambda c: c["script"].insert(
            1, {"stage": "types", "roles": {"P": c["script"][0]["roles"]["P"]}}
        ),
    ],
)
def test_belief_and_stakes_validation(mutate: Callable[[dict[str, Any]], object]) -> None:
    config = beliefs_crisis()
    mutate(config)
    with pytest.raises(ConfigError):
        crisis.build(config)
