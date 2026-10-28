"""Crises with incidents: delegated risk decided by an officer, and leaders' control."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from cold_war_sim.families import ConfigError, crisis
from cold_war_sim.games import CHANCE, build_tree, outcome_distribution
from cold_war_sim.games.algorithms.qre import logit_qre


def incident_crisis(
    hold_cost: float = 20.0,
    control: float | None = None,
    leader: str | None = None,
    leader_hold_cost: float = 0.5,
    base: float = 1.0,
) -> dict[str, Any]:
    """B may first order its officer's restraint; an incident then puts the officer to
    restrain or fire. The officer dreads war (-10) but holding costs it *hold_cost*."""

    def side(sign: float) -> dict[str, float]:
        return {
            "a_prevails": sign,
            "b_prevails": -sign,
            "compromise": 0.0,
            "limited_war": -5.0,
            "nuclear_war": -100.0,
        }

    roles = {
        "A": {"side": "a", "values": side(1.0)},
        "B": {"side": "b", "values": side(-1.0)},
        "OFF": {
            "side": "neutral",
            "values": {
                "a_prevails": 0.0,
                "b_prevails": 0.0,
                "compromise": 0.0,
                "limited_war": -2.0,
                "nuclear_war": -10.0,
            },
        },
    }
    script: list[dict[str, Any]] = []
    if control is not None:
        script.append(
            {
                "stage": "move",
                "actor": "B",
                "options": [
                    {"id": "hold_course", "kind": "hold"},
                    {"id": "order_restraint", "kind": "hold", "control": control},
                ],
            }
        )
    fire: Any = "nuclear_war"
    if leader is not None:
        fire = {"leader": leader, "launch": "nuclear_war", "hold_cost": leader_hold_cost}
    script.append(
        {
            "stage": "incident",
            "officer": "OFF",
            "command": "b",
            "base": base,
            "temperaments": {"jumpy": {"weight": 1.0, "hold_cost": hold_cost}},
            "fire": fire,
        }
    )
    return {"roles": roles, "script": script, "final_outcome": "compromise"}


def _outcomes(config: dict[str, Any]) -> dict[str, float]:
    game = crisis.build(config)
    tree = build_tree(game.new_initial_state())
    return outcome_distribution(tree, logit_qre(tree, temperature=0.05).policy)


def test_an_officer_fires_when_holding_costs_it_more_than_war() -> None:
    assert _outcomes(incident_crisis(hold_cost=20.0)).get("nuclear_war", 0.0) > 0.99
    assert _outcomes(incident_crisis(hold_cost=1.0)).get("compromise", 0.0) > 0.99


def test_a_control_order_restrains_the_officer() -> None:
    controlled = _outcomes(incident_crisis(hold_cost=20.0, control=1.0))
    assert controlled.get("compromise", 0.0) > 0.99  # B orders restraint, and the officer holds
    partial = _outcomes(incident_crisis(hold_cost=20.0, control=0.4))
    assert partial.get("nuclear_war", 0.0) > 0.99  # 60% of the hold cost still outweighs war


def test_a_steady_officer_holds_even_when_firing_would_cost_nothing_else() -> None:
    config = incident_crisis(leader="B", leader_hold_cost=0.5)
    config["script"][-1]["temperaments"] = {
        "steady": {"weight": 0.5, "hold_cost": 0.5, "fire_cost": 5.0},
        "by_the_book": {"weight": 0.5, "hold_cost": 15.0},
    }
    game = crisis.build(config)
    state = game.new_initial_state()
    state.apply_action("incident:steady")
    state.apply_action("restrain")
    assert state.returns()["OFF"] == pytest.approx(-0.5)
    state = game.new_initial_state()
    state.apply_action("incident:steady")
    state.apply_action("fire")
    state.apply_action("wait")
    assert state.returns()["OFF"] == pytest.approx(-5.0)  # a false report costs it 5
    # half the officers fire (the by-the-book ones); the leader waits each time
    assert _outcomes(config).get("compromise", 0.0) > 0.99


def test_a_leader_decides_after_an_early_warning() -> None:
    waits = _outcomes(incident_crisis(leader="B", leader_hold_cost=0.5))
    assert waits.get("compromise", 0.0) > 0.99  # the officer reports; the leader waits
    launches = _outcomes(incident_crisis(leader="B", leader_hold_cost=200.0))
    assert launches.get("nuclear_war", 0.0) > 0.99  # a leader sure the attack is real launches


def test_a_launch_turns_nuclear_only_with_ready_weapons() -> None:
    config = incident_crisis(leader="A", leader_hold_cost=50.0)
    config["script"][-1]["fire"].update(launch="limited_war", launch_nuclear=0.4)
    config["script"].insert(0, {"stage": "types", "roles": {"B": {"p_high_readiness": 0.5}}})
    for role in ("A", "B"):
        config["roles"][role]["values"]["nuclear_war"] = -20.0
    outcomes = _outcomes(config)
    # A strikes back (-5 at worst -20 x 0.4 against a hold cost of 50): war turns nuclear
    # only where B's weapons are ready
    assert outcomes["nuclear_war"] == pytest.approx(0.5 * 0.4, abs=0.01)
    assert outcomes["limited_war"] == pytest.approx(1.0 - 0.5 * 0.4, abs=0.01)


def test_an_impossible_incident_leaves_the_game_unchanged() -> None:
    with_stage = incident_crisis(control=0.5, base=0.0)
    without = incident_crisis(control=0.5, base=0.0)
    without["script"] = without["script"][:-1]
    tree_with = build_tree(crisis.build(with_stage).new_initial_state())
    tree_without = build_tree(crisis.build(without).new_initial_state())
    assert len(tree_with.nodes) == len(tree_without.nodes)
    assert _outcomes(with_stage) == pytest.approx(_outcomes(without))


def test_only_the_officer_sees_its_temperament() -> None:
    config = incident_crisis()
    config["script"][-1]["temperaments"] = {
        "steady": {"weight": 0.5, "hold_cost": 1.0},
        "jumpy": {"weight": 0.5, "hold_cost": 20.0},
    }
    state = crisis.build(config).new_initial_state()
    assert state.current_player() == CHANCE
    state.apply_action("incident:jumpy")
    assert state.current_player() == "OFF"
    assert any("temperament=jumpy" in str(item) for item in state.information_state("OFF"))
    for role in ("A", "B"):
        assert not any("temperament" in str(item) for item in state.information_state(role))
    state.apply_action("fire")
    assert state.is_terminal() and state.outcome().label == "nuclear_war"
    assert "OFF:alarm=fire" in state.outcome().tags


def test_an_incident_can_need_ready_weapons() -> None:
    config = incident_crisis(hold_cost=20.0)
    config["script"].insert(0, {"stage": "types", "roles": {"B": {"p_high_readiness": 0.25}}})
    config["script"][-1]["needs_readiness"] = True
    assert _outcomes(config)["nuclear_war"] == pytest.approx(0.25, abs=0.01)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["script"][-1].update(officer="NOBODY"),
        lambda c: c["script"][-1].update(fire="surrender"),
        lambda c: c["script"][-1].update(fire={"leader": "OFF"}),
        lambda c: c["script"][-1].update(command="c"),
        lambda c: c["script"][-1].update(temperaments={"jumpy": {"weight": 0.5}}),
        lambda c: c["script"][0]["options"][1].update(control=1.5),
        lambda c: c["script"][-1].update(fire={"leader": "A", "launch_nuclear": 2.0}),
    ],
)
def test_incident_validation(mutate: Callable[[dict[str, Any]], object]) -> None:
    config = incident_crisis(control=0.5)
    mutate(config)
    with pytest.raises(ConfigError):
        crisis.build(config)
