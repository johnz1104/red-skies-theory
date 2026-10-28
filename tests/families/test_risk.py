"""Risk attitudes: prospect theory's value function on a role's total payoff."""

from __future__ import annotations

from typing import Any

import pytest

from cold_war_sim.families import ConfigError, crisis, intervention
from cold_war_sim.families.common import prospect
from cold_war_sim.games import CHANCE, build_tree, outcome_distribution
from cold_war_sim.games.algorithms.qre import logit_qre

from .configs import intervention_config

TK = {"alpha": 0.88, "beta": 0.88, "lambda": 2.25}  # Tversky and Kahneman (1992)


def gamble_crisis(risk: dict[str, float] | None) -> dict[str, Any]:
    """A concedes for a sure loss of 1, or escalates into a 52% chance of a war that
    costs it 2 (else the crisis ends in a compromise worth 0)."""

    values = {
        "a_prevails": 1.0,
        "b_prevails": -1.0,
        "compromise": 0.0,
        "limited_war": -2.0,
        "nuclear_war": -2.0,
    }
    a: dict[str, Any] = {"side": "a", "values": values}
    if risk is not None:
        a["risk"] = risk
    return {
        "roles": {"A": a, "B": {"side": "b", "values": {k: -x for k, x in values.items()}}},
        "script": [
            {
                "stage": "move",
                "actor": "A",
                "options": [
                    {"id": "back_down", "kind": "concede"},
                    {"id": "gamble", "kind": "escalate"},
                ],
            },
            {"stage": "accident", "base": 0.52},
        ],
        "final_outcome": "compromise",
    }


def _outcomes(game: Any) -> dict[str, float]:
    tree = build_tree(game.new_initial_state())
    return outcome_distribution(tree, logit_qre(tree, temperature=0.01).policy)


def test_prospect_theory_values() -> None:
    assert prospect(2.0, None) == 2.0
    assert prospect(1.0, (0.88, 0.88, 2.25)) == pytest.approx(1.0)
    assert prospect(-1.0, (0.88, 0.88, 2.25)) == pytest.approx(-2.25)
    assert prospect(-2.0, (0.88, 0.88, 2.25)) == pytest.approx(-2.25 * 2.0**0.88)


def test_a_loss_averse_role_gambles_where_a_neutral_one_concedes() -> None:
    neutral = _outcomes(crisis.build(gamble_crisis(None)))
    assert neutral.get("b_prevails", 0.0) > 0.95  # -1 for sure beats 0.52 x -2
    averse = _outcomes(crisis.build(gamble_crisis(TK)))
    assert averse.get("limited_war", 0.0) + averse.get("compromise", 0.0) > 0.99


def test_risk_neutral_attitude_changes_nothing() -> None:
    plain = crisis.build(gamble_crisis(None)).new_initial_state()
    unit = crisis.build(gamble_crisis({"alpha": 1.0, "beta": 1.0, "lambda": 1.0}))
    state = unit.new_initial_state()
    for s in (plain, state):
        s.apply_action("back_down")
    assert state.returns() == plain.returns()


def test_an_intervention_patron_values_its_total_payoff_by_its_attitude() -> None:
    config = intervention_config()
    state = intervention.build(config).new_initial_state()
    config["risk"] = {"patron": TK}
    risky = intervention.build(config).new_initial_state()
    while not state.is_terminal():
        action = (
            state.chance_outcomes()[0][0]
            if state.current_player() == CHANCE
            else state.legal_actions()[0]
        )
        state.apply_action(action)
        risky.apply_action(action)
    patron = config["roles"]["patron"]
    assert risky.returns()[patron] == pytest.approx(
        prospect(state.returns()[patron], (0.88, 0.88, 2.25))
    )


@pytest.mark.parametrize(
    "risk",
    [{"alpha": 1.2}, {"beta": 0.0}, {"lambda": -1.0}, {"gamma": 0.5}],
)
def test_risk_validation(risk: dict[str, float]) -> None:
    with pytest.raises(ConfigError):
        crisis.build(gamble_crisis(risk))
