"""Attrition wars that bargain while fighting: rules, beliefs and reference play."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import pytest

from cold_war_sim.families import ConfigError, attrition
from cold_war_sim.families.attrition_beliefs import AttritionBeliefPolicy
from cold_war_sim.games import CHANCE
from cold_war_sim.games.algorithms import nash_conv
from cold_war_sim.games.traverse import uniform_policy

from . import configs
from .test_bargaining_beliefs import tree_policy


def play(game: attrition.AttritionGame, actions: Iterable[str]) -> attrition.AttritionState:
    state = game.new_initial_state()
    for action in actions:
        state.apply_action(action)
    return state


def test_plain_wars_are_unchanged() -> None:
    game = attrition.build(configs.attrition_config())
    assert game.outcome_labels == attrition.OUTCOMES
    assert not game.params.extended
    state = game.new_initial_state()
    belief = AttritionBeliefPolicy(game, temperature=0.1).action_probabilities(state)
    lookahead = attrition.AttritionLookaheadPolicy(game, temperature=0.1)
    assert belief == lookahead.action_probabilities(state)
    with pytest.raises(ConfigError, match="AttritionBeliefPolicy"):
        attrition.AttritionLookaheadPolicy(attrition.build(configs.attrition_offers_config()))


def test_stages_run_in_order() -> None:
    game = attrition.build(configs.attrition_offers_config(approval=True, consent=True))
    assert game.outcome_labels[-1] == "refused"
    state = game.new_initial_state()
    assert state.current_player() == CHANCE and state.stage == "types"
    state.apply_action("strong")  # side A's resolve
    state.apply_action("weak")  # side B's strength
    assert (state.stage, state.current_player()) == ("approval", "b1")
    state.apply_action("approve")
    assert (state.stage, state.current_player()) == ("offer", "a1")
    assert state.legal_actions() == ("no_offer", "offer_030", "offer_050", "offer_070")
    state.apply_action("offer_050")
    assert (state.stage, state.current_player()) == ("respond", "b1")
    state.apply_action("accept")
    assert (state.stage, state.current_player()) == ("ratify", "b2")
    state.apply_action("veto")  # the war goes on, and the veto costs the vetoer
    assert state.stage == "effort" and state.cost["b2"] == pytest.approx(0.1)
    for _ in range(3):
        state.apply_action("low")
    assert state.stage == "battle"
    state.apply_action("hold")
    outcome = state.outcome()
    assert outcome.label == "stalemate"  # a one-round war
    assert "b2:vetoed" in outcome.tags
    assert outcome.metrics["strong:a"] == 1.0 and outcome.metrics["strong:b"] == 0.0


def test_an_agreement_ends_the_war_at_its_share() -> None:
    game = attrition.build(configs.attrition_offers_config())
    state = play(game, ["weak", "strong", "offer_070", "accept"])
    outcome = state.outcome()
    assert outcome.label == "partition" and "agreed" in outcome.tags
    assert outcome.metrics["share"] == 0.7
    assert outcome.metrics["share_change"] == pytest.approx(0.2)
    returns = state.returns()
    assert returns["a1"] == pytest.approx(-1.5 + 3.0 * 0.7)
    assert returns["b1"] == pytest.approx(-1.5 + 3.0 * 0.3)


def test_a_refusal_ends_the_game_before_the_war() -> None:
    game = attrition.build(configs.attrition_offers_config(approval=True))
    state = play(game, ["strong", "strong", "refuse"])
    assert state.outcome().label == "refused"
    assert "b1:refused" in state.outcome().tags
    assert state.returns() == {"a1": 0.0, "b1": -0.2}


def test_types_are_private_to_their_side() -> None:
    game = attrition.build(configs.attrition_offers_config())
    state = play(game, ["strong", "weak"])
    view_a, view_b = state.information_state("a1")[1], state.information_state("b1")[1]
    assert "type:a=strong" in view_a and "type:b=weak" not in view_a
    assert "type:b=weak" in view_b and "type:a=strong" not in view_b


def test_the_front_holds_more_often_as_forces_grow() -> None:
    game = attrition.build(configs.attrition_offers_config())
    state = play(game, ["strong", "strong", "no_offer"])
    light = dict(state.battle_outcomes({"a1": "low", "b1": "low"}))
    heavy = dict(state.battle_outcomes({"a1": "high", "b1": "high"}))
    total = (1.0 + 1.0) + (1.0 + 1.0) * 1.6  # E_A + E_B at low effort, B strong
    assert light["hold"] == pytest.approx(1.0 - 1.0 / (1.0 + (total / 3.0) ** 2))
    assert heavy["hold"] > light["hold"]
    assert sum(light.values()) == pytest.approx(1.0)


def test_a_warning_commits_the_warner() -> None:
    config = configs.attrition_offers_config(max_rounds=3, red_line=True)
    config["offers"]["opens"] = 5  # no offers in this war
    config["step"] = 0.1
    game = attrition.build(config)
    state = play(game, ["strong", "strong"])
    assert (state.stage, state.current_player()) == ("warn", "b1")  # the front is at the line
    for action in ("warn", "low", "low", "a_gains"):  # A crosses the line
        state.apply_action(action)
    assert state.stage == "effort" and state.control == pytest.approx(0.6)
    bluff, honor = state.clone(), state.clone()
    for child, choice in ((bluff, "high"), (honor, "direct")):
        assert isinstance(child, attrition.AttritionState)
        child.apply_action("low")
        child.apply_action(choice)
    assert isinstance(bluff, attrition.AttritionState) and isinstance(
        honor, attrition.AttritionState
    )
    assert bluff.bluffed == {"b1"} and honor.honored == {"b1"}
    assert bluff.cost["b1"] - honor.cost["b1"] == pytest.approx(0.5 + 0.3 - 0.6)


def test_direct_intervention_takes_a_round_to_arrive() -> None:
    config = configs.attrition_offers_config(max_rounds=3, red_line=True)
    config.update(deployment=0.3, step=0.1)
    config["offers"]["opens"] = 5
    config.pop("red_lines")
    game = attrition.build(config)
    assert game.params.extended
    state = play(game, ["strong", "weak", "direct", "low"])  # A goes direct in round 0
    effort = state.side_effort("a", state.choices)
    assert effort == pytest.approx(1.0 + 1.0 * 4.0 * 0.3)  # local + capacity x direct x share
    state.apply_action("hold")
    for action in ("direct", "low"):
        state.apply_action(action)
    assert state.side_effort("a", state.choices) == pytest.approx(1.0 + 4.0)  # arrived
    plain = attrition.build(configs.attrition_offers_config(max_rounds=3, red_line=True))
    fresh = play(plain, ["strong", "weak", "silent", "no_offer", "direct", "low"])
    assert fresh.side_effort("a", {"a1": "direct", "b1": "low"}) == pytest.approx(5.0)


def test_deployment_can_differ_by_sponsor() -> None:
    config = configs.attrition_offers_config(max_rounds=3, red_line=True)
    config.update(deployment={"a1": 0.3}, step=0.1)  # b1 massed beforehand: arrives in full
    config["offers"]["opens"] = 5
    config.pop("red_lines")
    game = attrition.build(config)
    assert game.params.extended
    assert game.params.deployment_of("a1") == 0.3 and game.params.deployment_of("b1") == 1.0
    state = play(game, ["strong", "weak", "direct", "direct"])
    assert state.side_effort("a", state.choices) == pytest.approx(1.0 + 4.0 * 0.3)
    assert state.side_effort("b", state.choices) == pytest.approx((1.0 + 4.0) * 0.6)  # weak B


def test_a_commitment_is_public_and_a_concession_costs_the_committed() -> None:
    game = attrition.build(configs.attrition_offers_config(max_rounds=2, commitments=True))
    state = play(game, ["strong", "weak"])
    assert (state.stage, state.current_player()) == ("commit", "a1")
    assert state.legal_actions() == ("hold_off", "commit")
    state.apply_action("commit")
    assert (state.stage, state.current_player()) == ("commit", "b1")
    state.apply_action("hold_off")
    assert "r0:a1=commit" in state.information_state("b1")[1]
    kept, conceded = state.clone(), state.clone()
    for child, offer in ((kept, "offer_070"), (conceded, "offer_050")):
        assert isinstance(child, attrition.AttritionState)
        child.apply_action(offer)
        child.apply_action("accept")
    assert isinstance(kept, attrition.AttritionState)
    assert isinstance(conceded, attrition.AttritionState)
    assert "a1:committed" in kept.outcome().tags and "a1:conceded" not in kept.outcome().tags
    assert kept.returns()["a1"] == pytest.approx(-1.5 + 3.0 * 0.7)
    assert "a1:conceded" in conceded.outcome().tags
    assert conceded.returns()["a1"] == pytest.approx(-1.5 + 3.0 * 0.5 - 0.5)


def test_a_pledge_can_weigh_less_by_scale() -> None:
    config = configs.attrition_offers_config(max_rounds=2, commitments=True)
    config["commitments"]["a1"]["scale"] = 0.4  # the pledge weighs less in this world
    game = attrition.build(config)
    state = play(game, ["strong", "weak", "commit", "hold_off", "offer_050", "accept"])
    assert "a1:conceded" in state.outcome().tags
    assert state.returns()["a1"] == pytest.approx(-1.5 + 3.0 * 0.5 - 0.5 * 0.4)


def test_commitments_outlive_new_terms_and_are_asked_once() -> None:
    config = configs.attrition_offers_config(max_rounds=3, commitments=True)
    state = play(attrition.build(config), ["strong", "strong", "commit", "hold_off"])
    config["commitments"]["a1"]["audience_cost"] = 2.0  # a re-bound war: new values only
    state.params = attrition.build(config).params
    for action in ("no_offer", "low", "low", "hold"):
        state.apply_action(action)
    assert state.round == 1 and state.committed == {"a1"}
    assert (state.stage, state.current_player()) == ("commit", "b1")  # only B is asked again
    observation = state.observation("b1")
    assert observation[16:18] == (0.0, 1.0)  # own side uncommitted, the other side committed


def test_with_offers_settlements_come_only_through_offers() -> None:
    game = attrition.build(configs.attrition_offers_config(max_rounds=3, opens=1))
    state = play(game, ["strong", "strong"])
    assert state.stage == "effort" and "negotiate" not in state.legal_actions()
    for action in ("low", "low", "hold"):
        state.apply_action(action)
    assert state.stage == "offer"  # talks open in round 1
    state.apply_action("no_offer")
    assert state.stage == "effort" and "negotiate" not in state.legal_actions()
    plain = attrition.build(configs.attrition_config())  # without offers, leads negotiate
    assert "negotiate" in plain.new_initial_state().legal_actions()


def test_resolve_levels_are_drawn_privately_and_ranked() -> None:
    game = attrition.build(configs.attrition_offers_config(max_rounds=2, levels=True))
    a_types, b_types = game.params.a_types, game.params.b_types
    assert a_types is not None and b_types is not None
    assert a_types.kinds == ("resolute", "firm", "wavering")
    assert b_types.probabilities == pytest.approx((0.25, 0.5, 0.25))  # weights 1, 2, 1
    state = game.new_initial_state()
    assert dict(state.chance_outcomes()) == pytest.approx(
        {"resolute": 1 / 3, "firm": 1 / 3, "wavering": 1 / 3}
    )
    state = play(game, ["firm", "steady"])
    assert "type:a=firm" in state.information_state("a1")[1]
    assert "type:a=firm" not in state.information_state("b1")[1]
    assert state.cost_multiplier("a") == 1.5
    assert state.observation("a1")[10] == pytest.approx(0.5)  # the middle of three levels
    assert state.observation("b1")[10] == pytest.approx(0.5)
    state.apply_action("offer_050")
    state.apply_action("accept")
    metrics = state.outcome().metrics
    assert metrics["rank:a"] == 1.0 and metrics["strong:a"] == 0.0


def test_beliefs_over_levels_follow_bayes_rule() -> None:
    game = attrition.build(configs.attrition_offers_config(max_rounds=2, levels=True))
    policy = AttritionBeliefPolicy(game)
    state = play(game, ["resolute", "strong", "no_offer", "low", "low"])
    before = policy.joint(state)
    assert len(before) == 9
    likelihood = {
        cell: dict(state.battle_outcomes(state.choices, {"a": cell[0], "b": cell[1]}))["a_gains"]
        for cell in before
        if cell[0] is not None and cell[1] is not None
    }
    state.apply_action("a_gains")
    after = policy.joint(state)
    total = sum(before[cell] * likelihood[cell] for cell in before)
    for cell in before:
        assert after[cell] == pytest.approx(before[cell] * likelihood[cell] / total)
    weakest = sum(q for cell, q in after.items() if cell[1] == "weak")
    assert weakest > sum(q for cell, q in before.items() if cell[1] == "weak")


def test_a_sponsor_has_stakes_of_its_own_and_takes_the_lead_on_entry() -> None:
    game = attrition.build(configs.attrition_offers_config(max_rounds=3, opens=1, entrant=True))
    assert game.params.holders == ("a", "b", "c1")
    state = play(game, ["strong", "strong"])
    assert dict(state.chance_outcomes()) == {"committed": 0.5, "uncommitted": 0.5}
    state.apply_action("uncommitted")
    assert "type:c1=uncommitted" in state.information_state("b1")[1]  # its side sees it
    assert "type:c1=uncommitted" not in state.information_state("a1")[1]
    assert state.role_cost_multiplier("c1") == 6.0 and state.role_cost_multiplier("b1") == 1.0
    assert state.lead("b") == "b1"
    for action in ("low", "low", "direct"):  # c1 goes direct in round 0
        state.apply_action(action)
    assert state.lead("b") == "c1"
    assert state.cost["c1"] == pytest.approx(0.6 * 6.0)
    state.apply_action("hold")
    state.apply_action("offer_050")  # round 1: A proposes and the new lead responds
    assert (state.stage, state.current_player()) == ("respond", "c1")


def test_own_beliefs_differ_by_side_and_shape_choices() -> None:
    config = configs.attrition_offers_config(max_rounds=2, approval=True, levels=True)
    same = attrition.build(config)
    state = play(same, ["resolute", "strong"])
    policy = AttritionBeliefPolicy(same, temperature=0.1)
    assert policy.joint(state, "a") == policy.joint(state, "b")  # no beliefs: one view
    approve = {}
    for believed in ("resolute", "wavering"):  # B believes A almost surely of this level
        cfg = configs.attrition_offers_config(max_rounds=2, approval=True, levels=True)
        cfg["types"]["a"]["levels"]["wavering"]["cost"] = 40.0  # would not fight at all
        for kind, spec in cfg["types"]["a"]["levels"].items():
            spec["belief"] = 1.0 if kind == believed else 0.01
        game = attrition.build(cfg)
        policy = AttritionBeliefPolicy(game, temperature=0.1)
        state = play(game, ["resolute", "strong"])
        joint_a, joint_b = policy.joint(state, "a"), policy.joint(state, "b")
        assert joint_a != joint_b
        assert policy.public_belief(state)["a"] == pytest.approx(
            sum(q for cell, q in joint_b.items() if cell[0] == "resolute")
        )
        approve[believed] = policy.action_probabilities(state)["approve"]
    assert approve["wavering"] > approve["resolute"]  # B expects an easier war


def test_a_private_type_is_seen_by_its_holder_alone() -> None:
    config = configs.attrition_offers_config(
        max_rounds=2, consent=True, entrant=True, proxy=0.5, hidden=(0.5, 0.5)
    )
    game = attrition.build(config)
    assert game.params.holders == ("a", "b", "b2", "c1")
    state = play(game, ["strong", "strong", "holds_out", "committed"])
    assert "type:b2=holds_out" in state.information_state("b2")[1]
    for role in ("a1", "b1", "c1"):
        assert "type:b2=holds_out" not in state.information_state(role)[1]
    assert "type:c1=committed" in state.information_state("b2")[1]  # its side's public type
    assert state.proxy_multiplier("b2") == 1.0
    assert state.proxy_multiplier("b2", {"b2": "settles"}) == 0.0


def test_others_act_on_their_belief_about_a_private_type() -> None:
    """An entrant that believes its ally will settle chooses as if the ally had no gain,
    while the ally, which holds out, vetoes; the others learn from its veto by Bayes."""

    believing = attrition.build(
        configs.attrition_offers_config(
            max_rounds=2, consent=True, entrant=True, proxy=2.0, hidden=(1.0, 0.0)
        )
    )
    no_gain = attrition.build(
        configs.attrition_offers_config(max_rounds=2, consent=True, entrant=True)
    )
    choices = {}
    for game in (believing, no_gain):
        state = play(game, ["strong", "strong", "committed", "no_offer", "low", "low", "low"])
        assert state.current_player() == "c1"
        choices[game is believing] = AttritionBeliefPolicy(game, 0.1).action_probabilities(state)
    for action, p in choices[False].items():
        assert choices[True][action] == pytest.approx(p)
    learning = attrition.build(
        configs.attrition_offers_config(
            max_rounds=2, consent=True, entrant=True, proxy=2.0, hidden=(0.5, 0.5)
        )
    )
    policy = AttritionBeliefPolicy(learning, 0.1)
    state = play(
        learning,
        ["strong", "strong", "holds_out", "committed", "no_offer", "low", "low", "low", "direct"],
    )
    state.apply_action("hold")
    state.apply_action("offer_050")
    state.apply_action("accept")
    assert (state.stage, state.current_player()) == ("ratify", "b2")
    assert policy.action_probabilities(state)["veto"] > 0.9  # it holds out
    index = learning.params.holders.index("b2")

    def holds_out(view: str) -> float:
        return sum(q for cell, q in policy.joint(state, view).items() if cell[index] == "holds_out")

    before = {view: holds_out(view) for view in ("a", "b")}
    state.apply_action("veto")
    for view in ("a", "b"):
        assert holds_out(view) > before[view]


def test_withdrawing_costs_its_withdraw_cost() -> None:
    """A sponsor with no say quits to free-ride on its lead, unless quitting costs it."""

    choice, cost = {}, {}
    for withdraw_cost in (0.0, 1.0):
        config = configs.attrition_offers_config(max_rounds=2, consent=True)
        config["sponsors"]["b2"]["withdraw_cost"] = withdraw_cost
        game = attrition.build(config)
        state = play(game, ["strong", "strong", "no_offer", "low", "low"])
        assert state.current_player() == "b2"
        choice[withdraw_cost] = AttritionBeliefPolicy(game, 0.1).action_probabilities(state)
        state.apply_action("withdraw")
        cost[withdraw_cost] = state.cost["b2"]
    assert choice[0.0]["withdraw"] > 0.5 > choice[1.0]["withdraw"]
    assert cost[1.0] == pytest.approx(cost[0.0] + 1.0)


def test_momentum_shifts_the_local_balance_with_the_front() -> None:
    game = attrition.build(configs.attrition_offers_config(max_rounds=2, momentum=0.15))
    p = game.params
    assert p.extended and p.step == 0.175
    assert p.local_strength("a", 0.5) == p.local_strength("b", 0.5) == 1.0  # the start
    # one step gained by A moves 0.15 x (1 + 1) of local strength from B to A
    assert p.local_strength("a", 0.675) == pytest.approx(1.3)
    assert p.local_strength("b", 0.675) == pytest.approx(0.7)
    assert p.local_strength("b", 0.5 + 0.175 * 4) == pytest.approx(0.1)  # its floor
    state = play(game, ["strong", "strong"])
    choices = {"a1": "low", "b1": "low"}
    assert state.win_probability(choices, control=0.675) > state.win_probability(choices)
    plain = attrition.build(configs.attrition_offers_config(max_rounds=2))
    assert plain.params.local_strength("a", 0.9) == 1.0  # without momentum, fixed


def test_an_unseen_type_is_known_to_no_one_and_learned_from_battles() -> None:
    game = attrition.build(configs.attrition_offers_config(max_rounds=3, unseen=(0.5, 0.5)))
    state = play(game, ["strong", "weak"])  # B is weak, and no one is told
    for role in ("a1", "b1"):
        assert not any(t.startswith("type:b=") for t in state.information_state(role)[1])
    assert state.observation("b1")[10] == 0.5  # its own strength feature stays neutral
    policy = AttritionBeliefPolicy(game, temperature=0.1)
    index = game.params.holders.index("b")

    def weak(view: str) -> float:
        return sum(q for cell, q in policy.joint(state, view).items() if cell[index] == "weak")

    before = {view: weak(view) for view in ("a", "b")}
    assert before == {"a": pytest.approx(0.5), "b": pytest.approx(0.5)}
    for action in ("no_offer", "low", "low", "a_gains"):  # B loses a battle
        state.apply_action(action)
    for view in ("a", "b"):
        assert weak(view) > before[view]  # both sides learn that B is weak


def test_a_side_acts_on_its_own_belief_about_an_unseen_type() -> None:
    """A side sure it is strong chooses as if it knew it; the other side's view is the same."""

    unseen = attrition.build(configs.attrition_offers_config(max_rounds=2, unseen=(0.5, 1.0)))
    known = configs.attrition_offers_config(max_rounds=2)
    known["types"]["b"] = {
        "levels": {
            "strong": {"weight": 0.5, "strength": 1.6},
            "weak": {"weight": 0.5, "strength": 0.6},
        }
    }
    seen = attrition.build(known)
    choices = []
    for game in (unseen, seen):
        state = play(game, ["strong", "strong", "no_offer", "low"])
        assert state.current_player() == "b1"
        choices.append(AttritionBeliefPolicy(game, 0.1).action_probabilities(state))
    for action, p in choices[1].items():
        assert choices[0][action] == pytest.approx(p)


def test_a_direct_cost_is_charged_and_deters_direct_effort() -> None:
    direct = {}
    for extra in (0.0, 2.0):
        config = configs.attrition_offers_config(max_rounds=2, red_line=True)
        config["sponsors"]["b1"]["direct_cost"] = extra
        config.pop("red_lines")
        game = attrition.build(config)
        state = play(game, ["strong", "strong", "no_offer", "low"])
        assert state.current_player() == "b1"
        direct[extra] = AttritionBeliefPolicy(game, 0.1).action_probabilities(state)["direct"]
        cost = state.cost["b1"]
        state.apply_action("direct")
        assert state.cost["b1"] == pytest.approx(cost + 0.6 + extra)  # effort cost plus extra
    assert direct[2.0] < direct[0.0]


def test_a_proxy_gain_accrues_only_while_an_ally_fights_direct() -> None:
    config = configs.attrition_offers_config(max_rounds=2, consent=True, entrant=True, proxy=0.5)
    game = attrition.build(config)
    costs = {}
    for c1 in ("direct", "high"):
        for b2 in ("low", "withdraw"):
            state = play(game, ["strong", "strong", "committed", "no_offer", "low", "low", b2, c1])
            costs[(c1, b2)] = state.cost["b2"]
    assert costs[("direct", "low")] == pytest.approx(0.1 - 0.5)  # its low effort, less the gain
    assert costs[("high", "low")] == pytest.approx(0.1)
    assert costs[("direct", "withdraw")] == costs[("high", "withdraw")] == 0.0  # out of the war


def test_a_consent_role_with_a_proxy_gain_holds_out() -> None:
    """With an ally fighting direct, a consent role that gains from the war going on
    vetoes a settlement it would otherwise ratify."""

    veto = {}
    for gain in (0.0, 2.0):
        config = configs.attrition_offers_config(
            max_rounds=2, consent=True, entrant=True, proxy=gain
        )
        game = attrition.build(config)
        state = play(
            game,
            ["strong", "strong", "committed", "no_offer", "low", "low", "low", "direct", "hold"],
        )
        state.apply_action("offer_050")
        state.apply_action("accept")  # the new lead, c1, responds
        assert (state.stage, state.current_player()) == ("ratify", "b2")
        policy = AttritionBeliefPolicy(game, temperature=0.1)
        veto[gain] = policy.action_probabilities(state)["veto"]
    assert veto[2.0] > 0.9 and veto[2.0] > veto[0.0]


def _drop_refused(config: dict[str, Any]) -> None:
    config["approval"] = ["b1"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["offers"].update(grid=[0.0, 0.5]),
        lambda c: c["offers"].update(grid=[0.5, 0.3]),
        lambda c: c["offers"].update(proposer="c"),
        lambda c: c["offers"].update(consent=["a1"]),
        lambda c: c["offers"].update(consent=["zz"]),
        lambda c: c["offers"].update(opens=-1),
        lambda c: c["types"]["a"].update(prior=1.5),
        lambda c: c["types"]["b"]["strong"].update(strength=0.0),
        lambda c: c["types"].update(c={"prior": 0.5}),
        lambda c: c.update(mobility={"space": 0.0, "sharpness": 2.0}),
        lambda c: c["red_lines"]["b1"].update(below=0.4),
        lambda c: c.update(red_lines={"zz": {"above": 0.5}}),
        lambda c: c["sponsors"]["b1"].update(direct=False),  # a red line needs a direct option
        lambda c: c.update(approval=["zz"]),
        lambda c: c.update(deployment=0.0),
        _drop_refused,  # approval without a value for "refused"
        lambda c: c["sponsors"]["a1"]["values"].update(refused=0.0),  # "refused" without approval
        lambda c: c.update(commitments={"zz": {"at_least": 0.5}}),
        lambda c: c.update(commitments={"a1": {"at_least": 1.0}}),
        lambda c: c.update(commitments={"a1": {"at_least": 0.6, "audience_cost": -1.0}}),
        lambda c: c.update(commitments={"a1": {"at_least": 0.6}}) or c.pop("offers"),
        lambda c: c["types"].update(a={"levels": {"only": {"weight": 1.0}}}),
        lambda c: c["types"].update(a={"levels": {"x": {"weight": -1.0}, "y": {"weight": 1.0}}}),
        lambda c: c["types"].update(a={"levels": {"x": {"weight": 0.0}, "y": {"weight": 0.0}}}),
        lambda c: c["types"].update(a={"levels": {"x": {"weight": 1.0, "nerve": 2}, "y": {}}}),
        lambda c: c["types"].update(a={"levels": {"x": {"weight": 1.0, "strength": 0.0}}}),
        lambda c: c["types"].update(a={"levels": {"x": {"weight": 1.0}}, "prior": 0.5}),
        lambda c: c["types"].update(zz={"prior": 0.5}),  # neither a side nor a sponsor
        lambda c: c["types"].update(
            a={"levels": {"x": {"weight": 1.0, "belief": 0.0}, "y": {"weight": 1.0}}}
        ),
        lambda c: c.update(lead_on_entry={"a1": "b"}),  # not a sponsor of that side
        lambda c: c.update(lead_on_entry={"zz": "b"}),
        lambda c: c["sponsors"]["b1"].update(proxy_gain=-0.1),
        lambda c: c.update(deployment={"zz": 0.5}),
        lambda c: c["sponsors"]["b1"].update(withdraw_cost=-1.0),
        lambda c: c["sponsors"]["b1"].update(direct_cost=-1.0),
        lambda c: c.update(momentum=-0.1),
        lambda c: c["types"]["b"].update(own_belief=0.5),  # not a level key
        lambda c: c["types"].update(
            a={"levels": {"x": {"weight": 1.0, "own_belief": 1.0}, "y": {"weight": 1.0}}}
        ),  # own_belief needs unseen
        lambda c: c["types"].update(
            c1={"private": True, "unseen": True, "levels": {"x": {"weight": 1.0}, "y": {}}}
        ),
        lambda c: c.update(commitments={"a1": {"at_least": 0.6, "scale": -0.5}}),
        lambda c: c["types"]["a"].update(private=True),  # sides cannot hide from themselves
        lambda c: c["types"].update(
            a={"levels": {"x": {"weight": 1.0, "proxy": 0.0}, "y": {"weight": 1.0}}}
        ),
        lambda c: c.update(deployment={"a1": 0.0}),
    ],
)
def test_validation(mutate: Callable[[dict[str, Any]], None]) -> None:
    config = configs.attrition_offers_config(red_line=True)
    mutate(config)
    with pytest.raises(ConfigError):
        attrition.build(config)


def test_battles_update_beliefs_by_bayes_rule() -> None:
    game = attrition.build(configs.attrition_offers_config(max_rounds=2))
    policy = AttritionBeliefPolicy(game)
    state = play(game, ["strong", "strong", "no_offer", "low", "low"])
    assert state.stage == "battle"
    before = policy.joint(state)
    likelihood = {
        cell: dict(state.battle_outcomes(state.choices, {"a": cell[0], "b": cell[1]}))["a_gains"]
        for cell in before
        if cell[0] is not None and cell[1] is not None
    }
    # by hand: E_A = 2 and E_B = 2 x strength (1.6 strong, 0.6 weak); P(move) = 1/(1 + (T/3)^2)
    assert likelihood[("strong", "strong")] == pytest.approx(2 / 5.2 / (1 + (5.2 / 3) ** 2))
    assert likelihood[("weak", "weak")] == pytest.approx(2 / 3.2 / (1 + (3.2 / 3) ** 2))
    state.apply_action("a_gains")
    after = policy.joint(state)
    total = sum(before[cell] * likelihood[cell] for cell in before)
    for cell in before:
        assert after[cell] == pytest.approx(before[cell] * likelihood[cell] / total)
    strong_before = sum(q for cell, q in before.items() if cell[1] == "strong")
    assert policy.public_belief(state)["b"] < strong_before  # A's gain points to a weak B


def test_temperature_mixes_choices() -> None:
    game = attrition.build(configs.attrition_offers_config(max_rounds=2))
    state = play(game, ["strong", "weak"])
    exact = AttritionBeliefPolicy(game).action_probabilities(state)
    assert sorted(exact.values())[-1] == 1.0
    cold = AttritionBeliefPolicy(game, temperature=1e-4).action_probabilities(state)
    assert max(cold, key=lambda a: cold[a]) == max(exact, key=lambda a: exact[a])
    soft = AttritionBeliefPolicy(game, temperature=0.5).action_probabilities(state)
    assert sum(soft.values()) == pytest.approx(1.0) and min(soft.values()) > 0.0
    with pytest.raises(ConfigError):
        AttritionBeliefPolicy(game, temperature=-1.0)


@pytest.mark.parametrize(
    ("config", "bound"),
    [
        (configs.attrition_offers_config(), 0.15),
        (configs.attrition_offers_config(approval=True), 0.2),
        (configs.attrition_offers_config(max_rounds=2, opens=1), 0.2),
        (configs.attrition_offers_config(max_rounds=2, opens=1, commitments=True), 0.15),
        (configs.attrition_offers_config(max_rounds=2, opens=1, levels=True), 0.15),
        (configs.attrition_offers_config(entrant=True), 0.15),
        (configs.attrition_offers_config(consent=True, entrant=True, proxy=0.3), 0.2),
        (
            configs.attrition_offers_config(
                consent=True, red_line=True, proxy=0.3, hidden=(0.5, 0.5)
            ),
            0.2,
        ),
        (configs.attrition_offers_config(max_rounds=2, opens=1, momentum=0.15), 0.15),
        (configs.attrition_offers_config(max_rounds=2, opens=1, unseen=(0.5, 0.5)), 0.15),
    ],
)
def test_reference_play_is_hard_to_exploit_on_small_wars(
    config: dict[str, Any], bound: float
) -> None:
    """Not an equilibrium, but close: the policy scores 0.050, 0.075, 0.075, 0.105, 0.082,
    0.055, 0.108, 0.111, 0.105 and 0.085 on these wars, against 0.050, 0.057, 0.061,
    0.072, 0.074, 0.058, 0.117, 0.114, 0.105 and 0.071 for the exact logit QRE at the
    same temperature. tree_policy
    also checks that the policy reads only what the acting role can see."""

    game = attrition.build(config)
    tree, policy = tree_policy(game, AttritionBeliefPolicy(game, temperature=0.1))
    assert nash_conv(tree, policy).nash_conv < bound
    assert nash_conv(tree, uniform_policy(tree)).nash_conv > 0.8
