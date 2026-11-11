"""A role's observation is a function of its information state (E52).

A learner sees ``observation(role)``; tree scores evaluate one history per information
set. Both are sound only if every history in a set gives the role the same observation,
so nothing it may not know (another role's private report or type) leaks into it."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable
from typing import Any

import pytest

from cold_war_sim.families import build_family
from cold_war_sim.games import CHANCE, TERMINAL

from . import configs
from .test_bargaining_beliefs import MIXING

CASES: list[tuple[str, Any]] = [
    ("crisis", configs.signaling_crisis),  # A's private report about B
    ("crisis", configs.warning_crisis),
    ("crisis", configs.coalition_crisis),
    ("bargaining", lambda: configs.bargaining_config(horizon=2)),
    ("bargaining", lambda: MIXING),  # B's private type
    ("intervention", configs.intervention_config),
    ("alignment", configs.alignment_config),
    ("attrition", lambda: configs.attrition_config(max_rounds=2)),
    ("attrition", lambda: configs.attrition_offers_config(levels=True)),
    ("attrition", lambda: configs.attrition_offers_config(proxy=0.4, hidden=(0.5, 0.5))),
    ("attrition", lambda: configs.attrition_offers_config(unseen=(0.5, 0.5))),
]


def observations_by_information_set(game: Any) -> dict[tuple[str, Hashable], set[tuple]]:
    """Every decision history's observation, grouped by (role, information state)."""

    found: dict[tuple[str, Hashable], set[tuple]] = defaultdict(set)
    stack = [game.new_initial_state()]
    while stack:
        state = stack.pop()
        player = state.current_player()
        if player == TERMINAL:
            continue
        if player == CHANCE:
            actions = [a for a, p in state.chance_outcomes() if p > 0.0]
        else:
            actions = list(state.legal_actions())
            key = (player, state.information_state(player))
            found[key].add(tuple(state.observation(player)))
        for action in actions:
            child = state.clone()
            child.apply_action(action)
            stack.append(child)
    return found


@pytest.mark.parametrize(("family", "factory"), CASES)
def test_observations_depend_only_on_the_information_state(family: str, factory: Any) -> None:
    found = observations_by_information_set(build_family(family, factory()))
    split = [key for key, seen in found.items() if len(seen) > 1]
    assert found
    assert not split, f"{len(split)} of {len(found)} sets see more than one observation"


def test_a_private_report_reaches_only_its_observers() -> None:
    game = build_family("crisis", configs.signaling_crisis())
    found = observations_by_information_set(game)
    a_reports = {obs[8] for (role, _), seen in found.items() if role == "A" for obs in seen}
    b_reports = {obs[8] for (role, _), seen in found.items() if role == "B" for obs in seen}
    assert a_reports >= {0.0, 1.0}  # A reads its report
    assert b_reports == {0.5}  # B never sees what A was told


def test_intel_about_one_role_has_one_audience() -> None:
    config = configs.signaling_crisis()
    script = config["script"]
    script.insert(2, {"stage": "intel", "about": "B", "observers": ["B"], "accuracy": 0.6})
    with pytest.raises(Exception, match="two different audiences"):
        build_family("crisis", config)


def test_a_second_report_about_a_role_keeps_the_first() -> None:
    # Two reports about B (resolve, then readiness), both seen by A. The observation has
    # one report slot: it shows the first report A may see, as it does for reports about
    # different roles; the later one sits beside it in the state, not in its place.
    config = configs.signaling_crisis()
    intel = {"stage": "intel", "about": "B", "observers": ["A"], "accuracy": 0.75}
    config["script"][1:2] = [intel, {**intel, "dimension": "readiness"}]
    state = build_family("crisis", config).new_initial_state()
    reports = iter(("alarming", "reassuring"))
    while state.current_player() == CHANCE:
        actions = [a for a, p in state.chance_outcomes() if p > 0.0]
        state.apply_action(next(reports) if "alarming" in actions else actions[0])
    assert state.current_player() == "A"
    assert state.reports == {("B", "resolve"): "alarming", ("B", "readiness"): "reassuring"}
    assert state.observation("A")[8] == 1.0  # the first report: alarming
