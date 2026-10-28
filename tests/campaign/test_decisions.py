"""Decision events, optional roles, re-firing wars and re-binding on a leader change."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from cold_war_sim.campaign import (
    CampaignEngine,
    Catalog,
    EngineConfig,
    EquilibriumPolicy,
    HistoricalPolicy,
    calendar,
    derive,
    load_catalog,
    run_episode,
)
from cold_war_sim.campaign.conditions import CompileContext, ConditionError, compile_condition
from cold_war_sim.campaign.derive import Valuer, decision_values

FIXTURE = Path(__file__).with_name("fixture_decisions.yaml")


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return load_catalog(event_files=[FIXTURE])


@pytest.fixture(scope="module")
def replay(catalog: Catalog) -> CampaignEngine:
    engine = CampaignEngine(
        catalog, EngineConfig(chance_mode="historical", end_quarter=calendar.parse("1949-Q4"))
    )
    run_episode(engine, HistoricalPolicy(), 0)
    return engine


def _trace(engine: CampaignEngine, key: str) -> list[dict[str, object]]:
    return [item for item in engine.s.trace if key in item]


def test_quarter_of_year_condition() -> None:
    condition = compile_condition({"quarter_of_year": [1, 3]}, CompileContext())
    world = load_catalog(event_files=[FIXTURE]).initial_world()
    world.quarter = calendar.parse("1947-Q3")
    assert condition(world)
    world.quarter = calendar.parse("1947-Q2")
    assert not condition(world)
    with pytest.raises(ConditionError):
        compile_condition({"quarter_of_year": 5}, CompileContext())


def test_a_decision_is_asked_each_year_and_applies_in_resolve(replay: CampaignEngine) -> None:
    decisions = [(item["q"], item["c"]) for item in _trace(replay, "decision")]
    assert decisions == [("1946-Q1", "keep_garrison"), ("1947-Q1", "withdraw")]
    assert replay.world.outcomes["fx_garrison"] == "withdraw"
    assert "fx_garrison_gone" in replay.world.flags


def test_an_agent_answers_a_decision(catalog: Catalog) -> None:
    engine = CampaignEngine(catalog, EngineConfig(agents=("USA",)))
    engine.reset(0)
    while True:
        request = engine.pending()
        assert request is not None
        if request.kind == "event_decision":
            break
        engine.act(0)
    assert request.source == "fx_garrison" and request.power == "USA"
    assert [o.id for o in request.options] == ["keep_garrison", "withdraw"]
    mask = engine.action_mask("USA")
    assert mask.sum() == 2
    engine.act(1)
    assert "fx_garrison_gone" in engine.world.flags or engine.s.decided == [
        ("fx_garrison", "withdraw")
    ]


def test_a_refused_war_returns_and_uses_the_next_historical_plan(replay: CampaignEngine) -> None:
    ends = [(item["q"], item["outcome"]) for item in _trace(replay, "outcome")]
    assert ends[0] == ("1947-Q2", "refused")
    assert ends[-1][1] == "a_victory"
    assert replay.world.outcomes["fx_war"] == "a_victory"
    starts = [item["q"] for item in _trace(replay, "start")]
    assert starts == ["1947-Q2", "1947-Q3"]


def test_optional_roles_join_only_when_their_condition_holds(catalog: Catalog) -> None:
    major = catalog.events["fx_war"].major
    assert major is not None
    world = catalog.initial_world()
    assert world.powers["PRC"].status != "active"
    without = major.bound_config(world)
    assert "PRC" not in without["sponsors"]
    assert without["leads"]["b"] == "USSR" and without["approval"] == ["USSR"]
    world.powers["PRC"].status = "active"
    with_prc = major.bound_config(world)
    assert "PRC" in with_prc["sponsors"] and with_prc["leads"]["b"] == "PRC"
    assert with_prc["sponsors"]["PRC"]["capacity"] == pytest.approx(1.0 + 0.5 * 0.8)


def test_a_new_leader_rebinds_a_running_war(catalog: Catalog) -> None:
    engine = CampaignEngine(catalog, EngineConfig(chance_mode="historical"))
    engine.reset(0)
    policy = HistoricalPolicy()
    import numpy as np

    rng = np.random.default_rng(0)
    seen: dict[str, float] = {}
    while not engine.done and engine.world.quarter <= calendar.parse("1948-Q1"):
        active = engine.s.active.get("fx_war")
        if active is not None:
            seen[calendar.fmt(engine.world.quarter)] = active.config["sponsors"]["USSR"][
                "cost_scale"
            ]
        request = engine.pending()
        assert request is not None
        engine.act(policy.choose(engine, request, rng))
    assert seen["1947-Q3"] == pytest.approx(2.0 - 2.0 * 0.8)  # Stalin
    assert seen["1948-Q1"] == pytest.approx(2.0 - 2.0 * 0.4)  # Malenkov, re-bound
    assert [item["q"] for item in _trace(engine, "rebind")] == ["1947-Q4"]


def test_decision_values_count_the_war_a_withdrawal_enables(catalog: Catalog) -> None:
    world = catalog.initial_world()
    world.quarter = calendar.parse("1947-Q1")
    event = catalog.events["fx_garrison"]
    values = decision_values(catalog, event, world)
    valuer = Valuer(catalog, event, world, catalog.goals)
    spec = event.decision
    assert spec is not None
    after = valuer.after([spec.options["withdraw"].effects], label="withdraw")
    goal_only = valuer.value(after, "USA")
    war = catalog.events["fx_war"]
    change = valuer._chance(war, after) - valuer._chance(war, world)
    assert change > 0.9  # the war becomes all but certain to start within the lookahead
    expected = derive._expected_value(catalog, war, after, "USA", catalog.goals)
    assert values["keep_garrison"] == pytest.approx(0.0)
    assert values["withdraw"] == pytest.approx(round(goal_only + change * expected, 2))
    quiet = dataclasses.replace(catalog.goals, lookahead_quarters=0)
    assert decision_values(catalog, event, world, quiet)["withdraw"] == pytest.approx(
        valuer.value(after, "USA")
    )


def test_equilibrium_play_answers_decisions(catalog: Catalog) -> None:
    engine = CampaignEngine(
        catalog, EngineConfig(npc_policy="equilibrium", end_quarter=calendar.parse("1948-Q4"))
    )
    result = run_episode(engine, EquilibriumPolicy(), 1)
    assert any("decision" in item for item in engine.s.trace)
    assert result.final_quarter == "1948-Q4"
