from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from cold_war_sim.campaign import (
    MAX_OPTIONS,
    POWERS,
    CampaignEngine,
    Catalog,
    CatalogError,
    ColdWarEnv,
    EngineConfig,
    HistoricalPolicy,
    RandomPolicy,
    calendar,
    load_catalog,
    run_episode,
)
from cold_war_sim.campaign.director import run_director
from cold_war_sim.campaign.engine import Phase

from .conftest import FIXTURE_EVENTS


def test_random_rollouts_reach_the_end(fixture_catalog: Catalog) -> None:
    engine = CampaignEngine(fixture_catalog)
    for seed in range(5):
        result = run_episode(engine, RandomPolicy(), seed)
        assert result.final_quarter == "1991-Q4" or result.end_reason == "nuclear_war"
        assert all(math.isfinite(v) for v in result.returns.values())
        for region in engine.world.regions.values():
            assert sum(region.influence.values()) <= 1.0 + 1e-9
            assert 0.0 <= region.stability <= 1.0


def test_same_seed_same_trace(fixture_catalog: Catalog) -> None:
    engine = CampaignEngine(fixture_catalog)
    a = run_episode(engine, RandomPolicy(), 11)
    b = run_episode(engine, RandomPolicy(), 11)
    c = run_episode(engine, RandomPolicy(), 12)
    assert a.trace_hash == b.trace_hash
    assert a.returns == b.returns
    assert a.trace_hash != c.trace_hash


def test_hazard_draws_do_not_depend_on_policy(fixture_catalog: Catalog) -> None:
    engine = CampaignEngine(fixture_catalog)
    run_episode(engine, RandomPolicy(), 5)
    random_quarter = engine.world.fired.get("fx_hazard")
    run_episode(engine, HistoricalPolicy(), 5)
    assert engine.world.fired.get("fx_hazard") == random_quarter


def test_cascade_excludes_and_leadership(fixture_catalog: Catalog) -> None:
    engine = CampaignEngine(fixture_catalog, EngineConfig(chance_mode="historical"))
    run_episode(engine, HistoricalPolicy(), 0)
    world = engine.world
    assert world.fired["fx_telegram"] == world.fired["fx_cascade"] == calendar.parse("1946-Q1")
    assert "variant_a" in world.flags and "variant_b" not in world.flags
    assert "fx_variant_b" not in world.fired
    assert world.powers["USA"].leader == "eisenhower"


def test_historical_outcomes_follow_the_catalog(fixture_catalog: Catalog) -> None:
    engine = CampaignEngine(fixture_catalog, EngineConfig(chance_mode="historical"))
    result = run_episode(engine, HistoricalPolicy(), 0)
    assert result.outcomes["fx_crisis"] == "b_prevails"
    assert result.outcomes["fx_uprising"] == "crushed"
    assert result.outcomes["fx_canal"] == "b_prevails"
    assert result.outcomes["fx_civil_war"] == "b_withdrew"
    assert engine.world.powers["PRC"].status == "active"
    assert {
        "crisis_won",
        "ultimatum_issued",
        "eastern_bloc_consolidated",
        "uprising_crushed",
    } <= engine.world.flags


def test_events_above_the_playable_tier_run_on_rails(fixture_catalog: Catalog) -> None:
    engine = CampaignEngine(fixture_catalog, EngineConfig(playable_tier=1))
    sources: set[str] = set()
    for seed in range(3):
        run_episode(engine, RandomPolicy(), seed, on_decision=lambda e, r: sources.add(r.source))
        assert engine.world.outcomes["fx_canal"] == "b_prevails"
    assert "fx_canal" not in sources
    assert "fx_uprising" in sources
    with pytest.raises(ValueError, match="playable_tier"):
        EngineConfig(playable_tier=0)


def test_on_rails_replay_matches_the_historical_replay(fixture_catalog: Catalog) -> None:
    full = CampaignEngine(fixture_catalog, EngineConfig(chance_mode="historical"))
    railed = CampaignEngine(
        fixture_catalog, EngineConfig(chance_mode="historical", playable_tier=1)
    )
    a = run_episode(full, HistoricalPolicy(), 0)
    b = run_episode(railed, HistoricalPolicy(), 0)
    assert a.trace_hash == b.trace_hash
    assert b.decisions < a.decisions


def test_concurrent_events_share_a_quarter(fixture_catalog: Catalog) -> None:
    engine = CampaignEngine(fixture_catalog)
    run_episode(engine, HistoricalPolicy(), 3)
    starts = {item["start"]: item["q"] for item in engine.s.trace if "start" in item}
    assert starts["fx_uprising"] == starts["fx_canal"] == "1956-Q4"
    actors = {item["r"] for item in engine.s.trace if item.get("e") in ("fx_uprising", "fx_canal")}
    assert {"USSR", "USA", "UK", "FRA", "EGY"} <= actors


def test_annual_cadence_repeats_choices_between_first_quarters(fixture_catalog: Catalog) -> None:
    engine = CampaignEngine(fixture_catalog)
    run_episode(engine, RandomPolicy(), 4)
    decided = [item for item in engine.s.trace if item.get("e") == "fx_civil_war"]
    quarters = {item["q"] for item in decided}
    # Only the opening quarter and first quarters of later years carry new decisions.
    assert all(q == "1946-Q3" or q.endswith("Q1") for q in quarters)


def test_snapshot_restore_continues_identically(fixture_catalog: Catalog) -> None:
    engine = CampaignEngine(fixture_catalog)
    engine.reset(21)
    rng = np.random.default_rng(1)
    policy = RandomPolicy()
    for _ in range(150):
        request = engine.pending()
        assert request is not None
        engine.act(policy.choose(engine, request, rng))
    snapshot = engine.snapshot()
    rng_state = copy.deepcopy(rng)

    def finish(eng: CampaignEngine, r: np.random.Generator) -> str:
        while not eng.done:
            req = eng.pending()
            assert req is not None
            eng.act(policy.choose(eng, req, r))
        return eng.trace_hash()

    first = finish(engine, rng)
    engine.restore(snapshot)
    assert finish(engine, rng_state) == first


def test_strategy_commits_are_hidden_until_resolve(fixture_catalog: Catalog) -> None:
    engine = CampaignEngine(fixture_catalog)
    engine.reset(0)
    while not (
        engine.pending()
        and engine.pending().kind == "strategy_posture"
        and engine.pending().power == "USA"
    ):  # type: ignore[union-attr]
        request = engine.pending()
        assert request is not None
        engine.act(0)
    before = engine.world.powers["USA"].posture
    nuclear_index = engine.pending().index_of("nuclear")  # type: ignore[union-attr]
    engine.act(nuclear_index)
    assert engine.world.powers["USA"].posture == before  # not yet applied
    assert engine.s.commits["USA"]["posture"] == "nuclear"
    quarter = engine.world.quarter
    while engine.world.quarter == quarter and engine.s.phase in (Phase.STRATEGIC, Phase.SUBGAMES):
        engine.act(0)
    assert engine.world.quarter > quarter
    assert engine.world.powers["USA"].posture == "nuclear"


def test_observation_and_mask_shapes(fixture_catalog: Catalog) -> None:
    engine = CampaignEngine(fixture_catalog)
    engine.reset(0)
    for _ in range(40):
        request = engine.pending()
        assert request is not None
        for power in POWERS:
            vector = engine.observe(power)
            assert vector.shape == (len(engine.feature_names),)
            assert np.all(np.isfinite(vector))
        mask = engine.action_mask(request.power)
        assert mask.shape == (MAX_OPTIONS,) and mask.sum() == len(request.options)
        assert engine.action_mask("FRA" if request.power != "FRA" else "UK").sum() == 0
        engine.act(int(np.flatnonzero(mask)[-1]))
    assert "date" in engine.describe("USA")


def test_npc_powers_never_ask(fixture_catalog: Catalog) -> None:
    engine = CampaignEngine(fixture_catalog, EngineConfig(agents=("USA", "USSR")))
    engine.reset(2)
    asked = set()
    while not engine.done:
        request = engine.pending()
        assert request is not None
        asked.add(request.power)
        engine.act(0)
    assert asked <= {"USA", "USSR"}


def test_env_aec_loop(fixture_catalog: Catalog) -> None:
    env = ColdWarEnv(fixture_catalog)
    env.reset(seed=8)
    rng = np.random.default_rng(0)
    totals = dict.fromkeys(POWERS, 0.0)
    steps = 0
    while env.agents:
        agent = env.agent_selection
        obs, reward, terminated, truncated, _info = env.last()
        totals[agent] += reward
        assert obs.shape == (env.observation_size,)
        if terminated or truncated:
            env.step(None)
            continue
        mask = env.action_mask(agent)
        env.step(int(rng.choice(np.flatnonzero(mask))))
        steps += 1
    assert steps > 500
    assert any(abs(v) > 0 for v in totals.values())


def test_truncated_campaign(fixture_catalog: Catalog) -> None:
    env = ColdWarEnv(fixture_catalog, EngineConfig(end_quarter=calendar.parse("1950-Q4")))
    env.reset(seed=1)
    while env.agents:
        _, _, terminated, truncated, _ = env.last()
        if terminated or truncated:
            assert truncated
            env.step(None)
        else:
            env.step(0)
    assert env.engine.world.quarter == calendar.parse("1950-Q4")


def test_requested_events_bypass_window(fixture_catalog: Catalog) -> None:
    world = fixture_catalog.initial_world()
    result = run_director(fixture_catalog, world, 0, {"fx_war_after_crisis"})
    assert [e.id for e in result.majors] == ["fx_war_after_crisis"]


def _broken_catalog(tmp_path: Path, mutate: Any) -> None:
    doc = yaml.safe_load(FIXTURE_EVENTS.read_text())
    mutate(doc)
    path = tmp_path / "events.yaml"
    path.write_text(yaml.safe_dump(doc))
    load_catalog(event_files=[path])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["events"][0]["effects"].append(
            {"add": {"path": "powers.USA.morale", "value": 1}}
        ),
        lambda d: d["events"][1].update(when={"fired": "no_such_event"}),
        lambda d: d["events"][7]["outcomes"].pop("compromise"),
        lambda d: d["events"][7]["model"]["historical"]["choices"].update(USA=["nuke_them"]),
        lambda d: d["events"][7]["model"]["config"]["roles"]["USA"].update(side="west"),
        lambda d: d["events"].append(d["events"][0]),
        lambda d: d["events"][0].update(kind="major"),
        lambda d: d["events"][5]["effects"].append({"leader": "nobody"}),
    ],
)
def test_catalog_validation_errors(tmp_path: Path, mutate: Any) -> None:
    with pytest.raises(CatalogError):
        _broken_catalog(tmp_path, mutate)
