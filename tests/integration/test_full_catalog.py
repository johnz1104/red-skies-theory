"""End-to-end checks on the shipped 1945-1991 catalog."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from cold_war_sim.campaign import (
    POWERS,
    CampaignEngine,
    Catalog,
    ColdWarEnv,
    EngineConfig,
    EquilibriumPolicy,
    HistoricalPolicy,
    RandomPolicy,
    calendar,
    load_catalog,
    run_episode,
)
from cold_war_sim.campaign.assumptions import Sampling, check
from cold_war_sim.campaign.assumptions import render_markdown as render_register
from cold_war_sim.campaign.catalog_doc import render_markdown
from cold_war_sim.campaign.cli import main
from cold_war_sim.campaign.state import IDENTITIES, identity

DOCS = Path(__file__).resolve().parents[2] / "docs" / "EVENT_CATALOG.md"
REGISTER_DOC = Path(__file__).resolve().parents[2] / "docs" / "ASSUMPTIONS.md"


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return load_catalog()


@pytest.fixture(scope="module")
def historical_run(catalog: Catalog) -> CampaignEngine:
    engine = CampaignEngine(catalog, EngineConfig(chance_mode="historical"))
    run_episode(engine, HistoricalPolicy(), 0)
    return engine


def test_catalog_counts(catalog: Catalog) -> None:
    counts = catalog.counts()
    assert counts["major"] == 37  # Korea is one war (2026-10-11) with two Geneva variants (E31)
    assert counts["decision"] >= 1
    assert counts["cutscene"] >= 60
    assert counts["leadership"] >= 30
    families = {e.major.family for e in catalog.events.values() if e.major is not None}
    assert families == {"crisis", "bargaining", "intervention", "attrition", "alignment"}


def test_historical_replay_reaches_the_dissolution(historical_run: CampaignEngine) -> None:
    world = historical_run.world
    assert world.ended == "ussr_dissolved"
    assert calendar.fmt(world.quarter) == "1991-Q4"


@pytest.mark.parametrize(
    ("event_id", "date", "tolerance"),
    [
        ("prc_proclaimed", "1949-Q4", 2),
        ("korean_war", "1950-Q2", 2),
        ("cuban_missile_crisis", "1962-Q4", 2),
        ("afghanistan_decision", "1979-Q4", 2),
        ("ussr_dissolved", "1991-Q4", 2),
        ("soviet_atomic_bomb", "1949-Q3", 4),
        ("sputnik", "1957-Q4", 4),
        ("apollo_11", "1969-Q3", 4),
        ("prc_atomic_bomb", "1964-Q4", 4),
    ],
)
def test_historical_milestones(
    historical_run: CampaignEngine, event_id: str, date: str | None, tolerance: int
) -> None:
    world = historical_run.world
    assert event_id in world.fired, f"{event_id} never fired"
    if date is not None:
        assert abs(world.fired[event_id] - calendar.parse(date)) <= tolerance


def test_negotiated_milestones_end_on_time(historical_run: CampaignEngine) -> None:
    ends = {
        item["end"]: calendar.parse(item["q"]) for item in historical_run.s.trace if "end" in item
    }
    assert abs(ends["korean_war"] - calendar.parse("1953-Q3")) <= 2
    assert abs(ends["salt_i"] - calendar.parse("1972-Q2")) <= 2
    assert abs(ends["reykjavik_inf"] - calendar.parse("1987-Q4")) <= 2


def test_every_dated_event_fires_near_its_date(
    catalog: Catalog, historical_run: CampaignEngine
) -> None:
    world = historical_run.world
    late_or_missing = []
    for event in catalog.events.values():
        if event.historical_date is None:
            continue
        fired = world.fired.get(event.id)
        if fired is None or abs(fired - event.historical_date) > 2:
            late_or_missing.append(
                (event.id, calendar.fmt(event.historical_date), fired and calendar.fmt(fired))
            )
    assert not late_or_missing


def test_every_major_event_reaches_its_historical_outcome(
    catalog: Catalog, historical_run: CampaignEngine
) -> None:
    world = historical_run.world
    for event in catalog.events.values():
        if event.major is None or event.major.historical_outcome is None:
            continue
        assert world.outcomes.get(event.id) == event.major.historical_outcome, event.id


def test_tier_one_replay_matches_the_full_replay(
    catalog: Catalog, historical_run: CampaignEngine
) -> None:
    engine = CampaignEngine(catalog, EngineConfig(chance_mode="historical", playable_tier=1))
    sources: set[str] = set()
    run_episode(engine, HistoricalPolicy(), 0, on_decision=lambda e, r: sources.add(r.source))
    assert engine.trace_hash() == historical_run.trace_hash()
    playable = {e.id for e in catalog.events.values() if e.major is not None and e.major.tier == 1}
    playable |= {e.id for e in catalog.events.values() if e.decision and e.decision.tier == 1}
    assert sources - {"strategy"} <= playable


COMMUNIST_ONLY = (
    "indochina_war",
    "malayan_emergency",
    "geneva_conference",
    "taiwan_strait_1954",
    "taiwan_strait_1958",
    "vietnam_war",
    "sino_soviet_border_clash",
    "opening_to_china",
    "tiananmen",
    "angola",
    "sino_soviet_split",
    "sino_vietnamese_war",
)


def test_a_nationalist_china_skips_communist_only_events(catalog: Catalog) -> None:
    engine = CampaignEngine(catalog)
    for seed in range(40):
        result = run_episode(engine, RandomPolicy(), seed)
        if "kmt_holds_china" in engine.world.flags:
            break
    else:
        pytest.fail("no Nationalist victory in 40 random campaigns")
    world = engine.world
    handover = world.fired["ccp_defeated"]
    assert world.powers["PRC"].regime == "kmt" and identity(world, "PRC") == "ROC"
    assert world.powers["PRC"].leader in ("chiang_kai_shek", "chiang_ching_kuo", "lee_teng_hui")
    late = [e for e in COMMUNIST_ONLY if world.fired.get(e, -1) >= handover]
    assert not late
    assert engine.s.retired == {"PRC"}
    assert all(math.isfinite(v) for v in result.returns.values())


def test_random_rollouts_terminate_within_bounds(catalog: Catalog) -> None:
    engine = CampaignEngine(catalog)
    reasons = set()
    for seed in range(25):
        result = run_episode(engine, RandomPolicy(), seed)
        reasons.add(result.end_reason)
        assert all(math.isfinite(v) for v in result.returns.values())
        world = engine.world
        for region in world.regions.values():
            assert sum(region.influence.values()) <= 1.0 + 1e-9
            assert 0.0 <= region.stability <= 1.0
        for power in world.powers.values():
            assert 0.0 <= power.stability <= 1.0 and power.economy > 0.0
    assert reasons <= {"campaign_end", "ussr_dissolved", "nuclear_war"}


@pytest.mark.slow
def test_two_hundred_random_rollouts(catalog: Catalog) -> None:
    engine = CampaignEngine(catalog)
    for seed in range(200):
        result = run_episode(engine, RandomPolicy(), 1000 + seed)
        assert result.end_reason in {"campaign_end", "ussr_dissolved", "nuclear_war"}


def test_full_catalog_determinism_and_snapshots(catalog: Catalog) -> None:
    engine = CampaignEngine(catalog)
    first = run_episode(engine, RandomPolicy(), 77)
    second = run_episode(engine, RandomPolicy(), 77)
    assert first.trace_hash == second.trace_hash

    engine.reset(5)
    rng = np.random.default_rng(5)
    policy = RandomPolicy()
    for _ in range(400):
        request = engine.pending()
        assert request is not None
        engine.act(policy.choose(engine, request, rng))
    snapshot = engine.snapshot()
    branch_rng = np.random.default_rng(9)
    while not engine.done:
        request = engine.pending()
        assert request is not None
        engine.act(policy.choose(engine, request, branch_rng))
    expected = engine.trace_hash()
    engine.restore(snapshot)
    branch_rng = np.random.default_rng(9)
    while not engine.done:
        request = engine.pending()
        assert request is not None
        engine.act(policy.choose(engine, request, branch_rng))
    assert engine.trace_hash() == expected


@pytest.mark.slow  # reference play in Korea's war (E38)
def test_equilibrium_policy_plays_a_full_campaign(catalog: Catalog) -> None:
    engine = CampaignEngine(catalog, EngineConfig(npc_policy="equilibrium"))
    result = run_episode(engine, EquilibriumPolicy(), 3)
    assert result.end_reason in {"campaign_end", "ussr_dissolved", "nuclear_war"}


def test_aec_environment_on_the_full_catalog(catalog: Catalog) -> None:
    env = ColdWarEnv(catalog)
    env.reset(seed=2)
    rng = np.random.default_rng(2)
    seen = set()
    while env.agents:
        agent = env.agent_selection
        _, _, terminated, truncated, _ = env.last()
        if terminated or truncated:
            env.step(None)
            continue
        seen.add(agent)
        env.step(int(rng.choice(np.flatnonzero(env.action_mask(agent)))))
    assert set(POWERS) <= seen <= set(IDENTITIES)


def test_generated_catalog_doc_is_current(catalog: Catalog) -> None:
    if not DOCS.exists():  # the generated docs are not published yet
        pytest.skip("no docs/EVENT_CATALOG.md in this checkout")
    assert DOCS.read_text(encoding="utf-8") == render_markdown(catalog), (
        "run: python -m cold_war_sim.campaign catalog --markdown --output docs/EVENT_CATALOG.md"
    )


def test_the_assumptions_register_covers_every_number(catalog: Catalog) -> None:
    # coverage, entry values, ranges, and the catalog compiling (with every validation)
    # with every live number at the low end of its range, then at the high end (E42)
    assert check(catalog) == []


def test_generated_assumptions_doc_is_current(catalog: Catalog) -> None:
    if not REGISTER_DOC.exists():  # the generated docs are not published yet
        pytest.skip("no docs/ASSUMPTIONS.md in this checkout")
    assert REGISTER_DOC.read_text(encoding="utf-8") == render_register(catalog), (
        "run: python -m cold_war_sim.campaign assumptions doc --output docs/ASSUMPTIONS.md"
    )


def test_sampled_campaigns_are_deterministic(catalog: Catalog) -> None:
    engine = CampaignEngine(catalog, EngineConfig(sampling=Sampling()))
    first = run_episode(engine, RandomPolicy(), 21)
    second = run_episode(engine, RandomPolicy(), 21)
    assert first.trace_hash == second.trace_hash and len(first.assumptions) > 1000
    assert first.end_reason in {"campaign_end", "ussr_dissolved", "nuclear_war"}
    tier1 = CampaignEngine(catalog, EngineConfig(sampling=Sampling(), playable_tier=1))
    fewer = run_episode(tier1, RandomPolicy(), 21).assumptions
    assert set(fewer) < set(first.assumptions)  # tier-2 configs run on rails, undrawn


def test_aec_environment_with_sampling(catalog: Catalog) -> None:
    env = ColdWarEnv(catalog, EngineConfig(sampling=Sampling()))
    env.reset(seed=3)
    plain = ColdWarEnv(catalog)
    plain.reset(seed=3)
    assert env.observation_size == plain.observation_size  # the draw is never observed
    rng = np.random.default_rng(3)
    for _ in range(200):
        agent = env.agent_selection
        _, _, terminated, truncated, info = env.last()
        assert len(info["assumptions"]) == 64
        if terminated or truncated:
            env.step(None)
            continue
        env.step(int(rng.choice(np.flatnonzero(env.action_mask(agent)))))
    assert "assumptions" not in plain.last()[4]


@pytest.mark.slow  # reference play in Korea's war (E38)
def test_cli_commands(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["catalog", "--check"]) == 0
    assert "37 major" in capsys.readouterr().out
    assert main(["run", "--seed", "1", "--policy", "random", "--quiet"]) == 0
    assert main(["solve", "--event", "iran_crisis", "--iterations", "200"]) == 0
    assert "NashConv" in capsys.readouterr().out
    assert main(["goals", "show", "--every", "20"]) == 0
    assert "[Stalin]" in capsys.readouterr().out
    assert main(["goals", "derive", "--event", "cuban_missile_crisis"]) == 0
    assert "roles.USSR.values.compromise" in capsys.readouterr().out
    assert main(["goals", "check"]) == 0
    assert "chinese_civil_war" in capsys.readouterr().out  # the first event on derived payoffs
    assert main(["standing"]) == 0
    assert "sputnik" in capsys.readouterr().out
    assert main(["assumptions", "check", "--quick"]) == 0
    assert "ok" in capsys.readouterr().out
    assert main(["assumptions", "list", "--event", "suez_crisis", "--placeholder"]) == 0
    assert "stakes_beliefs" in capsys.readouterr().out
    assert main(["run", "--seed", "1", "--policy", "random", "--quiet", "--sample", "goals"]) == 0


def test_robust_runs_save_and_reload(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    saved = tmp_path / "runs.json"
    tiny = ["--draws", "2", "--seeds", "1", "--policy", "random", "--npc-policy", "random"]
    assert (
        main(
            ["assumptions", "robust", *tiny, "--no-confirm", "--workers", "1", "--save", str(saved)]
        )
        == 0
    )
    played = capsys.readouterr().out
    assert main(["assumptions", "robust", "--load", str(saved), "--no-confirm"]) == 0
    assert capsys.readouterr().out == played and "no_nuclear_war" in played


def test_korea_after_the_armistice(catalog: Catalog, historical_run: CampaignEngine) -> None:
    """The garrison stays to 1991 and no treaty is signed; once the troops leave, Kim may
    ask again only when the US is distracted, and a treaty ends the risk."""

    assert {"us_troops_korea", "korean_armistice"} <= historical_run.world.flags
    assert "korea_peace_treaty" not in historical_run.world.flags
    war = catalog.events["korean_war"]
    world = catalog.initial_world()
    world.quarter = calendar.parse("1966-Q1")
    world.flags |= {"korea_divided", "korean_armistice", "korean_war_fought"}
    world.flags -= {"us_troops_korea", "soviet_troops_korea"}
    assert not war.condition(world)  # no request without a distraction
    world.flags.add("kim_asks")
    assert war.condition(world)
    world.flags.add("korea_peace_treaty")
    assert not war.condition(world)


def test_every_tier_one_event_plays_as_a_scenario(catalog: Catalog) -> None:
    from cold_war_sim.campaign.scenario import ScenarioEnv, capture_entries

    tier1 = [e for e in catalog.major_ids if (m := catalog.events[e].major) and m.tier == 1]
    entries = capture_entries(catalog, tier1)
    assert set(entries) == set(tier1)
    for event_id, items in entries.items():
        (entry,) = items
        date = catalog.events[event_id].historical_date
        assert date is None or abs(entry.quarter - date) <= 4
    quick = {e: v for e, v in entries.items() if e not in SLOW_SCENARIOS}
    env = ScenarioEnv(quick, catalog)
    rng = np.random.default_rng(0)
    for event_id in sorted(quick):
        env.reset(1, event=event_id)
        assert env.agents
        while env.agents:
            agent = env.agent_selection
            _, _, term, trunc, _ = env.last()
            if term or trunc:
                env.step(None)
                continue
            env.step(int(rng.choice(np.flatnonzero(env.action_mask(agent)))))
        assert env.engine.s.scope_done
        assert env.outcome() is not None
        assert set(env.returns()) == set(env.roles)


# scenarios whose background wars make episodes slow on cold caches (E44)
SLOW_SCENARIOS = ("berlin_blockade", "chinese_civil_war", "korean_war")
