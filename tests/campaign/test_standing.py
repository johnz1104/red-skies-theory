from __future__ import annotations

import pytest

from cold_war_sim.campaign import (
    CampaignEngine,
    Catalog,
    EngineConfig,
    HistoricalPolicy,
    RandomPolicy,
    run_episode,
    standing,
)
from cold_war_sim.games import Outcome


def test_the_vietnam_example_from_the_design() -> None:
    k = 32.0
    # GOALS.md section 8: the US side is expected to win (E = 0.8).
    assert standing.delta(k, 1.0, 0.0, 0.8) == pytest.approx(-0.8 * k)
    assert standing.delta(k, 0.3, 1.0, 0.2) == pytest.approx(0.24 * k)
    assert standing.delta(k, 0.0, 1.0, 0.2) == 0.0


def test_results_by_family() -> None:
    crisis = {"roles": {"USA": {"side": "a"}, "USSR": {"side": "b"}}}
    assert standing.results("crisis", crisis, Outcome("a_prevails")) == {"a": 1.0, "b": 0.0}
    assert standing.results("crisis", crisis, Outcome("nuclear_war")) is None
    talks = {"roles": {"A": "USA", "B": "USSR"}}
    agreed = standing.results("bargaining", talks, Outcome("agreement", (), {"share": 0.75}))
    assert agreed == {"a": 0.75, "b": 0.25}
    aid = {"patrons": {"USA": {}, "USSR": {}}}
    tags = ("UK:aligned_USA", "FRA:nonaligned", "offer:USA=aid")
    assert standing.results("alignment", aid, Outcome("aligned_USA", tags)) == {
        "USA": 0.75,
        "USSR": 0.25,
    }


def test_standing_moves_with_contests_but_not_the_reward(fixture_catalog: Catalog) -> None:
    engine = CampaignEngine(fixture_catalog, EngineConfig(chance_mode="historical"))
    result = run_episode(engine, HistoricalPolicy(), 0)
    start = fixture_catalog.goals.standing_start
    assert engine.s.standing_log
    assert any(abs(v - start) > 1e-9 for v in result.standing.values())
    events = {event for _, event, _ in engine.s.standing_log}
    assert "fx_crisis" in events  # b_prevails: the USA wins
    _, _, crisis = next(entry for entry in engine.s.standing_log if entry[1] == "fx_crisis")
    assert crisis["USA"] > 0 > crisis["USSR"]
    assert all("standing" not in str(item) for item in engine.s.trace)


def test_standing_does_not_change_rewards(fixture_catalog: Catalog) -> None:
    a = run_episode(CampaignEngine(fixture_catalog), RandomPolicy(), 4)
    original = fixture_catalog.goals.standing_k
    object.__setattr__(fixture_catalog.goals, "standing_k", 1000.0)
    try:
        b = run_episode(CampaignEngine(fixture_catalog), RandomPolicy(), 4)
    finally:
        object.__setattr__(fixture_catalog.goals, "standing_k", original)
    assert a.returns == b.returns and a.trace_hash == b.trace_hash
    assert a.standing != b.standing


def test_alignment_results_ignore_forbid_tags() -> None:
    config = {"patrons": {"USA": {}, "USSR": {}}}
    outcome = Outcome("aligned_USA", ("UK:aligned_USA", "USSR:forbade", "offer:USA=aid"), {})
    assert standing.results("alignment", config, outcome) == {"USA": 1.0, "USSR": 0.0}
