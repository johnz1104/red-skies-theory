"""Single-event scenarios: entry capture, scoped play and the scenario environment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cold_war_sim.campaign import Catalog
from cold_war_sim.campaign.engine import CampaignEngine, EngineConfig, run_episode
from cold_war_sim.campaign.policies import HistoricalPolicy, make_policy
from cold_war_sim.campaign.scenario import (
    ScenarioEnv,
    capture_entries,
    load_entries,
    save_entries,
)

EVENTS = ("fx_civil_war", "fx_crisis", "fx_alignment", "fx_canal")


def _play(env: ScenarioEnv, rng: np.random.Generator) -> list[str]:
    """Random legal actions to the end; the sources of the requests agents saw."""

    sources = []
    while env.agents:
        agent = env.agent_selection
        _, _, term, trunc, _ = env.last()
        if term or trunc:
            env.step(None)
            continue
        request = env.engine.pending()
        assert request is not None
        sources.append(request.source)
        env.step(int(rng.choice(np.flatnonzero(env.action_mask(agent)))))
    return sources


def test_capture_leaves_the_replay_unchanged(fixture_catalog: Catalog) -> None:
    plain = CampaignEngine(fixture_catalog, EngineConfig(chance_mode="historical"))
    first = run_episode(plain, make_policy("historical"), 0).trace_hash
    capturing = CampaignEngine(
        fixture_catalog, EngineConfig(chance_mode="historical", capture=frozenset(EVENTS))
    )
    assert run_episode(capturing, make_policy("historical"), 0).trace_hash == first
    assert set(capturing.captured) == set(EVENTS)
    for event_id, state in capturing.captured.items():
        assert event_id in state.active
        assert state.active[event_id].started == state.world.quarter


def test_a_restored_entry_continues_like_the_replay(fixture_catalog: Catalog) -> None:
    config = EngineConfig(chance_mode="historical", capture=frozenset({"fx_crisis"}))
    engine = CampaignEngine(fixture_catalog, config)
    expected = run_episode(engine, make_policy("historical"), 0).trace_hash
    entry = engine.captured["fx_crisis"]
    resumed = CampaignEngine(fixture_catalog, EngineConfig(chance_mode="historical"))
    resumed.restore(entry)
    resumed.resume()
    policy, rng = HistoricalPolicy(), np.random.default_rng(0)
    while not resumed.done:
        request = resumed.pending()
        assert request is not None
        resumed.act(policy.choose(resumed, request, rng))
    assert resumed.trace_hash() == expected


def test_scoped_agents_play_only_the_event_and_stop_when_it_resolves(
    fixture_catalog: Catalog,
) -> None:
    entries = capture_entries(fixture_catalog, EVENTS, playable_tier=None)
    env = ScenarioEnv(entries, fixture_catalog, EngineConfig(npc_policy="equilibrium"))
    rng = np.random.default_rng(0)
    for event_id in EVENTS:
        for seed in range(3):
            env.reset(seed, event=event_id)
            agents = list(env.agents)
            assert agents and all(
                env.roles[a] in ("USA", "USSR", "PRC", "UK", "FRA") for a in agents
            )
            sources = _play(env, rng)
            assert set(sources) <= {event_id}
            assert env.engine.done and env.engine.s.scope_done
            outcome = env.outcome()
            assert outcome is not None
            returns = env.engine.s.last_returns[event_id]
            kappa = fixture_catalog.goals.kappa
            assert env.returns() == pytest.approx(
                {a: returns[env.roles[a]] / kappa for a in agents}
            )


def test_rewards_arrive_at_the_end_as_returns(fixture_catalog: Catalog) -> None:
    entries = capture_entries(fixture_catalog, ["fx_crisis"], playable_tier=None)
    env = ScenarioEnv(entries, fixture_catalog, EngineConfig(npc_policy="equilibrium"))
    env.reset(4)
    totals = dict.fromkeys(env.agents, 0.0)
    rng = np.random.default_rng(4)
    while env.agents:
        agent = env.agent_selection
        _, reward, term, trunc, _ = env.last()
        totals[agent] += reward
        if term or trunc:
            env.step(None)
            continue
        env.step(int(rng.choice(np.flatnonzero(env.action_mask(agent)))))
    assert totals == pytest.approx(env.returns())


def test_seeds_reseed_the_event(fixture_catalog: Catalog) -> None:
    entries = capture_entries(fixture_catalog, ["fx_civil_war"], playable_tier=None)
    env = ScenarioEnv(entries, fixture_catalog, EngineConfig(npc_policy="equilibrium"))
    histories = set()
    for seed in range(12):
        env.reset(seed)
        while env.agents:
            _, _, term, trunc, _ = env.last()
            if term or trunc:
                env.step(None)
                continue
            request = env.engine.pending()
            assert request is not None
            env.step(request.index_of("high"))  # the same choices every episode: fight on
        histories.add((env.outcome(), env.engine.world.quarter))
    assert len(histories) > 1  # chance differs between seeds from one entry
    env.reset(3)
    first = env.engine.trace_hash()
    env.reset(3)
    assert env.engine.trace_hash() == first


def test_entries_pickle(fixture_catalog: Catalog, tmp_path: Path) -> None:
    entries = capture_entries(fixture_catalog, ["fx_crisis", "fx_canal"])
    path = tmp_path / "entries.pkl"
    save_entries(path, entries)
    loaded = load_entries(path)
    assert {k: [e.label for e in v] for k, v in loaded.items()} == {
        k: [e.label for e in v] for k, v in entries.items()
    }
    campaign = capture_entries(
        fixture_catalog, ["fx_crisis"], source="campaign", seeds=range(2), playable_tier=None
    )
    assert all(e.source == "campaign" for e in campaign["fx_crisis"])
