"""Single-event scenarios: entry capture, scoped play and the scenario environment."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from cold_war_sim.campaign import POWERS, Catalog
from cold_war_sim.campaign.decisions.policies import HistoricalPolicy, make_policy
from cold_war_sim.campaign.engine import core as engine_module
from cold_war_sim.campaign.engine.core import (
    CampaignEngine,
    EngineConfig,
    reference_tree_policy,
    run_episode,
)
from cold_war_sim.campaign.environments.scenario import (
    Entry,
    ScenarioEnv,
    capture_entries,
    load_entries,
    save_entries,
)
from cold_war_sim.families import build_family
from cold_war_sim.games import build_tree
from cold_war_sim.games.tree import expected_returns

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


def _play_out(env: ScenarioEnv, seed: int) -> dict[str, float]:
    """Uniform agents from a reset at *seed*; the rewards each agent collected."""

    env.reset(seed)
    totals = dict.fromkeys(env.agents, 0.0)
    rng = np.random.default_rng(seed)
    while env.agents:
        agent = env.agent_selection
        _, reward, term, trunc, _ = env.last()
        totals[agent] += reward
        if term or trunc:
            env.step(None)
            continue
        env.step(int(rng.choice(np.flatnonzero(env.action_mask(agent)))))
    return totals


@pytest.mark.parametrize("event", ["fx_civil_war", "fx_talks"])
def test_events_over_quarters_keep_sampled_returns_under_the_flag(
    fixture_catalog: Catalog, event: str
) -> None:
    # E52: chance-expected returns cover instant events only; the flag used to zero every
    # reward of a war or talks, which the engine does not count in expectation
    entries = capture_entries(fixture_catalog, [event], playable_tier=None)
    runs: dict[bool, list[tuple[dict[str, float], dict[str, float], dict[str, float]]]] = {}
    for flag in (False, True):
        config = EngineConfig(npc_policy="equilibrium", chance_expected=flag)
        env = ScenarioEnv(entries, fixture_catalog, config)
        runs[flag] = []
        for seed in range(6):
            totals = _play_out(env, seed)
            assert not env.expected
            assert totals == pytest.approx(env.returns())
            runs[flag].append((totals, env.returns(), env.outcome_masses()))
    assert runs[True] == runs[False]
    assert any(value != 0.0 for totals, _, _ in runs[False] for value in totals.values())


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


def test_entries_saved_before_module_consolidation_still_play(
    fixture_catalog: Catalog, tmp_path: Path
) -> None:
    events = (*EVENTS, "fx_talks")
    entries = capture_entries(fixture_catalog, events, playable_tier=None)
    data = pickle.dumps(entries, protocol=0)
    # Protocol 0 stores module and class names as text. Recreate the names used
    # by existing entry files while keeping their state fields unchanged.
    earlier_names = {
        ("campaign.world.state", "WorldState"): "campaign.state",
        ("campaign.environments.scenario", "Entry"): "campaign.scenario",
        ("families.attrition.game", "AttritionState"): "families.attrition.state",
        ("families.bargaining.game", "BargainingParams"): "families.bargaining.params",
        ("families.bargaining.game", "SideParams"): "families.bargaining.params",
        ("campaign.engine.state", "Phase"): "campaign.engine.config",
        ("families.alignment", "AlignmentParams"): "families.alignment.params",
        ("families.alignment", "Client"): "families.alignment.params",
        ("families.alignment", "Patron"): "families.alignment.params",
        ("families.alignment", "AlignmentState"): "families.alignment.game",
        ("families.intervention", "InterventionParams"): "families.intervention.params",
        ("families.intervention", "InterventionState"): "families.intervention.game",
        ("families.crisis.params", "CrisisRole"): "families.crisis.roles",
        ("families.crisis.params", "MoveOption"): "families.crisis.options",
        ("families.crisis.params", "SignalOption"): "families.crisis.options",
        ("families.crisis.params", "PartnerOption"): "families.crisis.options",
    }
    for (module, name), earlier in earlier_names.items():
        data = data.replace(
            f"ccold_war_sim.{module}\n{name}\n".encode(),
            f"ccold_war_sim.{earlier}\n{name}\n".encode(),
        )
    path = tmp_path / "earlier_entries.pkl"
    path.write_bytes(data)
    loaded = load_entries(path)
    original = ScenarioEnv(entries, fixture_catalog)
    restored = ScenarioEnv(loaded, fixture_catalog)
    for event in events:
        original.reset(7, event=event)
        restored.reset(7, event=event)
        assert _play(original, np.random.default_rng(8)) == _play(
            restored, np.random.default_rng(8)
        )
        assert original.engine.trace_hash() == restored.engine.trace_hash()
        assert original.returns() == restored.returns()


def test_entries_from_another_catalog_are_refused(fixture_catalog: Catalog) -> None:
    entries = capture_entries(fixture_catalog, ["fx_crisis"], playable_tier=None)
    assert all(e.catalog == fixture_catalog.digest for e in entries["fx_crisis"])
    ScenarioEnv(entries, fixture_catalog)
    stale = {"fx_crisis": [entries["fx_crisis"][0].replace(catalog="")]}
    with pytest.raises(ValueError, match="recapture"):
        ScenarioEnv(stale, fixture_catalog)
    old = {"fx_crisis": [entries["fx_crisis"][0].replace(format=1)]}  # before E52's layout
    with pytest.raises(ValueError, match="older code"):
        ScenarioEnv(old, fixture_catalog)


@pytest.mark.parametrize(
    "stale",
    [
        b"ccold_war_sim.families.crisis\nStep\n.",  # a class that no longer exists
        b"ccold_war_sim.families.crisis_beliefs\nCrisisBeliefPolicy\n.",  # a module that moved
    ],
)
def test_an_entry_file_from_older_code_asks_to_be_recaptured(stale: bytes, tmp_path: Path) -> None:
    # entries pickle classes as they were when saved; a file that names a class or module
    # the code no longer has must say what to do, not fail inside pickle
    path = tmp_path / "old_entries.pkl"
    path.write_bytes(stale)
    with pytest.raises(ValueError, match="recapture"):
        load_entries(path)


def _uniform_play(
    env: ScenarioEnv, event: str, seeds: range, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Each agent's return per episode under uniformly random agents."""

    returns: dict[str, list[float]] = {}
    for seed in seeds:
        env.reset(seed, event=event)
        totals = dict.fromkeys(env.agents, 0.0)
        while env.agents:
            agent = env.agent_selection
            _, reward, term, trunc, info = env.last()
            totals[agent] += reward
            assert 0.0 <= info["survival"] <= 1.0
            if term or trunc:
                env.step(None)
                continue
            assert info["survival"] > 0.0  # a decision on the sampled path has weight
            env.step(int(rng.choice(np.flatnonzero(env.action_mask(agent)))))
        assert totals == pytest.approx(env.returns())  # the rewards add up to the estimate
        assert sum(env.outcome_masses().values()) == pytest.approx(1.0)
        for agent, value in env.returns().items():
            returns.setdefault(agent, []).append(value)
    return {a: np.asarray(v) for a, v in returns.items()}


def _uniform_value(catalog: Catalog, entry: Entry) -> dict[str, float]:
    """The exact expected return in goal units: agents uniform, reference play elsewhere."""

    event = entry.event_id
    engine = CampaignEngine(catalog, EngineConfig(npc_policy="equilibrium", scope=event))
    engine.restore(entry.state)
    major = catalog.events[event].major
    assert major is not None
    game = build_family(major.family, engine.s.active[event].config)
    tree = build_tree(game.new_initial_state())
    reference = reference_tree_policy(game, tree, 300, 0.1)
    joint = [
        [1.0 / len(info.actions)] * len(info.actions)
        if tree.roles[info.role] in POWERS
        else list(reference[k])
        for k, info in enumerate(tree.infosets)
    ]
    kappa = catalog.goals.kappa
    return {r: v / kappa for r, v in expected_returns(tree, joint).items() if r in POWERS}


def test_chance_expected_returns_are_unbiased_with_less_variance(
    fixture_catalog: Catalog,
) -> None:
    # E50: chance's stretches counted in expectation, the sampled path weighted
    event = "fx_uprising"
    entries = capture_entries(fixture_catalog, [event], playable_tier=None)
    exact = _uniform_value(fixture_catalog, entries[event][0])
    seeds = range(400)
    spread = {}
    for on in (False, True):
        env = ScenarioEnv(
            entries, fixture_catalog, EngineConfig(npc_policy="equilibrium", chance_expected=on)
        )
        returns = _uniform_play(env, event, seeds, np.random.default_rng(5))
        for agent, values in returns.items():
            error = values.std(ddof=1) / np.sqrt(len(values))
            assert abs(values.mean() - exact[agent]) < 3 * error
        spread[on] = {a: float(v.std()) for a, v in returns.items()}
    for agent, sd in spread[True].items():
        assert sd < 0.7 * spread[False][agent]


def test_chance_expected_episodes_are_reproducible(fixture_catalog: Catalog) -> None:
    entries = capture_entries(fixture_catalog, ["fx_crisis"], playable_tier=None)
    config = EngineConfig(npc_policy="equilibrium", chance_expected=True)
    env = ScenarioEnv(entries, fixture_catalog, config)
    runs = []
    for _ in range(2):
        returns = _uniform_play(env, "fx_crisis", range(3, 6), np.random.default_rng(2))
        runs.append(({a: list(v) for a, v in returns.items()}, env.engine.trace_hash()))
    assert runs[0] == runs[1]
    with pytest.raises(ValueError, match="sampled chance"):
        EngineConfig(chance_mode="historical", chance_expected=True)


def test_a_stretch_over_the_node_cap_is_sampled_as_without_the_flag(
    fixture_catalog: Catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(engine_module, "NATURE_NODE_CAP", 0)  # every stretch too large
    entries = capture_entries(fixture_catalog, ["fx_uprising"], playable_tier=None)
    runs = []
    for on in (False, True):
        config = EngineConfig(npc_policy="equilibrium", chance_expected=on)
        env = ScenarioEnv(entries, fixture_catalog, config)
        returns = _uniform_play(env, "fx_uprising", range(20), np.random.default_rng(7))
        runs.append(({a: list(v) for a, v in returns.items()}, env.engine.trace_hash()))
    assert runs[0] == runs[1]
