"""RL on scenarios: the PettingZoo adapter, tree scores and the PPO learner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("pettingzoo")

import torch
from pettingzoo.test import api_test

from cold_war_sim.campaign import Catalog
from cold_war_sim.campaign.engine import EngineConfig
from cold_war_sim.campaign.rl import evaluate, ppo
from cold_war_sim.campaign.rl.pz import CampaignAEC
from cold_war_sim.campaign.scenario import (
    Entry,
    ScenarioEnv,
    capture_entries,
    save_entries,
)
from cold_war_sim.families import build_family
from cold_war_sim.games import build_tree
from cold_war_sim.games.algorithms import nash_conv

TREE_EVENTS = ("fx_crisis", "fx_canal", "fx_uprising")
Entries = dict[str, list[Entry]]


class Uniform:
    def probabilities(self, ident: str, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return mask / mask.sum()


@pytest.fixture(scope="module")
def entries(fixture_catalog: Catalog) -> Entries:
    return capture_entries(fixture_catalog, TREE_EVENTS, playable_tier=None)


def _env(catalog: Catalog, entries: Entries, expected: bool = False) -> ScenarioEnv:
    config = EngineConfig(npc_policy="equilibrium", chance_expected=expected)
    return ScenarioEnv(entries, catalog, config)


# advisory: the Dict observation with an action mask is PettingZoo's own masked format,
# and agents are named by identity on purpose
@pytest.mark.filterwarnings("ignore:Observation space for each agent:UserWarning")
@pytest.mark.filterwarnings("ignore:We recommend agents to be named:UserWarning")
@pytest.mark.filterwarnings("ignore:Observation is not a NumPy array:UserWarning")
@pytest.mark.parametrize("expected", [False, True])
def test_the_adapter_passes_the_pettingzoo_api_test(
    fixture_catalog: Catalog, entries: Entries, expected: bool
) -> None:
    api_test(CampaignAEC(_env(fixture_catalog, entries, expected)), num_cycles=50)


def test_tokens_give_every_information_set_its_own_observation(
    fixture_catalog: Catalog, entries: Entries
) -> None:
    for entry in (items[0] for items in entries.values()):
        report = evaluate.tree_report(fixture_catalog, entry, Uniform())
        assert report.aliasing
        for sets, observations in report.aliasing.values():
            assert sets == observations


def test_path_regret_discounts_regret_off_the_qre_paths(
    fixture_catalog: Catalog, entries: Entries
) -> None:
    for event in ("fx_crisis", "fx_canal"):  # a sequentially rational profile: no regret
        exact = evaluate.tree_report(fixture_catalog, entries[event][0], None).scores["exact"]
        assert exact.credibility == pytest.approx(0.0, abs=1e-6)
        assert exact.path_regret == pytest.approx(0.0, abs=1e-6)
    # CFR+'s equilibrium of the uprising is unexploitable, but its worst regret sits at a
    # set the QRE seldom reaches, so the regret met along the QRE's paths is far smaller
    exact = evaluate.tree_report(fixture_catalog, entries["fx_uprising"][0], None).scores["exact"]
    assert exact.exploitability == pytest.approx(0.0, abs=1e-4)
    assert exact.credibility > 0.003
    assert exact.path_regret < 0.1 * exact.credibility


def test_reference_exploitability_matches_nash_conv(
    fixture_catalog: Catalog, entries: Entries
) -> None:
    from cold_war_sim.campaign.engine import CampaignEngine, reference_tree_policy

    entry = entries["fx_crisis"][0]
    report = evaluate.tree_report(fixture_catalog, entry, None)
    engine = CampaignEngine(fixture_catalog)
    engine.restore(entry.state)
    active = engine.s.active["fx_crisis"]
    game = build_family("crisis", active.config)
    tree = build_tree(game.new_initial_state())
    policy = reference_tree_policy(game, tree, 300, engine.config.baseline_temperature)
    direct = nash_conv(tree, policy).nash_conv / fixture_catalog.goals.kappa
    assert set(report.agent_roles) == set(tree.roles)  # both roles are powers here
    assert report.scores["reference"].exploitability == pytest.approx(direct, abs=1e-9)
    assert report.scores["exact"].exploitability <= report.scores["reference"].exploitability + 1e-6


def test_gae_on_one_terminal_reward() -> None:
    advantages = ppo.gae([0.0, 0.0, 1.0], [0.5, 0.5, 0.5], gamma=1.0, lam=1.0)
    assert advantages == pytest.approx([0.5, 0.5, 0.5])
    advantages = ppo.gae([0.0, 1.0], [0.2, 0.4], gamma=1.0, lam=0.5)
    assert advantages == pytest.approx([0.2 + 0.5 * 0.6, 0.6])


def test_gae_with_survival_weights() -> None:
    # every weight 1: the usual GAE
    plain = ppo.gae([0.0, 0.3, 1.0], [0.1, 0.4, 0.2], gamma=1.0, lam=0.9)
    ones = ppo.gae([0.0, 0.3, 1.0], [0.1, 0.4, 0.2], 1.0, 0.9, survival=[1.0, 1.0, 1.0])
    assert ones == pytest.approx(plain)
    # rewards weighted from the start: the second decision, reached at weight 0.5, sees
    # 0.3 / 0.5 = 0.6; the first sees the whole 0.3
    advantages = ppo.gae([0.0, 0.3], [0.1, 0.4], 1.0, 1.0, survival=[1.0, 0.5])
    assert advantages == pytest.approx([0.3 - 0.1, 0.6 - 0.4])


@pytest.mark.parametrize(
    ("objective", "expected"), [(o, False) for o in ppo.OBJECTIVES] + [("qre", True)]
)
def test_ppo_updates_are_reproducible(
    fixture_catalog: Catalog,
    entries: Entries,
    tmp_path: Path,
    objective: str,
    expected: bool,
) -> None:
    path = tmp_path / "entries.pkl"
    save_entries(path, entries)
    config = ppo.PPOConfig(
        entries=str(path),
        iterations=1,
        episodes=16,
        workers=1,
        seed=3,
        objective=objective,
        chance_expected=expected,
    )
    env = _env(fixture_catalog, entries, expected)

    def one_update() -> dict[str, np.ndarray]:
        learner = ppo.Learner.create(config, env.observation_size)
        policy = learner.policy()
        episodes = [
            ppo.play_episode(env, policy, s, np.random.default_rng(s + 7919)) for s in range(16)
        ]
        stats = learner.step(episodes)
        assert stats and all(np.isfinite(v) for s in stats.values() for v in s.values())
        return {
            k: v.detach().clone().numpy() for k, v in learner.models["USA"].state_dict().items()
        }

    first, second = one_update(), one_update()
    for key in first:
        assert np.array_equal(first[key], second[key])


def test_checkpoints_round_trip(tmp_path: Path) -> None:
    config = ppo.PPOConfig(entries="unused.pkl", seed=1)
    learner = ppo.Learner.create(config, 64)
    learner.obs_norms["USA"].update(np.random.default_rng(0).normal(size=(10, 64 - 16)))
    learner.save(tmp_path / "checkpoint.pt")
    loaded = ppo.Learner.load(tmp_path / "checkpoint.pt")
    assert loaded.config == config
    for key, value in learner.models["UK"].state_dict().items():
        assert torch.equal(value, loaded.models["UK"].state_dict()[key])
    assert np.allclose(loaded.obs_norms["USA"].mean, learner.obs_norms["USA"].mean)
    assert loaded.kappa == learner.kappa


def _bandit(objective: str, q: tuple[float, float], iterations: int = 150) -> np.ndarray:
    """Repeated updates on one decision with known action values; the final policy."""

    from cold_war_sim.campaign.observation import TOKEN_SLOTS

    torch.manual_seed(0)
    config = ppo.PPOConfig(entries="unused.pkl", objective=objective, lr=0.01, temperature=0.1)
    size = TOKEN_SLOTS + 4
    model = ppo.ActorCritic(size, 32)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    norm = ppo.RunningStats.of_size(4)
    generator = torch.Generator().manual_seed(0)
    rng = np.random.default_rng(0)
    obs = np.zeros((512, size), dtype=np.float32)
    mask = np.zeros((512, 20), dtype=np.int8)
    mask[:, :2] = 1
    values = np.asarray(q)

    def probabilities() -> np.ndarray:
        numeric, tokens = ppo.split(obs[:1])
        with torch.no_grad():
            logits, _ = model(
                torch.as_tensor(numeric), torch.as_tensor(tokens), torch.as_tensor(mask[:1])
            )
        result: np.ndarray = torch.softmax(logits, -1).numpy()[0, :2]
        return result

    for _ in range(iterations):
        p = probabilities()
        actions = rng.choice(2, size=512, p=p / p.sum())
        batch = ppo.Batch(
            obs,
            mask,
            actions,
            np.log(p[actions]),
            values[actions] - float(p @ values),  # exact advantages, in payoff units
            values[actions],
        )
        ppo.update(model, optimizer, batch, norm, config, generator)
    return probabilities()


def test_the_qre_objective_converges_to_the_logit_response() -> None:
    q = (0.0, 0.1)
    target = np.exp(np.asarray(q) / 0.1)
    target /= target.sum()  # softmax(Q / tau) = (0.269, 0.731)
    assert _bandit("qre", q) == pytest.approx(target, abs=0.05)
    # plain PPO normalizes advantages, so its entropy weight is no temperature: it all but
    # commits to the better action (E45)
    assert _bandit("ppo", q)[1] > 0.9


def test_the_qre_profile_is_at_distance_zero_from_itself(
    fixture_catalog: Catalog, entries: Entries
) -> None:
    report = evaluate.tree_report(fixture_catalog, entries["fx_crisis"][0], Uniform())
    assert report.scores["qre"].qre_distance == pytest.approx(0.0, abs=1e-12)
    assert report.scores["learned"].qre_distance is not None
    assert report.scores["learned"].qre_distance > 0.0


def test_the_temperature_cools_geometrically_to_its_target() -> None:
    config = ppo.PPOConfig(
        entries="unused.pkl", iterations=100, temperature=0.1, temperature_start=10.0, anneal=0.5
    )
    assert config.temperature_at(0) == pytest.approx(10.0)
    assert config.temperature_at(25) == pytest.approx(1.0)  # halfway in log space
    assert config.temperature_at(50) == pytest.approx(0.1)
    assert config.temperature_at(99) == pytest.approx(0.1)
    assert ppo.PPOConfig(entries="unused.pkl").temperature_at(7) == 0.1  # no schedule
    with pytest.raises(ValueError):
        ppo.PPOConfig(entries="unused.pkl", temperature=0.1, temperature_start=0.05)


def test_the_learning_rate_decays_after_cooling() -> None:
    config = ppo.PPOConfig(
        entries="unused.pkl",
        iterations=101,
        lr=1e-3,
        temperature_start=5.0,
        anneal=0.5,
        lr_final=1e-4,
    )
    assert config.lr_at(0) == pytest.approx(1e-3)
    assert config.lr_at(50) == pytest.approx(1e-3)  # constant while cooling
    assert config.lr_at(100) == pytest.approx(1e-4)
    assert ppo.PPOConfig(entries="unused.pkl").lr_at(40) == ppo.PPOConfig(entries="x").lr


def test_only_trainable_identities_learn(fixture_catalog: Catalog, entries: Entries) -> None:
    env = _env(fixture_catalog, entries)
    config = ppo.PPOConfig(entries="unused.pkl", seed=2, trainable=("USA",))
    learner = ppo.Learner.create(config, env.observation_size)
    before = {
        i: {k: v.clone() for k, v in m.state_dict().items()} for i, m in learner.models.items()
    }
    policy = learner.policy()
    episodes = [ppo.play_episode(env, policy, s, np.random.default_rng(s)) for s in range(24)]
    stats = learner.step(episodes)
    assert set(stats) <= {"USA"} and "USA" in stats
    changed = {
        i
        for i, m in learner.models.items()
        if any(not torch.equal(v, before[i][k]) for k, v in m.state_dict().items())
    }
    assert changed == {"USA"}


def test_frozen_identities_keep_their_normalizers(
    fixture_catalog: Catalog, entries: Entries
) -> None:
    # E52: best-response training moved the frozen opponents' observation normalizers,
    # so they played differently from the checkpoint they were scored against
    env = _env(fixture_catalog, entries)
    config = ppo.PPOConfig(entries="unused.pkl", seed=2, trainable=("USA",))
    learner = ppo.Learner.create(config, env.observation_size)
    before = {
        i: (learner.obs_norms[i].state(), learner.return_norms[i].state()) for i in ppo.IDENTITIES
    }
    policy = learner.policy()
    episodes = [ppo.play_episode(env, policy, s, np.random.default_rng(s)) for s in range(24)]
    learner.step(episodes)
    moved = {
        i
        for i in ppo.IDENTITIES
        if (learner.obs_norms[i].state(), learner.return_norms[i].state()) != before[i]
    }
    assert moved == {"USA"}


def test_league_snapshots_stay_as_they_were(fixture_catalog: Catalog, entries: Entries) -> None:
    # E52: snapshots were views of the live parameters, so every past opponent was the
    # current network
    env = _env(fixture_catalog, entries)
    learner = ppo.Learner.create(ppo.PPOConfig(entries="unused.pkl", seed=4), env.observation_size)
    snapshot = learner.weights()
    kept = {i: {k: v.copy() for k, v in w.items()} for i, w in snapshot.items()}
    policy = learner.policy()
    episodes = [ppo.play_episode(env, policy, s, np.random.default_rng(s)) for s in range(16)]
    learner.step(episodes)
    for ident, weights in snapshot.items():
        for key, value in weights.items():
            assert np.array_equal(value, kept[ident][key])
    after = learner.weights()
    assert any(not np.array_equal(after["USA"][k], v) for k, v in snapshot["USA"].items())


def test_a_league_opponent_plays_with_its_own_normalizer(
    fixture_catalog: Catalog, entries: Entries, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _env(fixture_catalog, entries)
    learner = ppo.Learner.create(ppo.PPOConfig(entries="unused.pkl"), env.observation_size)
    old = {i: n.state() for i, n in learner.obs_norms.items()}
    old = {i: dict(state, count=7.0) for i, state in old.items()}  # a recognizable normalizer
    current = {i: n.state() for i, n in learner.obs_norms.items()}
    used: list[dict[str, float]] = []

    class Recording(ppo.LearnedPolicy):
        def __init__(self, models: Any, norms: Any) -> None:
            super().__init__(models, norms)
            used.append({i: n.count for i, n in norms.items()})

    monkeypatch.setattr(ppo, "LearnedPolicy", Recording)
    monkeypatch.setitem(ppo._WORKER, "env", env)
    monkeypatch.setitem(ppo._WORKER, "models", {})
    weights = learner.weights()
    hidden = learner.config.hidden
    task = (weights, current, [0, 1], env.observation_size, hidden, [(weights, old)], 1.0)
    ppo._rollout(task)
    assert used and all(set(counts.values()) == {7.0} for counts in used)


def test_unrecorded_identities_leave_no_steps(fixture_catalog: Catalog, entries: Entries) -> None:
    env = _env(fixture_catalog, entries)
    learner = ppo.Learner.create(ppo.PPOConfig(entries="unused.pkl"), env.observation_size)
    policy = learner.policy()
    full = ppo.play_episode(env, policy, 5, np.random.default_rng(5))
    some = ppo.play_episode(env, policy, 5, np.random.default_rng(5), record={"USA"})
    assert set(some.steps) <= {"USA"} and set(full.steps) >= set(some.steps)
    assert some.returns == full.returns  # the same play, only fewer steps kept


def test_a_freestyle_campaign_plays_as_one_episode(fixture_catalog: Catalog) -> None:
    config = ppo.PPOConfig(entries="", mode="campaign", playable_tier=None)
    env = ppo.make_env(config, fixture_catalog)
    assert isinstance(env, ppo.CampaignTrainingEnv)
    learner = ppo.Learner.create(config, env.observation_size)
    episode = ppo.play_episode(env, learner.policy(), 0, np.random.default_rng(0))
    assert episode.event_id == "campaign" and episode.steps
    assert set(episode.returns) <= set(ppo.IDENTITIES)
    assert all(np.isfinite(r) for r in episode.returns.values())


@pytest.mark.slow
def test_a_best_response_reports_its_gain(tmp_path: Path) -> None:
    from cold_war_sim.campaign import load_catalog
    from cold_war_sim.campaign.rl import br

    catalog = load_catalog()
    path = tmp_path / "entries.pkl"
    save_entries(path, capture_entries(catalog, ["suez_crisis"]))
    config = ppo.PPOConfig(entries=str(path), workers=1, episodes=8, iterations=1)
    env = ppo.make_env(config, catalog)
    ppo.Learner.create(config, env.observation_size).save(tmp_path / "base.pt")
    result = br.best_response(
        tmp_path / "base.pt",
        "UK",
        tmp_path / "br",
        iterations=1,
        episodes=8,
        eval_episodes=16,
        workers=1,
        progress=lambda message: None,
    )
    assert result["episodes"] == 16 and np.isfinite(result["gain"])
