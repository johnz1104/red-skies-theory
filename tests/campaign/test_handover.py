from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from cold_war_sim.campaign import (
    CampaignEngine,
    Catalog,
    ColdWarEnv,
    EngineConfig,
    HistoricalPolicy,
    calendar,
    goals,
    load_catalog,
    run_episode,
)
from cold_war_sim.campaign.state import identity

from .conftest import FIXTURE_EVENTS

HANDOVER = [
    {"regime": {"power": "PRC", "to": "kmt"}},
    {"status": {"power": "PRC", "to": "active"}},
    {"flag": "kmt_holds_china"},
]


@pytest.fixture
def nationalist_win(tmp_path: Path) -> Catalog:
    """The fixture civil war, played so that the Nationalists win and take the seat."""

    doc = yaml.safe_load(FIXTURE_EVENTS.read_text())
    war = next(e for e in doc["events"] if e["id"] == "fx_civil_war")
    war["outcomes"]["b_victory"] = {"effects": HANDOVER}
    war["outcomes"]["a_withdrew"] = {"effects": HANDOVER}
    war["model"]["historical"] = {
        "choices": {"PRC": ["high"], "USSR": ["low"], "USA": ["high"]},
        "chance": ["b_gains"],
    }
    path = tmp_path / "events.yaml"
    path.write_text(yaml.safe_dump(doc))
    return load_catalog(event_files=[path])


def test_a_communist_defeat_hands_the_seat_to_the_nationalists(nationalist_win: Catalog) -> None:
    engine = CampaignEngine(nationalist_win, EngineConfig(chance_mode="historical"))
    deciders: set[str] = set()
    result = run_episode(
        engine,
        HistoricalPolicy(),
        0,
        on_decision=lambda e, r: deciders.add(e.agent_of(r.power)),
    )
    world = engine.world
    assert world.outcomes["fx_civil_war"] in ("b_victory", "a_withdrew")
    assert world.powers["PRC"].regime == "kmt" and world.powers["PRC"].status == "active"
    assert identity(world, "PRC") == "ROC"
    assert {"PRC", "ROC"} <= deciders
    assert engine.s.retired == {"PRC"}
    assert np.isfinite(result.returns["PRC"]) and np.isfinite(result.returns["ROC"])


def test_losing_a_seat_costs_the_score_held_with_a_floor(nationalist_win: Catalog) -> None:
    engine = CampaignEngine(nationalist_win)
    engine.reset(0)
    floor = nationalist_win.rewards["loss_floor"]
    parts = dict(engine.s.components["USA"])
    weights = goals.row(engine.world, "USA", nationalist_win).weights
    held = goals.combine(parts, weights)
    assert held > floor
    assert engine._loss_penalty("USA") == pytest.approx(held)
    engine.s.components["USA"] = dict.fromkeys(parts, -1.0)  # a negative score still pays
    assert engine._loss_penalty("USA") == pytest.approx(floor)
    score_mode = CampaignEngine(nationalist_win, EngineConfig(reward="score"))
    score_mode.reset(0)
    assert score_mode._loss_penalty("USA") == nationalist_win.rewards["elimination_penalty"]


def test_the_environment_adds_the_nationalist_agent(nationalist_win: Catalog) -> None:
    env = ColdWarEnv(nationalist_win, EngineConfig(chance_mode="historical"))
    assert env.possible_agents == ("USA", "USSR", "PRC", "ROC", "UK", "FRA")
    env.reset(seed=0)
    assert "ROC" not in env.agents
    policy, rng = HistoricalPolicy(), np.random.default_rng(0)
    seen: list[str] = []
    prc_terminated = False
    while env.agents:
        agent = env.agent_selection
        _, _, terminated, truncated, _ = env.last()
        if terminated or truncated:
            prc_terminated |= agent == "PRC" and env.engine.world.quarter < calendar.END
            env.step(None)
            continue
        if agent not in seen:
            seen.append(agent)
        request = env.engine.pending()
        assert request is not None and env.action_mask(agent).sum() == len(request.options)
        env.step(policy.choose(env.engine, request, rng))
    assert prc_terminated
    assert seen.index("PRC") < seen.index("ROC")


def test_masks_and_observations_follow_the_identity(nationalist_win: Catalog) -> None:
    engine = CampaignEngine(nationalist_win, EngineConfig(chance_mode="historical"))
    engine.reset(0)
    policy, rng = HistoricalPolicy(), np.random.default_rng(0)
    while engine.world.powers["PRC"].regime != "kmt":
        request = engine.pending()
        assert request is not None
        engine.act(policy.choose(engine, request, rng))
    while engine.pending() is not None and engine.pending().power != "PRC":  # type: ignore[union-attr]
        engine.act(policy.choose(engine, engine.pending(), rng))  # type: ignore[arg-type]
    request = engine.pending()
    assert request is not None and request.power == "PRC"
    assert engine.action_mask("ROC").sum() == len(request.options)
    assert engine.action_mask("PRC").sum() == 0
    assert (engine.observe("ROC") == engine.observe("PRC")).all()
