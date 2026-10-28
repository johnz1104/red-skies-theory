"""The shared effects contract (data/contract.yaml, E47) and the multiply effect."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from cold_war_sim.campaign import contract
from cold_war_sim.campaign.catalog import CatalogError, load_catalog
from cold_war_sim.campaign.conditions import CompileContext
from cold_war_sim.campaign.dynamics import DynamicsParams, step
from cold_war_sim.campaign.effects import EffectContext, compile_effects

from .conftest import FIXTURE_EVENTS

CRISIS = {
    "id": "fx_contract_crisis",
    "kind": "major",
    "title": "A crisis under the contract",
    "window": {"earliest": "1947-Q1", "latest": "1947-Q1"},
    "model": {
        "family": "crisis",
        "region": "persian_gulf",
        "effects": "contract",
        "config": {
            "roles": {
                "USSR": {
                    "side": "a",
                    "values": {
                        "a_prevails": 1,
                        "b_prevails": -1,
                        "compromise": 0,
                        "limited_war": -4,
                        "nuclear_war": -30,
                    },
                },
                "USA": {
                    "side": "b",
                    "values": {
                        "a_prevails": -1,
                        "b_prevails": 1,
                        "compromise": 0,
                        "limited_war": -4,
                        "nuclear_war": -30,
                    },
                },
            },
            "script": [
                {
                    "stage": "move",
                    "actor": "USSR",
                    "options": [
                        {"id": "withdraw", "kind": "concede"},
                        {"id": "stay", "kind": "escalate"},
                    ],
                },
                {
                    "stage": "move",
                    "actor": "USA",
                    "options": [{"id": "yield", "kind": "concede"}, {"id": "hold", "kind": "hold"}],
                },
            ],
            "final_outcome": "compromise",
        },
        "historical": {"choices": {"USSR": ["stay"], "USA": ["yield"]}, "outcome": "a_prevails"},
    },
    "outcomes": {
        "a_prevails": {"additions": [{"flag": "fx_contract_won"}]},
        "b_prevails": {},
        "compromise": {},
        "limited_war": {},
        "nuclear_war": {},
    },
}


def _catalog(tmp_path: Path, event: dict[str, Any]) -> Any:
    path = tmp_path / "contract_events.yaml"
    path.write_text(yaml.safe_dump({"era": 9, "title": "Contract fixture", "events": [event]}))
    return load_catalog(event_files=[FIXTURE_EVENTS, path])


def test_war_costs_match_the_dynamics() -> None:
    catalog = load_catalog(event_files=[FIXTURE_EVENTS])
    params = DynamicsParams.from_catalog(catalog)
    quiet, war = catalog.initial_world(), catalog.initial_world()
    for _ in range(4):
        step(quiet, params, {})
        step(war, params, {"USA": 1.0})
    economy, stability = contract.war_costs(contract.dynamics_of(catalog.raw.powers), 4)
    assert war.powers["USA"].economy / quiet.powers["USA"].economy == pytest.approx(
        economy, abs=1e-4
    )
    loss = quiet.powers["USA"].stability - war.powers["USA"].stability
    assert loss == pytest.approx(stability, abs=1e-6)


def test_an_opted_in_event_gets_the_contract_plus_its_additions(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path, copy.deepcopy(CRISIS))
    terms = catalog.raw.contract
    rule = terms["crisis"]["prevails"]
    major = catalog.events["fx_contract_crisis"].major
    world = catalog.initial_world()
    before = copy.deepcopy(world)
    major.outcomes["a_prevails"].effects.apply(world, EffectContext(leaders=catalog.leaders))
    importance = catalog.regions["persian_gulf"].importance
    assert "fx_contract_won" in world.flags
    gain = world.powers["USSR"].credibility - before.powers["USSR"].credibility
    loss = world.powers["USA"].credibility - before.powers["USA"].credibility
    assert gain == pytest.approx(rule["winner_credibility"])
    assert loss == pytest.approx(rule["loser_credibility"])
    moved = (
        world.regions["persian_gulf"].influence["USSR"]
        - before.regions["persian_gulf"].influence["USSR"]
    )
    assert moved == pytest.approx(
        rule["influence"] * importance / terms["reference_importance"], abs=1e-9
    )
    war = catalog.initial_world()
    major.outcomes["limited_war"].effects.apply(war, EffectContext(leaders=catalog.leaders))
    assert war.powers["USA"].economy < before.powers["USA"].economy
    assert war.powers["USSR"].conventional < before.powers["USSR"].conventional
    assert war.tension > before.tension


def test_a_contract_number_reaches_every_opted_in_event(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path, copy.deepcopy(CRISIS))
    doubled = catalog.variant({"contract:crisis/prevails/winner_credibility": 0.2})
    world = doubled.initial_world()
    start = world.powers["USSR"].credibility
    doubled.events["fx_contract_crisis"].major.outcomes["a_prevails"].effects.apply(
        world, EffectContext(leaders=doubled.leaders)
    )
    assert world.powers["USSR"].credibility - start == pytest.approx(0.2)


def test_an_opted_in_outcome_may_not_list_effects(tmp_path: Path) -> None:
    event = copy.deepcopy(CRISIS)
    event["outcomes"]["b_prevails"] = {"effects": [{"tension": 0.1}]}
    with pytest.raises(CatalogError):
        _catalog(tmp_path, event)


def test_multiply_scales_within_bounds() -> None:
    catalog = load_catalog(event_files=[FIXTURE_EVENTS])
    effects = compile_effects(
        [
            {"multiply": {"path": "powers.USA.economy", "factor": 0.5}},
            {"multiply": {"path": "powers.USA.stability", "factor": 10.0}},
        ],
        CompileContext(),
    )
    world = catalog.initial_world()
    economy = world.powers["USA"].economy
    effects.apply(world, EffectContext(leaders=catalog.leaders))
    assert world.powers["USA"].economy == pytest.approx(economy * 0.5)
    assert world.powers["USA"].stability == 1.0  # clamped to its bound
