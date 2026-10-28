from __future__ import annotations

import copy
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from cold_war_sim.campaign import (
    CampaignEngine,
    Catalog,
    CatalogError,
    EngineConfig,
    EquilibriumPolicy,
    RandomPolicy,
    calendar,
    derive,
    goalcheck,
    goals,
    load_catalog,
    run_episode,
)
from cold_war_sim.campaign.effects import EffectContext
from cold_war_sim.families import build_family

from .conftest import FIXTURE_EVENTS


def _derived_catalog(
    tmp_path: Path, extra: list[dict[str, Any]] | None = None, keep_binding: bool = False
) -> Catalog:
    doc = yaml.safe_load(FIXTURE_EVENTS.read_text())
    for event in doc["events"]:
        if event["kind"] == "major":
            event["model"]["payoffs"] = "derived"
            if event["id"] == "fx_crisis" and not keep_binding:
                event["model"].pop("bindings")
    doc["events"].extend(extra or [])
    path = tmp_path / "events.yaml"
    path.write_text(yaml.safe_dump(doc))
    return load_catalog(event_files=[path])


@pytest.fixture
def derived(tmp_path: Path) -> Catalog:
    return _derived_catalog(tmp_path)


def test_every_family_builds_with_derived_payoffs(derived: Catalog) -> None:
    world = derived.initial_world()
    families = set()
    for event in derived.events.values():
        if event.major is None:
            continue
        config = derived.instantiate_config(event.id, world)
        build_family(event.major.family, config)
        families.add(event.major.family)
        assert config != event.major.bound_config(world) or not event.major.power_roles()
    assert families == {"crisis", "attrition", "alignment", "bargaining", "intervention"}
    civil_war = derived.instantiate_config("fx_civil_war", world)
    held = max(goals.score(world, "PRC", derived), derived.rewards["loss_floor"])
    expected = round(-derived.goals.kappa * held, 2)
    assert civil_war["sponsors"]["PRC"]["values"]["b_victory"] == pytest.approx(expected)


def test_a_derived_value_matches_a_hand_calculation(derived: Catalog) -> None:
    world = derived.initial_world()
    config = derived.instantiate_config("fx_crisis", world)
    major = derived.events["fx_crisis"].major
    assert major is not None
    after = copy.deepcopy(world)
    major.outcomes["a_prevails"].effects.apply(after, EffectContext(leaders=derived.leaders))
    weights = goals.row(world, "USA", derived).weights
    change = goals.reward(
        goals.components(world, "USA", derived), goals.components(after, "USA", derived), weights
    )
    expected = round(derived.goals.kappa * change, 2)
    assert config["roles"]["USA"]["values"]["a_prevails"] == pytest.approx(expected)
    assert expected < 0  # a Soviet gain in the Persian Gulf costs the USA
    nuclear = derived.goals.kappa * derived.rewards["nuclear_war_penalty"]
    assert config["roles"]["USA"]["values"]["nuclear_war"] == pytest.approx(-nuclear)


def test_consequence_cutscenes_count_toward_the_outcome(tmp_path: Path) -> None:
    fallout = {
        "id": "fx_fallout",
        "kind": "cutscene",
        "title": "Fallout",
        "when": {"outcome": {"event": "fx_crisis", "in": ["a_prevails"]}},
        "effects": [{"status": {"power": "USA", "to": "eliminated"}}],
    }
    catalog = _derived_catalog(tmp_path, [fallout])
    world = catalog.initial_world()
    config = catalog.instantiate_config("fx_crisis", world)
    held = max(goals.score(world, "USA", catalog), catalog.rewards["loss_floor"])
    penalty = round(catalog.goals.kappa * held, 2)
    assert config["roles"]["USA"]["values"]["a_prevails"] == pytest.approx(-penalty)
    assert config["roles"]["USA"]["values"]["b_prevails"] > -penalty


def _follow_on_catalog(tmp_path: Path) -> Catalog:
    doc = yaml.safe_load(FIXTURE_EVENTS.read_text())
    crisis = next(e for e in doc["events"] if e["id"] == "fx_crisis")
    crisis["model"]["payoffs"] = "derived"
    crisis["model"].pop("bindings")
    doc["events"] += [
        {  # enabled by a_prevails, possible from three quarters after the crisis
            "id": "fx_later",
            "kind": "cutscene",
            "title": "Later",
            "window": {"earliest": "1947-Q4"},
            "when": {"outcome": {"event": "fx_crisis", "in": ["a_prevails"]}},
            "effects": [
                {"shift_influence": {"region": "persian_gulf", "toward": "USSR", "amount": 0.2}}
            ],
        },
        {  # likely under any outcome except b_prevails, which blocks it
            "id": "fx_blocked",
            "kind": "cutscene",
            "title": "Blocked",
            "window": {"earliest": "1947-Q4"},
            "when": {"no_flag": "crisis_won"},
            "effects": [{"add": {"path": "powers.USA.credibility", "value": 0.2}}],
        },
        {  # beyond the lookahead
            "id": "fx_far",
            "kind": "cutscene",
            "title": "Far",
            "window": {"earliest": "1960-Q1"},
            "when": {"outcome": {"event": "fx_crisis", "in": ["a_prevails"]}},
            "effects": [{"add": {"path": "powers.USA.credibility", "value": -0.5}}],
        },
    ]
    path = tmp_path / "events.yaml"
    path.write_text(yaml.safe_dump(doc))
    return load_catalog(event_files=[path])


def test_follow_on_events_count_at_their_discounted_chance(tmp_path: Path) -> None:
    catalog = _follow_on_catalog(tmp_path)
    event = catalog.events["fx_crisis"]
    major = event.major
    assert major is not None
    world = catalog.initial_world()
    world.quarter = calendar.parse("1947-Q1")
    valuer = derive.Valuer(catalog, event, world, catalog.goals)
    after_a = valuer.after([major.outcomes["a_prevails"].effects], label="a_prevails")
    after_b = valuer.after([major.outcomes["b_prevails"].effects], label="b_prevails")
    later = catalog.events["fx_later"]
    blocked = catalog.events["fx_blocked"]
    delta = catalog.goals.lookahead_discount
    # hazard 1, first possible quarter 1947-Q4 = three quarters after the crisis
    assert valuer._chance(later, after_a) == pytest.approx(delta**3)
    assert valuer._chance(later, world) == 0.0
    assert valuer._chance(blocked, world) == pytest.approx(delta**3)
    assert valuer._chance(blocked, after_b) == 0.0  # b_prevails sets crisis_won
    assert valuer._chance(catalog.events["fx_far"], after_a) == 0.0  # beyond the lookahead

    flat = replace(catalog.goals, lookahead_quarters=0)
    with_follow = derive.apply(catalog, event, world, major.bound_config(world))
    without = derive.apply(catalog, event, world, major.bound_config(world), flat)
    usa_with, usa_without = with_follow["roles"]["USA"]["values"], without["roles"]["USA"]["values"]
    assert usa_with["a_prevails"] < usa_without["a_prevails"]  # a later Soviet gain costs the US
    assert usa_with["b_prevails"] < usa_without["b_prevails"]  # winning blocks a later US gain
    assert usa_with["compromise"] == pytest.approx(usa_without["compromise"])  # blocked by neither


def test_a_vetoers_own_cost_counts_in_its_veto_value(tmp_path: Path) -> None:
    doc = yaml.safe_load(FIXTURE_EVENTS.read_text())
    talks = next(e for e in doc["events"] if e["id"] == "fx_talks")
    talks["model"]["payoffs"] = "derived"
    talks["model"]["config"]["consent"] = [{"role": "UK", "veto_value": 0.0}]
    talks["tags"] = {
        "UK:vetoed": {"effects": [{"relations": {"a": "USA", "b": "UK", "add": -0.3}}]}
    }
    path = tmp_path / "events.yaml"
    path.write_text(yaml.safe_dump(doc))
    catalog = load_catalog(event_files=[path])
    config = catalog.instantiate_config("fx_talks", catalog.initial_world())
    assert config["consent"][0]["veto_value"] < 0.0  # the vetoed label alone changes nothing
    assert config["sides"]["A"]["vetoed_value"] == 0.0  # principals value the label only


def test_bindings_on_derived_values_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(CatalogError, match="binding on derived value"):
        _derived_catalog(tmp_path, keep_binding=True)


def test_campaigns_run_with_derived_payoffs(derived: Catalog) -> None:
    for seed in range(2):
        result = run_episode(CampaignEngine(derived), RandomPolicy(), seed)
        assert all(math.isfinite(v) for v in result.returns.values())
    engine = CampaignEngine(derived, EngineConfig(npc_policy="equilibrium"))
    result = run_episode(engine, EquilibriumPolicy(), 0)
    assert result.final_quarter == "1991-Q4" or result.end_reason == "nuclear_war"


def test_review_tools_capture_solve_and_sweep(derived: Catalog) -> None:
    worlds = goalcheck.capture_worlds(derived, ["fx_crisis", "fx_talks"])
    world, config = worlds["fx_crisis"]
    assert world.quarter == calendar.parse("1947-Q1")
    crisis = goalcheck.outcome_distribution("crisis", config, iterations=100)
    talks = goalcheck.outcome_distribution("bargaining", worlds["fx_talks"][1], rollouts=50)
    assert sum(crisis.values()) == pytest.approx(1.0)
    assert sum(talks.values()) == pytest.approx(1.0)
    labels = [
        label for label, _ in goalcheck.variations(derived, derived.events["fx_crisis"], world)
    ]
    assert "kappa x2.0" in labels and "USA rivalry x0.5" in labels
    assert any(label.startswith("USSR security") for label in labels)


def test_a_lottery_is_valued_at_its_expectation(tmp_path: Path) -> None:
    doc = yaml.safe_load(FIXTURE_EVENTS.read_text())
    talks = next(e for e in doc["events"] if e["id"] == "fx_talks")
    talks["model"]["payoffs"] = "derived"
    win = [{"set_controller": {"region": "germany", "to": "USA"}}]
    lose = [{"set_controller": {"region": "germany", "to": "USSR"}}]
    talks["outcomes"]["agreement"] = {
        "effects": [{"lottery": {"chance": {"metric": "share"}, "win": win, "lose": lose}}]
    }
    talks["outcomes"]["breakdown"] = {"effects": win}  # stand-ins for the two branches
    talks["outcomes"]["impasse"] = {"effects": lose}
    path = tmp_path / "events.yaml"
    path.write_text(yaml.safe_dump(doc))
    catalog = load_catalog(event_files=[path])
    event = catalog.events["fx_talks"]
    valuer = derive.Valuer(catalog, event, catalog.initial_world(), catalog.goals)
    won, lost = valuer.label_value("breakdown", "USA"), valuer.label_value("impasse", "USA")
    assert won > lost
    for share in (0.0, 0.3, 1.0):
        expected = share * won + (1.0 - share) * lost
        value = valuer.label_value("agreement", "USA", {"share": share})
        assert value == pytest.approx(expected, abs=0.011)


def _nationalist_war(tmp_path: Path, **changes: Any) -> Catalog:
    """The fixture civil war with a computer-run Nationalist lead valued as the ROC."""

    doc = yaml.safe_load(FIXTURE_EVENTS.read_text())
    war = next(e for e in doc["events"] if e["id"] == "fx_civil_war")
    model = war["model"]
    model["payoffs"] = "derived"
    model["cadence"] = "quarterly"
    model.pop("historical")
    config = model["config"]
    hand = {"a_victory": -5, "b_victory": 3, "partition": 0, "a_withdrew": 3}
    hand |= {"b_withdrew": -5, "stalemate": 0}
    config["sponsors"]["KMT"] = {"side": "b", "capacity": 0.0, "values": hand}
    config["sponsors"]["KMT"]["settlement"] = {"base": -5.0, "per_share": 10.0}
    config["leads"] = {"a": "PRC", "b": "KMT"}
    config["initial_control"] = 0.3
    config["offers"] = {"grid": [0.3, 0.5, 0.7], "proposer": "alternate"}
    model["identities"] = {"KMT": {"identity": "ROC", "stalemate_share": 0.5}}
    handover = [
        {"regime": {"power": "PRC", "to": "kmt"}},
        {"status": {"power": "PRC", "to": "active"}},
    ]
    war["outcomes"]["b_victory"] = {"effects": handover}
    for key, value in changes.items():
        if value is None:
            model.pop(key)
        else:
            model[key] = value
    path = tmp_path / "events.yaml"
    path.write_text(yaml.safe_dump(doc))
    return load_catalog(event_files=[path])


def test_an_identity_role_values_outcomes_by_the_share_it_would_govern(tmp_path: Path) -> None:
    catalog = _nationalist_war(tmp_path)
    event = catalog.events["fx_civil_war"]
    major = event.major
    assert major is not None
    world = catalog.initial_world()
    valuer = derive.Valuer(catalog, event, world, catalog.goals)
    after = valuer.after([major.outcomes["b_victory"].effects], label="b_victory")
    weights = goals.row(after, "ROC", catalog).weights
    end = goals.components(after, "ROC", catalog)
    ruled = goals.combine(end, weights) + valuer.follow_on(after, "ROC", weights, end)
    kappa = catalog.goals.kappa
    config = catalog.instantiate_config("fx_civil_war", world)
    kmt = config["sponsors"]["KMT"]
    held = 0.7  # side B starts with 1 - initial_control
    assert kmt["values"]["b_victory"] == pytest.approx(round(kappa * ruled * 0.3, 2))
    assert kmt["values"]["a_withdrew"] == kmt["values"]["b_victory"]
    lost = -kappa * max(ruled * held, catalog.rewards["loss_floor"])
    assert kmt["values"]["a_victory"] == pytest.approx(round(lost, 2))
    assert kmt["values"]["stalemate"] == pytest.approx(round(kappa * ruled * (0.5 - held), 2))
    assert kmt["values"]["partition"] == 0.0  # at the starting line
    assert kmt["settlement"]["per_share"] == pytest.approx(round(kappa * ruled, 2))
    assert kmt["settlement"]["base"] == pytest.approx(round(-kappa * ruled * held, 2))
    build_family("attrition", config)


def test_identity_roles_are_validated(tmp_path: Path) -> None:
    with pytest.raises(CatalogError, match="computer-run sponsor"):
        _nationalist_war(tmp_path, identities={"USA": {"identity": "ROC", "stalemate_share": 0.5}})
    with pytest.raises(CatalogError, match="unknown identity"):
        _nationalist_war(tmp_path, identities={"KMT": {"identity": "KMT", "stalemate_share": 0.5}})
    with pytest.raises(CatalogError, match="derived payoffs"):
        _nationalist_war(tmp_path, payoffs=None)
    with pytest.raises(CatalogError, match="binding on derived value"):
        binding = {"at": "sponsors.KMT.values.b_victory", "terms": {"powers.USA.credibility": 1.0}}
        _nationalist_war(tmp_path, bindings=[binding])


def test_an_actions_tag_becomes_its_cost_to_each_power(tmp_path: Path) -> None:
    doc = yaml.safe_load(FIXTURE_EVENTS.read_text())
    crisis_event = next(e for e in doc["events"] if e["id"] == "fx_crisis")
    crisis_event["model"]["payoffs"] = "derived"
    crisis_event["model"].pop("bindings")
    crisis_event["tags"] = {
        "USSR:move=stay": {"effects": [{"relations": {"a": "USA", "b": "USSR", "add": -0.3}}]}
    }
    path = tmp_path / "events.yaml"
    path.write_text(yaml.safe_dump(doc))
    catalog = load_catalog(event_files=[path])
    event = catalog.events["fx_crisis"]
    world = catalog.initial_world()
    config = catalog.instantiate_config("fx_crisis", world)
    stay = next(o for o in config["script"][1]["options"] if o["id"] == "stay")
    valuer = derive.Valuer(catalog, event, world, catalog.goals)
    for power in ("USA", "USSR"):
        value = valuer.tag_value("USSR:move=stay", power)
        assert value is not None and value < 0.0
        assert stay["costs"][power] == pytest.approx(round(-value, 2))
    withdraw = next(o for o in config["script"][1]["options"] if o["id"] == "withdraw_troops")
    assert "costs" not in withdraw
    held = derive.slots("crisis", event.major.config, event.major.tags)  # type: ignore[union-attr]
    assert ("script", 1, "options", 1, "costs") in held
