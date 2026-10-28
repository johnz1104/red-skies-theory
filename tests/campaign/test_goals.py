from __future__ import annotations

import copy
import math
import shutil
from pathlib import Path

import pytest
import yaml

from cold_war_sim.campaign import (
    CampaignEngine,
    Catalog,
    CatalogError,
    EngineConfig,
    RandomPolicy,
    calendar,
    goals,
    load_catalog,
    run_episode,
)
from cold_war_sim.campaign.catalog import data_root
from cold_war_sim.campaign.conditions import CompileContext, compile_condition
from cold_war_sim.campaign.effects import (
    EffectContext,
    apply_leader,
    compile_effects,
    shift_influence,
)
from cold_war_sim.campaign.state import identity

from .conftest import FIXTURE_EVENTS


def test_threat_weight_counts_rivals_more_than_allies(fixture_catalog: Catalog) -> None:
    world = fixture_catalog.initial_world()
    world.powers["UK"].relations["USA"] = 0.8
    world.powers["UK"].relations["USSR"] = -0.6
    assert goals.threat_weight(world, "UK", "USA") == pytest.approx(0.1)
    assert goals.threat_weight(world, "UK", "USSR") == pytest.approx(0.8)
    allied = goals.components(world, "UK", fixture_catalog)["security"]
    world.powers["UK"].relations["USA"] = -0.6
    hostile = goals.components(world, "UK", fixture_catalog)["security"]
    assert allied > hostile


def test_the_allies_band_rises_plateaus_and_turns_into_a_cost(fixture_catalog: Catalog) -> None:
    allies = fixture_catalog.goals.allies
    assert allies is not None
    h, rho = allies.height, allies.cost
    assert abs(allies.value("USSR", "PRC", 0.0)) < 0.01 * h  # too weak to help
    assert allies.value("USSR", "PRC", 0.2) > 0.8 * h  # a useful junior partner
    assert allies.value("USSR", "PRC", 1.5) == pytest.approx(-rho * h, abs=0.01)  # an equal
    assert allies.value("UK", "USA", 10.0) == pytest.approx(h, abs=0.01)  # no ceiling


def test_an_unlisted_stronger_ally_is_welcome(fixture_catalog: Catalog) -> None:
    allies = fixture_catalog.goals.allies
    assert allies is not None
    assert "USA" not in allies.bands.get("FRA", {})  # France sets no band for the US
    assert allies.band("FRA", "USA").ceiling is None
    assert allies.value("FRA", "USA", 5.0) == pytest.approx(allies.height, abs=0.01)


def test_an_ally_helps_inside_its_band_and_hurts_beyond_it(fixture_catalog: Catalog) -> None:
    world = fixture_catalog.initial_world()
    for p, q in (("USSR", "PRC"), ("PRC", "USSR")):
        world.powers[p].relations[q] = 0.8
    world.powers["PRC"].status = "active"
    soviet = goals.military_strength(world, "USSR", fixture_catalog.goals)

    def security_with_china_at(ratio: float) -> float:
        world.powers["PRC"].conventional = ratio * soviet
        world.powers["PRC"].nuclear = 0.0
        return goals.components(world, "USSR", fixture_catalog)["security"]

    junior, equal = security_with_china_at(0.2), security_with_china_at(1.2)
    world.powers["USSR"].relations["PRC"] = 0.0
    neutral_junior = security_with_china_at(0.2)
    assert junior > neutral_junior  # a friendly junior partner adds to Soviet security
    assert equal < junior  # an ally grown into an equal is a burden


def test_a_rivals_gain_from_the_neutral_share_still_costs(fixture_catalog: Catalog) -> None:
    world = fixture_catalog.initial_world()
    region = world.regions["persian_gulf"]
    assert 1.0 - sum(region.influence.values()) >= 0.1
    own_share = region.influence["USA"]
    before = goals.components(world, "USA", fixture_catalog)["influence"]
    shift_influence(world, "persian_gulf", "USSR", 0.1)
    assert region.influence["USA"] == pytest.approx(own_share)
    assert goals.components(world, "USA", fixture_catalog)["influence"] < before


def test_an_empire_holding_can_be_an_asset_or_a_liability(fixture_catalog: Catalog) -> None:
    world = fixture_catalog.initial_world()
    held = [r for r in world.regions.values() if r.colonial_owner == "UK"]
    assert held

    def set_holds(influence: float, stability: float) -> float:
        for region in held:
            region.influence = dict.fromkeys(region.influence, 0.0)
            region.influence["UK"] = influence
            region.stability = stability
        return goals.components(world, "UK", fixture_catalog)["empire"]

    assert set_holds(0.8, 0.8) == pytest.approx(0.6)
    assert set_holds(0.5, 0.3) == pytest.approx(-0.2)
    released = compile_effects(
        [
            {"colonial_owner": {"region": r, "to": None}}
            for r in fixture_catalog.regions
            if fixture_catalog.regions[r].colonial_owner == "UK"
        ],
        CompileContext(),
    )
    released.apply(world, EffectContext(leaders=fixture_catalog.leaders))
    assert goals.components(world, "UK", fixture_catalog)["empire"] == 0.0


def test_autonomy_needs_a_bomb_and_no_superpower_command(fixture_catalog: Catalog) -> None:
    world = fixture_catalog.initial_world()
    world.powers["FRA"].tech_nuclear = 0.0
    world.flags.discard("nato")
    assert goals.components(world, "FRA", fixture_catalog)["autonomy"] == 0.5
    world.flags.add("nato")
    assert goals.components(world, "FRA", fixture_catalog)["autonomy"] == 0.0
    world.powers["FRA"].tech_nuclear = 1.2
    world.flags.add("france_out_of_nato_command")
    assert goals.components(world, "FRA", fixture_catalog)["autonomy"] == 1.0


def test_economy_is_log_growth_since_1945(fixture_catalog: Catalog) -> None:
    world = fixture_catalog.initial_world()
    world.powers["USA"].economy *= 2.0
    assert goals.components(world, "USA", fixture_catalog)["economy"] == pytest.approx(0.5)


def test_weight_rows_follow_leaders_and_dates(fixture_catalog: Catalog) -> None:
    world = fixture_catalog.initial_world()
    assert goals.row(world, "UK", fixture_catalog).label.startswith("before Suez")
    apply_leader(world, fixture_catalog.leaders["reagan"])
    apply_leader(world, fixture_catalog.leaders["gorbachev"])
    assert goals.row(world, "USA", fixture_catalog).label == "Reagan"
    assert goals.row(world, "USSR", fixture_catalog).label == "Gorbachev"
    world.quarter = calendar.parse("1968-Q2")
    assert goals.row(world, "UK", fixture_catalog).label.startswith("after the East of Suez")
    for identity_id, rows in fixture_catalog.goals.rows.items():
        for weight_row in rows:
            assert math.isclose(sum(weight_row.weights.values()), 1.0), identity_id


def test_a_weight_switch_alone_gives_no_reward(fixture_catalog: Catalog) -> None:
    engine = CampaignEngine(fixture_catalog)
    engine.reset(0)
    engine._rewards()  # bring the stored components up to date
    engine.pop_rewards()
    weights_before = goals.row(engine.world, "USSR", fixture_catalog).weights
    apply_leader(engine.world, fixture_catalog.leaders["gorbachev"])
    assert goals.row(engine.world, "USSR", fixture_catalog).weights != weights_before
    engine._rewards()
    assert engine.pop_rewards()["USSR"] == pytest.approx(0.0)


def test_an_insurgency_pays_for_its_war_effort(fixture_catalog: Catalog) -> None:
    from cold_war_sim.campaign.dynamics import DynamicsParams, step

    params = DynamicsParams.from_catalog(fixture_catalog)
    world = fixture_catalog.initial_world()
    assert world.powers["PRC"].status == "insurgent"
    peace, war = copy.deepcopy(world), copy.deepcopy(world)
    step(peace, params, {})
    step(war, params, {"PRC": 1.0})
    before = world.powers["PRC"]
    assert peace.powers["PRC"].economy == before.economy
    assert peace.powers["PRC"].stability == before.stability
    assert war.powers["PRC"].economy < before.economy
    assert war.powers["PRC"].stability < before.stability


def test_a_regime_governing_part_of_its_country_counts_its_share(fixture_catalog: Catalog) -> None:
    world = fixture_catalog.initial_world()
    china = world.regions["china_mainland"]
    prc = world.powers["PRC"]
    assert prc.status == "insurgent"
    regime = goals.components(world, "PRC", fixture_catalog)["regime"]
    assert regime == pytest.approx(prc.stability * china.influence["PRC"])
    prc.status = "active"  # ruling all of China: the full stability
    assert goals.components(world, "PRC", fixture_catalog)["regime"] == prc.stability
    world.flags.add("china_divided")  # ruling the north of a divided China
    regime = goals.components(world, "PRC", fixture_catalog)["regime"]
    assert regime == pytest.approx(prc.stability * china.influence["PRC"])
    assert (
        goals.components(world, "USA", fixture_catalog)["regime"] == world.powers["USA"].stability
    )


def test_regime_condition_effect_and_identity(fixture_catalog: Catalog) -> None:
    world = fixture_catalog.initial_world()
    ccp = compile_condition({"regime": {"power": "PRC", "in": ["ccp"]}}, CompileContext())
    assert world.powers["PRC"].regime == "ccp" and ccp(world)
    assert identity(world, "PRC") == "PRC"
    compile_effects([{"regime": {"power": "PRC", "to": "kmt"}}], CompileContext()).apply(
        world, EffectContext(leaders=fixture_catalog.leaders)
    )
    assert not ccp(world)
    assert identity(world, "PRC") == "ROC"
    assert goals.row(world, "ROC", fixture_catalog).label == "Nationalists"


def test_both_reward_modes_play_the_same_game(fixture_catalog: Catalog) -> None:
    a = run_episode(CampaignEngine(fixture_catalog), RandomPolicy(), 3)
    b = run_episode(
        CampaignEngine(fixture_catalog, EngineConfig(reward="score")), RandomPolicy(), 3
    )
    assert a.trace_hash == b.trace_hash
    assert all(math.isfinite(v) for v in a.returns.values())
    with pytest.raises(ValueError, match="reward"):
        EngineConfig(reward="prestige")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["weights"]["USA"][-1]["weights"].update(security=0.5),
        lambda d: d["weights"]["UK"][0].update(when={"fired": "suez_crisis"}),
        lambda d: d["weights"]["FRA"][-1].update(when={"after": "1950-Q1"}),
        lambda d: d["weights"].pop("ROC"),
        lambda d: d["components"]["influence"]["interest"]["USA"].update(atlantis=2),
        lambda d: d["weights"]["PRC"][0].update(when={"leader": {"power": "PRC", "in": ["sun"]}}),
        lambda d: d["components"]["regime"]["partial"]["PRC"].update(region="atlantis"),
        lambda d: d["components"]["regime"]["partial"]["PRC"].pop("when"),
    ],
)
def test_goal_validation_errors(tmp_path: Path, mutate: object) -> None:
    root = tmp_path / "data"
    shutil.copytree(data_root(), root)
    doc = yaml.safe_load((root / "goals.yaml").read_text())
    mutate(doc)  # type: ignore[operator]
    (root / "goals.yaml").write_text(yaml.safe_dump(doc))
    with pytest.raises(CatalogError, match=r"goals\.yaml"):
        load_catalog(root, event_files=[FIXTURE_EVENTS])


def test_buffer_regions_count_toward_security(fixture_catalog: Catalog) -> None:
    world = fixture_catalog.initial_world()
    assert "eastern_europe" in fixture_catalog.goals.buffers["USSR"]
    region = world.regions["eastern_europe"]
    base_ussr = goals.components(world, "USSR", fixture_catalog)["security"]
    base_uk = goals.components(world, "UK", fixture_catalog)["security"]
    shift_influence(world, "eastern_europe", "USA", 0.2)  # a rival on the Soviet approaches
    assert goals.components(world, "USSR", fixture_catalog)["security"] < base_ussr
    assert goals.components(world, "UK", fixture_catalog)["security"] == pytest.approx(base_uk)
    world2 = fixture_catalog.initial_world()
    shift_influence(world2, "eastern_europe", "USSR", 0.2)
    assert goals.components(world2, "USSR", fixture_catalog)["security"] > base_ussr
    assert region.importance > 0


def test_end_bonuses_belong_to_the_score_reward_only(fixture_catalog: Catalog) -> None:
    from cold_war_sim.campaign import rewards

    world = fixture_catalog.initial_world()
    world.ended = "ussr_dissolved"
    with_bonus = rewards.terminal_bonus(world, fixture_catalog)
    without = rewards.terminal_bonus(world, fixture_catalog, end_bonuses=False)
    assert with_bonus["USA"] == fixture_catalog.powers["USA"].end_bonus["ussr_dissolved"]
    assert without == dict.fromkeys(without, 0.0)
    world.nuclear_war = True  # the nuclear-war penalty applies under either reward
    penalty = float(fixture_catalog.rewards.get("nuclear_war_penalty", 20.0))
    assert rewards.terminal_bonus(world, fixture_catalog, end_bonuses=False)["UK"] == -penalty
