"""The crisis generator: casting, hazards, precedence, lazy specs and the engine hook."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from cold_war_sim.campaign import calendar, generator
from cold_war_sim.campaign.catalog import Catalog, CatalogError, load_catalog
from cold_war_sim.campaign.engine import CampaignEngine, EngineConfig, run_episode
from cold_war_sim.campaign.policies import make_policy
from cold_war_sim.campaign.state import WorldState


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return load_catalog()


def hostile_world(catalog: Catalog) -> WorldState:
    """1945's world with the superpowers fallen out and a contested Germany."""

    world = catalog.initial_world()
    world.powers["USA"].relations["USSR"] = -0.5
    world.powers["USSR"].relations["USA"] = -0.5
    world.quarter = calendar.parse("1962-Q1")
    return world


def test_ids_round_trip() -> None:
    eid = generator.event_id("standoff", "germany", ("USA", "USSR"))
    assert eid == "gen:standoff:germany:USA-USSR"
    assert generator.parse_id(eid) == ("standoff", "germany", ("USA", "USSR"))
    for bad in ("gen:war:germany:USA-USSR", "gen:standoff:mars:USA-USSR", "x:standoff"):
        with pytest.raises(generator.GeneratorError):
            generator.parse_id(bad)


def test_roles_go_to_the_leader_and_its_hostile_rival(catalog: Catalog) -> None:
    world = hostile_world(catalog)
    assert generator.cast(catalog.initial_world(), "germany") is None  # no hostility in 1945
    leader, rival = generator.cast(world, "germany")  # type: ignore[misc]
    influence = world.regions["germany"].influence
    assert influence[leader] == max(influence.values())
    assert world.powers[leader].relations[rival] < 0.0


def test_instability_raises_the_hazard(catalog: Catalog) -> None:
    doc = catalog.raw.generator  # type: ignore[union-attr]
    world = hostile_world(catalog)
    x = generator.features(catalog, world, "germany", ("USA", "USSR"))
    calm = generator.hazard(doc, "proxy_war", "germany", world.quarter, {**x, "instability": 0.2})
    torn = generator.hazard(doc, "proxy_war", "germany", world.quarter, {**x, "instability": 0.8})
    assert torn > calm > 0.0
    assert generator.hazard(doc, "settlement", "germany", world.quarter, x) == 0.0  # no war yet


def test_generated_events_resolve_but_stay_out_of_the_catalog(catalog: Catalog) -> None:
    named = len(catalog.events)
    eid = generator.event_id("uprising", "eastern_europe", ("USSR", "USA"))
    spec = catalog.events[eid]
    assert spec.major is not None and spec.major.family == "intervention"
    assert spec.major.payoffs == "derived" and eid not in list(catalog.events)
    assert len(catalog.events) == named and catalog.events.get("gen:nonsense") is None
    config = catalog.instantiate_config(eid, hostile_world(catalog))
    assert set(config["roles"].values()) == {"USSR", "USA"}
    with pytest.raises(KeyError):
        catalog.events["no_such_event"]


def test_a_variant_compiles_its_own_generated_events(catalog: Catalog) -> None:
    eid = generator.event_id("standoff", "germany", ("USA", "USSR"))
    variant = catalog.variant({"generator:templates/standoff/config/script/6/per_rung": 0.06})
    base = catalog.events[eid].major
    other = variant.events[eid].major
    assert base is not None and other is not None
    assert other.config["script"][6]["per_rung"] == 0.06 != base.config["script"][6]["per_rung"]


def test_precedence_blocks_a_region_with_a_crisis(catalog: Catalog) -> None:
    world = hostile_world(catalog)
    assert not generator.blocked(catalog, world, "caribbean", 4)
    world.active["cuban_missile_crisis"] = world.quarter
    assert generator.blocked(catalog, world, "caribbean", 4)
    del world.active["cuban_missile_crisis"]
    world.fired["cuban_missile_crisis"] = world.quarter - 2
    assert generator.blocked(catalog, world, "caribbean", 4)
    assert not generator.blocked(catalog, world, "caribbean", 2)


def test_proposals_are_deterministic_and_respect_one_per_region(catalog: Catalog) -> None:
    raw = catalog.raw
    assert raw is not None
    doc = copy.deepcopy(raw.generator)
    for spec in doc["templates"].values():
        spec["rate"] = 1.0  # every candidate fires
    hot = replace(catalog, raw=replace(raw, generator=doc))
    world = hostile_world(catalog)
    first = generator.propose(hot, world, seed=7)
    assert first == generator.propose(hot, world, seed=7)
    regions = [generator.parse_id(e)[1] for e in first]
    assert regions and len(regions) == len(set(regions))


def test_the_engine_starts_generated_crises_only_when_asked(catalog: Catalog) -> None:
    def campaign(config: EngineConfig) -> CampaignEngine:
        engine = CampaignEngine(catalog, config)
        run_episode(engine, make_policy("random"), 3)
        return engine

    end = calendar.parse("1960-Q4")
    plain = campaign(EngineConfig(end_quarter=end, playable_tier=1))
    assert not any(generator.is_generated(e) for e in plain.world.fired)
    on = campaign(EngineConfig(end_quarter=end, playable_tier=1, generator=True))
    started = [e for e in on.world.fired if generator.is_generated(e)]
    assert started, "the generator started nothing by 1960"
    again = campaign(EngineConfig(end_quarter=end, playable_tier=1, generator=True))
    assert again.trace_hash() == on.trace_hash()
    replay = campaign(
        EngineConfig(end_quarter=end, playable_tier=1, generator=True, chance_mode="historical")
    )
    assert not any(generator.is_generated(e) for e in replay.world.fired)


def test_an_unknown_generated_id_is_a_catalog_error(catalog: Catalog) -> None:
    with pytest.raises((KeyError, CatalogError)):
        catalog.events["gen:standoff:germany:USA-USA"]


def test_the_strategy_turn_offers_challenges_only_against_a_hostile_leader(
    catalog: Catalog,
) -> None:
    from cold_war_sim.campaign import strategic

    world = hostile_world(catalog)
    region = max(world.regions, key=lambda r: world.regions[r].influence["USSR"])
    plain = strategic.request_for(world, "USA", "mode", {"region": region}, {})
    offered = strategic.request_for(world, "USA", "mode", {"region": region}, {}, crises=True)
    assert plain is not None and offered is not None
    assert {o.id for o in plain.options} == {"aid", "covert"}
    assert {o.id for o in offered.options} == {"aid", "covert", "challenge", "insurgency"}
    world.powers["USA"].relations["USSR"] = 0.2
    friendly = strategic.request_for(world, "USA", "mode", {"region": region}, {}, crises=True)
    assert friendly is not None and {o.id for o in friendly.options} == {"aid", "covert"}


class Challenger:
    """Focuses on Eastern Europe (Moscow's) and challenges whenever it can."""

    def choose(self, engine: object, request: object, rng: object) -> int:
        ids = [o.id for o in request.options]  # type: ignore[attr-defined]
        for wanted in ("challenge", "eastern_europe"):
            if wanted in ids:
                return ids.index(wanted)
        return 0


def test_an_agents_challenge_starts_a_standoff_it_leads(catalog: Catalog) -> None:
    config = EngineConfig(
        end_quarter=calendar.parse("1956-Q4"), playable_tier=1, generator=True, agents=("USA",)
    )
    engine = CampaignEngine(catalog, config)
    run_episode(engine, {"USA": Challenger()}, 5)  # type: ignore[dict-item]
    started = [e for e in engine.world.fired if e.startswith("gen:standoff:")]
    assert any(generator.parse_id(e)[2][1] == "USA" for e in started), started
