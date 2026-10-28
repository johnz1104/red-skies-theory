"""World dynamics: diplomacy, containment and the growth effect of postures."""

from __future__ import annotations

import copy

import pytest

from cold_war_sim.campaign import Catalog
from cold_war_sim.campaign.dynamics import DT, DynamicsParams, step


def _world(catalog: Catalog):  # type: ignore[no-untyped-def]
    world = catalog.initial_world()
    for state in world.powers.values():
        state.status, state.diplomacy, state.focus_region = "active", None, None
    world.powers["USA"].relations["USSR"] = world.powers["USSR"].relations["USA"] = 0.0
    return world


def _relation_after(catalog: Catalog, usa: str | None, ussr: str | None) -> float:
    world = _world(catalog)
    world.powers["USA"].diplomacy, world.powers["USSR"].diplomacy = usa, ussr
    step(world, DynamicsParams.from_catalog(catalog), {})
    assert world.powers["USA"].relations["USSR"] == world.powers["USSR"].relations["USA"]
    return world.powers["USA"].relations["USSR"]


def test_warmth_needs_both_sides_but_hostility_does_not(fixture_catalog: Catalog) -> None:
    step_size = DynamicsParams.from_catalog(fixture_catalog)["relations_step"]
    assert _relation_after(fixture_catalog, None, None) == 0.0
    assert _relation_after(fixture_catalog, "engage:USSR", None) == pytest.approx(step_size / 2)
    assert _relation_after(fixture_catalog, "engage:USSR", "engage:USA") == pytest.approx(step_size)
    assert _relation_after(fixture_catalog, "pressure:USSR", None) == pytest.approx(-step_size)
    assert _relation_after(fixture_catalog, "engage:USSR", "pressure:USA") == pytest.approx(
        -step_size / 2
    )


def test_pressure_contains_a_rivals_influence_gains(fixture_catalog: Catalog) -> None:
    params = DynamicsParams.from_catalog(fixture_catalog)
    gains = {}
    for pressured in (False, True):
        world = _world(fixture_catalog)
        world.powers["USSR"].focus_region, world.powers["USSR"].focus_mode = "middle_east", "aid"
        world.powers["USA"].diplomacy = "pressure:USSR" if pressured else None
        before = world.regions["middle_east"].influence["USSR"]
        step(world, params, {})
        gains[pressured] = world.regions["middle_east"].influence["USSR"] - before
    assert gains[True] < gains[False]
    lost = params["aid_rate"] * params["containment"] * DT * _reach(fixture_catalog)
    assert gains[False] - gains[True] == pytest.approx(lost, rel=0.05)


def _reach(catalog: Catalog) -> float:
    return float(catalog.initial_world().powers["USSR"].economy) ** 0.5


def test_posture_growth_scales_with_the_share_of_gdp_moved(fixture_catalog: Catalog) -> None:
    params = DynamicsParams.from_catalog(fixture_catalog)
    economies = {}
    for posture in ("economy", "conventional"):
        world = _world(fixture_catalog)
        world.powers["USA"].posture = posture
        start = world.powers["USA"].economy
        step(world, params, {})
        economies[posture] = world.powers["USA"].economy / start - 1.0
    burden = copy.deepcopy(fixture_catalog.initial_world()).powers["USA"].burden
    expected = params["investment_growth"] * burden * (0.55 - 0.25) * DT
    assert economies["economy"] - economies["conventional"] == pytest.approx(expected, rel=1e-6)
