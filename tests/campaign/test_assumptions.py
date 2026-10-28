"""The assumptions register: addresses, ranges, liveness, draws and sampling (E42)."""

from __future__ import annotations

import copy
import math
from typing import Any

import pytest

from cold_war_sim.campaign import CampaignEngine, Catalog, EngineConfig, run_episode
from cold_war_sim.campaign.assumptions import (
    Number,
    Plan,
    RegisterError,
    Sampling,
    _glob,
    check,
    inventory,
    load_register,
    parse_register,
    plan,
)
from cold_war_sim.campaign.catalog import CatalogError, get_number, split_address
from cold_war_sim.campaign.policies import HistoricalPolicy, RandomPolicy
from cold_war_sim.campaign.robustness import Finding, FindingResult, spearman
from cold_war_sim.campaign.serialization import stable_hash

from .conftest import FIXTURE_EVENTS

RANGES = {
    "judgment": {"factor": 2.0, "logit": 1.0, "intercept": 0.25},
    "literature": {"factor": 1.333, "logit": 0.5, "intercept": 0.125},
    "calibrated": {"factor": 1.333, "logit": 0.5, "intercept": 0.125},
    "data": {"factor": 1.1, "logit": 0.25, "intercept": 0.05},
}


def _register(classes: list[dict[str, Any]], entries: dict[str, Any] | None = None) -> Any:
    return parse_register({"ranges": RANGES, "classes": classes, "entries": entries or {}})


def _number(address: str, value: float, **context: Any) -> Number:
    return Number(address, value, "test.yaml", 1, "", context)


@pytest.fixture(scope="module")
def numbers() -> list[Number]:
    return inventory(event_files=[FIXTURE_EVENTS])


@pytest.fixture(scope="module")
def fixture_plan(fixture_catalog: Catalog, numbers: list[Number]) -> Plan:
    return plan(fixture_catalog, load_register(), numbers)


def test_addresses_name_documents_and_paths(fixture_catalog: Catalog) -> None:
    assert split_address("goals:kappa") == ("goals", ("kappa",))
    assert split_address("dynamics:relations_step") == ("powers", ("dynamics", "relations_step"))
    doc, keys = split_address("event:fx_crisis/model/config/script/0/roles/USSR/p_high_resolve")
    assert doc == "event:fx_crisis"
    raw = fixture_catalog.raw
    assert raw is not None
    assert get_number(raw.document(doc), keys) == 0.3
    # binding terms are keyed by dotted world paths, hence "/" in addresses
    doc, keys = split_address("event:fx_crisis/model/bindings/0/terms/powers.USA.credibility")
    assert get_number(raw.document(doc), keys) == 0.5
    with pytest.raises(CatalogError):
        split_address("kappa")


def test_globs_cross_segments_only_with_two_stars() -> None:
    assert _glob("event:*/hazard").fullmatch("event:x/hazard")
    assert not _glob("event:*/hazard").fullmatch("event:x/y/hazard")
    assert _glob("event:*/when/**/compare/value").fullmatch("event:x/when/compare/value")
    assert _glob("event:*/when/**/compare/value").fullmatch("event:x/when/all/1/compare/value")
    assert _glob("goals:**").fullmatch("goals:components/allies/height")


def test_ranges_follow_the_kind_and_the_basis() -> None:
    register = _register(
        [
            {"match": "t:mag", "group": "g", "kind": "magnitude"},
            {"match": "t:sig", "group": "g", "kind": "signed"},
            {"match": "t:p", "group": "g", "kind": "probability"},
            {"match": "t:lit", "group": "g", "kind": "probability", "basis": "literature"},
            {"match": "t:rel", "group": "g", "kind": "relation"},
            {"match": "t:int", "group": "g", "kind": "intercept"},
        ]
    )
    mag = register.resolve(_number("t:mag", 0.15))
    assert (mag.low, mag.high) == pytest.approx((0.075, 0.3))
    sig = register.resolve(_number("t:sig", -0.05))
    assert (sig.low, sig.high) == pytest.approx((-0.1, -0.025))
    logit = math.log(0.7 / 0.3)
    p = register.resolve(_number("t:p", 0.7))
    assert p.low == pytest.approx(1 / (1 + math.exp(-(logit - 1))))
    assert p.high == pytest.approx(1 / (1 + math.exp(-(logit + 1))))
    assert (p.low, p.high) == pytest.approx((0.462, 0.864), abs=1e-3)
    lit = register.resolve(_number("t:lit", 0.7))
    assert (lit.low, lit.high) == pytest.approx((0.586, 0.794), abs=1e-3)
    rel = register.resolve(_number("t:rel", 0.8))
    assert -1.0 < rel.low < 0.8 < rel.high < 1.0  # type: ignore[operator]
    intercept = register.resolve(_number("t:int", 0.4, clamp=[0.2, 1.5]))
    assert (intercept.low, intercept.high) == pytest.approx((0.4 - 0.325, 0.4 + 0.325))


def test_zeros_bounds_and_structure_are_fixed() -> None:
    register = _register(
        [
            {"match": "t:tier", "group": "g", "fixed": "structural"},
            {"match": "t:*", "group": "g", "kind": "probability"},
        ]
    )
    assert register.resolve(_number("t:tier", 1)).fixed == "structural"
    assert register.resolve(_number("t:one", 1.0)).fixed == "bound"
    assert register.resolve(_number("t:zero", 0.0)).fixed == "bound"
    zero = _register([{"match": "t:*", "group": "g", "kind": "signed"}]).resolve(
        _number("t:x", 0.0)
    )
    assert zero.fixed == "zero" and not zero.drawable


def test_an_uncovered_number_and_a_stale_entry_are_reported(
    fixture_catalog: Catalog, numbers: list[Number]
) -> None:
    register = _register([{"match": "goals:**", "group": "goals"}])
    with pytest.raises(RegisterError, match="no class covers"):
        register.resolve(_number("powers:relations/USA-USSR", 0.3))
    stale = _register(
        [{"match": "**", "group": "all", "kind": "signed"}],
        {"goals:kappa": {"value": 99.0}},
    )
    problems = check(fixture_catalog, stale, numbers, extremes=False)
    assert any("goals:kappa" in p and "holds 100.0" in p for p in problems)


def test_entries_override_their_class(fixture_plan: Plan) -> None:
    kappa = next(a for a in fixture_plan.items if a.address == "goals:kappa")
    assert kappa.curated and (kappa.low, kappa.high) == (50.0, 200.0)
    assert kappa.placeholder


def test_liveness_marks_derived_bound_and_on_rails_numbers(
    fixture_catalog: Catalog, numbers: list[Number]
) -> None:
    full = plan(fixture_catalog, load_register(), numbers)
    tier1 = plan(fixture_catalog, load_register(), numbers, playable_tier=1)
    canal = "event:fx_canal/model/config/roles/EGY/values/a_prevails"
    assert full.status[canal] == "live" and tier1.status[canal] == "dead:rails"
    # a binding without a base starts from the config value, which stays live
    assert full.status["event:fx_crisis/model/config/roles/USA/values/b_prevails"] == "live"
    # outcome effects of an event on rails still act
    effects = [a for a in numbers if a.address.startswith("event:fx_canal/outcomes/")]
    assert all(tier1.status[a.address] == "live" for a in effects)


def test_draws_are_deterministic_and_address_stable(fixture_plan: Plan) -> None:
    first = fixture_plan.draw(5, Sampling())
    assert first == fixture_plan.draw(5, Sampling())
    assert first != fixture_plan.draw(6, Sampling())
    # a plan with one number fewer draws the others exactly as before
    smaller = Plan(fixture_plan.items[1:], fixture_plan.status)
    drawn = smaller.draw(5, Sampling())
    unlinked = [a.address for a in fixture_plan.items[1:] if a.linked is None and a.capped is None]
    assert all(drawn[k] == first[k] for k in unlinked if k in drawn)
    for item in fixture_plan.live():
        value = first[item.address]
        assert item.low is not None and item.high is not None
        if item.linked is None and item.capped is None:
            assert item.low - 1e-9 <= value <= item.high + 1e-9, item.address


def test_spread_zero_draws_the_catalogs_values(fixture_plan: Plan) -> None:
    values = fixture_plan.draw(3, Sampling(spread=0.0))
    by_address = {a.address: a.value for a in fixture_plan.items}
    assert all(values[k] == pytest.approx(by_address[k], rel=1e-8) for k in values)


def test_linked_sets_keep_their_total(fixture_plan: Plan) -> None:
    values = fixture_plan.draw(11, Sampling(select=("goals",)))
    row = {k: v for k, v in values.items() if k.startswith("goals:weights/USA/0/weights/")}
    assert len(row) >= 4 and sum(row.values()) == pytest.approx(1.0, abs=1e-6)
    base = {a.address: a.value for a in fixture_plan.items}
    assert any(row[k] != base[k] for k in row)
    setting = fixture_plan.setting("goals:weights/USA/0/weights/security", 0.4)
    assert setting["goals:weights/USA/0/weights/security"] == 0.4
    assert sum(setting.values()) == pytest.approx(1.0, abs=1e-6)
    postures = {
        k: v
        for k, v in fixture_plan.draw(2, Sampling()).items()
        if k.startswith("powers:postures/nuclear/")
    }
    assert sum(postures.values()) == pytest.approx(1.0, abs=1e-6)


def test_region_shares_are_capped(fixture_plan: Plan) -> None:
    for seed in range(20):
        values = fixture_plan.draw(seed, Sampling(select=("regions",)))
        shares = [
            v for k, v in values.items() if k.startswith("regions:regions/germany/influence/")
        ]
        assert sum(shares) <= 1.0 + 1e-9


def test_selection_by_group_event_and_glob(fixture_plan: Plan) -> None:
    goals = fixture_plan.draw(1, Sampling(select=("goals",)))
    # the goals group includes the goal reward's terminal terms
    assert goals and all(k.startswith(("goals:", "powers:rewards/")) for k in goals)
    event = fixture_plan.draw(1, Sampling(select=("event:fx_crisis",)))
    assert event and all(k.startswith("event:fx_crisis/") for k in event)
    glob = fixture_plan.draw(1, Sampling(select=("dynamics:*",)))
    assert glob and all(k.startswith("dynamics:") for k in glob)


def test_a_variant_patches_only_what_changes(fixture_catalog: Catalog) -> None:
    assert fixture_catalog.variant({}) is fixture_catalog
    variant = fixture_catalog.variant({"goals:kappa": 50.0, "dynamics:relations_step": 0.04})
    assert variant.goals.kappa == 50.0 and fixture_catalog.goals.kappa == 100.0
    assert variant.dynamics["relations_step"] == 0.04
    assert variant.events is not fixture_catalog.events
    assert variant.events["fx_crisis"] is fixture_catalog.events["fx_crisis"]
    changed = fixture_catalog.variant(
        {"event:fx_crisis/model/config/script/0/roles/USSR/p_high_resolve": 0.6}
    )
    major = changed.events["fx_crisis"].major
    assert major is not None and major.config["script"][0]["roles"]["USSR"]["p_high_resolve"] == 0.6
    assert major.roles == fixture_catalog.events["fx_crisis"].major.roles  # type: ignore[union-attr]
    base = fixture_catalog.events["fx_crisis"].major
    assert base is not None and base.config["script"][0]["roles"]["USSR"]["p_high_resolve"] == 0.3
    with pytest.raises(CatalogError, match="no number"):
        fixture_catalog.variant({"goals:no_such_number": 1.0})


def test_sampling_off_leaves_the_trace_unchanged(fixture_catalog: Catalog) -> None:
    plain = CampaignEngine(fixture_catalog, EngineConfig(chance_mode="historical"))
    first = run_episode(plain, HistoricalPolicy(), 0)
    again = CampaignEngine(fixture_catalog, EngineConfig(chance_mode="historical", sampling=None))
    assert run_episode(again, HistoricalPolicy(), 0).trace_hash == first.trace_hash
    assert first.assumptions == {}
    # spread 0 replays the same trace after the draw's own record
    zero = CampaignEngine(
        fixture_catalog, EngineConfig(chance_mode="historical", sampling=Sampling(spread=0.0))
    )
    result = run_episode(zero, HistoricalPolicy(), 0)
    assert "assumptions" in zero.s.trace[0]
    assert stable_hash(zero.s.trace[1:]) == first.trace_hash
    assert result.trace_hash != first.trace_hash


def test_sampled_episodes_are_deterministic_and_snapshots_keep_the_draw(
    fixture_catalog: Catalog,
) -> None:
    engine = CampaignEngine(fixture_catalog, EngineConfig(sampling=Sampling()))
    a = run_episode(engine, RandomPolicy(), 4)
    b = run_episode(engine, RandomPolicy(), 4)
    c = run_episode(engine, RandomPolicy(), 5)
    assert a.trace_hash == b.trace_hash and a.assumptions == b.assumptions
    assert a.assumptions != c.assumptions
    fixed = CampaignEngine(fixture_catalog, EngineConfig(sampling=Sampling(draw=2)))
    assert (
        run_episode(fixed, RandomPolicy(), 1).assumptions
        == run_episode(fixed, RandomPolicy(), 2).assumptions
    )
    engine.reset(7)
    snapshot = engine.snapshot()
    variant = engine.catalog
    engine.reset(8)
    assert engine.catalog is not variant
    engine.restore(snapshot)
    assert engine.catalog is variant and engine.s.assumptions == snapshot.assumptions
    plain = CampaignEngine(fixture_catalog)
    plain.restore(copy.deepcopy(snapshot))
    assert plain.catalog is variant


def test_findings_measure_rates_and_claims() -> None:
    runs = [
        {
            "outcomes": {"x": "a"},
            "tags": {"x": ("offer:USA=aid",)},
            "fired": {"x": 1},
            "ended": "campaign_end",
            "returns": {"USA": 1.0},
        },
        {
            "outcomes": {"x": "b"},
            "tags": {"x": ()},
            "fired": {"x": 1},
            "ended": "nuclear_war",
            "returns": {"USA": -1.0},
        },
        {"outcomes": {}, "tags": {}, "fired": {}, "ended": "campaign_end", "returns": {"USA": 0.0}},
    ]
    outcome = Finding("f", "", {"outcome": "x", "label": "a"}, {"at_least": 0.3})
    assert outcome.value(runs) == pytest.approx(1 / 3) and outcome.holds(outcome.value(runs))
    given = Finding("g", "", {"outcome": "x", "label": "!b", "given": "fired"}, {"at_least": 0.6})
    assert given.value(runs) == pytest.approx(0.5) and not given.holds(given.value(runs))
    tag = Finding("t", "", {"tag": "x", "has": "offer:USA=aid*"}, {"at_least": 0.3})
    assert tag.value(runs) == pytest.approx(1 / 3)
    ended = Finding("e", "", {"ended": "nuclear_war"}, {"at_most": 0.05})
    assert not ended.holds(ended.value(runs))
    assert Finding("r", "", {"return": "USA"}, {"at_least": 0.0}).value(runs) == pytest.approx(0.0)
    assert spearman([1, 2, 3, 4], [2, 4, 6, 9]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [1, 1, 1, 1]) == 0.0


def test_robust_verdicts_allow_for_seed_noise() -> None:
    claim = Finding("f", "", {"ended": "x"}, {"at_least": 0.5})
    assert FindingResult(claim, 0.4, [0.6] * 10, [0.6] * 10).verdict == "fails at base"
    assert FindingResult(claim, 0.7, [0.6] * 9 + [0.0], [0.6] * 10).verdict == "robust"
    # the base holds in every seed set, the draws in 8 of 10: fragile
    assert FindingResult(claim, 0.7, [0.6] * 8 + [0.0] * 2, [0.6] * 10).verdict == "fragile"
    # the base itself holds in only 7 of 10 seed sets: 6 of 10 draws is within the noise...
    base_sets = [0.6] * 7 + [0.2] * 3
    assert FindingResult(claim, 0.6, [0.6] * 6 + [0.2] * 4, base_sets).verdict == "noisy"
    # ...but 2 of 10 is not
    assert FindingResult(claim, 0.6, [0.6] * 2 + [0.2] * 8, base_sets).verdict == "fragile"
    # draws without a value (a given-fired event that never fired) are counted apart
    absent = FindingResult(claim, 0.7, [None, None] + [0.6] * 8, [0.6] * 10)
    assert absent.absent == 2 and absent.holds_share == 1.0 and absent.verdict == "robust"
