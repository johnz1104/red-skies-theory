from __future__ import annotations

import pytest

from cold_war_sim.campaign import calendar
from cold_war_sim.campaign.conditions import CompileContext, ConditionError, compile_condition
from cold_war_sim.campaign.effects import EffectContext, LeaderSpec, compile_effects
from cold_war_sim.campaign.paths import PathError, compile_path
from cold_war_sim.campaign.state import NEUTRAL, POWERS, default_world


def test_calendar_round_trip() -> None:
    assert calendar.fmt(calendar.parse("1962-Q4")) == "1962-Q4"
    assert calendar.parse("1946-Q1") - calendar.parse("1945-Q4") == 1
    assert calendar.quarter_of_year(calendar.parse("1950-Q1")) == 1
    assert calendar.progress(calendar.START) == 0.0 and calendar.progress(calendar.END) == 1.0
    with pytest.raises(ValueError):
        calendar.parse("1962-Q5")


def test_paths_validate_and_clamp() -> None:
    world = default_world(calendar.START)
    ref = compile_path("powers.USSR.stability")
    ref.set(world, 1.7)
    assert ref.get(world) == 1.0
    relation = compile_path("powers.USA.relations.USSR")
    relation.add(world, -0.4)
    assert world.powers["USSR"].relations["USA"] == pytest.approx(-0.4)
    influence = compile_path("regions.caribbean.influence.USSR")
    influence.set(world, 0.9)
    compile_path("regions.caribbean.influence.USA").set(world, 0.9)
    assert sum(world.regions["caribbean"].influence.values()) <= 1.0 + 1e-12
    for bad in (
        "powers.XX.stability",
        "powers.USA.leader",
        "regions.mars.stability",
        "tension.x",
        "foo",
    ):
        with pytest.raises(PathError):
            compile_path(bad)


def _cond(spec: object) -> object:
    return compile_condition(spec, CompileContext())


def test_condition_truth_tables() -> None:
    world = default_world(calendar.parse("1962-Q4"))
    world.flags.add("cuba_aligned")
    world.powers["USSR"].nuclear = 2.0
    world.powers["USA"].nuclear = 10.0
    world.outcomes["berlin"] = "agreement"
    world.outcome_tags["berlin"] = ("USSR:committed",)
    world.fired["berlin"] = world.quarter
    cases = [
        ({"after": "1962-Q1"}, True),
        ({"before": "1962-Q3"}, False),
        ({"flag": "cuba_aligned"}, True),
        ({"no_flag": "cuba_aligned"}, False),
        (
            {
                "compare": {
                    "path": "powers.USSR.nuclear",
                    "op": "lt",
                    "path_rhs": "powers.USA.nuclear",
                    "scale": 0.5,
                }
            },
            True,
        ),
        ({"compare": {"path": "powers.USSR.nuclear", "op": "ge", "value": 3}}, False),
        ({"outcome": {"event": "berlin", "in": ["agreement"]}}, True),
        ({"outcome": {"event": "berlin", "in": ["USSR:committed"]}}, True),
        ({"outcome": {"event": "berlin", "in": ["breakdown"]}}, False),
        ({"any": [{"flag": "nope"}, {"fired": "berlin"}]}, True),
        ({"not": {"fired": "berlin"}}, False),
        ([{"flag": "cuba_aligned"}, {"after": "1960-Q1"}], True),
        ({"status": {"power": "PRC", "in": ["active"]}}, True),
        ({"controller": {"region": "japan", "in": [NEUTRAL]}}, True),
    ]
    for spec, expected in cases:
        assert _cond(spec)(world) is expected, spec  # type: ignore[operator]


def test_focus_and_diplomacy_history_conditions() -> None:
    world = default_world(calendar.parse("1953-Q3"))
    world.powers["USA"].focus_history.append((1952, "persian_gulf", "covert"))
    world.powers["PRC"].diplomacy_history.append((1953, "pressure", "USSR"))
    assert _cond(
        {"focus": {"power": "USA", "region": "persian_gulf", "mode": "covert", "within_years": 1}}
    )(world)  # type: ignore[operator]
    assert not _cond({"focus": {"power": "USA", "mode": "aid", "within_years": 3}})(world)  # type: ignore[operator]
    assert _cond(
        {"diplomacy": {"power": "PRC", "action": "pressure", "target": "USSR", "within_years": 0}}
    )(world)  # type: ignore[operator]


@pytest.mark.parametrize(
    "spec",
    [
        {"unknown_op": 1},
        {"compare": {"path": "tension", "op": "between", "value": 1}},
        {"compare": {"path": "tension", "op": "lt"}},
        {"status": {"power": "ROC", "in": ["active"]}},
        {"after": "1962"},
        {"flag": []},
    ],
)
def test_bad_conditions_raise(spec: object) -> None:
    with pytest.raises((ConditionError, ValueError)):
        _cond(spec)


def test_effects_apply_with_scaling_and_leaders() -> None:
    ctx = CompileContext()
    effects = compile_effects(
        [
            {"flag": ["a", "b"]},
            {"unflag": "a"},
            {"add": {"path": "powers.USA.prestige", "value": 0.4, "scale_by": "share"}},
            {
                "shift_influence": {
                    "region": "germany",
                    "toward": "USSR",
                    "amount": 0.5,
                    "scale_by": {"metric": "share", "invert": True},
                }
            },
            {"set_controller": {"region": "caribbean", "to": "USSR"}},
            {"relations": {"a": "PRC", "b": "USSR", "add": -0.5}},
            {"leader": "khrushchev"},
            {"status": {"power": "PRC", "to": "eliminated"}},
            {"tension": 0.9},
            {"start_event": "next_one"},
        ],
        ctx,
    )
    assert effects.flags_set == {"a", "b"}
    assert ctx.event_refs == {"next_one"} and ctx.leaders == {"khrushchev"}
    world = default_world(calendar.START)
    world.powers["USA"].prestige = 0.5
    leaders = {"khrushchev": LeaderSpec("khrushchev", "USSR", "N. K.", 0.6, 0.8, 0.6)}
    ectx = EffectContext(leaders=leaders, metrics={"share": 0.25})
    effects.apply(world, ectx)
    assert world.flags == {"b"}
    assert world.powers["USA"].prestige == pytest.approx(0.6)
    assert world.regions["germany"].influence["USSR"] == pytest.approx(0.375)
    assert world.regions["caribbean"].controller == "USSR"
    assert world.powers["USSR"].relations["PRC"] == pytest.approx(-0.5)
    assert (
        world.powers["USSR"].leader == "khrushchev" and world.powers["USSR"].risk_tolerance == 0.8
    )
    assert world.powers["PRC"].status == "eliminated"
    assert world.tension == 1.0
    assert ectx.requested_events == ["next_one"]
    for region in world.regions.values():
        assert sum(region.influence[p] for p in POWERS) <= 1.0 + 1e-9


def test_end_campaign_effect_marks_nuclear_war() -> None:
    world = default_world(calendar.START)
    compile_effects([{"end_campaign": "nuclear_war"}], CompileContext()).apply(
        world, EffectContext(leaders={})
    )
    assert world.ended == "nuclear_war" and world.nuclear_war


def test_a_lottery_draws_by_chance_and_never_by_order() -> None:
    effects = compile_effects(
        [
            {
                "lottery": {
                    "chance": {"metric": "share"},
                    "win": [{"flag": "won"}],
                    "lose": [{"flag": "lost"}],
                }
            }
        ],
        CompileContext(),
    )
    assert effects.flags_set == {"won", "lost"} and len(effects.lotteries) == 1
    assert effects.lotteries[0](EffectContext(leaders={}, metrics={"share": 0.67})) == 0.67

    def outcome(seed: int, share: float, forced: dict[int, bool] | None = None) -> set[str]:
        world = default_world(calendar.START)
        ectx = EffectContext(
            leaders={}, metrics={"share": share}, seed=seed, source="talks", forced=forced
        )
        effects.apply(world, ectx)
        return set(world.flags)

    assert outcome(3, 0.5) == outcome(3, 0.5)  # the same seed, source and quarter
    assert outcome(3, 1.0) == {"won"} and outcome(3, 0.0) == {"lost"}
    assert outcome(3, 1.0, forced={0: False}) == {"lost"}  # derived payoffs fix the branch
    wins = sum(outcome(seed, 0.3) == {"won"} for seed in range(400))
    assert 90 < wins < 150  # about 0.3 of 400


@pytest.mark.parametrize(
    "spec",
    [
        {"lottery": {"chance": 1.5, "win": [], "lose": []}},
        {"lottery": {"chance": 0.5, "win": []}},
        {
            "lottery": {
                "chance": 0.5,
                "win": [{"lottery": {"chance": 0.5, "win": [], "lose": []}}],
                "lose": [],
            }
        },
    ],
)
def test_bad_lotteries_raise(spec: object) -> None:
    with pytest.raises(ConditionError):
        compile_effects([spec], CompileContext())
