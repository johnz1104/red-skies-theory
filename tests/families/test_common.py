from __future__ import annotations

import random
from typing import Any

import pytest

from cold_war_sim.families import ConfigError, build_family
from cold_war_sim.families.common import Section, distribution, value_table
from cold_war_sim.games import CHANCE

from . import configs


def test_a_section_reads_keys_with_defaults_under_their_dotted_names() -> None:
    data = {"cost": 2, "side": "a", "share": 0.25, "rounds": 3, "on": True, "items": (1, 2)}
    with Section("game.roles.X", data) as s:
        assert s.name == "game.roles.X" and s.at("cost") == "game.roles.X.cost"
        assert s.nonnegative("cost") == 2.0
        assert s.number("missing", -1) == -1.0  # a default makes the key optional
        assert s.choice("side", ("a", "b")) == "a"
        assert s.probability("share") == 0.25
        assert s.positive_int("rounds") == 3
        assert s.flag("on") is True and s.flag("off") is False
        assert s.sequence("items") == [1, 2]
        assert s.has("cost") and not s.has("absent")


def test_a_section_rejects_keys_nobody_asked_for() -> None:
    section = Section("game", {"cost": 1, "typo": 2})
    with pytest.raises(ConfigError, match=r"game: unknown keys \['typo'\]"), section as s:
        s.nonnegative("cost")


def test_a_section_reports_the_first_error_not_the_unread_keys() -> None:
    section = Section("game", {"cost": -1, "typo": 2})
    with pytest.raises(ConfigError, match=r"game\.cost must be nonnegative"), section as s:
        s.nonnegative("cost")


@pytest.mark.parametrize(
    ("read", "message"),
    [
        (lambda s: s.number("cost"), "game: missing required key 'cost'"),
        (lambda s: s.choice("side", ("a", "b")), r"game.side must be one of \('a', 'b'\)"),
        (lambda s: s.role("actor", {"X"}), "game.actor refers to unknown role 'Y'"),
        (lambda s: s.probability("share"), r"game.share must lie in \[0, 1\]"),
        (lambda s: s.positive_int("rounds"), "game.rounds must be a positive integer"),
        (lambda s: s.sequence("items"), "game.items must be a list"),
        (lambda s: s.section("block"), "game.block must be a mapping"),
    ],
)
def test_a_section_names_the_key_in_its_errors(read: Any, message: str) -> None:
    data = {"side": "c", "actor": "Y", "share": 1.5, "rounds": 0, "items": 3, "block": 4}
    with pytest.raises(ConfigError, match=message):
        read(Section("game", data))


def test_a_section_must_be_a_mapping() -> None:
    with pytest.raises(ConfigError, match="game must be a mapping"):
        Section("game", [1, 2])


def test_distribution_fills_missing_levels_and_must_sum_to_one() -> None:
    assert distribution("d", {"high": 0.25, "low": 0.75}, ("high", "mid", "low")) == {
        "high": 0.25,
        "mid": 0.0,
        "low": 0.75,
    }
    with pytest.raises(ConfigError, match="d must sum to 1"):
        distribution("d", {"high": 0.5}, ("high", "low"))


def test_value_table_keeps_the_label_order() -> None:
    table = value_table("v", {"lose": -1, "win": 2}, ("win", "lose"))
    assert list(table.items()) == [("win", 2.0), ("lose", -1.0)]
    with pytest.raises(ConfigError, match="missing values"):
        value_table("v", {"win": 2}, ("win", "lose"))


def _snapshot(value: Any, seen: set[int] | None = None) -> Any:
    """*value* as nested plain data, following objects into their attributes."""

    seen = set() if seen is None else seen
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if id(value) in seen:
        return "<seen>"
    seen = seen | {id(value)}
    if isinstance(value, dict):
        return {str(k): _snapshot(v, seen) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_snapshot(v, seen) for v in value]
    names = [*getattr(value, "__dict__", {}), *getattr(value, "__slots__", ())]
    if names:
        return {k: _snapshot(getattr(value, k), seen) for k in names}
    return repr(value)


@pytest.mark.parametrize(
    ("family", "config"),
    [
        ("crisis", configs.signaling_crisis()),
        ("crisis", configs.warning_crisis()),
        ("crisis", configs.coalition_crisis()),
        ("bargaining", configs.bargaining_config()),
        ("intervention", configs.intervention_config()),
        ("attrition", configs.attrition_config()),
        ("attrition", configs.attrition_offers_config(3, approval=True, consent=True)),
        ("alignment", configs.alignment_config()),
    ],
)
def test_playing_a_game_never_changes_its_parameters(family: str, config: dict[str, Any]) -> None:
    # parameter objects are shared by every copy of a state, so nothing may write to them
    game = build_family(family, config)
    before = _snapshot(game.params)
    for seed in range(40):
        rng = random.Random(seed)
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.current_player() == CHANCE:
                action = rng.choice([a for a, _ in state.chance_outcomes()])
            else:
                action = rng.choice(state.legal_actions())
            state.clone().apply_action(action)
            state.apply_action(action)
        state.returns()
        state.outcome()
    assert _snapshot(game.params) == before


@pytest.mark.parametrize("value", ["no", "false", 0, 1, 0.0, None])
def test_a_flag_is_only_true_or_false(value: Any) -> None:
    section = Section("game", {"on": value})
    with pytest.raises(ConfigError, match=r"game\.on must be true or false"), section as s:
        s.flag("on")
