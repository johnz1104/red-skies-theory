from __future__ import annotations

import copy

import pytest

from cold_war_sim.games.protocol import ImmutableParams, Outcome
from cold_war_sim.records import Record


class Point(Record):
    """A record with two attributes."""

    def __init__(self, x: float, y: float = 0.0) -> None:
        self.x = x
        self.y = y


class Fixed(ImmutableParams, Record):
    """A parameter object: shared, never copied."""

    def __init__(self, values: dict[str, float]) -> None:
        self.values = values


def test_records_compare_and_print_by_their_attributes() -> None:
    assert Point(1.0, 2.0) == Point(1.0, 2.0)
    assert Point(1.0, 2.0) != Point(1.0, 3.0)
    assert Point(1.0) != Fixed({"x": 1.0})  # another class is never equal
    assert repr(Point(1.0, 2.0)) == "Point(x=1.0, y=2.0)"
    assert Point(1.0, 2.0).to_dict() == {"x": 1.0, "y": 2.0}


def test_records_are_not_hashable() -> None:
    with pytest.raises(TypeError):
        hash(Point(1.0))


def test_replace_copies_and_leaves_the_original_alone() -> None:
    original = Point(1.0, 2.0)
    changed = original.replace(y=5.0)
    assert (changed.x, changed.y) == (1.0, 5.0)
    assert (original.x, original.y) == (1.0, 2.0)
    with pytest.raises(TypeError, match="no attributes"):
        original.replace(z=1.0)


def test_replace_copies_a_parameter_object_that_copying_would_share() -> None:
    original = Fixed({"a": 1.0})
    assert copy.deepcopy(original) is original  # parameter objects are shared
    changed = original.replace(values={"a": 2.0})
    assert changed is not original
    assert original.values == {"a": 1.0}


def test_outcome_defaults_and_equality() -> None:
    outcome = Outcome("deal")
    assert (outcome.label, outcome.tags, dict(outcome.metrics)) == ("deal", (), {})
    assert Outcome("deal", ("a",), {"share": 0.5}) == Outcome("deal", ("a",), {"share": 0.5})
