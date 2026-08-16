"""Validated illustrative parameters for the arms-competition models.

The numerical defaults in this module are modeling assumptions.  They are not
historical estimates.  Utilities are cardinal only within each constructed
model and should not be compared across event modules.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, ClassVar

ILLUSTRATIVE = "illustrative assumption; not historically calibrated"
NUMERICAL_CONVENIENCE = "numerical convenience"


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_players(player_names: tuple[str, str]) -> None:
    if len(player_names) != 2:
        raise ValueError("player_names must contain exactly two names")
    if any(not isinstance(name, str) or not name.strip() for name in player_names):
        raise ValueError("player names must be nonempty strings")
    if player_names[0] == player_names[1]:
        raise ValueError("player names must be distinct")


@dataclass(frozen=True, slots=True)
class RichardsonParameters:
    """Coefficients for two coupled affine Richardson arms-race equations.

    With state ``(x, y)``, the equations are

    ``x' = -fatigue_0*x + reaction_0*y + grievance_0``
    ``y' =  reaction_1*x - fatigue_1*y + grievance_1``.

    Reaction coefficients are nonnegative and fatigue coefficients are
    strictly positive.  Grievance terms may be negative in counterfactual
    numerical exercises; trajectories report when that causes the state to
    leave the modeled nonnegative domain.
    """

    reaction_0: float = 0.42
    reaction_1: float = 0.40
    fatigue_0: float = 0.36
    fatigue_1: float = 0.34
    grievance_0: float = 0.45
    grievance_1: float = 0.42
    player_names: tuple[str, str] = ("United States", "Soviet Union")

    ASSUMPTION_LABEL: ClassVar[str] = ILLUSTRATIVE

    def __post_init__(self) -> None:
        numeric_names = (
            "reaction_0",
            "reaction_1",
            "fatigue_0",
            "fatigue_1",
            "grievance_0",
            "grievance_1",
        )
        for name in numeric_names:
            object.__setattr__(self, name, _finite_number(name, getattr(self, name)))
        for name in ("reaction_0", "reaction_1"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        for name in ("fatigue_0", "fatigue_1"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be strictly positive")

        names = tuple(self.player_names)
        _validate_players(names)  # type: ignore[arg-type]
        object.__setattr__(self, "player_names", names)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> RichardsonParameters:
        if values is None:
            return cls()
        data = dict(values)
        if "player_names" in data:
            raw_names = data["player_names"]
            if not isinstance(raw_names, (list, tuple)):
                raise TypeError("player_names must be a two-item sequence")
            data["player_names"] = tuple(raw_names)
        try:
            return cls(**data)
        except TypeError as exc:
            raise ValueError(f"invalid Richardson parameter keys: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["player_names"] = list(self.player_names)
        result["provenance"] = self.provenance()
        return result

    @staticmethod
    def provenance() -> dict[str, str]:
        return {
            "reaction_0": ILLUSTRATIVE,
            "reaction_1": ILLUSTRATIVE,
            "fatigue_0": ILLUSTRATIVE,
            "fatigue_1": ILLUSTRATIVE,
            "grievance_0": ILLUSTRATIVE,
            "grievance_1": ILLUSTRATIVE,
        }
