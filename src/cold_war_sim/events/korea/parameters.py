"""Validated illustrative parameters for the Korean warning game."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Literal, cast

EntryTieBreak = Literal["stay_out", "intervene"]


PARAMETER_PROVENANCE: dict[str, str] = {
    "prior_high_resolve": "illustrative assumption",
    "low_warning_cost": "illustrative assumption",
    "high_warning_cost": "illustrative assumption",
    "low_threat_sensitivity": "illustrative assumption",
    "high_threat_sensitivity": "illustrative assumption",
    "low_intervention_fixed_cost": "illustrative assumption",
    "high_intervention_fixed_cost": "illustrative assumption",
    "intervention_effectiveness": "illustrative assumption",
    "restraint_threat": "normalization",
    "limited_threat": "illustrative assumption",
    "aggressive_threat": "illustrative assumption",
    "restraint_receiver_benefit": "normalization",
    "limited_receiver_benefit": "illustrative assumption",
    "aggressive_receiver_benefit": "illustrative assumption",
    "restraint_conflict_cost": "illustrative assumption",
    "limited_conflict_cost": "illustrative assumption",
    "aggressive_conflict_cost": "illustrative assumption",
    "restraint_intervention_cost": "illustrative assumption",
    "limited_intervention_cost": "illustrative assumption",
    "aggressive_intervention_cost": "illustrative assumption",
    "off_path_high_belief": "illustrative off-path convention",
    "entry_tie_break": "numerical convenience",
    "comparison_tolerance": "numerical convenience",
}


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _string(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


@dataclass(frozen=True, slots=True)
class KoreaParameters:
    """Parameters for warning, advance, and continuation entry choices.

    Every number is illustrative, normalized, or a numerical convention. No
    field is presented as an estimate of historical preferences or beliefs.
    """

    prior_high_resolve: float = 0.25
    low_warning_cost: float = 3.0
    high_warning_cost: float = 0.3
    low_threat_sensitivity: float = 0.8
    high_threat_sensitivity: float = 1.4
    low_intervention_fixed_cost: float = 4.0
    high_intervention_fixed_cost: float = 2.0
    intervention_effectiveness: float = 0.8
    restraint_threat: float = 0.0
    limited_threat: float = 2.0
    aggressive_threat: float = 5.0
    restraint_receiver_benefit: float = 1.0
    limited_receiver_benefit: float = 3.0
    aggressive_receiver_benefit: float = 6.0
    restraint_conflict_cost: float = 4.0
    limited_conflict_cost: float = 6.0
    aggressive_conflict_cost: float = 9.0
    restraint_intervention_cost: float = 0.5
    limited_intervention_cost: float = 1.0
    aggressive_intervention_cost: float = 2.0
    off_path_high_belief: float = 0.25
    entry_tie_break: EntryTieBreak = "stay_out"
    comparison_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        numeric_names = tuple(name for name in PARAMETER_PROVENANCE if name != "entry_tie_break")
        for name in numeric_names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 < self.prior_high_resolve < 1.0:
            raise ValueError("prior_high_resolve must lie strictly between zero and one")
        nonnegative = (
            self.low_warning_cost,
            self.high_warning_cost,
            self.low_threat_sensitivity,
            self.high_threat_sensitivity,
            self.low_intervention_fixed_cost,
            self.high_intervention_fixed_cost,
            self.restraint_threat,
            self.limited_threat,
            self.aggressive_threat,
            self.restraint_conflict_cost,
            self.limited_conflict_cost,
            self.aggressive_conflict_cost,
            self.restraint_intervention_cost,
            self.limited_intervention_cost,
            self.aggressive_intervention_cost,
            self.comparison_tolerance,
        )
        if any(value < 0.0 for value in nonnegative):
            raise ValueError("costs, threats, sensitivities, and tolerance must be nonnegative")
        if not 0.0 <= self.intervention_effectiveness <= 1.0:
            raise ValueError("intervention_effectiveness must lie in [0, 1]")
        if not 0.0 <= self.off_path_high_belief <= 1.0:
            raise ValueError("off_path_high_belief must lie in [0, 1]")
        if self.entry_tie_break not in ("stay_out", "intervene"):
            raise ValueError("entry_tie_break must be 'stay_out' or 'intervene'")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> KoreaParameters:
        """Build from input data while rejecting misspelled fields."""

        defaults = cls()
        known = set(PARAMETER_PROVENANCE)
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown Korea parameter(s): {sorted(unknown)}")

        def number(name: str) -> float:
            return _finite_float(name, values.get(name, getattr(defaults, name)))

        tie = _string("entry_tie_break", values.get("entry_tie_break", defaults.entry_tie_break))
        return cls(
            prior_high_resolve=number("prior_high_resolve"),
            low_warning_cost=number("low_warning_cost"),
            high_warning_cost=number("high_warning_cost"),
            low_threat_sensitivity=number("low_threat_sensitivity"),
            high_threat_sensitivity=number("high_threat_sensitivity"),
            low_intervention_fixed_cost=number("low_intervention_fixed_cost"),
            high_intervention_fixed_cost=number("high_intervention_fixed_cost"),
            intervention_effectiveness=number("intervention_effectiveness"),
            restraint_threat=number("restraint_threat"),
            limited_threat=number("limited_threat"),
            aggressive_threat=number("aggressive_threat"),
            restraint_receiver_benefit=number("restraint_receiver_benefit"),
            limited_receiver_benefit=number("limited_receiver_benefit"),
            aggressive_receiver_benefit=number("aggressive_receiver_benefit"),
            restraint_conflict_cost=number("restraint_conflict_cost"),
            limited_conflict_cost=number("limited_conflict_cost"),
            aggressive_conflict_cost=number("aggressive_conflict_cost"),
            restraint_intervention_cost=number("restraint_intervention_cost"),
            limited_intervention_cost=number("limited_intervention_cost"),
            aggressive_intervention_cost=number("aggressive_intervention_cost"),
            off_path_high_belief=number("off_path_high_belief"),
            entry_tie_break=cast(EntryTieBreak, tie),
            comparison_tolerance=number("comparison_tolerance"),
        )

    @property
    def prior(self) -> dict[str, float]:
        return {
            "low_resolve": 1.0 - self.prior_high_resolve,
            "high_resolve": self.prior_high_resolve,
        }
