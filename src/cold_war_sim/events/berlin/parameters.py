"""Validated parameters for the Berlin finite-horizon bargaining model.

The defaults are illustrative rather than estimates of the 1948--1949 crisis.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Literal, cast

PlayerName = Literal["west", "soviet"]
ResponderTieBreak = Literal["accept", "reject"]
OfferTieBreak = Literal["lowest_west_share", "highest_west_share"]


PARAMETER_PROVENANCE: dict[str, str] = {
    "horizon": "numerical convenience",
    "settlement_grid": "numerical convenience",
    "initial_proposer": "illustrative assumption",
    "settlement_surplus": "normalization",
    "west_discount": "illustrative assumption",
    "soviet_discount": "illustrative assumption",
    "west_delay_cost": "illustrative assumption",
    "soviet_delay_cost": "illustrative assumption",
    "west_reservation": "illustrative assumption",
    "soviet_reservation": "illustrative assumption",
    "west_commitment_floor": "illustrative assumption",
    "soviet_commitment_ceiling": "illustrative assumption",
    "west_commitment_cost": "illustrative assumption",
    "soviet_commitment_cost": "illustrative assumption",
    "base_escalation_risk": "illustrative assumption",
    "escalation_risk_growth": "illustrative assumption",
    "maximum_escalation_risk": "numerical convenience",
    "west_escalation_loss": "illustrative assumption",
    "soviet_escalation_loss": "illustrative assumption",
    "responder_tie_break": "numerical convenience",
    "offer_tie_break": "numerical convenience",
    "comparison_tolerance": "numerical convenience",
}


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _string(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _float_tuple(name: str, value: object) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{name} must be a list or tuple of real numbers")
    return tuple(_finite_float(name, item) for item in value)


@dataclass(frozen=True, slots=True)
class BerlinParameters:
    """Illustrative parameters for alternating-offers bargaining.

    Settlement-grid entries are the West's share of ``settlement_surplus``.
    Discounting and delay costs are both explicit: the former discounts the
    terminal prize while the latter is an additive cost of elapsed periods.
    """

    horizon: int = 4
    settlement_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    initial_proposer: PlayerName = "west"
    settlement_surplus: float = 10.0
    west_discount: float = 0.96
    soviet_discount: float = 0.94
    west_delay_cost: float = 0.15
    soviet_delay_cost: float = 0.15
    west_reservation: float = 0.0
    soviet_reservation: float = 0.0
    west_commitment_floor: float = 0.55
    soviet_commitment_ceiling: float = 0.45
    west_commitment_cost: float = 2.0
    soviet_commitment_cost: float = 2.0
    base_escalation_risk: float = 0.02
    escalation_risk_growth: float = 0.025
    maximum_escalation_risk: float = 0.30
    west_escalation_loss: float = 14.0
    soviet_escalation_loss: float = 14.0
    responder_tie_break: ResponderTieBreak = "accept"
    offer_tie_break: OfferTieBreak = "lowest_west_share"
    comparison_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int):
            raise TypeError("horizon must be an integer")
        if self.horizon < 1:
            raise ValueError("horizon must be at least one")
        if not isinstance(self.settlement_grid, tuple):
            raise TypeError("settlement_grid must be a tuple")
        if not self.settlement_grid:
            raise ValueError("settlement_grid must not be empty")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in self.settlement_grid
        ):
            raise TypeError("settlement_grid entries must be real numbers")
        if any(not isfinite(value) or value < 0.0 or value > 1.0 for value in self.settlement_grid):
            raise ValueError("settlement_grid entries must be finite values in [0, 1]")
        if tuple(sorted(set(self.settlement_grid))) != self.settlement_grid:
            raise ValueError("settlement_grid must be strictly increasing with no duplicates")
        if self.initial_proposer not in ("west", "soviet"):
            raise ValueError("initial_proposer must be 'west' or 'soviet'")
        finite_fields = (
            "settlement_surplus",
            "west_discount",
            "soviet_discount",
            "west_delay_cost",
            "soviet_delay_cost",
            "west_reservation",
            "soviet_reservation",
            "west_commitment_floor",
            "soviet_commitment_ceiling",
            "west_commitment_cost",
            "soviet_commitment_cost",
            "base_escalation_risk",
            "escalation_risk_growth",
            "maximum_escalation_risk",
            "west_escalation_loss",
            "soviet_escalation_loss",
            "comparison_tolerance",
        )
        for field_name in finite_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be a real number")
            if not isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
        if self.settlement_surplus <= 0.0:
            raise ValueError("settlement_surplus must be positive")
        if not 0.0 < self.west_discount <= 1.0 or not 0.0 < self.soviet_discount <= 1.0:
            raise ValueError("discount factors must lie in (0, 1]")
        if self.west_delay_cost < 0.0 or self.soviet_delay_cost < 0.0:
            raise ValueError("delay costs must be nonnegative")
        if not 0.0 <= self.west_commitment_floor <= 1.0:
            raise ValueError("west_commitment_floor must lie in [0, 1]")
        if not 0.0 <= self.soviet_commitment_ceiling <= 1.0:
            raise ValueError("soviet_commitment_ceiling must lie in [0, 1]")
        if self.west_commitment_cost < 0.0 or self.soviet_commitment_cost < 0.0:
            raise ValueError("commitment costs must be nonnegative")
        risks = (self.base_escalation_risk, self.maximum_escalation_risk)
        if any(risk < 0.0 or risk > 1.0 for risk in risks):
            raise ValueError("base and maximum escalation risks must lie in [0, 1]")
        if self.escalation_risk_growth < 0.0:
            raise ValueError("escalation_risk_growth must be nonnegative")
        if self.base_escalation_risk > self.maximum_escalation_risk:
            raise ValueError("base_escalation_risk cannot exceed maximum_escalation_risk")
        if self.west_escalation_loss < 0.0 or self.soviet_escalation_loss < 0.0:
            raise ValueError("escalation losses must be nonnegative")
        if self.responder_tie_break not in ("accept", "reject"):
            raise ValueError("responder_tie_break must be 'accept' or 'reject'")
        if self.offer_tie_break not in ("lowest_west_share", "highest_west_share"):
            raise ValueError("unsupported offer_tie_break")
        if self.comparison_tolerance < 0.0:
            raise ValueError("comparison_tolerance must be nonnegative")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> BerlinParameters:
        """Construct parameters from a JSON/YAML-style mapping.

        Unknown keys are rejected to prevent silently ignored inputs.
        """

        defaults = cls()
        known = set(PARAMETER_PROVENANCE)
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown Berlin parameter(s): {sorted(unknown)}")
        initial = _string("initial_proposer", values.get("initial_proposer", defaults.initial_proposer))
        response_tie = _string(
            "responder_tie_break", values.get("responder_tie_break", defaults.responder_tie_break)
        )
        offer_tie = _string("offer_tie_break", values.get("offer_tie_break", defaults.offer_tie_break))
        return cls(
            horizon=_integer("horizon", values.get("horizon", defaults.horizon)),
            settlement_grid=_float_tuple(
                "settlement_grid", values.get("settlement_grid", defaults.settlement_grid)
            ),
            initial_proposer=cast(PlayerName, initial),
            settlement_surplus=_finite_float(
                "settlement_surplus", values.get("settlement_surplus", defaults.settlement_surplus)
            ),
            west_discount=_finite_float(
                "west_discount", values.get("west_discount", defaults.west_discount)
            ),
            soviet_discount=_finite_float(
                "soviet_discount", values.get("soviet_discount", defaults.soviet_discount)
            ),
            west_delay_cost=_finite_float(
                "west_delay_cost", values.get("west_delay_cost", defaults.west_delay_cost)
            ),
            soviet_delay_cost=_finite_float(
                "soviet_delay_cost", values.get("soviet_delay_cost", defaults.soviet_delay_cost)
            ),
            west_reservation=_finite_float(
                "west_reservation", values.get("west_reservation", defaults.west_reservation)
            ),
            soviet_reservation=_finite_float(
                "soviet_reservation", values.get("soviet_reservation", defaults.soviet_reservation)
            ),
            west_commitment_floor=_finite_float(
                "west_commitment_floor",
                values.get("west_commitment_floor", defaults.west_commitment_floor),
            ),
            soviet_commitment_ceiling=_finite_float(
                "soviet_commitment_ceiling",
                values.get("soviet_commitment_ceiling", defaults.soviet_commitment_ceiling),
            ),
            west_commitment_cost=_finite_float(
                "west_commitment_cost", values.get("west_commitment_cost", defaults.west_commitment_cost)
            ),
            soviet_commitment_cost=_finite_float(
                "soviet_commitment_cost",
                values.get("soviet_commitment_cost", defaults.soviet_commitment_cost),
            ),
            base_escalation_risk=_finite_float(
                "base_escalation_risk",
                values.get("base_escalation_risk", defaults.base_escalation_risk),
            ),
            escalation_risk_growth=_finite_float(
                "escalation_risk_growth",
                values.get("escalation_risk_growth", defaults.escalation_risk_growth),
            ),
            maximum_escalation_risk=_finite_float(
                "maximum_escalation_risk",
                values.get("maximum_escalation_risk", defaults.maximum_escalation_risk),
            ),
            west_escalation_loss=_finite_float(
                "west_escalation_loss",
                values.get("west_escalation_loss", defaults.west_escalation_loss),
            ),
            soviet_escalation_loss=_finite_float(
                "soviet_escalation_loss",
                values.get("soviet_escalation_loss", defaults.soviet_escalation_loss),
            ),
            responder_tie_break=cast(ResponderTieBreak, response_tie),
            offer_tie_break=cast(OfferTieBreak, offer_tie),
            comparison_tolerance=_finite_float(
                "comparison_tolerance", values.get("comparison_tolerance", defaults.comparison_tolerance)
            ),
        )

    def escalation_risk(self, period: int) -> float:
        """Return rejection escalation risk in the zero-indexed period."""

        if period < 0 or period >= self.horizon:
            raise ValueError("period is outside the bargaining horizon")
        return min(
            self.maximum_escalation_risk,
            self.base_escalation_risk + period * self.escalation_risk_growth,
        )
