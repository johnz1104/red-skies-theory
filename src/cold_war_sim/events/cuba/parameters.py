"""Illustrative parameters for the Cuba-inspired crisis model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class CubaParameters:
    """Validated, historically inspired—not calibrated—model inputs.

    All defaults are illustrative assumptions except values explicitly tagged as
    normalizations or numerical conveniences in :attr:`provenance`.
    """

    prior_resolve_high: float = 0.45
    prior_readiness_high: float = 0.50
    intelligence_accuracy: float = 0.75

    first_conciliatory_cost_low: float = 0.10
    first_conciliatory_cost_high: float = 0.85
    first_hardline_cost_low: float = 0.70
    first_hardline_cost_high: float = 0.10
    second_reaffirm_cost: float = 0.05
    second_revise_cost: float = 0.45
    inconsistency_cost: float = 0.25

    quarantine_accident_probability: float = 0.015
    air_strike_accident_probability: float = 0.11
    negotiate_accident_probability: float = 0.002
    maintain_accident_probability: float = 0.025
    escalate_accident_probability: float = 0.16
    high_readiness_risk_multiplier: float = 1.30

    missile_value_us: float = 6.0
    missile_value_ussr: float = 5.0
    credibility_value: float = 2.0
    cuba_security_value: float = 2.5
    concession_value: float = 1.5
    conflict_cost_us: float = 12.0
    conflict_cost_ussr: float = 11.0
    catastrophe_loss: float = 100.0
    risk_sensitivity_us: float = 1.0
    risk_sensitivity_ussr: float = 1.0

    probability_tolerance: float = 1e-10
    best_response_tolerance: float = 1e-9

    provenance: ClassVar[dict[str, str]] = {
        "prior_resolve_high": "illustrative_assumption",
        "prior_readiness_high": "illustrative_assumption",
        "intelligence_accuracy": "illustrative_assumption",
        "first_conciliatory_cost_low": "illustrative_assumption",
        "first_conciliatory_cost_high": "illustrative_assumption",
        "first_hardline_cost_low": "illustrative_assumption",
        "first_hardline_cost_high": "illustrative_assumption",
        "second_reaffirm_cost": "illustrative_assumption",
        "second_revise_cost": "illustrative_assumption",
        "inconsistency_cost": "illustrative_assumption",
        "quarantine_accident_probability": "illustrative_assumption",
        "air_strike_accident_probability": "illustrative_assumption",
        "negotiate_accident_probability": "illustrative_assumption",
        "maintain_accident_probability": "illustrative_assumption",
        "escalate_accident_probability": "illustrative_assumption",
        "high_readiness_risk_multiplier": "illustrative_assumption",
        "missile_value_us": "utility_normalization",
        "missile_value_ussr": "utility_normalization",
        "credibility_value": "utility_normalization",
        "cuba_security_value": "utility_normalization",
        "concession_value": "utility_normalization",
        "conflict_cost_us": "illustrative_assumption",
        "conflict_cost_ussr": "illustrative_assumption",
        "catastrophe_loss": "utility_normalization",
        "risk_sensitivity_us": "utility_normalization",
        "risk_sensitivity_ussr": "utility_normalization",
        "probability_tolerance": "numerical_convenience",
        "best_response_tolerance": "numerical_convenience",
    }

    def __post_init__(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not isfinite(float(value)):
                raise ValueError(f"{name} must be a finite number")

        probabilities = (
            "prior_resolve_high",
            "prior_readiness_high",
            "intelligence_accuracy",
            "quarantine_accident_probability",
            "air_strike_accident_probability",
            "negotiate_accident_probability",
            "maintain_accident_probability",
            "escalate_accident_probability",
        )
        for name in probabilities:
            value = float(values[name])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.intelligence_accuracy <= 0.5:
            raise ValueError("intelligence_accuracy must exceed 0.5 in this baseline")

        nonnegative = set(values) - set(probabilities) - {
            "probability_tolerance",
            "best_response_tolerance",
        }
        for name in nonnegative:
            if float(values[name]) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if self.high_readiness_risk_multiplier <= 0.0:
            raise ValueError("high_readiness_risk_multiplier must be positive")
        if self.catastrophe_loss <= 0.0:
            raise ValueError("catastrophe_loss must be positive")
        if self.risk_sensitivity_us <= 0.0 or self.risk_sensitivity_ussr <= 0.0:
            raise ValueError("risk sensitivities must be positive")
        if self.probability_tolerance <= 0.0 or self.best_response_tolerance <= 0.0:
            raise ValueError("solver tolerances must be positive")

    def to_dict(self) -> dict[str, float]:
        """Return deterministic JSON-safe numeric parameters."""

        return {key: float(value) for key, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> CubaParameters:
        """Construct from a strict mapping; unknown fields are rejected."""

        allowed = set(cls.__dataclass_fields__) - {"provenance"}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown Cuba parameters: {sorted(unknown)}")
        numeric_values: dict[str, float] = {}
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{key} must be a real number")
            numeric_values[key] = float(value)
        return cls(**numeric_values)
