"""Structured baseline/counterfactual comparisons."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from cold_war_sim import __version__
from cold_war_sim.core.types import (
    SerializableMixin,
    canonical_json,
    deep_frozen_json,
    frozen_mapping,
    to_serializable,
    validate_stable_id,
)

from .feasibility import FeasibilityReport
from .outcomes import OutcomeDistribution
from .responses import EquilibriumCertificate
from .specs import (
    CounterfactualSpec,
    CounterfactualStatus,
    ResponseModel,
    SolutionConcept,
)


@dataclass(frozen=True)
class SerializationMetadata(SerializableMixin):
    package_version: str = __version__
    schema_version: str = "1.0"
    deterministic_key_order: bool = True
    nonfinite_values_allowed: bool = False

    def __post_init__(self) -> None:
        if not self.package_version.strip() or not self.schema_version.strip():
            raise ValueError("serialization versions must not be empty")
        if not isinstance(self.deterministic_key_order, bool) or not isinstance(
            self.nonfinite_values_allowed, bool
        ):
            raise TypeError("serialization flags must be booleans")
        if not self.deterministic_key_order:
            raise ValueError("counterfactual results require deterministic key order")
        if self.nonfinite_values_allowed:
            raise ValueError("counterfactual results cannot allow nonfinite values")


@dataclass(frozen=True)
class CounterfactualEvaluationResult(SerializableMixin):
    status: CounterfactualStatus
    baseline: Mapping[str, object]
    intervention: Mapping[str, object]
    response_model: str
    solution_concept: str
    baseline_strategy_set: tuple[Mapping[str, object], ...]
    counterfactual_strategy_set: tuple[Mapping[str, object], ...]
    baseline_outcome_distribution: OutcomeDistribution
    counterfactual_outcome_distribution: OutcomeDistribution
    outcome_feature_changes: Mapping[str, float]
    expected_utility_changes: Mapping[str, float]
    escalation_probability_change: float
    catastrophe_probability_change: float
    feasibility_report: FeasibilityReport
    equilibrium_certificates: tuple[EquilibriumCertificate, ...]
    multiplicity: int
    equilibrium_selection_treatment: Mapping[str, object]
    best_response_gaps: Mapping[str, float]
    warnings: tuple[str, ...]
    solver_runtime_seconds: float
    tolerance: float
    serialization_metadata: SerializationMetadata = SerializationMetadata()
    schema_version: str = "1.0"
    document_type: str = "counterfactual_result"

    def __post_init__(self) -> None:
        if not isinstance(self.status, CounterfactualStatus):
            raise TypeError("status must be a CounterfactualStatus")
        if self.response_model not in {item.value for item in ResponseModel}:
            raise ValueError("unsupported response model in counterfactual result")
        if self.solution_concept not in {item.value for item in SolutionConcept}:
            raise ValueError("unsupported solution concept in counterfactual result")
        if self.schema_version != "1.0" or self.document_type != "counterfactual_result":
            raise ValueError("unsupported counterfactual-result schema identity")
        baseline = deep_frozen_json(self.baseline)
        intervention = deep_frozen_json(self.intervention)
        if not isinstance(baseline, Mapping) or not isinstance(intervention, Mapping):
            raise TypeError("baseline and intervention must be mappings")
        baseline_strategies = tuple(
            deep_frozen_json(strategy) for strategy in self.baseline_strategy_set
        )
        counterfactual_strategies = tuple(
            deep_frozen_json(strategy) for strategy in self.counterfactual_strategy_set
        )
        if any(not isinstance(strategy, Mapping) for strategy in baseline_strategies):
            raise TypeError("baseline strategies must be mappings")
        if any(
            not isinstance(strategy, Mapping)
            for strategy in counterfactual_strategies
        ):
            raise TypeError("counterfactual strategies must be mappings")
        if not baseline_strategies:
            raise ValueError("a counterfactual result requires a baseline strategy set")
        if (
            isinstance(self.multiplicity, bool)
            or not isinstance(self.multiplicity, int)
            or self.multiplicity < 0
        ):
            raise ValueError("multiplicity must be a nonnegative integer")
        object.__setattr__(
            self,
            "equilibrium_selection_treatment",
            deep_frozen_json(self.equilibrium_selection_treatment),
        )
        if self.multiplicity != len(counterfactual_strategies):
            raise ValueError("multiplicity must equal the counterfactual strategy-set size")
        if not isinstance(self.equilibrium_selection_treatment, Mapping):
            raise TypeError("equilibrium_selection_treatment must be a mapping")
        feature_changes: dict[str, float] = {}
        for feature, raw in self.outcome_feature_changes.items():
            validate_stable_id(feature, field_name="changed outcome-feature id")
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError("outcome-feature changes must be finite")
            feature_changes[feature] = value
        utility_changes: dict[str, float] = {}
        for player, raw in self.expected_utility_changes.items():
            validate_stable_id(player, field_name="utility-change player id")
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError("expected-utility changes must be finite")
            utility_changes[player] = value
        gaps: dict[str, float] = {}
        for player, raw in self.best_response_gaps.items():
            validate_stable_id(player, field_name="best-response-gap player id")
            value = float(raw)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("best-response gaps must be finite and nonnegative")
            gaps[player] = value
        for name in (
            "escalation_probability_change",
            "catastrophe_probability_change",
            "solver_runtime_seconds",
            "tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name in {"solver_runtime_seconds", "tolerance"} and value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
            if name == "tolerance" and value == 0.0:
                raise ValueError("tolerance must be positive")
            object.__setattr__(self, name, value)
        if not -1.0 <= self.escalation_probability_change <= 1.0:
            raise ValueError("escalation probability change must lie in [-1, 1]")
        if not -1.0 <= self.catastrophe_probability_change <= 1.0:
            raise ValueError("catastrophe probability change must lie in [-1, 1]")
        certificates = tuple(self.equilibrium_certificates)
        if len(certificates) > self.multiplicity:
            raise ValueError("there cannot be more certificates than solutions")
        warnings = tuple(self.warnings)
        if any(not isinstance(warning, str) or not warning.strip() for warning in warnings):
            raise ValueError("counterfactual warnings must be nonempty strings")
        if not self.feasibility_report.feasible:
            if self.status is not CounterfactualStatus.INFEASIBLE_POLICY:
                raise ValueError("an infeasible report requires INFEASIBLE_POLICY status")
        elif self.status is CounterfactualStatus.INFEASIBLE_POLICY:
            raise ValueError("INFEASIBLE_POLICY requires a failed feasibility audit")
        if self.status in {
            CounterfactualStatus.NO_SUPPORTED_SOLUTION,
            CounterfactualStatus.NO_PURE_PBE_FOUND,
        } and self.multiplicity != 0:
            raise ValueError("a no-solution status requires zero multiplicity")
        if (
            self.status is CounterfactualStatus.MULTIPLE_SUPPORTED_EQUILIBRIA
            and self.multiplicity < 2
        ):
            raise ValueError("multiple-equilibrium status requires multiplicity of at least two")
        if (
            self.status is CounterfactualStatus.DEPENDENT_ON_OFF_PATH_BELIEFS
            and not any(item.depends_on_off_path_beliefs for item in certificates)
        ):
            raise ValueError(
                "off-path-dependence status requires a dependent certificate"
            )
        object.__setattr__(self, "baseline", baseline)
        object.__setattr__(self, "intervention", intervention)
        object.__setattr__(self, "baseline_strategy_set", baseline_strategies)
        object.__setattr__(
            self, "counterfactual_strategy_set", counterfactual_strategies
        )
        object.__setattr__(self, "outcome_feature_changes", frozen_mapping(feature_changes))
        object.__setattr__(self, "expected_utility_changes", frozen_mapping(utility_changes))
        object.__setattr__(self, "best_response_gaps", frozen_mapping(gaps))
        object.__setattr__(self, "equilibrium_certificates", certificates)
        object.__setattr__(self, "warnings", warnings)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "document_type": self.document_type,
            "status": self.status.value,
            "baseline": to_serializable(self.baseline),
            "intervention": to_serializable(self.intervention),
            "response_model": self.response_model,
            "solution_concept": self.solution_concept,
            "baseline_strategy_set": [
                to_serializable(item) for item in self.baseline_strategy_set
            ],
            "counterfactual_strategy_set": [
                to_serializable(item) for item in self.counterfactual_strategy_set
            ],
            "baseline_outcome_distribution": self.baseline_outcome_distribution.to_dict(),
            "counterfactual_outcome_distribution": (
                self.counterfactual_outcome_distribution.to_dict()
            ),
            "outcome_feature_changes": dict(self.outcome_feature_changes),
            "expected_utility_changes": dict(self.expected_utility_changes),
            "escalation_probability_change": self.escalation_probability_change,
            "catastrophe_probability_change": self.catastrophe_probability_change,
            "feasibility_report": self.feasibility_report.to_dict(),
            "equilibrium_certificates": [
                certificate.to_dict() for certificate in self.equilibrium_certificates
            ],
            "multiplicity": self.multiplicity,
            "equilibrium_selection_treatment": to_serializable(
                self.equilibrium_selection_treatment
            ),
            "best_response_gaps": dict(self.best_response_gaps),
            "warnings": list(self.warnings),
            "solver_runtime_seconds": self.solver_runtime_seconds,
            "tolerance": self.tolerance,
            "serialization_metadata": self.serialization_metadata.to_dict(),
        }


def evaluate_comparison(
    *,
    spec: CounterfactualSpec,
    baseline_strategies: Sequence[Mapping[str, object]],
    counterfactual_strategies: Sequence[Mapping[str, object]],
    baseline_outcomes: OutcomeDistribution,
    counterfactual_outcomes: OutcomeDistribution,
    baseline_utilities: Mapping[str, float],
    counterfactual_utilities: Mapping[str, float],
    feasibility: FeasibilityReport,
    certificates: Sequence[EquilibriumCertificate] = (),
    best_response_gaps: Mapping[str, float] | None = None,
    warnings: Sequence[str] = (),
    started_at: float | None = None,
) -> CounterfactualEvaluationResult:
    """Build a comparison while preserving both sides and all solution multiplicity."""

    if not baseline_strategies:
        raise ValueError("baseline_strategies must not be empty")
    if set(baseline_utilities) != set(counterfactual_utilities):
        raise ValueError("baseline and counterfactual utilities must cover the same players")
    converted_baseline_utilities = {
        player: float(value) for player, value in baseline_utilities.items()
    }
    converted_counterfactual_utilities = {
        player: float(value) for player, value in counterfactual_utilities.items()
    }
    for player in converted_baseline_utilities:
        validate_stable_id(player, field_name="utility player id")
    if not converted_baseline_utilities or not all(
        math.isfinite(value)
        for value in (
            *converted_baseline_utilities.values(),
            *converted_counterfactual_utilities.values(),
        )
    ):
        raise ValueError("baseline and counterfactual utilities must be finite and nonempty")
    claims_equilibrium = (
        spec.response_model is ResponseModel.REEQUILIBRATE or bool(certificates)
    )
    certificate_concepts = {certificate.equilibrium_concept for certificate in certificates}
    if claims_equilibrium and certificates and certificate_concepts != {
        spec.solution_concept.value
    }:
        raise ValueError("equilibrium certificates must match the requested solution concept")
    if any(certificate.tolerance > spec.tolerance for certificate in certificates):
        raise ValueError("certificate tolerance cannot be looser than the requested tolerance")
    if len(certificates) > len(counterfactual_strategies):
        raise ValueError("there cannot be more certificates than counterfactual solutions")
    if certificates and len(certificates) == len(counterfactual_strategies):
        certified_profiles = sorted(
            canonical_json(certificate.strategy_profile)
            for certificate in certificates
        )
        returned_profiles = sorted(
            canonical_json(strategy) for strategy in counterfactual_strategies
        )
        if certified_profiles != returned_profiles:
            raise ValueError(
                "equilibrium certificates must identify the returned strategy profiles"
            )
    if not feasibility.feasible:
        status = CounterfactualStatus.INFEASIBLE_POLICY
    elif not counterfactual_strategies:
        status = (
            CounterfactualStatus.NO_PURE_PBE_FOUND
            if spec.solution_concept.value == "PURE_PBE"
            else CounterfactualStatus.NO_SUPPORTED_SOLUTION
        )
    elif claims_equilibrium and (
        len(certificates) != len(counterfactual_strategies)
        or any(
            not certificate.exact
            or certificate.best_response_gap > spec.tolerance
            for certificate in certificates
        )
    ):
        status = CounterfactualStatus.UNVERIFIED_APPROXIMATION
    elif any(certificate.depends_on_off_path_beliefs for certificate in certificates):
        status = CounterfactualStatus.DEPENDENT_ON_OFF_PATH_BELIEFS
    elif claims_equilibrium and len(counterfactual_strategies) > 1:
        status = CounterfactualStatus.MULTIPLE_SUPPORTED_EQUILIBRIA
    else:
        status = CounterfactualStatus.VALID_COUNTERFACTUAL
    common_features = set(baseline_outcomes.outcomes[0].features.registry.common)
    feature_changes = {}
    for feature in sorted(common_features):
        try:
            change = counterfactual_outcomes.expected_common(feature) - baseline_outcomes.expected_common(feature)
        except (TypeError, ValueError):
            continue
        if math.isfinite(change):
            feature_changes[feature] = change
    runtime = 0.0 if started_at is None else max(0.0, time.perf_counter() - started_at)
    return CounterfactualEvaluationResult(
        status,
        spec.baseline.to_dict(),
        spec.intervention.to_dict(),
        spec.response_model.value,
        spec.solution_concept.value,
        tuple(baseline_strategies),
        tuple(counterfactual_strategies),
        baseline_outcomes,
        counterfactual_outcomes,
        feature_changes,
        {
            player: converted_counterfactual_utilities[player]
            - converted_baseline_utilities[player]
            for player in converted_baseline_utilities
        },
        counterfactual_outcomes.probability("military_escalation")
        - baseline_outcomes.probability("military_escalation"),
        counterfactual_outcomes.probability("catastrophic_escalation")
        - baseline_outcomes.probability("catastrophic_escalation"),
        feasibility,
        tuple(certificates),
        len(counterfactual_strategies),
        spec.equilibrium_handling.to_dict(),
        best_response_gaps or {},
        tuple(warnings),
        runtime,
        spec.tolerance,
    )
