"""Counterfactual analysis for model-feasible finite-game alternatives."""

from .evaluator import CounterfactualEvaluationResult, evaluate_comparison
from .feasibility import FeasibilityReport, PurePolicy, audit_policy
from .interventions import (
    ActionExpansion,
    ActionRestriction,
    Commitment,
    CommitmentScope,
    InformationIntervention,
    ParameterIntervention,
    PolicyReplacement,
    StructuralTransformation,
)
from .outcomes import OutcomeDistribution, OutcomeFeatures
from .policy_search import PolicySearchResult, search
from .specs import CounterfactualSpec, ResponseModel

__all__ = [
    "ActionExpansion",
    "ActionRestriction",
    "Commitment",
    "CommitmentScope",
    "CounterfactualEvaluationResult",
    "CounterfactualSpec",
    "FeasibilityReport",
    "InformationIntervention",
    "OutcomeDistribution",
    "OutcomeFeatures",
    "ParameterIntervention",
    "PolicyReplacement",
    "PolicySearchResult",
    "PurePolicy",
    "ResponseModel",
    "StructuralTransformation",
    "audit_policy",
    "evaluate_comparison",
    "search",
]
