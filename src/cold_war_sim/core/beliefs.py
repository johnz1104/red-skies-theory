"""Belief representations and Bayes-rule updates."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .probability import (
    DEFAULT_TOLERANCE,
    ProbabilityDistribution,
    validate_probability,
    validate_probability_distribution,
)
from .types import SerializableMixin, frozen_mapping, validate_stable_id


class OffPathBeliefRequired(ValueError):
    """Raised when Bayes' rule is undefined because an observation has zero mass."""


@dataclass(frozen=True)
class Belief(SerializableMixin):
    information_set_id: str
    probabilities: Mapping[str, float]
    source: str = "Bayes' rule"
    tolerance: float = DEFAULT_TOLERANCE

    def __post_init__(self) -> None:
        validate_stable_id(self.information_set_id, field_name="information-set id")
        distribution = ProbabilityDistribution(
            self.probabilities, tolerance=self.tolerance
        )
        object.__setattr__(self, "probabilities", distribution.probabilities)
        if not self.source:
            raise ValueError("belief source must be non-empty")


@dataclass(frozen=True)
class BeliefSystem(SerializableMixin):
    beliefs: Mapping[str, Belief]

    def __post_init__(self) -> None:
        converted: dict[str, Belief] = {}
        for information_set_id, belief in self.beliefs.items():
            validate_stable_id(
                information_set_id, field_name="belief-system information-set id"
            )
            if not isinstance(belief, Belief):
                raise TypeError("belief-system values must be Belief instances")
            if belief.information_set_id != information_set_id:
                raise ValueError(
                    "belief mapping key must equal Belief.information_set_id"
                )
            converted[information_set_id] = belief
        object.__setattr__(self, "beliefs", frozen_mapping(converted))

    def __getitem__(self, information_set_id: str) -> Belief:
        return self.beliefs[information_set_id]


def bayes_update(
    prior: Sequence[float] | Mapping[str, float],
    likelihood: Sequence[float] | Mapping[str, float],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[float, ...] | dict[str, float]:
    """Apply Bayes' rule to a prior and observation likelihood.

    A zero-probability observation raises :class:`OffPathBeliefRequired`.
    Callers must then supply and record an explicit off-path convention rather
    than silently substituting a posterior.
    """

    if isinstance(prior, Mapping) != isinstance(likelihood, Mapping):
        raise TypeError(
            "prior and likelihood must both be mappings or both be sequences"
        )

    if isinstance(prior, Mapping) and isinstance(likelihood, Mapping):
        prior_is_mapping = True
        prior_keys = tuple(prior.keys())
        if set(prior_keys) != set(likelihood.keys()):
            raise ValueError("prior and likelihood mappings must have identical keys")
        prior_values = validate_probability_distribution(
            [prior[key] for key in prior_keys], tolerance=tolerance, name="prior"
        )
        likelihood_values = tuple(
            validate_probability(likelihood[key], name=f"likelihood[{key!r}]")
            for key in prior_keys
        )
    else:
        if isinstance(prior, Mapping) or isinstance(likelihood, Mapping):
            raise AssertionError("mapping-kind mismatch should have been rejected")
        prior_is_mapping = False
        prior_values = validate_probability_distribution(
            prior, tolerance=tolerance, name="prior"
        )
        likelihood_values = tuple(
            validate_probability(value, name=f"likelihood[{index}]")
            for index, value in enumerate(likelihood)
        )
        if len(prior_values) != len(likelihood_values):
            raise ValueError("prior and likelihood must have the same length")
        prior_keys = ()

    weights = tuple(
        prior_probability * observation_probability
        for prior_probability, observation_probability in zip(
            prior_values, likelihood_values, strict=True
        )
    )
    evidence = math.fsum(weights)
    if evidence <= tolerance:
        raise OffPathBeliefRequired(
            "Bayes' rule is undefined because the observation has zero probability"
        )
    posterior = tuple(weight / evidence for weight in weights)
    # Validate the computed result too; this catches unforeseen numerical errors.
    validate_probability_distribution(
        posterior, tolerance=max(tolerance, 1e-12), name="posterior"
    )
    if prior_is_mapping:
        return {key: posterior[index] for index, key in enumerate(prior_keys)}
    return posterior


def posterior_from_reach(
    reach_probabilities: Mapping[str, float],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, float]:
    """Condition nonnegative node reach weights within one information set."""

    if not reach_probabilities:
        raise ValueError("reach probabilities must not be empty")
    weights: dict[str, float] = {}
    for node_id, value in reach_probabilities.items():
        validate_stable_id(node_id, field_name="reached node id")
        try:
            weight = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"reach probability for {node_id!r} must be numeric"
            ) from error
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"reach probability for {node_id!r} must be finite and nonnegative"
            )
        weights[node_id] = weight
    total = math.fsum(weights.values())
    if total <= tolerance:
        raise OffPathBeliefRequired(
            "Bayes' rule is undefined at an information set with zero reach probability"
        )
    posterior = {node_id: value / total for node_id, value in weights.items()}
    validate_probability_distribution(
        posterior, tolerance=tolerance, name="information-set belief"
    )
    return dict(sorted(posterior.items()))
