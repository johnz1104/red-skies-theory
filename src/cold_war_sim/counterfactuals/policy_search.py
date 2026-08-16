"""Exact bounded search over finite pure policies."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from cold_war_sim.core.types import SerializableMixin, frozen_mapping

from .constraints import Constraint, ConstraintContext
from .feasibility import FeasibilityReport, PurePolicy
from .objectives import CandidateMetrics, Objective
from .pareto import ParetoPoint, pareto_frontier
from .policy_space import PolicySpace


class PolicySearchStatus(StrEnum):
    COMPLETE = "COMPLETE"
    CAPACITY_EXCEEDED = "CAPACITY_EXCEEDED"
    NO_FEASIBLE_POLICY = "NO_FEASIBLE_POLICY"


@dataclass(frozen=True)
class EvaluatedPolicy(SerializableMixin):
    policy_id: str
    policy: Mapping[str, str]
    metrics: CandidateMetrics
    objective_score: tuple[float, ...]
    feasibility: FeasibilityReport
    independently_verified: bool
    verified_scenario_ids: tuple[str, ...]
    differences_from_baseline: Mapping[str, tuple[str | None, str | None]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy", frozen_mapping(self.policy))
        object.__setattr__(
            self, "differences_from_baseline", frozen_mapping(self.differences_from_baseline)
        )


@dataclass(frozen=True)
class PolicySearchResult(SerializableMixin):
    status: PolicySearchStatus
    exact: bool
    estimated_policy_count: int
    evaluated_policy_count: int
    feasible_policy_count: int
    retained: tuple[EvaluatedPolicy, ...]
    pareto_frontier: tuple[EvaluatedPolicy, ...]
    ties_at_cutoff: int
    narrower_legal_configuration: Mapping[str, object] | None
    warnings: tuple[str, ...]


PolicyEvaluator = Callable[[PurePolicy, Mapping[str, object]], CandidateMetrics]
PolicyAuditor = Callable[[PurePolicy], FeasibilityReport]
PolicyVerifier = Callable[[PurePolicy, Mapping[str, object], CandidateMetrics], bool]


def _policy_id(policy: PurePolicy) -> str:
    return "|".join(f"{key}={value}" for key, value in policy.actions.items())


def _differences(
    policy: Mapping[str, str], baseline: Mapping[str, str]
) -> Mapping[str, tuple[str | None, str | None]]:
    return frozen_mapping(
        {
            key: (baseline.get(key), policy.get(key))
            for key in sorted(set(policy) | set(baseline))
            if baseline.get(key) != policy.get(key)
        }
    )


def search(
    *,
    policy_space: PolicySpace,
    objective: Objective,
    evaluator: PolicyEvaluator,
    auditor: PolicyAuditor,
    verifier: PolicyVerifier,
    baseline_policy: Mapping[str, str],
    constraints: Sequence[Constraint] = (),
    parameter_scenarios: Sequence[
        Mapping[str, object] | tuple[str, Mapping[str, object]]
    ] = ({},),
    aggregation: str = "expected",
    maximum_search_size: int = 100_000,
    top_k: int = 10,
    tolerance: float = 1e-9,
) -> PolicySearchResult:
    """Enumerate every policy or stop before evaluation with a capacity status."""

    if maximum_search_size < 1 or top_k < 1:
        raise ValueError("search-size and top-k limits must be positive")
    if aggregation not in {"expected", "worst_case"}:
        raise ValueError("aggregation must be 'expected' or 'worst_case'")
    if not parameter_scenarios:
        raise ValueError("parameter_scenarios must not be empty")
    named_scenarios: list[tuple[str, Mapping[str, object]]] = []
    for index, scenario in enumerate(parameter_scenarios):
        if isinstance(scenario, tuple):
            if len(scenario) != 2:
                raise ValueError("named parameter scenarios must be (id, parameters)")
            scenario_id, parameters = scenario
        else:
            scenario_id, parameters = str(index), scenario
        if not scenario_id or any(scenario_id == known for known, _ in named_scenarios):
            raise ValueError("parameter scenario ids must be nonempty and unique")
        named_scenarios.append((scenario_id, parameters))
    size = policy_space.size
    if size > maximum_search_size:
        return PolicySearchResult(
            PolicySearchStatus.CAPACITY_EXCEEDED,
            True,
            size,
            0,
            0,
            (),
            (),
            0,
            policy_space.narrower_configuration(maximum_search_size),
            ("exact enumeration was not started; narrow the legal policy space",),
        )
    cache: dict[tuple[str, int], CandidateMetrics] = {}
    evaluated: list[EvaluatedPolicy] = []
    for policy in policy_space.enumerate():
        feasibility = auditor(policy)
        if not feasibility.feasible:
            continue
        scenario_metrics = []
        scenario_verification: list[str] = []
        for scenario_index, (scenario_id, scenario) in enumerate(named_scenarios):
            key = (_policy_id(policy), scenario_index)
            if key not in cache:
                cache[key] = evaluator(policy, scenario)
            scenario_metrics.append(cache[key])
            verified = verifier(policy, scenario, cache[key])
            if not verified:
                scenario_metrics = []
                break
            scenario_verification.append(scenario_id)
        if not scenario_metrics:
            continue
        first = scenario_metrics[0]
        if len(scenario_metrics) == 1:
            metrics = first
        else:
            players = tuple(first.expected_utilities)
            scenario_utilities = {
                scenario_id: metric.expected_utilities
                for (scenario_id, _), metric in zip(
                    named_scenarios, scenario_metrics, strict=True
                )
            }
            if aggregation == "worst_case":
                aggregate_utilities = {
                    player: min(metric.expected_utilities[player] for metric in scenario_metrics)
                    for player in players
                }
            else:
                aggregate_utilities = {
                    player: sum(metric.expected_utilities[player] for metric in scenario_metrics)
                    / len(scenario_metrics)
                    for player in players
                }
            if aggregation == "worst_case":
                escalation = max(
                    metric.escalation_probability for metric in scenario_metrics
                )
                catastrophe = max(
                    metric.catastrophe_probability for metric in scenario_metrics
                )
                settlement = min(
                    metric.negotiated_settlement_probability
                    for metric in scenario_metrics
                )
            else:
                escalation = sum(
                    metric.escalation_probability for metric in scenario_metrics
                ) / len(scenario_metrics)
                catastrophe = sum(
                    metric.catastrophe_probability for metric in scenario_metrics
                ) / len(scenario_metrics)
                settlement = sum(
                    metric.negotiated_settlement_probability
                    for metric in scenario_metrics
                ) / len(scenario_metrics)
            metrics = CandidateMetrics(
                aggregate_utilities,
                scenario_utilities,
                escalation,
                catastrophe,
                settlement,
            )
        context = ConstraintContext(metrics, policy.actions, baseline_policy)
        if not all(constraint.check(context, tolerance=tolerance) for constraint in constraints):
            continue
        evaluated.append(
            EvaluatedPolicy(
                _policy_id(policy),
                policy.actions,
                metrics,
                objective.score(metrics),
                feasibility,
                True,
                tuple(scenario_verification),
                _differences(policy.actions, baseline_policy),
            )
        )
    evaluated.sort(key=lambda item: (tuple(-x for x in item.objective_score), item.policy_id))
    if not evaluated:
        return PolicySearchResult(
            PolicySearchStatus.NO_FEASIBLE_POLICY,
            True,
            size,
            size,
            0,
            (),
            (),
            0,
            None,
            (),
        )
    cutoff_index = min(top_k, len(evaluated)) - 1
    cutoff = evaluated[cutoff_index].objective_score
    retained = tuple(
        item
        for item in evaluated
        if item.objective_score > cutoff
        or all(abs(a - b) <= tolerance for a, b in zip(item.objective_score, cutoff, strict=True))
    )
    ties = max(0, len(retained) - top_k)
    complete_frontier = frontier(evaluated)
    return PolicySearchResult(
        PolicySearchStatus.COMPLETE,
        True,
        size,
        size,
        len(evaluated),
        retained,
        complete_frontier,
        ties,
        None,
        (),
    )


def frontier(candidates: Sequence[EvaluatedPolicy], *, tolerance: float = 1e-9) -> tuple[EvaluatedPolicy, ...]:
    points = tuple(ParetoPoint(item.policy_id, item.metrics.expected_utilities) for item in candidates)
    retained_ids = {point.id for point in pareto_frontier(points, tolerance=tolerance)}
    return tuple(item for item in candidates if item.policy_id in retained_ids)
