"""Exact Cuba continuation PBE solver."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from itertools import product
from typing import Any, NotRequired, TypedDict, cast

from cold_war_sim.core.results import SolverResult
from cold_war_sim.events.cuba.model import (
    COMMUNICATION_HISTORIES,
    FINAL_RESPONSE_ORDER,
    SOVIET_TYPES,
    TYPE_ORDER,
    CubaModel,
    FinalResponse,
    InitialAction,
    Intelligence,
)
from cold_war_sim.solvers.pure_pbe import SignalingEquilibrium, SignalingGame, enumerate_pure_pbe


class _EquilibriumOutcomes(TypedDict):
    escalation_probability: float
    catastrophe_probability: float
    expected_utility_us: float
    expected_utility_ussr: float
    initial_action_frequencies: dict[str, float]
    ordered_signal_frequencies: dict[str, float]
    first_signal_frequencies: dict[str, float]
    second_signal_frequencies: dict[str, float]
    response_frequencies: dict[str, float]
    outcome_category_frequencies: dict[str, float]
    posterior_truth_probability: float
    paths: list[dict[str, object]]


class _SerializedEquilibrium(TypedDict):
    intelligence: str
    initial_action: str
    equilibrium_index: int
    classification: str
    sender_strategy: dict[str, str]
    receiver_strategy: dict[str, str]
    posteriors: dict[str, dict[str, float]]
    reach_probabilities: dict[str, float]
    expected_utilities: dict[str, float]
    sender_deviation_gains: dict[str, float]
    receiver_deviation_gains: dict[str, float]
    max_best_response_gain: float
    agent_form_nashconv: float
    off_path_messages: list[str]
    depends_on_off_path_beliefs: bool
    outcomes: _EquilibriumOutcomes
    sequential_communication_policy: dict[str, dict[str, object]]


class _ContinuationRecord(TypedDict):
    intelligence: str
    initial_action: str
    status: str
    equilibria: list[_SerializedEquilibrium]
    solver: NotRequired[dict[str, Any]]
    solver_metadata: NotRequired[object]
    solver_runtime_seconds: NotRequired[float]


class _RootAssessment(TypedDict):
    assessment_index: int
    intelligence: str
    continuation_selection_by_initial_action: dict[str, int]
    initial_action_values_us: dict[str, float]
    selected_initial_action: str
    selected_continuation_equilibrium_index: int
    selected_classification: str
    selected_outcomes: _EquilibriumOutcomes
    initial_action_tie: bool


class _CompleteAssessment(TypedDict):
    assessment_index: int
    root_assessment_by_intelligence: dict[str, int]
    selected_initial_action_by_intelligence: dict[str, str]
    regime_by_intelligence: dict[str, str]
    escalation_probability: float
    catastrophe_probability: float
    expected_utility_us: float
    expected_utility_ussr: float
    posterior_truth_probability: float


class _SolverSummary(TypedDict):
    name: str
    version: str
    equilibrium_concept: str
    found: bool
    multiple_solutions: bool
    convergence_or_exactness_status: str
    best_response_gap: float | None
    off_path_belief_convention: str
    runtime_seconds: float
    warnings: list[str]
    assumptions: list[str]
    seed: int | None


class _IncompleteInformationSolution(TypedDict):
    solver: _SolverSummary
    metrics: dict[str, object]
    continuations: list[_ContinuationRecord]
    root_assessments_by_intelligence: dict[str, list[_RootAssessment]]
    complete_game_assessments: list[_CompleteAssessment]


def _off_path_beliefs(
    model: CubaModel,
    intelligence: Intelligence,
    convention: object,
) -> str | dict[str, dict[str, float]]:
    if convention in (None, "posterior", "prior"):
        # The continuation game's prior is already P(type | intelligence), so
        # the enumerator's `prior` convention means that public posterior.
        return "prior"
    if convention == "uniform":
        return "uniform"
    if convention == "resolve_low":
        posterior = model.posterior_after_intelligence(intelligence)
        weights = {
            label: value if "resolve_low" in label else 0.0
            for label, value in posterior.items()
        }
    elif convention == "resolve_high":
        posterior = model.posterior_after_intelligence(intelligence)
        weights = {
            label: value if "resolve_high" in label else 0.0
            for label, value in posterior.items()
        }
    else:
        raise ValueError(
            "off_path_belief_convention must be posterior, prior, uniform, "
            "resolve_low, or resolve_high"
        )
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("configured off-path convention has zero feasible mass")
    belief = {label: value / total for label, value in weights.items()}
    return {history.label: dict(belief) for history in COMMUNICATION_HISTORIES}


def solve_continuation(
    model: CubaModel,
    intelligence: Intelligence,
    initial_action: InitialAction,
    *,
    off_path_belief_convention: object = "posterior",
) -> SolverResult:
    prior, sender_payoffs, receiver_payoffs = model.continuation_payoff_arrays(
        intelligence, initial_action
    )
    game = SignalingGame(
        sender="Soviet Union",
        receiver="United States",
        type_labels=TYPE_ORDER,
        message_labels=tuple(history.label for history in COMMUNICATION_HISTORIES),
        receiver_action_labels=FINAL_RESPONSE_ORDER,
        prior=cast(Sequence[float], prior),
        sender_payoffs=sender_payoffs,
        receiver_payoffs=receiver_payoffs,
        assumptions=(
            "The continuation follows a public intelligence report and a fixed initial US action.",
            "Every numerical payoff and probability is illustrative or normalized.",
            "The two sequential Soviet communications are represented by their ordered history.",
            "No player or chance move intervenes between the first and second communication.",
        ),
    )
    return enumerate_pure_pbe(
        game,
        off_path_beliefs=_off_path_beliefs(
            model, intelligence, off_path_belief_convention
        ),
        tolerance=model.parameters.best_response_tolerance,
    )


def _equilibrium_outcomes(
    model: CubaModel,
    intelligence: Intelligence,
    initial_action: InitialAction,
    equilibrium: SignalingEquilibrium,
) -> _EquilibriumOutcomes:
    posterior = model.posterior_after_intelligence(intelligence)
    expected = {
        "escalation_probability": 0.0,
        "catastrophe_probability": 0.0,
        "expected_utility_us": 0.0,
        "expected_utility_ussr": 0.0,
    }
    initial_frequency = {action.value: float(action is initial_action) for action in InitialAction}
    signal_frequencies = {history.label: 0.0 for history in COMMUNICATION_HISTORIES}
    first_frequencies = {"conciliatory": 0.0, "hardline": 0.0}
    second_frequencies = {"reaffirm": 0.0, "revise": 0.0}
    response_frequencies = {action.value: 0.0 for action in FinalResponse}
    category_frequencies: dict[str, float] = {}
    posterior_calibration = 0.0

    paths: list[dict[str, object]] = []
    for soviet_type in SOVIET_TYPES:
        probability = posterior[soviet_type.label]
        message_label = equilibrium.sender_strategy[soviet_type.label]
        communication = next(
            history for history in COMMUNICATION_HISTORIES if history.label == message_label
        )
        response = FinalResponse(equilibrium.receiver_strategy[message_label])
        terminal = model.terminal(
            soviet_type,
            intelligence,
            initial_action,
            communication,
            response,
        )
        expected["escalation_probability"] += probability * terminal.escalation_probability
        expected["catastrophe_probability"] += probability * terminal.catastrophe_probability
        expected["expected_utility_us"] += probability * terminal.expected_utility_us
        expected["expected_utility_ussr"] += probability * terminal.expected_utility_ussr
        signal_frequencies[message_label] += probability
        first_frequencies[communication.first.value] += probability
        second_frequencies[communication.second.value] += probability
        response_frequencies[response.value] += probability
        category_frequencies[terminal.peaceful_category] = (
            category_frequencies.get(terminal.peaceful_category, 0.0) + probability
        )
        posterior_calibration += probability * equilibrium.posteriors[message_label][soviet_type.label]
        paths.append(
            {
                "probability": probability,
                "type": soviet_type.label,
                "intelligence": intelligence.value,
                "initial_action": initial_action.value,
                "ordered_communication": message_label,
                "first_communication": communication.first.value,
                "second_communication": communication.second.value,
                "response": response.value,
                "terminal": terminal.to_dict(),
            }
        )
    return {
        "escalation_probability": expected["escalation_probability"],
        "catastrophe_probability": expected["catastrophe_probability"],
        "expected_utility_us": expected["expected_utility_us"],
        "expected_utility_ussr": expected["expected_utility_ussr"],
        "initial_action_frequencies": initial_frequency,
        "ordered_signal_frequencies": signal_frequencies,
        "first_signal_frequencies": first_frequencies,
        "second_signal_frequencies": second_frequencies,
        "response_frequencies": response_frequencies,
        "outcome_category_frequencies": dict(sorted(category_frequencies.items())),
        "posterior_truth_probability": posterior_calibration,
        "paths": paths,
    }


def _sequential_communication_policy(
    model: CubaModel,
    intelligence: Intelligence,
    initial_action: InitialAction,
    equilibrium: SignalingEquilibrium,
) -> dict[str, dict[str, object]]:
    """Extend a reduced ordered-history strategy to every second-message node.

    The terminal-history signaling reduction checks deviations to every ordered
    pair. This function additionally records sequentially rational second
    messages after both possible first communications, including an off-path
    first communication. The correspondence retains ties.
    """

    policy: dict[str, dict[str, object]] = {}
    tolerance = model.parameters.best_response_tolerance
    for soviet_type in SOVIET_TYPES:
        chosen_history = next(
            history
            for history in COMMUNICATION_HISTORIES
            if history.label == equilibrium.sender_strategy[soviet_type.label]
        )
        by_first: dict[str, object] = {}
        for first in {history.first for history in COMMUNICATION_HISTORIES}:
            values: dict[str, float] = {}
            for history in COMMUNICATION_HISTORIES:
                if history.first is not first:
                    continue
                response = FinalResponse(equilibrium.receiver_strategy[history.label])
                values[history.second.value] = model.terminal(
                    soviet_type,
                    intelligence,
                    initial_action,
                    history,
                    response,
                ).expected_utility_ussr
            best = max(values.values())
            optimal = sorted(
                second for second, value in values.items() if value >= best - tolerance
            )
            by_first[first.value] = {
                "optimal_second_communications": optimal,
                "continuation_utilities": dict(sorted(values.items())),
                "on_path_first_communication": first is chosen_history.first,
            }
            if first is chosen_history.first and chosen_history.second.value not in optimal:
                raise ArithmeticError(
                    "reduced signaling assessment failed second-stage sequential rationality"
                )
        policy[soviet_type.label] = by_first
    return policy


def solve_incomplete_information(
    model: CubaModel,
    *,
    off_path_belief_convention: object = "posterior",
) -> _IncompleteInformationSolution:
    """Enumerate continuation pure PBE and the initial US best-response correspondence.

    Each continuation equilibrium is retained. For each public intelligence
    report, the function enumerates every selection of one continuation PBE at
    every possible initial-action branch, then checks the initial US deviation.
    Finally it takes the Cartesian product across the two public-report
    subgames. Thus multiplicity is retained without silently choosing the
    continuation most favorable to either player.
    """

    continuation_records: list[_ContinuationRecord] = []
    root_assessments_by_intelligence: dict[str, list[_RootAssessment]] = {}
    tolerance = model.parameters.best_response_tolerance
    total_runtime = 0.0
    for intelligence in Intelligence:
        action_records: dict[str, list[_SerializedEquilibrium]] = {}
        for initial_action in InitialAction:
            result = solve_continuation(
                model,
                intelligence,
                initial_action,
                off_path_belief_convention=off_path_belief_convention,
            )
            total_runtime += result.runtime_seconds
            if not result.found:
                continuation_records.append(
                    {
                        "intelligence": intelligence.value,
                        "initial_action": initial_action.value,
                        "status": result.status,
                        "equilibria": [],
                        "solver": result.to_dict(),
                    }
                )
                continue
            serialized_equilibria: list[_SerializedEquilibrium] = []
            for equilibrium_index, raw_equilibrium in enumerate(result.equilibria):
                equilibrium = cast(SignalingEquilibrium, raw_equilibrium)
                outcomes = _equilibrium_outcomes(
                    model, intelligence, initial_action, equilibrium
                )
                record: _SerializedEquilibrium = {
                    "intelligence": intelligence.value,
                    "initial_action": initial_action.value,
                    "equilibrium_index": equilibrium_index,
                    "classification": equilibrium.classification,
                    "sender_strategy": dict(equilibrium.sender_strategy),
                    "receiver_strategy": dict(equilibrium.receiver_strategy),
                    "posteriors": {
                        key: dict(value) for key, value in equilibrium.posteriors.items()
                    },
                    "reach_probabilities": dict(equilibrium.reach_probabilities),
                    "expected_utilities": dict(equilibrium.expected_utilities),
                    "sender_deviation_gains": dict(equilibrium.sender_deviation_gains),
                    "receiver_deviation_gains": dict(equilibrium.receiver_deviation_gains),
                    "max_best_response_gain": equilibrium.max_best_response_gain,
                    "agent_form_nashconv": sum(
                        equilibrium.sender_deviation_gains.values()
                    )
                    + sum(equilibrium.receiver_deviation_gains.values()),
                    "off_path_messages": list(equilibrium.off_path_messages),
                    "depends_on_off_path_beliefs": equilibrium.depends_on_off_path_beliefs,
                    "outcomes": outcomes,
                    "sequential_communication_policy": _sequential_communication_policy(
                        model, intelligence, initial_action, equilibrium
                    ),
                }
                serialized_equilibria.append(record)
            action_records[initial_action.value] = serialized_equilibria
            continuation_records.append(
                {
                    "intelligence": intelligence.value,
                    "initial_action": initial_action.value,
                    "status": result.status,
                    "equilibria": serialized_equilibria,
                    "solver_metadata": result.to_dict()["metadata"],
                    "solver_runtime_seconds": result.runtime_seconds,
                }
            )
        root_assessments: list[_RootAssessment] = []
        if all(action_records.get(action.value) for action in InitialAction):
            ordered_actions = tuple(InitialAction)
            for continuation_selection in product(
                *(action_records[action.value] for action in ordered_actions)
            ):
                values = {
                    action.value: float(record["outcomes"]["expected_utility_us"])
                    for action, record in zip(
                        ordered_actions, continuation_selection, strict=True
                    )
                }
                best_us = max(values.values())
                best_actions = tuple(
                    action
                    for action in ordered_actions
                    if values[action.value] >= best_us - tolerance
                )
                selection_ids = {
                    action.value: int(record["equilibrium_index"])
                    for action, record in zip(
                        ordered_actions, continuation_selection, strict=True
                    )
                }
                for selected_action in best_actions:
                    selected = continuation_selection[ordered_actions.index(selected_action)]
                    root_assessments.append(
                        {
                            "assessment_index": len(root_assessments),
                            "intelligence": intelligence.value,
                            "continuation_selection_by_initial_action": selection_ids,
                            "initial_action_values_us": values,
                            "selected_initial_action": selected_action.value,
                            "selected_continuation_equilibrium_index": int(
                                selected["equilibrium_index"]
                            ),
                            "selected_classification": selected["classification"],
                            "selected_outcomes": selected["outcomes"],
                            "initial_action_tie": len(best_actions) > 1,
                        }
                    )
        root_assessments_by_intelligence[intelligence.value] = root_assessments

    complete_assessments: list[_CompleteAssessment] = []
    intelligence_order = tuple(Intelligence)
    assessment_lists = tuple(
        root_assessments_by_intelligence[item.value] for item in intelligence_order
    )
    if all(assessment_lists):
        for combination in product(*assessment_lists):
            weights = {
                intelligence.value: model.intelligence_probability(intelligence)
                for intelligence in intelligence_order
            }
            expected = {
                field: sum(
                    weights[intelligence.value]
                    * float(assessment["selected_outcomes"][field])
                    for intelligence, assessment in zip(
                        intelligence_order, combination, strict=True
                    )
                )
                for field in (
                    "escalation_probability",
                    "catastrophe_probability",
                    "expected_utility_us",
                    "expected_utility_ussr",
                    "posterior_truth_probability",
                )
            }
            complete_assessment: _CompleteAssessment = {
                "assessment_index": len(complete_assessments),
                "root_assessment_by_intelligence": {
                    intelligence.value: assessment["assessment_index"]
                    for intelligence, assessment in zip(
                        intelligence_order, combination, strict=True
                    )
                },
                "selected_initial_action_by_intelligence": {
                    intelligence.value: assessment["selected_initial_action"]
                    for intelligence, assessment in zip(
                        intelligence_order, combination, strict=True
                    )
                },
                "regime_by_intelligence": {
                    intelligence.value: assessment["selected_classification"]
                    for intelligence, assessment in zip(
                        intelligence_order, combination, strict=True
                    )
                },
                "escalation_probability": expected["escalation_probability"],
                "catastrophe_probability": expected["catastrophe_probability"],
                "expected_utility_us": expected["expected_utility_us"],
                "expected_utility_ussr": expected["expected_utility_ussr"],
                "posterior_truth_probability": expected[
                    "posterior_truth_probability"
                ],
            }
            complete_assessments.append(complete_assessment)

    classifications = Counter(
        str(record["classification"])
        for continuation in continuation_records
        for record in continuation["equilibria"]
    )
    total_equilibria = sum(len(record["equilibria"]) for record in continuation_records)
    no_pure = sum(record["status"] == "NO_PURE_PBE_FOUND" for record in continuation_records)
    solver_summary: _SolverSummary = {
            "name": "cuba_complete_pure_pbe_enumerator",
            "version": "1.0",
            "equilibrium_concept": (
                "restricted pure-strategy PBE continuations with initial-action "
                "sequential-rationality correspondence"
            ),
            "found": bool(complete_assessments),
            "multiple_solutions": len(complete_assessments) > 1,
            "convergence_or_exactness_status": "exact_finite_enumeration_with_tolerance",
            "best_response_gap": max(
                (
                    float(equilibrium["max_best_response_gain"])
                    for continuation in continuation_records
                    for equilibrium in continuation["equilibria"]
                ),
                default=None,
            ),
            "off_path_belief_convention": str(off_path_belief_convention),
            "runtime_seconds": total_runtime,
            "warnings": [
                "The search is restricted to pure continuation PBE.",
                "Tied sequential second-message choices are retained as correspondences rather than expanded into duplicate assessments.",
            ],
            "assumptions": [
                "All numerical inputs are illustrative assumptions or utility normalizations.",
                "The public intelligence report is observed before the initial US action.",
                "No equilibrium-selection rule is imposed across continuation multiplicity.",
            ],
            "seed": None,
    }
    metrics: dict[str, object] = {
            "continuation_equilibrium_count": total_equilibria,
            "root_assessment_count": sum(
                len(items) for items in root_assessments_by_intelligence.values()
            ),
            "complete_assessment_count": len(complete_assessments),
            "multiple_equilibria": len(complete_assessments) > 1,
            "no_pure_pbe_continuation_count": no_pure,
            "pooling_equilibrium_count": classifications["pooling"],
            "separating_equilibrium_count": classifications["separating"],
            "partially_pooling_equilibrium_count": classifications["partially_pooling"],
            "agent_form_nashconv_max": max(
                (
                    float(equilibrium["agent_form_nashconv"])
                    for continuation in continuation_records
                    for equilibrium in continuation["equilibria"]
                ),
                default=None,
            ),
    }
    return {
        "solver": solver_summary,
        "metrics": metrics,
        "continuations": continuation_records,
        "root_assessments_by_intelligence": root_assessments_by_intelligence,
        "complete_game_assessments": complete_assessments,
    }


__all__ = ["solve_continuation", "solve_incomplete_information"]
