"""Counterfactual adapter for the supported Cuba event model.

The adapter deliberately works with the event model's restricted pure-PBE
continuation solver.  It does not claim general extensive-form PBE support.
Policies are keyed by information sets whose identifiers encode only variables
observed at that decision.  In particular, no United States information-set
identifier contains the hidden Soviet type.
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, cast

from cold_war_sim import __version__
from cold_war_sim.core.types import canonical_json
from cold_war_sim.counterfactuals.feasibility import (
    InformationSetSpec,
    InformationStructure,
    InformationVariable,
    PurePolicy,
    audit_policy,
)
from cold_war_sim.counterfactuals.outcomes import (
    OutcomeDistribution,
    OutcomeFeatures,
    WeightedOutcome,
)
from cold_war_sim.counterfactuals.policy_space import PolicySpace
from cold_war_sim.events.cuba.solver import solve_incomplete_information
from cold_war_sim.events.cuba.model import (
    COMMUNICATION_HISTORIES,
    SOVIET_TYPES,
    CommunicationHistory,
    CubaModel,
    FinalResponse,
    FirstCommunication,
    InitialAction,
    Intelligence,
    SecondCommunication,
)
from cold_war_sim.events.cuba.parameters import CubaParameters

SCHEMA_VERSION = "1.0"
US = "united_states"
USSR = "soviet_union"
SUPPORTED_OPERATIONS = ("validate", "evaluate", "search", "paths", "frontier")


def _initial_set(intelligence: str) -> str:
    return f"initial_{intelligence}"


def _final_set(intelligence: str, initial: str, communication: str) -> str:
    return f"final_{intelligence}_{initial}_{communication}"


def _soviet_first_set(type_label: str, intelligence: str, initial: str) -> str:
    return f"soviet_first_{type_label}_{intelligence}_{initial}"


def _soviet_second_set(type_label: str, intelligence: str, initial: str, first: str) -> str:
    return f"soviet_second_{type_label}_{intelligence}_{initial}_{first}"


def information_structure() -> InformationStructure:
    """Return the event's explicit policy domains and observation chronology."""

    variables = {
        "soviet_type": InformationVariable("soviet_type", 0, (USSR,)),
        "intelligence": InformationVariable("intelligence", 1, (US, USSR)),
        "initial_action": InformationVariable("initial_action", 2, (US, USSR)),
        "first_communication": InformationVariable("first_communication", 3, (US, USSR)),
        "second_communication": InformationVariable("second_communication", 4, (US, USSR)),
        "modeler_parameter_truth": InformationVariable(
            "modeler_parameter_truth", 0, (), modeler_only=True
        ),
        "simulation_seed": InformationVariable("simulation_seed", 0, (), modeler_only=True),
    }
    sets: dict[str, InformationSetSpec] = {}
    for intelligence in Intelligence:
        intelligence_id = intelligence.value
        initial_id = _initial_set(intelligence_id)
        sets[initial_id] = InformationSetSpec(
            initial_id,
            US,
            2,
            tuple(f"{initial_id}_{soviet_type.label}" for soviet_type in SOVIET_TYPES),
            tuple(action.value for action in InitialAction),
            ("intelligence",),
            observation_signature={"intelligence": intelligence_id},
        )
        for initial in InitialAction:
            initial_action = initial.value
            for communication in COMMUNICATION_HISTORIES:
                final_id = _final_set(intelligence_id, initial_action, communication.label)
                sets[final_id] = InformationSetSpec(
                    final_id,
                    US,
                    5,
                    tuple(f"{final_id}_{soviet_type.label}" for soviet_type in SOVIET_TYPES),
                    tuple(response.value for response in FinalResponse),
                    (
                        "intelligence",
                        "initial_action",
                        "first_communication",
                        "second_communication",
                    ),
                    observation_signature={
                        "intelligence": intelligence_id,
                        "initial_action": initial_action,
                        "first_communication": communication.first.value,
                        "second_communication": communication.second.value,
                    },
                )
            for soviet_type in SOVIET_TYPES:
                first_id = _soviet_first_set(soviet_type.label, intelligence_id, initial_action)
                sets[first_id] = InformationSetSpec(
                    first_id,
                    USSR,
                    3,
                    (first_id,),
                    tuple(item.value for item in FirstCommunication),
                    ("soviet_type", "intelligence", "initial_action"),
                    observation_signature={
                        "soviet_type": soviet_type.label,
                        "intelligence": intelligence_id,
                        "initial_action": initial_action,
                    },
                )
                for first in FirstCommunication:
                    second_id = _soviet_second_set(
                        soviet_type.label,
                        intelligence_id,
                        initial_action,
                        first.value,
                    )
                    sets[second_id] = InformationSetSpec(
                        second_id,
                        USSR,
                        4,
                        (second_id,),
                        tuple(item.value for item in SecondCommunication),
                        (
                            "soviet_type",
                            "intelligence",
                            "initial_action",
                            "first_communication",
                        ),
                        observation_signature={
                            "soviet_type": soviet_type.label,
                            "intelligence": intelligence_id,
                            "initial_action": initial_action,
                            "first_communication": first.value,
                        },
                    )
    return InformationStructure(sets, variables)


def terminal_outcome_features(
    terminal: Any,
    *,
    branch: str = "no_escalation",
) -> OutcomeFeatures:
    """Convert a legacy terminal record into utility-independent features.

    The legacy event integrates accidental catastrophe into expected utility.
    The compatibility adapter therefore expands each record into catastrophe,
    controlled-escalation, and non-escalation branches.  It does not reinterpret
    the illustrative utility weights as outcome facts.
    """

    if branch not in {"catastrophe", "controlled_escalation", "no_escalation"}:
        raise ValueError("unknown Cuba outcome branch")
    catastrophic = branch == "catastrophe"
    escalated = branch in {"catastrophe", "controlled_escalation"}
    negotiated = terminal.peaceful_category == "negotiated" and not escalated
    peaceful = terminal.peaceful_category != "escalatory" and not escalated
    return OutcomeFeatures(
        common={
            "peaceful_settlement": peaceful,
            "negotiated_agreement": negotiated,
            "military_escalation": escalated,
            "catastrophic_escalation": catastrophic,
            "escalation_probability": terminal.escalation_probability,
            "catastrophe_probability": terminal.catastrophe_probability,
            "settlement_probability": float(peaceful),
            "missile_removal": terminal.missile_removal,
            "war_intensity": terminal.war_intensity,
        },
        by_player={
            US: {
                "military_cost": terminal.war_intensity,
            },
            USSR: {
                "political_cost": terminal.signal_cost,
                "military_cost": terminal.war_intensity,
            },
        },
    )


def _parameters(values: Mapping[str, Any]) -> CubaParameters:
    return CubaParameters.from_dict(dict(values))


def _spec(document: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = document.get("counterfactual")
    if isinstance(nested, Mapping):
        return cast(Mapping[str, Any], nested)
    return document


def _baseline(document: Mapping[str, Any]) -> Mapping[str, Any]:
    baseline = _spec(document).get("baseline")
    if not isinstance(baseline, Mapping) or baseline.get("event") != "cuba":
        raise ValueError("Cuba adapter requires baseline.event='cuba'")
    return cast(Mapping[str, Any], baseline)


def _plain_policy(raw: object) -> dict[str, str]:
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("policy must be a nonempty mapping")
    result: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("Cuba adapter currently supports deterministic pure policies")
        result[key] = value
    return result


def _normalize_player(value: object) -> str:
    aliases = {
        US: US,
        "United States": US,
        "us": US,
        USSR: USSR,
        "Soviet Union": USSR,
        "ussr": USSR,
    }
    try:
        return aliases[str(value)]
    except KeyError as error:
        raise ValueError(f"unknown Cuba player {value!r}") from error


def _continuation_lookup(
    solution: Mapping[str, Any],
) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    lookup: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    continuations = solution["continuations"]
    if not isinstance(continuations, Sequence):
        raise RuntimeError("Cuba solver returned malformed continuations")
    for continuation in continuations:
        if not isinstance(continuation, Mapping):
            continue
        intelligence = str(continuation["intelligence"])
        initial = str(continuation["initial_action"])
        equilibria = continuation["equilibria"]
        if not isinstance(equilibria, Sequence):
            continue
        for record in equilibria:
            if isinstance(record, Mapping):
                lookup[(intelligence, initial, int(record["equilibrium_index"]))] = record
    return lookup


def _root_options(
    solution: Mapping[str, Any],
    intelligence: str,
    allowed_initial: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    raw_roots = solution["root_assessments_by_intelligence"]
    if not isinstance(raw_roots, Mapping):
        raise RuntimeError("Cuba solver returned malformed root assessments")
    entries = raw_roots[intelligence]
    if not isinstance(entries, Sequence):
        raise RuntimeError("Cuba solver returned malformed root assessment list")
    lookup = _continuation_lookup(solution)
    selections: dict[tuple[tuple[str, int], ...], dict[str, int]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        raw_selection = entry["continuation_selection_by_initial_action"]
        if not isinstance(raw_selection, Mapping):
            continue
        selection = {str(key): int(value) for key, value in raw_selection.items()}
        selections[tuple(sorted(selection.items()))] = selection

    result: list[dict[str, Any]] = []
    tolerance = 1e-9
    for selection_key in sorted(selections):
        selection = selections[selection_key]
        values = {
            action: float(
                cast(
                    Mapping[str, Any],
                    lookup[(intelligence, action, selection[action])]["outcomes"],
                )["expected_utility_us"]
            )
            for action in allowed_initial
        }
        best = max(values.values())
        for selected in sorted(
            action for action, value in values.items() if value >= best - tolerance
        ):
            result.append(
                {
                    "intelligence": intelligence,
                    "continuation_selection_by_initial_action": dict(selection),
                    "initial_action_values_us": values,
                    "selected_initial_action": selected,
                    "selected_continuation_equilibrium_index": selection[selected],
                }
            )
    return tuple(result)


def _policy_from_roots(
    solution: Mapping[str, Any], roots: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, str], dict[str, Any]]:
    lookup = _continuation_lookup(solution)
    policy: dict[str, str] = {}
    certificate_parts: list[Mapping[str, Any]] = []
    for intelligence in sorted(roots):
        root = roots[intelligence]
        selected_initial = str(root["selected_initial_action"])
        policy[_initial_set(intelligence)] = selected_initial
        selection_raw = root["continuation_selection_by_initial_action"]
        if not isinstance(selection_raw, Mapping):
            raise RuntimeError("Cuba root selection is malformed")
        selection = {str(key): int(value) for key, value in selection_raw.items()}
        for initial in sorted(selection):
            record = lookup[(intelligence, initial, selection[initial])]
            certificate_parts.append(record)
            receiver_strategy = record["receiver_strategy"]
            sender_strategy = record["sender_strategy"]
            sequential = record["sequential_communication_policy"]
            if not isinstance(receiver_strategy, Mapping) or not isinstance(
                sender_strategy, Mapping
            ):
                raise RuntimeError("Cuba equilibrium strategy is malformed")
            for communication, response in receiver_strategy.items():
                policy[_final_set(intelligence, initial, str(communication))] = str(response)
            for soviet_type in SOVIET_TYPES:
                history_label = str(sender_strategy[soviet_type.label])
                history = next(
                    item for item in COMMUNICATION_HISTORIES if item.label == history_label
                )
                policy[_soviet_first_set(soviet_type.label, intelligence, initial)] = (
                    history.first.value
                )
                type_policy = cast(Mapping[str, Any], sequential)[soviet_type.label]
                if not isinstance(type_policy, Mapping):
                    raise RuntimeError("Cuba sequential policy is malformed")
                for first in FirstCommunication:
                    if first is history.first:
                        second = history.second.value
                    else:
                        first_record = type_policy[first.value]
                        if not isinstance(first_record, Mapping):
                            raise RuntimeError("Cuba second-stage correspondence is malformed")
                        optimal = first_record["optimal_second_communications"]
                        if not isinstance(optimal, Sequence) or not optimal:
                            raise RuntimeError("Cuba second-stage correspondence is empty")
                        # The existing event solver records tied off-path actions as a
                        # correspondence rather than multiplying assessments.  The
                        # lexicographic representative is only used to evaluate paths;
                        # the certificate retains that limitation explicitly.
                        second = sorted(str(item) for item in optimal)[0]
                    policy[
                        _soviet_second_set(soviet_type.label, intelligence, initial, first.value)
                    ] = second
    return policy, {"parts": certificate_parts}


def _joint_probability(model: CubaModel, soviet_type: Any, intelligence: Intelligence) -> float:
    return model.type_prior()[soviet_type.label] * model.intelligence_likelihood(
        soviet_type, intelligence
    )


def _evaluate_policy(
    model: CubaModel,
    policy: Mapping[str, str],
    *,
    equilibrium_supported: bool,
) -> dict[str, Any]:
    outcomes: list[WeightedOutcome] = []
    paths: list[dict[str, Any]] = []
    utilities = {US: 0.0, USSR: 0.0}
    for intelligence in Intelligence:
        initial = InitialAction(policy[_initial_set(intelligence.value)])
        for soviet_type in SOVIET_TYPES:
            probability = _joint_probability(model, soviet_type, intelligence)
            first = FirstCommunication(
                policy[_soviet_first_set(soviet_type.label, intelligence.value, initial.value)]
            )
            second = SecondCommunication(
                policy[
                    _soviet_second_set(
                        soviet_type.label,
                        intelligence.value,
                        initial.value,
                        first.value,
                    )
                ]
            )
            communication = CommunicationHistory(first, second)
            response = FinalResponse(
                policy[_final_set(intelligence.value, initial.value, communication.label)]
            )
            terminal = model.terminal(soviet_type, intelligence, initial, communication, response)
            utilities[US] += probability * terminal.expected_utility_us
            utilities[USSR] += probability * terminal.expected_utility_ussr
            branch_probabilities = {
                "catastrophe": terminal.catastrophe_probability,
                "controlled_escalation": (
                    terminal.escalation_probability - terminal.catastrophe_probability
                ),
                "no_escalation": 1.0 - terminal.escalation_probability,
            }
            path_id = (
                f"{soviet_type.label}|{intelligence.value}|{initial.value}|"
                f"{communication.label}|{response.value}"
            )
            for branch, conditional_probability in branch_probabilities.items():
                if conditional_probability <= 0.0:
                    continue
                features = terminal_outcome_features(terminal, branch=branch)
                outcome_id = f"{path_id}|{branch}"
                outcomes.append(
                    WeightedOutcome(
                        outcome_id,
                        probability * conditional_probability,
                        features,
                    )
                )
                catastrophe_utility_us = (
                    -model.parameters.catastrophe_loss * model.parameters.risk_sensitivity_us
                )
                catastrophe_utility_ussr = (
                    -model.parameters.catastrophe_loss * model.parameters.risk_sensitivity_ussr
                )
                branch_utilities = {
                    US: (
                        catastrophe_utility_us
                        if branch == "catastrophe"
                        else terminal.utility_us_no_catastrophe
                    ),
                    USSR: (
                        catastrophe_utility_ussr
                        if branch == "catastrophe"
                        else terminal.utility_ussr_no_catastrophe
                    ),
                }
                paths.append(
                    {
                        "terminal_id": outcome_id,
                        "events": [
                            {"actor": "nature", "action": soviet_type.label},
                            {"actor": "nature", "action": intelligence.value},
                            {"actor": US, "action": initial.value},
                            {"actor": USSR, "action": first.value},
                            {"actor": USSR, "action": second.value},
                            {"actor": US, "action": response.value},
                            {"actor": "nature", "action": branch},
                        ],
                        "reach_probability": probability * conditional_probability,
                        "terminal_outcome": features.to_dict(),
                        "utilities": branch_utilities,
                        "model_reachable": True,
                        "strategy_feasible": True,
                        "equilibrium_supported": equilibrium_supported,
                    }
                )
    distribution = OutcomeDistribution(tuple(outcomes))
    return {
        "distribution": distribution,
        "utilities": utilities,
        "paths": tuple(sorted(paths, key=lambda item: str(item["terminal_id"]))),
    }


def _assessment_set(
    model: CubaModel,
    *,
    allowed_initial: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[dict[str, Any], ...]:
    raw = cast(Mapping[str, Any], solve_incomplete_information(model))
    default_actions = tuple(item.value for item in InitialAction)
    allowed = {
        intelligence.value: (
            allowed_initial.get(intelligence.value, default_actions)
            if allowed_initial is not None
            else default_actions
        )
        for intelligence in Intelligence
    }
    roots = {
        intelligence.value: _root_options(raw, intelligence.value, allowed[intelligence.value])
        for intelligence in Intelligence
    }
    if any(not values for values in roots.values()):
        return ()
    assessments: list[dict[str, Any]] = []
    for combination in itertools.product(
        *(roots[intelligence.value] for intelligence in Intelligence)
    ):
        selected_roots = {
            intelligence.value: root
            for intelligence, root in zip(Intelligence, combination, strict=True)
        }
        policy, raw_certificate = _policy_from_roots(raw, selected_roots)
        evaluated = _evaluate_policy(model, policy, equilibrium_supported=True)
        parts = cast(Sequence[Mapping[str, Any]], raw_certificate["parts"])
        gaps = [float(part["max_best_response_gain"]) for part in parts]
        off_path_dependent = any(bool(part["depends_on_off_path_beliefs"]) for part in parts)
        assessment_id = f"cuba_eq_{len(assessments):04d}"
        beliefs: dict[str, Any] = {}
        continuation_reach: dict[str, float] = {}
        for part in parts:
            prefix = (
                f"{part['intelligence']}__{part['initial_action']}__"
                f"equilibrium_{int(part['equilibrium_index']):04d}"
            )
            raw_posteriors = part.get("posteriors", {})
            if isinstance(raw_posteriors, Mapping):
                for message, posterior in raw_posteriors.items():
                    beliefs[f"{prefix}__{message}"] = dict(cast(Mapping[str, float], posterior))
            raw_reach = part.get("reach_probabilities", {})
            if isinstance(raw_reach, Mapping):
                for message, probability in raw_reach.items():
                    continuation_reach[f"{prefix}__{message}"] = float(probability)
        certificate = {
            "equilibrium_concept": "PURE_PBE",
            "candidate_class": "pure strategies in the implemented Cuba reduction",
            "strategy_profile": dict(sorted(policy.items())),
            "beliefs": beliefs,
            "reach_probabilities": {
                **{
                    f"intelligence__{intelligence.value}": model.intelligence_probability(
                        intelligence
                    )
                    for intelligence in Intelligence
                },
                **continuation_reach,
            },
            "deviation_gains": {f"continuation_{index}": gap for index, gap in enumerate(gaps)},
            "best_response_gap": max(gaps, default=0.0),
            "off_path_convention": "configured by the existing Cuba solver",
            "exact": True,
            "tolerance": model.parameters.best_response_tolerance,
            "warnings": [
                "Pure continuation PBE only; mixed PBE were not searched.",
                "Tied off-path second messages remain a correspondence in the source solver.",
            ],
            "depends_on_off_path_beliefs": off_path_dependent,
        }
        assessments.append(
            {
                "id": assessment_id,
                "strategy_profile": dict(sorted(policy.items())),
                "expected_utilities": evaluated["utilities"],
                "outcome_distribution": evaluated["distribution"],
                "paths": evaluated["paths"],
                "certificate": certificate,
            }
        )
    # Equivalent root assessments may be generated by continuations that differ
    # only off path.  Retain distinct certified strategies but remove exact duplicates.
    unique: dict[str, dict[str, Any]] = {}
    for assessment in assessments:
        key = canonical_json(assessment["strategy_profile"])
        unique[key] = assessment
    ordered = tuple(unique[key] for key in sorted(unique))
    for index, assessment in enumerate(ordered):
        assessment["id"] = f"cuba_eq_{index:04d}"
    return ordered


def _selected(
    assessments: tuple[dict[str, Any], ...], spec: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    handling = spec.get("equilibrium_handling", {})
    if not isinstance(handling, Mapping):
        raise ValueError("equilibrium_handling must be a mapping")
    rule = str(handling.get("selection_rule", "RETAIN_ALL"))
    if rule == "RETAIN_ALL" or not assessments:
        return assessments
    if rule in {"PLAYER_OPTIMAL", "PLAYER_PESSIMAL"}:
        player = _normalize_player(handling.get("player"))
        values = [float(item["expected_utilities"][player]) for item in assessments]
        target = max(values) if rule == "PLAYER_OPTIMAL" else min(values)
        return tuple(
            item
            for item in assessments
            if math.isclose(float(item["expected_utilities"][player]), target, abs_tol=1e-9)
        )
    if rule == "SOCIAL_WELFARE":
        values = [sum(item["expected_utilities"].values()) for item in assessments]
        target = max(values)
        return tuple(
            item
            for item in assessments
            if math.isclose(sum(item["expected_utilities"].values()), target, abs_tol=1e-9)
        )
    if rule == "MINIMUM_CATASTROPHE":
        values = [
            item["outcome_distribution"].probability("catastrophic_escalation")
            for item in assessments
        ]
        target = min(values)
        return tuple(
            item
            for item, value in zip(assessments, values, strict=True)
            if math.isclose(value, target, abs_tol=1e-9)
        )
    raise ValueError(f"Cuba adapter does not support equilibrium selection rule {rule!r}")


def _apply_policy_replacement(
    model: CubaModel,
    baseline: tuple[dict[str, Any], ...],
    intervention: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    player = _normalize_player(intervention.get("player"))
    replacement = _plain_policy(intervention.get("policy"))
    report = audit_policy(
        PurePolicy.from_actions(player, replacement),
        information_structure(),
        require_complete=False,
    )
    if not report.feasible:
        return (
            {
                "infeasible": True,
                "feasibility": report,
            },
        )
    result: list[dict[str, Any]] = []
    for index, assessment in enumerate(baseline):
        policy = dict(assessment["strategy_profile"])
        policy.update(replacement)
        evaluated = _evaluate_policy(model, policy, equilibrium_supported=False)
        result.append(
            {
                "id": f"cuba_policy_{index:04d}",
                "strategy_profile": dict(sorted(policy.items())),
                "expected_utilities": evaluated["utilities"],
                "outcome_distribution": evaluated["distribution"],
                "paths": evaluated["paths"],
                "certificate": {
                    "equilibrium_concept": "SUPPLIED_PROFILE",
                    "candidate_class": "model-feasible pure policy profile",
                    "strategy_profile": dict(sorted(policy.items())),
                    "beliefs": {},
                    "reach_probabilities": {},
                    "deviation_gains": {},
                    "best_response_gap": 0.0,
                    "off_path_convention": "inherited from baseline assessment",
                    "exact": False,
                    "tolerance": model.parameters.best_response_tolerance,
                    "warnings": [
                        "Frozen opponents: this profile is not an equilibrium certificate."
                    ],
                },
                "feasibility": report,
            }
        )
    return tuple(result)


def _changed_model(base: CubaParameters, intervention: Mapping[str, Any]) -> CubaModel:
    kind = intervention.get("type")
    if kind == "PARAMETER_INTERVENTION":
        changes = intervention.get("changes")
        if not isinstance(changes, Mapping):
            raise ValueError("parameter intervention changes must be a mapping")
        return CubaModel(_parameters({**base.to_dict(), **dict(changes)}))
    if kind == "INFORMATION_INTERVENTION":
        if intervention.get("channel") != "intelligence":
            raise ValueError("Cuba supports only the 'intelligence' information channel")
        changes = intervention.get("changes")
        if not isinstance(changes, Mapping) or set(changes) != {"accuracy"}:
            raise ValueError("Cuba intelligence intervention requires only changes.accuracy")
        accuracy = float(changes["accuracy"])
        if accuracy <= 0.5:
            raise ValueError(
                "the implemented Cuba model requires intelligence accuracy above 0.5; "
                "information removal is unsupported"
            )
        return CubaModel(replace(base, intelligence_accuracy=accuracy))
    return CubaModel(base)


def _restriction(
    intervention: Mapping[str, Any],
) -> dict[str, tuple[str, ...]] | None:
    if intervention.get("type") != "ACTION_RESTRICTION":
        return None
    if _normalize_player(intervention.get("player")) != US:
        raise ValueError("Cuba action restriction currently supports US initial actions only")
    information_set = str(intervention.get("information_set"))
    if information_set not in {_initial_set(item.value) for item in Intelligence}:
        raise ValueError("Cuba action restriction currently supports initial_* information sets")
    raw_actions = intervention.get("actions")
    if not isinstance(raw_actions, Sequence) or isinstance(raw_actions, (str, bytes)):
        raise ValueError("restriction actions must be a sequence")
    removed = {str(item) for item in raw_actions}
    unknown = removed - {item.value for item in InitialAction}
    if unknown:
        raise ValueError(f"unknown Cuba initial actions: {sorted(unknown)}")
    allowed = tuple(item.value for item in InitialAction if item.value not in removed)
    if not allowed:
        raise ValueError("action restriction cannot remove every initial action")
    intelligence = information_set.removeprefix("initial_")
    return {intelligence: allowed}


def _commitment_restriction(
    intervention: Mapping[str, Any],
) -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    """Validate a binding public partial policy for U.S. initial decisions."""

    if intervention.get("binding") is not True or intervention.get("observable") is not True:
        raise ValueError("Cuba policy commitment must be observable and binding")
    if intervention.get("commitment_scope") != "PARTIAL_POLICY":
        raise ValueError("Cuba initial-policy commitment requires commitment_scope=PARTIAL_POLICY")
    player = _normalize_player(intervention.get("player"))
    if player != US:
        raise ValueError("Cuba commitment currently supports the U.S. initial policy only")
    policy = _plain_policy(intervention.get("policy"))
    supported_sets = {_initial_set(item.value) for item in Intelligence}
    if not set(policy) <= supported_sets:
        raise ValueError(
            "Cuba commitment can bind only initial_reassuring and initial_alarming"
        )
    report = audit_policy(
        PurePolicy.from_actions(player, policy),
        information_structure(),
        committed_actions=policy,
        require_complete=False,
    )
    if not report.feasible:
        raise ValueError(f"infeasible Cuba commitment: {report.to_dict()['violations']}")
    allowed: dict[str, tuple[str, ...]] = {
        identifier.removeprefix("initial_"): (action,)
        for identifier, action in sorted(policy.items())
    }
    return allowed, report.to_dict()


def _distribution_dict(assessment: Mapping[str, Any]) -> dict[str, Any]:
    distribution = assessment["outcome_distribution"]
    if not isinstance(distribution, OutcomeDistribution):
        raise RuntimeError("Cuba assessment contains malformed outcomes")
    return distribution.to_dict()


def _solution_set_dict(assessments: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "selection_treatment": "explicit equilibrium_handling from the request",
        "equilibria": [
            {
                "id": item["id"],
                "distribution": _distribution_dict(item),
            }
            for item in assessments
        ],
    }


def _utility_change_ranges(
    baseline: Sequence[Mapping[str, Any]], transformed: Sequence[Mapping[str, Any]]
) -> dict[str, float]:
    if not baseline or not transformed:
        return {f"{player}.minimum": 0.0 for player in (US, USSR)} | {
            f"{player}.maximum": 0.0 for player in (US, USSR)
        }
    result: dict[str, float] = {}
    for player in (US, USSR):
        changes = [
            float(after["expected_utilities"][player]) - float(before["expected_utilities"][player])
            for before in baseline
            for after in transformed
        ]
        result[f"{player}.minimum"] = min(changes)
        result[f"{player}.maximum"] = max(changes)
    return result


def _feature_change_ranges(
    baseline: Sequence[Mapping[str, Any]], transformed: Sequence[Mapping[str, Any]]
) -> dict[str, float]:
    result: dict[str, float] = {}
    for feature in ("military_escalation", "catastrophic_escalation", "negotiated_agreement"):
        before = [item["outcome_distribution"].probability(feature) for item in baseline]
        after = [item["outcome_distribution"].probability(feature) for item in transformed]
        if before and after:
            changes = [new - old for old in before for new in after]
            result[f"{feature}.minimum"] = min(changes)
            result[f"{feature}.maximum"] = max(changes)
    return result


def _metadata() -> dict[str, Any]:
    return {
        "package_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "seed_policy": "deterministic exact enumeration; configured seed is not consumed",
        "solver_version": "cuba counterfactual adapter 1.0",
    }


def validate_counterfactual(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate event-specific fields without running a strategic solve."""

    spec = _spec(document)
    baseline = _baseline(document)
    raw_parameters = baseline.get("parameters", {})
    if not isinstance(raw_parameters, Mapping):
        raise ValueError("baseline.parameters must be a mapping")
    base_parameters = _parameters(cast(Mapping[str, Any], raw_parameters))
    intervention = spec.get("intervention")
    if not isinstance(intervention, Mapping):
        raise ValueError("intervention must be a mapping")
    kind = intervention.get("type")
    if spec.get("uncertainty_set") is not None:
        raise ValueError(
            "Cuba event routing does not yet support uncertainty-set evaluation; "
            "no robust preference claim was computed"
        )
    if spec.get("solution_concept") not in {"PURE_PBE", "SUPPLIED_PROFILE"}:
        raise ValueError("Cuba supports PURE_PBE or a frozen SUPPLIED_PROFILE only")
    supported = {
        "POLICY_REPLACEMENT",
        "ACTION_RESTRICTION",
        "PARAMETER_INTERVENTION",
        "INFORMATION_INTERVENTION",
        "COMMITMENT",
    }
    if kind not in supported:
        raise ValueError(
            f"Cuba adapter does not implement intervention type {kind!r}; "
            "action expansion and unregistered structural transforms are unsupported"
        )
    feasibility: dict[str, Any] | None = None
    if kind == "POLICY_REPLACEMENT":
        player = _normalize_player(intervention.get("player"))
        policy = _plain_policy(intervention.get("policy"))
        report = audit_policy(
            PurePolicy.from_actions(player, policy),
            information_structure(),
            require_complete=False,
        )
        feasibility = report.to_dict()
        if not report.feasible:
            raise ValueError(f"infeasible Cuba policy: {report.to_dict()['violations']}")
        if spec.get("response_model") != "FROZEN_OPPONENTS":
            raise ValueError(
                "Cuba policy replacement currently supports only FROZEN_OPPONENTS; "
                "it is a diagnostic rather than a strategic prediction"
            )
    elif kind == "ACTION_RESTRICTION":
        _restriction(cast(Mapping[str, Any], intervention))
        if spec.get("response_model") != "REEQUILIBRATE":
            raise ValueError("Cuba action restriction requires REEQUILIBRATE")
    elif kind == "COMMITMENT":
        _, feasibility = _commitment_restriction(cast(Mapping[str, Any], intervention))
        if spec.get("response_model") != "REEQUILIBRATE":
            raise ValueError("Cuba public initial-policy commitment requires REEQUILIBRATE")
    else:
        _changed_model(base_parameters, cast(Mapping[str, Any], intervention))
        if spec.get("response_model") != "REEQUILIBRATE":
            raise ValueError("Cuba parameter and information changes require REEQUILIBRATE")
    return {
        "schema_version": SCHEMA_VERSION,
        "document_type": "counterfactual_validation",
        "event": "cuba",
        "valid": True,
        "supported": True,
        "information_structure": information_structure().to_dict(),
        "feasibility_report": feasibility,
        "warnings": [
            "The Cuba equilibrium solver is restricted to pure continuation PBE.",
            "No model-feasible result is a claim of historical plausibility.",
        ],
    }


def evaluate_counterfactual(document: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate one supported Cuba intervention with multiplicity retained."""

    started = time.perf_counter()
    validation = validate_counterfactual(document)
    del validation
    spec = _spec(document)
    baseline_config = _baseline(document)
    raw_parameters = baseline_config.get("parameters", {})
    assert isinstance(raw_parameters, Mapping)
    base_parameters = _parameters(cast(Mapping[str, Any], raw_parameters))
    intervention = cast(Mapping[str, Any], spec["intervention"])
    baseline_model = CubaModel(base_parameters)
    baseline_all = _assessment_set(baseline_model)
    baseline_assessments = _selected(baseline_all, spec)
    kind = intervention["type"]
    feasibility_report: dict[str, Any]
    warnings = [
        "All Cuba probabilities and utility parameters are illustrative or normalized.",
        "Expected-utility changes are pairwise ranges when multiplicity remains.",
    ]
    if kind == "POLICY_REPLACEMENT":
        transformed = _apply_policy_replacement(baseline_model, baseline_assessments, intervention)
        if transformed and transformed[0].get("infeasible"):
            report = transformed[0]["feasibility"]
            return {
                "schema_version": SCHEMA_VERSION,
                "document_type": "counterfactual_result",
                "status": "INFEASIBLE_POLICY",
                "baseline": dict(baseline_config),
                "intervention": dict(intervention),
                "response_model": spec["response_model"],
                "solution_concept": spec["solution_concept"],
                "baseline_strategy_set": [],
                "counterfactual_strategy_set": [],
                "baseline_outcome_distribution": {},
                "counterfactual_outcome_distribution": {},
                "outcome_feature_changes": {},
                "expected_utility_changes": {},
                "escalation_probability_change": 0.0,
                "catastrophe_probability_change": 0.0,
                "feasibility_report": report.to_dict(),
                "equilibrium_certificates": [],
                "multiplicity": 0,
                "equilibrium_selection_treatment": {
                    **dict(cast(Mapping[str, Any], spec["equilibrium_handling"])),
                    "diagnostic": "frozen_opponents",
                },
                "best_response_gaps": {},
                "warnings": warnings,
                "solver_runtime_seconds": time.perf_counter() - started,
                "tolerance": float(spec.get("tolerance", 1e-9)),
                "serialization_metadata": _metadata(),
            }
        counterfactual_assessments = transformed
        replacement = _plain_policy(intervention["policy"])
        player = _normalize_player(intervention["player"])
        feasibility_report = audit_policy(
            PurePolicy.from_actions(player, replacement),
            information_structure(),
            require_complete=False,
        ).to_dict()
        warnings.append("Frozen opponents do not constitute a re-equilibrated counterfactual.")
    else:
        changed_model = _changed_model(base_parameters, intervention)
        commitment_feasibility: dict[str, Any] | None = None
        allowed_initial: dict[str, tuple[str, ...]] | None
        if kind == "COMMITMENT":
            allowed_initial, commitment_feasibility = _commitment_restriction(intervention)
            warnings.append(
                "The observable binding U.S. initial-policy commitment is a model rule, "
                "not cheap talk or a historical claim."
            )
        else:
            allowed_initial = _restriction(intervention)
        counterfactual_all = _assessment_set(
            changed_model, allowed_initial=allowed_initial
        )
        counterfactual_assessments = _selected(counterfactual_all, spec)
        feasibility_report = commitment_feasibility or {
            "action_legality": {
                "status": "PASS",
                "checked_count": 0,
                "message": "structural validation",
            },
            "information_consistency": {
                "status": "PASS",
                "checked_count": 0,
                "message": "event model rebuilt",
            },
            "temporal_consistency": {
                "status": "PASS",
                "checked_count": 0,
                "message": "event chronology retained",
            },
            "commitment_consistency": {
                "status": "NOT_APPLICABLE",
                "checked_count": 0,
                "message": "no commitment",
            },
            "reachable_information_sets": [],
            "violations": [],
        }
    if not counterfactual_assessments:
        status = "NO_PURE_PBE_FOUND"
    elif kind == "POLICY_REPLACEMENT":
        status = "VALID_COUNTERFACTUAL"
    elif len(counterfactual_assessments) > 1:
        status = "MULTIPLE_SUPPORTED_EQUILIBRIA"
    elif bool(
        counterfactual_assessments[0]["certificate"].get("depends_on_off_path_beliefs", False)
    ):
        status = "DEPENDENT_ON_OFF_PATH_BELIEFS"
    else:
        status = "VALID_COUNTERFACTUAL"
    feature_changes = _feature_change_ranges(baseline_assessments, counterfactual_assessments)
    escalation_min = feature_changes.get("military_escalation.minimum", 0.0)
    catastrophe_min = feature_changes.get("catastrophic_escalation.minimum", 0.0)
    return {
        "schema_version": SCHEMA_VERSION,
        "document_type": "counterfactual_result",
        "status": status,
        "baseline": dict(baseline_config),
        "intervention": dict(intervention),
        "response_model": spec["response_model"],
        "solution_concept": spec["solution_concept"],
        "baseline_strategy_set": [
            {"id": item["id"], "policy": item["strategy_profile"]} for item in baseline_assessments
        ],
        "counterfactual_strategy_set": [
            {"id": item["id"], "policy": item["strategy_profile"]}
            for item in counterfactual_assessments
        ],
        "baseline_outcome_distribution": _solution_set_dict(baseline_assessments),
        "counterfactual_outcome_distribution": _solution_set_dict(counterfactual_assessments),
        "outcome_feature_changes": feature_changes,
        "expected_utility_changes": _utility_change_ranges(
            baseline_assessments, counterfactual_assessments
        ),
        "escalation_probability_change": escalation_min,
        "catastrophe_probability_change": catastrophe_min,
        "feasibility_report": feasibility_report,
        "equilibrium_certificates": [item["certificate"] for item in counterfactual_assessments],
        "multiplicity": len(counterfactual_assessments),
        "equilibrium_selection_treatment": dict(
            cast(Mapping[str, Any], spec["equilibrium_handling"])
        ),
        "best_response_gaps": {
            str(item["id"]): float(item["certificate"]["best_response_gap"])
            for item in counterfactual_assessments
        },
        "warnings": warnings,
        "solver_runtime_seconds": time.perf_counter() - started,
        "tolerance": float(spec.get("tolerance", 1e-9)),
        "serialization_metadata": _metadata(),
    }


def _objective_score(item: Mapping[str, Any], objective: Mapping[str, Any]) -> float:
    kind = objective.get("type")
    if kind in {"MAXIMIZE_EXPECTED_UTILITY", "MAXIMIZE_WORST_CASE_UTILITY"}:
        player = _normalize_player(objective.get("player"))
        return float(item["expected_utilities"][player])
    distribution = item["outcome_distribution"]
    if not isinstance(distribution, OutcomeDistribution):
        raise RuntimeError("malformed Cuba outcome distribution")
    if kind == "MINIMIZE_ESCALATION_PROBABILITY":
        return -distribution.probability("military_escalation")
    if kind == "MINIMIZE_CATASTROPHE_PROBABILITY":
        return -distribution.probability("catastrophic_escalation")
    if kind == "MAXIMIZE_NEGOTIATED_SETTLEMENT_PROBABILITY":
        return distribution.probability("negotiated_agreement")
    raise ValueError("Cuba adapter search does not support this objective")


def _search_constraints_pass(
    candidate: Mapping[str, Any],
    spec: Mapping[str, Any],
    baselines: Sequence[Mapping[str, Any]],
) -> bool:
    tolerance = float(spec.get("tolerance", 1e-9))
    constraints = spec.get("constraints", ())
    if not isinstance(constraints, Sequence) or isinstance(constraints, (str, bytes)):
        raise ValueError("constraints must be a sequence")
    policy = cast(Mapping[str, str], candidate["policy"])
    changed_sets = [
        {
            key
            for key, action in policy.items()
            if cast(Mapping[str, str], baseline["strategy_profile"]).get(key) != action
        }
        for baseline in baselines
    ]
    utility_ranges = cast(Mapping[str, Mapping[str, float]], candidate["expected_utility_ranges"])
    probability_ranges = cast(Mapping[str, Mapping[str, float]], candidate["probability_ranges"])
    for raw in constraints:
        if not isinstance(raw, Mapping):
            raise ValueError("each constraint must be a mapping")
        kind = raw.get("type")
        if kind == "MinimumExpectedUtility":
            if (
                utility_ranges[_normalize_player(raw.get("player"))]["minimum"]
                < float(raw["minimum"]) - tolerance
            ):
                return False
        elif kind == "MaximumEscalationProbability":
            if probability_ranges["escalation"]["maximum"] > float(raw["maximum"]) + tolerance:
                return False
        elif kind == "MaximumCatastropheProbability":
            if probability_ranges["catastrophe"]["maximum"] > float(raw["maximum"]) + tolerance:
                return False
        elif kind == "MinimumSettlementProbability":
            if probability_ranges["settlement"]["minimum"] < float(raw["minimum"]) - tolerance:
                return False
        elif kind == "MaximumPolicyChanges":
            if max((len(item) for item in changed_sets), default=0) > int(raw["maximum"]):
                return False
        elif kind == "AllowedInformationSets":
            allowed = {str(item) for item in cast(Sequence[Any], raw["information_sets"])}
            if any(not changed <= allowed for changed in changed_sets):
                return False
        else:
            raise ValueError(f"unsupported Cuba constraint {kind!r}")
    return True


def search_policies(document: Mapping[str, Any]) -> dict[str, Any]:
    """Enumerate a bounded pure-policy slice against every retained baseline."""

    if document.get("document_type") != "policy_search_request":
        raise ValueError("search requires a policy_search_request document")
    spec = _spec(document)
    validate_counterfactual(spec)
    player = _normalize_player(document.get("player"))
    allowed_raw = document.get("allowed_information_sets")
    allowed = (
        None
        if allowed_raw is None
        else tuple(str(item) for item in cast(Sequence[Any], allowed_raw))
    )
    baseline_config = _baseline(spec)
    raw_parameters = baseline_config.get("parameters", {})
    assert isinstance(raw_parameters, Mapping)
    model = CubaModel(_parameters(cast(Mapping[str, Any], raw_parameters)))
    baselines = _selected(_assessment_set(model), spec)
    if not baselines:
        raise ValueError("Cuba search cannot fix omitted decisions without a baseline PBE")
    structure = information_structure()
    owned = {key for key, info in structure.information_sets.items() if info.player_id == player}
    if allowed is not None and not set(allowed) <= owned:
        raise ValueError(
            f"unknown or unowned Cuba search information sets: {sorted(set(allowed) - owned)}"
        )
    variable = owned if allowed is None else set(allowed)
    supplied_fixed = document.get("fixed_actions", {})
    if not isinstance(supplied_fixed, Mapping):
        raise ValueError("fixed_actions must be a mapping")
    baseline_profile = cast(Mapping[str, str], baselines[0]["strategy_profile"])
    fixed_actions = {
        key: str(supplied_fixed.get(key, baseline_profile[key])) for key in sorted(owned - variable)
    }
    space = PolicySpace.from_information_structure(
        structure,
        player,
        allowed_information_sets=tuple(sorted(variable)),
        fixed_actions=fixed_actions,
    )
    maximum = int(document.get("maximum_policy_space", 100_000))
    if space.size > maximum:
        return {
            "schema_version": SCHEMA_VERSION,
            "document_type": "policy_search_result",
            "event": "cuba",
            "status": "CAPACITY_EXCEEDED",
            "exact": True,
            "estimated_policy_count": space.size,
            "evaluated_policy_count": 0,
            "retained": [],
            "narrower_legal_configuration": space.narrower_configuration(maximum),
            "warnings": ["Exact enumeration was not started."],
        }
    objective = spec.get("objective")
    if not isinstance(objective, Mapping):
        raise ValueError("objective must be a mapping")
    candidates: list[dict[str, Any]] = []
    for variable_policy in space.enumerate():
        policy = PurePolicy.from_actions(player, {**fixed_actions, **dict(variable_policy.actions)})
        report = audit_policy(policy, information_structure(), require_complete=True)
        if not report.feasible:
            continue
        transformed = _apply_policy_replacement(
            model,
            baselines,
            {"player": player, "policy": policy.actions},
        )
        scores = [
            _objective_score(item, cast(Mapping[str, Any], objective)) for item in transformed
        ]
        escalation_values = [
            cast(OutcomeDistribution, item["outcome_distribution"]).probability(
                "military_escalation"
            )
            for item in transformed
        ]
        catastrophe_values = [
            cast(OutcomeDistribution, item["outcome_distribution"]).probability(
                "catastrophic_escalation"
            )
            for item in transformed
        ]
        settlement_values = [
            cast(OutcomeDistribution, item["outcome_distribution"]).probability(
                "negotiated_agreement"
            )
            for item in transformed
        ]
        candidate = {
            "policy_id": "|".join(f"{key}={value}" for key, value in policy.actions.items()),
            "policy": dict(policy.actions),
            "objective_range": {"minimum": min(scores), "maximum": max(scores)},
            "expected_utility_ranges": {
                focal: {
                    "minimum": min(item["expected_utilities"][focal] for item in transformed),
                    "maximum": max(item["expected_utilities"][focal] for item in transformed),
                }
                for focal in (US, USSR)
            },
            "probability_ranges": {
                "escalation": {
                    "minimum": min(escalation_values),
                    "maximum": max(escalation_values),
                },
                "catastrophe": {
                    "minimum": min(catastrophe_values),
                    "maximum": max(catastrophe_values),
                },
                "settlement": {
                    "minimum": min(settlement_values),
                    "maximum": max(settlement_values),
                },
            },
            "independently_verified": True,
            "feasibility": report.to_dict(),
        }
        if _search_constraints_pass(candidate, spec, baselines):
            candidates.append(candidate)
    candidates.sort(
        key=lambda item: (
            -float(item["objective_range"]["minimum"]),
            -float(item["objective_range"]["maximum"]),
            str(item["policy_id"]),
        )
    )
    top_k = int(document.get("top_k", 10))
    cutoff = (
        float(candidates[min(top_k, len(candidates)) - 1]["objective_range"]["minimum"])
        if candidates
        else 0.0
    )
    if bool(document.get("retain_ties", True)):
        retained = [
            item
            for item in candidates
            if len(candidates) <= top_k
            or float(item["objective_range"]["minimum"])
            >= cutoff - float(spec.get("tolerance", 1e-9))
        ]
    else:
        retained = candidates[:top_k]
    return {
        "schema_version": SCHEMA_VERSION,
        "document_type": "policy_search_result",
        "event": "cuba",
        "status": "COMPLETE" if candidates else "NO_FEASIBLE_POLICY",
        "exact": True,
        "estimated_policy_count": space.size,
        "evaluated_policy_count": len(candidates),
        "feasible_policy_count": len(candidates),
        "ties_at_cutoff": max(0, len(retained) - top_k),
        "retained": retained,
        "multiplicity_treatment": "rank by worst retained-equilibrium score, then best score",
        "warnings": ["Mixed and behavioral policies were not searched."],
    }


def search_paths(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return deterministic top paths from all retained counterfactual assessments."""

    if document.get("document_type") != "path_search_request":
        raise ValueError("paths requires a path_search_request document")
    spec = _spec(document)
    evaluated = evaluate_counterfactual(spec)
    # Recompute the small internal objects because the public result intentionally
    # contains only serialized distributions, not path records.
    baseline_config = _baseline(spec)
    raw_parameters = baseline_config.get("parameters", {})
    assert isinstance(raw_parameters, Mapping)
    base = _parameters(cast(Mapping[str, Any], raw_parameters))
    intervention = cast(Mapping[str, Any], spec["intervention"])
    model = _changed_model(base, intervention)
    baseline_set = _selected(_assessment_set(CubaModel(base)), spec)
    if intervention["type"] == "POLICY_REPLACEMENT":
        assessments = _apply_policy_replacement(CubaModel(base), baseline_set, intervention)
    else:
        assessments = _selected(
            _assessment_set(model, allowed_initial=_restriction(intervention)), spec
        )
    if not assessments:
        return {
            "schema_version": SCHEMA_VERSION,
            "document_type": "path_search_result",
            "event": "cuba",
            "status": "NO_PURE_PBE_FOUND",
            "ranking": str(document.get("ranking", "EQUILIBRIUM_REACH_PROBABILITY")),
            "paths": [],
            "model_reachable_is_not_unilaterally_inducible": True,
            "warnings": ["No supported pure-PBE path set was found."],
        }

    def path_key(path: Mapping[str, Any]) -> tuple[str, str]:
        return str(path["terminal_id"]), canonical_json(path["events"])

    paths: list[dict[str, Any]] = []
    for counterfactual in assessments:
        for baseline_assessment in baseline_set:
            baseline_paths = tuple(
                cast(Sequence[Mapping[str, Any]], baseline_assessment.get("paths", ()))
            )
            baseline_lookup = {
                path_key(path): float(path["reach_probability"]) for path in baseline_paths
            }
            for raw_path in cast(Sequence[Mapping[str, Any]], counterfactual.get("paths", ())):
                path = dict(raw_path)
                path["baseline_solution_id"] = baseline_assessment["id"]
                path["counterfactual_solution_id"] = counterfactual["id"]
                path["baseline_reach_probability"] = baseline_lookup.get(path_key(raw_path), 0.0)
                path["equilibrium_reach_probability"] = (
                    float(raw_path["reach_probability"])
                    if bool(raw_path.get("equilibrium_supported"))
                    else None
                )
                counterfactual_decisions = [
                    event
                    for event in cast(Sequence[Mapping[str, Any]], raw_path["events"])
                    if event.get("actor") != "nature"
                ]
                closest = min(
                    (
                        [
                            event
                            for event in cast(Sequence[Mapping[str, Any]], baseline_path["events"])
                            if event.get("actor") != "nature"
                        ]
                        for baseline_path in baseline_paths
                    ),
                    key=lambda baseline_decisions: (
                        sum(
                            left.get("action") != right.get("action")
                            for left, right in zip(
                                counterfactual_decisions,
                                baseline_decisions,
                                strict=False,
                            )
                        )
                        + abs(len(counterfactual_decisions) - len(baseline_decisions))
                    ),
                )
                differences = [
                    {
                        "decision_index": index,
                        "actor": current.get("actor"),
                        "baseline_action": old.get("action"),
                        "path_action": current.get("action"),
                    }
                    for index, (current, old) in enumerate(
                        zip(counterfactual_decisions, closest, strict=False)
                    )
                    if current.get("action") != old.get("action")
                ]
                path["differences_from_baseline"] = differences
                path["deviation_count"] = len(differences) + abs(
                    len(counterfactual_decisions) - len(closest)
                )
                path.setdefault("information_available_at_changes", [])
                paths.append(path)
    ranking = str(document.get("ranking", "EQUILIBRIUM_REACH_PROBABILITY"))
    player = (
        _normalize_player(document.get("player")) if document.get("player") is not None else None
    )

    def score(path: Mapping[str, Any]) -> float:
        outcome = cast(Mapping[str, Any], path["terminal_outcome"])
        common = cast(Mapping[str, Any], outcome["common"])
        if ranking == "EQUILIBRIUM_REACH_PROBABILITY":
            value = path.get("equilibrium_reach_probability")
            if value is None:
                raise ValueError(
                    "equilibrium-reach ranking is unavailable for a frozen supplied policy"
                )
            return float(value)
        if ranking == "BASELINE_REACH_PROBABILITY":
            return float(path["baseline_reach_probability"])
        if ranking == "FOCAL_PLAYER_UTILITY":
            if player is None:
                raise ValueError("FOCAL_PLAYER_UTILITY requires player")
            return float(cast(Mapping[str, Any], path["utilities"])[player])
        if ranking == "JOINT_UTILITY":
            return sum(
                float(value) for value in cast(Mapping[str, Any], path["utilities"]).values()
            )
        if ranking == "ESCALATION_AVOIDANCE":
            return -float(bool(common.get("military_escalation", False)))
        if ranking == "CATASTROPHE_AVOIDANCE":
            return -float(bool(common.get("catastrophic_escalation", False)))
        if ranking == "NEGOTIATED_SETTLEMENT":
            return float(bool(common.get("negotiated_agreement", False)))
        if ranking in {"ROBUSTNESS", "ROBUST_WORST_CASE_UTILITY"}:
            raise ValueError(
                "Cuba robust path ranking requires an uncertainty set and is unsupported"
            )
        if ranking == "MODEL_REACHABILITY":
            return float(bool(path.get("model_reachable", False)))
        if ranking == "EQUILIBRIUM_SUPPORT":
            return float(bool(path.get("equilibrium_supported", False)))
        if ranking in {"MINIMUM_DEVIATION_COUNT", "MINIMUM_DEVIATIONS"}:
            return -float(path["deviation_count"])
        raise ValueError(f"unsupported Cuba path ranking {ranking!r}")

    unique = {
        (
            str(path["baseline_solution_id"]),
            str(path["counterfactual_solution_id"]),
            str(path["terminal_id"]),
            canonical_json(path["events"]),
        ): path
        for path in paths
    }
    ordered = sorted(unique.values(), key=lambda path: (-score(path), str(path["terminal_id"])))
    top_k = int(document.get("top_k", 10))
    selected = ordered[:top_k]
    if len(ordered) > top_k:
        cutoff = score(ordered[top_k - 1])
        selected = [
            path for path in ordered if score(path) >= cutoff - float(spec.get("tolerance", 1e-9))
        ]
    return {
        "schema_version": SCHEMA_VERSION,
        "document_type": "path_search_result",
        "event": "cuba",
        "status": evaluated["status"],
        "ranking": ranking,
        "paths": selected,
        "model_reachable_is_not_unilaterally_inducible": True,
        "warnings": [
            "Path existence does not establish that either player can unilaterally induce it."
        ],
    }


def frontier(document: Mapping[str, Any]) -> dict[str, Any]:
    expanded = dict(document)
    expanded["top_k"] = int(document.get("maximum_policy_space", 100_000))
    expanded["retain_ties"] = True
    result = search_policies(expanded)
    candidates = cast(list[dict[str, Any]], result.get("retained", []))
    frontier_items: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_ranges = cast(
            Mapping[str, Mapping[str, float]], candidate["expected_utility_ranges"]
        )
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            other_ranges = cast(Mapping[str, Mapping[str, float]], other["expected_utility_ranges"])
            weak = all(
                other_ranges[player]["minimum"] >= candidate_ranges[player]["minimum"] - 1e-9
                for player in (US, USSR)
            )
            strict = any(
                other_ranges[player]["minimum"] > candidate_ranges[player]["minimum"] + 1e-9
                for player in (US, USSR)
            )
            if weak and strict:
                dominated = True
                break
        if not dominated:
            frontier_items.append(candidate)
    return {
        "schema_version": SCHEMA_VERSION,
        "document_type": "pareto_frontier_result",
        "event": "cuba",
        "status": result["status"],
        "frontier": frontier_items,
        "comparison": "worst retained-equilibrium utility by player",
        "warnings": result.get("warnings", []),
    }


def route_counterfactual(document: Mapping[str, Any], operation: str) -> dict[str, Any]:
    """Route a deterministic counterfactual operation."""

    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError(f"unsupported Cuba counterfactual operation {operation!r}")
    if operation == "validate":
        return validate_counterfactual(document)
    if operation == "evaluate":
        return evaluate_counterfactual(document)
    if operation == "search":
        return search_policies(document)
    if operation == "paths":
        return search_paths(document)
    return frontier(document)


__all__ = [
    "US",
    "USSR",
    "evaluate_counterfactual",
    "frontier",
    "information_structure",
    "route_counterfactual",
    "search_paths",
    "search_policies",
    "terminal_outcome_features",
    "validate_counterfactual",
]
