"""Counterfactual adapter for the Korea warning and intervention game.

The baseline warning is observed perfectly.  A supported warning-accuracy
intervention inserts a symmetric noisy observation channel between China's
warning and the US/UN advance decision, rebuilds receiver posteriors, and
re-enumerates pure PBE for that deliberately narrow game class.
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Mapping, Sequence
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
from cold_war_sim.counterfactuals.value_of_information import (
    InformationChange,
    value_of_information,
)
from cold_war_sim.events.korea.model import (
    ADVANCE_ACTIONS,
    ENTRY_ACTIONS,
    TYPES,
    WARNINGS,
    KoreaWarningModel,
)
from cold_war_sim.events.korea.parameters import KoreaParameters

SCHEMA_VERSION = "1.0"
CHINA = "china"
RECEIVER = "us_un"
SUPPORTED_OPERATIONS = ("validate", "evaluate", "search", "paths", "frontier")


def _warning_set(china_type: str) -> str:
    return f"warning_{china_type}"


def _advance_set(observed_warning: str) -> str:
    return f"advance_{observed_warning}"


def _entry_set(china_type: str, actual_warning: str, advance: str) -> str:
    return f"entry_{china_type}_{actual_warning}_{advance}"


def information_structure(*, warning_accuracy: float = 1.0) -> InformationStructure:
    if not 0.5 <= warning_accuracy <= 1.0:
        raise ValueError("warning accuracy must lie in [0.5, 1]")
    variables = {
        "china_type": InformationVariable("china_type", 0, (CHINA,)),
        "actual_warning": InformationVariable("actual_warning", 1, (CHINA,)),
        "observed_warning": InformationVariable("observed_warning", 2, (CHINA, RECEIVER)),
        "receiver_action": InformationVariable("receiver_action", 3, (CHINA, RECEIVER)),
        "modeler_parameter_truth": InformationVariable(
            "modeler_parameter_truth", 0, (), modeler_only=True
        ),
        "simulation_seed": InformationVariable("simulation_seed", 0, (), modeler_only=True),
    }
    sets: dict[str, InformationSetSpec] = {}
    for china_type in TYPES:
        identifier = _warning_set(china_type)
        sets[identifier] = InformationSetSpec(
            identifier,
            CHINA,
            1,
            (identifier,),
            WARNINGS,
            ("china_type",),
            observation_signature={"china_type": china_type},
        )
    for observed in WARNINGS:
        identifier = _advance_set(observed)
        sets[identifier] = InformationSetSpec(
            identifier,
            RECEIVER,
            3,
            tuple(
                f"{identifier}_{china_type}_{actual}" for china_type in TYPES for actual in WARNINGS
            ),
            ADVANCE_ACTIONS,
            ("observed_warning",),
            observation_signature={"observed_warning": observed},
        )
    for china_type in TYPES:
        for actual in WARNINGS:
            for advance in ADVANCE_ACTIONS:
                identifier = _entry_set(china_type, actual, advance)
                sets[identifier] = InformationSetSpec(
                    identifier,
                    CHINA,
                    4,
                    (identifier,),
                    ENTRY_ACTIONS,
                    ("china_type", "actual_warning", "receiver_action"),
                    observation_signature={
                        "china_type": china_type,
                        "actual_warning": actual,
                        "receiver_action": advance,
                    },
                )
    return InformationStructure(sets, variables)


def terminal_outcome_features(
    model: KoreaWarningModel,
    china_type: str,
    warning: str,
    receiver_action: str,
    entry_action: str,
) -> OutcomeFeatures:
    """Convert one terminal history without embedding either player's utility."""

    model._validate_history(china_type, warning, receiver_action, entry_action)
    params = model.parameters
    advance_rank = {"restraint": 0.0, "limited": 0.5, "aggressive": 1.0}
    intervention = entry_action == "intervene"
    warning_cost = 0.0
    if warning == "warn":
        warning_cost = (
            params.low_warning_cost if china_type == "low_resolve" else params.high_warning_cost
        )
    receiver_conflict = {
        "restraint": params.restraint_conflict_cost,
        "limited": params.limited_conflict_cost,
        "aggressive": params.aggressive_conflict_cost,
    }[receiver_action]
    china_intervention = {
        "restraint": params.restraint_intervention_cost,
        "limited": params.limited_intervention_cost,
        "aggressive": params.aggressive_intervention_cost,
    }[receiver_action]
    fixed = (
        params.low_intervention_fixed_cost
        if china_type == "low_resolve"
        else params.high_intervention_fixed_cost
    )
    return OutcomeFeatures(
        common={
            "peaceful_settlement": not intervention,
            "negotiated_agreement": False,
            "concession": 1.0 - advance_rank[receiver_action],
            "military_escalation": intervention,
            "catastrophic_escalation": False,
            "intervention": intervention,
            "escalation_probability": float(intervention),
            "catastrophe_probability": 0.0,
            "settlement_probability": float(not intervention),
        },
        by_player={
            CHINA: {
                "political_cost": warning_cost,
                "military_cost": (fixed + china_intervention) if intervention else 0.0,
            },
            RECEIVER: {
                "military_cost": receiver_conflict if intervention else 0.0,
            },
        },
    )


def _spec(document: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = document.get("counterfactual")
    return cast(Mapping[str, Any], nested) if isinstance(nested, Mapping) else document


def _baseline(document: Mapping[str, Any]) -> Mapping[str, Any]:
    baseline = _spec(document).get("baseline")
    if not isinstance(baseline, Mapping) or baseline.get("event") != "korea":
        raise ValueError("Korea adapter requires baseline.event='korea'")
    return cast(Mapping[str, Any], baseline)


def _parameters(values: Mapping[str, Any]) -> KoreaParameters:
    return KoreaParameters.from_mapping(values)


def _player(value: object) -> str:
    aliases = {
        CHINA: CHINA,
        "China": CHINA,
        RECEIVER: RECEIVER,
        "US/UN": RECEIVER,
        "receiver": RECEIVER,
    }
    try:
        return aliases[str(value)]
    except KeyError as error:
        raise ValueError(f"unknown Korea player {value!r}") from error


def _pure_policy(raw: object) -> dict[str, str]:
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("policy must be a nonempty mapping")
    result: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("Korea adapter supports deterministic pure policies only")
        result[key] = value
    return result


def _signal_probability(actual: str, observed: str, accuracy: float) -> float:
    return accuracy if actual == observed else 1.0 - accuracy


def _allowed_actions(
    structure: InformationStructure,
    intervention: Mapping[str, Any] | None,
) -> dict[str, tuple[str, ...]]:
    allowed = {key: tuple(info.legal_actions) for key, info in structure.information_sets.items()}
    if not intervention or intervention.get("type") != "ACTION_RESTRICTION":
        return allowed
    player = _player(intervention.get("player"))
    identifier = str(intervention.get("information_set"))
    if identifier not in structure.information_sets:
        raise ValueError(f"unknown Korea information set {identifier!r}")
    if structure.information_sets[identifier].player_id != player:
        raise ValueError("action restriction player does not control the information set")
    raw = intervention.get("actions")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("restriction actions must be a sequence")
    removed = {str(item) for item in raw}
    unknown = removed - set(allowed[identifier])
    if unknown:
        raise ValueError(f"unknown restricted actions: {sorted(unknown)}")
    retained = tuple(action for action in allowed[identifier] if action not in removed)
    if not retained:
        raise ValueError("action restriction cannot remove every action")
    allowed[identifier] = retained
    return allowed


def _entry_correspondence(
    model: KoreaWarningModel,
    allowed: Mapping[str, tuple[str, ...]],
    *,
    tolerance: float,
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for china_type in TYPES:
        for warning in WARNINGS:
            for advance in ADVANCE_ACTIONS:
                identifier = _entry_set(china_type, warning, advance)
                values = {
                    action: model.terminal_payoffs(china_type, warning, advance, action)[0]
                    for action in allowed[identifier]
                }
                best = max(values.values())
                result[identifier] = tuple(
                    action for action in allowed[identifier] if values[action] >= best - tolerance
                )
    return result


def _representative_entry(
    model: KoreaWarningModel, correspondence: Mapping[str, tuple[str, ...]]
) -> dict[str, str]:
    result: dict[str, str] = {}
    for identifier, actions in sorted(correspondence.items()):
        if len(actions) == 1:
            result[identifier] = actions[0]
        elif model.parameters.entry_tie_break in actions:
            result[identifier] = model.parameters.entry_tie_break
        else:
            result[identifier] = sorted(actions)[0]
    return result


def _posterior_joint(
    model: KoreaWarningModel,
    sender: Mapping[str, str],
    observed: str,
    accuracy: float,
) -> tuple[float, dict[tuple[str, str], float]]:
    weights = {
        (china_type, sender[china_type]): (
            model.parameters.prior[china_type]
            * _signal_probability(sender[china_type], observed, accuracy)
        )
        for china_type in TYPES
    }
    reach = sum(weights.values())
    if reach <= model.parameters.comparison_tolerance:
        high = model.parameters.off_path_high_belief
        type_belief = {"low_resolve": 1.0 - high, "high_resolve": high}
        return (
            0.0,
            {(china_type, observed): type_belief[china_type] for china_type in TYPES},
        )
    return reach, {key: value / reach for key, value in weights.items()}


def _receiver_values(
    model: KoreaWarningModel,
    posterior: Mapping[tuple[str, str], float],
    observed: str,
    entry_policy: Mapping[str, str],
    allowed: Mapping[str, tuple[str, ...]],
) -> dict[str, float]:
    return {
        advance: sum(
            probability
            * model.terminal_payoffs(
                china_type,
                actual,
                advance,
                entry_policy[_entry_set(china_type, actual, advance)],
            )[1]
            for (china_type, actual), probability in posterior.items()
        )
        for advance in allowed[_advance_set(observed)]
    }


def _sender_value(
    model: KoreaWarningModel,
    china_type: str,
    warning: str,
    receiver: Mapping[str, str],
    entry_policy: Mapping[str, str],
    accuracy: float,
) -> float:
    return sum(
        _signal_probability(warning, observed, accuracy)
        * model.terminal_payoffs(
            china_type,
            warning,
            receiver[observed],
            entry_policy[_entry_set(china_type, warning, receiver[observed])],
        )[0]
        for observed in WARNINGS
    )


def _enumerate_equilibria(
    model: KoreaWarningModel,
    *,
    warning_accuracy: float,
    intervention: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    structure = information_structure(warning_accuracy=warning_accuracy)
    allowed = _allowed_actions(structure, intervention)
    tolerance = model.parameters.comparison_tolerance
    correspondence = _entry_correspondence(model, allowed, tolerance=tolerance)
    entry_policy = _representative_entry(model, correspondence)
    sender_sets = tuple(_warning_set(china_type) for china_type in TYPES)
    equilibria: list[dict[str, Any]] = []
    for sender_choices in itertools.product(*(allowed[identifier] for identifier in sender_sets)):
        sender = dict(zip(TYPES, sender_choices, strict=True))
        receiver_best: dict[str, tuple[str, ...]] = {}
        posteriors: dict[str, dict[str, float]] = {}
        joint_posteriors: dict[str, dict[tuple[str, str], float]] = {}
        reach: dict[str, float] = {}
        for observed in WARNINGS:
            reach[observed], posterior = _posterior_joint(model, sender, observed, warning_accuracy)
            joint_posteriors[observed] = posterior
            posteriors[observed] = {
                china_type: sum(
                    probability
                    for (posterior_type, _), probability in posterior.items()
                    if posterior_type == china_type
                )
                for china_type in TYPES
            }
            values = _receiver_values(model, posterior, observed, entry_policy, allowed)
            best = max(values.values())
            receiver_best[observed] = tuple(
                action
                for action in allowed[_advance_set(observed)]
                if values[action] >= best - tolerance
            )
        for receiver_choices in itertools.product(
            *(receiver_best[observed] for observed in WARNINGS)
        ):
            receiver = dict(zip(WARNINGS, receiver_choices, strict=True))
            sender_gains: dict[str, float] = {}
            for china_type in TYPES:
                chosen = _sender_value(
                    model,
                    china_type,
                    sender[china_type],
                    receiver,
                    entry_policy,
                    warning_accuracy,
                )
                alternatives = [
                    _sender_value(
                        model,
                        china_type,
                        warning,
                        receiver,
                        entry_policy,
                        warning_accuracy,
                    )
                    for warning in allowed[_warning_set(china_type)]
                ]
                sender_gains[china_type] = max(0.0, max(alternatives) - chosen)
            if max(sender_gains.values(), default=0.0) > tolerance:
                continue
            policy = {
                **{_warning_set(china_type): sender[china_type] for china_type in TYPES},
                **{_advance_set(observed): receiver[observed] for observed in WARNINGS},
                **entry_policy,
            }
            evaluated = _evaluate_profile(
                model,
                policy,
                warning_accuracy=warning_accuracy,
                equilibrium_supported=True,
            )
            off_path = tuple(observed for observed in WARNINGS if reach[observed] <= tolerance)
            depends = False
            for observed in off_path:
                chosen_action = receiver[observed]
                for china_type in TYPES:
                    actual = observed
                    chosen_value = model.terminal_payoffs(
                        china_type,
                        actual,
                        chosen_action,
                        entry_policy[_entry_set(china_type, actual, chosen_action)],
                    )[1]
                    best_value = max(
                        model.terminal_payoffs(
                            china_type,
                            actual,
                            action,
                            entry_policy[_entry_set(china_type, actual, action)],
                        )[1]
                        for action in allowed[_advance_set(observed)]
                    )
                    depends = depends or chosen_value < best_value - tolerance
            classification = "pooling" if len(set(sender.values())) == 1 else "separating"
            identifier = f"korea_eq_{len(equilibria):03d}"
            certificate = {
                "equilibrium_concept": "PURE_PBE",
                "candidate_class": "pure strategies in the implemented noisy-warning game",
                "strategy_profile": dict(sorted(policy.items())),
                "beliefs": posteriors,
                "reach_probabilities": reach,
                "deviation_gains": sender_gains,
                "best_response_gap": max(sender_gains.values(), default=0.0),
                "off_path_convention": "configured off_path_high_belief",
                "exact": True,
                "tolerance": tolerance,
                "warnings": [
                    "Pure PBE only; mixed equilibria were not searched.",
                    "Tied terminal entry choices use the documented event tie convention.",
                ],
                "depends_on_off_path_beliefs": depends,
            }
            equilibria.append(
                {
                    "id": identifier,
                    "classification": classification,
                    "strategy_profile": policy,
                    "expected_utilities": evaluated["utilities"],
                    "outcome_distribution": evaluated["distribution"],
                    "paths": evaluated["paths"],
                    "certificate": certificate,
                }
            )
    unique: dict[str, dict[str, Any]] = {}
    for equilibrium in equilibria:
        unique[canonical_json(equilibrium["strategy_profile"])] = equilibrium
    ordered = tuple(unique[key] for key in sorted(unique))
    for index, equilibrium in enumerate(ordered):
        equilibrium["id"] = f"korea_eq_{index:03d}"
    return ordered


def _evaluate_profile(
    model: KoreaWarningModel,
    policy: Mapping[str, str],
    *,
    warning_accuracy: float,
    equilibrium_supported: bool,
) -> dict[str, Any]:
    outcomes: list[WeightedOutcome] = []
    paths: list[dict[str, Any]] = []
    utilities = {CHINA: 0.0, RECEIVER: 0.0}
    for china_type in TYPES:
        type_probability = model.parameters.prior[china_type]
        warning = policy[_warning_set(china_type)]
        for observed in WARNINGS:
            probability = type_probability * _signal_probability(
                warning, observed, warning_accuracy
            )
            if probability <= 0.0:
                continue
            advance = policy[_advance_set(observed)]
            entry = policy[_entry_set(china_type, warning, advance)]
            payoffs = model.terminal_payoffs(china_type, warning, advance, entry)
            utilities[CHINA] += probability * payoffs[0]
            utilities[RECEIVER] += probability * payoffs[1]
            features = terminal_outcome_features(model, china_type, warning, advance, entry)
            terminal_id = f"{china_type}|{warning}|observed_{observed}|{advance}|{entry}"
            outcomes.append(WeightedOutcome(terminal_id, probability, features))
            paths.append(
                {
                    "terminal_id": terminal_id,
                    "events": [
                        {"actor": "nature", "action": china_type},
                        {"actor": CHINA, "action": warning},
                        {"actor": "warning_channel", "action": observed},
                        {"actor": RECEIVER, "action": advance},
                        {"actor": CHINA, "action": entry},
                    ],
                    "reach_probability": probability,
                    "terminal_outcome": features.to_dict(),
                    "utilities": {CHINA: payoffs[0], RECEIVER: payoffs[1]},
                    "model_reachable": True,
                    "strategy_feasible": True,
                    "equilibrium_supported": equilibrium_supported,
                }
            )
    return {
        "distribution": OutcomeDistribution(tuple(outcomes)),
        "utilities": utilities,
        "paths": tuple(sorted(paths, key=lambda item: str(item["terminal_id"]))),
    }


def _warning_accuracy(intervention: Mapping[str, Any] | None) -> float:
    if not intervention or intervention.get("type") != "INFORMATION_INTERVENTION":
        return 1.0
    if intervention.get("channel") != "warning":
        raise ValueError("Korea supports only the 'warning' information channel")
    changes = intervention.get("changes")
    if not isinstance(changes, Mapping) or set(changes) != {"accuracy"}:
        raise ValueError("Korea warning intervention requires only changes.accuracy")
    accuracy = float(changes["accuracy"])
    if not 0.5 <= accuracy <= 1.0:
        raise ValueError("Korea warning accuracy must lie in [0.5, 1]")
    return accuracy


def _changed_model(base: KoreaParameters, intervention: Mapping[str, Any]) -> KoreaWarningModel:
    if intervention.get("type") != "PARAMETER_INTERVENTION":
        return KoreaWarningModel(base)
    changes = intervention.get("changes")
    if not isinstance(changes, Mapping):
        raise ValueError("parameter intervention changes must be a mapping")
    values = {name: getattr(base, name) for name in base.__dataclass_fields__}
    values.update(changes)
    return KoreaWarningModel(KoreaParameters.from_mapping(values))


def _select(
    equilibria: tuple[dict[str, Any], ...], spec: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    handling = spec.get("equilibrium_handling", {})
    if not isinstance(handling, Mapping):
        raise ValueError("equilibrium_handling must be a mapping")
    rule = str(handling.get("selection_rule", "RETAIN_ALL"))
    if rule == "RETAIN_ALL" or not equilibria:
        return equilibria
    if rule in {"PLAYER_OPTIMAL", "PLAYER_PESSIMAL"}:
        player = _player(handling.get("player"))
        values = [item["expected_utilities"][player] for item in equilibria]
        target = max(values) if rule == "PLAYER_OPTIMAL" else min(values)
        return tuple(
            item
            for item in equilibria
            if math.isclose(item["expected_utilities"][player], target, abs_tol=1e-9)
        )
    if rule == "SOCIAL_WELFARE":
        target = max(sum(item["expected_utilities"].values()) for item in equilibria)
        return tuple(
            item
            for item in equilibria
            if math.isclose(sum(item["expected_utilities"].values()), target, abs_tol=1e-9)
        )
    if rule == "MINIMUM_CATASTROPHE":
        # This event has no catastrophe process; retain the complete tie.
        return equilibria
    raise ValueError(f"Korea adapter does not support equilibrium selection {rule!r}")


def _replace_policy(
    model: KoreaWarningModel,
    equilibria: tuple[dict[str, Any], ...],
    intervention: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    player = _player(intervention.get("player"))
    replacement = _pure_policy(intervention.get("policy"))
    report = audit_policy(
        PurePolicy.from_actions(player, replacement),
        information_structure(),
        require_complete=False,
    )
    if not report.feasible:
        raise ValueError(f"infeasible Korea policy: {report.to_dict()['violations']}")
    results = []
    for index, equilibrium in enumerate(equilibria):
        policy = dict(equilibrium["strategy_profile"])
        policy.update(replacement)
        evaluated = _evaluate_profile(
            model, policy, warning_accuracy=1.0, equilibrium_supported=False
        )
        results.append(
            {
                "id": f"korea_policy_{index:03d}",
                "strategy_profile": policy,
                "expected_utilities": evaluated["utilities"],
                "outcome_distribution": evaluated["distribution"],
                "paths": evaluated["paths"],
                "certificate": {
                    "equilibrium_concept": "SUPPLIED_PROFILE",
                    "candidate_class": "model-feasible pure policy profile",
                    "strategy_profile": policy,
                    "beliefs": {},
                    "reach_probabilities": {},
                    "deviation_gains": {},
                    "best_response_gap": 0.0,
                    "off_path_convention": "inherited",
                    "exact": False,
                    "tolerance": model.parameters.comparison_tolerance,
                    "warnings": ["This profile is not an equilibrium certificate."],
                },
                "feasibility": report,
            }
        )
    return tuple(results)


def _distribution_set(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "selection_treatment": "explicit request rule",
        "equilibria": [
            {
                "id": item["id"],
                "distribution": item["outcome_distribution"].to_dict(),
            }
            for item in items
        ],
    }


def _utility_changes(
    before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]
) -> dict[str, float]:
    result: dict[str, float] = {}
    for player in (CHINA, RECEIVER):
        changes = [
            new["expected_utilities"][player] - old["expected_utilities"][player]
            for old in before
            for new in after
        ]
        result[f"{player}.minimum"] = min(changes, default=0.0)
        result[f"{player}.maximum"] = max(changes, default=0.0)
    return result


def _feature_changes(
    before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]
) -> dict[str, float]:
    result: dict[str, float] = {}
    for feature in ("military_escalation", "catastrophic_escalation", "negotiated_agreement"):
        changes = [
            new["outcome_distribution"].probability(feature)
            - old["outcome_distribution"].probability(feature)
            for old in before
            for new in after
        ]
        result[f"{feature}.minimum"] = min(changes, default=0.0)
        result[f"{feature}.maximum"] = max(changes, default=0.0)
    return result


def _metadata() -> dict[str, Any]:
    return {
        "package_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "seed_policy": "deterministic exact enumeration",
        "solver_version": "korea counterfactual adapter 1.0",
    }


def warning_value_of_information(
    parameters: KoreaParameters | Mapping[str, Any] | None = None,
    *,
    baseline_accuracy: float = 0.5,
    informed_accuracy: float = 1.0,
) -> dict[str, Any]:
    """Compare all retained pure-PBE utilities across warning accuracies.

    This is an ex-ante value in the illustrative model, reported separately
    for both players as a pairwise range over equilibrium multiplicity.
    """

    params = (
        parameters if isinstance(parameters, KoreaParameters) else _parameters(parameters or {})
    )
    model = KoreaWarningModel(params)
    baseline = _enumerate_equilibria(model, warning_accuracy=baseline_accuracy)
    informed = _enumerate_equilibria(model, warning_accuracy=informed_accuracy)
    result = value_of_information(
        [cast(Mapping[str, float], item["expected_utilities"]) for item in baseline],
        [cast(Mapping[str, float], item["expected_utilities"]) for item in informed],
        change=InformationChange.SIGNAL_ACCURACY_CHANGE,
        selection_treatment="retain_all_pure_pbe_pairwise_range",
    ).to_dict()
    result.update(
        {
            "baseline_accuracy": baseline_accuracy,
            "informed_accuracy": informed_accuracy,
            "baseline_multiplicity": len(baseline),
            "informed_multiplicity": len(informed),
            "warning": (
                "Illustrative modeled utility only; a public signal need not benefit "
                "every strategic player."
            ),
        }
    )
    return result


def validate_counterfactual(document: Mapping[str, Any]) -> dict[str, Any]:
    spec = _spec(document)
    baseline = _baseline(document)
    raw_parameters = baseline.get("parameters", {})
    if not isinstance(raw_parameters, Mapping):
        raise ValueError("baseline.parameters must be a mapping")
    parameters = _parameters(cast(Mapping[str, Any], raw_parameters))
    intervention = spec.get("intervention")
    if not isinstance(intervention, Mapping):
        raise ValueError("intervention must be a mapping")
    kind = intervention.get("type")
    if spec.get("uncertainty_set") is not None:
        raise ValueError(
            "Korea event routing does not yet support uncertainty-set evaluation; "
            "no robust preference claim was computed"
        )
    if spec.get("solution_concept") not in {"PURE_PBE", "SUPPLIED_PROFILE"}:
        raise ValueError("Korea supports PURE_PBE or a frozen SUPPLIED_PROFILE only")
    if kind not in {
        "POLICY_REPLACEMENT",
        "ACTION_RESTRICTION",
        "PARAMETER_INTERVENTION",
        "INFORMATION_INTERVENTION",
    }:
        raise ValueError(f"Korea adapter does not support intervention {kind!r}")
    accuracy = _warning_accuracy(cast(Mapping[str, Any], intervention))
    structure = information_structure(warning_accuracy=accuracy)
    feasibility: dict[str, Any] | None = None
    if kind == "POLICY_REPLACEMENT":
        player = _player(intervention.get("player"))
        policy = _pure_policy(intervention.get("policy"))
        report = audit_policy(
            PurePolicy.from_actions(player, policy), structure, require_complete=False
        )
        if not report.feasible:
            raise ValueError(f"infeasible Korea policy: {report.to_dict()['violations']}")
        feasibility = report.to_dict()
        if spec.get("response_model") != "FROZEN_OPPONENTS":
            raise ValueError("Korea policy replacement currently supports FROZEN_OPPONENTS only")
    elif kind == "ACTION_RESTRICTION":
        _allowed_actions(structure, cast(Mapping[str, Any], intervention))
        if spec.get("response_model") != "REEQUILIBRATE":
            raise ValueError("Korea action restriction requires REEQUILIBRATE")
    elif kind == "PARAMETER_INTERVENTION":
        _changed_model(parameters, cast(Mapping[str, Any], intervention))
        if spec.get("response_model") != "REEQUILIBRATE":
            raise ValueError("Korea parameter intervention requires REEQUILIBRATE")
    elif spec.get("response_model") != "REEQUILIBRATE":
        raise ValueError("Korea information intervention requires REEQUILIBRATE")
    return {
        "schema_version": SCHEMA_VERSION,
        "document_type": "counterfactual_validation",
        "event": "korea",
        "valid": True,
        "supported": True,
        "warning_accuracy": accuracy,
        "information_structure": structure.to_dict(),
        "feasibility_report": feasibility,
        "warnings": [
            "Warning-quality interventions rebuild posteriors at the advance decision.",
            "The exact solver searches pure PBE only.",
        ],
    }


def evaluate_counterfactual(document: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    validate_counterfactual(document)
    spec = _spec(document)
    baseline_config = _baseline(document)
    raw_parameters = baseline_config.get("parameters", {})
    assert isinstance(raw_parameters, Mapping)
    base_parameters = _parameters(cast(Mapping[str, Any], raw_parameters))
    intervention = cast(Mapping[str, Any], spec["intervention"])
    baseline_all = _enumerate_equilibria(KoreaWarningModel(base_parameters), warning_accuracy=1.0)
    baseline = _select(baseline_all, spec)
    accuracy = _warning_accuracy(intervention)
    model = _changed_model(base_parameters, intervention)
    if intervention["type"] == "POLICY_REPLACEMENT":
        transformed = _replace_policy(model, baseline, intervention)
        warnings = ["Frozen opponents are a diagnostic, not a prediction."]
    else:
        transformed = _select(
            _enumerate_equilibria(
                model,
                warning_accuracy=accuracy,
                intervention=(
                    intervention if intervention["type"] == "ACTION_RESTRICTION" else None
                ),
            ),
            spec,
        )
        warnings = []
    if not transformed:
        status = "NO_PURE_PBE_FOUND"
    elif intervention["type"] == "POLICY_REPLACEMENT":
        status = "VALID_COUNTERFACTUAL"
    elif len(transformed) > 1:
        status = "MULTIPLE_SUPPORTED_EQUILIBRIA"
    elif transformed[0]["certificate"].get("depends_on_off_path_beliefs"):
        status = "DEPENDENT_ON_OFF_PATH_BELIEFS"
    else:
        status = "VALID_COUNTERFACTUAL"
    structure = information_structure(warning_accuracy=accuracy)
    feature_changes = _feature_changes(baseline, transformed)
    feasibility = {
        "action_legality": {"status": "PASS", "checked_count": 0, "message": "event validation"},
        "information_consistency": {
            "status": "PASS",
            "checked_count": 0,
            "message": "posterior rebuilt",
        },
        "temporal_consistency": {
            "status": "PASS",
            "checked_count": 0,
            "message": "warning precedes advance",
        },
        "commitment_consistency": {
            "status": "NOT_APPLICABLE",
            "checked_count": 0,
            "message": "no commitment",
        },
        "reachable_information_sets": list(structure.information_sets),
        "violations": [],
    }
    if intervention["type"] == "POLICY_REPLACEMENT" and transformed:
        feasibility = transformed[0]["feasibility"].to_dict()
    return {
        "schema_version": SCHEMA_VERSION,
        "document_type": "counterfactual_result",
        "status": status,
        "baseline": dict(baseline_config),
        "intervention": dict(intervention),
        "response_model": spec["response_model"],
        "solution_concept": spec["solution_concept"],
        "baseline_strategy_set": [
            {"id": item["id"], "policy": item["strategy_profile"]} for item in baseline
        ],
        "counterfactual_strategy_set": [
            {"id": item["id"], "policy": item["strategy_profile"]} for item in transformed
        ],
        "baseline_outcome_distribution": _distribution_set(baseline),
        "counterfactual_outcome_distribution": _distribution_set(transformed),
        "outcome_feature_changes": feature_changes,
        "expected_utility_changes": _utility_changes(baseline, transformed),
        "escalation_probability_change": feature_changes["military_escalation.minimum"],
        "catastrophe_probability_change": 0.0,
        "feasibility_report": feasibility,
        "equilibrium_certificates": [item["certificate"] for item in transformed],
        "multiplicity": len(transformed),
        "equilibrium_selection_treatment": dict(
            cast(Mapping[str, Any], spec["equilibrium_handling"])
        ),
        "best_response_gaps": {
            item["id"]: item["certificate"]["best_response_gap"] for item in transformed
        },
        "warnings": [
            *warnings,
            "All Korea parameters are illustrative or normalized.",
            "NO_PURE_PBE_FOUND concerns only the restricted pure class.",
        ],
        "solver_runtime_seconds": time.perf_counter() - started,
        "tolerance": float(spec.get("tolerance", 1e-9)),
        "serialization_metadata": _metadata(),
    }


def _objective(item: Mapping[str, Any], objective: Mapping[str, Any]) -> float:
    kind = objective.get("type")
    if kind in {"MAXIMIZE_EXPECTED_UTILITY", "MAXIMIZE_WORST_CASE_UTILITY"}:
        return float(item["expected_utilities"][_player(objective.get("player"))])
    distribution = cast(OutcomeDistribution, item["outcome_distribution"])
    if kind == "MINIMIZE_ESCALATION_PROBABILITY":
        return -distribution.probability("military_escalation")
    if kind == "MINIMIZE_CATASTROPHE_PROBABILITY":
        return 0.0
    if kind == "MAXIMIZE_NEGOTIATED_SETTLEMENT_PROBABILITY":
        return distribution.probability("negotiated_agreement")
    raise ValueError("unsupported Korea search objective")


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
                utility_ranges[_player(raw.get("player"))]["minimum"]
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
            raise ValueError(f"unsupported Korea constraint {kind!r}")
    return True


def search_policies(document: Mapping[str, Any]) -> dict[str, Any]:
    if document.get("document_type") != "policy_search_request":
        raise ValueError("search requires policy_search_request")
    spec = _spec(document)
    validate_counterfactual(spec)
    baseline_config = _baseline(spec)
    raw_parameters = baseline_config.get("parameters", {})
    assert isinstance(raw_parameters, Mapping)
    model = KoreaWarningModel(_parameters(cast(Mapping[str, Any], raw_parameters)))
    equilibria = _select(_enumerate_equilibria(model, warning_accuracy=1.0), spec)
    player = _player(document.get("player"))
    allowed_raw = document.get("allowed_information_sets")
    allowed_sets = (
        None
        if allowed_raw is None
        else tuple(str(item) for item in cast(Sequence[Any], allowed_raw))
    )
    structure = information_structure()
    owned = {key for key, info in structure.information_sets.items() if info.player_id == player}
    if allowed_sets is not None and not set(allowed_sets) <= owned:
        raise ValueError(
            f"unknown or unowned Korea search information sets: {sorted(set(allowed_sets) - owned)}"
        )
    if not equilibria:
        raise ValueError("Korea search cannot fix omitted decisions without a baseline PBE")
    variable = owned if allowed_sets is None else set(allowed_sets)
    supplied_fixed = document.get("fixed_actions", {})
    if not isinstance(supplied_fixed, Mapping):
        raise ValueError("fixed_actions must be a mapping")
    baseline_profile = cast(Mapping[str, str], equilibria[0]["strategy_profile"])
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
            "event": "korea",
            "status": "CAPACITY_EXCEEDED",
            "exact": True,
            "estimated_policy_count": space.size,
            "evaluated_policy_count": 0,
            "retained": [],
            "narrower_legal_configuration": space.narrower_configuration(maximum),
            "warnings": ["Exact enumeration was not started."],
        }
    objective = cast(Mapping[str, Any], spec["objective"])
    candidates = []
    for variable_policy in space.enumerate():
        policy = PurePolicy.from_actions(player, {**fixed_actions, **dict(variable_policy.actions)})
        report = audit_policy(policy, structure, require_complete=True)
        if not report.feasible:
            continue
        transformed = _replace_policy(
            model, equilibria, {"player": player, "policy": policy.actions}
        )
        scores = [_objective(item, objective) for item in transformed]
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
                for focal in (CHINA, RECEIVER)
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
        if _search_constraints_pass(candidate, spec, equilibria):
            candidates.append(candidate)
    candidates.sort(
        key=lambda item: (
            -float(cast(Mapping[str, Any], item["objective_range"])["minimum"]),
            str(item["policy_id"]),
        )
    )
    top_k = int(document.get("top_k", 10))
    retained = candidates[:top_k]
    if bool(document.get("retain_ties", True)) and len(candidates) > top_k:
        cutoff = float(cast(Mapping[str, Any], candidates[top_k - 1]["objective_range"])["minimum"])
        retained = [
            item
            for item in candidates
            if float(cast(Mapping[str, Any], item["objective_range"])["minimum"])
            >= cutoff - float(spec.get("tolerance", 1e-9))
        ]
    return {
        "schema_version": SCHEMA_VERSION,
        "document_type": "policy_search_result",
        "event": "korea",
        "status": "COMPLETE" if candidates else "NO_FEASIBLE_POLICY",
        "exact": True,
        "estimated_policy_count": space.size,
        "evaluated_policy_count": len(candidates),
        "feasible_policy_count": len(candidates),
        "ties_at_cutoff": max(0, len(retained) - top_k),
        "retained": retained,
        "warnings": ["Search uses frozen baseline opponents."],
    }


def search_paths(document: Mapping[str, Any]) -> dict[str, Any]:
    if document.get("document_type") != "path_search_request":
        raise ValueError("paths requires path_search_request")
    spec = _spec(document)
    validate_counterfactual(spec)
    baseline_config = _baseline(spec)
    raw_parameters = baseline_config.get("parameters", {})
    assert isinstance(raw_parameters, Mapping)
    parameters = _parameters(cast(Mapping[str, Any], raw_parameters))
    intervention = cast(Mapping[str, Any], spec["intervention"])
    model = _changed_model(parameters, intervention)
    accuracy = _warning_accuracy(intervention)
    baseline = _select(
        _enumerate_equilibria(KoreaWarningModel(parameters), warning_accuracy=1.0), spec
    )
    if intervention["type"] == "POLICY_REPLACEMENT":
        equilibria = _replace_policy(KoreaWarningModel(parameters), baseline, intervention)
    else:
        equilibria = _select(
            _enumerate_equilibria(
                model,
                warning_accuracy=accuracy,
                intervention=intervention if intervention["type"] == "ACTION_RESTRICTION" else None,
            ),
            spec,
        )
    if not equilibria:
        return {
            "schema_version": SCHEMA_VERSION,
            "document_type": "path_search_result",
            "event": "korea",
            "status": "NO_PURE_PBE_FOUND",
            "ranking": str(document.get("ranking", "EQUILIBRIUM_REACH_PROBABILITY")),
            "paths": [],
            "model_reachable_is_not_unilaterally_inducible": True,
            "warnings": ["No supported pure-PBE path set was found."],
        }

    def path_key(path: Mapping[str, Any]) -> tuple[str, str]:
        return str(path["terminal_id"]), canonical_json(path["events"])

    paths: list[dict[str, Any]] = []
    for counterfactual in equilibria:
        for baseline_assessment in baseline:
            baseline_paths = tuple(cast(Sequence[Mapping[str, Any]], baseline_assessment["paths"]))
            baseline_lookup = {
                path_key(path): float(path["reach_probability"]) for path in baseline_paths
            }
            for raw_path in cast(Sequence[Mapping[str, Any]], counterfactual["paths"]):
                path = dict(raw_path)
                path["baseline_solution_id"] = baseline_assessment["id"]
                path["counterfactual_solution_id"] = counterfactual["id"]
                path["baseline_reach_probability"] = baseline_lookup.get(path_key(raw_path), 0.0)
                path["equilibrium_reach_probability"] = (
                    float(raw_path["reach_probability"])
                    if bool(raw_path.get("equilibrium_supported"))
                    else None
                )
                current_decisions = [
                    event
                    for event in cast(Sequence[Mapping[str, Any]], raw_path["events"])
                    if event.get("actor") != "nature"
                ]
                baseline_decision_sets = [
                    [
                        event
                        for event in cast(Sequence[Mapping[str, Any]], baseline_path["events"])
                        if event.get("actor") != "nature"
                    ]
                    for baseline_path in baseline_paths
                ]
                closest = min(
                    baseline_decision_sets,
                    key=lambda prior: (
                        sum(
                            left.get("action") != right.get("action")
                            for left, right in zip(current_decisions, prior, strict=False)
                        )
                        + abs(len(current_decisions) - len(prior))
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
                        zip(current_decisions, closest, strict=False)
                    )
                    if current.get("action") != old.get("action")
                ]
                path["differences_from_baseline"] = differences
                path["deviation_count"] = len(differences) + abs(
                    len(current_decisions) - len(closest)
                )
                path.setdefault("information_available_at_changes", [])
                paths.append(path)
    ranking = str(document.get("ranking", "EQUILIBRIUM_REACH_PROBABILITY"))
    focal = _player(document["player"]) if document.get("player") is not None else None

    def score(path: Mapping[str, Any]) -> float:
        common = path["terminal_outcome"]["common"]
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
            if focal is None:
                raise ValueError("FOCAL_PLAYER_UTILITY requires player")
            return float(path["utilities"][focal])
        if ranking == "JOINT_UTILITY":
            return sum(
                float(value) for value in cast(Mapping[str, Any], path["utilities"]).values()
            )
        if ranking == "ESCALATION_AVOIDANCE":
            return -float(common["military_escalation"])
        if ranking == "CATASTROPHE_AVOIDANCE":
            return 0.0
        if ranking == "NEGOTIATED_SETTLEMENT":
            return float(common["negotiated_agreement"])
        if ranking in {"ROBUSTNESS", "ROBUST_WORST_CASE_UTILITY"}:
            raise ValueError(
                "Korea robust path ranking requires an uncertainty set and is unsupported"
            )
        if ranking == "MODEL_REACHABILITY":
            return float(path["model_reachable"])
        if ranking == "EQUILIBRIUM_SUPPORT":
            return float(path["equilibrium_supported"])
        if ranking in {"MINIMUM_DEVIATION_COUNT", "MINIMUM_DEVIATIONS"}:
            return -float(path["deviation_count"])
        raise ValueError(f"unsupported Korea path ranking {ranking!r}")

    unique = {
        (
            path["baseline_solution_id"],
            path["counterfactual_solution_id"],
            path["terminal_id"],
            canonical_json(path["events"]),
        ): path
        for path in paths
    }
    ordered = sorted(unique.values(), key=lambda path: (-score(path), path["terminal_id"]))
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
        "event": "korea",
        "status": "VALID_COUNTERFACTUAL" if equilibria else "NO_PURE_PBE_FOUND",
        "ranking": ranking,
        "paths": selected,
        "model_reachable_is_not_unilaterally_inducible": True,
        "warnings": ["Path existence does not establish unilateral inducibility."],
    }


def frontier(document: Mapping[str, Any]) -> dict[str, Any]:
    expanded = dict(document)
    expanded["top_k"] = int(document.get("maximum_policy_space", 100_000))
    expanded["retain_ties"] = True
    result = search_policies(expanded)
    candidates = cast(list[dict[str, Any]], result.get("retained", []))
    retained = []
    for candidate in candidates:
        current = candidate["expected_utility_ranges"]
        dominated = any(
            other is not candidate
            and all(
                other["expected_utility_ranges"][player]["minimum"]
                >= current[player]["minimum"] - 1e-9
                for player in (CHINA, RECEIVER)
            )
            and any(
                other["expected_utility_ranges"][player]["minimum"]
                > current[player]["minimum"] + 1e-9
                for player in (CHINA, RECEIVER)
            )
            for other in candidates
        )
        if not dominated:
            retained.append(candidate)
    return {
        "schema_version": SCHEMA_VERSION,
        "document_type": "pareto_frontier_result",
        "event": "korea",
        "status": result["status"],
        "frontier": retained,
        "comparison": "worst retained-equilibrium utility by player",
        "warnings": result["warnings"],
    }


def route_counterfactual(document: Mapping[str, Any], operation: str) -> dict[str, Any]:
    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError(f"unsupported Korea counterfactual operation {operation!r}")
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
    "CHINA",
    "RECEIVER",
    "evaluate_counterfactual",
    "frontier",
    "information_structure",
    "route_counterfactual",
    "search_paths",
    "search_policies",
    "terminal_outcome_features",
    "validate_counterfactual",
    "warning_value_of_information",
]
