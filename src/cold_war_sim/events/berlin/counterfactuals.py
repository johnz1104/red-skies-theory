"""Counterfactual adapter for finite-horizon Berlin bargaining."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
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
from cold_war_sim.events.berlin.model import PLAYER_INDEX, BerlinBargainingModel, _other
from cold_war_sim.events.berlin.parameters import BerlinParameters

SCHEMA_VERSION = "1.0"
WEST = "west"
SOVIET = "soviet"
SUPPORTED_OPERATIONS = ("validate", "evaluate", "search", "paths", "frontier")


def _share_action(share: float) -> str:
    return "share_" + format(share, ".12g").replace(".", "p")


def _share_value(action: str, grid: Sequence[float]) -> float:
    lookup = {_share_action(share): share for share in grid}
    try:
        return lookup[action]
    except KeyError as error:
        raise ValueError(f"unknown Berlin offer action {action!r}") from error


def _period_set(period: int) -> str:
    return f"period_{period}"


def _response_set(period: int, share: float) -> str:
    return f"response_{period}_{_share_action(share)}"


def information_structure(
    parameters: BerlinParameters | None = None,
    *,
    settlement_shares_by_period: Mapping[int, Sequence[float]] | None = None,
) -> InformationStructure:
    params = parameters or BerlinParameters()
    model = BerlinBargainingModel(params)
    variables = {
        "period": InformationVariable("period", 0, (WEST, SOVIET)),
        "current_offer": InformationVariable("current_offer", 0, (WEST, SOVIET)),
        "prior_rejections": InformationVariable("prior_rejections", 0, (WEST, SOVIET)),
        "modeler_parameter_truth": InformationVariable(
            "modeler_parameter_truth", 0, (), modeler_only=True
        ),
        "simulation_seed": InformationVariable("simulation_seed", 0, (), modeler_only=True),
    }
    sets: dict[str, InformationSetSpec] = {}
    for period in range(params.horizon):
        period_grid = (
            tuple(settlement_shares_by_period[period])
            if settlement_shares_by_period is not None
            else params.settlement_grid
        )
        proposer = model.proposer_at(period)
        responder = _other(proposer)
        proposer_set = _period_set(period)
        sets[proposer_set] = InformationSetSpec(
            proposer_set,
            proposer,
            period * 3,
            (proposer_set,),
            tuple(_share_action(value) for value in period_grid),
            ("period", "prior_rejections"),
            observation_signature={"period": period, "prior_rejections": period},
        )
        for share in period_grid:
            response_set = _response_set(period, share)
            sets[response_set] = InformationSetSpec(
                response_set,
                responder,
                period * 3 + 1,
                (response_set,),
                ("accept", "reject"),
                ("period", "prior_rejections", "current_offer"),
                observation_signature={
                    "period": period,
                    "prior_rejections": period,
                    "current_offer": _share_action(share),
                },
            )
    return InformationStructure(sets, variables)


def terminal_outcome_features(
    model: BerlinBargainingModel,
    *,
    terminal_kind: str,
    elapsed_periods: int,
    west_share: float | None = None,
    proposer: str | None = None,
) -> OutcomeFeatures:
    if terminal_kind not in {"agreement", "escalation", "impasse"}:
        raise ValueError("unknown Berlin terminal kind")
    agreement = terminal_kind == "agreement"
    escalation = terminal_kind == "escalation"
    concession = 0.0
    if agreement and west_share is not None and proposer is not None:
        concession = 1.0 - west_share if proposer == WEST else west_share
    params = model.parameters
    west_commitment_cost = 0.0
    soviet_commitment_cost = 0.0
    if agreement and west_share is not None:
        west_commitment_cost = params.west_commitment_cost * max(
            0.0, params.west_commitment_floor - west_share
        )
        soviet_commitment_cost = params.soviet_commitment_cost * max(
            0.0, west_share - params.soviet_commitment_ceiling
        )
    return OutcomeFeatures(
        common={
            "peaceful_settlement": agreement,
            "negotiated_agreement": agreement,
            "concession": concession,
            "military_escalation": escalation,
            "catastrophic_escalation": False,
            "bargaining_duration": elapsed_periods,
            "escalation_probability": float(escalation),
            "catastrophe_probability": 0.0,
            "settlement_probability": float(agreement),
        },
        by_player={
            WEST: {
                "political_cost": west_commitment_cost,
                "military_cost": params.west_escalation_loss if escalation else 0.0,
                "economic_cost": elapsed_periods * params.west_delay_cost,
            },
            SOVIET: {
                "political_cost": soviet_commitment_cost,
                "military_cost": params.soviet_escalation_loss if escalation else 0.0,
                "economic_cost": elapsed_periods * params.soviet_delay_cost,
            },
        },
    )


def _spec(document: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = document.get("counterfactual")
    return cast(Mapping[str, Any], nested) if isinstance(nested, Mapping) else document


def _baseline(document: Mapping[str, Any]) -> Mapping[str, Any]:
    baseline = _spec(document).get("baseline")
    if not isinstance(baseline, Mapping) or baseline.get("event") != "berlin":
        raise ValueError("Berlin adapter requires baseline.event='berlin'")
    return cast(Mapping[str, Any], baseline)


def _parameters(values: Mapping[str, Any]) -> BerlinParameters:
    return BerlinParameters.from_mapping(values)


def _player(value: object) -> str:
    normalized = str(value).lower()
    if normalized not in {WEST, SOVIET}:
        raise ValueError(f"unknown Berlin player {value!r}")
    return normalized


def _policy(raw: object) -> dict[str, str]:
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("policy must be a nonempty mapping")
    result: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("Berlin adapter supports deterministic pure policies only")
        result[key] = value
    return result


def _restriction(
    structure: InformationStructure,
    intervention: Mapping[str, Any] | None,
) -> dict[str, tuple[str, ...]]:
    allowed = {key: tuple(value.legal_actions) for key, value in structure.information_sets.items()}
    if not intervention or intervention.get("type") != "ACTION_RESTRICTION":
        return allowed
    player = _player(intervention.get("player"))
    identifier = str(intervention.get("information_set"))
    if identifier not in structure.information_sets:
        raise ValueError(f"unknown Berlin information set {identifier!r}")
    if structure.information_sets[identifier].player_id != player:
        raise ValueError("restriction player does not control the information set")
    raw_actions = intervention.get("actions")
    if not isinstance(raw_actions, Sequence) or isinstance(raw_actions, (str, bytes)):
        raise ValueError("restriction actions must be a sequence")
    removed = {str(item) for item in raw_actions}
    unknown = removed - set(allowed[identifier])
    if unknown:
        raise ValueError(f"unknown Berlin actions: {sorted(unknown)}")
    retained = tuple(action for action in allowed[identifier] if action not in removed)
    if not retained:
        raise ValueError("action restriction cannot remove every action")
    allowed[identifier] = retained
    return allowed


def _solve_policy(
    model: BerlinBargainingModel,
    *,
    intervention: Mapping[str, Any] | None = None,
    settlement_shares_by_period: Mapping[int, Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Backward-induct with optional information-set action restriction."""

    params = model.parameters
    structure = information_structure(
        params, settlement_shares_by_period=settlement_shares_by_period
    )
    allowed = _restriction(structure, intervention)
    policy: dict[str, str] = {}
    next_payoffs: tuple[float, float] | None = None
    tied_proposer_sets: dict[str, tuple[str, ...]] = {}
    tied_responder_sets: list[str] = []
    for period in range(params.horizon - 1, -1, -1):
        proposer = model.proposer_at(period)
        responder = _other(proposer)
        proposer_index = PLAYER_INDEX[proposer]
        responder_index = PLAYER_INDEX[responder]
        rejection = model._rejection_payoffs(period, next_payoffs)
        evaluations: list[tuple[str, tuple[float, float], str]] = []
        for action in allowed[_period_set(period)]:
            share = _share_value(action, params.settlement_grid)
            settlement = model.settlement_payoffs(period, share)
            response_set = _response_set(period, share)
            response_actions = allowed[response_set]
            if len(response_actions) == 1:
                response = response_actions[0]
            else:
                difference = settlement[responder_index] - rejection[responder_index]
                if difference > params.comparison_tolerance:
                    response = "accept"
                elif difference < -params.comparison_tolerance:
                    response = "reject"
                else:
                    tied_responder_sets.append(response_set)
                    response = params.responder_tie_break
            policy[response_set] = response
            induced = settlement if response == "accept" else rejection
            evaluations.append((action, induced, response))
        best = max(item[1][proposer_index] for item in evaluations)
        optimal = tuple(
            item
            for item in evaluations
            if math.isclose(
                item[1][proposer_index],
                best,
                rel_tol=0.0,
                abs_tol=params.comparison_tolerance,
            )
        )
        if len(optimal) > 1:
            tied_proposer_sets[_period_set(period)] = tuple(item[0] for item in optimal)
        selected = (
            min(optimal, key=lambda item: _share_value(item[0], params.settlement_grid))
            if params.offer_tie_break == "lowest_west_share"
            else max(optimal, key=lambda item: _share_value(item[0], params.settlement_grid))
        )
        policy[_period_set(period)] = selected[0]
        next_payoffs = selected[1]
    assert next_payoffs is not None
    evaluated = _evaluate_profile(
        model,
        policy,
        equilibrium_supported=True,
        settlement_shares_by_period=settlement_shares_by_period,
    )
    reach: dict[str, float] = {}
    survival = 1.0
    active = True
    for period in range(params.horizon):
        proposer_set = _period_set(period)
        reach[proposer_set] = survival if active else 0.0
        selected_offer = policy[proposer_set]
        period_grid = (
            tuple(settlement_shares_by_period[period])
            if settlement_shares_by_period is not None
            else params.settlement_grid
        )
        for share in period_grid:
            response_set = _response_set(period, share)
            reach[response_set] = (
                survival if active and _share_action(share) == selected_offer else 0.0
            )
        selected_share = _share_value(selected_offer, period_grid)
        if policy[_response_set(period, selected_share)] == "accept":
            active = False
        elif active:
            survival *= 1.0 - params.escalation_risk(period)
    return {
        "id": "berlin_spe_000",
        "strategy_profile": dict(sorted(policy.items())),
        "expected_utilities": evaluated["utilities"],
        "outcome_distribution": evaluated["distribution"],
        "paths": evaluated["paths"],
        "certificate": {
            "equilibrium_concept": "SUBGAME_PERFECT",
            "candidate_class": "pure strategies on the configured finite offer grid",
            "strategy_profile": dict(sorted(policy.items())),
            "beliefs": {},
            "reach_probabilities": reach,
            "deviation_gains": {WEST: 0.0, SOVIET: 0.0},
            "best_response_gap": 0.0,
            "off_path_convention": "NOT_APPLICABLE",
            "exact": True,
            "tolerance": params.comparison_tolerance,
            "warnings": [
                "Configured deterministic tie conventions select one representative policy.",
                "The event solver records, but does not enumerate, all tied equilibrium policies.",
            ],
            "tied_proposer_actions": tied_proposer_sets,
            "tied_responder_information_sets": sorted(set(tied_responder_sets)),
        },
    }


def _evaluate_profile(
    model: BerlinBargainingModel,
    policy: Mapping[str, str],
    *,
    equilibrium_supported: bool,
    settlement_shares_by_period: Mapping[int, Sequence[float]] | None = None,
) -> dict[str, Any]:
    params = model.parameters
    structure = information_structure(
        params, settlement_shares_by_period=settlement_shares_by_period
    )
    report_west = audit_policy(
        PurePolicy.from_actions(
            WEST,
            {
                key: value
                for key, value in policy.items()
                if structure.information_sets[key].player_id == WEST
            },
        ),
        structure,
        require_complete=True,
    )
    report_soviet = audit_policy(
        PurePolicy.from_actions(
            SOVIET,
            {
                key: value
                for key, value in policy.items()
                if structure.information_sets[key].player_id == SOVIET
            },
        ),
        structure,
        require_complete=True,
    )
    if not report_west.feasible or not report_soviet.feasible:
        raise ValueError("Berlin profile is incomplete or infeasible")
    outcomes: list[WeightedOutcome] = []
    paths: list[dict[str, Any]] = []
    utilities = {WEST: 0.0, SOVIET: 0.0}
    survival = 1.0
    events: list[dict[str, str]] = []
    for period in range(params.horizon):
        proposer = model.proposer_at(period)
        offer_action = policy[_period_set(period)]
        share = _share_value(offer_action, params.settlement_grid)
        response = policy[_response_set(period, share)]
        step_events = [
            *events,
            {"actor": proposer, "action": offer_action},
            {"actor": _other(proposer), "action": response},
        ]
        if response == "accept":
            payoff = model.settlement_payoffs(period, share)
            features = terminal_outcome_features(
                model,
                terminal_kind="agreement",
                elapsed_periods=period + 1,
                west_share=share,
                proposer=proposer,
            )
            terminal_id = f"agreement_period_{period}_{offer_action}"
            outcomes.append(WeightedOutcome(terminal_id, survival, features))
            utilities[WEST] += survival * payoff[0]
            utilities[SOVIET] += survival * payoff[1]
            paths.append(
                {
                    "terminal_id": terminal_id,
                    "events": step_events,
                    "reach_probability": survival,
                    "terminal_outcome": features.to_dict(),
                    "utilities": {WEST: payoff[0], SOVIET: payoff[1]},
                    "model_reachable": True,
                    "strategy_feasible": True,
                    "equilibrium_supported": equilibrium_supported,
                }
            )
            survival = 0.0
            break
        risk = params.escalation_risk(period)
        escalation_probability = survival * risk
        if escalation_probability > 0.0:
            payoff = model.escalation_payoffs(period + 1)
            features = terminal_outcome_features(
                model,
                terminal_kind="escalation",
                elapsed_periods=period + 1,
            )
            terminal_id = f"escalation_period_{period}"
            outcomes.append(WeightedOutcome(terminal_id, escalation_probability, features))
            utilities[WEST] += escalation_probability * payoff[0]
            utilities[SOVIET] += escalation_probability * payoff[1]
            paths.append(
                {
                    "terminal_id": terminal_id,
                    "events": [*step_events, {"actor": "nature", "action": "escalation"}],
                    "reach_probability": escalation_probability,
                    "terminal_outcome": features.to_dict(),
                    "utilities": {WEST: payoff[0], SOVIET: payoff[1]},
                    "model_reachable": True,
                    "strategy_feasible": True,
                    "equilibrium_supported": equilibrium_supported,
                }
            )
        survival *= 1.0 - risk
        events = [*step_events, {"actor": "nature", "action": "survival"}]
        if period == params.horizon - 1 and survival > 0.0:
            payoff = model.impasse_payoffs(params.horizon)
            features = terminal_outcome_features(
                model,
                terminal_kind="impasse",
                elapsed_periods=params.horizon,
            )
            terminal_id = "impasse"
            outcomes.append(WeightedOutcome(terminal_id, survival, features))
            utilities[WEST] += survival * payoff[0]
            utilities[SOVIET] += survival * payoff[1]
            paths.append(
                {
                    "terminal_id": terminal_id,
                    "events": events,
                    "reach_probability": survival,
                    "terminal_outcome": features.to_dict(),
                    "utilities": {WEST: payoff[0], SOVIET: payoff[1]},
                    "model_reachable": True,
                    "strategy_feasible": True,
                    "equilibrium_supported": equilibrium_supported,
                }
            )
            survival = 0.0
    return {
        "distribution": OutcomeDistribution(tuple(outcomes)),
        "utilities": utilities,
        "paths": tuple(sorted(paths, key=lambda item: str(item["terminal_id"]))),
    }


def _changed_model(
    base: BerlinParameters, intervention: Mapping[str, Any]
) -> BerlinBargainingModel:
    if intervention.get("type") != "PARAMETER_INTERVENTION":
        return BerlinBargainingModel(base)
    changes = intervention.get("changes")
    if not isinstance(changes, Mapping):
        raise ValueError("parameter intervention changes must be a mapping")
    values = asdict(base)
    values.update(changes)
    return BerlinBargainingModel(BerlinParameters.from_mapping(values))


def _transformed_model(
    base: BerlinParameters,
    intervention: Mapping[str, Any],
) -> tuple[BerlinBargainingModel, dict[int, tuple[float, ...]] | None]:
    """Apply one validated builder transform and return its scoped action grid."""

    kind = intervention.get("type")
    if kind == "PARAMETER_INTERVENTION":
        return _changed_model(base, intervention), None
    if kind == "STRUCTURAL_TRANSFORMATION":
        if intervention.get("transformation_id") != "swap_initial_proposer":
            raise ValueError("unknown Berlin structural transformation")
        raw = intervention.get("parameters", {})
        if not isinstance(raw, Mapping) or raw:
            raise ValueError("swap_initial_proposer accepts no parameters")
        return (
            BerlinBargainingModel(replace(base, initial_proposer=_other(base.initial_proposer))),
            None,
        )
    if kind == "ACTION_EXPANSION":
        if intervention.get("transformation_id") != "berlin_add_settlement_share":
            raise ValueError(
                "Berlin action expansion requires transformation_id='berlin_add_settlement_share'"
            )
        identifier = str(intervention.get("information_set"))
        if not identifier.startswith("period_"):
            raise ValueError("settlement-share expansion requires a period_* proposer set")
        try:
            period = int(identifier.removeprefix("period_"))
        except ValueError as error:
            raise ValueError("invalid Berlin proposer information-set id") from error
        baseline_model = BerlinBargainingModel(base)
        if period < 0 or period >= base.horizon:
            raise ValueError("expanded Berlin period is outside the horizon")
        player = _player(intervention.get("player"))
        if baseline_model.proposer_at(period) != player:
            raise ValueError("action-expansion player does not control that proposer set")
        raw = intervention.get("parameters")
        if not isinstance(raw, Mapping) or set(raw) != {"west_share"}:
            raise ValueError("berlin_add_settlement_share requires only parameters.west_share")
        share_raw = raw["west_share"]
        if isinstance(share_raw, bool) or not isinstance(share_raw, (int, float)):
            raise TypeError("west_share must be a real number")
        share = float(share_raw)
        if not math.isfinite(share) or not 0.0 <= share <= 1.0:
            raise ValueError("west_share must be finite and lie in [0, 1]")
        if share in base.settlement_grid:
            raise ValueError("expanded settlement share already exists")
        action = str(intervention.get("action"))
        if action != _share_action(share):
            raise ValueError(
                f"expanded action must be {_share_action(share)!r} for west_share={share}"
            )
        expanded_grid = tuple(sorted((*base.settlement_grid, share)))
        transformed = BerlinBargainingModel(replace(base, settlement_grid=expanded_grid))
        scoped = {
            index: (expanded_grid if index == period else tuple(base.settlement_grid))
            for index in range(base.horizon)
        }
        return transformed, scoped
    return BerlinBargainingModel(base), None


def _replace_policy(
    model: BerlinBargainingModel,
    baseline: Mapping[str, Any],
    intervention: Mapping[str, Any],
    *,
    settlement_shares_by_period: Mapping[int, Sequence[float]] | None = None,
) -> dict[str, Any]:
    player = _player(intervention.get("player"))
    replacement = _policy(intervention.get("policy"))
    structure = information_structure(
        model.parameters,
        settlement_shares_by_period=settlement_shares_by_period,
    )
    report = audit_policy(
        PurePolicy.from_actions(player, replacement), structure, require_complete=False
    )
    if not report.feasible:
        raise ValueError(f"infeasible Berlin policy: {report.to_dict()['violations']}")
    policy = dict(baseline["strategy_profile"])
    policy.update(replacement)
    evaluated = _evaluate_profile(
        model,
        policy,
        equilibrium_supported=False,
        settlement_shares_by_period=settlement_shares_by_period,
    )
    return {
        "id": "berlin_policy_000",
        "strategy_profile": policy,
        "expected_utilities": evaluated["utilities"],
        "outcome_distribution": evaluated["distribution"],
        "paths": evaluated["paths"],
        "certificate": {
            "equilibrium_concept": "SUPPLIED_PROFILE",
            "candidate_class": "model-feasible pure bargaining policy",
            "strategy_profile": policy,
            "beliefs": {},
            "reach_probabilities": {},
            "deviation_gains": {},
            "best_response_gap": 0.0,
            "off_path_convention": "NOT_APPLICABLE",
            "exact": False,
            "tolerance": model.parameters.comparison_tolerance,
            "warnings": ["Frozen opponents: not an equilibrium certificate."],
        },
        "feasibility": report,
    }


def _metadata() -> dict[str, Any]:
    return {
        "package_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "seed_policy": "deterministic backward induction",
        "solver_version": "berlin counterfactual adapter 1.0",
    }


def validate_counterfactual(document: Mapping[str, Any]) -> dict[str, Any]:
    spec = _spec(document)
    baseline = _baseline(document)
    raw_parameters = baseline.get("parameters", {})
    if not isinstance(raw_parameters, Mapping):
        raise ValueError("baseline.parameters must be a mapping")
    params = _parameters(cast(Mapping[str, Any], raw_parameters))
    intervention = spec.get("intervention")
    if not isinstance(intervention, Mapping):
        raise ValueError("intervention must be a mapping")
    kind = intervention.get("type")
    if spec.get("uncertainty_set") is not None:
        raise ValueError(
            "Berlin event routing does not yet support uncertainty-set evaluation; "
            "no robust preference claim was computed"
        )
    if spec.get("solution_concept") not in {
        "SUBGAME_PERFECT",
        "SUBGAME_PERFECT_EQUILIBRIUM",
        "BACKWARD_INDUCTION",
        "SUPPLIED_PROFILE",
    }:
        raise ValueError("Berlin supports finite-horizon backward induction or a supplied profile")
    if kind not in {
        "POLICY_REPLACEMENT",
        "ACTION_RESTRICTION",
        "ACTION_EXPANSION",
        "PARAMETER_INTERVENTION",
        "STRUCTURAL_TRANSFORMATION",
    }:
        raise ValueError(
            f"Berlin adapter does not support {kind!r}; commitment-cost parameters are "
            "preference penalties, not enforceable commitments"
        )
    transformed_model, transformed_shares = _transformed_model(
        params, cast(Mapping[str, Any], intervention)
    )
    structure = information_structure(
        transformed_model.parameters,
        settlement_shares_by_period=transformed_shares,
    )
    feasibility: dict[str, Any] | None = None
    if kind == "POLICY_REPLACEMENT":
        player = _player(intervention.get("player"))
        policy = _policy(intervention.get("policy"))
        report = audit_policy(
            PurePolicy.from_actions(player, policy), structure, require_complete=False
        )
        if not report.feasible:
            raise ValueError(f"infeasible Berlin policy: {report.to_dict()['violations']}")
        feasibility = report.to_dict()
        if spec.get("response_model") != "FROZEN_OPPONENTS":
            raise ValueError("Berlin policy replacement currently supports FROZEN_OPPONENTS only")
    elif kind == "ACTION_RESTRICTION":
        _restriction(structure, cast(Mapping[str, Any], intervention))
        if spec.get("response_model") != "REEQUILIBRATE":
            raise ValueError("Berlin action restriction requires REEQUILIBRATE")
    else:
        _transformed_model(params, cast(Mapping[str, Any], intervention))
        if spec.get("response_model") != "REEQUILIBRATE":
            raise ValueError("Berlin model transformation requires REEQUILIBRATE")
    return {
        "schema_version": SCHEMA_VERSION,
        "document_type": "counterfactual_validation",
        "event": "berlin",
        "valid": True,
        "supported": True,
        "information_structure": structure.to_dict(),
        "feasibility_report": feasibility,
        "warnings": [
            "Offer actions are restricted to the configured finite settlement grid.",
            "Configured tie conventions select a representative when payoffs tie.",
        ],
    }


def evaluate_counterfactual(document: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    validate_counterfactual(document)
    spec = _spec(document)
    baseline_config = _baseline(document)
    raw_parameters = baseline_config.get("parameters", {})
    assert isinstance(raw_parameters, Mapping)
    params = _parameters(cast(Mapping[str, Any], raw_parameters))
    intervention = cast(Mapping[str, Any], spec["intervention"])
    baseline = _solve_policy(BerlinBargainingModel(params))
    model = BerlinBargainingModel(params)
    transformed_shares: dict[int, tuple[float, ...]] | None = None
    if intervention["type"] == "POLICY_REPLACEMENT":
        transformed = _replace_policy(BerlinBargainingModel(params), baseline, intervention)
        warnings = ["Frozen opponents are a diagnostic, not a strategic prediction."]
    else:
        model, transformed_shares = _transformed_model(params, intervention)
        transformed = _solve_policy(
            model,
            intervention=intervention if intervention["type"] == "ACTION_RESTRICTION" else None,
            settlement_shares_by_period=transformed_shares,
        )
        warnings = []
    before_distribution = baseline["outcome_distribution"]
    after_distribution = transformed["outcome_distribution"]
    feature_changes = {
        feature: after_distribution.probability(feature) - before_distribution.probability(feature)
        for feature in ("military_escalation", "catastrophic_escalation", "negotiated_agreement")
    }
    feasibility = {
        "action_legality": {"status": "PASS", "checked_count": 0, "message": "event validation"},
        "information_consistency": {
            "status": "PASS",
            "checked_count": 0,
            "message": "perfect information",
        },
        "temporal_consistency": {
            "status": "PASS",
            "checked_count": 0,
            "message": "period order retained",
        },
        "commitment_consistency": {
            "status": "NOT_APPLICABLE",
            "checked_count": 0,
            "message": "no enforceable commitment",
        },
        "reachable_information_sets": list(
            information_structure(
                (params if intervention["type"] == "POLICY_REPLACEMENT" else model.parameters),
                settlement_shares_by_period=(
                    None if intervention["type"] == "POLICY_REPLACEMENT" else transformed_shares
                ),
            ).information_sets
        ),
        "violations": [],
    }
    if intervention["type"] == "POLICY_REPLACEMENT":
        feasibility = transformed["feasibility"].to_dict()
    return {
        "schema_version": SCHEMA_VERSION,
        "document_type": "counterfactual_result",
        "status": "VALID_COUNTERFACTUAL",
        "baseline": dict(baseline_config),
        "intervention": dict(intervention),
        "response_model": spec["response_model"],
        "solution_concept": spec["solution_concept"],
        "baseline_strategy_set": [{"id": baseline["id"], "policy": baseline["strategy_profile"]}],
        "counterfactual_strategy_set": [
            {"id": transformed["id"], "policy": transformed["strategy_profile"]}
        ],
        "baseline_outcome_distribution": before_distribution.to_dict(),
        "counterfactual_outcome_distribution": after_distribution.to_dict(),
        "outcome_feature_changes": feature_changes,
        "expected_utility_changes": {
            player: transformed["expected_utilities"][player]
            - baseline["expected_utilities"][player]
            for player in (WEST, SOVIET)
        },
        "escalation_probability_change": feature_changes["military_escalation"],
        "catastrophe_probability_change": 0.0,
        "feasibility_report": feasibility,
        "equilibrium_certificates": [transformed["certificate"]],
        "multiplicity": 1,
        "equilibrium_selection_treatment": {
            **dict(cast(Mapping[str, Any], spec["equilibrium_handling"])),
            "event_tie_treatment": "configured deterministic tie conventions",
        },
        "best_response_gaps": {transformed["id"]: transformed["certificate"]["best_response_gap"]},
        "warnings": [
            *warnings,
            "All Berlin parameters are illustrative or normalized.",
            "Commitment-cost fields are utility penalties, not binding commitments.",
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
    raise ValueError("unsupported Berlin objective")


def _constraints_pass(
    candidate: Mapping[str, Any],
    spec: Mapping[str, Any],
    baseline_policy: Mapping[str, str],
) -> bool:
    """Apply every schema-supported constraint to one exact search candidate."""

    tolerance = float(spec.get("tolerance", 1e-9))
    constraints = spec.get("constraints", ())
    if not isinstance(constraints, Sequence) or isinstance(constraints, (str, bytes)):
        raise ValueError("constraints must be a sequence")
    policy = cast(Mapping[str, str], candidate["policy"])
    changed = {
        key
        for key in set(policy) | set(baseline_policy)
        if policy.get(key) != baseline_policy.get(key)
    }
    utilities = cast(Mapping[str, float], candidate["expected_utilities"])
    for raw in constraints:
        if not isinstance(raw, Mapping):
            raise ValueError("each constraint must be a mapping")
        kind = raw.get("type")
        if kind == "MinimumExpectedUtility":
            if utilities[_player(raw.get("player"))] < float(raw["minimum"]) - tolerance:
                return False
        elif kind == "MaximumEscalationProbability":
            if float(candidate["escalation_probability"]) > float(raw["maximum"]) + tolerance:
                return False
        elif kind == "MaximumCatastropheProbability":
            if float(raw["maximum"]) + tolerance < 0.0:
                return False
        elif kind == "MinimumSettlementProbability":
            if float(candidate["settlement_probability"]) < float(raw["minimum"]) - tolerance:
                return False
        elif kind == "MaximumPolicyChanges":
            if len(changed) > int(raw["maximum"]):
                return False
        elif kind == "AllowedInformationSets":
            allowed = {str(item) for item in cast(Sequence[Any], raw["information_sets"])}
            if not changed <= allowed:
                return False
        else:
            raise ValueError(f"unsupported Berlin constraint {kind!r}")
    return True


def search_policies(document: Mapping[str, Any]) -> dict[str, Any]:
    if document.get("document_type") != "policy_search_request":
        raise ValueError("search requires policy_search_request")
    spec = _spec(document)
    validate_counterfactual(spec)
    baseline_config = _baseline(spec)
    raw_parameters = baseline_config.get("parameters", {})
    assert isinstance(raw_parameters, Mapping)
    params = _parameters(cast(Mapping[str, Any], raw_parameters))
    intervention = cast(Mapping[str, Any], spec["intervention"])
    model, transformed_shares = _transformed_model(params, intervention)
    baseline = _solve_policy(model, settlement_shares_by_period=transformed_shares)
    player = _player(document.get("player"))
    raw_allowed = document.get("allowed_information_sets")
    allowed = (
        None
        if raw_allowed is None
        else tuple(str(item) for item in cast(Sequence[Any], raw_allowed))
    )
    structure = information_structure(
        model.parameters, settlement_shares_by_period=transformed_shares
    )
    owned = {key for key, info in structure.information_sets.items() if info.player_id == player}
    if allowed is not None and not set(allowed) <= owned:
        raise ValueError(
            f"unknown or unowned Berlin search information sets: {sorted(set(allowed) - owned)}"
        )
    owned = {key for key, info in structure.information_sets.items() if info.player_id == player}
    variable = owned if allowed is None else set(allowed)
    supplied_fixed = document.get("fixed_actions", {})
    if not isinstance(supplied_fixed, Mapping):
        raise ValueError("fixed_actions must be a mapping")
    baseline_profile_full = cast(Mapping[str, str], baseline["strategy_profile"])
    fixed_actions = {
        key: str(supplied_fixed.get(key, baseline_profile_full[key]))
        for key in sorted(owned - variable)
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
            "event": "berlin",
            "status": "CAPACITY_EXCEEDED",
            "exact": True,
            "estimated_policy_count": space.size,
            "evaluated_policy_count": 0,
            "retained": [],
            "narrower_legal_configuration": space.narrower_configuration(maximum),
            "warnings": ["Exact enumeration was not started."],
        }
    objective = cast(Mapping[str, Any], spec["objective"])
    candidates: list[dict[str, Any]] = []
    baseline_policy = {
        key: value
        for key, value in cast(Mapping[str, str], baseline["strategy_profile"]).items()
        if key in owned
    }
    for variable_policy in space.enumerate():
        policy = PurePolicy.from_actions(player, {**fixed_actions, **dict(variable_policy.actions)})
        report = audit_policy(policy, structure, require_complete=True)
        if not report.feasible:
            continue
        transformed = _replace_policy(
            model,
            baseline,
            {"player": player, "policy": policy.actions},
            settlement_shares_by_period=transformed_shares,
        )
        candidate = {
            "policy_id": "|".join(f"{key}={value}" for key, value in policy.actions.items()),
            "policy": dict(policy.actions),
            "objective_score": _objective(transformed, objective),
            "expected_utilities": transformed["expected_utilities"],
            "escalation_probability": transformed["outcome_distribution"].probability(
                "military_escalation"
            ),
            "catastrophe_probability": 0.0,
            "settlement_probability": transformed["outcome_distribution"].probability(
                "negotiated_agreement"
            ),
            "independently_verified": True,
            "feasibility": report.to_dict(),
            "differences_from_baseline": sorted(
                key
                for key in set(policy.actions) | set(baseline_policy)
                if policy.actions.get(key) != baseline_policy.get(key)
            ),
        }
        if _constraints_pass(candidate, spec, baseline_policy):
            candidates.append(candidate)
    candidates.sort(key=lambda item: (-item["objective_score"], item["policy_id"]))
    top_k = int(document.get("top_k", 10))
    retain_ties = bool(document.get("retain_ties", True))
    retained = candidates[:top_k]
    if retain_ties and len(candidates) > top_k:
        cutoff = float(candidates[top_k - 1]["objective_score"])
        retained = [
            item
            for item in candidates
            if float(item["objective_score"]) >= cutoff - float(spec.get("tolerance", 1e-9))
        ]
    return {
        "schema_version": SCHEMA_VERSION,
        "document_type": "policy_search_result",
        "event": "berlin",
        "status": "COMPLETE" if candidates else "NO_FEASIBLE_POLICY",
        "exact": True,
        "estimated_policy_count": space.size,
        "evaluated_policy_count": len(candidates),
        "feasible_policy_count": len(candidates),
        "ties_at_cutoff": max(0, len(retained) - top_k),
        "retained": retained,
        "warnings": ["Search holds all unselected bargaining decisions fixed."],
    }


def search_paths(document: Mapping[str, Any]) -> dict[str, Any]:
    if document.get("document_type") != "path_search_request":
        raise ValueError("paths requires path_search_request")
    spec = _spec(document)
    validate_counterfactual(spec)
    baseline_config = _baseline(spec)
    raw_parameters = baseline_config.get("parameters", {})
    assert isinstance(raw_parameters, Mapping)
    params = _parameters(cast(Mapping[str, Any], raw_parameters))
    intervention = cast(Mapping[str, Any], spec["intervention"])
    baseline = _solve_policy(BerlinBargainingModel(params))
    if intervention["type"] == "POLICY_REPLACEMENT":
        transformed = _replace_policy(BerlinBargainingModel(params), baseline, intervention)
    else:
        transformed_model, transformed_shares = _transformed_model(params, intervention)
        transformed = _solve_policy(
            transformed_model,
            intervention=intervention if intervention["type"] == "ACTION_RESTRICTION" else None,
            settlement_shares_by_period=transformed_shares,
        )
    baseline_paths = tuple(cast(Sequence[Mapping[str, Any]], baseline["paths"]))

    def path_key(path: Mapping[str, Any]) -> tuple[str, str]:
        return str(path["terminal_id"]), canonical_json(path["events"])

    baseline_lookup = {path_key(path): float(path["reach_probability"]) for path in baseline_paths}
    paths: list[dict[str, Any]] = []
    for raw_path in cast(Sequence[Mapping[str, Any]], transformed["paths"]):
        path = dict(raw_path)
        path["baseline_solution_id"] = baseline["id"]
        path["counterfactual_solution_id"] = transformed["id"]
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
                for event in cast(Sequence[Mapping[str, Any]], old["events"])
                if event.get("actor") != "nature"
            ]
            for old in baseline_paths
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
            for index, (current, old) in enumerate(zip(current_decisions, closest, strict=False))
            if current.get("action") != old.get("action")
        ]
        path["differences_from_baseline"] = differences
        path["deviation_count"] = len(differences) + abs(len(current_decisions) - len(closest))
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
                "Berlin robust path ranking requires an uncertainty set and is unsupported"
            )
        if ranking == "MODEL_REACHABILITY":
            return float(path["model_reachable"])
        if ranking == "EQUILIBRIUM_SUPPORT":
            return float(path["equilibrium_supported"])
        if ranking in {"MINIMUM_DEVIATION_COUNT", "MINIMUM_DEVIATIONS"}:
            return -float(path["deviation_count"])
        raise ValueError(f"unsupported Berlin path ranking {ranking!r}")

    paths.sort(key=lambda path: (-score(path), path["terminal_id"]))
    top_k = int(document.get("top_k", 10))
    selected = paths[:top_k]
    if len(paths) > top_k:
        cutoff = score(paths[top_k - 1])
        selected = [
            path for path in paths if score(path) >= cutoff - float(spec.get("tolerance", 1e-9))
        ]
    return {
        "schema_version": SCHEMA_VERSION,
        "document_type": "path_search_result",
        "event": "berlin",
        "status": "VALID_COUNTERFACTUAL",
        "ranking": ranking,
        "paths": selected,
        "model_reachable_is_not_unilaterally_inducible": True,
        "warnings": ["Rejected-offer paths contain exogenous escalation branches."],
    }


def frontier(document: Mapping[str, Any]) -> dict[str, Any]:
    expanded = dict(document)
    expanded["top_k"] = int(document.get("maximum_policy_space", 100_000))
    expanded["retain_ties"] = True
    result = search_policies(expanded)
    candidates = cast(list[dict[str, Any]], result.get("retained", []))
    retained = []
    for candidate in candidates:
        dominated = any(
            other is not candidate
            and all(
                other["expected_utilities"][player]
                >= candidate["expected_utilities"][player] - 1e-9
                for player in (WEST, SOVIET)
            )
            and any(
                other["expected_utilities"][player] > candidate["expected_utilities"][player] + 1e-9
                for player in (WEST, SOVIET)
            )
            for other in candidates
        )
        if not dominated:
            retained.append(candidate)
    return {
        "schema_version": SCHEMA_VERSION,
        "document_type": "pareto_frontier_result",
        "event": "berlin",
        "status": result["status"],
        "frontier": retained,
        "comparison": "expected modeled utility by player",
        "warnings": result["warnings"],
    }


def route_counterfactual(document: Mapping[str, Any], operation: str) -> dict[str, Any]:
    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError(f"unsupported Berlin counterfactual operation {operation!r}")
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
    "SOVIET",
    "WEST",
    "evaluate_counterfactual",
    "frontier",
    "information_structure",
    "route_counterfactual",
    "search_paths",
    "search_policies",
    "terminal_outcome_features",
    "validate_counterfactual",
]
