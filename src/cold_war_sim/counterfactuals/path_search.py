"""Top-k terminal path queries with separate reachability/support flags."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from cold_war_sim.core.extensive_form import (
    BehavioralStrategy,
    ChanceNode,
    DecisionNode,
    ExtensiveFormGame,
    TerminalNode,
)
from cold_war_sim.core.types import (
    SerializableMixin,
    deep_frozen_json,
    frozen_mapping,
    validate_stable_id,
)

from .feasibility import FeasibilityReport
from .outcomes import OutcomeFeatures
from .responses import EquilibriumCertificate


class PathRanking(StrEnum):
    EQUILIBRIUM_REACH_PROBABILITY = "EQUILIBRIUM_REACH_PROBABILITY"
    BASELINE_REACH_PROBABILITY = "BASELINE_REACH_PROBABILITY"
    FOCAL_PLAYER_UTILITY = "FOCAL_PLAYER_UTILITY"
    JOINT_UTILITY = "JOINT_UTILITY"
    ESCALATION_AVOIDANCE = "ESCALATION_AVOIDANCE"
    CATASTROPHE_AVOIDANCE = "CATASTROPHE_AVOIDANCE"
    NEGOTIATED_SETTLEMENT = "NEGOTIATED_SETTLEMENT"
    MINIMUM_DEVIATIONS = "MINIMUM_DEVIATIONS"
    MINIMUM_DEVIATION_COUNT = "MINIMUM_DEVIATION_COUNT"
    MODEL_REACHABILITY = "MODEL_REACHABILITY"
    EQUILIBRIUM_SUPPORT = "EQUILIBRIUM_SUPPORT"
    ROBUSTNESS = "ROBUSTNESS"
    ROBUST_WORST_CASE_UTILITY = "ROBUST_WORST_CASE_UTILITY"


@dataclass(frozen=True)
class ActionEvent(SerializableMixin):
    node_id: str
    actor: str
    action: str
    information_set_id: str | None

    def __post_init__(self) -> None:
        validate_stable_id(self.node_id, field_name="path-event node id")
        validate_stable_id(self.actor, field_name="path-event actor id")
        validate_stable_id(self.action, field_name="path-event action id")
        if self.information_set_id is not None:
            validate_stable_id(
                self.information_set_id, field_name="path-event information-set id"
            )


@dataclass(frozen=True)
class DecisionDifference(SerializableMixin):
    information_set_id: str
    baseline_action: str | None
    path_action: str

    def __post_init__(self) -> None:
        validate_stable_id(
            self.information_set_id, field_name="difference information-set id"
        )
        if self.baseline_action is not None:
            validate_stable_id(self.baseline_action, field_name="baseline action id")
        validate_stable_id(self.path_action, field_name="path action id")
        if self.baseline_action == self.path_action:
            raise ValueError("a decision difference must change the baseline action")


@dataclass(frozen=True)
class InformationSnapshot(SerializableMixin):
    information_set_id: str
    observed_variables: Mapping[str, object]

    def __post_init__(self) -> None:
        validate_stable_id(
            self.information_set_id, field_name="snapshot information-set id"
        )
        observed = deep_frozen_json(self.observed_variables)
        if not isinstance(observed, Mapping):
            raise TypeError("observed_variables must be a mapping")
        object.__setattr__(self, "observed_variables", observed)


@dataclass(frozen=True)
class CounterfactualPath(SerializableMixin):
    terminal_id: str
    events: tuple[ActionEvent, ...]
    reach_probability: float
    terminal_outcome: OutcomeFeatures
    utilities: Mapping[str, float]
    differences_from_baseline: tuple[DecisionDifference, ...]
    information_available_at_changes: tuple[InformationSnapshot, ...]
    model_reachable: bool
    strategy_feasible: bool
    equilibrium_supported: bool
    baseline_reach_probability: float | None = None
    equilibrium_reach_probability: float | None = None
    robust_utilities: Mapping[str, tuple[float, ...]] | None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.terminal_id, field_name="terminal id")
        if any(not isinstance(value, bool) for value in (
            self.model_reachable,
            self.strategy_feasible,
            self.equilibrium_supported,
        )):
            raise TypeError("path support flags must be booleans")
        events = tuple(self.events)
        if not events:
            raise ValueError("a counterfactual path must contain at least one event")
        differences = tuple(self.differences_from_baseline)
        snapshots = tuple(self.information_available_at_changes)
        difference_ids = tuple(item.information_set_id for item in differences)
        if len(difference_ids) != len(set(difference_ids)):
            raise ValueError("path decision differences must have unique information sets")
        snapshot_ids = tuple(item.information_set_id for item in snapshots)
        if len(snapshot_ids) != len(set(snapshot_ids)):
            raise ValueError("information snapshots must have unique information sets")
        if not set(snapshot_ids) <= set(difference_ids):
            raise ValueError("information snapshots must correspond to decision differences")
        event_decisions = {
            (event.information_set_id, event.action)
            for event in events
            if event.information_set_id is not None
        }
        if any(
            (difference.information_set_id, difference.path_action)
            not in event_decisions
            for difference in differences
        ):
            raise ValueError("each decision difference must identify an event on the path")
        for name in (
            "reach_probability",
            "baseline_reach_probability",
            "equilibrium_reach_probability",
        ):
            raw = getattr(self, name)
            if raw is None:
                continue
            value = float(raw)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and lie in [0, 1]")
            object.__setattr__(self, name, value)
        converted_utilities = {}
        for player, raw in self.utilities.items():
            validate_stable_id(player, field_name="path-utility player id")
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError(f"utility for {player!r} must be finite")
            converted_utilities[player] = value
        object.__setattr__(self, "utilities", frozen_mapping(converted_utilities))
        if not converted_utilities:
            raise ValueError("path utilities must report at least one player")
        if self.strategy_feasible and not self.model_reachable:
            raise ValueError("a strategy-feasible path must be model-reachable")
        if self.equilibrium_supported and not self.strategy_feasible:
            raise ValueError("an equilibrium-supported path must be strategy-feasible")
        if self.robust_utilities is not None:
            if set(self.robust_utilities) != set(self.utilities):
                raise ValueError("robust utilities must cover the path utility players")
            lengths = {len(values) for values in self.robust_utilities.values()}
            if not lengths or 0 in lengths or len(lengths) != 1:
                raise ValueError(
                    "robust utility vectors must be nonempty and have equal length"
                )
            converted_robust = {}
            for player, values in self.robust_utilities.items():
                converted = tuple(float(value) for value in values)
                if not all(math.isfinite(value) for value in converted):
                    raise ValueError("robust utilities must be finite")
                converted_robust[player] = converted
            object.__setattr__(
                self, "robust_utilities", frozen_mapping(converted_robust)
            )
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "differences_from_baseline", differences)
        object.__setattr__(self, "information_available_at_changes", snapshots)


def enumerate_paths(
    game: ExtensiveFormGame,
    strategy: BehavioralStrategy,
    *,
    outcome_features: Mapping[str, OutcomeFeatures],
    utilities: Mapping[str, Mapping[str, float]] | None = None,
    baseline_policy: Mapping[str, str] | None = None,
    baseline_profile: BehavioralStrategy | None = None,
    information_snapshots: Mapping[str, InformationSnapshot] | None = None,
    equilibrium_profile: BehavioralStrategy | None = None,
    feasibility_report: FeasibilityReport | None = None,
    equilibrium_certificate: EquilibriumCertificate | None = None,
    robust_utilities: Mapping[str, Mapping[str, tuple[float, ...]]] | None = None,
    tolerance: float = 1e-12,
) -> tuple[CounterfactualPath, ...]:
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    game.validate_strategy(strategy)
    if equilibrium_profile is not None:
        game.validate_strategy(equilibrium_profile)
    if baseline_profile is not None:
        game.validate_strategy(baseline_profile)
    if equilibrium_profile is not None and equilibrium_certificate is None:
        raise ValueError(
            "an equilibrium profile requires an independently checked certificate"
        )
    if equilibrium_certificate is not None and equilibrium_profile is None:
        raise ValueError("an equilibrium certificate requires an equilibrium profile")
    certificate_supported = bool(
        equilibrium_certificate is not None
        and equilibrium_certificate.exact
        and equilibrium_certificate.best_response_gap <= equilibrium_certificate.tolerance
    )
    equilibrium_reach = (
        game.realization_probabilities(equilibrium_profile)
        if equilibrium_profile is not None
        else None
    )
    baseline_reach = (
        game.realization_probabilities(baseline_profile)
        if baseline_profile is not None
        else None
    )
    baseline_policy = baseline_policy or {}
    information_snapshots = information_snapshots or {}
    unknown_baseline = set(baseline_policy) - set(game.information_set_map)
    if unknown_baseline:
        raise ValueError(
            f"baseline policy references unknown information sets {sorted(unknown_baseline)}"
        )
    for identifier, action in baseline_policy.items():
        if action not in game.information_set_map[identifier].action_ids:
            raise ValueError(f"baseline policy selects an illegal action at {identifier!r}")
    unknown_snapshots = set(information_snapshots) - set(game.information_set_map)
    if unknown_snapshots:
        raise ValueError(
            "information snapshots reference unknown information sets "
            f"{sorted(unknown_snapshots)}"
        )
    for identifier, snapshot in information_snapshots.items():
        if snapshot.information_set_id != identifier:
            raise ValueError("information-snapshot key must match the snapshot id")
    expected_terminals = set(game.terminal_ids())
    if set(outcome_features) != expected_terminals:
        raise ValueError("outcome features must cover every terminal node exactly")
    if utilities is not None:
        if set(utilities) != expected_terminals:
            raise ValueError("utilities must cover every terminal node exactly")
        expected_players = set(game.player_ids)
        if any(set(values) != expected_players for values in utilities.values()):
            raise ValueError("terminal utilities must cover every game player")
    if robust_utilities is not None and set(robust_utilities) != expected_terminals:
        raise ValueError("robust utilities must cover every terminal node exactly")
    paths: list[CounterfactualPath] = []

    def walk(node_id: str, probability: float, events: tuple[ActionEvent, ...]) -> None:
        node = game.nodes[node_id]
        if isinstance(node, TerminalNode):
            differences = tuple(
                DecisionDifference(
                    event.information_set_id,
                    baseline_policy.get(event.information_set_id),
                    event.action,
                )
                for event in events
                if event.information_set_id is not None
                and baseline_policy.get(event.information_set_id) != event.action
            )
            snapshots = tuple(
                information_snapshots[difference.information_set_id]
                for difference in differences
                if difference.information_set_id in information_snapshots
            )
            strategy_audited = bool(
                feasibility_report is not None and feasibility_report.feasible
            )
            eq_supported = (
                equilibrium_reach is not None
                and certificate_supported
                and strategy_audited
                and equilibrium_reach[node_id] > tolerance
            )
            terminal_utilities = (
                dict(utilities[node_id])
                if utilities is not None
                else dict(node.terminal_utilities.utilities)
            )
            paths.append(
                CounterfactualPath(
                    node_id,
                    events,
                    probability,
                    outcome_features[node_id],
                    terminal_utilities,
                    differences,
                    snapshots,
                    True,
                    strategy_audited and probability > tolerance,
                    eq_supported,
                    (
                        float(baseline_reach[node_id])
                        if baseline_reach is not None
                        else None
                    ),
                    (
                        float(equilibrium_reach[node_id])
                        if equilibrium_reach is not None
                        else None
                    ),
                    (
                        None
                        if robust_utilities is None
                        else robust_utilities.get(node_id)
                    ),
                )
            )
            return
        if isinstance(node, DecisionNode):
            for action, child in node.actions.items():
                walk(
                    child,
                    probability * strategy[node.information_set_id][action],
                    (*events, ActionEvent(node.id, node.player_id, action, node.information_set_id)),
                )
            return
        if isinstance(node, ChanceNode):
            for branch in node.branches:
                walk(
                    branch.child_id,
                    probability * branch.probability,
                    (*events, ActionEvent(node.id, "nature", branch.outcome_id, None)),
                )

    walk(game.root_id, 1.0, ())
    return tuple(sorted(paths, key=lambda path: path.terminal_id))


def top_paths(
    paths: tuple[CounterfactualPath, ...],
    *,
    ranking: PathRanking,
    k: int,
    focal_player: str | None = None,
    include_ties: bool = True,
    tolerance: float = 1e-12,
) -> tuple[CounterfactualPath, ...]:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be positive")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if not isinstance(ranking, PathRanking):
        raise TypeError("ranking must be a PathRanking")
    if ranking in {PathRanking.FOCAL_PLAYER_UTILITY, PathRanking.ROBUST_WORST_CASE_UTILITY} and focal_player is None:
        raise ValueError("this path ranking requires focal_player")

    def score(path: CounterfactualPath) -> float:
        if ranking is PathRanking.EQUILIBRIUM_REACH_PROBABILITY:
            if path.equilibrium_reach_probability is None:
                raise ValueError(
                    "equilibrium reach ranking requires an equilibrium profile"
                )
            return path.equilibrium_reach_probability
        if ranking is PathRanking.BASELINE_REACH_PROBABILITY:
            if path.baseline_reach_probability is None:
                raise ValueError("baseline reach ranking requires a baseline profile")
            return path.baseline_reach_probability
        if ranking is PathRanking.FOCAL_PLAYER_UTILITY:
            assert focal_player is not None
            if focal_player not in path.utilities:
                raise ValueError("focal_player is absent from path utilities")
            return path.utilities[focal_player]
        if ranking is PathRanking.JOINT_UTILITY:
            return sum(path.utilities.values())
        if ranking is PathRanking.ESCALATION_AVOIDANCE:
            return -float(path.terminal_outcome.common.get("military_escalation", False))
        if ranking is PathRanking.CATASTROPHE_AVOIDANCE:
            return -float(path.terminal_outcome.common.get("catastrophic_escalation", False))
        if ranking is PathRanking.NEGOTIATED_SETTLEMENT:
            return float(path.terminal_outcome.common.get("negotiated_agreement", False))
        if ranking in {
            PathRanking.MINIMUM_DEVIATIONS,
            PathRanking.MINIMUM_DEVIATION_COUNT,
        }:
            return -float(len(path.differences_from_baseline))
        if ranking is PathRanking.MODEL_REACHABILITY:
            return float(path.model_reachable)
        if ranking is PathRanking.EQUILIBRIUM_SUPPORT:
            return float(path.equilibrium_supported)
        if ranking is PathRanking.ROBUSTNESS:
            if path.robust_utilities is None:
                raise ValueError("robust path ranking requires scenario utilities")
            scenario_count = len(next(iter(path.robust_utilities.values())))
            return min(
                sum(path.robust_utilities[player][index] for player in path.utilities)
                for index in range(scenario_count)
            )
        assert focal_player is not None
        if path.robust_utilities is None:
            raise ValueError("robust path ranking requires scenario utilities")
        if focal_player not in path.robust_utilities:
            raise ValueError("focal_player is absent from robust path utilities")
        return min(path.robust_utilities[focal_player])

    ordered = tuple(sorted(paths, key=lambda item: (-score(item), item.terminal_id)))
    if len(ordered) <= k or not include_ties:
        return ordered[:k]
    cutoff = score(ordered[k - 1])
    return tuple(path for path in ordered if score(path) >= cutoff - tolerance)
