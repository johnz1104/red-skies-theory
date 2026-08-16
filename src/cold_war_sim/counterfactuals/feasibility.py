"""Information-feasibility audits that reject hindsight leakage."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from cold_war_sim.core.types import (
    SerializableMixin,
    deep_frozen_json,
    frozen_mapping,
    validate_stable_id,
)


class CheckStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ViolationKind(StrEnum):
    ILLEGAL_ACTION = "ILLEGAL_ACTION"
    UNKNOWN_INFORMATION_SET = "UNKNOWN_INFORMATION_SET"
    WRONG_PLAYER = "WRONG_PLAYER"
    UNOBSERVED_VARIABLE = "UNOBSERVED_VARIABLE"
    FUTURE_INFORMATION = "FUTURE_INFORMATION"
    MODELER_PARAMETER_TRUTH = "MODELER_PARAMETER_TRUTH"
    SIMULATION_SEED = "SIMULATION_SEED"
    NODE_CONDITIONING = "NODE_CONDITIONING"
    INCONSISTENT_INFORMATION_SET_ACTION = "INCONSISTENT_INFORMATION_SET_ACTION"
    COMMITMENT_VIOLATION = "COMMITMENT_VIOLATION"


@dataclass(frozen=True)
class CheckResult(SerializableMixin):
    status: CheckStatus
    checked_count: int
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, CheckStatus):
            raise TypeError("check status must be a CheckStatus")
        if (
            isinstance(self.checked_count, bool)
            or not isinstance(self.checked_count, int)
            or self.checked_count < 0
        ):
            raise ValueError("checked_count must be a nonnegative integer")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("check-result message must not be empty")


@dataclass(frozen=True)
class InformationVariable(SerializableMixin):
    name: str
    reveal_stage: int
    observed_by: tuple[str, ...]
    modeler_only: bool = False

    def __post_init__(self) -> None:
        validate_stable_id(self.name, field_name="information-variable name")
        if (
            isinstance(self.reveal_stage, bool)
            or not isinstance(self.reveal_stage, int)
            or self.reveal_stage < 0
        ):
            raise ValueError("reveal stage must be a nonnegative integer")
        if not isinstance(self.modeler_only, bool):
            raise TypeError("modeler_only must be a boolean")
        observers = tuple(sorted(self.observed_by))
        if len(observers) != len(set(observers)):
            raise ValueError("observed_by must not contain duplicates")
        for observer in observers:
            validate_stable_id(observer, field_name="observer id")
        object.__setattr__(self, "observed_by", observers)


@dataclass(frozen=True)
class InformationSetSpec(SerializableMixin):
    id: str
    player_id: str
    stage: int
    node_ids: tuple[str, ...]
    legal_actions: tuple[str, ...]
    available_information: tuple[str, ...]
    observation_signature: Mapping[str, object] | None = None
    reachable: bool = True

    def __post_init__(self) -> None:
        validate_stable_id(self.id, field_name="information-set id")
        validate_stable_id(self.player_id, field_name="player id")
        if isinstance(self.stage, bool) or not isinstance(self.stage, int) or self.stage < 0:
            raise ValueError("information-set stage must be a nonnegative integer")
        if not isinstance(self.reachable, bool):
            raise TypeError("reachable must be a boolean")
        for name, values in (
            ("node ids", self.node_ids),
            ("legal actions", self.legal_actions),
            ("available information", self.available_information),
        ):
            if name != "available information" and not values:
                raise ValueError(f"{name} must not be empty")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")
            for value in values:
                validate_stable_id(value, field_name=name)
            object.__setattr__(self, name.replace(" ", "_") if name != "node ids" else "node_ids", tuple(sorted(values)))
        if self.observation_signature is None:
            raise ValueError(
                "an information set requires an explicit observation_signature"
            )
        signature = deep_frozen_json(self.observation_signature)
        if not isinstance(signature, Mapping):
            raise TypeError("observation_signature must be a mapping")
        if set(signature) != set(self.available_information):
            raise ValueError(
                "observation_signature keys must exactly match available_information"
            )
        try:
            json.dumps(dict(signature), allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError("observation_signature must contain strict JSON values") from error
        object.__setattr__(self, "observation_signature", signature)


@dataclass(frozen=True)
class InformationStructure(SerializableMixin):
    information_sets: Mapping[str, InformationSetSpec]
    variables: Mapping[str, InformationVariable]

    def __post_init__(self) -> None:
        information_sets = dict(sorted(self.information_sets.items()))
        variables = dict(sorted(self.variables.items()))
        for key, information_set in information_sets.items():
            if key != information_set.id:
                raise ValueError("information-set key must match its id")
            missing = set(information_set.available_information) - set(variables)
            if missing:
                raise ValueError(f"information set {key!r} references unknown variables {sorted(missing)}")
            for variable_name in information_set.available_information:
                variable = variables[variable_name]
                if variable.modeler_only:
                    raise ValueError(
                        f"information set {key!r} exposes modeler-only variable "
                        f"{variable_name!r}"
                    )
                if information_set.player_id not in variable.observed_by:
                    raise ValueError(
                        f"information set {key!r} exposes unobserved variable "
                        f"{variable_name!r}"
                    )
                if variable.reveal_stage > information_set.stage:
                    raise ValueError(
                        f"information set {key!r} exposes future variable "
                        f"{variable_name!r}"
                    )
        for key, variable in variables.items():
            if key != variable.name:
                raise ValueError("information-variable key must match its name")
        partitions: dict[tuple[object, ...], str] = {}
        for identifier, information_set in information_sets.items():
            signature = json.dumps(
                dict(information_set.observation_signature or {}),
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            partition_key = (
                information_set.player_id,
                information_set.stage,
                information_set.legal_actions,
                signature,
            )
            previous = partitions.get(partition_key)
            if previous is not None:
                raise ValueError(
                    "observationally indistinguishable nodes were split across "
                    f"information sets {previous!r} and {identifier!r}"
                )
            partitions[partition_key] = identifier
        object.__setattr__(self, "information_sets", frozen_mapping(information_sets))
        object.__setattr__(self, "variables", frozen_mapping(variables))


@dataclass(frozen=True)
class PolicyDecision(SerializableMixin):
    action: str
    conditions_on: tuple[str, ...] = ()
    node_actions: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.action, field_name="policy action")
        conditions = tuple(sorted(self.conditions_on))
        for condition in conditions:
            validate_stable_id(condition, field_name="conditioned variable")
        object.__setattr__(self, "conditions_on", conditions)
        if self.node_actions is not None:
            if not self.node_actions:
                raise ValueError("node_actions must not be empty when supplied")
            converted = {}
            for node_id, action in self.node_actions.items():
                validate_stable_id(node_id, field_name="policy node id")
                validate_stable_id(action, field_name="node-specific action id")
                converted[node_id] = action
            object.__setattr__(self, "node_actions", frozen_mapping(converted))


@dataclass(frozen=True)
class PurePolicy(SerializableMixin):
    player_id: str
    decisions: Mapping[str, PolicyDecision]

    def __post_init__(self) -> None:
        validate_stable_id(self.player_id, field_name="policy player id")
        if not self.decisions:
            raise ValueError("a pure policy must include at least one decision")
        converted = {}
        for identifier, decision in self.decisions.items():
            validate_stable_id(identifier, field_name="policy information-set id")
            if not isinstance(decision, PolicyDecision):
                raise TypeError("pure-policy decisions must be PolicyDecision values")
            converted[identifier] = decision
        object.__setattr__(self, "decisions", frozen_mapping(converted))

    @classmethod
    def from_actions(cls, player_id: str, actions: Mapping[str, str]) -> PurePolicy:
        return cls(player_id, {key: PolicyDecision(value) for key, value in actions.items()})

    @property
    def actions(self) -> Mapping[str, str]:
        return frozen_mapping({key: decision.action for key, decision in self.decisions.items()})


@dataclass(frozen=True)
class FeasibilityViolation(SerializableMixin):
    kind: ViolationKind
    information_set_id: str
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ViolationKind):
            raise TypeError("violation kind must be a ViolationKind")
        validate_stable_id(
            self.information_set_id, field_name="violation information-set id"
        )
        if not isinstance(self.detail, str) or not self.detail.strip():
            raise ValueError("violation detail must not be empty")


@dataclass(frozen=True)
class FeasibilityReport(SerializableMixin):
    action_legality: CheckResult
    information_consistency: CheckResult
    temporal_consistency: CheckResult
    commitment_consistency: CheckResult
    reachable_information_sets: tuple[str, ...]
    violations: tuple[FeasibilityViolation, ...]

    def __post_init__(self) -> None:
        reachable = tuple(sorted(self.reachable_information_sets))
        if len(reachable) != len(set(reachable)):
            raise ValueError("reachable information sets must not contain duplicates")
        for identifier in reachable:
            validate_stable_id(identifier, field_name="reachable information-set id")
        object.__setattr__(
            self, "reachable_information_sets", reachable
        )
        violations = tuple(self.violations)
        if any(not isinstance(item, FeasibilityViolation) for item in violations):
            raise TypeError("violations must contain FeasibilityViolation values")
        object.__setattr__(self, "violations", violations)
        statuses = (
            self.action_legality.status,
            self.information_consistency.status,
            self.temporal_consistency.status,
            self.commitment_consistency.status,
        )
        if self.violations and CheckStatus.FAIL not in statuses:
            raise ValueError("a report with violations must contain a failed check")
        if CheckStatus.FAIL in statuses and not self.violations:
            raise ValueError("a failed feasibility check must report a violation")

    @property
    def feasible(self) -> bool:
        return not self.violations and all(
            check.status is not CheckStatus.FAIL
            for check in (
                self.action_legality,
                self.information_consistency,
                self.temporal_consistency,
                self.commitment_consistency,
            )
        )


RESERVED_SEED_VARIABLES = frozenset({"seed", "simulation_seed", "rng_state"})
RESERVED_FUTURE_VARIABLES = frozenset({"terminal_outcome", "future_action"})


def audit_policy(
    policy: PurePolicy,
    information: InformationStructure,
    *,
    committed_actions: Mapping[str, str] | None = None,
    require_complete: bool = True,
) -> FeasibilityReport:
    """Audit legality, information consistency, chronology, and commitments."""

    violations: list[FeasibilityViolation] = []
    action_checks = 0
    information_checks = 0
    temporal_checks = 0
    commitment_checks = 0
    owned = {
        identifier: info
        for identifier, info in information.information_sets.items()
        if info.player_id == policy.player_id
    }
    if committed_actions:
        for identifier, committed_action in committed_actions.items():
            info = information.information_sets.get(identifier)
            if info is None:
                violations.append(
                    FeasibilityViolation(
                        ViolationKind.UNKNOWN_INFORMATION_SET,
                        identifier,
                        "binding commitment references an unknown information set",
                    )
                )
                continue
            if info.player_id != policy.player_id:
                violations.append(
                    FeasibilityViolation(
                        ViolationKind.WRONG_PLAYER,
                        identifier,
                        "binding commitment references another player's decision",
                    )
                )
                continue
            commitment_checks += 1
            if committed_action not in info.legal_actions:
                violations.append(
                    FeasibilityViolation(
                        ViolationKind.ILLEGAL_ACTION,
                        identifier,
                        "binding commitment selects an illegal action",
                    )
                )
    unknown = set(policy.decisions) - set(information.information_sets)
    for identifier in sorted(unknown):
        violations.append(
            FeasibilityViolation(
                ViolationKind.UNKNOWN_INFORMATION_SET,
                identifier,
                "policy references an information set absent from the model",
            )
        )
    if require_complete:
        for identifier in sorted(set(owned) - set(policy.decisions)):
            violations.append(
                FeasibilityViolation(
                    ViolationKind.UNKNOWN_INFORMATION_SET,
                    identifier,
                    "complete policy omits a player-controlled information set",
                )
            )
    for identifier, decision in policy.decisions.items():
        info = information.information_sets.get(identifier)
        if info is None:
            continue
        if info.player_id != policy.player_id:
            violations.append(
                FeasibilityViolation(
                    ViolationKind.WRONG_PLAYER,
                    identifier,
                    f"information set is controlled by {info.player_id!r}",
                )
            )
            continue
        action_checks += 1
        if decision.action not in info.legal_actions:
            violations.append(
                FeasibilityViolation(
                    ViolationKind.ILLEGAL_ACTION,
                    identifier,
                    f"action {decision.action!r} is not legal",
                )
            )
        if decision.node_actions is not None:
            information_checks += 1
            if set(decision.node_actions) != set(info.node_ids):
                violations.append(
                    FeasibilityViolation(
                        ViolationKind.NODE_CONDITIONING,
                        identifier,
                        "node-specific actions must cover the information set exactly",
                    )
                )
            if set(decision.node_actions.values()) != {decision.action}:
                violations.append(
                    FeasibilityViolation(
                        ViolationKind.INCONSISTENT_INFORMATION_SET_ACTION,
                        identifier,
                        "a policy cannot choose different actions at nodes in one information set",
                    )
                )
        for variable_name in decision.conditions_on:
            information_checks += 1
            variable = information.variables.get(variable_name)
            if variable_name in RESERVED_SEED_VARIABLES:
                violations.append(
                    FeasibilityViolation(
                        ViolationKind.SIMULATION_SEED,
                        identifier,
                        "a policy cannot condition on a simulation seed",
                    )
                )
                continue
            if variable_name in RESERVED_FUTURE_VARIABLES:
                violations.append(
                    FeasibilityViolation(
                        ViolationKind.FUTURE_INFORMATION,
                        identifier,
                        "a policy cannot condition on a future action or outcome",
                    )
                )
                continue
            if variable is None:
                violations.append(
                    FeasibilityViolation(
                        ViolationKind.UNOBSERVED_VARIABLE,
                        identifier,
                        f"{variable_name!r} is not available at this information set",
                    )
                )
                continue
            if variable_name not in info.available_information:
                violations.append(
                    FeasibilityViolation(
                        ViolationKind.UNOBSERVED_VARIABLE,
                        identifier,
                        f"{variable_name!r} is not available at this information set",
                    )
                )
            if variable.modeler_only:
                violations.append(
                    FeasibilityViolation(
                        ViolationKind.MODELER_PARAMETER_TRUTH,
                        identifier,
                        f"{variable_name!r} is modeler-only truth",
                    )
                )
            if policy.player_id not in variable.observed_by:
                violations.append(
                    FeasibilityViolation(
                        ViolationKind.UNOBSERVED_VARIABLE,
                        identifier,
                        f"{variable_name!r} is not observed by this player",
                    )
                )
            temporal_checks += 1
            if variable.reveal_stage > info.stage:
                violations.append(
                    FeasibilityViolation(
                        ViolationKind.FUTURE_INFORMATION,
                        identifier,
                        f"{variable_name!r} arrives after the decision",
                    )
                )
        if (
            committed_actions
            and identifier in committed_actions
            and decision.action != committed_actions[identifier]
        ):
            violations.append(
                FeasibilityViolation(
                    ViolationKind.COMMITMENT_VIOLATION,
                    identifier,
                    "policy conflicts with a binding commitment",
                )
            )
    kinds = {item.kind for item in violations}
    action_failed = bool(kinds & {ViolationKind.ILLEGAL_ACTION, ViolationKind.UNKNOWN_INFORMATION_SET, ViolationKind.WRONG_PLAYER})
    info_failed = bool(kinds & {ViolationKind.UNOBSERVED_VARIABLE, ViolationKind.MODELER_PARAMETER_TRUTH, ViolationKind.SIMULATION_SEED, ViolationKind.NODE_CONDITIONING, ViolationKind.INCONSISTENT_INFORMATION_SET_ACTION})
    temporal_failed = ViolationKind.FUTURE_INFORMATION in kinds
    commitment_failed = ViolationKind.COMMITMENT_VIOLATION in kinds
    return FeasibilityReport(
        action_legality=CheckResult(CheckStatus.FAIL if action_failed else CheckStatus.PASS, action_checks, "legal action and ownership checks"),
        information_consistency=CheckResult(CheckStatus.FAIL if info_failed else CheckStatus.PASS, information_checks, "information-set and observation checks"),
        temporal_consistency=CheckResult(CheckStatus.FAIL if temporal_failed else CheckStatus.PASS, temporal_checks, "information arrival checks"),
        commitment_consistency=CheckResult(CheckStatus.FAIL if commitment_failed else CheckStatus.PASS, commitment_checks, "binding commitment checks"),
        reachable_information_sets=tuple(
            identifier for identifier, info in owned.items() if info.reachable
        ),
        violations=tuple(violations),
    )
