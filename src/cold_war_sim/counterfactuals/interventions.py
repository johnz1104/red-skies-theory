"""Concrete transformations supported by counterfactual adapters."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from cold_war_sim.core.types import (
    SerializableMixin,
    deep_frozen_json,
    frozen_mapping,
    to_serializable,
    validate_stable_id,
)


@runtime_checkable
class Intervention(Protocol):
    """Marker protocol implemented by immutable concrete interventions."""

    @property
    def kind(self) -> str: ...

    def to_dict(self) -> dict[str, Any]: ...


class CommitmentScope(StrEnum):
    ACTION = "ACTION"
    PARTIAL_POLICY = "PARTIAL_POLICY"
    COMPLETE_POLICY = "COMPLETE_POLICY"


def _ids(values: tuple[str, ...], *, name: str) -> tuple[str, ...]:
    result = tuple(sorted(values))
    if not result:
        raise ValueError(f"{name} must not be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    for value in result:
        validate_stable_id(value, field_name=name)
    return result


@dataclass(frozen=True)
class PolicyReplacement(SerializableMixin):
    player_id: str
    policy: Mapping[str, str]

    @property
    def kind(self) -> str:
        return "policy_replacement"

    def __post_init__(self) -> None:
        validate_stable_id(self.player_id, field_name="player id")
        if not self.policy:
            raise ValueError("replacement policy must not be empty")
        for information_set, action in self.policy.items():
            validate_stable_id(information_set, field_name="information-set id")
            validate_stable_id(action, field_name="action id")
        object.__setattr__(self, "policy", frozen_mapping(self.policy))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "POLICY_REPLACEMENT", "player": self.player_id, "policy": dict(self.policy)}


@dataclass(frozen=True)
class ActionRestriction(SerializableMixin):
    player_id: str
    information_set_id: str
    removed_actions: tuple[str, ...]

    @property
    def kind(self) -> str:
        return "action_restriction"

    def __post_init__(self) -> None:
        validate_stable_id(self.player_id, field_name="player id")
        validate_stable_id(self.information_set_id, field_name="information-set id")
        object.__setattr__(
            self, "removed_actions", _ids(self.removed_actions, name="removed actions")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "ACTION_RESTRICTION",
            "player": self.player_id,
            "information_set": self.information_set_id,
            "actions": list(self.removed_actions),
        }


@dataclass(frozen=True)
class ActionExpansion(SerializableMixin):
    player_id: str
    information_set_id: str
    added_action: str
    transformation_id: str

    @property
    def kind(self) -> str:
        return "action_expansion"

    def __post_init__(self) -> None:
        validate_stable_id(self.player_id, field_name="player id")
        validate_stable_id(self.information_set_id, field_name="information-set id")
        validate_stable_id(self.added_action, field_name="added action")
        validate_stable_id(self.transformation_id, field_name="transformation id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "ACTION_EXPANSION",
            "transformation_id": self.transformation_id,
            "player": self.player_id,
            "information_set": self.information_set_id,
            "action": self.added_action,
        }


@dataclass(frozen=True)
class Commitment(SerializableMixin):
    player_id: str
    policy: Mapping[str, str]
    observable: bool
    binding: bool
    scope: CommitmentScope

    @property
    def kind(self) -> str:
        return "commitment"

    def __post_init__(self) -> None:
        validate_stable_id(self.player_id, field_name="player id")
        if not self.policy:
            raise ValueError("commitment policy must not be empty")
        if not isinstance(self.observable, bool) or not isinstance(self.binding, bool):
            raise TypeError("observable and binding must be booleans")
        if self.scope is CommitmentScope.ACTION and len(self.policy) != 1:
            raise ValueError("an ACTION commitment must contain exactly one decision")
        for information_set, action in self.policy.items():
            validate_stable_id(information_set, field_name="information-set id")
            validate_stable_id(action, field_name="action id")
        object.__setattr__(self, "policy", frozen_mapping(self.policy))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "COMMITMENT",
            "player": self.player_id,
            "policy": dict(self.policy),
            "observable": self.observable,
            "binding": self.binding,
            "commitment_scope": self.scope.value,
        }


@dataclass(frozen=True)
class ParameterIntervention(SerializableMixin):
    parameters: Mapping[str, Any]

    @property
    def kind(self) -> str:
        return "parameter_intervention"

    def __post_init__(self) -> None:
        if not self.parameters:
            raise ValueError("parameter intervention must change at least one parameter")
        for name in self.parameters:
            validate_stable_id(name, field_name="parameter name")
        to_serializable(self.parameters)
        object.__setattr__(self, "parameters", deep_frozen_json(self.parameters))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "PARAMETER_INTERVENTION", "changes": dict(self.parameters)}


@dataclass(frozen=True)
class InformationIntervention(SerializableMixin):
    channel_id: str
    observers: tuple[str, ...]
    accuracy: float | None = None
    remove: bool = False
    public: bool = False

    @property
    def kind(self) -> str:
        return "information_intervention"

    def __post_init__(self) -> None:
        validate_stable_id(self.channel_id, field_name="information-channel id")
        object.__setattr__(self, "observers", _ids(self.observers, name="observers"))
        if not isinstance(self.remove, bool) or not isinstance(self.public, bool):
            raise TypeError("remove and public must be booleans")
        if self.accuracy is not None:
            accuracy = float(self.accuracy)
            if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
                raise ValueError("information accuracy must lie in [0, 1]")
            object.__setattr__(self, "accuracy", accuracy)
        if self.remove and self.accuracy is not None:
            raise ValueError("a removed channel cannot also set an accuracy")

    def to_dict(self) -> dict[str, Any]:
        changes: dict[str, Any] = {"remove": self.remove}
        if self.accuracy is not None:
            changes["accuracy"] = self.accuracy
        return {
            "type": "INFORMATION_INTERVENTION",
            "channel": self.channel_id,
            "changes": changes,
            "public": self.public,
            "recipients": list(self.observers),
        }


@dataclass(frozen=True)
class StructuralTransformation(SerializableMixin):
    transformation_id: str
    parameters: Mapping[str, Any]

    @property
    def kind(self) -> str:
        return "structural_transformation"

    def __post_init__(self) -> None:
        validate_stable_id(self.transformation_id, field_name="transformation id")
        to_serializable(self.parameters)
        object.__setattr__(self, "parameters", deep_frozen_json(self.parameters))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "STRUCTURAL_TRANSFORMATION",
            "transformation_id": self.transformation_id,
            "parameters": dict(self.parameters),
        }


class StructuralTransformer(Protocol):
    supported_transformations: tuple[str, ...]

    def transform(self, intervention: StructuralTransformation) -> Any: ...


def apply_policy_replacement(
    game: Any,
    profile: Any,
    intervention: PolicyReplacement,
) -> Any:
    """Replace selected information-set choices in a behavioral profile."""

    from cold_war_sim.core.extensive_form import BehavioralStrategy, ExtensiveFormGame

    if not isinstance(game, ExtensiveFormGame) or not isinstance(
        profile, BehavioralStrategy
    ):
        raise TypeError(
            "generic policy replacement requires ExtensiveFormGame and BehavioralStrategy"
        )
    game.validate_strategy(profile)
    probabilities = {
        identifier: dict(distribution)
        for identifier, distribution in profile.probabilities.items()
    }
    for identifier, selected in intervention.policy.items():
        if identifier not in probabilities:
            raise ValueError(f"unknown information set {identifier!r}")
        information_set = game.information_set_map[identifier]
        if information_set.player_id != intervention.player_id:
            raise ValueError(
                f"information set {identifier!r} is not controlled by "
                f"{intervention.player_id!r}"
            )
        if selected not in probabilities[identifier]:
            raise ValueError(f"illegal action {selected!r} at {identifier!r}")
        probabilities[identifier] = {
            action: float(action == selected) for action in probabilities[identifier]
        }
    return BehavioralStrategy(probabilities)


def apply_action_restriction(game: Any, intervention: ActionRestriction) -> Any:
    """Remove actions from an extensive-form information set and prune unreachable nodes."""

    from cold_war_sim.core.extensive_form import (
        ChanceNode,
        DecisionNode,
        ExtensiveFormGame,
    )

    if not isinstance(game, ExtensiveFormGame):
        raise TypeError("generic action restriction requires ExtensiveFormGame")
    info = game.information_set_map.get(intervention.information_set_id)
    if info is None:
        raise ValueError("action restriction references an unknown information set")
    if info.player_id != intervention.player_id:
        raise ValueError("action restriction is assigned to the wrong player")
    removed = set(intervention.removed_actions)
    unknown = removed - set(info.action_ids)
    if unknown:
        raise ValueError(f"cannot remove unknown actions {sorted(unknown)}")
    if removed == set(info.action_ids):
        raise ValueError("an action restriction cannot remove every legal action")
    nodes = dict(game.nodes)
    for node_id in info.node_ids:
        node = nodes[node_id]
        assert isinstance(node, DecisionNode)
        nodes[node_id] = DecisionNode(
            node.id,
            node.player_id,
            node.information_set_id,
            {action: child for action, child in node.actions.items() if action not in removed},
        )
    reachable: set[str] = set()
    stack = [game.root_id]
    while stack:
        node_id = stack.pop()
        if node_id in reachable:
            continue
        reachable.add(node_id)
        node = nodes[node_id]
        if isinstance(node, DecisionNode):
            stack.extend(node.actions.values())
        elif isinstance(node, ChanceNode):
            stack.extend(branch.child_id for branch in node.branches)
    return ExtensiveFormGame(game.player_ids, game.root_id, {key: nodes[key] for key in reachable})


def apply_parameter_intervention(parameters: Any, intervention: ParameterIntervention) -> Any:
    """Strictly replace documented dataclass parameters; unknown fields are rejected."""

    if not is_dataclass(parameters) or isinstance(parameters, type):
        raise TypeError("generic parameter intervention requires a dataclass instance")
    known = {field.name for field in fields(parameters)}
    unknown = set(intervention.parameters) - known
    if unknown:
        raise ValueError(f"unknown parameter fields {sorted(unknown)}")
    return replace(parameters, **intervention.parameters)


def apply_structural_transformation(
    transformer: StructuralTransformer,
    intervention: StructuralTransformation,
) -> Any:
    if intervention.transformation_id not in transformer.supported_transformations:
        raise ValueError(
            f"unsupported structural transformation {intervention.transformation_id!r}"
        )
    return transformer.transform(intervention)


InterventionType = (
    PolicyReplacement
    | ActionRestriction
    | ActionExpansion
    | Commitment
    | ParameterIntervention
    | InformationIntervention
    | StructuralTransformation
)
