"""Small, explicit value objects shared by the supported models.

The classes in this module deliberately distinguish stable identifiers from
display labels.  Stable identifiers are what appear in strategies, histories,
and serialized results; labels are presentation metadata only.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, StrEnum
from types import MappingProxyType
from typing import Any, TypeVar

K = TypeVar("K")
V = TypeVar("V")


def validate_stable_id(value: str, *, field_name: str = "identifier") -> str:
    """Return *value* after validating that it is a usable stable identifier."""

    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty, trimmed string")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{field_name} must not contain control characters")
    return value


def _finite_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def frozen_mapping(mapping: Mapping[K, V]) -> Mapping[K, V]:
    """Create an immutable mapping with deterministic key iteration order."""

    return MappingProxyType(
        dict(sorted(mapping.items(), key=lambda item: str(item[0])))
    )


def deep_frozen_json(value: Any) -> Any:
    """Recursively copy strict JSON-like data into immutable containers.

    This is intended for value objects that must not retain mutable aliases to
    caller-owned nested dictionaries or lists.  Mapping keys remain strings,
    mappings become deterministic ``MappingProxyType`` objects, and sequences
    become tuples.  Non-finite and unsupported leaves are rejected.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("cannot freeze a non-finite floating-point value")
        return value
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            if not isinstance(key, str):
                raise TypeError("JSON-like mapping keys must be strings")
            converted[key] = deep_frozen_json(item)
        return MappingProxyType(converted)
    if isinstance(value, (tuple, list)):
        return tuple(deep_frozen_json(item) for item in value)
    raise TypeError(f"unsupported immutable JSON value: {type(value).__name__}")


def to_serializable(value: Any) -> Any:
    """Recursively convert supported objects to deterministic JSON data.

    Non-finite floating-point values are rejected because JSON encodings of
    NaN and infinity are not portable and can conceal failed simulations.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("cannot serialize a non-finite floating-point value")
        return value
    if isinstance(value, Enum):
        return to_serializable(value.value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return to_serializable(value.item())
        except (TypeError, ValueError):
            pass
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_serializable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): to_serializable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [to_serializable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [to_serializable(item) for item in value]
        return sorted(converted, key=lambda item: json.dumps(item, sort_keys=True))
    if hasattr(value, "tolist") and callable(value.tolist):
        return to_serializable(value.tolist())
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return to_serializable(value.to_dict())
    raise TypeError(f"unsupported serialization type: {type(value).__name__}")


def canonical_json(value: Any, *, indent: int | None = None) -> str:
    """Serialize *value* with stable ordering and strict finite-number rules."""

    return json.dumps(
        to_serializable(value),
        allow_nan=False,
        ensure_ascii=False,
        indent=indent,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
    )


class SerializableMixin:
    """Mixin providing deterministic dictionary and JSON representations."""

    def to_dict(self) -> dict[str, Any]:
        result = to_serializable(self)
        if not isinstance(result, dict):
            raise TypeError("top-level serializable representation must be a mapping")
        return result

    def to_json(self, *, indent: int | None = None) -> str:
        return canonical_json(self, indent=indent)


@dataclass(frozen=True, order=True)
class Player(SerializableMixin):
    id: str
    name: str | None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.id, field_name="player id")
        if self.name is not None:
            validate_stable_id(self.name, field_name="player name")

    @property
    def label(self) -> str:
        return self.name or self.id


@dataclass(frozen=True, order=True)
class PlayerType(SerializableMixin):
    id: str
    player_id: str
    label: str | None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.id, field_name="type id")
        validate_stable_id(self.player_id, field_name="type player id")
        if self.label is not None:
            validate_stable_id(self.label, field_name="type label")


@dataclass(frozen=True, order=True)
class Action(SerializableMixin):
    id: str
    label: str | None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.id, field_name="action id")
        if self.label is not None:
            validate_stable_id(self.label, field_name="action label")


@dataclass(frozen=True, order=True)
class ChanceOutcome(SerializableMixin):
    id: str
    probability: float
    label: str | None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.id, field_name="chance-outcome id")
        probability = _finite_number(self.probability, field_name="chance probability")
        if probability < 0.0 or probability > 1.0:
            raise ValueError("chance probability must lie in [0, 1]")
        object.__setattr__(self, "probability", probability)
        if self.label is not None:
            validate_stable_id(self.label, field_name="chance-outcome label")


@dataclass(frozen=True, order=True)
class HistoryEntry(SerializableMixin):
    node_id: str
    actor_id: str
    action_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.node_id, field_name="history node id")
        validate_stable_id(self.actor_id, field_name="history actor id")
        validate_stable_id(self.action_id, field_name="history action id")


@dataclass(frozen=True)
class History(SerializableMixin):
    entries: tuple[HistoryEntry, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(self.entries))

    def append(self, entry: HistoryEntry) -> History:
        return History((*self.entries, entry))


class ParameterProvenance(StrEnum):
    ILLUSTRATIVE_ASSUMPTION = "illustrative assumption"
    NORMALIZATION = "normalization"
    NUMERICAL_CONVENIENCE = "numerical convenience"
    EXTERNALLY_SOURCED_ESTIMATE = "externally sourced estimate"


@dataclass(frozen=True)
class ModelParameter(SerializableMixin):
    name: str
    value: Any
    provenance: ParameterProvenance
    description: str = ""
    citation: str | None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.name, field_name="parameter name")
        if not isinstance(self.provenance, ParameterProvenance):
            object.__setattr__(self, "provenance", ParameterProvenance(self.provenance))
        if (
            self.provenance is ParameterProvenance.EXTERNALLY_SOURCED_ESTIMATE
            and not self.citation
        ):
            raise ValueError("an externally sourced estimate requires a citation")
        # This also rejects unsupported values and non-finite nested numbers.
        to_serializable(self.value)


@dataclass(frozen=True)
class ModelParameters(SerializableMixin):
    """A deterministic, immutable collection of explicitly sourced parameters."""

    parameters: tuple[ModelParameter, ...]
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        parameters = tuple(
            sorted(self.parameters, key=lambda parameter: parameter.name)
        )
        names = [parameter.name for parameter in parameters]
        if len(names) != len(set(names)):
            raise ValueError("model parameter names must be unique")
        object.__setattr__(self, "parameters", parameters)
        validate_stable_id(self.schema_version, field_name="parameter schema version")

    def __getitem__(self, name: str) -> Any:
        for parameter in self.parameters:
            if parameter.name == name:
                return parameter.value
        raise KeyError(name)

    def as_value_dict(self) -> dict[str, Any]:
        return {parameter.name: parameter.value for parameter in self.parameters}


def labels(values: Sequence[str | Player | Action]) -> tuple[str, ...]:
    """Extract stable IDs from a homogeneous sequence of labels/value objects."""

    result: list[str] = []
    for value in values:
        item = value.id if isinstance(value, (Player, Action)) else value
        result.append(validate_stable_id(item))
    if len(result) != len(set(result)):
        raise ValueError("identifiers must be unique")
    if not result:
        raise ValueError("at least one identifier is required")
    return tuple(result)
