"""Deterministic JSON conversion shared by traces, caches and replays."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any


def to_jsonable(value: Any) -> Any:
    """Recursively convert *value* to plain JSON data with a stable ordering.

    Non-finite floats are rejected: NaN and infinity are not portable JSON and
    usually indicate a failed computation that should not be recorded silently.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("cannot serialize a non-finite float")
        return value
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {
            str(key): to_jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((to_jsonable(item) for item in value), key=_sort_key)
    if hasattr(value, "item") and callable(value.item):  # numpy scalars
        return to_jsonable(value.item())
    if hasattr(value, "tolist") and callable(value.tolist):  # numpy arrays
        return to_jsonable(value.tolist())
    raise TypeError(f"unsupported serialization type: {type(value).__name__}")


def _sort_key(item: Any) -> str:
    return json.dumps(item, sort_keys=True)


def canonical_json(value: Any, *, indent: int | None = None) -> str:
    """Serialize *value* with sorted keys, no NaN, and compact separators."""

    return json.dumps(
        to_jsonable(value),
        allow_nan=False,
        ensure_ascii=False,
        indent=indent,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
    )


def stable_hash(value: Any) -> str:
    """SHA-256 of the canonical JSON form of *value*."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
