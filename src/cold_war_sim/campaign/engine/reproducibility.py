"""Reproducibility: seeded random streams, and canonical JSON for traces, caches and hashes.

* Sub-games draw from ``SeedSequence(seed, spawn_key=(crc32(event_id), n))``,
  so adding or reordering events never shifts another event's randomness.
* Director hazards use a counter-based uniform (BLAKE2b of seed, event and
  quarter): whether an event fires never depends on evaluation order or on
  which policy is playing, which gives common random numbers across policies.
* Traces, solver caches and replays share one deterministic JSON form.
"""

from __future__ import annotations

import hashlib
import json
import math
import zlib
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

import numpy as np

from ...records import Record


def stable_key(text: str) -> int:
    """A hash of *text* that is the same in every process (CRC-32)."""

    return zlib.crc32(text.encode("utf-8"))


def stream(seed: int, name: str, instance: int = 0) -> np.random.Generator:
    """The random stream of sub-game *name*, instance *instance*, under *seed*."""

    sequence = np.random.SeedSequence(seed, spawn_key=(stable_key(name), instance))
    return np.random.Generator(np.random.PCG64(sequence))


def counter_uniform(seed: int, *parts: object) -> float:
    """A uniform in [0, 1) that depends only on ``seed`` and ``parts``."""

    digest = hashlib.blake2b(
        "|".join(str(p) for p in (seed, *parts)).encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") / 2**64


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
    if isinstance(value, Record):
        return {name: to_jsonable(item) for name, item in value.to_dict().items()}
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
