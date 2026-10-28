"""Seeded random streams.

* Sub-games draw from ``SeedSequence(seed, spawn_key=(crc32(event_id), n))``,
  so adding or reordering events never shifts another event's randomness.
* Director hazards use a counter-based uniform (BLAKE2b of seed, event and
  quarter): whether an event fires never depends on evaluation order or on
  which policy is playing, which gives common random numbers across policies.
"""

from __future__ import annotations

import hashlib
import zlib

import numpy as np


def stable_key(text: str) -> int:
    return zlib.crc32(text.encode("utf-8"))


def stream(seed: int, name: str, instance: int = 0) -> np.random.Generator:
    sequence = np.random.SeedSequence(seed, spawn_key=(stable_key(name), instance))
    return np.random.Generator(np.random.PCG64(sequence))


def counter_uniform(seed: int, *parts: object) -> float:
    """A uniform in [0, 1) that depends only on ``seed`` and ``parts``."""

    digest = hashlib.blake2b(
        "|".join(str(p) for p in (seed, *parts)).encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") / 2**64
