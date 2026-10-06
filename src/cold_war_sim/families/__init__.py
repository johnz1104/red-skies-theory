"""Mechanism-based game families.

Each family is a role-agnostic game on the kernel protocol, configured from a
plain mapping by its ``build`` function; role ids are chosen by the caller.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from . import alignment, attrition, bargaining, crisis, intervention
from .common import ConfigError

Builder = Callable[[Mapping[str, Any]], Any]

FAMILIES: dict[str, Builder] = {
    crisis.NAME: crisis.build,
    bargaining.NAME: bargaining.build,
    intervention.NAME: intervention.build,
    attrition.NAME: attrition.build,
    alignment.NAME: alignment.build,
}


def build_family(family: str, config: Mapping[str, Any]) -> Any:
    try:
        builder = FAMILIES[family]
    except KeyError as exc:
        raise ConfigError(f"unknown family {family!r}; known: {sorted(FAMILIES)}") from exc
    return builder(config)


__all__ = ["FAMILIES", "ConfigError", "build_family"]
