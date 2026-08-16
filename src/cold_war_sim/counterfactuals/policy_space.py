"""Deterministic pure-policy spaces defined on information sets."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator
from dataclasses import dataclass

from cold_war_sim.core.types import SerializableMixin, validate_stable_id

from .feasibility import InformationStructure, PurePolicy


@dataclass(frozen=True)
class PolicySpace(SerializableMixin):
    player_id: str
    information_set_ids: tuple[str, ...]
    action_sets: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.player_id, field_name="policy-space player id")
        if len(self.information_set_ids) != len(self.action_sets):
            raise ValueError("one action set is required per information set")
        if len(self.information_set_ids) != len(set(self.information_set_ids)):
            raise ValueError("information-set ids must be unique")
        if any(not actions for actions in self.action_sets):
            raise ValueError("every information set must have a legal action")
        for identifier, actions in zip(
            self.information_set_ids, self.action_sets, strict=True
        ):
            validate_stable_id(identifier, field_name="information-set id")
            if len(actions) != len(set(actions)):
                raise ValueError(
                    f"action set at {identifier!r} must not contain duplicates"
                )
            for action in actions:
                validate_stable_id(action, field_name="action id")

    @classmethod
    def from_information_structure(
        cls,
        information: InformationStructure,
        player_id: str,
        *,
        allowed_information_sets: tuple[str, ...] | None = None,
        fixed_actions: dict[str, str] | None = None,
    ) -> PolicySpace:
        allowed = None if allowed_information_sets is None else set(allowed_information_sets)
        fixed_actions = fixed_actions or {}
        owned = {
            identifier
            for identifier, info in information.information_sets.items()
            if info.player_id == player_id
        }
        if allowed is not None:
            unknown = allowed - set(information.information_sets)
            if unknown:
                raise ValueError(
                    f"unknown allowed information sets: {sorted(unknown)}"
                )
            wrong_player = allowed - owned
            if wrong_player:
                raise ValueError(
                    "allowed information sets are not controlled by "
                    f"{player_id!r}: {sorted(wrong_player)}"
                )
            omitted = owned - allowed
            missing_fixed = omitted - set(fixed_actions)
            if missing_fixed:
                raise ValueError(
                    "a complete policy requires fixed actions for every omitted "
                    f"information set: {sorted(missing_fixed)}"
                )
        unknown_fixed = set(fixed_actions) - owned
        if unknown_fixed:
            raise ValueError(
                "fixed actions reference unknown or unowned information sets: "
                f"{sorted(unknown_fixed)}"
            )
        records = []
        for identifier, info in information.information_sets.items():
            if info.player_id != player_id:
                continue
            actions = (
                (fixed_actions[identifier],)
                if identifier in fixed_actions
                else tuple(sorted(info.legal_actions))
            )
            if any(action not in info.legal_actions for action in actions):
                raise ValueError(f"fixed action at {identifier!r} is illegal")
            records.append((identifier, actions))
        records.sort()
        if not records:
            raise ValueError("player controls no selected information sets")
        return cls(
            player_id,
            tuple(identifier for identifier, _ in records),
            tuple(actions for _, actions in records),
        )

    @property
    def size(self) -> int:
        return math.prod(len(actions) for actions in self.action_sets)

    def enumerate(self) -> Iterator[PurePolicy]:
        for actions in itertools.product(*self.action_sets):
            yield PurePolicy.from_actions(
                self.player_id,
                dict(zip(self.information_set_ids, actions, strict=True)),
            )

    def narrower_configuration(self, maximum_size: int) -> dict[str, object]:
        if maximum_size < 1:
            raise ValueError("maximum size must be positive")
        retained: list[str] = []
        fixed: dict[str, str] = {}
        size = 1
        for identifier, actions in zip(
            self.information_set_ids, self.action_sets, strict=True
        ):
            candidate = size * len(actions)
            if candidate <= maximum_size:
                retained.append(identifier)
                size = candidate
            else:
                fixed[identifier] = actions[0]
        result: dict[str, object] = {
            "fixed_actions": fixed,
            "estimated_size": size,
            "omitted_information_sets": [
                identifier
                for identifier in self.information_set_ids
                if identifier not in retained
            ],
        }
        if retained:
            result["allowed_information_sets"] = retained
        return result
