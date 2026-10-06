"""State-machine protocol shared by every game in the package.

The shape follows OpenSpiel: a game produces an initial state; a state reports
whose turn it is (a role, chance, or terminal), its legal actions in a declared
order, and is advanced one action at a time. Information states are hashable
keys that identify what a role has observed, so tree algorithms can aggregate
histories a role cannot tell apart.
"""

from __future__ import annotations

import copy
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

CHANCE = "__chance__"
TERMINAL = "__terminal__"
PUBLIC = "__public__"


class IllegalActionError(ValueError):
    """Raised when an action is not legal in the current state."""


@dataclass(frozen=True, slots=True)
class Outcome:
    """Terminal classification of a game.

    ``label`` is the outcome class; ``tags`` are additional facts about how
    the game ended (for example ``"deal"`` or ``"client:aligned_A"``);
    ``metrics`` are numbers such as a settlement share.
    """

    label: str
    tags: tuple[str, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)


@runtime_checkable
class GameState(Protocol):
    """A mutable game state advanced by :meth:`apply_action`."""

    @property
    def roles(self) -> tuple[str, ...]: ...

    def current_player(self) -> str: ...

    def legal_actions(self) -> tuple[str, ...]: ...

    def chance_outcomes(self) -> tuple[tuple[str, float], ...]: ...

    def apply_action(self, action: str) -> None: ...

    def information_state(self, role: str) -> Hashable: ...

    def observation(self, role: str) -> tuple[float, ...]: ...

    def is_terminal(self) -> bool: ...

    def returns(self) -> dict[str, float]: ...

    def outcome(self) -> Outcome: ...

    def history(self) -> tuple[str, ...]: ...

    def clone(self) -> GameState: ...


@runtime_checkable
class Game(Protocol):
    """Factory for initial states of one parameterized game."""

    @property
    def name(self) -> str: ...

    @property
    def roles(self) -> tuple[str, ...]: ...

    def new_initial_state(self) -> GameState: ...


class ImmutableParams:
    """Mixin for frozen parameter dataclasses shared by reference across clones."""

    def __deepcopy__(self, memo: dict[int, object]) -> ImmutableParams:
        return self

    def __copy__(self) -> ImmutableParams:
        return self


class LoggedState:
    """Base class implementing history, per-role views and cloning.

    Subclasses log every event with the roles allowed to see it. The
    information state of a role is its own view log, which gives perfect
    recall as long as a role's own actions are logged as visible to it.
    """

    _roles: tuple[str, ...]
    _history: list[str]
    _views: dict[str, list[str]]

    def __init__(self, roles: Iterable[str]) -> None:
        self._roles = tuple(roles)
        if len(set(self._roles)) != len(self._roles):
            raise ValueError("roles must be unique")
        self._history = []
        self._views = {role: [] for role in self._roles}

    @property
    def roles(self) -> tuple[str, ...]:
        return self._roles

    def history(self) -> tuple[str, ...]:
        return tuple(self._history)

    def information_state(self, role: str) -> Hashable:
        return (role, tuple(self._views[role]))

    def log(self, token: str, visible_to: Iterable[str] | str = PUBLIC) -> None:
        roles = self._roles if visible_to == PUBLIC else tuple(visible_to)
        for role in roles:
            self._views[role].append(token)

    def clone(self) -> LoggedState:
        return copy.deepcopy(self)

    # Subclasses implement the transition.
    def apply_action(self, action: str) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


def check_legal(action: str, legal: tuple[str, ...]) -> None:
    if action not in legal:
        raise IllegalActionError(f"illegal action {action!r}; legal actions are {legal}")
