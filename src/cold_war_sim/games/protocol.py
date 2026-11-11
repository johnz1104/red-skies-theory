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
from typing import Protocol, runtime_checkable

from ..records import Record

CHANCE = "__chance__"
TERMINAL = "__terminal__"
PUBLIC = "__public__"


class IllegalActionError(ValueError):
    """Raised when an action is not legal in the current state."""


class Outcome(Record):
    """How a game ended.

    ``label`` is the outcome class; ``tags`` are further facts about the ending (for
    example ``"deal"`` or ``"client:aligned_A"``); ``metrics`` are numbers such as a
    settlement share.
    """

    def __init__(
        self,
        label: str,
        tags: tuple[str, ...] = (),
        metrics: Mapping[str, float] | None = None,
    ) -> None:
        self.label = label
        self.tags = tags
        self.metrics: Mapping[str, float] = {} if metrics is None else metrics


@runtime_checkable
class GameState(Protocol):
    """A mutable game state advanced by :meth:`apply_action`."""

    @property
    def roles(self) -> tuple[str, ...]:
        """Every role in the game, in a fixed order."""
        ...

    def current_player(self) -> str:
        """The role to act, or ``CHANCE``, or ``TERMINAL``."""
        ...

    def legal_actions(self) -> tuple[str, ...]:
        """The acting role's actions, in a fixed order (empty at chance and at the end)."""
        ...

    def chance_outcomes(self) -> tuple[tuple[str, float], ...]:
        """At a chance node, each possible draw with its probability."""
        ...

    def apply_action(self, action: str) -> None:
        """Advance the state by one action or chance draw."""
        ...

    def information_state(self, role: str) -> Hashable:
        """A key for everything *role* has observed so far."""
        ...

    def observation(self, role: str) -> tuple[float, ...]:
        """A fixed-length numeric summary of what *role* sees."""
        ...

    def is_terminal(self) -> bool:
        """Whether the game has ended."""
        ...

    def returns(self) -> dict[str, float]:
        """Each role's payoff, once the game has ended."""
        ...

    def outcome(self) -> Outcome:
        """How the game ended."""
        ...

    def history(self) -> tuple[str, ...]:
        """Every action and chance draw so far, in order."""
        ...

    def clone(self) -> GameState:
        """An independent copy of the state."""
        ...


@runtime_checkable
class Game(Protocol):
    """Factory for initial states of one parameterized game."""

    @property
    def name(self) -> str:
        """The family's name."""
        ...

    @property
    def roles(self) -> tuple[str, ...]:
        """Every role in the game, in a fixed order."""
        ...

    def new_initial_state(self) -> GameState:
        """A fresh state at the start of the game."""
        ...


class ImmutableParams:
    """Mixin for parameter objects that are never changed after they are built.

    Copies of a state share them: copying returns the object itself.
    """

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
        """Every role in the game, in a fixed order."""

        return self._roles

    def history(self) -> tuple[str, ...]:
        """Every action and chance draw so far, in order."""

        return tuple(self._history)

    def information_state(self, role: str) -> Hashable:
        """The role and the log of everything it was allowed to see."""

        return (role, tuple(self._views[role]))

    def log(self, token: str, visible_to: Iterable[str] | str = PUBLIC) -> None:
        """Record *token* in the view of each role allowed to see it (everyone by default)."""

        roles = self._roles if visible_to == PUBLIC else tuple(visible_to)
        for role in roles:
            self._views[role].append(token)

    def clone(self) -> LoggedState:
        """An independent deep copy (parameter objects are shared)."""

        return copy.deepcopy(self)

    def apply_action(self, action: str) -> None:  # pragma: no cover - abstract
        """Advance the state; subclasses implement the transition."""

        raise NotImplementedError


def check_legal(action: str, legal: tuple[str, ...]) -> None:
    """Raise :class:`IllegalActionError` unless *action* is one of *legal*."""

    if action not in legal:
        raise IllegalActionError(f"illegal action {action!r}; legal actions are {legal}")
