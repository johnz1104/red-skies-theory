"""A small base class for value objects."""

from __future__ import annotations

from typing import Any, Self


class Record:
    """A plain value object: compared, printed and copied by its attributes.

    A subclass sets its attributes in ``__init__``; nothing is generated. Because records
    compare by value they are not hashable; a class that must serve as a dictionary key
    defines its own ``__hash__``.
    """

    def to_dict(self) -> dict[str, Any]:
        """The attributes by name, in the order ``__init__`` set them."""

        return dict(vars(self))

    def replace(self, **changes: Any) -> Self:
        """A copy with the named attributes changed.

        ``__init__`` is not run again; a class that validates itself defines ``_check``,
        which is run on the copy.
        """

        unknown = sorted(set(changes) - set(vars(self)))
        if unknown:
            raise TypeError(f"{type(self).__name__} has no attributes {unknown}")
        new = object.__new__(type(self))
        vars(new).update(vars(self))
        vars(new).update(changes)
        new._check()
        return new

    def _check(self) -> None:
        """Raise if the attributes are inconsistent (nothing to check by default)."""

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return vars(self) == vars(other)

    def __repr__(self) -> str:
        fields = ", ".join(f"{name}={value!r}" for name, value in vars(self).items())
        return f"{type(self).__name__}({fields})"
