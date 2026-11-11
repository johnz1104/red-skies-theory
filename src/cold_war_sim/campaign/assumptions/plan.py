"""Build, draw and check a coherent set of uncertain model inputs."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import cache
from typing import Any

from ...records import Record
from ..catalog.core import Catalog, split_address
from ..engine.reproducibility import counter_uniform
from ..payoffs import derive
from ..world.dynamics import DEFAULTS
from ..world.state import POWER_FIELDS, REGION_FIELDS
from .register import (
    _SIG,
    GLOBAL_FILES,
    LOG_KINDS,
    Assumption,
    Number,
    Register,
    RegisterError,
    _from_scale,
    _glob,
    _to_scale,
    inventory,
    load_register,
)


class Sampling(Record):
    """Draw the catalog's numbers within their ranges (off unless set on EngineConfig).

    * ``select``: groups (``goals``), events (``event:korean_war``) or address globs
      (``goals:weights/**``); empty: every live number.
    * ``spread``: the share of each range used, on its natural scale (0: the catalog's
      values).
    * ``draw``: ``None`` draws anew for each episode from its seed; an int fixes one draw
      shared by every episode (robustness runs).
    """

    def __init__(
        self,
        select: tuple[str, ...] = (),
        spread: float = 1.0,
        draw: int | None = None,
    ) -> None:
        self.select = select
        self.spread = spread
        self.draw = draw
        self._check()

    def _check(self) -> None:
        """The spread must lie in [0, 1]."""

        if not 0.0 <= self.spread <= 1.0:
            raise ValueError("spread must lie in [0, 1]")

    def selects(self, item: Assumption) -> bool:
        """Whether *item* is drawn: by group, event or address glob (every live number if the
        selection is empty).
        """

        if not self.select:
            return True
        for token in self.select:
            if ":" not in token:
                if item.group == token:
                    return True
            elif token.startswith("event:") and "/" not in token and "*" not in token:
                if item.address.startswith(token + "/"):
                    return True
            elif _glob(token).fullmatch(item.address):
                return True
        return False


class Plan(Record):
    """The register resolved against one catalog, with liveness for a playable tier."""

    def __init__(
        self,
        items: tuple[Assumption, ...],
        status: Mapping[str, str],
    ) -> None:
        self.items = items
        self.status = status  # address -> live | dead:derived | dead:bound | dead:rails

    def live(self) -> list[Assumption]:
        """The numbers a draw moves: drawable and live."""

        return [a for a in self.items if a.drawable and self.status[a.address] == "live"]

    def draw(self, seed: int, sampling: Sampling) -> dict[str, float]:
        """Drawn values for the selected live numbers, by address."""

        return self.values(lambda a: counter_uniform(seed, "assumption", a.address), sampling)

    def values(self, u: Callable[[Assumption], float], sampling: Sampling) -> dict[str, float]:
        """Values at the points ``u(item)`` in [0, 1] of the selected live numbers'
        ranges (0: the low end, 1: the high end), with linked sets kept and caps applied."""

        chosen = [a for a in self.live() if sampling.selects(a)]
        values = {a.address: _point(a, u(a), sampling.spread) for a in chosen}
        linked: dict[str, list[Assumption]] = {}
        for a in chosen:
            if a.linked is not None:
                linked.setdefault(a.linked, []).append(a)
        for members in linked.values():
            # the drawn members keep their total, so fixed members (and the set) keep theirs
            total = sum(m.value for m in members)
            if len(members) == 1:
                values[members[0].address] = members[0].value
                continue
            if len(members) == 2:
                rest = total - values[members[0].address]
                if 0.0 < rest < total:
                    values[members[1].address] = _round(rest)
                    continue
            drawn = sum(values[m.address] for m in members)
            for m in members:
                values[m.address] = _round(values[m.address] * total / drawn)
        capped: dict[str, list[Assumption]] = {}
        for a in self.items:
            if a.capped is not None:
                capped.setdefault(a.capped, []).append(a)
        for members in capped.values():
            if not any(m.address in values for m in members):
                continue
            total = sum(values.get(m.address, m.value) for m in members)
            if total > 1.0:
                for m in members:
                    values[m.address] = _round(values.get(m.address, m.value) / total)
        return values

    def setting(self, address: str, value: float) -> dict[str, float]:
        """The values that set one number: the other members of its linked set are
        rescaled to keep the set's total, and its region's shares are capped at 1."""

        item = next(a for a in self.items if a.address == address)
        values = {address: _round(value)}
        if item.linked is not None:
            others = [
                a
                for a in self.items
                if a.linked == item.linked and a.address != address and a.drawable
            ]
            rest = sum(a.value for a in others)
            if rest > 0.0:
                target = rest + item.value - value
                for a in others:
                    values[a.address] = _round(a.value * target / rest)
        if item.capped is not None:
            members = [a for a in self.items if a.capped == item.capped]
            total = sum(values.get(a.address, a.value) for a in members)
            if total > 1.0:
                for a in members:
                    values[a.address] = _round(values.get(a.address, a.value) / total)
        return values


def _round(value: float) -> float:
    """*value* to the significant digits a draw keeps."""

    return float(f"{value:.{_SIG}g}")


def _point(item: Assumption, u: float, spread: float) -> float:
    """The value at point *u* of the item's range on its natural scale, the range shrunk
    toward the catalog's value by *spread*."""

    assert item.low is not None and item.high is not None
    kind = item.kind if item.kind != "intercept" else "additive"
    sign = -1.0 if item.kind in LOG_KINDS and item.value < 0 else 1.0
    t_value = _to_scale(kind, item.value)
    t_low, t_high = _to_scale(kind, item.low), _to_scale(kind, item.high)
    if sign < 0:
        t_low, t_high = t_high, t_low  # on the log scale of |v| the low end is the smaller |v|
    t_low = t_value + spread * (t_low - t_value)
    t_high = t_value + spread * (t_high - t_value)
    return _round(_from_scale(kind, t_low + u * (t_high - t_low), sign))


def plan(
    catalog: Catalog,
    register: Register | None = None,
    numbers: Sequence[Number] | None = None,
    playable_tier: int | None = None,
) -> Plan:
    """Resolve every number of *catalog* (or *numbers*) against the register, with each
    number's liveness in *catalog* at *playable_tier*."""

    register = register or load_register()
    numbers = numbers if numbers is not None else numbers_of(catalog)
    items = tuple(register.resolve(n) for n in numbers)
    dead = {
        event_id: _dead_paths(catalog, event_id)
        for event_id in {a.address.partition(":")[2].partition("/")[0] for a in items}
        if event_id in catalog.events
    }
    status = {a.address: _liveness(catalog, a.address, playable_tier, dead) for a in items}
    return Plan(items, status)


_PLANS: dict[tuple[int, int | None], tuple[Catalog, Plan]] = {}


def plan_for(catalog: Catalog, playable_tier: int | None = None) -> Plan:
    """:func:`plan` with the package register, cached per catalog and tier."""

    key = (id(catalog), playable_tier)
    cached = _PLANS.get(key)
    if cached is None or cached[0] is not catalog:
        if len(_PLANS) > 16:
            _PLANS.clear()
        cached = _PLANS[key] = (catalog, plan(catalog, playable_tier=playable_tier))
    return cached[1]


@cache
def base_inventory() -> tuple[Number, ...]:
    """The inventory of the package data (read once)."""

    return tuple(inventory())


def numbers_of(catalog: Catalog) -> list[Number]:
    """Every number in *catalog*'s own source documents (so a catalog with other events,
    such as a test fixture, gets its own), with lines and comments from the package data's
    files where the address is found there."""

    raw = catalog.raw
    if raw is None:
        raise RegisterError("this catalog keeps no source documents")
    files = {n.address: n for n in base_inventory()}
    found: list[tuple[str, float, Mapping[str, Any]]] = []
    for doc in GLOBAL_FILES:
        for keys, value, context in _walk_raw(raw.document(doc), (), {}):
            if doc == "powers" and keys[0] == "dynamics":
                continue
            found.append((f"{doc}:{'/'.join(keys)}", value, context))
    overrides = raw.powers.get("dynamics", {})
    found += [(f"dynamics:{k}", float(overrides.get(k, v)), {}) for k, v in DEFAULTS.items()]
    for _, event in raw.events:
        for keys, value, context in _walk_raw(event, (), {}):
            found.append((f"event:{event['id']}/{'/'.join(keys)}", value, context))
    result = []
    for address, value, context in found:
        known = files.get(address)
        if known is not None:
            result.append(Number(address, value, known.file, known.line, known.comment, context))
        else:
            result.append(Number(address, value, "", 0, "", context))
    return result


def _walk_raw(
    value: Any, keys: tuple[str, ...], context: Mapping[str, Any]
) -> Iterator[tuple[tuple[str, ...], float, Mapping[str, Any]]]:
    """Every number below *value*, with its path and the mapping it sits in."""

    if isinstance(value, dict):
        siblings = {
            str(k): v
            for k, v in value.items()
            if not isinstance(v, dict)
            and not (isinstance(v, list) and any(isinstance(i, (dict, list)) for i in v))
        }
        for k, v in value.items():
            yield from _walk_raw(v, (*keys, str(k)), siblings)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from _walk_raw(v, (*keys, str(i)), {})
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield keys, float(value), context


def _dead_paths(catalog: Catalog, event_id: str) -> tuple[tuple[tuple[str, ...], str], ...]:
    """Config path prefixes whose numbers never act: slots that derived payoffs fill, and
    targets of bindings with a base (the binding replaces the value)."""

    major = catalog.events[event_id].major
    if major is None:
        return ()
    result = [
        (tuple(str(k) for k in binding.at), "dead:bound")
        for binding in major.bindings
        if binding.base is not None
    ]
    if major.payoffs == "derived":
        found = derive.slots(
            major.family, major.config, major.tags, major.effort_costs, tuple(major.identities)
        )
        result += [(tuple(str(k) for k in slot), "dead:derived") for slot in found]
    return tuple(result)


def _liveness(
    catalog: Catalog,
    address: str,
    playable_tier: int | None,
    dead: Mapping[str, tuple[tuple[tuple[str, ...], str], ...]],
) -> str:
    """Whether the number at *address* acts in a campaign at *playable_tier*: ``live``, or
    the reason it is dead (a derived slot, a binding target, an event on rails).
    """

    doc, keys = split_address(address)
    if not doc.startswith("event:") or keys[0] != "model":
        return "live"
    event_id = doc.removeprefix("event:")
    event = catalog.events.get(event_id)
    if event is None or event.major is None:
        return "live"
    if keys[1] == "config":
        for prefix, reason in dead.get(event_id, ()):
            if keys[2 : 2 + len(prefix)] == prefix:
                return reason
    if (
        playable_tier is not None
        and event.major.tier > playable_tier
        and keys[1] in ("config", "bindings")
    ):
        return "dead:rails"
    return "live"


def code_value(address: str) -> float:
    """The value of a ``code:<module>.<NAME>[/<key>...]`` or ``engine:<field>`` constant."""

    doc, _, path = address.partition(":")
    if doc == "engine":
        from ..engine.core import EngineConfig

        return float(getattr(EngineConfig(), path))
    if doc != "code":
        raise RegisterError(f"{address}: not a code constant")
    head, *keys = path.split("/")
    module_name, _, name = head.rpartition(".")
    value: Any = getattr(importlib.import_module(f"cold_war_sim.campaign.{module_name}"), name)
    for key in keys:
        value = value[int(key)] if isinstance(value, (list, tuple)) else value[key]
    return float(value)


def check(
    catalog: Catalog,
    register: Register | None = None,
    numbers: Sequence[Number] | None = None,
    *,
    extremes: bool = True,
) -> list[str]:
    """Problems with the register: uncovered numbers, stale entries, invalid ranges, and
    (with *extremes*) catalogs that fail to compile with every live number at either end
    of its range."""

    from ..catalog.core import CatalogError, compile_catalog

    register = register or load_register()
    numbers = numbers if numbers is not None else numbers_of(catalog)
    problems: list[str] = []
    by_address = {n.address: n for n in numbers}
    items = []
    for number in numbers:
        try:
            items.append(register.resolve(number))
        except RegisterError as exc:
            problems.append(str(exc))
    for address, entry in register.entries.items():
        try:
            if address.startswith(("code:", "engine:")):
                actual = code_value(address)
            elif address in by_address:
                actual = by_address[address].value
            else:
                problems.append(f"entry {address}: no such number in the data")
                continue
        except (AttributeError, ImportError, KeyError, IndexError, ValueError) as exc:
            problems.append(f"entry {address}: cannot read it ({exc})")
            continue
        if not math.isclose(actual, entry.value, rel_tol=1e-9, abs_tol=1e-12):
            problems.append(f"entry {address}: the data holds {actual}, the entry {entry.value}")
    for item in items:
        problems.extend(_range_problems(item))
    if extremes and catalog.raw is not None and not problems:
        current = plan(catalog, register, numbers)
        for end in (0.0, 1.0):
            values = current.values(_constant(end), Sampling())
            variant = catalog.variant(values)
            assert variant.raw is not None
            try:
                compile_catalog(variant.raw)
            except (CatalogError, ValueError, KeyError, TypeError) as exc:
                problems.append(
                    f"every live number at the {'low' if end == 0 else 'high'} end: {exc}"
                )
    return problems


def _constant(u: float) -> Callable[[Assumption], float]:
    """A point function that puts every number at the same position *u* of its range."""

    return lambda _item: u


def _range_problems(item: Assumption) -> list[str]:
    """What is wrong with a number's range: it must contain the value and keep its sign or
    its bounds.
    """

    if item.low is None or item.high is None:
        return []
    low, high, value = item.low, item.high, item.value
    where = f"{item.address} ({item.kind}, {value})"
    if not low - 1e-12 <= value <= high + 1e-12:
        return [f"{where}: the range [{low}, {high}] does not contain the value"]
    if item.kind in LOG_KINDS and (low * value <= 0 or high * value <= 0):
        return [f"{where}: a log-scale range may not reach or cross zero"]
    if item.kind in ("probability", "share") and not (low > 0.0 and high < 1.0):
        return [f"{where}: the range must lie inside (0, 1)"]
    if item.kind == "relation" and not (low > -1.0 and high < 1.0):
        return [f"{where}: the range must lie inside (-1, 1)"]
    bounds = _field_bounds(item.address)
    if bounds is not None and not (bounds[0] <= low and high <= bounds[1]):
        return [f"{where}: the range [{low}, {high}] leaves the field's bounds {bounds}"]
    return []


def _field_bounds(address: str) -> tuple[float, float] | None:
    """The declared bounds of a world-state field that *address* sets, if it sets one."""

    doc, keys = split_address(address)
    if doc == "powers" and len(keys) == 4 and keys[0] == "powers" and keys[2] == "initial":
        return POWER_FIELDS.get(keys[3])
    if doc == "regions" and len(keys) == 3 and keys[2] in REGION_FIELDS:
        return REGION_FIELDS[keys[2]]
    if doc == "regions" and keys[-1] == "base_stability":
        return (0.0, 1.0)
    return None
