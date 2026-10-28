"""The assumptions register: every hand-set number, its basis, a plausible range, and draws.

User decision 7 (2026-10-14): the register comes first. It lists every hand-set number with
its value, its source or judgment, a plausible range and what it affects; episodes can
sample within the ranges; sweeps report which findings are robust and which are fragile.

* **Addresses** name a document and a ``/``-separated path (:func:`catalog.split_address`):
  ``goals:kappa``, ``powers:powers/USA/initial/economy``, ``dynamics:relations_step`` (a
  ``dynamics.DEFAULTS`` coefficient, overridden in powers.yaml), ``event:<id>/<path>``.
  ``code:<module>.<NAME>[/<key>]`` and ``engine:<field>`` name constants left in Python.
* **The register file** (``data/assumptions.yaml``) has ordered class rules, which cover
  every number (first match wins), and entries, which override single numbers with a
  source. Each number gets a basis (data, literature, calibrated or judgment), a kind
  (which fixes the scale its range lives on), a group, and a range, or a reason it is fixed
  (structural counts, historical plans, binding clamps, exact bounds, zeros, the pre-goals
  score only, numerical method).
* **Ranges** follow the basis (user decision 2026-10-17): judgment x0.5 to x2 for
  magnitudes (sign kept) and logit +-1 for probabilities and shares; literature or
  calibrated x0.75 to x1.33 and logit +-0.5; a binding's intercept moves by a share of its
  clamp's width. The factors live in the register file.
* **Liveness:** a hand payoff that derived payoffs replace (``derive.slots``) is dead, and
  so are the config and bindings of an event that runs on rails (above ``playable_tier``).
* **A draw** sets each live number to a uniform point of its range on its natural scale
  (log for magnitudes, logit for probabilities and shares), from a counter-based uniform of
  the seed and the address, so adding a number never moves another number's draw. Linked
  numbers (a type's level weights, a belief pair, a goal weight row, a posture's shares)
  keep their total: a pair draws its first member and the second takes the rest, a larger
  set is renormalized. A region's influence shares are scaled down if they pass 1.
"""

from __future__ import annotations

import fnmatch
import importlib
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from . import derive
from .catalog import Catalog, data_root, split_address
from .dynamics import DEFAULTS
from .rng import counter_uniform
from .state import POWER_FIELDS, REGION_FIELDS

BASES = ("data", "literature", "calibrated", "judgment")
KINDS = ("magnitude", "signed", "probability", "share", "relation", "intercept", "auto")
FIXED = ("structural", "historical", "bound", "zero", "score", "method", "code")
LOG_KINDS = ("magnitude", "signed")
LOGIT_KINDS = ("probability", "share", "relation")
REGISTER_FILE = "assumptions.yaml"
GLOBAL_FILES = {
    "leaders": "leaders.yaml",
    "powers": "powers.yaml",
    "regions": "regions.yaml",
    "initial": "initial_1945.yaml",
    "strategy": "historical_strategy.yaml",
    "goals": "goals.yaml",
    "contract": "contract.yaml",
    "generator": "generator.yaml",
}
_INT = "tag:yaml.org,2002:int"
_FLOAT = "tag:yaml.org,2002:float"
_SIG = 9  # significant digits kept in a draw


class RegisterError(ValueError):
    """Raised when the register is malformed or does not match the data."""


# ------------------------------------------------------------------ inventory
@dataclass(frozen=True)
class Number:
    """One number found in the campaign data."""

    address: str
    value: float
    file: str
    line: int  # 1-based
    comment: str  # the comment on its line, or the nearest comment block above its item
    context: Mapping[str, Any] = field(default_factory=dict)  # the mapping that holds it

    @property
    def event_id(self) -> str | None:
        doc = self.address.partition(":")[0]
        if doc != "event":
            return None
        return self.address.partition(":")[2].partition("/")[0]

    @property
    def keys(self) -> tuple[str, ...]:
        return split_address(self.address)[1]


def inventory(root: Path | None = None, event_files: Iterable[Path] | None = None) -> list[Number]:
    """Every number in the data files under *root* (default: package data), with its line
    and nearby comment, plus every ``dynamics.DEFAULTS`` coefficient (as overridden in
    powers.yaml). The order is stable: files, then document order."""

    root = root or data_root()
    numbers: list[Number] = []
    for doc, name in GLOBAL_FILES.items():
        path = root / name
        text = path.read_text(encoding="utf-8")
        node = yaml.compose(text)
        if node is None:
            continue
        lines = text.splitlines()
        for keys, value, line, chain, context in _walk(node, (), (), {}):
            if doc == "powers" and keys[0] == "dynamics":
                continue  # listed below with the defaults
            address = f"{doc}:{'/'.join(keys)}"
            comment = _comment(lines, line, chain)
            numbers.append(Number(address, value, name, line + 1, comment, context))
    numbers.extend(_dynamics_numbers(root))
    files = (
        sorted(event_files) if event_files is not None else sorted((root / "events").glob("*.yaml"))
    )
    for path in files:
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        events = _child(yaml.compose(text), "events")
        if events is None:
            continue
        for item in events.value:
            event_id = _child(item, "id").value
            for keys, value, line, chain, context in _walk(item, (), (), {}):
                address = f"event:{event_id}/{'/'.join(keys)}"
                comment = _comment(lines, line, chain)
                numbers.append(Number(address, value, path.name, line + 1, comment, context))
    return numbers


def _child(node: Any, key: str) -> Any:
    if isinstance(node, yaml.MappingNode):
        for k, v in node.value:
            if k.value == key:
                return v
    return None


def _scalar(node: Any) -> Any:
    """A scalar's value, or a flat sequence's values (a clamp); None for mappings."""

    if isinstance(node, yaml.ScalarNode):
        return yaml.safe_load(node.value) if node.tag in (_INT, _FLOAT) else node.value
    if isinstance(node, yaml.SequenceNode) and all(
        isinstance(item, yaml.ScalarNode) for item in node.value
    ):
        return [_scalar(item) for item in node.value]
    return None


def _walk(
    node: Any, keys: tuple[str, ...], chain: tuple[int, ...], context: Mapping[str, Any]
) -> Iterator[tuple[tuple[str, ...], float, int, tuple[int, ...], Mapping[str, Any]]]:
    """(keys, value, 0-based line, lines of the enclosing items, sibling scalars) for each
    numeric scalar."""

    if isinstance(node, yaml.MappingNode):
        siblings = {str(k.value): _scalar(v) for k, v in node.value}
        for k, v in node.value:
            yield from _walk(v, (*keys, str(k.value)), (*chain, k.start_mark.line), siblings)
    elif isinstance(node, yaml.SequenceNode):
        for i, v in enumerate(node.value):
            yield from _walk(v, (*keys, str(i)), (*chain, v.start_mark.line), {})
    elif isinstance(node, yaml.ScalarNode) and node.tag in (_INT, _FLOAT):
        value = yaml.safe_load(node.value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            yield keys, float(value), node.start_mark.line, chain, context


def _inline(line: str) -> str:
    quote = ""
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "'\"":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            return line[i + 1 :].strip()
    return ""


def _block_above(lines: Sequence[str], line: int) -> str:
    block: list[str] = []
    i = line - 1
    while i >= 0 and lines[i].lstrip().startswith("#"):
        block.append(lines[i].lstrip()[1:].strip())
        i -= 1
    return " ".join(reversed(block))


def _comment(lines: Sequence[str], line: int, chain: tuple[int, ...]) -> str:
    """The number's inline comment, else the comment block just above its line, else the
    block above one of its three nearest enclosing items (never an event's own header)."""

    text = _inline(lines[line]) or _block_above(lines, line)
    if not text:
        for start in reversed(chain[1:][-3:]):
            if start == line:
                continue
            text = _inline(lines[start]) or _block_above(lines, start)
            if text:
                break
    return text[:400]


def _dynamics_numbers(root: Path) -> list[Number]:
    overrides = (yaml.safe_load((root / "powers.yaml").read_text(encoding="utf-8")) or {}).get(
        "dynamics", {}
    )
    lines = Path(__file__).with_name("dynamics.py").read_text(encoding="utf-8").splitlines()
    result = []
    for key, default in DEFAULTS.items():
        line = next(i for i, text in enumerate(lines) if text.strip().startswith(f'"{key}":'))
        value = float(overrides.get(key, default))
        note = _inline(lines[line]) or _block_above(lines, line)
        if key in overrides:
            note = f"powers.yaml overrides the default {default}. {note}".strip()
        result.append(Number(f"dynamics:{key}", value, "dynamics.py", line + 1, note[:400]))
    return result


def pattern_of(address: str) -> str:
    """The address with event ids and list indices replaced by ``*`` (for grouping)."""

    doc, _, path = address.partition(":")
    sep = ":"
    if doc == "event":
        path, doc, sep = path.partition("/")[2], "event:*", "/"
    return doc + sep + "/".join("*" if k.isdigit() else k for k in path.split("/"))


# ------------------------------------------------------------------- register
@dataclass(frozen=True)
class RangeRule:
    factor: float  # magnitudes: [v / factor, v * factor]
    logit: float  # probabilities and shares: logit(v) +- logit
    intercept: float  # binding bases: +- this share of the clamp's width


@dataclass(frozen=True)
class ClassRule:
    match: str
    regex: re.Pattern[str]
    group: str
    kind: str = "auto"
    basis: str = "judgment"
    fixed: str | None = None
    path: str | None = None  # glob on the sibling `path` (effects, conditions)
    linked: int | None = None  # the address segment (from the end) that varies in a set
    capped: int | None = None  # likewise, for shares whose sum may not pass 1
    why: str = ""

    @property
    def label(self) -> str:
        return self.match + (f" (path {self.path})" if self.path else "")

    def matches(self, number: Number) -> bool:
        if not self.regex.fullmatch(number.address):
            return False
        if self.path is not None:
            sibling = number.context.get("path")
            return isinstance(sibling, str) and fnmatch.fnmatchcase(sibling, self.path)
        return True


@dataclass(frozen=True)
class Entry:
    address: str
    value: float
    basis: str | None = None
    kind: str | None = None
    range: tuple[float, float] | None = None
    factor: float | None = None
    logit: float | None = None
    fixed: str | None = None
    sample: bool = True  # False: listed with a range, never drawn
    source: str = ""
    note: str = ""
    placeholder: bool = False


@dataclass(frozen=True)
class Assumption:
    """A number with everything the register says about it."""

    address: str
    value: float
    kind: str
    basis: str
    group: str
    fixed: str | None
    low: float | None
    high: float | None
    rule: str  # the class rule's pattern
    curated: bool  # an entry describes it
    sample: bool
    source: str
    note: str
    placeholder: bool
    linked: str | None  # the key of its linked set
    capped: str | None  # the key of its capped set
    file: str
    line: int

    @property
    def drawable(self) -> bool:
        return self.fixed is None and self.sample and self.low is not None


@dataclass(frozen=True)
class Register:
    ranges: Mapping[str, RangeRule]
    classes: tuple[ClassRule, ...]
    entries: Mapping[str, Entry]

    def rule_for(self, number: Number) -> ClassRule | None:
        return next((rule for rule in self.classes if rule.matches(number)), None)

    def resolve(self, number: Number) -> Assumption:
        rule = self.rule_for(number)
        if rule is None:
            raise RegisterError(f"no class covers {number.address}")
        entry = self.entries.get(number.address)
        kind = (entry.kind if entry and entry.kind else None) or rule.kind
        if kind == "auto":
            kind = _auto_kind(number)
        basis = (entry.basis if entry and entry.basis else None) or rule.basis
        fixed = (entry.fixed if entry and entry.fixed else None) or rule.fixed
        if fixed is None:
            fixed = _bound_fixed(kind, number.value)
        low = high = None
        if fixed is None:
            low, high = self._range(number, kind, basis, entry)
        keys = number.address.split("/")
        linked = _set_key(keys, rule.linked)
        capped = _set_key(keys, rule.capped)
        note = entry.note if entry and entry.note else number.comment
        placeholder = bool(entry.placeholder) if entry else _says_placeholder(number.comment)
        return Assumption(
            address=number.address,
            value=number.value,
            kind=kind,
            basis=basis,
            group=rule.group,
            fixed=fixed,
            low=low,
            high=high,
            rule=rule.label,
            curated=entry is not None,
            sample=entry.sample if entry else True,
            source=entry.source if entry else "",
            note=note,
            placeholder=placeholder,
            linked=linked,
            capped=capped,
            file=number.file,
            line=number.line,
        )

    def _range(
        self, number: Number, kind: str, basis: str, entry: Entry | None
    ) -> tuple[float, float]:
        value = number.value
        if entry is not None and entry.range is not None:
            return entry.range
        rule = self.ranges[basis]
        factor = entry.factor if entry is not None and entry.factor else rule.factor
        width = entry.logit if entry is not None and entry.logit else rule.logit
        if kind == "intercept":
            clamp = number.context.get("clamp")
            if clamp is None:
                kind = "signed"
            else:
                half = rule.intercept * (float(clamp[1]) - float(clamp[0]))
                return value - half, value + half
        if kind in LOG_KINDS:
            a, b = value / factor, value * factor
            return (a, b) if value > 0 else (b, a)
        t = _to_scale(kind, value)
        return _from_scale(kind, t - width), _from_scale(kind, t + width)


def _says_placeholder(comment: str) -> bool:
    return "placeholder" in comment.lower()


def _auto_kind(number: Number) -> str:
    """A threshold's or a set value's kind follows the path it compares or sets."""

    path = number.context.get("path")
    if not isinstance(path, str):
        return "signed"
    field_name = path.split(".")[-1]
    parts = path.split(".")
    if "relations" in parts:
        return "relation"
    if "influence" in parts or field_name in REGION_FIELDS or field_name == "tension":
        return "share"
    bounds = POWER_FIELDS.get(field_name)
    if bounds is not None and bounds == (0.0, 1.0):
        return "share"
    if field_name in ("arms_control",):
        return "share"
    return "magnitude" if number.value > 0 else "signed"


def _bound_fixed(kind: str, value: float) -> str | None:
    if kind in LOG_KINDS and value == 0.0:
        return "zero"
    if kind in ("probability", "share") and not 0.0 < value < 1.0:
        return "bound"
    if kind == "relation" and not -1.0 < value < 1.0:
        return "bound"
    if kind == "magnitude" and value < 0.0:
        raise RegisterError(f"a magnitude must be positive, got {value}")
    return None


def _to_scale(kind: str, value: float) -> float:
    if kind in LOG_KINDS:
        return math.log(abs(value))
    if kind == "relation":
        value = (value + 1.0) / 2.0
    if kind in LOGIT_KINDS:
        return math.log(value / (1.0 - value))
    return value


def _from_scale(kind: str, t: float, sign: float = 1.0) -> float:
    if kind in LOG_KINDS:
        return sign * math.exp(t)
    if kind in LOGIT_KINDS:
        p = 1.0 / (1.0 + math.exp(-t))
        return 2.0 * p - 1.0 if kind == "relation" else p
    return t


def _glob(pattern: str) -> re.Pattern[str]:
    """``*`` matches within one segment; ``**`` matches any run of segments, and ``/**/``
    also matches a single ``/``."""

    parts = []
    for i, chunk in enumerate(pattern.replace("/**/", "/\0").split("\0")):
        if i:
            parts.append("(?:.*/)?")
        for j, piece in enumerate(chunk.split("**")):
            if j:
                parts.append(".*")
            parts.append("[^/]*".join(re.escape(p) for p in piece.split("*")))
    return re.compile("".join(parts))


def parse_register(doc: Mapping[str, Any]) -> Register:
    ranges = {}
    for basis in BASES:
        spec = doc.get("ranges", {}).get(basis)
        if spec is None:
            raise RegisterError(f"ranges: missing basis {basis!r}")
        ranges[basis] = RangeRule(
            float(spec["factor"]), float(spec["logit"]), float(spec["intercept"])
        )
    classes = []
    for item in doc.get("classes", []):
        unknown = set(item) - {
            "match",
            "group",
            "kind",
            "basis",
            "fixed",
            "path",
            "linked",
            "capped",
            "why",
        }
        if unknown:
            raise RegisterError(f"class {item.get('match')!r}: unknown keys {sorted(unknown)}")
        kind = item.get("kind", "auto")
        basis = item.get("basis", "judgment")
        fixed = item.get("fixed")
        _check_terms(item["match"], kind, basis, fixed)
        classes.append(
            ClassRule(
                match=item["match"],
                regex=_glob(item["match"]),
                group=str(item["group"]),
                kind=kind,
                basis=basis,
                fixed=fixed,
                path=item.get("path"),
                linked=item.get("linked"),
                capped=item.get("capped"),
                why=str(item.get("why", "")),
            )
        )
    entries = {}
    for address, item in (doc.get("entries") or {}).items():
        unknown = set(item) - {
            "value", "basis", "kind", "range", "factor", "logit", "fixed", "sample",
            "source", "note", "placeholder",
        }  # fmt: skip
        if unknown:
            raise RegisterError(f"entry {address}: unknown keys {sorted(unknown)}")
        if "value" not in item:
            raise RegisterError(f"entry {address}: a value (the check) is required")
        _check_terms(
            address, item.get("kind", "auto"), item.get("basis", "judgment"), item.get("fixed")
        )
        span = item.get("range")
        entries[address] = Entry(
            address=address,
            value=float(item["value"]),
            basis=item.get("basis"),
            kind=item.get("kind"),
            range=(float(span[0]), float(span[1])) if span is not None else None,
            factor=float(item["factor"]) if "factor" in item else None,
            logit=float(item["logit"]) if "logit" in item else None,
            fixed=item.get("fixed"),
            sample=bool(item.get("sample", True)),
            source=str(item.get("source", "")),
            note=str(item.get("note", "")),
            placeholder=bool(item.get("placeholder", False)),
        )
    return Register(ranges, tuple(classes), entries)


def _check_terms(where: str, kind: str, basis: str, fixed: str | None) -> None:
    if kind not in KINDS:
        raise RegisterError(f"{where}: unknown kind {kind!r}")
    if basis not in BASES:
        raise RegisterError(f"{where}: unknown basis {basis!r}")
    if fixed is not None and fixed not in FIXED:
        raise RegisterError(f"{where}: unknown fixed reason {fixed!r}")


@cache
def load_register(root: Path | None = None) -> Register:
    root = root or data_root()
    with (root / REGISTER_FILE).open("r", encoding="utf-8") as handle:
        return parse_register(yaml.safe_load(handle) or {})


# ------------------------------------------------------------------ resolved
@dataclass(frozen=True)
class Sampling:
    """Draw the catalog's numbers within their ranges (off unless set on EngineConfig).

    * ``select``: groups (``goals``), events (``event:korean_war``) or address globs
      (``goals:weights/**``); empty: every live number.
    * ``spread``: the share of each range used, on its natural scale (0: the catalog's
      values).
    * ``draw``: ``None`` draws anew for each episode from its seed; an int fixes one draw
      shared by every episode (robustness runs).
    """

    select: tuple[str, ...] = ()
    spread: float = 1.0
    draw: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.spread <= 1.0:
            raise ValueError("spread must lie in [0, 1]")

    def selects(self, item: Assumption) -> bool:
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


@dataclass(frozen=True)
class Plan:
    """The register resolved against one catalog, with liveness for a playable tier."""

    items: tuple[Assumption, ...]
    status: Mapping[str, str]  # address -> live | dead:derived | dead:bound | dead:rails

    def live(self) -> list[Assumption]:
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


def _set_key(keys: list[str], position: int | None) -> str | None:
    """The address with the segment at *position* (from the end) replaced by ``*``."""

    if position is None:
        return None
    index = len(keys) + position
    return "/".join([*keys[:index], "*", *keys[index + 1 :]])


def _round(value: float) -> float:
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


# ---------------------------------------------------------------------- check
def code_value(address: str) -> float:
    """The value of a ``code:<module>.<NAME>[/<key>...]`` or ``engine:<field>`` constant."""

    doc, _, path = address.partition(":")
    if doc == "engine":
        from .engine import EngineConfig

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

    from .catalog import CatalogError, compile_catalog

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
    return lambda _item: u


def _range_problems(item: Assumption) -> list[str]:
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
    doc, keys = split_address(address)
    if doc == "powers" and len(keys) == 4 and keys[0] == "powers" and keys[2] == "initial":
        return POWER_FIELDS.get(keys[3])
    if doc == "regions" and len(keys) == 3 and keys[2] in REGION_FIELDS:
        return REGION_FIELDS[keys[2]]
    if doc == "regions" and keys[-1] == "base_stability":
        return (0.0, 1.0)
    return None


# -------------------------------------------------------------------- reports
def affects(address: str, catalog: Catalog, context: Mapping[str, Any] | None = None) -> str:
    """A short description of where a number acts."""

    context = context or {}
    if address.startswith("dynamics:"):
        return "the quarterly dynamics"
    doc, keys = split_address(address)
    if not doc.startswith("event:"):
        return f"{doc} {'/'.join(keys)}"
    event_id = doc.removeprefix("event:")
    event = catalog.events.get(event_id)
    head = event_id
    if event is not None:
        head += f" ({event.kind}" + (f", tier {event.major.tier}" if event.major else "") + ")"
    target = _target(context)
    if keys[0] == "outcomes":
        return f"{head}: outcome {keys[1]}, {target}"
    if keys[0] == "tags":
        return f"{head}: tag {keys[1]}, {target}"
    if keys[0] == "effects":
        return f"{head}: effect, {target}"
    if keys[0] == "when":
        return f"{head}: condition on {context.get('path', '?')}"
    if keys[0] == "hazard":
        return f"{head}: hazard"
    if keys[:2] == ("model", "bindings") and event is not None and event.major is not None:
        binding = event.major.bindings[int(keys[2])]
        at = ".".join(str(k) for k in binding.at)
        return f"{head}: binding of {at}" + (
            f" on {keys[4]}" if keys[3] == "terms" else f" ({keys[3]})"
        )
    return f"{head}: {'/'.join(keys[1:])}"


def _target(context: Mapping[str, Any]) -> str:
    if "region" in context and "toward" in context:
        return f"influence in {context['region']} toward {context['toward']}"
    if "path" in context:
        return str(context["path"])
    if "a" in context and "b" in context:
        return f"relations {context['a']}-{context['b']}"
    return "tension" if not context else ", ".join(sorted(context))


def render_markdown(catalog: Catalog, register: Register | None = None) -> str:
    """docs/ASSUMPTIONS.md: the register resolved against the catalog."""

    register = register or load_register()
    numbers = numbers_of(catalog)
    context = {n.address: n.context for n in numbers}
    full = plan(catalog, register, numbers)
    tier1 = plan(catalog, register, numbers, playable_tier=1)
    items = full.items
    lines = [
        "# The assumptions register",
        "",
        "Generated by `python -m cold_war_sim.campaign assumptions doc --output "
        "docs/ASSUMPTIONS.md` from `campaign/data/assumptions.yaml` and the data files; do not "
        "edit by hand. User decision 7 (2026-10-14): every hand-set number with its basis, a "
        "plausible range and what it affects, so that episodes can sample within the ranges "
        "and sweeps can tell robust findings from fragile ones.",
        "",
        "## Summary",
        "",
    ]
    drawable = [a for a in items if a.drawable and full.status[a.address] == "live"]
    dead = [a for a in items if a.fixed is None and full.status[a.address] != "live"]
    fixed = Counter(a.fixed for a in items if a.fixed is not None)
    rails = sum(1 for a in items if a.drawable and tier1.status[a.address] == "dead:rails")
    lines += [
        f"- **{len(items)} numbers** in the data, plus {len(_code_entries(register))} constants in "
        "the code.",
        f"- **{len(drawable)} are live and drawn** when sampling is on; "
        f"{rails} of them are the config and bindings of tier-2 events, which run on rails in "
        "tier-1 mode (`--playable-tier 1`) and are then not drawn.",
        f"- **{len(dead)} more are dead:** "
        + ", ".join(
            f"{n} {_DEAD[reason]}"
            for reason, n in sorted(Counter(full.status[a.address] for a in dead).items())
        )
        + ".",
        f"- **{sum(fixed.values())} are fixed:** "
        + ", ".join(f"{reason} {n}" for reason, n in sorted(fixed.items()))
        + ".",
        f"- **{sum(1 for a in items if a.curated)} have curated entries;** "
        f"{sum(1 for a in items if a.placeholder)} are marked placeholders.",
        "",
        "| Group | Numbers | Drawn | Fixed | Dead | Curated | Placeholders | Bases (drawn) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for group in sorted({a.group for a in items}):
        members = [a for a in items if a.group == group]
        drawn = [a for a in members if a.drawable and full.status[a.address] == "live"]
        bases = Counter(a.basis for a in drawn)
        lines.append(
            f"| {group} | {len(members)} | {len(drawn)} "
            f"| {sum(1 for a in members if a.fixed)} "
            f"| {sum(1 for a in members if a.fixed is None and full.status[a.address] != 'live')} "
            f"| {sum(1 for a in members if a.curated)} "
            f"| {sum(1 for a in members if a.placeholder)} "
            f"| {', '.join(f'{b} {n}' for b, n in sorted(bases.items())) or '-'} |"
        )
    lines += [
        "",
        "## Range rules",
        "",
        "A number's range follows its basis unless its entry gives one. Magnitudes run from "
        "v / factor to v * factor (sign kept); probabilities and shares over logit(v) ± logit; "
        "relations over the logit of (r + 1)/2; a binding's intercept by ± the given share of "
        "its clamp's width. A draw is uniform on that scale.",
        "",
        "| Basis | Factor | Logit | Intercept | 0.15 → | p 0.7 → |",
        "|---|---|---|---|---|---|",
    ]
    for basis, spec in register.ranges.items():
        lo, hi = (
            _from_scale("probability", _to_scale("probability", 0.7) - spec.logit),
            _from_scale("probability", _to_scale("probability", 0.7) + spec.logit),
        )
        lines.append(
            f"| {basis} | {spec.factor:g} | {spec.logit:g} | {spec.intercept:g} "
            f"| {0.15 / spec.factor:.3g} .. {0.15 * spec.factor:.3g} | {lo:.2f} .. {hi:.2f} |"
        )
    lines += [
        "",
        "## Class rules",
        "",
        "Tried in order; the first rule that matches a number covers it.",
        "",
        "| # | Match | Group | Kind | Basis | Fixed | Numbers | Why |",
        "|---|---|---|---|---|---|---|---|",
    ]
    counts = Counter(a.rule for a in items)
    for i, rule in enumerate(register.classes, 1):
        lines.append(
            f"| {i} | `{rule.label}` | {rule.group} | {rule.kind} | {rule.basis} "
            f"| {rule.fixed or ''} | {counts.get(rule.label, 0)} | {rule.why} |"
        )
    lines += ["", "## Curated entries", ""]
    curated = [a for a in items if a.curated]
    for group in sorted({a.group for a in curated}):
        lines += [
            f"### {group}",
            "",
            "| Address | Value | Range | Basis | Affects | Source and note |",
            "|---|---|---|---|---|---|",
        ]
        for a in curated:
            if a.group != group:
                continue
            span = a.fixed or ("not drawn " if not a.sample else "") + _span(a)
            text = "; ".join(t for t in (a.source, a.note) if t).replace("|", "/")
            mark = " (placeholder)" if a.placeholder else ""
            lines.append(
                f"| `{a.address}` | {a.value:g} | {span} | {a.basis}{mark} "
                f"| {affects(a.address, catalog, context[a.address])} | {text} |"
            )
        lines.append("")
    code = _code_entries(register)
    if code:
        lines += [
            "## Constants in the code",
            "",
            "| Address | Value | Range | Basis | Note |",
            "|---|---|---|---|---|",
        ]
        for entry in code:
            span = entry.fixed or (
                f"{entry.range[0]:g} .. {entry.range[1]:g}" if entry.range else "-"
            )
            lines.append(
                f"| `{entry.address}` | {entry.value:g} | {span} | {entry.basis or 'judgment'} "
                f"| {'; '.join(t for t in (entry.source, entry.note) if t)} |"
            )
        lines.append("")
    placeholders = [a for a in items if a.placeholder and not a.curated]
    lines += [
        "## Placeholders not yet curated",
        "",
        f"{len(placeholders)} numbers whose comment in the data says placeholder.",
        "",
        "| Address | Value | Range | Comment |",
        "|---|---|---|---|",
    ]
    for a in placeholders:
        span = a.fixed or _span(a)
        lines.append(f"| `{a.address}` | {a.value:g} | {span} | {a.note[:160].replace('|', '/')} |")
    lines.append("")
    return "\n".join(lines)


_DEAD = {
    "dead:derived": "hand payoffs that derived payoffs replace",
    "dead:bound": "config values that a binding replaces",
}


def _span(item: Assumption) -> str:
    if item.low is None or item.high is None:
        return "-"
    return f"{item.low:.3g} .. {item.high:.3g}"


def _code_entries(register: Register) -> list[Entry]:
    return [e for a, e in register.entries.items() if a.startswith(("code:", "engine:"))]


__all__ = [
    "Assumption",
    "Entry",
    "Number",
    "Plan",
    "Register",
    "RegisterError",
    "Sampling",
    "check",
    "code_value",
    "inventory",
    "load_register",
    "pattern_of",
    "plan",
    "plan_for",
    "render_markdown",
]
