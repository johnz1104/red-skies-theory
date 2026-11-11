"""Inventory model numbers and assign their ranges, sources and sampling rules."""

from __future__ import annotations

import fnmatch
import math
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from ...records import Record
from ..catalog.core import data_root, split_address
from ..world.dynamics import DEFAULTS
from ..world.state import POWER_FIELDS, REGION_FIELDS

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


class RangeRule(Record):
    """The range of a basis: magnitudes from ``value / factor`` to ``value x factor``,
    probabilities and shares ``logit`` either side on the logit scale, an intercept a share
    ``intercept`` of its clamp's width.
    """

    def __init__(
        self,
        factor: float,
        logit: float,
        intercept: float,
    ) -> None:
        self.factor = factor  # magnitudes: [v / factor, v * factor]
        self.logit = logit  # probabilities and shares: logit(v) +- logit
        self.intercept = intercept  # binding bases: +- this share of the clamp's width


class ClassRule(Record):
    """A class of numbers, by address glob (and optionally a sibling path): its group, kind,
    basis, and whether it is linked or capped with its siblings.
    """

    def __init__(
        self,
        match: str,
        regex: re.Pattern[str],
        group: str,
        kind: str = "auto",
        basis: str = "judgment",
        fixed: str | None = None,
        path: str | None = None,
        linked: int | None = None,
        capped: int | None = None,
        why: str = "",
    ) -> None:
        self.match = match
        self.regex = regex
        self.group = group
        self.kind = kind
        self.basis = basis
        self.fixed = fixed
        self.path = path  # glob on the sibling `path` (effects, conditions)
        self.linked = linked  # the address segment (from the end) that varies in a set
        self.capped = capped  # likewise, for shares whose sum may not pass 1
        self.why = why

    @property
    def label(self) -> str:
        """The rule's glob, with its path condition."""

        return self.match + (f" (path {self.path})" if self.path else "")

    def matches(self, number: Number) -> bool:
        """Whether *number* belongs to the class."""

        if not self.regex.fullmatch(number.address):
            return False
        if self.path is not None:
            sibling = number.context.get("path")
            return isinstance(sibling, str) and fnmatch.fnmatchcase(sibling, self.path)
        return True


class Entry(Record):
    """One number's own entry: its checked value and anything it overrides (basis, kind,
    range, fixed reason), with its source.
    """

    def __init__(
        self,
        address: str,
        value: float,
        basis: str | None = None,
        kind: str | None = None,
        range: tuple[float, float] | None = None,
        factor: float | None = None,
        logit: float | None = None,
        fixed: str | None = None,
        sample: bool = True,
        source: str = "",
        note: str = "",
        placeholder: bool = False,
    ) -> None:
        self.address = address
        self.value = value
        self.basis = basis
        self.kind = kind
        self.range = range
        self.factor = factor
        self.logit = logit
        self.fixed = fixed
        self.sample = sample  # False: listed with a range, never drawn
        self.source = source
        self.note = note
        self.placeholder = placeholder


class Assumption(Record):
    """A number with everything the register says about it."""

    def __init__(
        self,
        address: str,
        value: float,
        kind: str,
        basis: str,
        group: str,
        fixed: str | None,
        low: float | None,
        high: float | None,
        rule: str,
        curated: bool,
        sample: bool,
        source: str,
        note: str,
        placeholder: bool,
        linked: str | None,
        capped: str | None,
        file: str,
        line: int,
    ) -> None:
        self.address = address
        self.value = value
        self.kind = kind
        self.basis = basis
        self.group = group
        self.fixed = fixed
        self.low = low
        self.high = high
        self.rule = rule  # the class rule's pattern
        self.curated = curated  # an entry describes it
        self.sample = sample
        self.source = source
        self.note = note
        self.placeholder = placeholder
        self.linked = linked  # the key of its linked set
        self.capped = capped  # the key of its capped set
        self.file = file
        self.line = line

    @property
    def drawable(self) -> bool:
        """Whether a draw may move the number: not fixed, sampled, and with a range."""

        return self.fixed is None and self.sample and self.low is not None


class Register(Record):
    """The register file: the ranges by basis, the class rules in order, and the entries."""

    def __init__(
        self,
        ranges: Mapping[str, RangeRule],
        classes: tuple[ClassRule, ...],
        entries: Mapping[str, Entry],
    ) -> None:
        self.ranges = ranges
        self.classes = classes
        self.entries = entries

    def rule_for(self, number: Number) -> ClassRule | None:
        """The first class rule that covers *number*."""

        return next((rule for rule in self.classes if rule.matches(number)), None)

    def resolve(self, number: Number) -> Assumption:
        """*number* as an assumption: its class, kind, basis and range, with its entry's overrides."""

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
        """The range of *number*: its entry's, or its basis's rule on its kind's scale."""

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
    """Whether a comment marks the number as a placeholder."""

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
    """The fixed reason of a number its kind's scale cannot move: a zero magnitude, or a
    probability or relation at a bound.
    """

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
    """*value* on its kind's scale: log of the magnitude, or logit of the probability."""

    if kind in LOG_KINDS:
        return math.log(abs(value))
    if kind == "relation":
        value = (value + 1.0) / 2.0
    if kind in LOGIT_KINDS:
        return math.log(value / (1.0 - value))
    return value


def _from_scale(kind: str, t: float, sign: float = 1.0) -> float:
    """The inverse of :func:`_to_scale`."""

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
    """The register of a parsed register document."""

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
    """Raise unless the kind, basis and fixed reason are known words."""

    if kind not in KINDS:
        raise RegisterError(f"{where}: unknown kind {kind!r}")
    if basis not in BASES:
        raise RegisterError(f"{where}: unknown basis {basis!r}")
    if fixed is not None and fixed not in FIXED:
        raise RegisterError(f"{where}: unknown fixed reason {fixed!r}")


@cache
def load_register(root: Path | None = None) -> Register:
    """The register file of the data directory *root*."""

    root = root or data_root()
    with (root / REGISTER_FILE).open("r", encoding="utf-8") as handle:
        return parse_register(yaml.safe_load(handle) or {})


def _set_key(keys: list[str], position: int | None) -> str | None:
    """The address with the segment at *position* (from the end) replaced by ``*``."""

    if position is None:
        return None
    index = len(keys) + position
    return "/".join([*keys[:index], "*", *keys[index + 1 :]])


class Number(Record):
    """One number found in the campaign data."""

    def __init__(
        self,
        address: str,
        value: float,
        file: str,
        line: int,
        comment: str,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        self.address = address
        self.value = value
        self.file = file
        self.line = line  # 1-based
        self.comment = (
            comment  # the comment on its line, or the nearest comment block above its item
        )
        self.context: Mapping[str, Any] = (
            {} if context is None else context
        )  # the mapping that holds it

    @property
    def event_id(self) -> str | None:
        """The event an ``event:`` address belongs to."""

        doc = self.address.partition(":")[0]
        if doc != "event":
            return None
        return self.address.partition(":")[2].partition("/")[0]

    @property
    def keys(self) -> tuple[str, ...]:
        """The path of the address inside its document."""

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
    """The value node under *key* of a YAML mapping node."""

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
    """The comment at the end of a YAML line (outside quotes), if any."""

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
    """The comment lines directly above *line*, joined."""

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
    """The coefficients of ``dynamics.DEFAULTS``, with their comments in ``dynamics.py`` and
    any override in ``powers.yaml``.
    """

    overrides = (yaml.safe_load((root / "powers.yaml").read_text(encoding="utf-8")) or {}).get(
        "dynamics", {}
    )
    lines = (
        Path(__file__)
        .parents[1]
        .joinpath("world", "dynamics.py")
        .read_text(encoding="utf-8")
        .splitlines()
    )
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
