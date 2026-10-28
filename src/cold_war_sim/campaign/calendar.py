"""Quarter arithmetic. A quarter is the integer ``year * 4 + (q - 1)``."""

from __future__ import annotations

import re

_PATTERN = re.compile(r"^(\d{4})-Q([1-4])$")


def parse(text: str) -> int:
    """Parse ``"1962-Q4"`` into an absolute quarter index."""

    match = _PATTERN.match(str(text).strip())
    if match is None:
        raise ValueError(f"invalid quarter {text!r}; expected YYYY-Qn")
    return int(match.group(1)) * 4 + int(match.group(2)) - 1


def fmt(quarter: int) -> str:
    return f"{quarter // 4}-Q{quarter % 4 + 1}"


def year(quarter: int) -> int:
    return quarter // 4


def quarter_of_year(quarter: int) -> int:
    """1 to 4."""

    return quarter % 4 + 1


START = parse("1945-Q3")
END = parse("1991-Q4")
N_QUARTERS = END - START + 1


def progress(quarter: int) -> float:
    """Fraction of the campaign elapsed, in [0, 1]."""

    return (quarter - START) / (END - START)
