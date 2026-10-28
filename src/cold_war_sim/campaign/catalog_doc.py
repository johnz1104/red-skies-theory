"""Render the event catalog as Markdown (``docs/EVENT_CATALOG.md`` is generated)."""

from __future__ import annotations

from . import calendar
from .catalog import Catalog, EventSpec

HEADER = """# Event catalog

Generated from `src/cold_war_sim/campaign/data/events/*.yaml` by
`python -m cold_war_sim.campaign catalog --markdown`. Do not edit by hand.

All parameters are illustrative. **Major** events are playable sub-games built
from a mechanism family; **cutscenes** apply fixed effects when their window
and condition hold; **leadership** events swap a power's leader.

**Tier** is the review priority of a major event. With `run --playable-tier N`
(`EngineConfig.playable_tier`), major events above tier N run on rails: they
still fire under their conditions but play their historical choices and chance.
A **hold-out** event's historical choices are never used when adjusting goals
(`docs/GOALS.md` section 11).

**Payoffs** are `hand` (set in the YAML) or `derived` (power roles value each
outcome by the change in their goal score; `campaign/derive.py`).
"""


def _window(event: EventSpec) -> str:
    start = calendar.fmt(event.earliest) if event.earliest is not None else "open"
    end = calendar.fmt(event.latest) if event.latest is not None else "open"
    return f"{start} to {end}"


def _date(event: EventSpec) -> str:
    return calendar.fmt(event.historical_date) if event.historical_date is not None else ""


def render_markdown(catalog: Catalog) -> str:
    counts = catalog.counts()
    lines = [
        HEADER,
        f"**{counts['major']}** major events, **{counts['cutscene']}** cutscenes, "
        f"**{counts['leadership']}** leadership changes.\n",
    ]
    by_era: dict[int, list[EventSpec]] = {}
    for event in catalog.events.values():
        by_era.setdefault(event.era, []).append(event)
    for era in sorted(by_era):
        events = sorted(by_era[era], key=lambda e: (e.historical_date or 0, e.id))
        lines.append(f"## Era {era}: {catalog.era_titles.get(era, '')}\n")
        majors = [e for e in events if e.kind == "major"]
        if majors:
            lines.append(
                "| Major event | Tier | Historical date | Window | Family | Payoffs | Cadence "
                "| Roles | Outcomes |"
            )
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for event in majors:
                major = event.major
                assert major is not None
                tier = f"{major.tier}, hold-out" if major.holdout else str(major.tier)
                lines.append(
                    f"| {event.title} (`{event.id}`) | {tier} | {_date(event)} | "
                    f"{_window(event)} | {major.family} | {major.payoffs} | {major.cadence} | "
                    f"{', '.join(major.roles)} | "
                    f"{', '.join(sorted(major.outcomes))} |"
                )
            lines.append("")
        cutscenes = [e for e in events if e.kind == "cutscene"]
        if cutscenes:
            lines.append("| Cutscene | Historical date | Window |")
            lines.append("|---|---|---|")
            for event in cutscenes:
                lines.append(
                    f"| {event.title} (`{event.id}`) | {_date(event)} | {_window(event)} |"
                )
            lines.append("")
        leaders = [e for e in events if e.kind == "leadership"]
        if leaders:
            lines.append(
                "Leadership: " + "; ".join(f"{e.title} ({_date(e)})" for e in leaders) + "\n"
            )
    return "\n".join(lines).rstrip() + "\n"
