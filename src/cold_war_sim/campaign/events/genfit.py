"""Fit crisis rates to observed counts by type, region group, and decade.

Simulated campaigns count both named and generated crises. Each fitting round
adjusts the base rate, then the region and decade factors, toward the counts
left after named crises are included. The adjustment ratio is raised to
``power``: 0.5 takes a square-root step, while 1 takes the full step. Crisis
types without data keep their assumed rates.

Exposure counts the region-quarters where a crisis could start: the region is
unblocked and has a hostile rival. A floor keeps each level at least at
``FLOOR * observed crises per eligible region-quarter * its exposure``, even
where named events already meet the observed count. Across a group or decade
margin, these floors add at most ``FLOOR`` times the observed total. Levels
below ``MIN_EXPOSURE`` keep their factors because increasing them would have
little effect.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from ...records import Record
from ..catalog.core import Catalog, EventTable, load_catalog
from ..decisions.policies import make_policy
from ..engine.core import CampaignEngine, EngineConfig, run_episode
from ..world import calendar
from . import generator

Cell = tuple[str, str, int]  # (template, group, decade)
Exposure = Counter[tuple[str, int]]  # (group, decade) -> castable, unblocked region-quarters
FLOOR = 0.25  # a level keeps this share of the data's crises per castable region-quarter
MIN_EXPOSURE = 1.0  # castable region-quarters a campaign below which a factor is left alone


class Counts(Record):
    """Crises counted over reference-play campaigns, by (template, region group, decade):
    those from named events, those generated, and the exposure (castable, unblocked
    region-quarters).
    """

    def __init__(
        self,
        named: Counter[Cell],
        generated: Counter[Cell],
        campaigns: int,
        exposure: Exposure | None = None,
    ) -> None:
        self.named = named
        self.generated = generated
        self.campaigns = campaigns
        self.exposure: Exposure = (
            Counter() if exposure is None else exposure
        )  # summed over the campaigns


def _cells(catalog: Catalog, engine: CampaignEngine) -> tuple[Counter[Cell], Counter[Cell]]:
    """The crises that started in one campaign, named and generated, by cell."""

    doc = catalog.raw.generator  # type: ignore[union-attr]
    named_map: Mapping[str, str] = doc["named"]
    named: Counter[Cell] = Counter()
    generated: Counter[Cell] = Counter()
    for item in engine.s.trace:
        event = item.get("start")
        if event is None:
            continue
        quarter = calendar.parse(item["q"])
        decade = generator.decade_of(doc, quarter)
        if generator.is_generated(event):
            template, region, _ = generator.parse_id(event)
            generated[(template, generator.group_of(doc, region), decade)] += 1
            continue
        major = catalog.events[event].major
        if major is None or major.region is None or major.family not in named_map:
            continue
        named[(named_map[major.family], generator.group_of(doc, major.region), decade)] += 1
    return named, generated


_WORKER: dict[str, Any] = {}


def with_generator(catalog: Catalog, generator_doc: Mapping[str, Any]) -> Catalog:
    """*catalog* with another generator document; its generated events compile from it."""

    raw = catalog.raw
    assert raw is not None
    table = EventTable(catalog.events)
    new = catalog.replace(raw=raw.replace(generator=generator_doc), events=table)
    table.owner = new
    return new


def _init(generator_doc: Mapping[str, Any]) -> None:
    """Give a worker process its catalog with the generator document under fit."""

    _WORKER["catalog"] = with_generator(load_catalog(), generator_doc)


def _campaign(seed: int) -> tuple[Counter[Cell], Counter[Cell], Exposure]:
    """One reference-play campaign with the generator on: its crisis counts and exposure."""

    catalog: Catalog = _WORKER["catalog"]
    config = EngineConfig(agents=(), playable_tier=1, npc_policy="equilibrium", generator=True)
    engine = CampaignEngine(catalog, config)
    exposure: Exposure = Counter()
    generator.count_exposure(exposure)
    try:
        run_episode(engine, make_policy("equilibrium"), seed)
    finally:
        generator.count_exposure(None)
    named, generated = _cells(catalog, engine)
    return named, generated, exposure


def count(generator_doc: Mapping[str, Any], seeds: list[int], workers: int) -> Counts:
    """The crisis counts of the campaigns *seeds*, on *workers* processes."""

    named: Counter[Cell] = Counter()
    generated: Counter[Cell] = Counter()
    exposure: Exposure = Counter()
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(generator_doc,)) as pool:
        for n, g, e in pool.map(_campaign, seeds):
            named.update(n)
            generated.update(g)
            exposure.update(e)
    return Counts(named, generated, len(seeds), exposure)


def _margin(counter: Counter[Cell], template: str, axis: int) -> Counter[Any]:
    """The counts of *template* summed onto one axis (1: region group, 2: decade)."""

    out: Counter[Any] = Counter()
    for cell, value in counter.items():
        if cell[0] == template:
            out[cell[axis]] += value
    return out


def update(doc: Mapping[str, Any], counts: Counts, power: float = 0.5) -> dict[str, Any]:
    """One round of fitting: a copy of the generator document with fitted rates."""

    new = copy.deepcopy(dict(doc))
    per = 1.0 / counts.campaigns
    exposed: dict[int, dict[Any, float]] = {1: {}, 2: {}}  # a campaign's, by group and decade
    for (group, decade), n in counts.exposure.items():
        exposed[1][group] = exposed[1].get(group, 0.0) + per * n
        exposed[2][decade] = exposed[2].get(decade, 0.0) + per * n
    total_exposure = sum(exposed[1].values())
    for template, target in doc["targets"].items():
        spec = new["templates"][template]
        gen_total = per * sum(_margin(counts.generated, template, 1).values())
        total_target = float(sum(target["decades"].values()))
        # the floor: a share of the data's crises per castable region-quarter
        intensity = FLOOR * total_target / total_exposure if total_exposure else 0.0
        wanted = per  # the generated total: the larger margin's sum of wants it can reach
        for axis, key, factors in ((1, "groups", "group_factor"), (2, "decades", "decade_factor")):
            named_m = _margin(counts.named, template, axis)
            gen_m = _margin(counts.generated, template, axis)
            margin_total = 0.0
            for level, value in target[key].items():
                level_key = int(level) if axis == 2 else level
                exposure = exposed[axis].get(level_key, 0.0)
                if exposure < MIN_EXPOSURE:
                    continue  # nothing can start here, so its factor would grow without effect
                want = max(intensity * exposure, float(value) - per * named_m[level_key])
                margin_total += want
                # a zero count reads as one crisis over the counted campaigns (its resolution),
                # or as the want itself when that is finer than the resolution
                have = max(per * gen_m[level_key], min(per, want))
                spec[factors][level] = float(spec[factors][level]) * (want / have) ** power
            wanted = max(wanted, margin_total)
            mean = sum(spec[factors].values()) / len(spec[factors])
            for level in spec[factors]:  # factors average 1; the rate carries the total
                spec[factors][level] = round(spec[factors][level] / mean, 4)
        spec["rate"] = round(float(spec["rate"]) * (wanted / max(gen_total, per)) ** power, 6)
    return new


def table(doc: Mapping[str, Any], counts: Counts) -> list[str]:
    """Per template and margin: the data's count against named + generated per campaign,
    with the generated part in brackets (the part ``FLOOR`` keeps above zero)."""

    per = 1.0 / counts.campaigns
    lines = []
    chi2 = 0.0
    for template, target in doc["targets"].items():
        named_g = _margin(counts.named, template, 1)
        gen_g = _margin(counts.generated, template, 1)
        total = float(sum(target["decades"].values()))
        sim = per * (sum(named_g.values()) + sum(gen_g.values()))
        lines.append(
            f"{template}: data {total:.0f}, simulated {sim:.1f} per campaign "
            f"(named {per * sum(named_g.values()):.1f}, generated {per * sum(gen_g.values()):.1f})"
        )
        for axis, key in ((1, "groups"), (2, "decades")):
            named_m = _margin(counts.named, template, axis)
            gen_m = _margin(counts.generated, template, axis)
            parts = []
            for level, value in target[key].items():
                level_key = int(level) if axis == 2 else level
                s = per * (named_m[level_key] + gen_m[level_key])
                parts.append(f"{level} {value}/{s:.1f} [{per * gen_m[level_key]:.2f}]")
                if axis == 1:
                    chi2 += (float(value) - s) ** 2 / max(float(value), 1.0)
            lines.append(f"  {key}: " + ", ".join(parts))
    lines.append(f"chi-square over the template x group cells (data as variance): {chi2:.1f}")
    return lines


def fit(
    doc: Mapping[str, Any],
    seeds: list[int],
    rounds: int,
    workers: int,
    progress: Callable[[str], None] = print,
    power: float = 0.5,
    on_round: Callable[[int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """*on_round* receives each round's updated document (a long fit keeps its progress)."""

    current = copy.deepcopy(dict(doc))
    for round_ in range(rounds):
        counts = count(current, seeds, workers)
        progress(f"== round {round_ + 1}: {counts.campaigns} campaigns")
        for line in table(current, counts):
            progress(line)
        current = update(current, counts, power)
        if on_round is not None:
            on_round(round_ + 1, current)
    counts = count(current, seeds, workers)
    progress("== fitted")
    for line in table(current, counts):
        progress(line)
    return current


__all__ = ["Counts", "count", "fit", "table", "update", "with_generator"]
