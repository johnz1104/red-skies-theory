"""Sweeps over the assumptions register: which findings are robust, which are fragile.

User decision 7 (2026-10-14; tools chosen 2026-10-17). Three tools:

* :func:`event_sweep` (``assumptions sweep --event X``): at the event's historical start
  (the replay's world), each live number goes to the low and high ends of its range; the
  event is re-solved wherever its bound, derived config changes, and the numbers are ranked
  by how far the outcome distribution moves (total variation), flagging changes of the most
  likely outcome. It generalizes ``goals sweep``. World-state numbers act upstream, so the
  candidates are the event's own numbers, the goal functions, the dynamics and terminal
  terms (derived effort costs and loss terms), and the cutscenes (derived payoffs count
  their follow-on chances).
* :func:`robust` (``assumptions robust``): D fixed draws x S seeds of campaigns, plus the
  same campaigns at the catalog's values (common seeds). Each finding of ``findings.yaml``
  is measured per draw; it is **robust** if its claim holds in at least 90% of draws. With
  S seeds a draw's rate is itself noisy, so the claim is also checked on each S-seed set at
  the catalog's values (the base share). A finding is **fragile** if it is not robust and
  either the base share reaches 90% or the draws' share falls more than two standard
  errors below it; otherwise S is too small to tell (**noisy**; add seeds). A claim that
  fails over all the campaigns at the catalog's values **fails at base**. A draw in which a
  ``given: fired`` finding's event never fires has no value; such draws are counted apart
  (absent) and left out of the shares. Suspects are the numbers whose position in
  their range correlates most with the finding across draws (Spearman); with many numbers
  and few draws, chance correlations near 0.4 are expected, so each suspect is checked by
  S-seed runs with only that number at either end of its range.
* :func:`history` (``assumptions history``): the historical replay (historical choices and
  chance) under D draws: for each dated event, the share of draws in which it still fires
  within two quarters of its date and reaches its historical outcome. History must stay
  possible from 1945 (user, 2026-10-14); this measures how much of it rests on the numbers.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import calendar, goalcheck
from .assumptions import Assumption, Plan, Sampling, plan_for
from .catalog import Catalog, data_root
from .engine import CampaignEngine, EngineConfig, run_episode
from .policies import make_policy
from .rng import counter_uniform
from .serialization import canonical_json

ROBUST_SHARE = 0.9  # a finding holding in at least this share of draws is robust
MOVED = 0.01  # total variation below this is solver noise or rounding
FINDINGS_FILE = "findings.yaml"
# what a finding claims (user, 2026-10-20): "conclusion", what the model concludes at its
# values (a rate); "possible", that a historical outcome stays possible (history must stay
# possible from 1945 but is not a target, 2026-10-14)
FINDING_KINDS = ("conclusion", "possible")


# ---------------------------------------------------------------- event sweep
@dataclass
class SweepLine:
    address: str
    value: float
    low: float
    high: float
    at_low: dict[str, float] | None  # None: the bound config did not change
    at_high: dict[str, float] | None
    distance: float  # the larger total variation of the two ends
    flips: bool  # an end changes the most likely outcome


@dataclass
class SweepReport:
    event_id: str
    quarter: int
    base: dict[str, float]
    candidates: int
    lines: list[SweepLine]

    def text(self, top: int = 25) -> list[str]:
        moved = [line for line in self.lines if line.distance >= MOVED]
        flips = [line for line in self.lines if line.flips]
        out = [
            f"{self.event_id} at {calendar.fmt(self.quarter)} (the replay's world): "
            f"base {_dist(self.base)}",
            f"  {self.candidates} candidate numbers; {len(self.lines)} change the bound config; "
            f"{len(moved)} move the outcome by {MOVED:g} or more; {len(flips)} change the most "
            "likely outcome",
        ]
        for line in sorted(self.lines, key=lambda x: -x.distance)[:top]:
            mark = "FLIP " if line.flips else "     "
            out.append(
                f"  {mark}{line.distance:4.2f}  {line.address}  {line.value:g} "
                f"[{line.low:.3g} .. {line.high:.3g}]"
            )
            for end, dist in (("low", line.at_low), ("high", line.at_high)):
                if dist is not None:
                    out.append(f"           {end:<4} {_dist(dist)}")
        return out


def _dist(dist: Mapping[str, float]) -> str:
    return ", ".join(
        f"{k} {v:.2f}" for k, v in sorted(dist.items(), key=lambda kv: -kv[1]) if v >= 0.005
    )


def _modal(dist: Mapping[str, float]) -> str:
    return max(sorted(dist), key=lambda k: dist[k])


def sweep_candidates(plan: Plan, catalog: Catalog, event_id: str) -> list[Assumption]:
    """Live numbers that can change the event's bound, derived config at a fixed world."""

    cutscenes = {e.id for e in catalog.events.values() if e.kind in ("cutscene", "leadership")}
    result = []
    for item in plan.live():
        doc = item.address.partition(":")[0]
        if doc == "event":
            owner = item.address.partition(":")[2].partition("/")[0]
            if owner == event_id or owner in cutscenes:
                result.append(item)
        elif doc in ("goals", "dynamics") or item.address.startswith("powers:rewards/"):
            result.append(item)
    return result


def event_sweep(
    catalog: Catalog,
    event_id: str,
    *,
    seed: int = 0,
    iterations: int = 300,
    rollouts: int = 300,
    temperature: float = goalcheck.DEFAULT_TEMPERATURE,
    select: Sequence[str] = (),
    progress: Callable[[str], None] | None = None,
) -> SweepReport:
    event = catalog.events[event_id]
    major = event.major
    if major is None:
        raise ValueError(f"{event_id} is not a major event")
    world, _ = goalcheck.capture_worlds(catalog, [event_id], seed)[event_id]

    solved: dict[str, dict[str, float]] = {}

    def solve(config: Mapping[str, Any]) -> dict[str, float]:
        key = canonical_json(config)
        if key not in solved:
            solved[key] = goalcheck.outcome_distribution(
                major.family,
                config,
                iterations=iterations,
                rollouts=rollouts,
                temperature=temperature,
            )
        return solved[key]

    base_config = catalog.instantiate_config(event_id, world)
    base_key = canonical_json(base_config)
    base = solve(base_config)
    sampling = Sampling(select=tuple(select))
    current = plan_for(catalog)
    candidates = [a for a in sweep_candidates(current, catalog, event_id) if sampling.selects(a)]
    lines = []
    for i, item in enumerate(candidates):
        if progress is not None and i % 50 == 0:
            progress(f"  {i} of {len(candidates)} candidates ...")
        assert item.low is not None and item.high is not None
        ends: list[dict[str, float] | None] = []
        for value in (item.low, item.high):
            variant = catalog.variant(current.setting(item.address, value))
            config = variant.instantiate_config(event_id, world)
            ends.append(None if canonical_json(config) == base_key else solve(config))
        if ends[0] is None and ends[1] is None:
            continue
        distance = max(goalcheck.total_variation(base, d) for d in ends if d is not None)
        flips = any(_modal(d) != _modal(base) for d in ends if d is not None)
        lines.append(
            SweepLine(
                item.address, item.value, item.low, item.high, ends[0], ends[1], distance, flips
            )
        )
    return SweepReport(event_id, world.quarter, base, len(candidates), lines)


# ------------------------------------------------------------------- findings
@dataclass(frozen=True)
class Finding:
    id: str
    text: str
    metric: Mapping[str, Any]
    claim: Mapping[str, float]
    source: str = ""
    kind: str = "conclusion"

    def value(self, runs: Sequence[Mapping[str, Any]]) -> float | None:
        """The finding's rate (or mean) over *runs*; None when no run qualifies."""

        metric = self.metric
        if "return" in metric:
            values = [run["returns"].get(metric["return"], 0.0) for run in runs]
            return float(np.mean(values)) if values else None
        pool = runs
        event = metric.get("outcome") or metric.get("tag")
        if metric.get("given") == "fired" and event is not None:
            pool = [run for run in runs if event in run["outcomes"]]
        if not pool:
            return None
        return sum(1.0 for run in pool if self._holds(run)) / len(pool)

    def _holds(self, run: Mapping[str, Any]) -> bool:
        metric = self.metric
        if "outcome" in metric:
            label = run["outcomes"].get(metric["outcome"])
            return label is not None and _matches(label, metric.get("label", "*"))
        if "tag" in metric:
            tags = run["tags"].get(metric["tag"], ())
            return any(_matches(tag, metric["has"]) for tag in tags)
        if "fired" in metric:
            return metric["fired"] in run["fired"]
        if "ended" in metric:
            return bool(run["ended"] == metric["ended"])
        raise ValueError(f"finding {self.id}: unknown metric {dict(metric)}")

    def holds(self, value: float | None) -> bool:
        if value is None:
            return False
        if "at_least" in self.claim and value < self.claim["at_least"]:
            return False
        return not ("at_most" in self.claim and value > self.claim["at_most"])

    def claim_text(self) -> str:
        parts = []
        if "at_least" in self.claim:
            parts.append(f">= {self.claim['at_least']:g}")
        if "at_most" in self.claim:
            parts.append(f"<= {self.claim['at_most']:g}")
        return " and ".join(parts)


def _matches(value: str, pattern: str | Sequence[str]) -> bool:
    patterns = [pattern] if isinstance(pattern, str) else list(pattern)
    hits = [p for p in patterns if not p.startswith("!")]
    misses = [p[1:] for p in patterns if p.startswith("!")]
    if any(fnmatchcase(value, p) for p in misses):
        return False
    return not hits or any(fnmatchcase(value, p) for p in hits)


def load_findings(path: Path | None = None) -> list[Finding]:
    path = path or data_root() / FINDINGS_FILE
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    result = []
    for item in doc.get("findings", []):
        kind = str(item.get("kind", "conclusion"))
        if kind not in FINDING_KINDS:
            raise ValueError(f"finding {item['id']}: kind must be one of {FINDING_KINDS}")
        result.append(
            Finding(
                id=str(item["id"]),
                text=str(item.get("text", item["id"])),
                metric=dict(item["metric"]),
                claim={k: float(v) for k, v in item["claim"].items()},
                source=str(item.get("source", "")),
                kind=kind,
            )
        )
    return result


# ------------------------------------------------------------------- workers
@dataclass(frozen=True)
class RunSpec:
    policy: str = "equilibrium"
    npc_policy: str = "equilibrium"
    chance: str = "sample"
    playable_tier: int | None = None
    temperature: float = goalcheck.DEFAULT_TEMPERATURE
    select: tuple[str, ...] = ()
    spread: float = 1.0


# ("base", None, seed), ("draw", (draw id, selection), seed) or ("set", (address, value), seed)
Job = tuple[str, Any, int]

_CATALOG: Catalog | None = None
_SPEC: RunSpec | None = None
_ENGINES: dict[Any, CampaignEngine] = {}


def _init(spec: RunSpec) -> None:
    global _CATALOG, _SPEC
    from . import load_catalog

    _CATALOG, _SPEC = load_catalog(), spec
    _ENGINES.clear()


def _engine(kind: str, key: Any) -> CampaignEngine:
    assert _CATALOG is not None and _SPEC is not None
    tag = (kind, key)
    engine = _ENGINES.get(tag)
    if engine is None:
        if len(_ENGINES) > 8:
            _ENGINES.clear()
        catalog = (
            _CATALOG.variant(plan_for(_CATALOG, _SPEC.playable_tier).setting(key[0], key[1]))
            if kind == "set"
            else _CATALOG
        )
        sampling = (
            Sampling(select=tuple(key[1]), spread=_SPEC.spread, draw=int(key[0]))
            if kind == "draw"
            else None
        )
        engine = CampaignEngine(
            catalog,
            EngineConfig(
                chance_mode=_SPEC.chance,
                npc_policy=_SPEC.npc_policy,
                playable_tier=_SPEC.playable_tier,
                baseline_temperature=_SPEC.temperature,
                sampling=sampling,
            ),
        )
        _ENGINES[tag] = engine
    return engine


def _play(job: Job) -> tuple[Job, dict[str, Any]]:
    assert _SPEC is not None
    kind, key, seed = job
    engine = _engine(kind, key)
    result = run_episode(engine, make_policy(_SPEC.policy), seed)
    world = engine.world
    summary = {
        "outcomes": dict(world.outcomes),
        "tags": {k: tuple(v) for k, v in world.outcome_tags.items()},
        "fired": dict(world.fired),
        "ended": result.end_reason,
        "returns": dict(result.returns),
    }
    return job, summary


def run_jobs(
    jobs: Sequence[Job], spec: RunSpec, workers: int | None = None
) -> dict[Job, dict[str, Any]]:
    """Play every job (in worker processes when *workers* > 1)."""

    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if workers <= 1:
        _init(spec)
        return dict(_play(job) for job in jobs)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(spec,)) as pool:
        return dict(pool.map(_play, jobs, chunksize=1))


# --------------------------------------------------------------------- robust
@dataclass
class FindingResult:
    finding: Finding
    base: float | None  # over every campaign at the catalog's values
    per_draw: list[float | None]
    base_sets: list[float | None] = field(default_factory=list)  # per seed set, catalog values
    suspects: list[tuple[str, float]] = field(default_factory=list)  # (address, rho)
    confirmed: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)
    by_group: dict[str, float] = field(default_factory=dict)  # group drawn alone -> holds share

    def share(self, values: Sequence[float | None]) -> float:
        """The share of *values* (absent ones left out) for which the claim holds."""

        known = [v for v in values if v is not None]
        return sum(1.0 for v in known if self.finding.holds(v)) / len(known) if known else 0.0

    @property
    def holds_share(self) -> float:
        return self.share(self.per_draw)

    @property
    def base_holds(self) -> float:
        return self.share(self.base_sets) if self.base_sets else 1.0

    @property
    def absent(self) -> int:
        return sum(1 for v in self.per_draw if v is None)

    @property
    def verdict(self) -> str:
        if not self.finding.holds(self.base):
            return "fails at base"
        if self.holds_share >= ROBUST_SHARE:
            return "robust"
        known = len(self.per_draw) - self.absent
        p = self.base_holds
        error = math.sqrt(p * (1.0 - p) / known) if known else 0.0
        if p >= ROBUST_SHARE or self.holds_share < p - 2.0 * error:
            return "fragile"
        return "noisy"


@dataclass
class RobustReport:
    draws: int
    seeds: int
    spec: RunSpec
    results: list[FindingResult]
    numbers: int = 0  # numbers drawn

    def text(self) -> list[str]:
        chance = math.sqrt(2.0 * math.log(max(self.numbers, 2))) / math.sqrt(max(self.draws - 1, 1))
        out = [
            f"{self.draws} draws x {self.seeds} seeds ({self.spec.policy} play; spread "
            f"{self.spec.spread:g}; select {','.join(self.spec.select) or 'all live'}), "
            f"plus the same seeds at the catalog's values",
            f"  suspects are leads: with {self.numbers} numbers and {self.draws} draws the largest "
            f"|rho| by chance alone is about {min(chance, 1.0):.2f}; the runs with a suspect "
            "alone at its range ends are the test",
            f"  base: the rate over all {self.draws * self.seeds} campaigns at the catalog's "
            f"values, and the share of its {self.seeds}-seed sets in which the claim holds; "
            "draws: the per-draw rate (mean, median [5% .. 95%]), the share of draws in which "
            "it holds, and draws in which a given-fired event never fired",
            f"  {'finding':<36} {'claim':<8} {'base':>5} {'holds':>5}  "
            f"{'draws: mean, median [5% .. 95%]':<32} {'holds':>5} {'absent':>6}  verdict",
        ]
        headers = {
            "conclusion": "  what the model concludes at its values:",
            "possible": "  history stays possible:",
        }
        ordered = sorted(self.results, key=lambda r: FINDING_KINDS.index(r.finding.kind))
        for index, r in enumerate(ordered):
            if index == 0 or r.finding.kind != ordered[index - 1].finding.kind:
                out.append(headers[r.finding.kind])
            values = [v for v in r.per_draw if v is not None]
            spread = (
                f"{np.mean(values):.2f}, {np.median(values):.2f} "
                f"[{np.quantile(values, 0.05):.2f} .. {np.quantile(values, 0.95):.2f}]"
                if values
                else "-"
            )
            base = "-" if r.base is None else f"{r.base:.2f}"
            out.append(
                f"  {r.finding.id:<36} {r.finding.claim_text():<8} {base:>5} "
                f"{r.base_holds:5.2f}  {spread:<32} {r.holds_share:5.2f} {r.absent:6d}  "
                f"{r.verdict}"
            )
            for address, rho in r.suspects:
                ends = r.confirmed.get(address)
                tail = ""
                if ends is not None:
                    low, high = (("-" if v is None else f"{v:.2f}") for v in ends)
                    tail = f"; alone at its ends: low {low}, high {high}"
                out.append(f"      suspect {address} (rho {rho:+.2f}){tail}")
            if r.by_group:
                shares = sorted(r.by_group.items(), key=lambda kv: kv[1])
                text = ", ".join(f"{g} {v:.2f}" for g, v in shares)
                out.append(f"      holds when one group is drawn alone: {text}")
        return out


def _ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = array.argsort(kind="stable")
    ranks = np.empty(len(array))
    ranks[order] = np.arange(len(array), dtype=float)
    for value in np.unique(array):  # average ranks for ties
        tied = array == value
        ranks[tied] = ranks[tied].mean()
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    ra, rb = _ranks(a), _ranks(b)
    if ra.std() == 0.0 or rb.std() == 0.0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def robust(
    catalog: Catalog,
    findings: Sequence[Finding],
    *,
    draws: int,
    seeds: int,
    spec: RunSpec,
    workers: int | None = None,
    suspects: int = 3,
    confirm: bool = True,
    by_group: bool = False,
    save: Path | None = None,
    runs: Mapping[Job, dict[str, Any]] | None = None,
    progress: Callable[[str], None] | None = None,
) -> RobustReport:
    """Measure *findings* over D draws x S seeds. *save* writes every campaign's summary;
    *runs* (from :func:`load_runs`) reuses saved campaigns instead of playing them."""

    note = progress or (lambda _text: None)
    seed_sets = {d: [d * seeds + j for j in range(seeds)] for d in range(draws)}
    if runs is None:
        jobs: list[Job] = [
            ("draw", (d, spec.select), s) for d in range(draws) for s in seed_sets[d]
        ]
        jobs += [("base", None, s) for d in range(draws) for s in seed_sets[d]]
        note(f"{len(jobs)} campaigns ...")
        runs = run_jobs(jobs, spec, workers)
    if save is not None:
        records = [{"job": [kind, key, seed], **run} for (kind, key, seed), run in runs.items()]
        save.write_text(canonical_json({"spec": spec, "seeds": seeds, "runs": records}) + "\n")
    base_runs = [runs[("base", None, s)] for d in range(draws) for s in seed_sets[d]]
    per_draw_runs = {
        d: [runs[("draw", (d, spec.select), s)] for s in seed_sets[d]] for d in range(draws)
    }
    current = plan_for(catalog, spec.playable_tier)
    sampling = Sampling(select=spec.select, spread=spec.spread)
    drawn = [a for a in current.live() if sampling.selects(a)]
    positions = {
        a.address: [counter_uniform(d, "assumption", a.address) for d in range(draws)]
        for a in drawn
        if a.linked is None or _first_of_set(a, drawn)
    }
    results = []
    for finding in findings:
        per_draw = [finding.value(per_draw_runs[d]) for d in range(draws)]
        base_sets = [
            finding.value([runs[("base", None, s)] for s in seed_sets[d]]) for d in range(draws)
        ]
        result = FindingResult(finding, finding.value(base_runs), per_draw, base_sets)
        known = [(d, v) for d, v in enumerate(per_draw) if v is not None]
        if len(known) >= 4 and result.verdict != "robust":
            scored = [
                (address, spearman([u[d] for d, _ in known], [v for _, v in known]))
                for address, u in positions.items()
            ]
            scored = [item for item in scored if item[1] != 0.0]
            scored.sort(key=lambda item: -abs(item[1]))
            result.suspects = scored[:suspects]
        results.append(result)
    if confirm:
        targets = {address for r in results for address, _ in r.suspects}
        by_address = {a.address: a for a in drawn}
        jobs = [
            ("set", (address, end), s)
            for address in sorted(targets)
            for end in (by_address[address].low, by_address[address].high)
            for s in range(seeds)
        ]
        if jobs:
            note(f"{len(jobs)} confirmation campaigns ...")
            ends = run_jobs(jobs, spec, workers)
            for r in results:
                for address, _ in r.suspects:
                    item = by_address[address]
                    values = []
                    for end in (item.low, item.high):
                        pool = [ends[("set", (address, end), s)] for s in range(seeds)]
                        values.append(r.finding.value(pool))
                    r.confirmed[address] = (values[0], values[1])
    fragile = [r for r in results if r.verdict != "robust"]
    if by_group and fragile:
        groups = sorted({a.group for a in drawn})
        jobs = [("draw", (d, (g,)), s) for g in groups for d in range(draws) for s in seed_sets[d]]
        note(f"{len(jobs)} campaigns with one group drawn at a time ...")
        alone = run_jobs(jobs, spec, workers)
        for g in groups:
            for r in fragile:
                values = [
                    r.finding.value([alone[("draw", (d, (g,)), s)] for s in seed_sets[d]])
                    for d in range(draws)
                ]
                r.by_group[g] = r.share(values)
    return RobustReport(draws, seeds, spec, results, len(drawn))


def load_runs(path: Path) -> tuple[RunSpec, int, int, dict[Job, dict[str, Any]]]:
    """Read a file written by ``robust(save=...)``: its spec, draws, seeds and campaigns."""

    data = json.loads(path.read_text())
    raw = data["spec"]
    spec = RunSpec(**{**raw, "select": tuple(raw["select"])})
    runs: dict[Job, dict[str, Any]] = {}
    draws = 0
    for record in data["runs"]:
        kind, key, seed = record.pop("job")
        if kind == "draw":
            key = (int(key[0]), tuple(key[1]))
            draws = max(draws, key[0] + 1)
        runs[(kind, key, int(seed))] = record
    return spec, draws, int(data["seeds"]), runs


def _first_of_set(item: Assumption, drawn: Iterable[Assumption]) -> bool:
    """A linked pair is drawn through its first member (the second takes the rest)."""

    for other in drawn:
        if other.linked == item.linked:
            return other.address == item.address
    return False


# -------------------------------------------------------------------- history
@dataclass
class HistoryReport:
    """For each dated event: the share of draws in which it breaks (never fires, or ends
    otherwise than in history) and the share in which only its timing moves (outside two
    quarters of its date), for the joint draw and for each group drawn alone."""

    draws: int
    spec: RunSpec
    base_failures: list[str]
    selections: list[str]  # "all" first, then one group at a time
    broken: dict[str, dict[str, list[int]]]  # selection -> event -> draws in which it broke
    timing: dict[str, dict[str, list[int]]]  # selection -> event -> draws, timing only
    ended: dict[str, Counter[str]]  # selection -> end reasons

    def share(self, table: str, selection: str, event_id: str) -> float:
        source = self.broken if table == "broken" else self.timing
        return len(source[selection].get(event_id, [])) / self.draws

    def text(self, top: int = 40) -> list[str]:
        out = [
            f"the historical replay (historical choices and chance) under {self.draws} draws "
            f"(spread {self.spec.spread:g})",
            f"  at the catalog's values: {len(self.base_failures)} dated events miss (should be 0)",
        ]
        for selection in self.selections:
            ends = ", ".join(f"{k} {v}" for k, v in self.ended[selection].most_common())
            out.append(f"  endings, {selection}: {ends}")
        events = sorted(
            {e for table in (self.broken, self.timing) for e in table["all"]},
            key=lambda e: (-self.share("broken", "all", e), -self.share("timing", "all", e), e),
        )
        groups = self.selections[1:]
        out.append(
            "  share of draws in which an event breaks / only its timing moves; by group: the "
            "groups that break it when drawn alone"
        )
        for event_id in events[:top]:
            causes = [
                f"{g} {self.share('broken', g, event_id):.2f}"
                for g in sorted(groups, key=lambda g: -self.share("broken", g, event_id))
                if self.share("broken", g, event_id) > 0
            ]
            timing = [
                f"{g} {self.share('timing', g, event_id):.2f}"
                for g in sorted(groups, key=lambda g: -self.share("timing", g, event_id))
                if self.share("timing", g, event_id) > 0
            ]
            out.append(
                f"    {self.share('broken', 'all', event_id):4.2f} / "
                f"{self.share('timing', 'all', event_id):4.2f}  {event_id}"
            )
            if causes:
                out.append(f"        breaks: {', '.join(causes[:5])}")
            if timing:
                out.append(f"        timing: {', '.join(timing[:5])}")
        return out


def _history_failures(catalog: Catalog, run: Mapping[str, Any]) -> dict[str, str]:
    """Dated events that never fire or fire more than two quarters off ("early", "late"),
    and major events that end otherwise than in history."""

    failures = {}
    for event in catalog.events.values():
        if event.historical_date is not None:
            fired = run["fired"].get(event.id)
            if fired is None:
                failures[event.id] = "never fires"
                continue
            if abs(fired - event.historical_date) > 2:
                failures[event.id] = "late" if fired > event.historical_date else "early"
        major = event.major
        if major is not None and major.historical_outcome is not None:
            label = run["outcomes"].get(event.id)
            if label is not None and label != major.historical_outcome:
                failures[event.id] = f"ends {label}"
    return failures


def history(
    catalog: Catalog,
    *,
    draws: int,
    spec: RunSpec,
    workers: int | None = None,
    by_group: bool = True,
) -> HistoryReport:
    spec = RunSpec(
        policy="historical",
        npc_policy="historical",
        chance="historical",
        playable_tier=spec.playable_tier,
        temperature=spec.temperature,
        select=spec.select,
        spread=spec.spread,
    )
    current = plan_for(catalog, spec.playable_tier)
    groups = sorted({a.group for a in current.live() if Sampling(select=spec.select).selects(a)})
    selections: list[tuple[str, tuple[str, ...]]] = [("all", spec.select)]
    if by_group:
        selections += [(g, (g,)) for g in groups]
    jobs: list[Job] = [("base", None, 0)]
    jobs += [("draw", (d, select), 0) for _, select in selections for d in range(draws)]
    runs = run_jobs(jobs, spec, workers)
    base_failures = sorted(_history_failures(catalog, runs[("base", None, 0)]))
    broken: dict[str, dict[str, list[int]]] = {}
    timing: dict[str, dict[str, list[int]]] = {}
    ended: dict[str, Counter[str]] = {}
    for name, select in selections:
        broken[name], timing[name], ended[name] = defaultdict(list), defaultdict(list), Counter()
        for d in range(draws):
            run = runs[("draw", (d, select), 0)]
            ended[name][run["ended"]] += 1
            for event_id, reason in _history_failures(catalog, run).items():
                table = timing if reason in ("early", "late") else broken
                table[name][event_id].append(d)
    return HistoryReport(
        draws, spec, base_failures, [n for n, _ in selections], broken, timing, ended
    )


__all__ = [
    "Finding",
    "HistoryReport",
    "RobustReport",
    "RunSpec",
    "SweepReport",
    "event_sweep",
    "history",
    "load_findings",
    "load_runs",
    "robust",
    "spearman",
]
