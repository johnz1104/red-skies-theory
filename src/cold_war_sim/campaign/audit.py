"""Strategy-layer interest audit: does any yearly strategy dominate?

Before RL design, the user asked (2026-10-12) whether the campaign has a single
solved answer that every agent would converge to. Sub-games are deliberately
small and solvable; the open question is the yearly strategy layer (posture,
focus region and mode, diplomacy). This audit plays scripted doctrines
(``policies.DOCTRINES``) on strategy turns, with reference play in every
sub-game and decision event, and reports:

* **unilateral deviations:** each power's goal return under each doctrine while
  every other power follows history;
* **dominance:** against random profiles of the other powers' doctrines, how
  often one doctrine is a power's best, the margin to the runner-up in units of
  the seed-to-seed spread, and the regret of always playing the most frequent
  best doctrine instead of each profile's best;
* **path diversity:** how many distinct paths the runs take through the tier-1
  events, and their entropy.

Returns are goal returns per identity (the RL reward), on common random numbers:
every doctrine sees the same seeds. Under the goal reward there are no end bonuses
(E33); under ``reward="score"`` the report also gives returns net of them, since
the Soviet dissolution's bonuses dwarf the strategy effects (E32). The dissolution
rate is reported separately.
"""

from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .catalog import Catalog
from .engine import CampaignEngine, EngineConfig, run_episode
from .policies import DOCTRINES, DoctrinePolicy
from .state import POWERS

Profile = tuple[tuple[str, str], ...]  # (power, doctrine) for every power
Job = tuple[str, int, Profile, int]  # (label, focal index, profile, seed)

_ENGINE: CampaignEngine | None = None
_TIER1: tuple[str, ...] = ()


def _init(temperature: float) -> None:
    global _ENGINE, _TIER1
    from . import load_catalog

    catalog = load_catalog()
    _ENGINE = CampaignEngine(
        catalog, EngineConfig(npc_policy="equilibrium", baseline_temperature=temperature)
    )
    _TIER1 = tuple(e.id for e in catalog.ordered if e.major is not None and e.major.tier == 1)


def _play(job: Job) -> tuple[Job, dict[str, float], dict[str, float], str, tuple[str, ...]]:
    """Total returns, returns net of end bonuses, the end reason and the tier-1 path."""

    assert _ENGINE is not None
    _, _, profile, seed = job
    policies = {power: DoctrinePolicy(doctrine) for power, doctrine in profile}
    result = run_episode(_ENGINE, policies, seed)
    world = _ENGINE.world
    path = tuple(f"{e}={world.outcomes[e]}" for e in _TIER1 if e in world.outcomes)
    net = dict(result.returns)
    if world.ended is not None and _ENGINE.config.reward == "score":
        for power in POWERS:
            net[power] -= _ENGINE.catalog.powers[power].end_bonus.get(world.ended, 0.0)
    return job, dict(result.returns), net, result.end_reason, path


@dataclass
class AuditReport:
    seeds: tuple[int, ...]
    profiles: int
    unilateral: dict[str, dict[str, list[float]]] = field(default_factory=dict)  # net
    unilateral_total: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    dissolved: dict[str, Counter[str]] = field(default_factory=dict)  # unilateral runs
    # power -> profile index -> doctrine -> returns net of end bonuses, over seeds
    against: dict[str, dict[int, dict[str, list[float]]]] = field(default_factory=dict)
    paths: Counter[tuple[str, ...]] = field(default_factory=Counter)

    def lines(self) -> list[str]:
        out = [
            f"strategy audit: {len(DOCTRINES)} doctrines, seeds {list(self.seeds)}, "
            f"{self.profiles} random co-player profiles"
        ]
        names = list(DOCTRINES)
        tables = [("goal returns", self.unilateral_total)]
        if self.unilateral != self.unilateral_total:  # the score reward pays end bonuses
            tables = [("net of end bonuses", self.unilateral), ("total", self.unilateral_total)]
        for title, table in tables:
            out.append(f"unilateral deviations from history, {title} (mean; change vs historical)")
            out.append("  " + " " * 6 + "".join(f"{n:>13}" for n in names))
            for power in POWERS:
                row = table.get(power, {})
                base = float(np.mean(row["historical"])) if "historical" in row else 0.0
                cells = []
                for name in names:
                    values = row.get(name)
                    if not values:
                        cells.append(f"{'-':>13}")
                        continue
                    mean = float(np.mean(values))
                    cells.append(f"{mean:>7.2f}{mean - base:>+6.2f}")
                out.append(f"  {power:<6}" + "".join(cells))
        if self.dissolved:
            out.append("Soviet dissolution in the unilateral runs (share of runs)")
            for power in POWERS:
                counts = self.dissolved.get(power, Counter())
                runs = len(self.seeds)
                out.append(f"  {power:<6}" + "".join(f"{counts[n] / runs:>13.2f}" for n in names))
        if self.against:
            out.append("against random profiles of the others' doctrines")
            for power in POWERS:
                summary = self._dominance(power)
                if summary:
                    out.append(f"  {power:<6}" + summary)
        if self.paths:
            total = sum(self.paths.values())
            entropy = -sum(c / total * math.log2(c / total) for c in self.paths.values())
            out.append(
                f"paths through tier-1 events: {len(self.paths)} distinct in {total} runs, "
                f"entropy {entropy:.2f} bits"
            )
            for path, count in self.paths.most_common(5):
                out.append(f"  {count:4d}  {' > '.join(path)}")
        return out

    def _dominance(self, power: str) -> str:
        table = self.against.get(power, {})
        if not table:
            return ""
        best: list[str] = []
        margins: list[float] = []
        regrets: list[float] = []
        spread = []
        for by_doctrine in table.values():
            means = {d: float(np.mean(v)) for d, v in by_doctrine.items() if v}
            ranked = sorted(means, key=lambda d: -means[d])
            best.append(ranked[0])
            sd = [float(np.std(v, ddof=1)) for v in by_doctrine.values() if len(v) > 1]
            spread.append(float(np.mean(sd)) if sd else 0.0)
            scale = spread[-1] if spread[-1] > 0 else 1.0
            margins.append((means[ranked[0]] - means[ranked[1]]) / scale)
        modal, count = Counter(best).most_common(1)[0]
        for by_doctrine in table.values():
            means = {d: float(np.mean(v)) for d, v in by_doctrine.items() if v}
            regrets.append(max(means.values()) - means[modal])
        verdict = "DOMINANT" if count / len(best) >= 0.9 else "depends on the others"
        mix = ", ".join(f"{d} {c}" for d, c in Counter(best).most_common())
        return (
            f"best doctrine by profile: {mix} | modal {modal} best in {count}/{len(best)} "
            f"({verdict}) | median margin {float(np.median(margins)):.2f} sd | "
            f"regret of always {modal}: mean {float(np.mean(regrets)):.2f}"
        )


def _profiles(count: int, rng: np.random.Generator) -> list[dict[str, str]]:
    names = list(DOCTRINES)
    return [{p: names[int(rng.integers(len(names)))] for p in POWERS} for _ in range(count)]


def audit_strategy(
    catalog: Catalog,
    seeds: Iterable[int],
    *,
    profiles: int = 20,
    workers: int | None = None,
    temperature: float = 0.1,
    profile_seed: int = 0,
) -> AuditReport:
    del catalog  # each worker loads its own
    seed_list = tuple(seeds)
    rng = np.random.default_rng(profile_seed)
    jobs: list[Job] = []
    for index, power in enumerate(POWERS):
        for doctrine in DOCTRINES:
            profile = tuple((p, doctrine if p == power else "historical") for p in POWERS)
            jobs += [("unilateral", index, profile, s) for s in seed_list]
    for r, others in enumerate(_profiles(profiles, rng)):
        for index, power in enumerate(POWERS):
            for doctrine in DOCTRINES:
                profile = tuple((p, doctrine if p == power else others[p]) for p in POWERS)
                jobs += [(f"profile:{r}", index, profile, s) for s in seed_list]
    report = AuditReport(seed_list, profiles)
    unilateral: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    unilateral_total: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    dissolved: dict[str, Counter[str]] = defaultdict(Counter)
    against: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")  # one thread per worker process
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(temperature,)) as pool:
        for job, returns, net, reason, path in pool.map(_play, jobs, chunksize=4):
            label, index, profile, _ = job
            power = POWERS[index]
            doctrine = dict(profile)[power]
            value = net.get(power, 0.0)
            if label == "unilateral":
                unilateral[power][doctrine].append(value)
                unilateral_total[power][doctrine].append(returns.get(power, 0.0))
                dissolved[power][doctrine] += reason == "ussr_dissolved"
            else:
                against[power][int(label.split(":")[1])][doctrine].append(value)
            report.paths[path] += 1
    report.unilateral = {p: dict(v) for p, v in unilateral.items()}
    report.unilateral_total = {p: dict(v) for p, v in unilateral_total.items()}
    report.dissolved = dict(dissolved)
    report.against = {p: {r: dict(d) for r, d in v.items()} for p, v in against.items()}
    return report


def job_count(seeds: Sequence[int], profiles: int) -> int:
    return len(POWERS) * len(DOCTRINES) * len(seeds) * (1 + profiles)
