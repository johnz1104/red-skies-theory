"""Strategy-layer interest audit: does any yearly strategy dominate?

Each power plays each scripted doctrine on its strategy turns while the other powers play
random doctrine profiles, with reference play in the sub-games. A doctrine that is best
against every profile dominates; a real trade-off should not be dominated in every
plausible world.
"""

from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..campaign.catalog.core import Catalog
from ..campaign.decisions.policies import DOCTRINES, DoctrinePolicy
from ..campaign.engine.core import CampaignEngine, EngineConfig, run_episode
from ..campaign.world.state import POWERS
from ..records import Record

Profile = tuple[tuple[str, str], ...]  # (power, doctrine) for every power
Job = tuple[str, int, Profile, int]  # (label, focal index, profile, seed)

_ENGINE: CampaignEngine | None = None
_TIER1: tuple[str, ...] = ()


def _init(temperature: float) -> None:
    """Give a worker process its engine and the tier-1 event list."""

    global _ENGINE, _TIER1
    from ..campaign import load_catalog

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


class AuditReport(Record):
    """An audit: each power's mean return by its doctrine, unilaterally and against each
    co-player profile.
    """

    def __init__(
        self,
        seeds: tuple[int, ...],
        profiles: int,
        unilateral: dict[str, dict[str, list[float]]] | None = None,
        unilateral_total: dict[str, dict[str, list[float]]] | None = None,
        dissolved: dict[str, Counter[str]] | None = None,
        against: dict[str, dict[int, dict[str, list[float]]]] | None = None,
        paths: Counter[tuple[str, ...]] | None = None,
    ) -> None:
        self.seeds = seeds
        self.profiles = profiles
        self.unilateral: dict[str, dict[str, list[float]]] = (
            {} if unilateral is None else unilateral
        )  # net
        self.unilateral_total: dict[str, dict[str, list[float]]] = (
            {} if unilateral_total is None else unilateral_total
        )
        self.dissolved: dict[str, Counter[str]] = (
            {} if dissolved is None else dissolved
        )  # unilateral runs
        # power -> profile index -> doctrine -> returns net of end bonuses, over seeds
        self.against: dict[str, dict[int, dict[str, list[float]]]] = (
            {} if against is None else against
        )
        self.paths: Counter[tuple[str, ...]] = Counter() if paths is None else paths

    def lines(self) -> list[str]:
        """The report: each power's doctrines ranked by mean return, and which dominate."""

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
        """The doctrines that are best for *power* against every profile, with their margins."""

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
    """*count* random doctrine profiles, one doctrine per power."""

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
    """Play the audit's campaigns on *workers* processes and collect the report."""

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
    """The number of campaigns an audit plays."""

    return len(POWERS) * len(DOCTRINES) * len(seeds) * (1 + profiles)
