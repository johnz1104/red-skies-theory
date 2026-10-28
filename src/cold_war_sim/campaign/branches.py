"""Branch audit: what a set of campaigns actually did, and where they got stuck.

Runs campaigns and reports, for the user's map of alternative worlds
(``docs/BRANCHES.md``):

* each major event's firing rate and final outcomes, and each decision event's choices;
* the most common paths through chosen events (their final outcomes, in order);
* events that never fired;
* **stuck-flag candidates**: flags still set at the end although some effect in the
  catalog clears them, with how often. Some are legitimate (Germany stays divided in a
  campaign without reunification); others mark dead ends (a war flag whose war never
  ended).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from .catalog import Catalog
from .engine import CampaignEngine, EngineConfig, run_episode
from .policies import make_policy


@dataclass
class BranchReport:
    campaigns: int
    fired: Counter[str] = field(default_factory=Counter)
    outcomes: dict[str, Counter[str]] = field(default_factory=dict)
    choices: dict[str, Counter[str]] = field(default_factory=dict)
    paths: Counter[str] = field(default_factory=Counter)
    stuck: Counter[str] = field(default_factory=Counter)
    never: tuple[str, ...] = ()

    def lines(self, catalog: Catalog, top: int = 12) -> list[str]:
        n = self.campaigns
        out = [f"{n} campaigns"]
        out.append("major events (fired share: final outcomes)")
        for event_id in catalog.major_ids:
            if not self.fired[event_id]:
                continue
            counts = self.outcomes.get(event_id, Counter())
            mix = ", ".join(f"{label} {c}" for label, c in counts.most_common())
            out.append(f"  {event_id:<28} {self.fired[event_id] / n:5.2f}: {mix}")
        if self.choices:
            out.append("decision events (choices over all firings)")
            for event_id, counts in sorted(self.choices.items()):
                mix = ", ".join(f"{option} {c}" for option, c in counts.most_common())
                out.append(f"  {event_id:<28} {mix}")
        if self.paths:
            out.append("most common paths")
            out += [f"  {c:4d}  {path}" for path, c in self.paths.most_common(top)]
        out.append(
            "flags still set at the end that the catalog clears somewhere (dead-end candidates)"
        )
        out += [f"  {flag:<32} {c}" for flag, c in self.stuck.most_common()] or ["  none"]
        out.append(f"never fired: {len(self.never)}")
        out += [f"  {event_id}" for event_id in self.never]
        return out


def audit(
    catalog: Catalog,
    seeds: Iterable[int],
    *,
    policy: str = "equilibrium",
    npc_policy: str = "equilibrium",
    temperature: float = 0.1,
    events: Iterable[str] | None = None,
) -> BranchReport:
    """Play one campaign per seed and collect the branch statistics."""

    engine = CampaignEngine(
        catalog, EngineConfig(npc_policy=npc_policy, baseline_temperature=temperature)
    )
    chosen = tuple(events) if events is not None else ()
    seen: set[str] = set()
    seeds = list(seeds)
    report = BranchReport(campaigns=len(seeds))
    for seed in seeds:
        run_episode(engine, make_policy(policy), seed)
        world = engine.world
        seen |= set(world.fired)
        for event_id in world.fired:
            event = catalog.events[event_id]
            if event.major is not None:
                report.fired[event_id] += 1
                label = world.outcomes.get(event_id, "(running)")
                report.outcomes.setdefault(event_id, Counter())[label] += 1
        for item in engine.s.trace:
            if "decision" in item:
                report.choices.setdefault(str(item["decision"]), Counter())[str(item["c"])] += 1
        if chosen:
            ends = [
                f"{item['end']}={item['outcome']}"
                for item in engine.s.trace
                if item.get("end") in chosen and item.get("outcome") != "refused"
            ]
            report.paths[" > ".join(ends) or "(none)"] += 1
        for flag in world.flags & catalog.cleared_flags:
            report.stuck[flag] += 1
    report.never = tuple(sorted(set(catalog.events) - seen))
    return report


__all__ = ["BranchReport", "audit"]
