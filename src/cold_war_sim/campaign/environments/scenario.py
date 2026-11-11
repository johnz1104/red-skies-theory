"""Single-event scenarios: one major event played from an entry state.

An **entry** is the engine's state at the end of the director phase in the quarter an
event starts, with the event instantiated at that world. An episode restores an entry,
reseeds the event's chance, the computer-run roles and the director's hazards from the
episode seed, and plays until the event resolves. The agents are the identities of the
event's power roles; each is rewarded with its return in the event's game divided by
kappa, the quantity the game's equilibria are computed on.
"""

from __future__ import annotations

import pickle
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ...records import Record
from ..catalog.core import Catalog, load_catalog
from ..decisions.policies import make_policy
from ..engine.core import CampaignEngine, EngineConfig, EngineState
from ..engine.reproducibility import stream
from ..world import calendar
from ..world.state import IDENTITIES, POWERS, identity, seat
from .campaign import AgentCycle

ENTRY_SOURCES = ("historical", "campaign")
# Version of the saved state layout. Increase this when a change to stored fields
# would make earlier scenarios incompatible.
ENTRY_FORMAT = 5

# Saved scenarios record the module where each class was defined. These moves
# kept the same fields, so existing scenarios can still be restored.
_ENTRY_MODULES = {
    "cold_war_sim.campaign.engine.config": "cold_war_sim.campaign.engine.state",
    "cold_war_sim.campaign.engine_state": "cold_war_sim.campaign.engine.state",
    "cold_war_sim.campaign.state": "cold_war_sim.campaign.world.state",
    "cold_war_sim.campaign.scenario": "cold_war_sim.campaign.environments.scenario",
    "cold_war_sim.families.alignment.game": "cold_war_sim.families.alignment",
    "cold_war_sim.families.alignment.params": "cold_war_sim.families.alignment",
    "cold_war_sim.families.intervention.game": "cold_war_sim.families.intervention",
    "cold_war_sim.families.intervention.params": "cold_war_sim.families.intervention",
    "cold_war_sim.families.crisis.roles": "cold_war_sim.families.crisis.params",
    "cold_war_sim.families.crisis.options": "cold_war_sim.families.crisis.params",
    "cold_war_sim.families.attrition.state": "cold_war_sim.families.attrition.game",
    "cold_war_sim.families.bargaining.params": "cold_war_sim.families.bargaining.game",
}


class _EntryUnpickler(pickle.Unpickler):
    """Restore scenario classes saved under their earlier module names."""

    def find_class(self, module: str, name: str) -> Any:
        return super().find_class(_ENTRY_MODULES.get(module, module), name)


class Entry(Record):
    """An event's entry state: where it came from, and the engine snapshot."""

    def __init__(
        self,
        event_id: str,
        source: str,
        seed: int,
        quarter: int,
        state: EngineState,
        catalog: str = "",
        format: int = 1,
    ) -> None:
        self.event_id = event_id
        self.source = source  # "historical" or "campaign"
        self.seed = seed
        self.quarter = quarter
        self.state = state
        self.catalog = catalog  # Catalog.digest at capture ("" in entries saved before)
        self.format = format  # ENTRY_FORMAT at capture (1 in entries saved before)

    def __deepcopy__(self, memo: dict[int, object]) -> Entry:
        return self

    @property
    def label(self) -> str:
        """A readable name: the event, its date, and where the entry came from."""

        return f"{self.event_id}@{calendar.fmt(self.quarter)}:{self.source}/{self.seed}"


def capture_entries(
    catalog: Catalog,
    events: Iterable[str],
    *,
    source: str = "historical",
    seeds: Iterable[int] = (0,),
    playable_tier: int | None = 1,
    temperature: float | None = None,
) -> dict[str, list[Entry]]:
    """Entry states of *events*: the historical replay (``source="historical"``: one per
    event) or reference-play campaigns, one per seed in which the event fires."""

    if source not in ENTRY_SOURCES:
        raise ValueError(f"source must be one of {ENTRY_SOURCES}")
    wanted = frozenset(events)
    unknown = sorted(
        e for e in wanted if e not in catalog.events or catalog.events[e].major is None
    )
    if unknown:
        raise ValueError(f"not major events: {unknown}")
    historical = source == "historical"
    config = EngineConfig(
        chance_mode="historical" if historical else "sample",
        npc_policy="historical" if historical else "equilibrium",
        playable_tier=playable_tier,
        capture=wanted,
    )
    if temperature is not None:
        config = config.replace(baseline_temperature=temperature)
    entries: dict[str, list[Entry]] = {e: [] for e in sorted(wanted)}
    for seed in seeds if not historical else (0,):
        engine = CampaignEngine(catalog, config)
        policy = make_policy("historical" if historical else "equilibrium")
        rng = np.random.default_rng(seed)
        engine.reset(seed)
        while not engine.done and wanted - set(engine.captured):
            request = engine.pending()
            if request is None:
                break
            engine.act(policy.choose(engine, request, rng))
        for event_id, state in engine.captured.items():
            entries[event_id].append(
                Entry(
                    event_id,
                    source,
                    seed,
                    state.world.quarter,
                    state,
                    catalog.digest,
                    ENTRY_FORMAT,
                )
            )
    return entries


def check_entries(entries: Mapping[str, Sequence[Entry]], catalog: Catalog) -> None:
    """Reject saved scenarios whose format or catalog differs from the current one.

    Each entry keeps the game as it was when the event started. Changing the
    catalog cannot update that saved game, so the entry must be captured again.
    """

    old = [e for items in entries.values() for e in items if e.format != ENTRY_FORMAT]
    if old:
        raise ValueError(
            f"{len(old)} entries were saved by older code (first: {old[0].label}); "
            "recapture them with `rl entries`"
        )
    stale = [e for items in entries.values() for e in items if e.catalog != catalog.digest]
    if stale:
        raise ValueError(
            f"{len(stale)} entries were captured under a different catalog (first: "
            f"{stale[0].label}); recapture them with `rl entries`"
        )


_CATALOG: Catalog | None = None


def _capture_one(job: tuple[tuple[str, ...], int, int | None]) -> dict[str, list[Entry]]:
    """The entries of one reference-play campaign, on a worker process."""

    global _CATALOG
    if _CATALOG is None:
        _CATALOG = load_catalog()
    events, seed, tier = job
    return capture_entries(_CATALOG, events, source="campaign", seeds=(seed,), playable_tier=tier)


def capture_campaign_entries(
    events: Iterable[str],
    seeds: Iterable[int],
    *,
    workers: int = 1,
    playable_tier: int | None = 1,
) -> dict[str, list[Entry]]:
    """Entries from reference-play campaigns (one per seed in which each event fires),
    on worker processes (the base catalog, loaded in each)."""

    wanted = tuple(sorted(set(events)))
    jobs = [(wanted, int(seed), playable_tier) for seed in seeds]
    if workers <= 1:
        return merge_entries(*(_capture_one(job) for job in jobs))
    with ProcessPoolExecutor(workers) as pool:
        return merge_entries(*pool.map(_capture_one, jobs))


def save_entries(path: Path, entries: Mapping[str, Sequence[Entry]]) -> None:
    """Pickle *entries* to *path*."""

    with path.open("wb") as handle:
        pickle.dump({k: list(v) for k, v in entries.items()}, handle)


def load_entries(path: Path) -> dict[str, list[Entry]]:
    """Entries saved by :func:`save_entries`. They are pickles, which can run code when
    loaded: load only files you made yourself."""

    try:
        with path.open("rb") as handle:
            loaded: dict[str, list[Entry]] = _EntryUnpickler(handle).load()  # trusted cache
    except (pickle.UnpicklingError, AttributeError, ImportError, TypeError, IndexError) as error:
        # the file pickles classes as they were when it was saved
        raise ValueError(
            f"{path} was saved by older code and no longer loads ({type(error).__name__}); "
            "recapture it with `rl entries`"
        ) from error
    return loaded


def merge_entries(*pools: Mapping[str, Sequence[Entry]]) -> dict[str, list[Entry]]:
    """The union of several sets of entries, by event."""

    merged: dict[str, list[Entry]] = {}
    for pool in pools:
        for event_id, items in pool.items():
            merged.setdefault(event_id, []).extend(items)
    return merged


def power_roles(engine: CampaignEngine, event_id: str) -> tuple[str, ...]:
    """The power roles taking part in the running *event_id* (optional roles that were
    dropped at its start are left out)."""

    active = engine.s.active[event_id]
    major = engine.catalog.events[event_id].major
    assert major is not None
    roles = set(getattr(active.state, "roles", major.roles))
    return tuple(r for r in major.power_roles() if r in roles)


class ScenarioEnv(AgentCycle):
    """An AEC-shaped multi-agent environment over single-event scenarios (the same
    interface as :class:`~.env.ColdWarEnv`)."""

    metadata: ClassVar[dict[str, Any]] = {
        "name": "cold_war_scenario_v0",
        "is_parallelizable": False,
    }

    def __init__(
        self,
        entries: Mapping[str, Sequence[Entry]],
        catalog: Catalog | None = None,
        config: EngineConfig | None = None,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        self.entries = {e: list(v) for e, v in entries.items() if v}
        if not self.entries:
            raise ValueError("no entries")
        self.events = tuple(sorted(self.entries))
        raw = np.array([float((weights or {}).get(e, 1.0)) for e in self.events])
        if (raw < 0).any() or raw.sum() <= 0:
            raise ValueError("event weights must be non-negative and not all zero")
        self.weights = raw / raw.sum()
        base = config or EngineConfig(npc_policy="equilibrium", playable_tier=1)
        self.config = base.replace(capture=frozenset(), scope=None)
        catalog = catalog or load_catalog()
        check_entries(self.entries, catalog)
        self.engine = CampaignEngine(catalog, self.config)
        self.kappa = self.engine.catalog.goals.kappa
        self.possible_agents: tuple[str, ...] = tuple(
            i for i in IDENTITIES if seat(i) in self.config.agents
        )
        self.event_id = ""
        self.entry: Entry | None = None
        self.roles: dict[str, str] = {}  # agent identity -> its power role in the event
        self.agents: list[str] = []
        self.rewards: dict[str, float] = {}
        self._cumulative_rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict[str, Any]] = {}
        self.agent_selection: str = ""
        self.expected = False  # this episode's returns are chance-expected (instant events)

    def reset(self, seed: int = 0, event: str | None = None, entry: int | None = None) -> None:
        """Restore an entry (drawn from *seed* unless *event* or *entry* is given), reseed
        its randomness from *seed*, and run to the first agent decision.
        """

        rng = np.random.default_rng(seed)
        if event is None:
            event = self.events[int(rng.choice(len(self.events), p=self.weights))]
        pool = self.entries[event]
        chosen = pool[int(rng.integers(len(pool))) if entry is None else entry]
        self.event_id, self.entry = event, chosen
        engine = self.engine
        engine.config = self.config.replace(scope=event)
        engine.restore(chosen.state)
        _reseed(engine.s, event, int(seed))
        self.expected = engine.expects_scope()
        world = engine.world
        self.roles = {
            identity(world, p): p
            for p in power_roles(engine, event)
            if p in self.config.agents and world.powers[p].status != "eliminated"
        }
        self.agents = [a for a in self.possible_agents if a in self.roles]
        self.rewards = dict.fromkeys(self.agents, 0.0)
        self._cumulative_rewards = dict.fromkeys(self.agents, 0.0)
        self.terminations = dict.fromkeys(self.agents, False)
        self.truncations = dict.fromkeys(self.agents, False)
        self.infos = {a: {} for a in self.agents}
        self._totals = dict.fromkeys(self.agents, 0.0)  # chance-expected returns
        engine.resume()
        self._collect()

    def returns(self) -> dict[str, float]:
        """Each agent's return in the event's game, in goal units (once it resolved); with
        chance-expected returns (instant events), the estimate the rewards add up to."""

        returns = self.engine.s.last_returns.get(self.event_id)
        if not self.engine.s.scope_done or returns is None:
            return {}
        if self.expected:
            return dict(self._totals)
        return {a: float(returns.get(role, 0.0)) / self.kappa for a, role in self.roles.items()}

    def outcome_masses(self) -> dict[str, float]:
        """The outcome distribution the episode stands for (once it resolved): its outcome,
        or with chance-expected returns the probability mass of every terminal counted."""

        if not self.engine.s.scope_done:
            return {}
        expected = self.engine.s.expected
        if self.expected and expected is not None:
            return dict(expected.outcomes)
        outcome = self.outcome()
        return {outcome: 1.0} if outcome is not None else {}

    def outcome(self) -> str | None:
        """The event's outcome label, once it has resolved."""

        if not self.engine.s.scope_done:
            return None
        return self.engine.world.outcomes.get(self.event_id)

    # ---------------------------------------------------------------- helpers
    def _collect(self) -> None:
        """Pay each agent its return in the event's game over kappa: as it accrues in
        expectation, or at the event's end.
        """

        self.engine.pop_rewards()  # the campaign's goal reward is not a scenario's reward
        self.rewards = dict.fromkeys(self.agents, 0.0)
        if self.expected:  # payoffs as the engine counts them in expectation
            accrued = self.engine.pop_expected()
            for agent in self.agents:
                value = accrued.get(self.roles[agent], 0.0) / self.kappa
                self.rewards[agent] = value
                self._cumulative_rewards[agent] += value
                self._totals[agent] += value
                self.terminations[agent] = self.engine.done
        elif self.engine.done:
            final = self.returns()
            for agent in self.agents:
                self.rewards[agent] = final.get(agent, 0.0)
                self._cumulative_rewards[agent] += self.rewards[agent]
                self.terminations[agent] = True
        for agent in self.agents:
            self.infos[agent] = self._info()
        self._select_next()

    def _info(self) -> dict[str, Any]:
        """The event, the entry, the date, and the path's survival weight."""

        info: dict[str, Any] = {
            "event": self.event_id,
            "entry": self.entry.label if self.entry is not None else "",
            "quarter": calendar.fmt(self.engine.world.quarter),
            # the weight of this episode's path so far (chance-expected returns)
            "survival": 1.0 if self.engine.s.expected is None else self.engine.s.expected.survival,
        }
        if self.engine.done:
            info["outcome"] = self.outcome()
            info["resolved"] = self.engine.s.scope_done
        return info


def _reseed(state: EngineState, event_id: str, seed: int) -> None:
    """Fresh randomness for an episode from an entry: the event's chance, the computer-run
    roles and the director's hazards and effect lotteries."""

    state.seed = seed
    state.npc_rng = stream(seed, "__npc__")
    for active_id, active in state.active.items():
        instance = max(state.instances.get(active_id, 1) - 1, 0)
        active.rng = stream(seed, active_id, instance)
    state.trace.append({"scenario": event_id, "seed": seed})


__all__ = [
    "ENTRY_FORMAT",
    "ENTRY_SOURCES",
    "POWERS",
    "Entry",
    "ScenarioEnv",
    "capture_campaign_entries",
    "capture_entries",
    "check_entries",
    "load_entries",
    "merge_entries",
    "power_roles",
    "save_entries",
]
