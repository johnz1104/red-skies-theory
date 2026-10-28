"""Single-event scenarios: one major event played from an entry state (user, 2026-10-20).

An **entry** is the engine's state at the end of the director phase in the quarter an
event starts (``EngineConfig.capture``), with the event instantiated at that world: from
the historical replay (historical choices and chance), or from other campaigns. An
episode restores an entry, reseeds the event's chance, the computer-run roles and the
director's hazards from the episode seed, and plays until the event resolves
(``EngineConfig.scope``).

* **Agents** are the learning identities of the event's power roles that take part (the
  China seat follows its regime). They are asked only inside the event; their strategy
  turns, decision events and every other event follow ``npc_policy`` (reference play,
  historical strategy), and events above ``playable_tier`` run on rails.
* **Observation** is the campaign's layout (``CampaignEngine.observe``), so a policy can
  move between scenarios and the campaign.
* **Reward** is each agent's return in the event's game divided by kappa, at the end:
  for an event with derived payoffs, the change in its goal score that the outcome
  causes (with follow-ons), net of its action and effort costs. It is the quantity the
  game's equilibria are computed on, so learned play can be scored by exploitability.

Entries hold no compiled catalog (sampling off), so they pickle (:func:`save_entries`).
"""

from __future__ import annotations

import pickle
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from . import calendar
from .catalog import Catalog, load_catalog
from .decisions import MAX_OPTIONS
from .engine import CampaignEngine, EngineConfig, EngineState
from .policies import make_policy
from .rng import stream
from .state import IDENTITIES, POWERS, identity, seat

ENTRY_SOURCES = ("historical", "campaign")


@dataclass(frozen=True)
class Entry:
    """An event's entry state: where it came from, and the engine snapshot."""

    event_id: str
    source: str  # "historical" or "campaign"
    seed: int
    quarter: int
    state: EngineState

    def __deepcopy__(self, memo: dict[int, object]) -> Entry:
        return self

    @property
    def label(self) -> str:
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
        config = replace(config, baseline_temperature=temperature)
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
            entries[event_id].append(Entry(event_id, source, seed, state.world.quarter, state))
    return entries


_CATALOG: Catalog | None = None


def _capture_one(job: tuple[tuple[str, ...], int, int | None]) -> dict[str, list[Entry]]:
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
    with path.open("wb") as handle:
        pickle.dump({k: list(v) for k, v in entries.items()}, handle)


def load_entries(path: Path) -> dict[str, list[Entry]]:
    with path.open("rb") as handle:
        loaded: dict[str, list[Entry]] = pickle.load(handle)  # our own cache: trusted
    return loaded


def merge_entries(*pools: Mapping[str, Sequence[Entry]]) -> dict[str, list[Entry]]:
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


class ScenarioEnv:
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
        self.config = replace(base, capture=frozenset(), scope=None)
        self.engine = CampaignEngine(catalog or load_catalog(), self.config)
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

    @property
    def observation_size(self) -> int:
        return len(self.engine.feature_names)

    @property
    def action_size(self) -> int:
        return MAX_OPTIONS

    def reset(self, seed: int = 0, event: str | None = None, entry: int | None = None) -> None:
        rng = np.random.default_rng(seed)
        if event is None:
            event = self.events[int(rng.choice(len(self.events), p=self.weights))]
        pool = self.entries[event]
        chosen = pool[int(rng.integers(len(pool))) if entry is None else entry]
        self.event_id, self.entry = event, chosen
        engine = self.engine
        engine.config = replace(self.config, scope=event)
        engine.restore(chosen.state)
        _reseed(engine.s, event, int(seed))
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
        engine.resume()
        self._collect()

    def observe(self, agent: str) -> np.ndarray:
        return self.engine.observe(agent)

    def action_mask(self, agent: str) -> np.ndarray:
        return self.engine.action_mask(agent)

    def last(self) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        agent = self.agent_selection
        return (
            self.observe(agent),
            self._cumulative_rewards[agent],
            self.terminations[agent],
            self.truncations[agent],
            self.infos[agent],
        )

    def step(self, action: int | None) -> None:
        agent = self.agent_selection
        if self.terminations.get(agent) or self.truncations.get(agent):
            if action is not None:
                raise ValueError("terminated or truncated agents must be stepped with None")
            self._remove(agent)
            self.rewards = dict.fromkeys(self.agents, 0.0)  # a removal step earns nothing
            self._select_next()
            return
        if action is None:
            raise ValueError("a live agent needs an action")
        request = self.engine.pending()
        if request is None or self.engine.agent_of(request.power) != agent:
            raise RuntimeError("environment is out of sync with the engine")
        self._cumulative_rewards[agent] = 0.0
        self.engine.act(int(action))
        self._collect()

    def returns(self) -> dict[str, float]:
        """Each agent's return in the event's game, in goal units (once it resolved)."""

        returns = self.engine.s.last_returns.get(self.event_id)
        if not self.engine.s.scope_done or returns is None:
            return {}
        return {a: float(returns.get(role, 0.0)) / self.kappa for a, role in self.roles.items()}

    def outcome(self) -> str | None:
        if not self.engine.s.scope_done:
            return None
        return self.engine.world.outcomes.get(self.event_id)

    # ---------------------------------------------------------------- helpers
    def _collect(self) -> None:
        self.engine.pop_rewards()  # the campaign's goal reward is not a scenario's reward
        self.rewards = dict.fromkeys(self.agents, 0.0)
        if self.engine.done:
            final = self.returns()
            for agent in self.agents:
                self.rewards[agent] = final.get(agent, 0.0)
                self._cumulative_rewards[agent] += self.rewards[agent]
                self.terminations[agent] = True
        for agent in self.agents:
            self.infos[agent] = self._info()
        self._select_next()

    def _select_next(self) -> None:
        for agent in self.agents:
            if self.terminations[agent] or self.truncations[agent]:
                self.agent_selection = agent
                return
        request = self.engine.pending()
        self.agent_selection = self.engine.agent_of(request.power) if request is not None else ""

    def _info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "event": self.event_id,
            "entry": self.entry.label if self.entry is not None else "",
            "quarter": calendar.fmt(self.engine.world.quarter),
        }
        if self.engine.done:
            info["outcome"] = self.outcome()
            info["resolved"] = self.engine.s.scope_done
        return info

    def _remove(self, agent: str) -> None:
        self.agents.remove(agent)
        for table in (
            self.rewards,
            self._cumulative_rewards,
            self.terminations,
            self.truncations,
            self.infos,
        ):
            table.pop(agent, None)


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
    "ENTRY_SOURCES",
    "POWERS",
    "Entry",
    "ScenarioEnv",
    "capture_campaign_entries",
    "capture_entries",
    "load_entries",
    "merge_entries",
    "power_roles",
    "save_entries",
]
