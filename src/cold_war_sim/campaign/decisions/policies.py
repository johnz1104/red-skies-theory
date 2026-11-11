"""Campaign policies: random, historical, and equilibrium baselines.

A policy maps the engine's pending request to an option index. Policies own
their random generators; the engine's streams are never shared with them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

from ...games import sample_action
from .requests import DecisionRequest

if TYPE_CHECKING:
    from ..engine.core import CampaignEngine


class CampaignPolicy(Protocol):
    """A campaign policy: a choice of option for the engine's pending request."""

    def choose(
        self, engine: CampaignEngine, request: DecisionRequest, rng: np.random.Generator
    ) -> int:
        """The index of the option chosen for *request*."""
        ...


class RandomPolicy:
    """Uniform over the request's options."""

    def choose(
        self, engine: CampaignEngine, request: DecisionRequest, rng: np.random.Generator
    ) -> int:
        """A uniform draw over the options."""

        return int(rng.integers(len(request.options)))


class HistoricalPolicy:
    """Chooses the option tagged historical; otherwise the family baseline's most
    likely action (sub-games) or the first option (strategy)."""

    def choose(
        self, engine: CampaignEngine, request: DecisionRequest, rng: np.random.Generator
    ) -> int:
        """The historical option; where history marks none, the modal choice of reference
        play in a sub-game or decision event, else the first option.
        """

        index = request.historical_index()
        if index is not None:
            return index
        if request.role is not None or request.kind == "event_decision":
            dist = engine.baseline_distribution(request)
            if dist:
                best = max(dist, key=lambda a: (dist[a], -request.index_of(a)))
                return request.index_of(best)
        return 0


class EquilibriumPolicy:
    """Samples the reference play in sub-games (logit QRE, dynamic programming, or
    the attrition belief policy) and in decision events (a logit over derived option
    values); historical choices for strategy."""

    def __init__(self) -> None:
        self._fallback = HistoricalPolicy()

    def choose(
        self, engine: CampaignEngine, request: DecisionRequest, rng: np.random.Generator
    ) -> int:
        """A draw from reference play in a sub-game or decision event; elsewhere the fallback's choice."""

        if request.role is None and request.kind != "event_decision":
            return self._fallback.choose(engine, request, rng)
        dist = engine.baseline_distribution(request)
        if not dist:
            return self._fallback.choose(engine, request, rng)
        return request.index_of(sample_action(dist, rng))


# Scripted yearly strategies for the strategy-layer audit (campaign/audit.py). Each sets
# the posture, a focus region ("contested": where the power and its main rival both hold
# the most influence), a focus mode and a diplomatic initiative toward the main rival.
DOCTRINES: dict[str, dict[str, str]] = {
    "historical": {},
    "guns": {
        "posture": "conventional",
        "region": "contested",
        "mode": "aid",
        "diplomacy": "pressure",
    },
    "nuclear": {"posture": "nuclear", "region": "none", "diplomacy": "pressure"},
    "butter": {"posture": "economy", "region": "none", "diplomacy": "engage"},
    "science": {"posture": "science", "region": "none", "diplomacy": "none"},
    "covert": {"posture": "balanced", "region": "contested", "mode": "covert", "diplomacy": "none"},
    "random": {},
}
MAIN_RIVAL = {"USA": "USSR", "USSR": "USA", "PRC": "USA", "UK": "USSR", "FRA": "USSR"}


class DoctrinePolicy:
    """A scripted doctrine on strategy turns; reference play everywhere else."""

    def __init__(self, doctrine: str) -> None:
        if doctrine not in DOCTRINES:
            raise ValueError(f"unknown doctrine {doctrine!r}; choose from {sorted(DOCTRINES)}")
        self.doctrine = doctrine
        self._reference = EquilibriumPolicy()
        self._historical = HistoricalPolicy()

    def choose(
        self, engine: CampaignEngine, request: DecisionRequest, rng: np.random.Generator
    ) -> int:
        """The doctrine's choice on a strategy turn; reference play everywhere else."""

        if request.source != "strategy":
            return self._reference.choose(engine, request, rng)
        if self.doctrine == "historical":
            return self._historical.choose(engine, request, rng)
        if self.doctrine == "random":
            return int(rng.integers(len(request.options)))
        spec = DOCTRINES[self.doctrine]
        ids = [o.id for o in request.options]
        rival = MAIN_RIVAL[request.power]
        stage = request.kind.removeprefix("strategy_")
        wanted = spec.get(stage, "none")
        if stage == "region" and wanted == "contested":
            wanted = _contested(engine, request.power, rival)
        elif stage == "diplomacy" and wanted != "none":
            wanted = f"{wanted}:{rival}"
        return ids.index(wanted) if wanted in ids else 0


def _contested(engine: CampaignEngine, power: str, rival: str) -> str:
    """The region most contested between *power* and *rival*: the largest minimum of
    their influence shares.
    """

    regions = engine.world.regions
    best = max(
        regions,
        key=lambda r: (
            min(regions[r].influence[power], regions[r].influence[rival]),
            regions[r].influence[rival],
            r,
        ),
    )
    return best


POLICIES: dict[str, type] = {
    "random": RandomPolicy,
    "historical": HistoricalPolicy,
    "equilibrium": EquilibriumPolicy,
}


def make_policy(name: str) -> CampaignPolicy:
    """The policy named *name* (a key of ``POLICIES``)."""

    try:
        policy: CampaignPolicy = POLICIES[name]()
    except KeyError as exc:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(POLICIES)}") from exc
    return policy
