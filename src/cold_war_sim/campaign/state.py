"""Persistent world state of the campaign.

The state is a tree of plain mutable dataclasses so it can be deep-copied for
snapshots and serialized for traces. Numeric fields have declared bounds
(:data:`POWER_FIELDS`, :data:`REGION_FIELDS`, :data:`GLOBAL_FIELDS`); effects
and dynamics clamp to them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

POWERS = ("USA", "USSR", "PRC", "UK", "FRA")
REGIONS = (
    "western_europe",
    "germany",
    "eastern_europe",
    "mediterranean",
    "middle_east",
    "persian_gulf",
    "north_africa",
    "south_asia",
    "afghanistan",
    "china_mainland",
    "taiwan",
    "korea",
    "japan",
    "indochina",
    "southeast_asia",
    "latin_america",
    "caribbean",
    "central_africa",
    "horn_of_africa",
)
STATUSES = ("dormant", "insurgent", "active", "eliminated")
REGIMES = ("ccp", "kmt")  # who governs the China seat
# Learning identities: one per seat, except that the China seat changes hands
# between the Communists (PRC) and the Nationalists (ROC).
IDENTITIES = ("USA", "USSR", "PRC", "ROC", "UK", "FRA")
POSTURES = ("economy", "balanced", "conventional", "nuclear", "science")
FOCUS_MODES = ("aid", "covert")
DIPLOMACY_ACTIONS = ("engage", "pressure")
NEUTRAL = "neutral"
CONTROL_THRESHOLD = 0.5

# Numeric fields and their bounds.
POWER_FIELDS: dict[str, tuple[float, float]] = {
    "economy": (0.01, 100.0),
    "growth": (-0.2, 0.2),
    "burden": (0.0, 0.6),
    "conventional": (0.0, 100.0),
    "nuclear": (0.0, 100.0),
    "tech_nuclear": (0.0, 10.0),
    "tech_missile": (0.0, 10.0),
    "tech_space": (0.0, 10.0),
    "stability": (0.0, 1.0),
    # no working ceiling (user decision 2026-10-13, E39): at a cap of 1 a highly
    # credible power gained nothing from further success; the drift toward 0.5 keeps
    # it near 1, and 2 is only a safety bound
    "credibility": (0.0, 2.0),
    "prestige": (0.0, 1.0),
    "hawkishness": (0.0, 1.0),
    "risk_tolerance": (0.0, 1.0),
    "reformism": (0.0, 1.0),
}
REGION_FIELDS: dict[str, tuple[float, float]] = {
    "stability": (0.0, 1.0),
    "importance": (0.0, 10.0),
}
GLOBAL_FIELDS: dict[str, tuple[float, float]] = {
    "tension": (0.0, 1.0),
    "arms_control": (0.0, 1.0),
}
RELATION_BOUNDS = (-1.0, 1.0)
INFLUENCE_BOUNDS = (0.0, 1.0)


@dataclass(slots=True)
class PowerState:
    status: str = "active"
    economy: float = 1.0
    growth: float = 0.03
    burden: float = 0.1
    conventional: float = 1.0
    nuclear: float = 0.0
    tech_nuclear: float = 0.0
    tech_missile: float = 0.0
    tech_space: float = 0.0
    stability: float = 0.7
    credibility: float = 0.5
    prestige: float = 0.5
    hawkishness: float = 0.5
    risk_tolerance: float = 0.5
    reformism: float = 0.5
    leader: str = ""
    regime: str = ""
    relations: dict[str, float] = field(default_factory=dict)
    posture: str = "balanced"
    focus_region: str | None = None
    focus_mode: str | None = None
    diplomacy: str | None = None  # "engage:USSR" or "pressure:PRC"
    focus_history: list[tuple[int, str, str]] = field(default_factory=list)
    diplomacy_history: list[tuple[int, str, str]] = field(default_factory=list)


@dataclass(slots=True)
class RegionState:
    influence: dict[str, float] = field(default_factory=lambda: dict.fromkeys(POWERS, 0.0))
    stability: float = 0.6
    importance: float = 1.0
    base_stability: float = 0.6
    controller: str = NEUTRAL
    colonial_owner: str | None = None
    conflict: str | None = None


@dataclass(slots=True)
class WorldState:
    quarter: int
    tension: float = 0.3
    arms_control: float = 0.0
    flags: set[str] = field(default_factory=set)
    fired: dict[str, int] = field(default_factory=dict)
    active: dict[str, int] = field(default_factory=dict)
    outcomes: dict[str, str] = field(default_factory=dict)
    outcome_tags: dict[str, tuple[str, ...]] = field(default_factory=dict)
    powers: dict[str, PowerState] = field(default_factory=dict)
    regions: dict[str, RegionState] = field(default_factory=dict)
    ended: str | None = None
    nuclear_war: bool = False
    log: list[tuple[int, str, str]] = field(default_factory=list)

    def power(self, power_id: str) -> PowerState:
        return self.powers[power_id]

    def region(self, region_id: str) -> RegionState:
        return self.regions[region_id]

    def active_powers(self) -> tuple[str, ...]:
        return tuple(p for p in POWERS if self.powers[p].status == "active")


def identity(world: WorldState, power: str) -> str:
    """The learning identity that holds *power*'s seat."""

    if power == "PRC" and world.powers["PRC"].regime == "kmt":
        return "ROC"
    return power


def seat(identity_id: str) -> str:
    """The seat (power) an identity plays."""

    return "PRC" if identity_id == "ROC" else identity_id


def default_world(quarter: int) -> WorldState:
    """A structurally complete world with neutral values (used for validation)."""

    powers = {p: PowerState(relations={q: 0.0 for q in POWERS if q != p}) for p in POWERS}
    regions = {r: RegionState() for r in REGIONS}
    return WorldState(quarter=quarter, powers=powers, regions=regions)


def normalize_influence(region: RegionState) -> None:
    """Clamp each power's influence to [0, 1] and rescale so the sum is at most 1."""

    for p in POWERS:
        region.influence[p] = min(1.0, max(0.0, region.influence.get(p, 0.0)))
    total = sum(region.influence.values())
    if total > 1.0:
        for p in POWERS:
            region.influence[p] /= total


def update_controller(region: RegionState) -> None:
    leader = max(POWERS, key=lambda p: (region.influence[p], -POWERS.index(p)))
    region.controller = leader if region.influence[leader] >= CONTROL_THRESHOLD else NEUTRAL
