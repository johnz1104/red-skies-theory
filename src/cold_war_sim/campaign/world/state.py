"""The world state of the campaign: powers, regions, flags and the record of events.

Numeric fields have declared bounds (``POWER_FIELDS``, ``REGION_FIELDS``,
``GLOBAL_FIELDS``); effects and dynamics clamp to them.
"""

from __future__ import annotations

from ...records import Record

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
    # no working ceiling: at a cap of 1 a highly credible power gained nothing from further
    # success; the drift toward 0.5 keeps it near 1, and 2 is only a safety bound
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


class PowerState(Record):
    """One power: its economy, forces, technology, domestic state, leader and yearly strategy."""

    def __init__(
        self,
        status: str = "active",
        economy: float = 1.0,
        growth: float = 0.03,
        burden: float = 0.1,
        conventional: float = 1.0,
        nuclear: float = 0.0,
        tech_nuclear: float = 0.0,
        tech_missile: float = 0.0,
        tech_space: float = 0.0,
        stability: float = 0.7,
        credibility: float = 0.5,
        prestige: float = 0.5,
        hawkishness: float = 0.5,
        risk_tolerance: float = 0.5,
        reformism: float = 0.5,
        leader: str = "",
        regime: str = "",
        relations: dict[str, float] | None = None,
        posture: str = "balanced",
        focus_region: str | None = None,
        focus_mode: str | None = None,
        diplomacy: str | None = None,
        focus_history: list[tuple[int, str, str]] | None = None,
        diplomacy_history: list[tuple[int, str, str]] | None = None,
    ) -> None:
        self.status = status
        self.economy = economy
        self.growth = growth
        self.burden = burden
        self.conventional = conventional
        self.nuclear = nuclear
        self.tech_nuclear = tech_nuclear
        self.tech_missile = tech_missile
        self.tech_space = tech_space
        self.stability = stability
        self.credibility = credibility
        self.prestige = prestige
        self.hawkishness = hawkishness
        self.risk_tolerance = risk_tolerance
        self.reformism = reformism
        self.leader = leader
        self.regime = regime
        self.relations: dict[str, float] = {} if relations is None else relations
        self.posture = posture
        self.focus_region = focus_region
        self.focus_mode = focus_mode
        self.diplomacy = diplomacy  # "engage:USSR" or "pressure:PRC"
        self.focus_history: list[tuple[int, str, str]] = (
            [] if focus_history is None else focus_history
        )
        self.diplomacy_history: list[tuple[int, str, str]] = (
            [] if diplomacy_history is None else diplomacy_history
        )


class RegionState(Record):
    """One region: each power's influence share (summing to at most 1), stability,
    importance, and who controls or owns it.
    """

    def __init__(
        self,
        influence: dict[str, float] | None = None,
        stability: float = 0.6,
        importance: float = 1.0,
        base_stability: float = 0.6,
        controller: str = NEUTRAL,
        colonial_owner: str | None = None,
        conflict: str | None = None,
    ) -> None:
        self.influence: dict[str, float] = (
            dict.fromkeys(POWERS, 0.0) if influence is None else influence
        )
        self.stability = stability
        self.importance = importance
        self.base_stability = base_stability
        self.controller = controller
        self.colonial_owner = colonial_owner
        self.conflict = conflict


class WorldState(Record):
    """The state of the world in one quarter.

    ``fired``, ``active`` and ``resolved`` map a major event to the quarter it started,
    has been running since, and last ended; ``outcomes`` and ``outcome_tags`` record how
    each ended.
    """

    def __init__(
        self,
        quarter: int,
        tension: float = 0.3,
        arms_control: float = 0.0,
        flags: set[str] | None = None,
        fired: dict[str, int] | None = None,
        active: dict[str, int] | None = None,
        resolved: dict[str, int] | None = None,
        outcomes: dict[str, str] | None = None,
        outcome_tags: dict[str, tuple[str, ...]] | None = None,
        powers: dict[str, PowerState] | None = None,
        regions: dict[str, RegionState] | None = None,
        ended: str | None = None,
        nuclear_war: bool = False,
        log: list[tuple[int, str, str]] | None = None,
    ) -> None:
        self.quarter = quarter
        self.tension = tension
        self.arms_control = arms_control
        self.flags: set[str] = set() if flags is None else flags
        self.fired: dict[str, int] = (
            {} if fired is None else fired
        )  # major event -> quarter it started
        self.active: dict[str, int] = {} if active is None else active
        self.resolved: dict[str, int] = (
            {} if resolved is None else resolved
        )  # quarter it last ended
        self.outcomes: dict[str, str] = {} if outcomes is None else outcomes
        self.outcome_tags: dict[str, tuple[str, ...]] = {} if outcome_tags is None else outcome_tags
        self.powers: dict[str, PowerState] = {} if powers is None else powers
        self.regions: dict[str, RegionState] = {} if regions is None else regions
        self.ended = ended
        self.nuclear_war = nuclear_war
        self.log: list[tuple[int, str, str]] = [] if log is None else log

    def power(self, power_id: str) -> PowerState:
        return self.powers[power_id]

    def region(self, region_id: str) -> RegionState:
        return self.regions[region_id]

    def active_powers(self) -> tuple[str, ...]:
        """The powers that hold a seat and act this quarter."""

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
    """The region's controller: the power with the largest influence if it reaches the
    control threshold (the earlier power on ties), else neutral.
    """

    leader = max(POWERS, key=lambda p: (region.influence[p], -POWERS.index(p)))
    region.controller = leader if region.influence[leader] >= CONTROL_THRESHOLD else NEUTRAL
