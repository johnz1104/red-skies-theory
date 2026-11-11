"""Quarterly world dynamics (all coefficients illustrative).

Each power's yearly posture splits its state budget (economy x burden) among
conventional forces, nuclear forces, science and the economy. Force stocks follow the
discretized arms law ``x += dt (-delta x + b u)``. Science advances nuclear, missile and
space technology. Region focus moves influence, diplomacy moves relations, and arms
growth and wars raise tension.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

from ...records import Record
from ..catalog.effects import shift_influence
from .state import (
    POWER_FIELDS,
    POWERS,
    REGIONS,
    PowerState,
    WorldState,
    normalize_influence,
    update_controller,
)

if TYPE_CHECKING:
    from ..catalog.core import Catalog

DT = 0.25  # years per quarter
TECH_CAP = 10.0  # technology stocks stop here (state.POWER_FIELDS)
WAR_EFFORT_SCALE = 4.0  # a war effort of 1 counts four times in the stability formulas
NUCLEAR_STRENGTH_WEIGHT = 0.6  # nuclear forces in military strength (the pre-goals score)

# Budget shares for each posture are in powers.yaml's `postures` block, where
# the assumptions register can include them in sensitivity checks.
POSTURE_KEYS = ("economy", "conventional", "nuclear", "science")

DEFAULTS: dict[str, float] = {
    "depreciation_conventional": 0.12,
    "depreciation_nuclear": 0.05,
    "control_effectiveness": 0.8,
    "conventional_scale": 15.0,
    "nuclear_scale": 30.0,
    "tech_scale": 5.0,
    # growth points per unit of GDP moved between investment and the military: meta-analyses find
    # small effects (Alptekin and Levine 2012, small positive; Simpartl 2024, small negative;
    # Dunne and Tian's surveys). Placeholder within "small": 0.15 points per 1% of GDP (was 0.08
    # per unit of budget share, about 0.8 points per 1% of US GDP)
    "investment_growth": 0.15,
    "burden_drag": 0.05,
    "aid_rate": 0.06,
    "covert_rate": 0.05,
    "covert_destabilize": 0.08,
    "containment": 0.3,  # share of a pressured power's aid and covert gains lost (placeholder)
    "influence_decay": 0.02,
    "region_stability_reversion": 0.25,
    "relations_step": 0.03,
    "pressure_tension": 0.004,
    "arms_tension": 0.01,
    "tension_floor": 0.15,
    "tension_decay": 0.25,
    "stability_reversion": 0.2,
    "stability_target": 0.7,
    "stability_growth_effect": 2.0,
    "reputation_reversion": 0.03,
    "war_growth_drag": 0.01,
    "war_stability_drain": 0.01,
    "war_tension": 0.002,
    # the domestic cost of fighting with one's own troops: a quarter of direct effort lowers the
    # stability target by this, times the power's casualty_sensitivity (powers.yaml: democracies
    # 1, autocracies less). Public support falls as casualties mount (Mueller 1973; Gartner and
    # Segura 1998); stability's reversion toward its target gives the concave path of support
    # over a long war
    "casualty_drain": 0.25,
    # Reference points: a burden of 10% of GDP costs no growth; growth above 3% a year
    # raises stability; half of nuclear spending advances missiles; and credibility
    # drifts toward 0.5. Investment is measured against the balanced posture's share.
    "burden_reference": 0.1,
    "stability_growth_reference": 0.03,
    "missile_from_nuclear": 0.5,
    "reputation_target": 0.5,
    # one engaging side warms a relation by this share of a step (warmth needs both)
    "one_sided_warmth": 0.5,
}


class DynamicsParams(Record):
    """The dynamics' coefficients (``DEFAULTS`` overridden by ``powers.yaml``), with each
    power's technology rates, military efficiency and casualty sensitivity and the
    postures' budget shares.
    """

    def __init__(
        self,
        values: Mapping[str, float],
        tech_rates: Mapping[str, Mapping[str, float]],
        efficiency: Mapping[str, float],
        postures: Mapping[str, tuple[float, float, float, float]],
        casualty_sensitivity: Mapping[str, float] | None = None,
    ) -> None:
        self.values = values
        self.tech_rates = tech_rates
        self.efficiency = efficiency
        self.postures = postures
        self.casualty_sensitivity: Mapping[str, float] = (
            {} if casualty_sensitivity is None else casualty_sensitivity
        )

    def __getitem__(self, key: str) -> float:
        return self.values[key]

    def __deepcopy__(self, memo: dict[int, object]) -> DynamicsParams:
        return self

    @classmethod
    def from_catalog(cls, catalog: Catalog) -> DynamicsParams:
        """The parameters of *catalog*; an override of an unknown coefficient is an error."""

        dynamics, powers = catalog.dynamics, catalog.powers
        unknown = set(dynamics) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown dynamics coefficients {sorted(unknown)}")
        values = {**DEFAULTS, **{k: float(v) for k, v in dynamics.items()}}
        rates = {p: dict(powers[p].tech_rates) for p in POWERS}
        efficiency = {p: float(powers[p].military_efficiency) for p in POWERS}
        sensitivity = {p: float(powers[p].casualty_sensitivity) for p in POWERS}
        return cls(values, rates, efficiency, dict(catalog.postures), sensitivity)


def military_strength(world: WorldState, power: str) -> float:
    """Conventional forces plus weighted nuclear forces (the pre-goals score's measure)."""

    state = world.powers[power]
    return state.conventional + NUCLEAR_STRENGTH_WEIGHT * state.nuclear


def _insurgent_war_costs(state: PowerState, params: DynamicsParams, effort: float) -> None:
    """An insurgency has no growth, force or technology dynamics, but fighting still
    drains its economy and stability as it does an active power's."""

    if effort <= 0.0:
        return
    state.economy = max(0.01, state.economy * (1.0 - params["war_growth_drag"] * effort * DT))
    drain = (
        params["stability_reversion"]
        * DT
        * params["war_stability_drain"]
        * effort
        * WAR_EFFORT_SCALE
    )
    state.stability = _clamp01(state.stability - drain)


def step(
    world: WorldState,
    params: DynamicsParams,
    war_effort: Mapping[str, float],
    direct_effort: Mapping[str, float] | None = None,
) -> None:
    """Advance the world by one quarter. ``war_effort[p]`` is p's proxy-war load, and
    ``direct_effort[p]`` the part of it fought with its own troops."""

    nuclear_before = sum(world.powers[p].nuclear for p in ("USA", "USSR"))
    for power in POWERS:
        state = world.powers[power]
        if state.status == "insurgent":
            _insurgent_war_costs(state, params, war_effort.get(power, 0.0))
            continue
        if state.status != "active":
            continue
        econ, conv, nuc, sci = params.postures[state.posture]
        effort = war_effort.get(power, 0.0)
        growth = (
            state.growth
            + params["investment_growth"] * state.burden * (econ - params.postures["balanced"][0])
            - params["burden_drag"] * (state.burden - params["burden_reference"])
            - params["war_growth_drag"] * effort
        )
        state.economy = max(0.01, state.economy * (1.0 + growth * DT))
        budget = state.economy * state.burden * params.efficiency[power]
        b = params["control_effectiveness"]
        state.conventional += DT * (
            -params["depreciation_conventional"] * state.conventional
            + b * budget * conv * params["conventional_scale"]
        )
        if state.tech_nuclear >= 1.0:
            restraint = 1.0 - world.arms_control if power in ("USA", "USSR") else 1.0
            state.nuclear += DT * (
                -params["depreciation_nuclear"] * state.nuclear
                + b * budget * nuc * params["nuclear_scale"] * restraint
            )
        rates = params.tech_rates[power]
        scale = params["tech_scale"] * budget * DT
        state.tech_nuclear = min(
            TECH_CAP, state.tech_nuclear + scale * rates.get("nuclear", 1.0) * (nuc + sci)
        )
        state.tech_missile = min(
            TECH_CAP,
            state.tech_missile
            + scale * rates.get("missile", 1.0) * (params["missile_from_nuclear"] * nuc + sci),
        )
        state.tech_space = min(TECH_CAP, state.tech_space + scale * rates.get("space", 1.0) * sci)
        target = (
            params["stability_target"]
            + params["stability_growth_effect"] * (growth - params["stability_growth_reference"])
            - params["war_stability_drain"] * effort * WAR_EFFORT_SCALE
            - params["casualty_drain"]
            * params.casualty_sensitivity.get(power, 1.0)
            * (direct_effort or {}).get(power, 0.0)
        )
        state.stability = _clamp01(
            state.stability + params["stability_reversion"] * DT * (target - state.stability)
        )
        for attr in ("credibility", "prestige"):
            value = getattr(state, attr)
            low, high = POWER_FIELDS[attr]
            value += params["reputation_reversion"] * DT * (params["reputation_target"] - value)
            setattr(state, attr, min(high, max(low, value)))
        _focus(world, power, params)
    _diplomacy(world, params)
    for region_id in REGIONS:
        region = world.regions[region_id]
        for p in POWERS:
            region.influence[p] *= 1.0 - params["influence_decay"] * DT
        normalize_influence(region)
        update_controller(region)
        region.stability = _clamp01(
            region.stability
            + params["region_stability_reversion"] * DT * (region.base_stability - region.stability)
        )
    nuclear_after = sum(world.powers[p].nuclear for p in ("USA", "USSR"))
    arms_growth = max(0.0, nuclear_after - nuclear_before)
    total_effort = sum(war_effort.values())
    world.tension = _clamp01(
        world.tension
        + params["arms_tension"] * arms_growth
        + params["war_tension"] * total_effort
        - params["tension_decay"] * DT * max(0.0, world.tension - params["tension_floor"])
    )


def _focus(world: WorldState, power: str, params: DynamicsParams) -> None:
    """A power's region focus: aid or covert action shifts influence toward it in
    proportion to the square root of its economy, less under pressure (containment);
    covert action also destabilizes the region.
    """

    state = world.powers[power]
    if state.focus_region is None:
        return
    reach = math.sqrt(max(state.economy, 0.0))
    if under_pressure(world, power):
        reach *= 1.0 - params["containment"]
    if state.focus_mode == "challenge":
        return  # the standoff it starts is the action
    if state.focus_mode in ("covert", "insurgency"):
        shift_influence(world, state.focus_region, power, params["covert_rate"] * reach * DT)
        region = world.regions[state.focus_region]
        region.stability = _clamp01(region.stability - params["covert_destabilize"] * DT)
    else:
        shift_influence(world, state.focus_region, power, params["aid_rate"] * reach * DT)


def under_pressure(world: WorldState, power: str) -> bool:
    """Whether an active power is pressuring *power* this year."""

    return any(
        world.powers[q].status == "active" and world.powers[q].diplomacy == f"pressure:{power}"
        for q in POWERS
        if q != power
    )


def _diplomacy(world: WorldState, params: DynamicsParams) -> None:
    """Move each pair's shared relation: each side pressuring cools it by a step,
    each side engaging warms it by half a step (warmth needs both)."""

    step = params["relations_step"]
    active = [p for p in POWERS if world.powers[p].status == "active"]
    for i, p in enumerate(active):
        for q in active[i + 1 :]:
            actions = (world.powers[p].diplomacy, world.powers[q].diplomacy)
            towards = (f":{q}", f":{p}")
            engaging = sum(1 for a, t in zip(actions, towards, strict=True) if a == f"engage{t}")
            pressuring = sum(
                1 for a, t in zip(actions, towards, strict=True) if a == f"pressure{t}"
            )
            if not engaging and not pressuring:
                continue
            delta = step * (params["one_sided_warmth"] * engaging - pressuring)
            value = max(-1.0, min(1.0, world.powers[p].relations[q] + delta))
            world.powers[p].relations[q] = value
            world.powers[q].relations[p] = value
            world.tension = _clamp01(world.tension + params["pressure_tension"] * pressuring)


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))
