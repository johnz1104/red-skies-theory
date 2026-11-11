"""The shared effects contract: generic outcome effects per family (``data/contract.yaml``).

An event with ``model.effects: contract`` lists only documented ``additions`` on its
outcomes; :func:`expand` writes the contract's generic effects in front of them, in the
effect language, before the event is compiled. The raw documents the catalog keeps are
never changed, so each contract number lives in one place.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import Any

from ..world.dynamics import DEFAULTS, DT, WAR_EFFORT_SCALE
from ..world.state import POWERS

FAMILIES = ("crisis", "attrition", "bargaining", "alignment", "intervention")


class ContractError(ValueError):
    """Raised for an invalid effects contract."""

    pass


def dynamics_of(powers_doc: Mapping[str, Any]) -> dict[str, float]:
    """The dynamics coefficients: the defaults with powers.yaml's overrides."""

    merged = dict(DEFAULTS)
    merged.update({k: float(v) for k, v in (powers_doc.get("dynamics") or {}).items()})
    return merged


def sensitivity_of(powers_doc: Mapping[str, Any]) -> dict[str, float]:
    """Each power's ``casualty_sensitivity`` (powers.yaml; 1 when unset)."""

    entries = powers_doc.get("powers") or {}
    return {p: float((entries.get(p) or {}).get("casualty_sensitivity", 1.0)) for p in POWERS}


def war_costs(
    dynamics: Mapping[str, float], quarters: float, sensitivity: float = 0.0
) -> tuple[float, float]:
    """What *quarters* of full war effort cost a power under the dynamics: the factor
    on its economy, and its stability loss when the fighting stops.

    Each quarter of full effort lowers growth by ``war_growth_drag`` (so the economy
    by that times DT) and the stability target by ``war_stability_drain`` x
    WAR_EFFORT_SCALE plus the growth effect, plus ``casualty_drain`` x *sensitivity*
    when the power fights with its own troops; stability closes ``stability_reversion`` x
    DT of its gap to the target each quarter (``dynamics.step``)."""

    drag = dynamics["war_growth_drag"]
    economy = (1.0 - drag * DT) ** quarters
    target_drop = (
        dynamics["war_stability_drain"] * WAR_EFFORT_SCALE
        + dynamics["stability_growth_effect"] * drag
        + dynamics["casualty_drain"] * sensitivity
    )
    closing = dynamics["stability_reversion"] * DT
    stability = target_drop * (1.0 - (1.0 - closing) ** quarters)
    return economy, stability


def _a_and_b(config: Mapping[str, Any], a_key: str, b_key: str) -> dict[str, list[str]]:
    """Sides ``a`` and ``b`` as the roles under *a_key* and *b_key*, where those are powers."""

    roles = config["roles"]
    return {s: [roles[k]] if roles[k] in POWERS else [] for s, k in (("a", a_key), ("b", b_key))}


def _by_side(entries: Mapping[str, Any]) -> dict[str, list[str]]:
    """The powers among *entries* (role -> spec with a ``side``), by side."""

    sides: dict[str, list[str]] = {"a": [], "b": []}
    for role, spec in entries.items():
        side = spec.get("side")
        if role in POWERS and side in sides:
            sides[side].append(role)
    return sides


SIDES: dict[str, Callable[[Mapping[str, Any]], dict[str, list[str]]]] = {
    "crisis": lambda config: _by_side(config["roles"]),
    "attrition": lambda config: _by_side(config["sponsors"]),
    "bargaining": lambda config: _a_and_b(config, "A", "B"),
    "intervention": lambda config: _a_and_b(config, "patron", "rival"),
    "alignment": lambda config: {"a": [p for p in config["patrons"] if p in POWERS], "b": []},
}


def _sides(family: str, config: Mapping[str, Any]) -> dict[str, list[str]]:
    """The powers on each side, in config order."""

    return SIDES[family](config)


def _war(
    rule: Mapping[str, Any],
    fighters: list[str],
    sides: Mapping[str, list[str]],
    region: str | None,
    dynamics: Mapping[str, float],
    sensitivity: Mapping[str, float],
    relations: bool = True,
) -> list[Any]:
    """A war that is not played out: each fighter pays ``quarters`` of full effort with its
    own troops (:func:`war_costs`) and loses a share of its forces; the region's
    stability falls, the two sides' leading powers fall out, and tension rises."""

    items: list[Any] = []
    for power in fighters:
        economy, stability = war_costs(
            dynamics, float(rule["quarters"]), sensitivity.get(power, 1.0)
        )
        items += [
            {"multiply": {"path": f"powers.{power}.economy", "factor": economy}},
            {"add": {"path": f"powers.{power}.stability", "value": -stability}},
        ]
        if "forces" in rule:
            items.append(
                {
                    "multiply": {
                        "path": f"powers.{power}.conventional",
                        "factor": 1.0 - rule["forces"],
                    }
                }
            )
    if region is not None and "region_stability" in rule:
        items.append(
            {"add": {"path": f"regions.{region}.stability", "value": rule["region_stability"]}}
        )
    if relations and "relations" in rule and sides["a"] and sides["b"]:
        items.append(
            {"relations": {"a": sides["a"][0], "b": sides["b"][0], "add": rule["relations"]}}
        )
    if "tension" in rule:
        items.append({"tension": rule["tension"]})
    return items


def _credibility(winners: list[str], losers: list[str], rule: Mapping[str, Any]) -> list[Any]:
    """Build the configured credibility changes for winners and losers."""

    items: list[Any] = []
    for power in winners if "winner_credibility" in rule else ():
        items.append(
            {"add": {"path": f"powers.{power}.credibility", "value": rule["winner_credibility"]}}
        )
    for power in losers if "loser_credibility" in rule else ():
        items.append(
            {"add": {"path": f"powers.{power}.credibility", "value": rule["loser_credibility"]}}
        )
    return items


class _Ending:
    """One outcome of an event, as the contract's rules read it."""

    def __init__(
        self,
        label: str,
        rules: Mapping[str, Any],
        sides: dict[str, list[str]],
        region: str | None,
        scale: float,
        dynamics: Mapping[str, float],
        sensitivity: Mapping[str, float],
    ) -> None:
        self.label = label
        self.rules = rules  # the family's rules
        self.sides = sides  # the powers on sides a and b
        self.region = region
        self.scale = scale  # the region's importance relative to the reference
        self.dynamics = dynamics
        self.sensitivity = sensitivity

    def influence(self, toward: str, amount: float, **extra: Any) -> dict[str, Any]:
        """The effect that shifts *amount* of the region's influence toward a power."""

        return {
            "shift_influence": {"region": self.region, "toward": toward, "amount": amount, **extra}
        }

    def war(self, rule: Mapping[str, Any], fighters: list[str], **options: Any) -> list[Any]:
        """The costs of an unplayed war to *fighters*."""

        return _war(
            rule, fighters, self.sides, self.region, self.dynamics, self.sensitivity, **options
        )


OTHER = {"a": "b", "b": "a"}


def _crisis(e: _Ending) -> list[Any]:
    """``a_prevails``/``b_prevails`` move influence to the winning side's powers, and
    credibility up for them and down for the losers. ``compromise`` eases tension.
    ``limited_war`` costs every power on either side an unplayed war. ``nuclear_war``
    ends the campaign."""

    if e.label in ("a_prevails", "b_prevails"):
        rule = e.rules["prevails"]
        side = e.label[0]
        winners, losers = e.sides[side], e.sides[OTHER[side]]
        items: list[Any] = []
        if e.region is not None and winners:
            amount = rule["influence"] * e.scale / len(winners)
            items += [e.influence(w, amount) for w in winners]
        return items + _credibility(winners, losers, rule)
    if e.label == "compromise":
        return [{"tension": e.rules["compromise"]["tension"]}]
    if e.label == "limited_war":
        return e.war(e.rules["limited_war"], e.sides["a"] + e.sides["b"])
    if e.label == "nuclear_war":
        return [{"end_campaign": "nuclear_war"}]
    return []


def _bargaining(e: _Ending) -> list[Any]:
    """An agreement warms the two parties' relations and eases tension; a breakdown does
    the reverse; impasses and vetoes change nothing generic."""

    rule = e.rules.get(e.label)
    if rule is None or not (e.sides["a"] and e.sides["b"]):
        return []
    return [
        {"relations": {"a": e.sides["a"][0], "b": e.sides["b"][0], "add": rule["relations"]}},
        {"tension": rule["tension"]},
    ]


def _alignment(e: _Ending) -> list[Any]:
    """``aligned_<P>`` moves influence toward patron P, with credibility up for P and down
    for the other patrons; ``nonaligned`` changes nothing."""

    if not e.label.startswith("aligned_"):
        return []
    rule, winner = e.rules["aligned"], e.label.removeprefix("aligned_")
    losers = [p for p in e.sides["a"] if p != winner]
    items = []
    if e.region is not None and winner in POWERS:
        items.append(e.influence(winner, rule["influence"] * e.scale))
    return items + _credibility([winner] if winner in POWERS else [], losers, rule)


def _intervention(e: _Ending) -> list[Any]:
    """``crushed`` keeps the patron's hold (its credibility up, its relations with the
    rival down); ``defected`` moves influence to the rival (the patron's credibility
    down); ``reformed`` moves a little; ``quagmire`` costs the patron a war; ``clash``
    costs both sides one."""

    patron, rival = e.sides["a"], e.sides["b"]
    rule = e.rules.get(e.label)
    if rule is None:
        return []
    if e.label == "crushed":
        items = _credibility(patron, [], rule)
        if patron and rival:
            items.append({"relations": {"a": patron[0], "b": rival[0], "add": rule["relations"]}})
        return items
    if e.label in ("defected", "reformed"):
        items = []
        if e.region is not None and rival:
            items.append(e.influence(rival[0], rule["influence"] * e.scale))
        if e.label == "defected":
            items += _credibility(rival, patron, rule)
        return items
    if e.label == "quagmire":
        return e.war(rule, patron, relations=False) + _credibility([], patron, rule)
    if e.label == "clash":
        return e.war(rule, patron + rival)
    return []


def _attrition(e: _Ending) -> list[Any]:
    """A victory, or the other side's withdrawal, moves influence to the winning side's
    powers, split by their share of the side's effort (``side_share``), with credibility
    as in a crisis; partitions, stalemates and refusals change nothing generic."""

    if e.label not in ("a_victory", "b_victory", "a_withdrew", "b_withdrew"):
        return []
    if e.label.endswith("victory"):
        rule, side = e.rules["victory"], e.label[0]
    else:
        rule, side = e.rules["withdrawal"], OTHER[e.label[0]]  # the side that stayed wins
    winners, losers = e.sides[side], e.sides[OTHER[side]]
    items = []
    if e.region is not None:
        amount = rule["influence"] * e.scale
        items += [e.influence(w, amount, scale_by=f"side_share:{w}") for w in winners]
    return items + _credibility(winners, losers, rule)


GENERIC: dict[str, Callable[[_Ending], list[Any]]] = {
    "crisis": _crisis,
    "attrition": _attrition,
    "bargaining": _bargaining,
    "alignment": _alignment,
    "intervention": _intervention,
}


def generic_effects(
    family: str,
    label: str,
    config: Mapping[str, Any],
    region: str | None,
    contract: Mapping[str, Any],
    importance: float,
    dynamics: Mapping[str, float],
    sensitivity: Mapping[str, float],
) -> list[Any]:
    """The contract's effects for outcome *label* of an event (effect-language specs)."""

    scale = importance / float(contract["reference_importance"])
    ending = _Ending(
        label, contract[family], _sides(family, config), region, scale, dynamics, sensitivity
    )
    return GENERIC[family](ending)


def expand(
    raw_event: Mapping[str, Any],
    contract: Mapping[str, Any] | None,
    regions_doc: Mapping[str, Any],
    dynamics: Mapping[str, float],
    sensitivity: Mapping[str, float],
) -> Mapping[str, Any]:
    """*raw_event* with the contract's effects written into its outcomes (a copy), or the
    event itself when it does not opt in."""

    model = raw_event.get("model") or {}
    if model.get("effects") != "contract":
        return raw_event
    family = model["family"]
    if family not in FAMILIES:
        raise ContractError(f"event {raw_event['id']}: the contract covers {FAMILIES}")
    if not contract or family not in contract:
        raise ContractError(f"event {raw_event['id']}: no contract for {family}")
    region = model.get("region")
    importance = float(regions_doc["regions"][region]["importance"]) if region else 0.0
    event = copy.deepcopy(dict(raw_event))
    for label, spec in event["outcomes"].items():
        if "effects" in spec:
            raise ContractError(
                f"event {event['id']}: outcome {label} lists effects; under the contract it "
                "lists only additions"
            )
        generated = generic_effects(
            family, label, model["config"], region, contract, importance, dynamics, sensitivity
        )
        spec["effects"] = generated + list(spec.pop("additions", []) or [])
    return event


__all__ = [
    "ContractError",
    "dynamics_of",
    "expand",
    "generic_effects",
    "sensitivity_of",
    "war_costs",
]
