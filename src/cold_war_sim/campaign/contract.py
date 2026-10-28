"""The shared effects contract: generic outcome effects per template (``data/contract.yaml``).

An event with ``model.effects: contract`` lists only documented ``additions`` on its
outcomes; :func:`expand` writes the contract's generic effects in front of them, in the
effect language, before the event is compiled. The raw documents the catalog keeps (and
the assumptions register reads) are never changed, so each contract number lives in one
place.

* **crisis:** ``a_prevails``/``b_prevails`` move influence to the winning side's powers
  and credibility up for them and down for the losing side's; ``compromise`` eases
  tension; ``limited_war`` (a war not played out) costs every power on either side a
  year of full war effort fought with its own troops, as the dynamics would charge it
  (:func:`war_costs`, with the domestic cost of casualties), plus
  forces, the region's stability, relations between the sides' leading powers and
  tension; ``nuclear_war`` ends the campaign.
* **attrition:** a victory or the other side's withdrawal moves influence to the winning
  side's powers, split by their share of the side's effort (``side_share``), with
  credibility as above; partitions, stalemates and refusals change nothing generic.
* **bargaining** (E48): an agreement warms the two parties' relations and eases tension;
  a breakdown does the reverse; impasses and vetoes change nothing generic.
* **alignment** (E48): ``aligned_<P>`` moves influence in the region toward patron P, with
  credibility up for P and down for the other patrons; ``nonaligned`` changes nothing.
* **intervention** (E48): ``crushed`` keeps the patron's hold (its credibility up, its
  relations with the rival down); ``defected`` moves influence to the rival (the patron's
  credibility down); ``reformed`` moves a little; ``quagmire`` costs the patron a war's
  costs; ``clash`` costs both sides as a crisis's unplayed war does.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from .dynamics import DEFAULTS, DT, WAR_EFFORT_SCALE
from .state import POWERS

FAMILIES = ("crisis", "attrition", "bargaining", "alignment", "intervention")


class ContractError(ValueError):
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


def _sides(family: str, config: Mapping[str, Any]) -> dict[str, list[str]]:
    """The powers on each side, in config order."""

    if family == "bargaining":
        roles = config["roles"]
        return {s: [roles[k]] if roles[k] in POWERS else [] for s, k in (("a", "A"), ("b", "B"))}
    if family == "intervention":
        roles = config["roles"]
        return {
            s: [roles[k]] if roles[k] in POWERS else []
            for s, k in (("a", "patron"), ("b", "rival"))
        }
    if family == "alignment":
        return {"a": [p for p in config["patrons"] if p in POWERS], "b": []}
    entries = config["roles"] if family == "crisis" else config["sponsors"]
    sides: dict[str, list[str]] = {"a": [], "b": []}
    for role, spec in entries.items():
        side = spec.get("side")
        if role in POWERS and side in sides:
            sides[side].append(role)
    return sides


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

    rules = contract[family]
    sides = _sides(family, config)
    other = {"a": "b", "b": "a"}
    scale = importance / float(contract["reference_importance"])
    if family == "crisis":
        if label in ("a_prevails", "b_prevails"):
            rule = rules["prevails"]
            side = label[0]
            winners, losers = sides[side], sides[other[side]]
            items: list[Any] = []
            if region is not None and winners:
                amount = rule["influence"] * scale / len(winners)
                items += [
                    {"shift_influence": {"region": region, "toward": w, "amount": amount}}
                    for w in winners
                ]
            return items + _credibility(winners, losers, rule)
        if label == "compromise":
            return [{"tension": rules["compromise"]["tension"]}]
        if label == "limited_war":
            return _war(
                rules["limited_war"], sides["a"] + sides["b"], sides, region, dynamics, sensitivity
            )
        if label == "nuclear_war":
            return [{"end_campaign": "nuclear_war"}]
        return []
    if family == "bargaining":
        rule = rules.get(label)
        if rule is None or not (sides["a"] and sides["b"]):
            return []
        return [
            {"relations": {"a": sides["a"][0], "b": sides["b"][0], "add": rule["relations"]}},
            {"tension": rule["tension"]},
        ]
    if family == "alignment":
        if not label.startswith("aligned_"):
            return []
        rule, winner = rules["aligned"], label.removeprefix("aligned_")
        losers = [p for p in sides["a"] if p != winner]
        items = []
        if region is not None and winner in POWERS:
            items.append(
                {
                    "shift_influence": {
                        "region": region,
                        "toward": winner,
                        "amount": rule["influence"] * scale,
                    }
                }
            )
        return items + _credibility([winner] if winner in POWERS else [], losers, rule)
    if family == "intervention":
        patron, rival = sides["a"], sides["b"]
        rule = rules.get(label)
        if rule is None:
            return []
        if label == "crushed":
            items = _credibility(patron, [], rule)
            if patron and rival:
                items.append(
                    {"relations": {"a": patron[0], "b": rival[0], "add": rule["relations"]}}
                )
            return items
        if label in ("defected", "reformed"):
            items = []
            if region is not None and rival:
                items.append(
                    {
                        "shift_influence": {
                            "region": region,
                            "toward": rival[0],
                            "amount": rule["influence"] * scale,
                        }
                    }
                )
            if label == "defected":
                items += _credibility(rival, patron, rule)
            return items
        if label == "quagmire":
            items = _war(rule, patron, sides, region, dynamics, sensitivity, relations=False)
            return items + _credibility([], patron, rule)
        if label == "clash":
            return _war(rule, patron + rival, sides, region, dynamics, sensitivity)
        return []
    if label in ("a_victory", "b_victory", "a_withdrew", "b_withdrew"):
        if label.endswith("victory"):
            rule, side = rules["victory"], label[0]
        else:
            rule, side = rules["withdrawal"], other[label[0]]  # the side that stayed wins
        winners, losers = sides[side], sides[other[side]]
        items = []
        if region is not None:
            items += [
                {
                    "shift_influence": {
                        "region": region,
                        "toward": w,
                        "amount": rule["influence"] * scale,
                        "scale_by": f"side_share:{w}",
                    }
                }
                for w in winners
            ]
        return items + _credibility(winners, losers, rule)
    return []


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
