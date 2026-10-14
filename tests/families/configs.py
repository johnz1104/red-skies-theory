"""Representative family configurations used across the family tests."""

from __future__ import annotations

from typing import Any


def crisis_values(ap: float, bp: float, c: float, lw: float, nw: float) -> dict[str, float]:
    return {
        "a_prevails": ap,
        "b_prevails": bp,
        "compromise": c,
        "limited_war": lw,
        "nuclear_war": nw,
    }


def signaling_crisis() -> dict[str, Any]:
    """Private resolve and readiness, noisy intelligence, revisable signals, a deal."""

    return {
        "roles": {
            "A": {
                "side": "a",
                "values": crisis_values(2, -2, 0.5, -6, -40),
                "escalation_cost": 0.2,
                "audience_cost": 1.0,
            },
            "B": {
                "side": "b",
                "values": crisis_values(-2, 2, 0.5, -6, -40),
                "escalation_cost": 0.2,
                "audience_cost": 1.5,
                "high_resolve_war_factor": 0.5,
            },
        },
        "script": [
            {"stage": "types", "roles": {"B": {"p_high_resolve": 0.4, "p_high_readiness": 0.5}}},
            {"stage": "intel", "about": "B", "observers": ["A"], "accuracy": 0.75},
            {
                "stage": "move",
                "actor": "A",
                "options": [
                    {"id": "blockade", "kind": "escalate", "rungs": 1},
                    {"id": "strike", "kind": "escalate", "rungs": 2},
                ],
            },
            {
                "stage": "accident",
                "per_rung": 0.02,
                "nuclear_per_rung": 0.1,
                "readiness_multiplier": 1.5,
            },
            {
                "stage": "signal",
                "actor": "B",
                "options": [
                    {"id": "conciliatory", "stance": "soft"},
                    {"id": "hardline", "stance": "hard"},
                ],
            },
            {
                "stage": "signal",
                "actor": "B",
                "options": [
                    {"id": "reaffirm", "stance": "keep"},
                    {"id": "revise", "stance": "flip", "cost": 0.1},
                ],
            },
            {
                "stage": "move",
                "actor": "A",
                "options": [
                    {"id": "negotiate", "kind": "deal"},
                    {"id": "maintain_pressure", "kind": "hold"},
                    {"id": "escalate", "kind": "escalate"},
                ],
            },
            {"stage": "respond", "actor": "B"},
            {
                "stage": "move",
                "actor": "B",
                "options": [
                    {"id": "withdraw", "kind": "concede"},
                    {"id": "stand_firm", "kind": "hold"},
                ],
            },
            {"stage": "accident", "base": 0.01, "per_rung": 0.03, "nuclear_per_rung": 0.1},
        ],
    }


def warning_crisis() -> dict[str, Any]:
    """A privately typed defender warns; the challenger chooses how far to advance."""

    return {
        "roles": {
            "A": {"side": "a", "values": crisis_values(2, 0, 0, -4, -30), "escalation_cost": 0.1},
            "B": {
                "side": "b",
                "values": crisis_values(-2, 1, 0, -5, -30),
                "audience_cost": 1.0,
                "high_resolve_war_factor": 0.4,
            },
        },
        "script": [
            {"stage": "types", "roles": {"B": {"p_high_resolve": 0.5}}},
            {
                "stage": "signal",
                "actor": "B",
                "options": [{"id": "quiet", "stance": "soft"}, {"id": "warn", "stance": "hard"}],
            },
            {
                "stage": "move",
                "actor": "A",
                "options": [
                    {"id": "restraint", "kind": "concede"},
                    {"id": "limited_advance", "kind": "escalate", "rungs": 1},
                    {"id": "full_advance", "kind": "escalate", "rungs": 2},
                ],
            },
            {
                "stage": "move",
                "actor": "B",
                "options": [
                    {"id": "stay_out", "kind": "concede"},
                    {"id": "intervene", "kind": "fight"},
                ],
            },
        ],
    }


def coalition_crisis() -> dict[str, Any]:
    """Two-member coalitions on each side and a neutral role that can apply pressure."""

    return {
        "roles": {
            "A1": {"side": "a", "values": crisis_values(1.5, -1, 0, -2, -30)},
            "A2": {"side": "a", "values": crisis_values(1.5, -1, 0, -2, -30)},
            "B1": {"side": "b", "values": crisis_values(-2, 1, 0, -2, -30)},
            "B2": {
                "side": "b",
                "values": crisis_values(-1, 1, 0, -3, -30),
                "escalation_cost": 0.3,
            },
            "N": {"side": "neutral", "values": crisis_values(-0.5, 0.3, 0.5, -2, -30)},
        },
        "script": [
            {
                "stage": "move",
                "actor": "A1",
                "options": [
                    {"id": "refrain", "kind": "concede"},
                    {"id": "invade", "kind": "escalate"},
                ],
            },
            {
                "stage": "move",
                "actor": "A2",
                "options": [{"id": "stay_out", "kind": "hold"}, {"id": "join", "kind": "escalate"}],
            },
            {
                "stage": "signal",
                "actor": "B2",
                "options": [
                    {"id": "silent", "stance": "soft"},
                    {"id": "threaten", "stance": "hard", "cost": 0.1},
                ],
            },
            {
                "stage": "partner",
                "actor": "N",
                "options": [
                    {"id": "back_side_b", "target": "b", "risk_multiplier": 0.8},
                    {"id": "pressure_side_a", "target": "a", "continue_cost": 1.5},
                ],
            },
            {
                "stage": "move",
                "actor": "A1",
                "options": [
                    {"id": "withdraw", "kind": "concede"},
                    {"id": "continue", "kind": "hold"},
                ],
            },
            {"stage": "accident", "base": 0.02, "per_rung": 0.03, "nuclear_per_rung": 0.05},
        ],
        "final_outcome": "a_prevails",
    }


def bargaining_config(
    horizon: int = 3, commitments: bool = True, consent: bool = True
) -> dict[str, Any]:
    side_a: dict[str, Any] = {
        "discount": 0.9,
        "delay_cost": 0.05,
        "breakdown_value": -3,
        "impasse_value": 0.2,
    }
    side_b: dict[str, Any] = {
        "discount": 0.8,
        "delay_cost": 0.1,
        "breakdown_value": -3,
        "impasse_value": 0.3,
    }
    if commitments:
        side_a |= {"commitment": 0.75, "commitment_cost": 1.0}
        side_b |= {"commitment": 0.25, "commitment_cost": 1.0}
    config: dict[str, Any] = {
        "roles": {"A": "P1", "B": "P2"},
        "horizon": horizon,
        "surplus": 2.0,
        "sides": {"A": side_a, "B": side_b},
        "breakdown": {"base": 0.05, "growth": 0.05, "max": 0.5},
    }
    if consent:
        config["consent"] = [{"role": "C", "base": -0.2, "per_share": 1.0, "veto_value": 0.0}]
    return config


def intervention_values(r: float, c: float, d: float, q: float, cl: float) -> dict[str, float]:
    return {"reformed": r, "crushed": c, "defected": d, "quagmire": q, "clash": cl}


def intervention_config() -> dict[str, Any]:
    return {
        "roles": {"patron": "patron", "rival": "rival"},
        "p_strong": 0.5,
        "quagmire_prob": 0.2,
        "values": {
            "patron": intervention_values(0.2, 1, -2, -1, -8),
            "rival": intervention_values(0.3, -0.5, 1.5, 0.5, -8),
        },
        "costs": {
            "crackdown": 0.1,
            "intervene": 0.5,
            "back_down": 0.5,
            "condemn": 0.05,
            "covert_aid": 0.2,
            "confront": 0.5,
        },
    }


def attrition_values(
    av: float, bv: float, part: float, aw: float, bw: float, st: float
) -> dict[str, float]:
    return {
        "a_victory": av,
        "b_victory": bv,
        "partition": part,
        "a_withdrew": aw,
        "b_withdrew": bw,
        "stalemate": st,
    }


def attrition_config(max_rounds: int = 12) -> dict[str, Any]:
    return {
        "sponsors": {
            "a1": {
                "side": "a",
                "capacity": 1.0,
                "direct": True,
                "values": attrition_values(3, -3, 0, -2, 2, -1),
            },
            "b1": {"side": "b", "capacity": 0.5, "values": attrition_values(-2, 2, 0, -1, 1, 0)},
            "b2": {
                "side": "b",
                "capacity": 0.8,
                "direct": True,
                "values": attrition_values(-3, 3, 0.5, -2, 2, 0),
            },
        },
        "leads": {"a": "a1", "b": "b2"},
        "local_strength": {"a": 0.8, "b": 1.0},
        "max_rounds": max_rounds,
        "weariness": {"high": 0.05, "direct": 0.2},
        "weariness_limit": 1.0,
    }


def alignment_config() -> dict[str, Any]:
    return {
        "patrons": {
            "p1": {"gain": {"c1": 1.0, "c2": 1.0, "c3": 1.5}, "loss": {"c3": 0.5}},
            "p2": {"gain": {"c3": 2.0, "c2": 1.0}, "loss": {"c1": 0.5, "c2": 1.0}},
        },
        "clients": {
            "c1": {"mode": "player", "affinity": {"p1": 0.8, "p2": -1.0}, "nonaligned": 0.3},
            "c2": {"mode": "player", "affinity": {"p1": 0.4, "p2": -0.3}, "nonaligned": 0.6},
            "c3": {
                "mode": "chance",
                "affinity": {"p1": 0.2, "p2": 0.6},
                "nonaligned": 0.0,
                "temperature": 0.5,
            },
        },
    }


def attrition_offers_config(
    max_rounds: int = 1,
    *,
    approval: bool = False,
    consent: bool = False,
    red_line: bool = False,
    opens: int = 0,
    commitments: bool = False,
    levels: bool = False,
    entrant: bool = False,
    proxy: float = 0.0,
    hidden: tuple[float, float] | None = None,
    momentum: float = 0.0,
    unseen: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """A two-sponsor war that bargains while fighting: A's resolve and B's strength
    are private, the front can hold, and A offers a share each round. With
    *commitments*, A may commit to at least 0.7 and B to at least half. With *levels*,
    each side has three kinds instead of two. With *entrant*, a third power ``c1`` on
    B's side has stakes of its own (committed or not) and leads B once it goes direct.
    A positive *proxy* gives the consent role ``b2`` that gain per round in which another
    B sponsor fights direct. *hidden* = (nature's weight, others' belief) makes that
    gain a private type of ``b2``: it holds out (gain in full) or settles (none). A
    positive *momentum* shifts the local balance with the front, on a finer step (five
    cells, the starting line on the grid) so that it matters within two rounds.
    *unseen* = (A's belief, B's own belief) that B is strong makes B's strength a type
    no one sees."""

    def values(av: float, bv: float, aw: float, bw: float, refused: float) -> dict[str, float]:
        table = attrition_values(av, bv, 0.0, aw, bw, 0.0)
        if approval:
            table["refused"] = refused
        return table

    config: dict[str, Any] = {
        "sponsors": {
            "a1": {
                "side": "a",
                "capacity": 1.0,
                "direct": red_line,
                "values": values(2, -2, -2.5, 2, 0.0),
                "settlement": {"base": -1.5, "per_share": 3.0},
            },
            "b1": {
                "side": "b",
                "capacity": 1.0,
                "direct": red_line,
                "values": values(-2, 2, 2, -2.5, -0.2),
                "settlement": {"base": -1.5, "per_share": 3.0},
            },
        },
        "local_strength": {"a": 1.0, "b": 1.0},
        "initial_control": 0.5,
        "step": 0.4,
        "victory_threshold": 0.85,
        "max_rounds": max_rounds,
        "weariness": {"high": 0.0, "direct": 0.0},
        "costs": {"low": 0.1, "high": 0.3, "direct": 0.6},
        "offers": {"grid": [0.3, 0.5, 0.7], "proposer": "a", "opens": opens},
        "types": {
            "a": {"prior": 0.5, "strong": {"cost": 0.5}, "weak": {"cost": 3.0}},
            "b": {"prior": 0.5, "strong": {"strength": 1.6}, "weak": {"strength": 0.6}},
        },
        "mobility": {"space": 3.0, "sharpness": 2.0},
    }
    if approval:
        config["approval"] = ["b1"]
    if consent:
        config["sponsors"]["b2"] = {
            "side": "b",
            "capacity": 0.0,
            "values": values(-1, 1, 1, -1, 0.0),
            "settlement": {"base": -0.5, "per_share": 0.5},
        }
        config["offers"]["consent"] = [{"role": "b2", "veto_cost": 0.1}]
        if proxy > 0.0:
            config["sponsors"]["b2"]["proxy_gain"] = proxy
        if hidden is not None:
            weight, belief = hidden
            config["types"]["b2"] = {
                "private": True,
                "levels": {
                    "holds_out": {"weight": weight, "belief": belief, "proxy": 1.0},
                    "settles": {"weight": 1.0 - weight, "belief": 1.0 - belief, "proxy": 0.0},
                },
            }
    if red_line:
        config["red_lines"] = {"b1": {"above": 0.5, "audience_cost": 0.5}}
    if levels:
        config["types"] = {
            "a": {
                "levels": {
                    "resolute": {"weight": 1.0, "cost": 0.5},
                    "firm": {"weight": 1.0, "cost": 1.5},
                    "wavering": {"weight": 1.0, "cost": 3.0},
                }
            },
            "b": {
                "levels": {
                    "strong": {"weight": 1.0, "strength": 1.6},
                    "steady": {"weight": 2.0, "strength": 1.0},
                    "weak": {"weight": 1.0, "strength": 0.6},
                }
            },
        }
    if entrant:
        config["sponsors"]["c1"] = {
            "side": "b",
            "capacity": 1.0,
            "direct": True,
            "values": values(-1.5, 1.5, 1.5, -1.5, 0.0),
            "settlement": {"base": -1.0, "per_share": 2.0},
        }
        config["types"]["c1"] = {
            "levels": {
                "committed": {"weight": 1.0, "cost": 1.0},
                "uncommitted": {"weight": 1.0, "cost": 6.0},
            }
        }
        config["lead_on_entry"] = {"c1": "b"}
    if unseen is not None:
        other, own = unseen
        config["types"]["b"] = {
            "unseen": True,
            "levels": {
                "strong": {"weight": 0.5, "belief": other, "own_belief": own, "strength": 1.6},
                "weak": {
                    "weight": 0.5,
                    "belief": 1.0 - other,
                    "own_belief": 1.0 - own,
                    "strength": 0.6,
                },
            },
        }
    if momentum > 0.0:
        config["momentum"] = momentum
        config["step"] = 0.175
    if commitments:
        config["commitments"] = {
            "a1": {"at_least": 0.7, "audience_cost": 0.5},
            "b1": {"at_least": 0.5, "audience_cost": 0.5},
        }
    return config
