"""Proxy war as a stochastic war of attrition between sponsor coalitions.

Two local sides fight over *control* (side A's share of the ground, in [0, 1]). Each
round every sponsor secretly chooses an effort level; the choices are revealed together
and one battle moves control a step toward its winner. Effort costs accumulate, and
weariness forces a sponsor out at its limit. Optional blocks turn the war into
bargaining while fighting: approval, offers, private types, mobility, red lines,
deployment lags, momentum, entry as lead and commitments.

Outcomes: ``a_victory``, ``b_victory``, ``partition``, ``a_withdrew``, ``b_withdrew``,
``stalemate``, and ``refused`` when the war needs approval.
"""

from __future__ import annotations

from .game import AttritionGame, AttritionState
from .params import (
    EFFORT,
    KINDS,
    LEVELS,
    NAME,
    OUTCOMES,
    PROPOSERS,
    REFUSED,
    STAGES,
    STRONG,
    WEAK,
    AttritionParams,
    Commitment,
    Consent,
    Mobility,
    OfferParams,
    RedLine,
    SideTypes,
    Sponsor,
    build,
    other_side,
)
from .policies import (
    AttritionBeliefPolicy,
    AttritionHeuristicPolicy,
    AttritionLookaheadPolicy,
    choose,
)

__all__ = [
    "EFFORT",
    "KINDS",
    "LEVELS",
    "NAME",
    "OUTCOMES",
    "PROPOSERS",
    "REFUSED",
    "STAGES",
    "STRONG",
    "WEAK",
    "AttritionBeliefPolicy",
    "AttritionGame",
    "AttritionHeuristicPolicy",
    "AttritionLookaheadPolicy",
    "AttritionParams",
    "AttritionState",
    "Commitment",
    "Consent",
    "Mobility",
    "OfferParams",
    "RedLine",
    "SideTypes",
    "Sponsor",
    "build",
    "choose",
    "other_side",
]
