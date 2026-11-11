"""Alternating-offers bargaining under a risk of breakdown.

Two sides divide a stake; ``share`` always means side A's share. Each period the
proposer offers a share from a grid and the responder accepts or rejects; a rejection
risks breakdown, and the horizon ends the talks in impasse (the status quo). Optional
extensions: commitments before the talks, consent roles that ratify or veto, a contest
between periods (bargaining while fighting), and hidden resolve.

Outcomes: ``agreement`` (metric ``share``), ``breakdown``, ``impasse``, ``vetoed``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .game import (
    NAME,
    OUTCOMES,
    SELECTIONS,
    SIDE_ROLES,
    STRONG,
    TOLERANCE,
    TYPES,
    WEAK,
    BargainingGame,
    BargainingParams,
    BargainingState,
    ConsentParams,
    ContestParams,
    SideParams,
    TypeParams,
    side_payoff,
)
from .solve import (
    BargainingEquilibriumPolicy,
    BargainingSolution,
    LogitSolution,
    ratify_choices,
    ratify_probabilities,
    solve,
    solve_logit,
)


def build(config: Mapping[str, Any]) -> BargainingGame:
    """The bargaining game of *config* (keys: :class:`BargainingParams`)."""

    return BargainingGame(BargainingParams.from_config(config))


__all__ = [
    "NAME",
    "OUTCOMES",
    "SELECTIONS",
    "SIDE_ROLES",
    "STRONG",
    "TOLERANCE",
    "TYPES",
    "WEAK",
    "BargainingEquilibriumPolicy",
    "BargainingGame",
    "BargainingParams",
    "BargainingSolution",
    "BargainingState",
    "ConsentParams",
    "ContestParams",
    "LogitSolution",
    "SideParams",
    "TypeParams",
    "build",
    "ratify_choices",
    "ratify_probabilities",
    "side_payoff",
    "solve",
    "solve_logit",
]
