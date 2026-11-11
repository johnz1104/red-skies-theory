"""Coercive crisis: an escalation ladder with risk left to chance.

Two sides (``a`` and ``b``) contest a stake; neutral roles may pressure either. The game
is a script of stages: private types, signals about them, public messages, moves on
the ladder, deals, third-party pressure, accidents and delegated incidents.

Outcomes: ``a_prevails``, ``b_prevails``, ``compromise``, ``limited_war``,
``nuclear_war``. A role's payoff is its value of the outcome less the costs of its
escalation, signals, audience and pressure, and the costs options list for it.
"""

from __future__ import annotations

from .game import STAGES, CrisisBeliefPolicy, CrisisGame, CrisisParams, CrisisState, build
from .params import MOVE_KINDS, NAME, OUTCOMES, SIDES, STANCES, WAR_OUTCOMES, CrisisRole

__all__ = [
    "MOVE_KINDS",
    "NAME",
    "OUTCOMES",
    "SIDES",
    "STAGES",
    "STANCES",
    "WAR_OUTCOMES",
    "CrisisBeliefPolicy",
    "CrisisGame",
    "CrisisParams",
    "CrisisRole",
    "CrisisState",
    "build",
]
