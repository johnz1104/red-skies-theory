"""Fixed-width observation vectors and readable observation dicts.

The vector layout is fixed by the catalog (its flag list) and published as
``feature_names``. Each power sees its own state exactly and the other powers
through the same public fields. A context block describes the pending
decision: its kind, round, option count and the family's own observation of
the sub-game (padded to ``FAMILY_FEATURES``), which encodes only what that
role is allowed to know.

The family's observation is a summary, and several information sets can share it
(E44: Moscow's own stakes at Suez, other roles' earlier moves). The last block therefore
carries the role's information state itself: its most recent ``TOKEN_SLOTS`` log tokens,
newest first, each as a hashed id in 1..``TOKEN_SPACE`` - 1 (0 = empty). The ids are
categories, not magnitudes: a network embeds them (``rl.ppo``).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from . import calendar
from .decisions import DECISION_KINDS, MAX_OPTIONS
from .rng import stable_key
from .state import POSTURES, POWERS, REGIONS, STATUSES

if TYPE_CHECKING:
    from .engine import CampaignEngine

FAMILY_FEATURES = 19
TOKEN_SLOTS = 16  # the information state's most recent log tokens
TOKEN_SPACE = 4096  # hashed token ids; 0 marks an empty slot
OWN_FIELDS = (
    "growth",
    "conventional",
    "nuclear",
    "tech_nuclear",
    "tech_missile",
    "tech_space",
    "stability",
    "credibility",
    "prestige",
    "hawkishness",
    "risk_tolerance",
    "reformism",
)
SCALES = {
    "growth": 10.0,
    "conventional": 0.1,
    "nuclear": 0.05,
    "tech_nuclear": 1 / 3,
    "tech_missile": 1 / 3,
    "tech_space": 1 / 3,
}


def feature_names(flags: tuple[str, ...]) -> tuple[str, ...]:
    names = ["global.progress", "global.quarter", "global.tension", "global.arms_control"]
    names += [f"flag.{f}" for f in flags]
    names += [f"self.status.{s}" for s in STATUSES]
    names += ["self.log_economy"] + [f"self.{f}" for f in OWN_FIELDS]
    names += [f"self.posture.{p}" for p in POSTURES] + ["self.has_focus"]
    for i in range(len(POWERS) - 1):
        names += [
            f"other{i}.active",
            f"other{i}.insurgent",
            f"other{i}.eliminated",
            f"other{i}.log_economy",
            f"other{i}.conventional",
            f"other{i}.nuclear",
            f"other{i}.tech_missile",
            f"other{i}.stability",
            f"other{i}.prestige",
            f"other{i}.relation",
        ]
    for region in REGIONS:
        names += [f"region.{region}.influence.{p}" for p in POWERS]
        names += [
            f"region.{region}.stability",
            f"region.{region}.conflict",
            f"region.{region}.own_colony",
            f"region.{region}.controlled_by_self",
        ]
    names += [f"context.kind.{k}" for k in DECISION_KINDS]
    names += ["context.round", "context.options", "context.active_events"]
    names += [f"context.family.{i}" for i in range(FAMILY_FEATURES)]
    names += [f"context.token.{i}" for i in range(TOKEN_SLOTS)]
    return tuple(names)


def token_id(token: str) -> int:
    """A log token's id in 1..TOKEN_SPACE - 1 (stable across runs and processes)."""

    return stable_key(token) % (TOKEN_SPACE - 1) + 1


def encode(engine: CampaignEngine, power: str) -> np.ndarray:
    world = engine.world
    flags = engine.observed_flags
    values: list[float] = [
        calendar.progress(world.quarter),
        (calendar.quarter_of_year(world.quarter) - 1) / 3.0,
        world.tension,
        world.arms_control,
    ]
    values += [1.0 if f in world.flags else 0.0 for f in flags]
    own = world.powers[power]
    values += [1.0 if own.status == s else 0.0 for s in STATUSES]
    values.append(math.log(own.economy))
    values += [getattr(own, f) * SCALES.get(f, 1.0) for f in OWN_FIELDS]
    values += [1.0 if own.posture == p else 0.0 for p in POSTURES]
    values.append(1.0 if own.focus_region else 0.0)
    for other in POWERS:
        if other == power:
            continue
        state = world.powers[other]
        values += [
            1.0 if state.status == "active" else 0.0,
            1.0 if state.status == "insurgent" else 0.0,
            1.0 if state.status == "eliminated" else 0.0,
            math.log(state.economy),
            state.conventional * 0.1,
            state.nuclear * 0.05,
            state.tech_missile / 3.0,
            state.stability,
            state.prestige,
            own.relations.get(other, 0.0),
        ]
    for region_id in REGIONS:
        region = world.regions[region_id]
        values += [region.influence[p] for p in POWERS]
        values += [
            region.stability,
            1.0 if region.conflict else 0.0,
            1.0 if region.colonial_owner == power else 0.0,
            1.0 if region.controller == power else 0.0,
        ]
    request = engine.pending()
    kind = [0.0] * len(DECISION_KINDS)
    context = [0.0, 0.0, len(world.active) / 10.0]
    family = [0.0] * FAMILY_FEATURES
    tokens = [0.0] * TOKEN_SLOTS
    if request is not None and request.power == power:
        kind[DECISION_KINDS.index(request.kind)] = 1.0
        context[0] = min(1.0, request.round / 20.0)
        context[1] = len(request.options) / MAX_OPTIONS
        if request.role is not None:
            features = engine.subgame_observation(request.source, request.role)
            family[: len(features)] = features[:FAMILY_FEATURES]
            recent = engine.subgame_information(request.source, request.role)[::-1]
            tokens[: min(len(recent), TOKEN_SLOTS)] = [
                float(token_id(t)) for t in recent[:TOKEN_SLOTS]
            ]
    values += kind + context + family + tokens
    return np.asarray(values, dtype=np.float32)


def describe(engine: CampaignEngine, power: str) -> dict[str, Any]:
    """A readable observation for debugging, logging and language-model agents."""

    world = engine.world
    own = world.powers[power]
    request = engine.pending()
    view: dict[str, Any] = {
        "date": calendar.fmt(world.quarter),
        "tension": round(world.tension, 3),
        "power": power,
        "status": own.status,
        "leader": own.leader,
        "economy": round(own.economy, 3),
        "conventional": round(own.conventional, 3),
        "nuclear": round(own.nuclear, 3),
        "stability": round(own.stability, 3),
        "prestige": round(own.prestige, 3),
        "relations": {k: round(v, 2) for k, v in own.relations.items()},
        "active_events": sorted(world.active),
        "recent": [f"{calendar.fmt(q)} {text}" for q, _, text in world.log[-5:]],
    }
    if request is not None and request.power == power:
        view["decision"] = {
            "kind": request.kind,
            "source": request.source,
            "role": request.role,
            "options": [o.id for o in request.options],
        }
    return view
