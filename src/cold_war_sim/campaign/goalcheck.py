"""Tools for reviewing goal-derived payoffs: capture, solve, sweep, check.

* :func:`capture_worlds` replays the historical campaign and records the world
  and bound config at the moment each requested event starts.
* :func:`outcome_distribution` gives the baseline (equilibrium) outcome
  distribution of a bound config: exact for tree families, by simulated
  rollouts for bargaining (dynamic programming policy) and attrition
  (heuristic policy).
* :func:`variations` lists the sensitivity-sweep settings of GOALS.md §11.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from ..families import build_family
from ..games import CHANCE, build_tree, sample_action
from ..games import outcome_distribution as tree_distribution
from . import goals
from .catalog import Catalog, EventSpec
from .engine import CampaignEngine, EngineConfig, baseline_policy, reference_tree_policy
from .goals import GoalConfig
from .policies import make_policy
from .state import WorldState, identity

TREE_FAMILIES = ("crisis", "intervention", "alignment")
DEFAULT_TEMPERATURE = EngineConfig().baseline_temperature


def capture_worlds(
    catalog: Catalog, event_ids: Iterable[str], seed: int = 0
) -> dict[str, tuple[WorldState, dict[str, Any]]]:
    """World and bound config when each event starts in the historical replay."""

    wanted = set(event_ids)
    found: dict[str, tuple[WorldState, dict[str, Any]]] = {}
    engine = CampaignEngine(catalog, EngineConfig(chance_mode="historical"))
    policy = make_policy("historical")
    rng = np.random.default_rng(seed)
    engine.reset(seed)
    while not engine.done and wanted - set(found):
        for event_id, active in engine.s.active.items():
            if event_id in wanted and event_id not in found:
                found[event_id] = (copy.deepcopy(engine.world), copy.deepcopy(active.config))
        request = engine.pending()
        if request is None:
            break
        engine.act(policy.choose(engine, request, rng))
    return found


def outcome_distribution(
    family: str,
    config: Mapping[str, Any],
    *,
    iterations: int = 300,
    rollouts: int = 300,
    seed: int = 0,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict[str, float]:
    game = build_family(family, config)
    if family in TREE_FAMILIES:
        root = game.solver_state() if hasattr(game, "solver_state") else game.new_initial_state()
        tree = build_tree(root)
        policy = reference_tree_policy(game, tree, iterations, temperature)
        return tree_distribution(tree, policy)
    live = baseline_policy(family, config, iterations, temperature)
    rng = np.random.default_rng(seed)
    counts: Counter[str] = Counter()
    for _ in range(rollouts):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.current_player() == CHANCE:
                state.apply_action(sample_action(dict(state.chance_outcomes()), rng))
            else:
                dist = live.action_probabilities(state)
                state.apply_action(sample_action(dist, rng))
        counts[state.outcome().label] += 1
    return {label: n / rollouts for label, n in sorted(counts.items())}


def total_variation(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def variations(
    catalog: Catalog, event: EventSpec, world: WorldState
) -> list[tuple[str, GoalConfig]]:
    """Sweep settings: weights x0.5/x1.5 (row renormalized) and rivalry x0.5/x1.5 for
    each power role, the core multiplier at 1.5 and 3, the allies band height
    x0.5/x1.5, the buffer weight x0.5/x1.5, and kappa x0.5/x2."""

    major = event.major
    assert major is not None
    base = catalog.goals
    result: list[tuple[str, GoalConfig]] = []
    for power in major.power_roles():
        ident = identity(world, power)
        current = goals.row(world, ident, catalog)
        rows = list(base.rows[ident])
        index = rows.index(current)
        for component, weight in current.weights.items():
            if weight <= 0.0:
                continue
            for factor in (0.5, 1.5):
                scaled = dict(current.weights)
                scaled[component] = weight * factor
                total = sum(scaled.values())
                changed = list(rows)
                changed[index] = replace(current, weights={k: v / total for k, v in scaled.items()})
                config = replace(base, rows={**base.rows, ident: tuple(changed)})
                result.append((f"{ident} {component} x{factor}", config))
        for factor in (0.5, 1.5):
            rivalry = {**base.rivalry, ident: base.rivalry[ident] * factor}
            result.append((f"{ident} rivalry x{factor}", replace(base, rivalry=rivalry)))
    for multiplier in (1.5, 3.0):
        interest = {
            i: {r: (multiplier if m == 2.0 else m) for r, m in regions.items()}
            for i, regions in base.interest.items()
        }
        result.append((f"core multiplier {multiplier}", replace(base, interest=interest)))
    if base.allies is not None:
        for factor in (0.5, 1.5):
            allies = replace(base.allies, height=base.allies.height * factor)
            result.append((f"allies height x{factor}", replace(base, allies=allies)))
    if base.buffer_weight:
        for factor in (0.5, 1.5):
            weight = base.buffer_weight * factor
            result.append((f"buffer weight x{factor}", replace(base, buffer_weight=weight)))
    for factor in (0.5, 2.0):
        result.append((f"kappa x{factor}", replace(base, kappa=base.kappa * factor)))
    return result


__all__ = ["capture_worlds", "outcome_distribution", "total_variation", "variations"]
