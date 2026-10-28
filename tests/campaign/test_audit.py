"""Scripted doctrines for the strategy-layer audit."""

from __future__ import annotations

import numpy as np
import pytest

from cold_war_sim.campaign import CampaignEngine, Catalog, EngineConfig, audit
from cold_war_sim.campaign.policies import DOCTRINES, DoctrinePolicy


def test_doctrines_choose_their_strategy_and_defer_elsewhere(fixture_catalog: Catalog) -> None:
    engine = CampaignEngine(fixture_catalog, EngineConfig(npc_policy="equilibrium"))
    engine.reset(0)
    rng = np.random.default_rng(0)
    policy = DoctrinePolicy("butter")
    chosen: dict[str, str] = {}
    while not engine.done and len(chosen) < 3:
        request = engine.pending()
        assert request is not None
        index = policy.choose(engine, request, rng)
        if request.source == "strategy" and request.power == "USA":
            chosen[request.kind] = request.options[index].id
        engine.act(index)
    assert chosen["strategy_posture"] == "economy"
    assert chosen["strategy_region"] == "none"
    assert chosen["strategy_diplomacy"] in ("engage:USSR", "none")  # none if the USSR is gone
    with pytest.raises(ValueError):
        DoctrinePolicy("pacifism")


def test_the_audit_counts_its_campaigns() -> None:
    assert audit.job_count([0, 1], 3) == 5 * len(DOCTRINES) * 2 * 4
