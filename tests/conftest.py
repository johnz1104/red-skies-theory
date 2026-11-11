from __future__ import annotations

from pathlib import Path

import pytest

from cold_war_sim.campaign import Catalog, load_catalog

FIXTURE_EVENTS = Path(__file__).with_name("fixture_events.yaml")


@pytest.fixture(scope="session")
def fixture_catalog() -> Catalog:
    return load_catalog(event_files=[FIXTURE_EVENTS])
