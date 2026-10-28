"""The 1945-1991 campaign: world state, event catalog, engine and environment."""

from .catalog import Catalog, CatalogError, load_catalog
from .decisions import MAX_OPTIONS, DecisionRequest, Option
from .engine import CampaignEngine, EngineConfig, EpisodeResult, run_episode
from .env import ColdWarEnv
from .policies import EquilibriumPolicy, HistoricalPolicy, RandomPolicy, make_policy
from .state import POWERS, REGIONS, WorldState

__all__ = [
    "MAX_OPTIONS",
    "POWERS",
    "REGIONS",
    "CampaignEngine",
    "Catalog",
    "CatalogError",
    "ColdWarEnv",
    "DecisionRequest",
    "EngineConfig",
    "EpisodeResult",
    "EquilibriumPolicy",
    "HistoricalPolicy",
    "Option",
    "RandomPolicy",
    "WorldState",
    "load_catalog",
    "make_policy",
    "run_episode",
]
