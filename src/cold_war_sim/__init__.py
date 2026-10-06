"""Game-theory engine and strategic-interaction game families.

* :mod:`cold_war_sim.games`: a game-state protocol and exact tree algorithms
  (backward induction, best responses, NashConv, CFR+).
* :mod:`cold_war_sim.families`: crisis, bargaining, intervention, attrition and
  alignment games, configured from plain mappings.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cold-war-sim")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0.2.0"

__all__ = ["__version__"]
