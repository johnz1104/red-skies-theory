"""Cold War Simulator: finite strategic models under uncertainty."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cold-war-sim")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0.1.0"

__all__ = ["__version__"]
