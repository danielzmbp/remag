"""
REMAG: Recovery of Eukaryotic Metagenome-Assembled Genomes using contrastive learning
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("remag")
except PackageNotFoundError:
    # The package metadata is unavailable when importing directly from a source tree.
    __version__ = "0+unknown"

__author__ = "Daniel Gómez-Pérez"
__email__ = "daniel.gomez-perez@earlham.ac.uk"

__all__ = ["main", "main_cli"]


def __getattr__(name):
    if name == "main_cli":
        from .cli import main_cli

        return main_cli
    if name == "main":
        from .core import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
