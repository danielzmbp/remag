"""
REMAG: Recovery of Eukaryotic Metagenome-Assembled Genomes using contrastive learning
"""

try:
    from ._version import __version__
except ImportError:
    # Fallback for Bioconda/conda installations without git
    __version__ = "0.4.3"

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
