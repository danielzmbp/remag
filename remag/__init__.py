"""
REMAG: Recovery of eukaryotic genomes using contrastive learning
"""

try:
    from ._version import __version__
except ImportError:
    # Fallback for Bioconda/conda installations without git
    __version__ = "0.4.1"

__author__ = "Daniel Gómez-Pérez"
__email__ = "daniel.gomez-perez@earlham.ac.uk"

try:
    from .cli import main_cli
    from .core import main

    __all__ = ["main", "main_cli"]
except ImportError:
    __all__ = []
