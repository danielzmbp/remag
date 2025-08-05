"""
REMAG: Recovery of eukaryotic genomes using contrastive learning
"""

try:
    from ._version import __version__
except ImportError:
    # Fallback for development or when _version.py doesn't exist
    __version__ = "0.0.0+unknown"

__author__ = "Daniel Gómez-Pérez"
__email__ = "daniel.gomez-perez@earlham.ac.uk"

try:
    from .core import main
    from .cli import main_cli
    from .xgbclass import xgbClass
    __all__ = ["main", "main_cli", "xgbClass"]
except ImportError:
    __all__ = []
