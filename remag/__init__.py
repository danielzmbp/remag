"""
REMAG: Recovery of eukaryotic genomes using contrastive learning
"""

__version__ = "0.1.0"
__author__ = "Daniel Gómez-Pérez"
__email__ = "daniel.gomez-perez@earlham.ac.uk"

try:
    from .core import main
    from .cli import main_cli
    from .xgbclass import xgbClass
    __all__ = ["main", "main_cli", "xgbClass"]
except ImportError:
    __all__ = []
