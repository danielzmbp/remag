"""
REMAG: Recovery of eukaryotic genomes using contrastive learning
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

try:
    from .core import main
    from .cli import main_cli
    from .xgbclass import xgbClass
    __all__ = ["main", "main_cli", "xgbClass"]
except ImportError:
    __all__ = []
