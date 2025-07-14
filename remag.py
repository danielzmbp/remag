#!/usr/bin/env python3
"""
REMAG: Metagenomic binning using neural networks and contrastive learning

This is a standalone script version that can be run directly with:
python remag.py

For the package version, install with pip and use:
remag
"""

if __name__ == "__main__":
    from remag.cli import main_cli
    main_cli.main(standalone_mode=False)
