#!/usr/bin/env python3
"""
REMAG entry point script.

This script allows running REMAG as 'python remag.py' while maintaining
compatibility with the installed package.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from remag.cli import main_cli

if __name__ == "__main__":
    main_cli()