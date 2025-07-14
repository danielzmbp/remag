"""
Entry point for python -m remag
"""

from .cli import main_cli

if __name__ == "__main__":
    main_cli.main(standalone_mode=False)
