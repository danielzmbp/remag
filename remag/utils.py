"""
Utility functions for REMAG
"""

import gzip
import os
import sys
from loguru import logger
from typing import Dict, List, Union


def setup_logging(output_dir=None, verbose=False):
    """Setup logging with optional file output."""
    logger.remove()
    log_level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        colorize=True,
    )
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        logger.add(
            os.path.join(output_dir, "remag.log"),
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            rotation="10 MB",
        )


def is_gzipped(file_path):
    """Check if a file is gzipped based on its extension."""
    return file_path.endswith(".gz")


def open_file(file_path, mode="r"):
    """Open a file, handling gzipped files if necessary."""
    if is_gzipped(file_path):
        return gzip.open(
            file_path, mode + "t" if "b" not in mode else mode, encoding="utf-8"
        )
    return open(file_path, mode, encoding="utf-8")


def fasta_iter(fasta_file):
    """Iterate over sequences in a FASTA file."""
    with open_file(fasta_file, "r") as f:
        header = ""
        seq = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    yield header, seq
                header = line.lstrip(">")  # Remove the ">" character from the header
                seq = ""
            else:
                seq += line
        if header:
            yield header, seq


import re

def extract_base_contig_name(fragment_header: str) -> str:
    """Extract the base contig name from a fragment header.

    Handles various fragment header formats:
    - contig.original -> contig
    - contig.h1.0 -> contig
    - contig.h2.1 -> contig
    - contig.0 -> contig

    Args:
        fragment_header: Fragment header string

    Returns:
        Base contig name without fragment suffixes
    """
    # Match patterns: .original, .h1.N, .h2.N, or .N (where N is a number)
    # First try to match the half identifier pattern: base.h1.N or base.h2.N
    match = re.match(r"(.+)\.h[12]\.\d+$", fragment_header)
    if match:
        return match.group(1)

    # Then try other patterns: base.original or base.N
    match = re.match(r"(.+)\.(?:\d+|original)$", fragment_header)
    if match:
        return match.group(1)

    # If no pattern matches, return as-is
    return fragment_header


# Type aliases for better code clarity
FragmentDict = Dict[str, Dict[str, Union[str, List[str]]]]
CoverageDict = Dict[str, float]
