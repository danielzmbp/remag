"""
Utility functions for REMAG
"""

import gzip
import os
import re
import sys
from typing import Container, Dict, List, Optional, Union

import torch
from loguru import logger


def get_torch_device():
    """Get the appropriate torch device (CUDA, MPS, or CPU)."""
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    return device


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
        seq_lines = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    yield header, "".join(seq_lines)
                header = line.lstrip(">")  # Remove the ">" character from the header
                seq_lines = []
            else:
                seq_lines.append(line)
        if header:
            yield header, "".join(seq_lines)


def extract_base_contig_name(
    fragment_header: str, known_headers: Optional[Container[str]] = None
) -> str:
    """Extract the base contig name from a fragment header.

    Handles REMAG fragment header formats:
    - contig.original -> contig
    - contig.h1.0 -> contig
    - contig.h2.1 -> contig
    - contig.0 -> contig, only when contig.original is present in known_headers

    Args:
        fragment_header: Fragment header string
        known_headers: Optional collection of all fragment headers in the same set.
            Plain numeric suffixes are only stripped when the matching original
            contig header is present, avoiding SPAdes decimal coverage truncation.

    Returns:
        Base contig name without fragment suffixes
    """
    if fragment_header.endswith(".original"):
        return fragment_header[: -len(".original")]

    # Match REMAG half-fragment identifiers: base.h1.N or base.h2.N.
    match = re.match(r"(.+)\.h[12]\.\d+$", fragment_header)
    if match:
        return match.group(1)

    # Plain numeric suffixes are ambiguous because assembler headers may end in
    # decimals, e.g. SPAdes NODE_*_cov_5.491044. Only strip them when this is a
    # known REMAG fragment with a matching base.original header in the same set.
    match = re.match(r"(.+)\.\d+$", fragment_header)
    if match and known_headers is not None:
        base_name = match.group(1)
        if f"{base_name}.original" in known_headers:
            return base_name

    # If no pattern matches, return as-is
    return fragment_header


# Type aliases for better code clarity
FragmentDict = Dict[str, Dict[str, Union[str, List[str]]]]
CoverageDict = Dict[str, float]


class ContigHeaderMapper:
    """Efficient mapping between contig names and their headers in fragments_dict.

    This class eliminates the O(n*m) complexity of repeatedly searching through
    fragments_dict to find headers matching contig names.
    """

    def __init__(self, fragments_dict: FragmentDict):
        """Initialize the mapper with a fragments dictionary.

        Args:
            fragments_dict: Dictionary with headers as keys and fragment data as values
        """
        self._fragments_dict = fragments_dict
        self._contig_to_header_map = {}
        self._build_mapping()

    def _build_mapping(self):
        """Build the contig name to header mapping."""
        known_headers = set(self._fragments_dict)
        for header in self._fragments_dict:
            contig_name = extract_base_contig_name(header, known_headers=known_headers)
            # In case of duplicates, keep the first one (consistent with original behavior)
            if contig_name not in self._contig_to_header_map:
                self._contig_to_header_map[contig_name] = header

    def get_header(self, contig_name: str) -> Union[str, None]:
        """Get the header for a given contig name.

        Args:
            contig_name: The base contig name

        Returns:
            The corresponding header from fragments_dict, or None if not found
        """
        return self._contig_to_header_map.get(contig_name)

    def get_mapping(self) -> Dict[str, str]:
        """Get the complete contig to header mapping.

        Returns:
            Dictionary mapping contig names to headers
        """
        return self._contig_to_header_map.copy()

    def has_contig(self, contig_name: str) -> bool:
        """Check if a contig name exists in the mapping.

        Args:
            contig_name: The base contig name to check

        Returns:
            True if the contig exists in the mapping
        """
        return contig_name in self._contig_to_header_map


def group_contigs_by_cluster(clusters_df):
    """Group contigs by their cluster assignments.

    Replaces the repeated pattern of manually building cluster_contig_counts
    dictionaries throughout the codebase.

    Args:
        clusters_df: DataFrame with 'contig' and 'cluster' columns

    Returns:
        Dictionary mapping cluster IDs to sets of contig names
    """
    cluster_groups = clusters_df.groupby("cluster")["contig"].apply(set).to_dict()
    return cluster_groups


def initialize_duplication_columns(clusters_df):
    """Initialize core gene duplication columns in clusters DataFrame.

    Common pattern used throughout miniprot_utils to set default values
    for duplication analysis columns.

    Args:
        clusters_df: DataFrame with cluster assignments

    Returns:
        DataFrame: Copy with initialized duplication columns
    """
    clusters_df = clusters_df.copy()
    clusters_df["has_duplicated_core_genes"] = False
    clusters_df["duplicated_core_genes_count"] = 0
    clusters_df["total_core_genes_found"] = 0
    clusters_df["single_copy_genes_count"] = 0
    return clusters_df
