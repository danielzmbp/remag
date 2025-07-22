"""
Utility functions for REMAG
"""

import gzip
import os
import sys
from loguru import logger
from typing import Dict, List, Union
import torch


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
        for header in self._fragments_dict.keys():
            contig_name = extract_base_contig_name(header)
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


def build_contig_to_header_map(fragments_dict: FragmentDict) -> Dict[str, str]:
    """Build a mapping from contig names to their headers in fragments_dict.
    
    This is a convenience function for simple use cases. For repeated lookups,
    use ContigHeaderMapper class instead.
    
    Args:
        fragments_dict: Dictionary with headers as keys and fragment data as values
        
    Returns:
        Dictionary mapping base contig names to their headers
    """
    mapper = ContigHeaderMapper(fragments_dict)
    return mapper.get_mapping()


def group_contigs_by_cluster(clusters_df):
    """Group contigs by their cluster assignments.
    
    Replaces the repeated pattern of manually building cluster_contig_counts
    dictionaries throughout the codebase.
    
    Args:
        clusters_df: DataFrame with 'contig' and 'cluster' columns
        
    Returns:
        Dictionary mapping cluster IDs to sets of contig names
    """
    cluster_groups = clusters_df.groupby('cluster')['contig'].apply(set).to_dict()
    return cluster_groups


def count_contigs_per_cluster(clusters_df):
    """Count the number of contigs in each cluster.
    
    Args:
        clusters_df: DataFrame with 'contig' and 'cluster' columns
        
    Returns:
        Dictionary mapping cluster IDs to contig counts
    """
    return clusters_df['cluster'].value_counts().to_dict()


def filter_clusters_by_size(cluster_contig_dict, fragments_dict, min_size, exclude_noise=True):
    """Filter clusters based on total sequence size.
    
    Common pattern used in multiple modules to filter out small bins.
    
    Args:
        cluster_contig_dict: Dictionary mapping cluster IDs to sets of contig headers
        fragments_dict: Dictionary containing fragment sequences
        min_size: Minimum total sequence size for a cluster
        exclude_noise: Whether to exclude 'noise' cluster
        
    Returns:
        Dictionary with filtered clusters meeting size criteria
    """
    filtered_clusters = {}
    
    for cluster_id, contig_headers in cluster_contig_dict.items():
        if exclude_noise and cluster_id == "noise":
            continue
            
        # Calculate total size
        total_size = sum(
            len(fragments_dict[h]["sequence"]) 
            for h in contig_headers 
            if h in fragments_dict
        )
        
        if total_size >= min_size:
            filtered_clusters[cluster_id] = contig_headers
            
    return filtered_clusters
