"""
Quality control module for REMAG
"""

from loguru import logger


def check_core_gene_duplications(clusters_df, fragments_dict, args):
    """
    Check for duplicated core genes using miniprot.
    
    This is a wrapper function that calls the consolidated miniprot utility
    with quality.py-specific thresholds (target_coverage >= 0.50, identity >= 0.50).
    """
    # Import here to avoid circular imports
    from .miniprot_utils import check_core_gene_duplications as _check_core_gene_duplications
    
    return _check_core_gene_duplications(
        clusters_df, 
        fragments_dict, 
        args,
        target_coverage_threshold=0.50,
        identity_threshold=0.50,
        use_header_cache=False  # Use simple approach from quality.py
    )
