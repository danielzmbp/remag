"""
Single Copy Core Gene (SCG) feature extraction and management.

This module handles the extraction of SCG annotations from contigs using miniprot,
creating sparse feature matrices for use in the hybrid contrastive learning loss.
"""

import json
import os
import numpy as np
import pandas as pd
from scipy import sparse
from loguru import logger

from .miniprot_utils import (
    check_miniprot_available,
    get_gene_mappings_cache_path
)


def get_scg_annotations_path(args):
    """Get the path for the SCG annotations file."""
    return os.path.join(args.output, "scg_annotations.json")


def extract_scg_features(fragments_dict, args, gene_mappings=None, use_cached=True):
    """
    Extract SCG features for all contigs using miniprot annotations.

    This function creates a sparse matrix representation of SCG presence/absence
    for use in the hybrid contrastive learning loss. It leverages cached gene
    mappings from the adaptive resolution step if available.

    Args:
        fragments_dict: Dictionary mapping headers to sequences
        args: Arguments object containing output directory, cores, etc.
        gene_mappings: Optional pre-computed gene-to-contig mappings from miniprot.
                      If None, will try to load from cache or run miniprot.
        use_cached: Whether to use cached gene mappings if available (only used if gene_mappings is None)

    Returns:
        tuple: (scg_matrix_df, gene_family_index, contig_to_scg_dict)
            - scg_matrix_df: DataFrame with contigs as index, gene families as columns
            - gene_family_index: List of gene family codes
            - contig_to_scg_dict: Dict mapping contig names to their SCG families
    """
    logger.info("Extracting SCG features for contrastive learning...")

    # Check if miniprot is available
    if not check_miniprot_available():
        logger.warning("miniprot not found - SCG features will be empty")
        return _create_empty_scg_features(fragments_dict)

    # Use provided gene_mappings if available
    if gene_mappings is not None:
        logger.debug("Using pre-computed gene mappings for SCG features")
    else:
        # Try to load cached gene mappings from adaptive resolution step
        cache_path = get_gene_mappings_cache_path(args)

        if use_cached and os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    gene_mappings = json.load(f)
                logger.info(f"Loaded cached gene mappings for {len(gene_mappings)} contigs")
            except Exception as e:
                logger.warning(f"Failed to load cached gene mappings: {e}")

        # If no cache exists, run miniprot now to generate gene mappings
        if gene_mappings is None:
            logger.info("No cached gene mappings found - running miniprot to extract SCG annotations...")
            from .miniprot_utils import estimate_organisms_from_all_contigs

            # This will run miniprot and create the cache
            gene_counts = estimate_organisms_from_all_contigs(fragments_dict, args)

            # Try loading the cache again after miniprot run
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "r") as f:
                        gene_mappings = json.load(f)
                    logger.info(f"Successfully generated and loaded gene mappings for {len(gene_mappings)} contigs")
                except Exception as e:
                    logger.warning(f"Failed to load gene mappings after miniprot run: {e}")
                    return _create_empty_scg_features(fragments_dict)
            else:
                logger.warning("Miniprot completed but no cache file was created - SCG loss will be disabled")
                return _create_empty_scg_features(fragments_dict)

    # Build SCG feature matrix
    return _build_scg_matrix(gene_mappings, fragments_dict, args)


def _create_empty_scg_features(fragments_dict):
    """Create empty SCG features when miniprot is unavailable or no cache exists."""
    contig_names = list(fragments_dict.keys())
    empty_df = pd.DataFrame(index=contig_names)
    empty_dict = {contig: set() for contig in contig_names}
    return empty_df, [], empty_dict


def _build_scg_matrix(gene_mappings, fragments_dict, args):
    """
    Build sparse SCG matrix from gene mappings.

    Args:
        gene_mappings: Dict of {contig_name: {gene_family: {score, coverage, identity}}}
        fragments_dict: Dictionary mapping headers to sequences
        args: Arguments object

    Returns:
        tuple: (scg_matrix_df, gene_family_index, contig_to_scg_dict)
    """
    logger.debug("Building SCG feature matrix...")

    # Extract all unique gene families
    all_gene_families = set()
    for contig_genes in gene_mappings.values():
        all_gene_families.update(contig_genes.keys())

    gene_family_index = sorted(list(all_gene_families))
    logger.debug(f"Found {len(gene_family_index)} unique gene families")

    # Create mapping from gene family to column index
    gene_to_col = {gene: idx for idx, gene in enumerate(gene_family_index)}

    # Get all contig names (including fragments)
    all_contig_names = list(fragments_dict.keys())
    contig_to_row = {contig: idx for idx, contig in enumerate(all_contig_names)}

    # Build sparse matrix representation
    rows = []
    cols = []
    data = []
    contig_to_scg_dict = {}

    for contig_name in all_contig_names:
        # Extract base contig name for fragments
        base_contig = _get_base_contig_name(contig_name)

        # Look up SCG annotations for base contig
        if base_contig in gene_mappings:
            contig_genes = gene_mappings[base_contig]
            contig_to_scg_dict[contig_name] = set(contig_genes.keys())

            # Add entries for each gene family present in this contig
            row_idx = contig_to_row[contig_name]
            for gene_family, gene_info in contig_genes.items():
                col_idx = gene_to_col[gene_family]
                # Use score (coverage * identity) as feature value
                score = gene_info["score"]

                rows.append(row_idx)
                cols.append(col_idx)
                data.append(score)
        else:
            contig_to_scg_dict[contig_name] = set()

    # Create sparse CSR matrix for efficient operations
    n_contigs = len(all_contig_names)
    n_genes = len(gene_family_index)

    if len(rows) > 0:
        scg_matrix = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n_contigs, n_genes),
            dtype=np.float32
        )

        # Convert to DataFrame for easier handling
        scg_matrix_dense = scg_matrix.toarray()
        scg_matrix_df = pd.DataFrame(
            scg_matrix_dense,
            index=all_contig_names,
            columns=gene_family_index
        )
    else:
        # No SCG annotations found
        logger.warning("No SCG annotations found for any contigs")
        scg_matrix_df = pd.DataFrame(
            np.zeros((n_contigs, n_genes), dtype=np.float32),
            index=all_contig_names,
            columns=gene_family_index
        )

    # Log statistics
    n_contigs_with_scg = sum(1 for scgs in contig_to_scg_dict.values() if len(scgs) > 0)
    logger.debug(f"SCG matrix: {n_contigs} contigs × {n_genes} gene families")
    logger.info(f"Contigs with SCG annotations: {n_contigs_with_scg}/{n_contigs} ({100*n_contigs_with_scg/n_contigs:.1f}%)")

    # Save annotations for reference
    _save_scg_annotations(contig_to_scg_dict, gene_family_index, args)

    return scg_matrix_df, gene_family_index, contig_to_scg_dict


def _get_base_contig_name(contig_name):
    """
    Extract base contig name from fragment identifier.

    Fragment naming patterns:
    - Original: contig_name.original
    - Fragments: contig_name.h1.0, contig_name.h2.1, etc.

    Args:
        contig_name: Full contig/fragment identifier

    Returns:
        str: Base contig name without fragment suffixes
    """
    if contig_name.endswith(".original"):
        return contig_name.replace(".original", "")

    # Handle fragment patterns (.h1.M, .h2.M, etc.)
    parts = contig_name.split(".")
    if len(parts) >= 3 and parts[-2].startswith("h"):
        # Remove fragment suffix (.hN.M)
        return ".".join(parts[:-2])

    return contig_name


def _save_scg_annotations(contig_to_scg_dict, gene_family_index, args):
    """Save SCG annotations to JSON file for reference."""
    annotations_path = get_scg_annotations_path(args)

    try:
        # Convert sets to lists for JSON serialization
        annotations_data = {
            "gene_families": gene_family_index,
            "contig_annotations": {
                contig: sorted(list(scgs))
                for contig, scgs in contig_to_scg_dict.items()
                if len(scgs) > 0  # Only save contigs with annotations
            },
            "summary": {
                "total_contigs": len(contig_to_scg_dict),
                "contigs_with_scg": sum(1 for scgs in contig_to_scg_dict.values() if len(scgs) > 0),
                "total_gene_families": len(gene_family_index)
            }
        }

        with open(annotations_path, "w") as f:
            json.dump(annotations_data, f, indent=2)

        logger.info(f"Saved SCG annotations to {annotations_path}")
    except Exception as e:
        logger.warning(f"Failed to save SCG annotations: {e}")


def load_scg_features(args):
    """
    Load pre-computed SCG features from file.

    Args:
        args: Arguments object containing output directory

    Returns:
        tuple: (scg_matrix_df, gene_family_index, contig_to_scg_dict) or None if not found
    """
    annotations_path = get_scg_annotations_path(args)

    if not os.path.exists(annotations_path):
        return None

    try:
        with open(annotations_path, "r") as f:
            annotations_data = json.load(f)

        gene_family_index = annotations_data["gene_families"]
        contig_annotations = annotations_data["contig_annotations"]

        # Reconstruct contig_to_scg_dict
        contig_to_scg_dict = {
            contig: set(scgs) for contig, scgs in contig_annotations.items()
        }

        logger.info(f"Loaded SCG annotations: {annotations_data['summary']}")

        # Note: We don't reconstruct the full matrix here, as it's more efficient
        # to load it on-demand during training
        return None, gene_family_index, contig_to_scg_dict

    except Exception as e:
        logger.warning(f"Failed to load SCG annotations: {e}")
        return None


class SCGFeatureManager:
    """
    Manager for SCG features during training.

    This class handles efficient lookup of SCG features for contigs during
    training, supporting batch operations and sparse representations.
    """

    def __init__(self, scg_matrix_df, contig_to_scg_dict):
        """
        Initialize SCG feature manager.

        Args:
            scg_matrix_df: DataFrame with SCG features (contigs × gene families)
            contig_to_scg_dict: Dict mapping contig names to their SCG sets
        """
        self.scg_matrix_df = scg_matrix_df
        self.contig_to_scg_dict = contig_to_scg_dict
        self.has_scg_features = (scg_matrix_df is not None and
                                 not scg_matrix_df.empty and
                                 len(contig_to_scg_dict) > 0)

        if self.has_scg_features:
            self.n_gene_families = scg_matrix_df.shape[1]
            logger.debug(f"Initialized SCG manager with {self.n_gene_families} gene families")
        else:
            self.n_gene_families = 0
            logger.debug("Initialized SCG manager with no features (will use dummy tensors)")

    def get_scg_features(self, contig_names):
        """
        Get SCG features for a list of contigs.

        Args:
            contig_names: List of contig names

        Returns:
            np.ndarray: Binary matrix (n_contigs × n_gene_families)
        """
        if not self.has_scg_features:
            # Return zero matrix if no SCG features available
            return np.zeros((len(contig_names), max(1, self.n_gene_families)), dtype=np.float32)

        # Look up features from DataFrame
        features = []
        for contig in contig_names:
            if contig in self.scg_matrix_df.index:
                features.append(self.scg_matrix_df.loc[contig].values)
            else:
                # Contig not in matrix - use zero vector
                features.append(np.zeros(self.n_gene_families, dtype=np.float32))

        return np.array(features, dtype=np.float32)

    def get_scg_overlap(self, contigs1, contigs2):
        """
        Compute SCG overlap between two sets of contigs.

        Args:
            contigs1: List of contig names (batch 1)
            contigs2: List of contig names (batch 2)

        Returns:
            np.ndarray: Overlap matrix (len(contigs1) × len(contigs2))
                Each entry is the number of shared SCG families
        """
        if not self.has_scg_features:
            # Return zero matrix if no SCG features
            return np.zeros((len(contigs1), len(contigs2)), dtype=np.float32)

        overlap_matrix = np.zeros((len(contigs1), len(contigs2)), dtype=np.float32)

        for i, c1 in enumerate(contigs1):
            scg1 = self.contig_to_scg_dict.get(c1, set())
            for j, c2 in enumerate(contigs2):
                scg2 = self.contig_to_scg_dict.get(c2, set())
                # Count shared SCG families
                overlap_matrix[i, j] = len(scg1.intersection(scg2))

        return overlap_matrix
