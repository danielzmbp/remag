"""
Adaptive resolution module for REMAG

This module provides functionality to automatically determine the optimal Leiden
resolution parameter based on core gene duplication analysis.
"""

import os
import json
import numpy as np
import pandas as pd
from loguru import logger

from .miniprot_utils import estimate_organisms_from_all_contigs, check_core_gene_duplications_from_cache
from .clustering import _leiden_clustering


def estimate_resolution_from_organisms(estimated_organisms, base_resolution=0.1, reference_organisms=100):
    """
    Estimate Leiden resolution parameter based on estimated organism count.

    Uses square root scaling to map organism counts to resolution values,
    with clamping to prevent extreme values.

    Args:
        estimated_organisms: Estimated number of organisms from core gene analysis
        base_resolution: Default resolution for reference_organisms (default: 0.1)
        reference_organisms: Number of organisms for which base_resolution is optimal (default: 100)

    Returns:
        float: Estimated resolution parameter
    """
    if estimated_organisms <= 0:
        logger.warning("Invalid organism estimate, using base resolution")
        return base_resolution

    # Square root scaling (empirically reasonable for graph clustering)
    resolution = base_resolution * (estimated_organisms / reference_organisms) ** 0.5

    # Clamp to reasonable bounds (minimum 0.05 ensures proper separation even for low-diversity samples)
    resolution = np.clip(resolution, 0.05, 5.0)

    logger.info(f"Estimated {estimated_organisms:.1f} organisms → resolution={resolution:.2f}")

    return resolution


def test_multiple_resolutions(embeddings_df, gene_mappings_cache, args, test_resolutions):
    """
    Test multiple resolution values and pick the best based on core gene duplications.

    Args:
        embeddings_df: DataFrame with embeddings for all contigs
        gene_mappings_cache: Cached gene-to-contig mappings from miniprot
        args: Arguments object
        test_resolutions: List of resolution values to test

    Returns:
        tuple: (best_resolution, results_dict)
    """
    logger.info(f"Testing {len(test_resolutions)} resolution values: {[f'{r:.2f}' for r in test_resolutions]}")

    results = {}

    for resolution in test_resolutions:
        logger.info(f"Testing resolution={resolution:.2f}...")

        # Perform clustering with this resolution
        cluster_labels = _leiden_clustering(
            embeddings_df.values,
            k=getattr(args, 'leiden_k_neighbors', 15),
            similarity_threshold=getattr(args, 'leiden_similarity_threshold', 0.1),
            resolution=resolution,
            random_state=42,
            n_jobs=getattr(args, 'cores', 1),
            args=None  # Don't save intermediate graphs during testing
        )

        # Convert cluster labels to DataFrame format for duplication checking
        contig_names = list(embeddings_df.index)
        formatted_labels = [
            f"bin_{label}" if label != -1 else "noise" for label in cluster_labels
        ]

        test_clusters_df = pd.DataFrame({
            'contig': contig_names,
            'cluster': formatted_labels
        })

        # Count clusters
        n_clusters = len([c for c in test_clusters_df['cluster'].unique() if c != 'noise'])

        # Check duplications using cached mappings
        try:
            test_clusters_df = check_core_gene_duplications_from_cache(
                test_clusters_df, gene_mappings_cache, args
            )

            # Calculate total duplications across all bins (sum unique values per bin)
            total_duplications = int(test_clusters_df.groupby('cluster')['duplicated_core_genes_count'].first().sum())
            bins_with_duplications = int(test_clusters_df.groupby('cluster')['has_duplicated_core_genes'].first().sum())

            logger.info(f"Resolution {resolution:.2f}: {n_clusters} clusters, "
                       f"{bins_with_duplications} bins with duplications, "
                       f"{total_duplications} total duplicated genes")

            results[resolution] = {
                'n_clusters': n_clusters,
                'bins_with_duplications': bins_with_duplications,
                'total_duplications': total_duplications,
                'clusters_df': test_clusters_df
            }

        except Exception as e:
            logger.warning(f"Failed to check duplications for resolution {resolution:.2f}: {e}")
            results[resolution] = {
                'n_clusters': n_clusters,
                'bins_with_duplications': float('inf'),
                'total_duplications': float('inf'),
                'clusters_df': test_clusters_df
            }

    # Pick the resolution with the fewest total duplications
    best_resolution = min(results.keys(), key=lambda r: results[r]['total_duplications'])
    best_result = results[best_resolution]

    logger.info(f"Best resolution: {best_resolution:.2f} with {best_result['n_clusters']} clusters, "
               f"{best_result['total_duplications']} total duplications")

    return best_resolution, results


def determine_optimal_resolution(embeddings_df, fragments_dict, args):
    """
    Determine optimal Leiden resolution by analyzing core gene duplications.

    This is the main function that orchestrates the adaptive resolution process:
    1. Run miniprot on all contigs to estimate organism count
    2. Calculate base resolution from organism estimate
    3. Test multiple resolution values (base * [0.7, 1.0, 1.4])
    4. Pick the resolution with fewest core gene duplications

    Args:
        embeddings_df: DataFrame with embeddings for all contigs
        fragments_dict: Dictionary mapping headers to sequences
        args: Arguments object

    Returns:
        float: Optimal resolution parameter
    """
    logger.info("=== ADAPTIVE RESOLUTION DETERMINATION ===")

    # Step 1: Estimate organism count from all contigs
    gene_counts = estimate_organisms_from_all_contigs(fragments_dict, args)

    if not gene_counts:
        logger.warning("No core genes found, falling back to default resolution")
        return getattr(args, 'leiden_resolution', 1.0)

    # Save gene counts if keeping intermediate files
    if getattr(args, "keep_intermediate", False):
        gene_counts_path = os.path.join(args.output, "organism_estimation_gene_counts.json")
        try:
            with open(gene_counts_path, "w") as f:
                json.dump(gene_counts, f, indent=2)
            logger.info(f"Saved gene counts for organism estimation to {gene_counts_path}")
        except Exception as e:
            logger.warning(f"Failed to save gene counts: {e}")

    # Step 2: Estimate organism count using max gene occurrence
    # Since these are single-copy genes, the max count indicates the minimum number of organisms
    counts_list = list(gene_counts.values())
    median_count = np.median(counts_list)
    percentile_90 = np.percentile(counts_list, 90)
    max_count = np.max(counts_list)

    # Use maximum for estimation (most conservative, ensures we don't underestimate diversity)
    estimated_organisms = max_count

    logger.info(f"Core gene statistics: median={median_count:.1f}, 90th percentile={percentile_90:.1f}, max={max_count:.1f}")
    logger.info(f"Estimated number of organisms: {estimated_organisms:.1f} (using max gene count)")

    # Step 3: Calculate base resolution
    # Note: Always use 1.0/100 as the base/reference for the formula, NOT args.leiden_resolution
    # args.leiden_resolution is only used as a fallback if auto-resolution fails
    # This scaling gives: 1 organism → 0.05 (min clamp), 25 organisms → 0.5, 100 organisms → 1.0, 1000 organisms → 3.16
    base_resolution = estimate_resolution_from_organisms(
        estimated_organisms,
        base_resolution=1.0,  # Fixed base for formula
        reference_organisms=100  # Reference point: 100 organisms
    )

    # Step 4: Test multiple resolutions around the base estimate
    test_resolutions = [
        max(base_resolution * 0.7, 0.05),  # Conservative (fewer bins), min 0.05
        base_resolution,                    # Base estimate
        base_resolution * 1.4               # Aggressive (more bins)
    ]

    # Remove duplicates and sort
    test_resolutions = sorted(set(test_resolutions))

    # Load gene mappings cache for quick duplication checking
    # The cache was created during organism estimation and contains:
    # {contig_name: {gene_family: {score, coverage, identity}}}
    logger.info("Loading gene mappings cache for duplication checking...")

    # Import needed for cache path function
    from .miniprot_utils import get_gene_mappings_cache_path

    # Check if cache already exists from organism estimation
    cache_path = get_gene_mappings_cache_path(args)
    gene_mappings_cache = None

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                gene_mappings_cache = json.load(f)
            logger.info(f"Loaded existing gene mappings cache with {len(gene_mappings_cache)} contigs")
        except Exception as e:
            logger.warning(f"Failed to load gene mappings cache: {e}")

    if gene_mappings_cache is None:
        logger.warning("No gene mappings cache available - cannot test multiple resolutions")
        logger.info(f"Using base resolution estimate: {base_resolution:.2f}")
        return base_resolution

    # Step 5: Test resolutions and pick the best
    best_resolution, results = test_multiple_resolutions(
        embeddings_df, gene_mappings_cache, args, test_resolutions
    )

    # Save resolution testing results if keeping intermediate files
    if getattr(args, "keep_intermediate", False):
        resolution_results_path = os.path.join(args.output, "resolution_testing_results.json")
        try:
            # Convert results to a serializable format (exclude clusters_df)
            serializable_results = {}
            for res, data in results.items():
                serializable_results[f"{res:.4f}"] = {
                    'n_clusters': data['n_clusters'],
                    'bins_with_duplications': data['bins_with_duplications'],
                    'total_duplications': data['total_duplications']
                }
            serializable_results['selected_resolution'] = f"{best_resolution:.4f}"
            serializable_results['estimated_organisms'] = float(estimated_organisms)
            serializable_results['median_gene_count'] = float(median_count)
            serializable_results['percentile_90_gene_count'] = float(percentile_90)
            serializable_results['max_gene_count'] = float(max_count)

            with open(resolution_results_path, "w") as f:
                json.dump(serializable_results, f, indent=2)
            logger.info(f"Saved resolution testing results to {resolution_results_path}")
        except Exception as e:
            logger.warning(f"Failed to save resolution testing results: {e}")

    logger.info(f"=== ADAPTIVE RESOLUTION COMPLETE: {best_resolution:.2f} ===")

    return best_resolution
