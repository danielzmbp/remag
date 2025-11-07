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
from .clustering import _leiden_clustering, _construct_knn_graph, _leiden_clustering_on_graph


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

    # Fix other parameters - only vary resolution
    fixed_k_neighbors = getattr(args, 'leiden_k_neighbors', 15)
    fixed_similarity_threshold = getattr(args, 'leiden_similarity_threshold', 0.1)
    fixed_n_jobs = getattr(args, 'cores', 1)

    # Construct k-NN graph ONCE (reuse for all resolution tests for performance)
    graph = _construct_knn_graph(
        embeddings_df.values,
        k=fixed_k_neighbors,
        similarity_threshold=fixed_similarity_threshold,
        n_jobs=fixed_n_jobs,
        args=None  # Don't save graph during testing
    )

    results = {}

    for resolution in test_resolutions:
        logger.debug(f"Testing resolution={resolution:.2f}...")

        # Apply Leiden clustering on pre-built graph (fast - no graph construction)
        cluster_labels = _leiden_clustering_on_graph(
            graph,
            resolution=resolution,
            random_state=42
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

            # Calculate per-bin completeness metrics (using single-copy genes only)
            bin_completeness = test_clusters_df.groupby('cluster')['single_copy_genes_count'].first()
            total_duplications = int(test_clusters_df.groupby('cluster')['duplicated_core_genes_count'].first().sum())
            bins_with_duplications = int(test_clusters_df.groupby('cluster')['has_duplicated_core_genes'].first().sum())

            # Completeness quality metrics (single-copy genes)
            max_bin_completeness = int(bin_completeness.max()) if len(bin_completeness) > 0 else 0
            median_bin_completeness = int(bin_completeness.median()) if len(bin_completeness) > 0 else 0

            logger.info(f"Resolution {resolution:.2f}: {n_clusters} clusters, "
                       f"max completeness={max_bin_completeness}, median={median_bin_completeness}, "
                       f"{bins_with_duplications} contaminated, {total_duplications} total duplications")

            results[resolution] = {
                'n_clusters': n_clusters,
                'bins_with_duplications': bins_with_duplications,
                'total_duplications': total_duplications,
                'max_bin_completeness': max_bin_completeness,
                'median_bin_completeness': median_bin_completeness,
                'clusters_df': test_clusters_df
            }

        except Exception as e:
            logger.warning(f"Failed to check duplications for resolution {resolution:.2f}: {e}")
            results[resolution] = {
                'n_clusters': n_clusters,
                'bins_with_duplications': float('inf'),
                'total_duplications': float('inf'),
                'max_bin_completeness': 0,
                'median_bin_completeness': 0,
                'clusters_df': test_clusters_df
            }

    # Pick the resolution that maximizes genome completeness (single-copy genes only)
    # Priority: 1) Highest max completeness (recover complete, clean genomes)
    #           2) Highest median completeness (overall bin quality)
    #           3) Fewest duplications (contamination)
    # Rationale: Completeness now counts only single-copy genes (non-duplicated),
    # preventing contaminated bins from being rewarded with high scores. This avoids
    # bias towards over-consolidation (fewer, larger, contaminated bins).
    best_resolution = max(results.keys(), key=lambda r: (
        results[r]['max_bin_completeness'],    # Primary: recover complete genomes
        results[r]['median_bin_completeness'], # Secondary: overall bin quality
        -results[r]['total_duplications']      # Tertiary: contamination (negated for max)
    ))
    best_result = results[best_resolution]

    logger.info(f"Best resolution: {best_resolution:.2f} with {best_result['n_clusters']} clusters, "
               f"max completeness={best_result['max_bin_completeness']}, "
               f"median={best_result['median_bin_completeness']}, "
               f"{best_result['total_duplications']} total duplications")

    return best_resolution, results


def determine_optimal_resolution(embeddings_df, fragments_dict, args, gene_mappings=None):
    """
    Determine optimal Leiden resolution by analyzing core gene duplications.

    This is the main function that orchestrates the adaptive resolution process:
    1. Use existing gene mappings or run miniprot to estimate organism count
    2. Calculate base resolution from organism estimate
    3. Test multiple resolution values (base * [0.7, 1.0, 1.4])
    4. Pick the resolution with fewest core gene duplications

    Args:
        embeddings_df: DataFrame with embeddings for all contigs
        fragments_dict: Dictionary mapping headers to sequences
        args: Arguments object
        gene_mappings: Optional pre-computed gene-to-contig mappings from miniprot.
                      If None, will run miniprot to generate them.

    Returns:
        float: Optimal resolution parameter
    """
    # Step 1: Get gene counts from existing mappings or run miniprot
    if gene_mappings is not None:
        # Extract gene counts from existing mappings (inline to avoid extra function)
        gene_counts = {}
        for contig_name, genes in gene_mappings.items():
            for gene_family in genes.keys():
                gene_counts[gene_family] = gene_counts.get(gene_family, 0) + 1
    else:
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

    logger.debug(f"Core gene statistics: median={median_count:.1f}, 90th percentile={percentile_90:.1f}, max={max_count:.1f}")
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
        max(base_resolution * 0.3, 0.05),  # Very conservative (fewer bins), min 0.05
        max(base_resolution * 0.6, 0.05),  # Conservative (fewer bins), min 0.05
        base_resolution,                    # Base estimate
        base_resolution * 2.0,              # Aggressive (more bins)
        base_resolution * 3.0               # Very aggressive (more bins)
    ]

    # Remove duplicates and sort
    test_resolutions = sorted(set(test_resolutions))

    # Load gene mappings cache for quick duplication checking
    # The cache was created during organism estimation and contains:
    # {contig_name: {gene_family: {score, coverage, identity}}}
    logger.debug("Loading gene mappings cache for duplication checking...")

    # Import needed for cache path function
    from .miniprot_utils import get_gene_mappings_cache_path

    # Use provided gene_mappings if available, otherwise try to load from cache
    gene_mappings_cache = gene_mappings

    if gene_mappings_cache is None:
        # Check if cache already exists from organism estimation
        cache_path = get_gene_mappings_cache_path(args)

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

    return best_resolution
