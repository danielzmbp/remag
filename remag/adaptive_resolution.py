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

from .miniprot_utils import estimate_organisms_from_all_contigs, check_core_gene_duplications_from_cache, extract_gene_counts_from_mappings
from .clustering import _construct_knn_graph, _leiden_clustering_on_graph


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

    # Use same min_duplications threshold as refinement for consistency
    min_duplications_threshold = getattr(args, 'min_duplications_for_refinement', 1)

    # Construct k-NN graph ONCE (reuse for all resolution tests for performance)
    # Save graph to disk so it can be reused during final clustering (saves ~1 minute)
    graph = _construct_knn_graph(
        embeddings_df.values,
        k=fixed_k_neighbors,
        similarity_threshold=fixed_similarity_threshold,
        n_jobs=fixed_n_jobs,
        args=args  # Save graph for reuse in final clustering
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

            # Calculate per-bin metrics
            bin_scg = test_clusters_df.groupby('cluster')['single_copy_genes_count'].first()
            bin_dups = test_clusters_df.groupby('cluster')['duplicated_core_genes_count'].first()

            # Calculate quality scores for each bin: scg - 5*dups
            bin_quality_scores = bin_scg - (5 * bin_dups)

            # Aggregate metrics
            total_duplications = int(bin_dups.sum())
            bins_with_duplications = int((bin_dups >= min_duplications_threshold).sum())

            # Quality metrics
            max_bin_completeness = int(bin_scg.max()) if len(bin_scg) > 0 else 0
            p90_bin_completeness = int(np.percentile(bin_scg, 90)) if len(bin_scg) > 0 else 0
            p90_quality_score = float(np.percentile(bin_quality_scores, 90)) if len(bin_quality_scores) > 0 else 0

            logger.info(f"Resolution {resolution:.2f}: {n_clusters} clusters, "
                       f"max SCG={max_bin_completeness}, p90 SCG={p90_bin_completeness}, "
                       f"p90 quality (SCG-5*dups)={p90_quality_score:.1f}, "
                       f"{bins_with_duplications} contaminated, {total_duplications} total duplications")

            results[resolution] = {
                'n_clusters': n_clusters,
                'bins_with_duplications': bins_with_duplications,
                'total_duplications': total_duplications,
                'max_bin_completeness': max_bin_completeness,
                'p90_bin_completeness': p90_bin_completeness,
                'p90_quality_score': p90_quality_score,
                'clusters_df': test_clusters_df
            }

        except Exception as e:
            logger.warning(f"Failed to check duplications for resolution {resolution:.2f}: {e}")
            results[resolution] = {
                'n_clusters': n_clusters,
                'bins_with_duplications': float('inf'),
                'total_duplications': float('inf'),
                'max_bin_completeness': 0,
                'p90_bin_completeness': 0,
                'p90_quality_score': float('-inf'),
                'clusters_df': test_clusters_df
            }

    # Pick the resolution using quality-aware selection
    # Filter out biologically impossible solutions where p90 SCG > 200 (indicates merged organisms)
    # Database has 133 BUSCO eukaryotic core gene families, so p90 > 200 suggests multiple organisms merged
    MAX_REALISTIC_P90 = 200
    valid_resolutions = {
        r: data for r, data in results.items()
        if data['p90_bin_completeness'] <= MAX_REALISTIC_P90
    }

    if valid_resolutions:
        # Among valid solutions, maximize p90 SCG completeness, then max SCG completeness
        best_resolution = max(valid_resolutions.keys(), key=lambda r: (
            valid_resolutions[r]['p90_bin_completeness'],
            valid_resolutions[r]['max_bin_completeness']
        ))
        best_result = valid_resolutions[best_resolution]
        logger.info(f"Selected from {len(valid_resolutions)}/{len(results)} valid resolutions (p90 SCG ≤ {MAX_REALISTIC_P90})")
    else:
        # Fallback: if all resolutions exceed threshold, pick one with best completeness
        logger.warning(f"All {len(results)} resolutions exceed p90 SCG={MAX_REALISTIC_P90} threshold - picking best completeness")
        best_resolution = max(results.keys(), key=lambda r: (
            results[r]['p90_bin_completeness'],
            results[r]['max_bin_completeness']
        ))
        best_result = results[best_resolution]

    logger.info(f"Best resolution: {best_resolution:.2f} with {best_result['n_clusters']} clusters, "
               f"max SCG={best_result['max_bin_completeness']}, "
               f"p90 SCG={best_result['p90_bin_completeness']}, "
               f"p90 quality score={best_result['p90_quality_score']:.1f}, "
               f"{best_result['bins_with_duplications']} contaminated bins, "
               f"{best_result['total_duplications']} total duplications")

    return best_resolution, results


def determine_optimal_resolution(embeddings_df, fragments_dict, args, gene_mappings=None):
    """
    Determine optimal Leiden resolution by analyzing core gene duplications.

    This is the main function that orchestrates the adaptive resolution process:
    1. Use existing gene mappings or run miniprot to estimate organism count
    2. Calculate base resolution from organism estimate
    3. Test 16 resolution values (0.01x to 3.0x multipliers with emphasis on lower range)
    4. Pick the resolution that maximizes completeness while minimizing contamination

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
        gene_counts = extract_gene_counts_from_mappings(gene_mappings)
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
    # Testing range: 0.01x to 3.0x around base estimate (e.g., base=0.5 tests 0.005-1.5)
    base_resolution = estimate_resolution_from_organisms(
        estimated_organisms,
        base_resolution=1.0,  # Fixed base for formula
        reference_organisms=100  # Reference point: 100 organisms
    )

    # Step 4: Test multiple resolutions around the base estimate
    # Balanced exploration: 7 below, 1.0 at center, 8 above (16 total)
    test_resolutions = [
        base_resolution * 0.2,   # Conservative clustering
        base_resolution * 0.3,
        base_resolution * 0.4,
        base_resolution * 0.5,
        base_resolution * 0.6,
        base_resolution * 0.7,
        base_resolution * 0.8,
        base_resolution * 1.0,   # Base estimate
        base_resolution * 1.2,   # Aggressive splitting
        base_resolution * 1.5,
        base_resolution * 1.75,
        base_resolution * 2.0,
        base_resolution * 2.5,
        base_resolution * 3.0,
        base_resolution * 3.5,
        base_resolution * 4.0
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
                    'total_duplications': data['total_duplications'],
                    'max_bin_completeness': data['max_bin_completeness'],
                    'p90_bin_completeness': data['p90_bin_completeness'],
                    'p90_quality_score': data['p90_quality_score']
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
