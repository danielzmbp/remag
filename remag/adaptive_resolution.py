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
from .clustering import _construct_knn_graph, _leiden_clustering_on_graph


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
    # Save graph to disk so it can be reused during final clustering (saves ~1 minute)
    graph = _construct_knn_graph(
        embeddings_df.values,
        k=fixed_k_neighbors,
        similarity_threshold=fixed_similarity_threshold,
        n_jobs=fixed_n_jobs,
        args=args  # Save graph for reuse in final clustering
    )

    results = {}
    tested_resolutions = []
    peak_completeness = 0

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

        tested_resolutions.append(resolution)

        # Early stop if completeness drops below 50% of the best seen so far (avoid over-splitting)
        current_max_comp = results[resolution]['max_bin_completeness']
        prev_peak = peak_completeness
        if prev_peak > 0 and current_max_comp < 0.50 * prev_peak:
            logger.info(
                f"Stopping resolution sweep early: resolution {resolution:.2f} max completeness "
                f"{current_max_comp} < 50% of peak {prev_peak}"
            )
            # Drop this over-split result from consideration
            tested_resolutions.pop()
            results.pop(resolution, None)
            break

        peak_completeness = max(prev_peak, current_max_comp)

    # Pick resolution: minimize duplications, then maximize clusters without tanking max completeness
    usable_results = {r: results[r] for r in tested_resolutions}
    min_dup = min(res['total_duplications'] for res in usable_results.values())

    # Candidates with minimal duplications
    dup_candidates = {
        r: res for r, res in usable_results.items() if res['total_duplications'] == min_dup
    }

    mode = getattr(args, "mode", "metagenomics").lower()
    if mode == "single-cell":
        # Prefer fewer clusters (coarser) while still minimizing duplications
        best_resolution = min(
            dup_candidates.keys(),
            key=lambda r: (dup_candidates[r]['n_clusters'], -dup_candidates[r]['max_bin_completeness'])
        )
    else:
        # Default: prefer more clusters (finer) after minimizing duplications
        best_resolution = max(
            dup_candidates.keys(),
            key=lambda r: (dup_candidates[r]['n_clusters'], dup_candidates[r]['max_bin_completeness'])
        )
    best_result = dup_candidates[best_resolution]

    logger.info(
        f"Best resolution: {best_resolution:.2f} with {best_result['n_clusters']} clusters, "
        f"{best_result['total_duplications']} total duplications, "
        f"max completeness={best_result['max_bin_completeness']}"
    )

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

    # Step 3: Choose candidate resolutions
    mode = getattr(args, "mode", "metagenomics").lower()
    coverage_count = (len(args.bam) if getattr(args, "bam", None) else 0) + (len(args.tsv) if getattr(args, "tsv", None) else 0)

    if mode == "single-cell":
        # Single-cell: focus on coarse resolutions, cap at 0.5
        test_resolutions = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    else:
        test_resolutions = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0, 1.2, 1.5]
        if coverage_count > 1:
            # Coassembly: skip very low resolutions and include higher ones
            test_resolutions = [0.60, 0.80, 1.0, 1.2, 1.5, 2.0]

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
        logger.info("Falling back to default resolution: 1.0")
        return 1.0

    # Step 4: Test resolutions and pick the best
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
