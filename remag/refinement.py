"""
Refinement module for REMAG
"""

import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger

from .miniprot_utils import check_core_gene_duplications, check_core_gene_duplications_from_cache, get_core_gene_duplication_results_path, get_gene_mappings_cache_path
from .clustering import _leiden_clustering


def refine_bin_with_leiden_clustering(
    bin_contigs, embeddings_df, fragments_dict, args, bin_id, duplication_results
):
    """
    Refine a single contaminated bin using existing embeddings with k-NN graph and Leiden clustering.
    
    Args:
        bin_contigs: List of contig names in this bin
        embeddings_df: DataFrame with embeddings for all contigs
        fragments_dict: Dictionary containing fragment sequences
        args: Command line arguments
        bin_id: Original bin ID being refined
        duplication_results: Results from core gene duplication analysis
        
    Returns:
        DataFrame with cluster assignments or None if refinement failed
    """
    logger.info(f"Refining bin {bin_id} using Leiden clustering on existing embeddings...")
    
    # Extract embeddings for contigs in this bin
    # Note: embeddings are saved without the .original suffix
    bin_embedding_names = bin_contigs
    
    # Filter to contigs that have embeddings
    available_embeddings = [name for name in bin_embedding_names if name in embeddings_df.index]
    
    if len(available_embeddings) < 2:
        logger.warning(f"Bin {bin_id} has insufficient contigs with embeddings ({len(available_embeddings)})")
        return None
        
    bin_embeddings = embeddings_df.loc[available_embeddings]
    logger.info(f"Using embeddings for {len(bin_embeddings)} contigs in bin {bin_id}")
    
    # Use conservative Leiden parameters for refinement only (initial clustering unaffected)
    def get_conservative_leiden_params(args):
        """Get conservative Leiden parameters for refinement based on original clustering settings."""
        base_resolution = getattr(args, 'leiden_resolution', 1.0)
        base_k_neighbors = getattr(args, 'leiden_k_neighbors', 15)
        base_threshold = getattr(args, 'leiden_similarity_threshold', 0.1)
        
        # Conservative adjustments for refinement
        conservative_resolution = base_resolution * 0.5  # Lower resolution = fewer clusters
        conservative_k_neighbors = min(50, base_k_neighbors + 10)  # More neighbors = more connectivity
        conservative_threshold = max(0.2, base_threshold * 2)  # Higher threshold = stronger connections
        
        return conservative_resolution, conservative_k_neighbors, conservative_threshold
    
    leiden_resolution, leiden_k_neighbors, leiden_similarity_threshold = get_conservative_leiden_params(args)
    
    # Log duplication info for reference
    if bin_id in duplication_results:
        duplicated_genes_count = len(duplication_results[bin_id]["duplicated_genes"])
        total_genes_found = duplication_results[bin_id]["total_genes_found"]
        logger.info(
            f"Bin {bin_id} has {duplicated_genes_count} duplicated core genes out of {total_genes_found} total genes"
        )
    
    # leiden_k_neighbors and leiden_similarity_threshold already set above with conservative values
    
    logger.info(f"Leiden parameters for bin {bin_id}: resolution={leiden_resolution:.2f}, k={leiden_k_neighbors}")
    
    # Apply Leiden clustering
    cluster_labels = _leiden_clustering(
        bin_embeddings.values,  # Use the normalized embedding values
        k=leiden_k_neighbors,
        similarity_threshold=leiden_similarity_threshold,
        resolution=leiden_resolution,
        random_state=42,
        n_jobs=getattr(args, 'cores', 1)
    )
    
    # Check clustering results
    n_clusters = len(set(cluster_labels))
    
    logger.info(f"Bin {bin_id} Leiden clustering: {n_clusters} clusters")
    
    # Merge clusters that are too small to avoid over-fragmentation
    min_cluster_size = 5  # Hardcoded for refinement
    cluster_sizes = pd.Series(cluster_labels).value_counts()
    small_clusters = cluster_sizes[cluster_sizes < min_cluster_size].index
    
    if len(small_clusters) > 0:
        largest_cluster = cluster_sizes.idxmax()
        # Merge small clusters into the largest one
        cluster_labels = np.array([
            largest_cluster if c in small_clusters else c 
            for c in cluster_labels
        ])
        # Recalculate number of clusters after merging
        n_clusters = len(set(cluster_labels))
        logger.info(f"Merged {len(small_clusters)} small clusters into largest cluster, now {n_clusters} clusters")
    
    if n_clusters < 2:
        logger.warning(f"Bin {bin_id} refinement produced insufficient clusters ({n_clusters})")
        return None
    
    # Create cluster assignments DataFrame
    # Embeddings already use base contig names without .original suffix
    contig_names = available_embeddings
    
    # Format cluster labels with bin prefix
    formatted_labels = [
        f"{bin_id}_refined_bin_{label}" for label in cluster_labels
    ]
    
    refined_clusters_df = pd.DataFrame({
        'contig': contig_names,
        'cluster': formatted_labels,
        'original_bin': bin_id
    })
    
    # Validate refinement produces meaningful splits without over-fragmentation
    def validate_refinement_quality(original_contigs, refined_clusters_df, bin_id):
        """Validate that refinement produces meaningful splits without over-fragmentation."""
        cluster_sizes = refined_clusters_df.groupby('cluster').size()
        largest_cluster_size = cluster_sizes.max()
        original_size = len(original_contigs)
        
        # Check 1: Avoid extreme fragmentation
        if largest_cluster_size < original_size * 0.3:
            logger.warning(f"Bin {bin_id}: extreme fragmentation detected (largest={largest_cluster_size}/{original_size})")
            return False
        
        # Check 2: Ensure meaningful split (not just removing a few contigs)
        second_largest = cluster_sizes.nlargest(2).iloc[-1] if len(cluster_sizes) > 1 else 0
        if second_largest < max(5, original_size * 0.1):
            logger.info(f"Bin {bin_id}: refinement produced trivial split, likely over-sensitive")
            return False
            
        # Check 3: Don't create too many small fragments
        small_clusters = (cluster_sizes < 5).sum()
        if small_clusters > 1:
            logger.warning(f"Bin {bin_id}: refinement created {small_clusters} small clusters")
            return False
        
        return True
    
    if not validate_refinement_quality(contig_names, refined_clusters_df, bin_id):
        return None
        
    largest_subbin_size = refined_clusters_df.groupby('cluster').size().max()
    retention_ratio = largest_subbin_size / len(contig_names)
    logger.info(f"Bin {bin_id} successfully refined into {n_clusters} sub-bins (largest retains {retention_ratio:.1%})")
    
    return refined_clusters_df


def refine_contaminated_bins_with_embeddings(
    clusters_df, embeddings_df, fragments_dict, args, refinement_round=1, max_refinement_rounds=2
):
    """
    Refine bins that have duplicated core genes using existing embeddings with k-NN graph
    construction and Leiden clustering. This approach:

    1. Identifies bins with duplicated core genes
    2. For each contaminated bin, extracts embeddings of its contigs
    3. Constructs a k-NN graph and applies Leiden clustering
    4. Checks for duplications in refined sub-bins
    5. Iteratively refines still-contaminated sub-bins

    This approach is much more efficient than retraining the entire pipeline as it
    reuses existing embeddings and applies the same clustering method used in the
    main pipeline.

    Args:
        clusters_df: DataFrame with cluster assignments and duplication flags
        embeddings_df: DataFrame with embeddings for all contigs  
        fragments_dict: Dictionary containing fragment sequences
        args: Command line arguments
        refinement_round: Current refinement round (1-indexed)
        max_refinement_rounds: Maximum number of refinement rounds to perform

    Returns:
        tuple: (refined_clusters_df, refined_fragments_dict, refinement_summary)
    """
    # Identify contaminated bins - only refine if multiple genes are duplicated (reduce false positives)
    min_duplications = getattr(args, 'min_duplications_for_refinement', 2)
    
    # Load duplication results to check counts
    duplication_results = {}
    results_path = get_core_gene_duplication_results_path(args)
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                duplication_results = json.load(f)
            logger.info(f"Loaded duplication results for {len(duplication_results)} bins")
        except Exception as e:
            logger.warning(f"Failed to load duplication results: {e}")
    else:
        logger.warning("No duplication results file found; refinement will only run for bins with verified duplication counts")
    
    # Filter for bins with multiple duplications
    contaminated_bins = []
    if "has_duplicated_core_genes" in clusters_df.columns:
        contaminated_clusters = clusters_df[
            clusters_df["has_duplicated_core_genes"] == True
        ]["cluster"].unique()
        
        # Additional filter for multiple duplications
        for bin_id in contaminated_clusters:
            if bin_id in duplication_results:
                duplicated_count = len(duplication_results[bin_id].get("duplicated_genes", {}))
                if duplicated_count >= min_duplications:
                    contaminated_bins.append(bin_id)
                    logger.info(f"REFINEMENT: {bin_id} selected - {duplicated_count} duplicated genes (>= {min_duplications})")
                else:
                    logger.debug(f"REFINEMENT: {bin_id} skipped - only {duplicated_count} duplicated genes (< {min_duplications})")
            else:
                # If no duplication data for this bin, skip refinement
                logger.warning(f"REFINEMENT: {bin_id} skipped - no duplication data available")

    if not contaminated_bins:
        logger.info("No contaminated bins found, skipping refinement")
        return clusters_df, fragments_dict, {}

    if refinement_round > max_refinement_rounds:
        logger.info(
            f"Maximum refinement rounds ({max_refinement_rounds}) reached, marking remaining contaminated bins without further refinement"
        )
        return clusters_df, fragments_dict, {}

    logger.info(
        f"REFINEMENT: Starting round {refinement_round} - evaluating {len(contaminated_bins)} contaminated bins (min {min_duplications} duplicated genes)"
    )
    logger.info("Using existing embeddings with k-NN graph construction and Leiden clustering")
    
    # duplication_results already loaded above for filtering
    
    all_refined_clusters = []
    refinement_summary = {}
    
    # Process each contaminated bin
    for bin_id in tqdm(contaminated_bins, desc="Refining contaminated bins with embeddings"):
        try:
            # Get contigs belonging to this bin
            bin_contigs_df = clusters_df[clusters_df["cluster"] == bin_id]
            
            if bin_contigs_df.empty:
                logger.warning(f"No contigs found for bin {bin_id}")
                refinement_summary[bin_id] = {
                    "status": "failed",
                    "reason": "no_contigs",
                    "sub_bins": 0,
                }
                continue
                
            bin_contigs = bin_contigs_df["contig"].tolist()
            
            # Refine this bin using Leiden clustering
            refined_clusters_df = refine_bin_with_leiden_clustering(
                bin_contigs, embeddings_df, fragments_dict, args, bin_id, duplication_results
            )
            
            if refined_clusters_df is None:
                refinement_summary[bin_id] = {
                    "status": "failed", 
                    "reason": "clustering_failed",
                    "sub_bins": 0,
                }
                continue
                
            # Check for duplicated core genes in refined bins using cached mappings
            logger.debug(f"Checking core gene duplications in {bin_id} refined sub-bins...")
            
            # Try to use cached gene mappings first (much faster)
            gene_mappings_cache = getattr(args, '_gene_mappings_cache', None)
            
            if gene_mappings_cache is None:
                # Fallback: try to load from file if keeping intermediate files
                cache_path = get_gene_mappings_cache_path(args)
                if os.path.exists(cache_path):
                    try:
                        with open(cache_path, "r") as f:
                            gene_mappings_cache = json.load(f)
                        logger.debug(f"Loaded gene mappings cache from {cache_path}")
                    except Exception as e:
                        logger.warning(f"Failed to load gene mappings cache: {e}")
                        gene_mappings_cache = None
            
            # Try cached approach first, with comprehensive error recovery
            duplication_check_failed = False
            if gene_mappings_cache is not None:
                try:
                    # Use fast cached approach
                    refined_clusters_df = check_core_gene_duplications_from_cache(
                        refined_clusters_df, gene_mappings_cache, args
                    )
                    logger.debug(f"Successfully used cached duplication check for {bin_id}")
                except Exception as e:
                    logger.warning(f"Cache-based duplication check failed for {bin_id}: {e}")
                    gene_mappings_cache = None  # Force fallback to miniprot
            
            if gene_mappings_cache is None:
                try:
                    # Fallback to miniprot (slower but still works)
                    logger.warning(f"Gene mappings cache not available, falling back to miniprot for {bin_id}")
                    refined_clusters_df = check_core_gene_duplications(
                        refined_clusters_df,
                        fragments_dict,
                        args,
                        target_coverage_threshold=0.55,
                        identity_threshold=0.35,
                        use_header_cache=True
                    )
                    logger.debug(f"Successfully used miniprot duplication check for {bin_id}")
                except Exception as e:
                    logger.error(f"Both cache and miniprot duplication checks failed for {bin_id}: {e}")
                    logger.warning(f"Proceeding with {bin_id} refinement without duplication validation")
                    # Mark all refined bins as potentially having duplications for safety
                    refined_clusters_df['has_duplicated_core_genes'] = True
                    duplication_check_failed = True
            
            # Count successful sub-bins
            sub_bins = refined_clusters_df["cluster"].nunique()
            
            if sub_bins > 1:
                all_refined_clusters.append(refined_clusters_df)
                refinement_summary[bin_id] = {
                    "status": "success",
                    "sub_bins": sub_bins,
                }
                logger.info(f"Successfully refined {bin_id} into {sub_bins} sub-bins")
            else:
                refinement_summary[bin_id] = {
                    "status": "insufficient_split",
                    "sub_bins": sub_bins,
                }
                logger.warning(f"Refinement of {bin_id} produced only {sub_bins} sub-bins")
                
        except Exception as e:
            logger.error(f"Error during refinement of {bin_id}: {e}")
            refinement_summary[bin_id] = {
                "status": "error",
                "reason": str(e),
                "sub_bins": 0,
            }
    
    # Combine all refined clusters
    if all_refined_clusters:
        logger.info("Integrating refined bins into final results...")
        
        # Remove contaminated bins from original results
        clean_original_clusters = clusters_df[
            ~clusters_df["cluster"].isin(contaminated_bins)
        ].copy()
        
        # Add refined clusters
        all_refined_df = pd.concat(all_refined_clusters, ignore_index=True)
        
        # Combine clean original + refined clusters
        final_clusters_df = pd.concat(
            [clean_original_clusters, all_refined_df], ignore_index=True
        )
        
        logger.info(f"Refinement round {refinement_round} complete!")
        success_count = sum(1 for s in refinement_summary.values() if s["status"] == "success")
        logger.info(f"Refinement summary: {success_count}/{len(refinement_summary)} bins successfully refined")
        
        # Check if we should perform another round of refinement
        if refinement_round < max_refinement_rounds:
            logger.info(
                f"Checking for contaminated bins requiring round {refinement_round+1} refinement..."
            )
            
            # Check for contaminated bins in the current result
            still_contaminated_bins = []
            if "has_duplicated_core_genes" in final_clusters_df.columns:
                still_contaminated_clusters = final_clusters_df[
                    final_clusters_df["has_duplicated_core_genes"] == True
                ]["cluster"].unique()
                still_contaminated_bins = [
                    c for c in still_contaminated_clusters
                ]
            
            if still_contaminated_bins:
                logger.info(
                    f"Found {len(still_contaminated_bins)} bins still needing refinement, starting round {refinement_round+1}"
                )
                
                # Recursively refine the still-contaminated bins
                final_clusters_df, fragments_dict, additional_refinement_summary = (
                    refine_contaminated_bins_with_embeddings(
                        final_clusters_df,
                        embeddings_df,
                        fragments_dict,
                        args,
                        refinement_round=refinement_round + 1,
                        max_refinement_rounds=max_refinement_rounds,
                    )
                )
                
                # Merge refinement summaries
                refinement_summary.update(additional_refinement_summary)
            else:
                logger.info("No more contaminated bins found, refinement complete!")
        
        return final_clusters_df, fragments_dict, refinement_summary
    else:
        logger.warning("No bins were successfully refined, keeping original results")
        return clusters_df, fragments_dict, refinement_summary






def refine_contaminated_bins(
    clusters_df, fragments_dict, args, refinement_round=1, max_refinement_rounds=2
):
    """
    Refine bins that have duplicated core genes using existing embeddings with k-NN graph
    construction and Leiden clustering. This is a wrapper function that loads embeddings
    and calls the new embedding-based refinement approach.

    Args:
        clusters_df: DataFrame with cluster assignments and duplication flags
        fragments_dict: Dictionary containing fragment sequences
        args: Command line arguments
        refinement_round: Current refinement round (1-indexed)
        max_refinement_rounds: Maximum number of refinement rounds to perform

    Returns:
        tuple: (refined_clusters_df, refined_fragments_dict, refinement_summary)
    """
    # Load embeddings from CSV file
    embeddings_csv_path = os.path.join(args.output, "embeddings.csv")
    
    if not os.path.exists(embeddings_csv_path):
        logger.error(f"Embeddings file not found at {embeddings_csv_path}")
        logger.error("Cannot perform embedding-based refinement without embeddings")
        return clusters_df, fragments_dict, {}
    
    try:
        embeddings_df = pd.read_csv(embeddings_csv_path, index_col=0)
        logger.info(f"Loaded embeddings for {len(embeddings_df)} contigs from {embeddings_csv_path}")
    except Exception as e:
        logger.error(f"Failed to load embeddings from {embeddings_csv_path}: {e}")
        return clusters_df, fragments_dict, {}
    
    # Call the new embedding-based refinement function
    return refine_contaminated_bins_with_embeddings(
        clusters_df, embeddings_df, fragments_dict, args, refinement_round, max_refinement_rounds
    )
