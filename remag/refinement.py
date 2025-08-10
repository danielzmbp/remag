"""
Refinement module for REMAG
"""

import json
import os
import pandas as pd
from tqdm import tqdm
from loguru import logger

from .miniprot_utils import check_core_gene_duplications, get_core_gene_duplication_results_path
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
    
    # Use standard Leiden parameters (no adaptive resolution tuning for now)
    leiden_resolution = getattr(args, 'leiden_resolution', 1.0)
    
    # Log duplication info for reference
    if bin_id in duplication_results:
        duplicated_genes_count = len(duplication_results[bin_id]["duplicated_genes"])
        total_genes_found = duplication_results[bin_id]["total_genes_found"]
        logger.info(
            f"Bin {bin_id} has {duplicated_genes_count} duplicated core genes out of {total_genes_found} total genes"
        )
    
    # Use standard k-NN parameters
    leiden_k_neighbors = getattr(args, 'leiden_k_neighbors', 15)
    leiden_similarity_threshold = getattr(args, 'leiden_similarity_threshold', 0.1)
    
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
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = sum(1 for label in cluster_labels if label == -1)
    
    logger.info(f"Bin {bin_id} Leiden clustering: {n_clusters} clusters, {n_noise} noise points")
    
    if n_clusters < 2:
        logger.warning(f"Bin {bin_id} refinement produced insufficient clusters ({n_clusters})")
        return None
    
    # Create cluster assignments DataFrame
    # Embeddings already use base contig names without .original suffix
    contig_names = available_embeddings
    
    # Format cluster labels with bin prefix
    formatted_labels = [
        f"{bin_id}_refined_bin_{label}" if label != -1 else "noise"
        for label in cluster_labels
    ]
    
    refined_clusters_df = pd.DataFrame({
        'contig': contig_names,
        'cluster': formatted_labels,
        'original_bin': bin_id
    })
    
    # Remove noise contigs from results
    refined_clusters_df = refined_clusters_df[refined_clusters_df['cluster'] != 'noise']
    
    if refined_clusters_df.empty:
        logger.warning(f"Bin {bin_id} refinement resulted in all noise contigs")
        return None
        
    logger.info(f"Bin {bin_id} successfully refined into {n_clusters} sub-bins")
    
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
    # Identify contaminated bins
    contaminated_bins = []
    if "has_duplicated_core_genes" in clusters_df.columns:
        contaminated_clusters = clusters_df[
            clusters_df["has_duplicated_core_genes"] == True
        ]["cluster"].unique()
        contaminated_bins = [c for c in contaminated_clusters if c != "noise"]

    if not contaminated_bins:
        logger.info("No contaminated bins found, skipping refinement")
        return clusters_df, fragments_dict, {}

    if refinement_round > max_refinement_rounds:
        logger.info(
            f"Maximum refinement rounds ({max_refinement_rounds}) reached, marking remaining contaminated bins without further refinement"
        )
        return clusters_df, fragments_dict, {}

    logger.info(
        f"Starting embedding-based refinement round {refinement_round} of {len(contaminated_bins)} contaminated bins..."
    )
    logger.info("Using existing embeddings with k-NN graph construction and Leiden clustering")
    
    # Load duplication results for parameter tuning
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
        logger.warning("No duplication results file found, will use default clustering parameters")
    
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
                
            # Check for duplicated core genes in refined bins
            logger.debug(f"Checking core gene duplications in {bin_id} refined sub-bins...")
            refined_clusters_df = check_core_gene_duplications(
                refined_clusters_df,
                fragments_dict,
                args,
                target_coverage_threshold=0.60,  # Match compleasm standard (60%)
                identity_threshold=0.40,  # Match compleasm standard (40%)
                use_header_cache=True
            )
            
            # Count successful sub-bins (exclude noise)
            sub_bins = refined_clusters_df[refined_clusters_df["cluster"] != "noise"]["cluster"].nunique()
            
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
                    c for c in still_contaminated_clusters if c != "noise"
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
