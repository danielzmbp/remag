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
    
    # Load gene mappings cache for marker-based validation
    gene_mappings_cache = getattr(args, '_gene_mappings_cache', None)
    if gene_mappings_cache is None:
        # Try to load from file if keeping intermediate files
        cache_path = get_gene_mappings_cache_path(args)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    gene_mappings_cache = json.load(f)
                logger.debug(f"Loaded gene mappings cache from {cache_path} for validation")
            except Exception as e:
                logger.warning(f"Failed to load gene mappings cache for validation: {e}")
                gene_mappings_cache = None
    
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
    
    # Define validation functions (used in the retry loop below)
    def validate_refinement_quality_fallback(original_contigs, refined_clusters_df, bin_id):
        """Fallback contig-based validation when gene mappings unavailable."""
        cluster_sizes = refined_clusters_df.groupby('cluster').size()
        largest_cluster_size = cluster_sizes.max()
        original_size = len(original_contigs)
        
        if largest_cluster_size < original_size * 0.4:
            logger.warning(f"Bin {bin_id} extreme fragmentation detected (largest={largest_cluster_size}/{original_size}, <40% retention)")
            return False
        
        small_clusters = (cluster_sizes < 5).sum()
        if small_clusters > 1:
            logger.warning(f"Bin {bin_id} refinement created {small_clusters} small clusters")
            return False
        
        return True
    
    def validate_refinement_with_markers(original_contigs, refined_clusters_df, bin_id, gene_mappings_cache, duplication_results, margin_factor=2.0):
        """
        Validate refinement using marker genes with intelligent trade-offs.
        Allow minimal splitting of single-copy genes ONLY if it significantly reduces contamination.
        
        Returns:
            str: 'success', 'no_duplications_resolved', 'excessive_fragmentation', 'trade_off_unfavorable', 'no_gene_mappings'
        """
        if gene_mappings_cache is None:
            logger.warning(f"Bin {bin_id} no gene mappings available, falling back to contig-based validation")
            fallback_result = validate_refinement_quality_fallback(original_contigs, refined_clusters_df, bin_id)
            return 'success' if fallback_result else 'no_gene_mappings'
        
        # Get original bin's gene composition
        original_gene_counts = {}
        for contig_name in original_contigs:
            if contig_name in gene_mappings_cache:
                for gene_family in gene_mappings_cache[contig_name].keys():
                    original_gene_counts[gene_family] = original_gene_counts.get(gene_family, 0) + 1
        
        if not original_gene_counts:
            logger.warning(f"Bin {bin_id} no genes found in original bin, falling back to contig-based validation")
            fallback_result = validate_refinement_quality_fallback(original_contigs, refined_clusters_df, bin_id)
            return 'success' if fallback_result else 'no_gene_mappings'
        
        # Analyze gene distribution across refined clusters
        cluster_genes = {}
        for _, row in refined_clusters_df.iterrows():
            contig_name = row['contig']
            cluster_id = row['cluster']
            if cluster_id not in cluster_genes:
                cluster_genes[cluster_id] = {}
            
            if contig_name in gene_mappings_cache:
                for gene_family in gene_mappings_cache[contig_name].keys():
                    cluster_genes[cluster_id][gene_family] = cluster_genes[cluster_id].get(gene_family, 0) + 1
        
        # Calculate split penalty and contamination reduction with detailed tracking
        split_penalty = 0
        contamination_reduction = 0
        split_genes = []
        resolved_genes = []
        
        for gene_family, original_count in original_gene_counts.items():
            clusters_with_gene = sum(1 for cluster_gene_counts in cluster_genes.values() 
                                   if gene_family in cluster_gene_counts)
            
            if original_count == 1:
                # Single-copy gene in original bin
                if clusters_with_gene > 1:
                    split_penalty += 1  # Penalty for splitting single-copy gene
                    split_genes.append(gene_family)
            elif original_count > 1:
                # Duplicated gene in original bin
                if clusters_with_gene == 1:
                    contamination_reduction += 1  # Benefit for resolving duplication
                    resolved_genes.append(f"{gene_family}({original_count}->1)")
        
        # Apply trade-off formula: contamination_reduction > (split_penalty * margin_factor)
        trade_off_ratio = contamination_reduction / max(1, split_penalty) if split_penalty > 0 else float('inf')
        trade_off_acceptable = contamination_reduction > (split_penalty * margin_factor)
        
        # Additional basic checks
        cluster_gene_counts = [len(genes) for genes in cluster_genes.values()]
        largest_cluster_genes = max(cluster_gene_counts) if cluster_gene_counts else 0
        total_original_genes = len(original_gene_counts)
        
        # Check 0: Must resolve at least some contamination to justify any splitting
        if contamination_reduction == 0:
            logger.warning(f"Bin {bin_id} refinement resolves no duplications - keeping original bin")
            return 'no_duplications_resolved'
        
        # Check 1: Trade-off assessment
        if split_penalty > 0 and not trade_off_acceptable:
            split_details = f" (genes: {', '.join(split_genes)})" if split_genes else ""
            resolved_details = f" (genes: {', '.join(resolved_genes)})" if resolved_genes else ""
            logger.warning(f"Bin {bin_id} trade-off unfavorable: splits {split_penalty} single-copy genes{split_details} but only resolves {contamination_reduction} duplications{resolved_details} (ratio {trade_off_ratio:.2f} < {margin_factor})")
            return 'trade_off_unfavorable'
        
        # Check 2: Single-copy gene integrity - ensure most single-copy genes stay together
        single_copy_genes = [gene for gene, count in original_gene_counts.items() if count == 1]
        if single_copy_genes:
            # Find which cluster has the most single-copy genes
            cluster_single_copy_counts = {}
            for cluster_id in cluster_genes:
                cluster_single_copy_counts[cluster_id] = sum(1 for gene in single_copy_genes 
                                                            if gene in cluster_genes[cluster_id])
            
            # Calculate retention ratio for the main cluster
            max_single_copy_retention = max(cluster_single_copy_counts.values()) if cluster_single_copy_counts else 0
            single_copy_retention_ratio = max_single_copy_retention / len(single_copy_genes)
            
            if single_copy_retention_ratio < 0.9:  # Less than 90% stay together
                logger.warning(f"Bin {bin_id} excessive fragmentation of single-copy genes "
                              f"(only {single_copy_retention_ratio:.1%} stay together, {max_single_copy_retention}/{len(single_copy_genes)}) - keeping original bin")
                return 'excessive_fragmentation'
        
        # Log success with details
        if split_penalty == 0:
            resolved_details = f" (genes: {', '.join(resolved_genes)})" if resolved_genes else ""
            logger.info(f"Bin {bin_id} perfect separation: no single-copy genes split, resolves {contamination_reduction} duplications{resolved_details}")
        else:
            split_details = f" (genes: {', '.join(split_genes)})" if split_genes else ""
            resolved_details = f" (genes: {', '.join(resolved_genes)})" if resolved_genes else ""
            logger.info(f"Bin {bin_id} acceptable trade-off: splits {split_penalty} single-copy genes{split_details} but resolves {contamination_reduction} duplications{resolved_details} (ratio {trade_off_ratio:.2f} > {margin_factor})")
        
        return 'success'
    
    # Adaptive parameter selection with retry logic
    def get_adaptive_leiden_params(args, attempt, failure_reason=None):
        """Get adaptive Leiden parameters based on attempt and previous failure.

        Note: base_resolution comes from args.leiden_resolution, which contains:
        - The auto-calculated optimal resolution (when auto-resolution is enabled, the default)
        - The user-specified resolution (when --leiden-resolution is provided)
        - Fallback of 1.0 (only if auto-resolution fails and no manual value provided)

        This ensures refinement adjustments are proportional to the sample's diversity level.
        """
        base_resolution = getattr(args, 'leiden_resolution', 1.0)
        base_k_neighbors = getattr(args, 'leiden_k_neighbors', 15)
        base_threshold = getattr(args, 'leiden_similarity_threshold', 0.1)
        
        if attempt == 0:
            # First try: Standard refinement parameters (not overly conservative)
            resolution_multiplier = 1.0
            k_adjustment = 0
            threshold_multiplier = 1.0
        elif failure_reason == "no_duplications_resolved":
            # Need MORE clusters - increase resolution, reduce connectivity
            resolution_multiplier = 1.5 * (attempt)  # 1.5x, 3.0x
            k_adjustment = -5 * attempt  # Fewer neighbors = less connectivity
            threshold_multiplier = 0.7  # Lower threshold = weaker connections
        elif failure_reason in ["excessive_fragmentation", "trade_off_unfavorable"]:
            # Need FEWER clusters - decrease resolution, increase connectivity
            resolution_multiplier = 0.5 / attempt  # 0.5x, 0.25x
            k_adjustment = 10 * attempt  # More neighbors = more connectivity
            threshold_multiplier = 1.5  # Higher threshold = stronger connections
        else:
            # Default progression for unknown failures
            resolution_multiplier = 1.0 + (attempt - 1) * 0.3  # 1.0x, 1.3x, 1.6x
            k_adjustment = 0
            threshold_multiplier = 1.0
        
        # Apply adjustments with bounds
        leiden_resolution = base_resolution * resolution_multiplier
        leiden_k_neighbors = max(5, min(50, base_k_neighbors + k_adjustment))
        leiden_similarity_threshold = max(0.05, min(0.5, base_threshold * threshold_multiplier))
        
        return leiden_resolution, leiden_k_neighbors, leiden_similarity_threshold
    
    # Try adaptive refinement with up to 3 attempts
    max_attempts = 3
    failure_reason = None
    refined_clusters_df = None
    n_clusters = 0

    # Log the base resolution being used for refinement
    base_resolution_value = getattr(args, 'leiden_resolution', 1.0)
    logger.info(f"Bin {bin_id} refinement using base resolution: {base_resolution_value:.2f} (from auto-resolution or manual setting)")

    # Log duplication info for reference
    if bin_id in duplication_results:
        duplicated_genes_count = len(duplication_results[bin_id]["duplicated_genes"])
        total_genes_found = duplication_results[bin_id]["total_genes_found"]
        logger.info(
            f"Bin {bin_id} has {duplicated_genes_count} duplicated core genes out of {total_genes_found} total genes"
        )

    for attempt in range(max_attempts):
        leiden_resolution, leiden_k_neighbors, leiden_similarity_threshold = get_adaptive_leiden_params(args, attempt, failure_reason)
        
        attempt_info = f"attempt {attempt+1}/{max_attempts}"
        if attempt > 0:
            attempt_info += f" (after {failure_reason})"
        
        logger.info(f"Bin {bin_id} {attempt_info}: resolution={leiden_resolution:.2f}, k={leiden_k_neighbors}, threshold={leiden_similarity_threshold:.2f}")
        
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
        logger.info(f"Bin {bin_id} {attempt_info}: Leiden clustering produced {n_clusters} clusters")
        
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
            logger.info(f"Bin {bin_id} {attempt_info}: Merged {len(small_clusters)} small clusters, now {n_clusters} clusters")
        
        if n_clusters < 2:
            failure_reason = "no_duplications_resolved"  # Assume insufficient splitting
            logger.warning(f"Bin {bin_id} {attempt_info}: Insufficient clusters ({n_clusters})")
            if attempt == max_attempts - 1:
                logger.warning(f"Bin {bin_id} failed after {max_attempts} attempts")
                return None
            continue
        
        # Create cluster assignments DataFrame
        # Embeddings already use base contig names without .original suffix
        contig_names = available_embeddings
        
        # Format cluster labels with clean naming scheme (bin_X becomes bin_X_0, bin_X_1, etc.)
        formatted_labels = [
            f"{bin_id}_{label}" for label in cluster_labels
        ]
        
        refined_clusters_df = pd.DataFrame({
            'contig': contig_names,
            'cluster': formatted_labels,
            'original_bin': bin_id
        })
        
        # Validate refinement FIRST using only contigs that were actually refined (have embeddings)
        validation_result = validate_refinement_with_markers(available_embeddings, refined_clusters_df, bin_id, gene_mappings_cache, duplication_results)
        
        if validation_result == 'success':
            logger.info(f"Bin {bin_id} {attempt_info}: Validation passed!")
            break  # Success - exit retry loop
        else:
            failure_reason = validation_result
            logger.warning(f"Bin {bin_id} {attempt_info}: Validation failed - {failure_reason}")
            
            if attempt == max_attempts - 1:
                logger.warning(f"Bin {bin_id} failed validation after {max_attempts} attempts, keeping original")
                return None
            # Continue to next attempt with adjusted parameters
    
    # If we get here, validation passed - continue with success path
    
    # AFTER validation passes, handle contigs without embeddings - assign them to the largest refined cluster
    contigs_without_embeddings = [contig for contig in bin_contigs if contig not in available_embeddings]
    if contigs_without_embeddings:
        # Find the largest refined cluster
        cluster_sizes = refined_clusters_df.groupby('cluster').size()
        largest_cluster = cluster_sizes.idxmax()
        
        # Create entries for contigs without embeddings
        additional_rows = []
        for contig in contigs_without_embeddings:
            additional_rows.append({
                'contig': contig,
                'cluster': largest_cluster,
                'original_bin': bin_id
            })
        
        # Add to refined_clusters_df
        additional_df = pd.DataFrame(additional_rows)
        refined_clusters_df = pd.concat([refined_clusters_df, additional_df], ignore_index=True)
        
        logger.debug(f"Bin {bin_id} assigned {len(contigs_without_embeddings)} contigs without embeddings to largest cluster {largest_cluster}")
        
    # Calculate final statistics
    n_refined_clusters = refined_clusters_df['cluster'].nunique()
    largest_subbin_size = refined_clusters_df.groupby('cluster').size().max()
    retention_ratio = largest_subbin_size / len(available_embeddings)
    logger.info(f"Bin {bin_id} successfully refined into {n_refined_clusters} sub-bins (largest retains {retention_ratio:.1%})")
    
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
    # Identify contaminated bins - attempt refinement with even single duplicated genes
    min_duplications = getattr(args, 'min_duplications_for_refinement', 1)
    
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
    failed_refinement_bins = []
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
                failed_refinement_bins.append(bin_id)
                continue
                
            bin_contigs = bin_contigs_df["contig"].tolist()
            
            # Refine this bin using Leiden clustering
            refined_clusters_df = refine_bin_with_leiden_clustering(
                bin_contigs, embeddings_df, fragments_dict, args, bin_id, duplication_results
            )
            
            if refined_clusters_df is None:
                refinement_summary[bin_id] = {
                    "status": "failed", 
                    "reason": "clustering_failed_or_too_fragmented",
                    "sub_bins": 0,
                }
                failed_refinement_bins.append(bin_id)
                logger.info(f"Bin {bin_id} kept original due to failed refinement")
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
            
            # Extract duplication results for refined sub-bins and add to duplication_results dictionary
            # This ensures they're available for subsequent refinement rounds
            if not duplication_check_failed and 'has_duplicated_core_genes' in refined_clusters_df.columns:
                for refined_bin_id in refined_clusters_df['cluster'].unique():
                    bin_data = refined_clusters_df[refined_clusters_df['cluster'] == refined_bin_id].iloc[0]
                    has_dups = bin_data.get('has_duplicated_core_genes', False)
                    dup_count = bin_data.get('duplicated_core_genes_count', 0)
                    total_genes = bin_data.get('total_core_genes_found', 0)
                    
                    # Create fake duplicated_genes dict with count matching dup_count
                    fake_duplicated_genes = {}
                    if isinstance(dup_count, (int, float)) and dup_count > 0:
                        for i in range(int(dup_count)):
                            fake_duplicated_genes[f"gene_{i}"] = 2  # Fake gene with 2 copies
                    
                    # Add to duplication_results for next refinement round
                    duplication_results[refined_bin_id] = {
                        "has_duplications": bool(has_dups),
                        "duplicated_genes": fake_duplicated_genes,
                        "total_genes_found": int(total_genes) if total_genes else 0
                    }
                
                # Save updated duplication results back to file for next refinement round
                try:
                    with open(results_path, "w") as f:
                        json.dump(duplication_results, f, indent=2)
                    logger.debug(f"Updated duplication results file with {len(refined_clusters_df['cluster'].unique())} refined sub-bins from {bin_id}")
                except Exception as e:
                    logger.warning(f"Failed to save updated duplication results: {e}")
                    
                logger.debug(f"Added duplication results for {len(refined_clusters_df['cluster'].unique())} refined sub-bins from {bin_id}")
            
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
                failed_refinement_bins.append(bin_id)
                logger.warning(f"Refinement of {bin_id} produced only {sub_bins} sub-bins, keeping original")
                
        except Exception as e:
            logger.error(f"Error during refinement of {bin_id}: {e}")
            refinement_summary[bin_id] = {
                "status": "error",
                "reason": str(e),
                "sub_bins": 0,
            }
            failed_refinement_bins.append(bin_id)
    
    # Combine all refined clusters
    logger.info("Integrating refined bins into final results...")
    
    # Remove contaminated bins from original results (both successfully refined and failed)
    successfully_refined_bins = [bin_id for bin_id in contaminated_bins if bin_id not in failed_refinement_bins]
    clean_original_clusters = clusters_df[
        ~clusters_df["cluster"].isin(successfully_refined_bins)
    ].copy()
    
    # Add back failed refinement bins with a flag
    if failed_refinement_bins:
        logger.info(f"Keeping {len(failed_refinement_bins)} bins that failed refinement in their original form")
        failed_bins_df = clusters_df[clusters_df["cluster"].isin(failed_refinement_bins)].copy()
        failed_bins_df["refinement_failed"] = True
        # Keep the original contamination flag for failed bins
        clean_original_clusters = clean_original_clusters[~clean_original_clusters["cluster"].isin(failed_refinement_bins)]
        clean_original_clusters = pd.concat([clean_original_clusters, failed_bins_df], ignore_index=True)
    
    if all_refined_clusters:
        # Add refined clusters
        all_refined_df = pd.concat(all_refined_clusters, ignore_index=True)
        
        # Combine clean original + refined clusters
        final_clusters_df = pd.concat(
            [clean_original_clusters, all_refined_df], ignore_index=True
        )
    else:
        final_clusters_df = clean_original_clusters
    
    logger.info(f"Refinement round {refinement_round} complete!")
    success_count = sum(1 for s in refinement_summary.values() if s["status"] == "success")
    failed_count = len(failed_refinement_bins)
    logger.info(f"Refinement summary: {success_count}/{len(refinement_summary)} bins successfully refined, {failed_count} kept original due to failed refinement")
    
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
