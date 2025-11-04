"""
Refinement module for REMAG
"""

import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger
from sklearn.cluster import KMeans

from .miniprot_utils import check_core_gene_duplications, check_core_gene_duplications_from_cache, get_core_gene_duplication_results_path, get_gene_mappings_cache_path


def estimate_organisms_from_scg(bin_contigs, gene_mappings_cache):
    """
    Estimate the number of organisms in a bin based on single-copy gene (SCG) duplications.

    Args:
        bin_contigs: List of contig names in the bin
        gene_mappings_cache: Cached gene-to-contig mappings from miniprot

    Returns:
        int: Estimated number of organisms (minimum 2)
    """
    if gene_mappings_cache is None:
        logger.warning("No gene mappings available, defaulting to 2 organisms")
        return 2

    # Count occurrences of each gene family across contigs in this bin
    gene_counts = {}
    for contig_name in bin_contigs:
        if contig_name in gene_mappings_cache:
            for gene_family in gene_mappings_cache[contig_name].keys():
                gene_counts[gene_family] = gene_counts.get(gene_family, 0) + 1

    if not gene_counts:
        logger.warning("No genes found in bin, defaulting to 2 organisms")
        return 2

    # Estimate organisms as the maximum duplication count across all genes
    # This represents the worst-case contamination
    max_duplication = max(gene_counts.values())
    estimated_organisms = max(2, max_duplication)  # Minimum 2 for splitting

    logger.debug(f"SCG analysis: max duplication={max_duplication}, estimated organisms={estimated_organisms}")

    return estimated_organisms


def refine_bin_with_kmeans_clustering(
    bin_contigs, embeddings_df, fragments_dict, args, bin_id, duplication_results
):
    """
    Refine a single contaminated bin using SCG-guided KMeans clustering.

    This approach:
    1. Estimates number of organisms from SCG duplication counts
    2. Tests multiple cluster numbers around that estimate (±50%)
    3. Evaluates each clustering by core gene duplications
    4. Selects the clustering that minimizes duplications while keeping completeness high

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
    logger.info(f"Refining bin {bin_id} using SCG-guided KMeans clustering...")

    # Load gene mappings cache for SCG-based estimation and validation
    gene_mappings_cache = getattr(args, '_gene_mappings_cache', None)
    if gene_mappings_cache is None:
        cache_path = get_gene_mappings_cache_path(args)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    gene_mappings_cache = json.load(f)
                logger.debug(f"Loaded gene mappings cache from {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to load gene mappings cache: {e}")
                gene_mappings_cache = None

    # Extract embeddings for contigs in this bin
    bin_embedding_names = bin_contigs
    available_embeddings = [name for name in bin_embedding_names if name in embeddings_df.index]

    if len(available_embeddings) < 2:
        logger.warning(f"Bin {bin_id} has insufficient contigs with embeddings ({len(available_embeddings)})")
        return None

    bin_embeddings = embeddings_df.loc[available_embeddings]
    logger.info(f"Using embeddings for {len(bin_embeddings)} contigs in bin {bin_id}")

    # Estimate number of organisms from SCG duplications
    estimated_organisms = estimate_organisms_from_scg(bin_contigs, gene_mappings_cache)
    logger.info(f"Bin {bin_id} estimated to contain {estimated_organisms} organisms based on SCG analysis")

    # Calculate original bin quality before refinement (for comparison)
    original_bin_quality = None
    if gene_mappings_cache is not None:
        try:
            # Create a temporary DataFrame with all contigs in a single cluster
            temp_clusters_df = pd.DataFrame({
                'contig': bin_contigs,
                'cluster': [bin_id] * len(bin_contigs)
            })

            # Check duplications for the original bin
            temp_clusters_df = check_core_gene_duplications_from_cache(
                temp_clusters_df, gene_mappings_cache, args
            )

            # Extract metrics
            original_scg = int(temp_clusters_df['single_copy_genes_count'].iloc[0])
            original_dups = int(temp_clusters_df['duplicated_core_genes_count'].iloc[0])
            original_bin_quality = original_scg - (5 * original_dups)

            logger.info(f"Bin {bin_id} original quality: {original_bin_quality:.1f} (SCG={original_scg}, dups={original_dups})")
        except Exception as e:
            logger.warning(f"Failed to calculate original bin quality: {e}")
            original_bin_quality = None

    # Generate cluster number variations with balanced exploration
    # Test conservative to aggressive splitting: 0.5x to 3.0x of estimated organisms
    # This provides symmetric exploration around the SCG estimate
    test_cluster_counts = []
    for multiplier in [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]:
        n_clusters = max(2, int(estimated_organisms * multiplier))
        if n_clusters not in test_cluster_counts:
            test_cluster_counts.append(n_clusters)

    test_cluster_counts.sort()
    logger.info(f"Testing {len(test_cluster_counts)} cluster counts: {test_cluster_counts}")

    # Track successful solutions
    successful_attempts = []

    # Test each cluster count
    for n_clusters in test_cluster_counts:
        logger.debug(f"Bin {bin_id}: Testing k={n_clusters} clusters...")

        # Run KMeans
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300,
            algorithm='lloyd'  # Standard k-means algorithm
        )

        cluster_labels = kmeans.fit_predict(bin_embeddings.values)

        # Create cluster assignments DataFrame
        formatted_labels = [f"{bin_id}_{label}" for label in cluster_labels]

        refined_clusters_df = pd.DataFrame({
            'contig': available_embeddings,
            'cluster': formatted_labels,
            'original_bin': bin_id
        })

        # Check duplications using cached mappings
        if gene_mappings_cache is not None:
            try:
                refined_clusters_df = check_core_gene_duplications_from_cache(
                    refined_clusters_df, gene_mappings_cache, args
                )
            except Exception as e:
                logger.warning(f"Failed to check duplications for k={n_clusters}: {e}")
                continue
        else:
            logger.warning("No gene mappings cache, skipping duplication check")
            continue

        # Calculate metrics for this clustering
        bin_scg = refined_clusters_df.groupby('cluster')['single_copy_genes_count'].first()
        bin_dups = refined_clusters_df.groupby('cluster')['duplicated_core_genes_count'].first()

        # Calculate per-bin quality scores (completeness - 5*contamination)
        bin_quality_scores = bin_scg - (5 * bin_dups)

        total_duplications = int(bin_dups.sum())
        max_completeness = int(bin_scg.max()) if len(bin_scg) > 0 else 0
        p90_completeness = int(np.percentile(bin_scg, 90)) if len(bin_scg) > 0 else 0
        p90_quality = float(np.percentile(bin_quality_scores, 90)) if len(bin_quality_scores) > 0 else 0.0

        # Store this attempt
        successful_attempts.append({
            'n_clusters': n_clusters,
            'clusters_df': refined_clusters_df,
            'total_duplications': total_duplications,
            'max_completeness': max_completeness,
            'p90_completeness': p90_completeness,
            'p90_quality': p90_quality,
            'inertia': kmeans.inertia_
        })

        logger.debug(f"Bin {bin_id} k={n_clusters}: p90 quality={p90_quality:.1f}, {total_duplications} total dups, p90 completeness={p90_completeness}, max completeness={max_completeness}")

    if not successful_attempts:
        logger.warning(f"Bin {bin_id} failed all clustering attempts")
        return None

    # Select best clustering: maximize p90 quality (completeness - 5*contamination) to avoid over-fragmentation
    best_attempt = max(successful_attempts, key=lambda x: (
        x['p90_quality'],                   # Primary: maximize p90 quality (completeness - 5*contamination)
        x['p90_completeness'],              # Secondary: maximize p90 completeness
        x['max_completeness']               # Tertiary: maximize max completeness
    ))

    logger.info(f"Bin {bin_id} refinement selected: k={best_attempt['n_clusters']} clusters, "
                f"p90 quality={best_attempt['p90_quality']:.1f}, "
                f"{best_attempt['total_duplications']} total duplications, "
                f"p90 completeness={best_attempt['p90_completeness']} SCGs, "
                f"max completeness={best_attempt['max_completeness']} SCGs")

    # Compare to original bin quality - only refine if it improves quality
    if original_bin_quality is not None:
        if best_attempt['p90_quality'] <= original_bin_quality:
            logger.info(f"Bin {bin_id} refinement rejected: p90 quality {best_attempt['p90_quality']:.1f} "
                       f"does not improve original quality {original_bin_quality:.1f}")
            return None
        else:
            logger.info(f"Bin {bin_id} refinement accepted: p90 quality improved from "
                       f"{original_bin_quality:.1f} to {best_attempt['p90_quality']:.1f}")

    refined_clusters_df = best_attempt['clusters_df']
    n_refined_clusters = refined_clusters_df['cluster'].nunique()

    # Check if refinement actually helped
    if n_refined_clusters < 2:
        logger.warning(f"Bin {bin_id} refinement produced only {n_refined_clusters} cluster(s)")
        return None

    logger.info(f"Bin {bin_id} successfully refined into {n_refined_clusters} sub-bins")

    return refined_clusters_df


def refine_contaminated_bins_with_embeddings(
    clusters_df, embeddings_df, fragments_dict, args, refinement_round=1, max_refinement_rounds=16
):
    """
    Refine bins that have duplicated core genes using SCG-guided KMeans clustering.

    This approach:
    1. Identifies bins with duplicated core genes
    2. For each contaminated bin, extracts embeddings of its contigs
    3. Estimates number of organisms from single-copy gene (SCG) duplications
    4. Tests multiple cluster counts around the SCG estimate (±50%)
    5. Applies KMeans clustering with each cluster count
    6. Selects the clustering that minimizes duplications while keeping completeness high
    7. Checks for duplications in refined sub-bins
    8. Iteratively refines still-contaminated sub-bins

    This approach is efficient as it:
    - Reuses existing embeddings (no retraining)
    - Uses SCG counts to guide cluster number selection
    - Uses standard KMeans algorithm for accurate clustering
    - Prioritizes decontamination while maintaining genome completeness

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

    # Load duplication results from memory or file
    duplication_results = {}

    # Get results path (needed for saving updated results later)
    results_path = get_core_gene_duplication_results_path(args)

    # First try to get from args (available even without -k flag)
    if hasattr(args, '_duplication_results'):
        duplication_results = args._duplication_results
        logger.debug(f"Using in-memory duplication results for {len(duplication_results)} bins")
    else:
        # Fall back to loading from file (only available with -k flag)
        if os.path.exists(results_path):
            try:
                with open(results_path, "r") as f:
                    duplication_results = json.load(f)
                logger.info(f"Loaded duplication results from file for {len(duplication_results)} bins")
            except Exception as e:
                logger.warning(f"Failed to load duplication results from file: {e}")
        else:
            logger.warning("No duplication results available (neither in memory nor file); skipping refinement")
    
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
                # Bin is flagged as contaminated but missing from duplication_results
                # This can happen if bins were created/renamed during refinement
                # Get duplication count from clusters_df instead
                bin_rows = clusters_df[clusters_df["cluster"] == bin_id]
                if not bin_rows.empty:
                    bin_data = bin_rows.iloc[0]
                    dup_count = bin_data["duplicated_core_genes_count"] if "duplicated_core_genes_count" in bin_data.index else 0
                    # Convert to int (handles numpy types from pandas)
                    try:
                        dup_count_int = int(dup_count)
                    except (ValueError, TypeError):
                        dup_count_int = 0

                    if dup_count_int >= min_duplications:
                        logger.info(f"REFINEMENT: {bin_id} selected - {dup_count_int} duplicated genes from clusters_df (not in duplication_results)")
                        contaminated_bins.append(bin_id)
                        # Add to duplication_results for consistency
                        fake_duplicated_genes = {f"gene_{i}": 2 for i in range(dup_count_int)}
                        total_genes_found = bin_data["total_core_genes_found"] if "total_core_genes_found" in bin_data.index else 0
                        duplication_results[bin_id] = {
                            "has_duplications": True,
                            "duplicated_genes": fake_duplicated_genes,
                            "total_genes_found": int(total_genes_found) if total_genes_found else 0
                        }
                        args._duplication_results = duplication_results
                    else:
                        logger.debug(f"REFINEMENT: {bin_id} skipped - only {dup_count_int} duplicated genes (< {min_duplications})")
                else:
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
    logger.info("Using existing embeddings with SCG-guided KMeans clustering")
    
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

            # Refine this bin using SCG-guided KMeans clustering
            refined_clusters_df = refine_bin_with_kmeans_clustering(
                bin_contigs, embeddings_df, fragments_dict, args, bin_id, duplication_results
            )
            
            if refined_clusters_df is None:
                refinement_summary[bin_id] = {
                    "status": "failed", 
                    "reason": "clustering_failed_or_too_fragmented",
                    "sub_bins": 0,
                }
                failed_refinement_bins.append(bin_id)
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
                        target_coverage_threshold=0.50,
                        identity_threshold=0.30,
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
                logger.debug(f"Extracting duplication data from {len(refined_clusters_df['cluster'].unique())} refined sub-bins of {bin_id}")
                logger.debug(f"DataFrame columns: {refined_clusters_df.columns.tolist()}")

                for refined_bin_id in refined_clusters_df['cluster'].unique():
                    bin_data = refined_clusters_df[refined_clusters_df['cluster'] == refined_bin_id].iloc[0]

                    # Log RAW values (use .get() for logging only to avoid KeyError)
                    raw_has_dups = bin_data.get('has_duplicated_core_genes', 'MISSING')
                    raw_dup_count = bin_data.get('duplicated_core_genes_count', 'MISSING')
                    raw_total_genes = bin_data.get('total_core_genes_found', 'MISSING')
                    logger.debug(f"RAW {refined_bin_id}: has_dups={raw_has_dups}, dup_count={raw_dup_count}, total={raw_total_genes}")

                    # Use bracket notation for pandas Series, not .get()
                    has_dups = bin_data['has_duplicated_core_genes'] if 'has_duplicated_core_genes' in bin_data.index else False
                    dup_count = bin_data['duplicated_core_genes_count'] if 'duplicated_core_genes_count' in bin_data.index else 0
                    total_genes = bin_data['total_core_genes_found'] if 'total_core_genes_found' in bin_data.index else 0

                    # Handle NaN values explicitly to prevent extraction failures
                    if pd.isna(dup_count):
                        logger.debug(f"{refined_bin_id}: dup_count was NaN, converting to 0")
                        dup_count = 0
                    if pd.isna(total_genes):
                        logger.debug(f"{refined_bin_id}: total_genes was NaN, converting to 0")
                        total_genes = 0
                    if pd.isna(has_dups):
                        logger.debug(f"{refined_bin_id}: has_dups was NaN, converting to False")
                        has_dups = False

                    # Ensure consistency: if has_dups is True, dup_count should be > 0
                    if has_dups and dup_count == 0:
                        logger.warning(
                            f"INCONSISTENCY for {refined_bin_id}: "
                            f"has_duplicated_core_genes={has_dups} (type {type(has_dups).__name__}) "
                            f"but duplicated_core_genes_count={dup_count} (type {type(dup_count).__name__}). "
                            f"Setting count to 1."
                        )
                        dup_count = 1  # Minimal contamination assumption

                    # Create fake duplicated_genes dict with count matching dup_count
                    # Convert to int first (handles numpy types from pandas DataFrame)
                    fake_duplicated_genes = {}
                    try:
                        dup_count_int = int(dup_count)
                        if dup_count_int > 0:
                            for i in range(dup_count_int):
                                fake_duplicated_genes[f"gene_{i}"] = 2  # Fake gene with 2 copies
                    except (ValueError, TypeError):
                        logger.debug(f"{refined_bin_id}: Could not convert dup_count={dup_count} (type={type(dup_count)}) to int")

                    # Add to duplication_results for next refinement round
                    duplication_results[refined_bin_id] = {
                        "has_duplications": bool(has_dups),
                        "duplicated_genes": fake_duplicated_genes,
                        "total_genes_found": int(total_genes) if total_genes else 0
                    }

                    # Log ALL bins with what was stored
                    logger.debug(
                        f"Stored {refined_bin_id}: has_duplications={bool(has_dups)}, "
                        f"duplicated_genes={len(fake_duplicated_genes)}, total_genes={int(total_genes)}"
                    )
                
                # Update in-memory duplication results for next refinement round
                args._duplication_results = duplication_results
                logger.debug(f"Updated in-memory duplication results with {len(refined_clusters_df['cluster'].unique())} refined sub-bins from {bin_id}")

                # Save updated duplication results back to file if keeping intermediate files
                if getattr(args, "keep_intermediate", False):
                    try:
                        with open(results_path, "w") as f:
                            json.dump(duplication_results, f, indent=2)
                        logger.debug(f"Saved updated duplication results file with {len(refined_clusters_df['cluster'].unique())} refined sub-bins from {bin_id}")
                    except Exception as e:
                        logger.warning(f"Failed to save updated duplication results: {e}")
            
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
        # Apply same min_duplications filter to avoid false positives
        # Also exclude bins that already failed refinement (refinement_failed == True)
        still_contaminated_bins = []

        # Log duplication_results contents at start of round 2+ detection
        logger.debug(f"Round {refinement_round+1} detection: duplication_results has {len(duplication_results)} bins")
        if duplication_results:
            contaminated_in_results = sum(1 for d in duplication_results.values() if len(d.get('duplicated_genes', {})) > 0)
            logger.debug(f"  - {contaminated_in_results} bins in duplication_results have >0 duplicated genes")

        if "has_duplicated_core_genes" in final_clusters_df.columns:
            # Filter contaminated bins, excluding those that already failed refinement in previous rounds
            contaminated_mask = final_clusters_df["has_duplicated_core_genes"] == True

            # Also exclude bins with refinement_failed=True if the column exists
            if "refinement_failed" in final_clusters_df.columns:
                # Use fillna(False) to handle NaN values - only exclude if explicitly True
                not_failed_mask = final_clusters_df["refinement_failed"].fillna(False).infer_objects(copy=False) == False
                still_contaminated_clusters = final_clusters_df[contaminated_mask & not_failed_mask]["cluster"].unique()

                # Count how many were excluded
                all_contaminated = final_clusters_df[contaminated_mask]["cluster"].nunique()
                failed_contaminated = all_contaminated - len(still_contaminated_clusters)
                logger.debug(f"Found {all_contaminated} contaminated bins, excluding {failed_contaminated} that failed refinement previously")
            else:
                still_contaminated_clusters = final_clusters_df[contaminated_mask]["cluster"].unique()
                logger.debug(f"Found {len(still_contaminated_clusters)} bins flagged as contaminated in final_clusters_df")

            # Filter by minimum duplications threshold
            for bin_id in still_contaminated_clusters:

                if bin_id in duplication_results:
                    dup_data = duplication_results[bin_id]
                    duplicated_count = len(dup_data.get("duplicated_genes", {}))
                    has_dups_flag = dup_data.get("has_duplications", False)
                    logger.debug(
                        f"Checking {bin_id} in duplication_results: has_duplications={has_dups_flag}, "
                        f"duplicated_genes count={duplicated_count}, total_genes={dup_data.get('total_genes_found', 0)}"
                    )
                    if duplicated_count >= min_duplications:
                        still_contaminated_bins.append(bin_id)
                        logger.debug(f"✓ Selected {bin_id} for refinement: {duplicated_count} duplicated genes (>= {min_duplications})")
                    else:
                        logger.debug(f"✗ Skipping {bin_id}: only {duplicated_count} duplicated genes (< {min_duplications})")
                else:
                    # Get count from clusters_df (fallback when not in duplication_results)
                    logger.debug(f"{bin_id} NOT in duplication_results, checking clusters_df")
                    if not bin_rows.empty:
                        bin_data = bin_rows.iloc[0]
                        dup_count = bin_data["duplicated_core_genes_count"] if "duplicated_core_genes_count" in bin_data.index else 0
                        # Handle NaN values and convert to int (handles numpy types)
                        if pd.isna(dup_count):
                            dup_count = 0
                        try:
                            dup_count_int = int(dup_count)
                        except (ValueError, TypeError):
                            dup_count_int = 0

                        logger.debug(f"{bin_id} from clusters_df: duplicated_core_genes_count={dup_count_int}")
                        if dup_count_int >= min_duplications:
                            still_contaminated_bins.append(bin_id)
                            logger.debug(f"✓ Selected {bin_id} for refinement: {dup_count_int} duplicated genes (from clusters_df)")
                        else:
                            logger.debug(f"✗ Skipping {bin_id}: only {dup_count_int} duplicated genes (< {min_duplications})")

        # Summary of round 2+ detection
        logger.debug(f"Round {refinement_round+1} detection complete: {len(still_contaminated_bins)} bins selected for refinement")
        if still_contaminated_bins:
            logger.debug(f"  Selected bins: {', '.join(still_contaminated_bins)}")

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

            # Log summary of bins that still have contamination but couldn't be refined
            if "refinement_failed" in final_clusters_df.columns:
                failed_bins = final_clusters_df[
                    final_clusters_df["refinement_failed"] == True
                ]["cluster"].unique()
                if len(failed_bins) > 0:
                    logger.info(f"{len(failed_bins)} bins remain with contamination but could not be refined: {', '.join(failed_bins)}")

    return final_clusters_df, fragments_dict, refinement_summary






def refine_contaminated_bins(
    clusters_df, fragments_dict, args, refinement_round=1, max_refinement_rounds=16
):
    """
    Refine bins that have duplicated core genes using SCG-guided KMeans clustering.
    This is a wrapper function that loads embeddings and calls the embedding-based
    refinement approach with automatic cluster number selection based on SCG counts.

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
