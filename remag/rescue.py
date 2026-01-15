"""
Rescue module for REMAG.
Implements "Satellite Rescue" strategy to merge fragmented bins based on embedding similarity and SCG safety.
"""

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

from .miniprot_utils import get_gene_mappings_cache_path

def get_bin_scg_stats(bin_contigs, gene_mappings_cache):
    """Calculate SCG duplication % for a list of contigs."""
    if not bin_contigs: return 0.0, 0.0
    
    bin_genes = {}
    for c in bin_contigs:
        if c in gene_mappings_cache:
            # gene_mappings_cache[c] is a dict of {gene_id: info}
            for gene_id in gene_mappings_cache[c].keys():
                bin_genes[gene_id] = bin_genes.get(gene_id, 0) + 1
    
    present = len(bin_genes)
    if present == 0: return 0.0, 0.0
    
    duplicated = len([g for g, count in bin_genes.items() if count > 1])
    duplication_rate = (duplicated / present) * 100.0
    return duplication_rate, present


def rescue_fragmented_bins(
    clusters_df, 
    embeddings_df, 
    fragments_dict, 
    args, 
    similarity_threshold=0.70, 
    max_duplication_increase=3.0
):
    """
    Attempt to merge smaller bins (or split parts of genomes) into larger "Core Bins"
    based on global embedding centroid similarity, provided it is safe (SCG check).
    """
    logger.info("Running Satellite Rescue to merge fragmented bins...")

    # 1. Load Gene Mappings Cache
    gene_mappings_cache = getattr(args, "_gene_mappings_cache", None)
    if gene_mappings_cache is None:
        cache_path = get_gene_mappings_cache_path(args)
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                gene_mappings_cache = json.load(f)
    
    if not gene_mappings_cache:
        logger.warning("No gene mappings cache found. Skipping rescue step as safety checks require SCGs.")
        return clusters_df

    # 2. Prepare Data
    # Calculate contig lengths
    contig_lengths = {k: len(v['sequence']) for k, v in fragments_dict.items()}
    
    # Identify Core Bins (potential targets) - anything that is NOT noise
    core_bins = set(clusters_df[clusters_df['cluster'] != 'noise']['cluster'].unique())
    
    if len(core_bins) == 0:
        logger.info("No core bins found. Skipping rescue.")
        return clusters_df

    # Expand noise into temporary singleton bins for rescue attempts
    # We modify the dataframe in place (it will be returned)
    
    # Label noise as 'singleton_idx'
    noise_mask = clusters_df['cluster'] == 'noise'
    n_noise_initial = noise_mask.sum()
    
    if n_noise_initial > 0:
        logger.info(f"Preparing {n_noise_initial} noise contigs for potential rescue...")
        # Assign unique singleton IDs
        # We use a loop or vectorized assignment
        new_labels = [f"singleton_{i}" for i in range(n_noise_initial)]
        clusters_df.loc[noise_mask, 'cluster'] = new_labels
    
    # Now all_bins includes Core Bins + Temporary Singletons
    all_bins = clusters_df['cluster'].unique()
    
    # 3. Calculate Centroids for ALL Bins
    bin_centroids = {}
    bin_sizes = {} # in bp
    
    # Store intermediate sums for efficient centroid updates
    bin_weighted_sums = {}
    bin_total_weights = {}

    logger.debug(f"Calculating centroids for {len(all_bins)} bins (including candidates)...")

    for b in all_bins:
        members = clusters_df[clusters_df['cluster'] == b]['contig'].values
        valid_members = [c for c in members if c in embeddings_df.index]
        
        if not valid_members: continue
        
        # Calculate size
        size = sum(contig_lengths.get(c, 0) for c in members)
        bin_sizes[b] = size
        
        # Calculate weighted centroid
        vecs = embeddings_df.loc[valid_members].values
        weights = np.array([contig_lengths.get(c, 1000) for c in valid_members]).reshape(-1, 1)
        
        # Store un-normalized weighted sum and total weight for dynamic updates
        weighted_sum = np.sum(vecs * weights, axis=0)
        total_weight = weights.sum()
        
        bin_weighted_sums[b] = weighted_sum
        bin_total_weights[b] = total_weight
        
        bin_centroids[b] = weighted_sum / total_weight

    # Sort bins by size (smallest first) so we merge small into large
    # Filter out bins that had no valid embeddings (not in bin_centroids)
    sorted_bins = sorted(bin_centroids.keys(), key=lambda b: bin_sizes[b])

    logger.info(f"Attempting merge on {len(sorted_bins)} bins/fragments (Threshold > {similarity_threshold})...")

    merged_count = 0
    merged_map = {} # source -> target (for tracking chains if needed, though we do single pass)
    final_clusters = clusters_df['cluster'].copy()
    
    # Keep track of "active" bin members to calculate cumulative SCG stats correctly
    # Initialize with current members
    bin_members_map = {b: list(clusters_df[clusters_df['cluster'] == b]['contig'].values) for b in sorted_bins}

    for source_bin in sorted_bins:
        # If this bin has already been merged into something else, skip it
        # (Though with smallest-to-largest sort, we usually haven't processed it as a target yet)
        if source_bin in merged_map: continue
        
        source_vec = bin_centroids[source_bin].reshape(1, -1)
        
        best_target = None
        best_score = -1.0
        
        # Compare against all LARGER bins
        for target_bin in sorted_bins:
            if source_bin == target_bin: continue
            if target_bin in merged_map: continue # Don't merge into a bin that's already gone
            
            # CRITICAL: Only merge into Core Bins (original clusters). 
            # Do not merge debris into debris (singleton -> singleton).
            if target_bin not in core_bins: continue

            # Only merge into strictly larger bins (or equal, tie-break by name) to maintain stability
            # and ensure flow towards anchors.
            if bin_sizes[target_bin] < bin_sizes[source_bin]: continue
            
            target_vec = bin_centroids[target_bin].reshape(1, -1)
            sim = cosine_similarity(source_vec, target_vec)[0][0]
            
            if sim > best_score:
                best_score = sim
                best_target = target_bin
        
        if best_target and best_score >= similarity_threshold:
            # Check SCG Safety
            source_members = bin_members_map[source_bin]
            target_members = bin_members_map[best_target]
            
            current_dup, _ = get_bin_scg_stats(target_members, gene_mappings_cache)
            # Hypothetical merge
            new_dup, _ = get_bin_scg_stats(target_members + source_members, gene_mappings_cache)
            
            if (new_dup - current_dup) < max_duplication_increase:
                # MERGE!
                # Don't log every single noise merge, it's too verbose
                if not source_bin.startswith("singleton_"):
                    logger.info(f"Merging {source_bin} ({bin_sizes[source_bin]/1e6:.2f}Mb) -> {best_target} ({bin_sizes[best_target]/1e6:.2f}Mb) | Sim: {best_score:.3f} | Dup: {current_dup:.1f}%->{new_dup:.1f}%")
                
                # Update final clusters
                mask = final_clusters == source_bin
                final_clusters[mask] = best_target
                
                # Mark as merged
                merged_map[source_bin] = best_target
                merged_count += 1
                
                # Update target bin members map so future merges into this target see the accumulated genes
                bin_members_map[best_target].extend(source_members)
                
                # Update size estimate for future iterations? 
                # Yes, technically the target is now bigger.
                bin_sizes[best_target] += bin_sizes[source_bin]
                
                # Update Target Centroid
                bin_weighted_sums[best_target] += bin_weighted_sums[source_bin]
                bin_total_weights[best_target] += bin_total_weights[source_bin]
                bin_centroids[best_target] = bin_weighted_sums[best_target] / bin_total_weights[best_target]

    # Update dataframe with merged results
    clusters_df['cluster'] = final_clusters

    # Revert un-merged singletons back to 'noise'
    # Any cluster label starting with 'singleton_' is a failed rescue
    # Use string accessors on the Series
    final_mask = clusters_df['cluster'].astype(str).str.startswith('singleton_')
    if final_mask.any():
        clusters_df.loc[final_mask, 'cluster'] = 'noise'
    
    rescued_noise = n_noise_initial - final_mask.sum()
    
    if merged_count > 0:
        logger.info(f"Rescue complete: Merged {merged_count} total items (including {rescued_noise} noise contigs).")
    else:
        logger.info("Rescue complete: No safe merges found.")

    return clusters_df