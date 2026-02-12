"""
Rescue module for REMAG.
Implements "Satellite Rescue" strategy to merge fragmented bins based on embedding similarity and SCG safety.
"""

import json
import os

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

from .miniprot_utils import get_gene_mappings_cache_path

MAX_MERGE_DUPLICATION_PERCENT = 10.0


def get_bin_scg_stats(bin_contigs, gene_mappings_cache):
    """Calculate SCG duplication % for a list of contigs."""
    if not bin_contigs:
        return 0.0, 0.0

    bin_genes = {}
    for c in bin_contigs:
        if c in gene_mappings_cache:
            # gene_mappings_cache[c] is a dict of {gene_id: info}
            for gene_id in gene_mappings_cache[c].keys():
                bin_genes[gene_id] = bin_genes.get(gene_id, 0) + 1

    present = len(bin_genes)
    if present == 0:
        return 0.0, 0.0

    duplicated = len([g for g, count in bin_genes.items() if count > 1])
    duplication_rate = (duplicated / present) * 100.0
    return duplication_rate, present


def rescue_fragmented_bins(
    clusters_df,
    embeddings_df,
    fragments_dict,
    args,
    similarity_threshold=0.70,
    max_duplication_increase=5.0,
    max_total_duplication=5.0,
):
    """
    Attempt to merge smaller bins (or split parts of genomes) into larger "Core Bins"
    based on global embedding centroid similarity, provided it is safe (SCG check).
    """
    logger.info("Running Satellite Rescue to merge fragmented bins...")

    merge_duplication_cap = min(max_total_duplication, MAX_MERGE_DUPLICATION_PERCENT)

    # 1. Load Gene Mappings Cache
    gene_mappings_cache = getattr(args, "_gene_mappings_cache", None)
    if gene_mappings_cache is None:
        cache_path = get_gene_mappings_cache_path(args)
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                gene_mappings_cache = json.load(f)

    if not gene_mappings_cache:
        logger.warning(
            "No gene mappings cache found. Skipping rescue step as safety checks require SCGs."
        )
        return clusters_df

    # 2. Prepare Data
    # Calculate contig lengths
    contig_lengths = {k: len(v["sequence"]) for k, v in fragments_dict.items()}

    # Filter out noise for initial bin list
    valid_bins_df = clusters_df[clusters_df["cluster"] != "noise"].copy()
    all_bins = valid_bins_df["cluster"].unique()

    if len(all_bins) < 2:
        logger.info("Fewer than 2 bins. Skipping rescue.")
        return clusters_df

    # 3. Calculate Centroids for ALL Bins
    bin_centroids = {}
    bin_sizes = {}  # in bp

    logger.debug(f"Calculating centroids for {len(all_bins)} bins...")

    for b in all_bins:
        members = valid_bins_df[valid_bins_df["cluster"] == b]["contig"].values
        valid_members = [c for c in members if c in embeddings_df.index]

        if not valid_members:
            continue

        # Calculate size
        size = sum(contig_lengths.get(c, 0) for c in members)
        bin_sizes[b] = size

        # Calculate weighted centroid
        vecs = embeddings_df.loc[valid_members].values
        weights = np.array(
            [contig_lengths.get(c, 1000) for c in valid_members]
        ).reshape(-1, 1)
        weights = weights / weights.sum()
        centroid = np.sum(vecs * weights, axis=0)

        bin_centroids[b] = centroid

    # Sort bins by size (smallest first) so we merge small into large
    # Filter out bins that had no valid embeddings (not in bin_centroids)
    sorted_bins = sorted(bin_centroids.keys(), key=lambda b: bin_sizes[b])

    logger.info(
        f"Attempting merge on {len(sorted_bins)} bins (Threshold > {similarity_threshold})..."
    )

    merged_count = 0
    merged_map = (
        {}
    )  # source -> target (for tracking chains if needed, though we do single pass)
    final_clusters = clusters_df["cluster"].copy()

    # Keep track of "active" bin members to calculate cumulative SCG stats correctly
    # Initialize with current members
    bin_members_map = {
        b: list(valid_bins_df[valid_bins_df["cluster"] == b]["contig"].values)
        for b in sorted_bins
    }

    for source_bin in sorted_bins:
        # If this bin has already been merged into something else, skip it
        # (Though with smallest-to-largest sort, we usually haven't processed it as a target yet)
        if source_bin in merged_map:
            continue

        source_vec = bin_centroids[source_bin].reshape(1, -1)

        best_target = None
        best_score = -1.0

        # Compare against all LARGER bins
        for target_bin in sorted_bins:
            if source_bin == target_bin:
                continue
            if target_bin in merged_map:
                continue  # Don't merge into a bin that's already gone

            # Only merge into strictly larger bins (or equal, tie-break by name) to maintain stability
            # and ensure flow towards anchors.
            if bin_sizes[target_bin] < bin_sizes[source_bin]:
                continue

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
            new_dup, _ = get_bin_scg_stats(
                target_members + source_members, gene_mappings_cache
            )

            if (
                new_dup - current_dup
            ) < max_duplication_increase and new_dup <= merge_duplication_cap:
                # MERGE!
                logger.info(
                    f"Merging {source_bin} ({bin_sizes[source_bin]/1e6:.2f}Mb) -> {best_target} ({bin_sizes[best_target]/1e6:.2f}Mb) | Sim: {best_score:.3f} | Dup: {current_dup:.1f}%->{new_dup:.1f}%"
                )

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

    if merged_count > 0:
        logger.info(f"Rescue complete: Merged {merged_count} bins.")
        # Update dataframe
        clusters_df["cluster"] = final_clusters
    else:
        logger.info("Rescue complete: No safe merges found.")

    # 4. Singleton Rescue (NEW)
    # Attempt to assign unbinned "noise" contigs to the best matching bin
    # Re-calculate valid bins and centroids after the merge step

    # Get current valid bins (excluding noise)
    current_valid_bins = clusters_df[clusters_df["cluster"] != "noise"][
        "cluster"
    ].unique()

    if len(current_valid_bins) == 0:
        return clusters_df

    logger.info("Running Singleton Rescue for unbinned contigs...")

    # Calculate updated centroids
    updated_centroids = {}
    updated_bin_members = {}

    for b in current_valid_bins:
        members = clusters_df[clusters_df["cluster"] == b]["contig"].values
        valid_members = [c for c in members if c in embeddings_df.index]

        if not valid_members:
            continue

        updated_bin_members[b] = list(members)

        # Calculate weighted centroid
        vecs = embeddings_df.loc[valid_members].values
        weights = np.array(
            [contig_lengths.get(c, 1000) for c in valid_members]
        ).reshape(-1, 1)
        weights = weights / weights.sum()
        centroid = np.sum(vecs * weights, axis=0)
        updated_centroids[b] = centroid

    # Identify noise contigs
    noise_contigs = clusters_df[clusters_df["cluster"] == "noise"]["contig"].values
    valid_noise_contigs = [c for c in noise_contigs if c in embeddings_df.index]

    logger.info(f"Found {len(valid_noise_contigs)} unbinned contigs with embeddings.")

    rescued_singletons = 0

    for contig in valid_noise_contigs:
        contig_vec = embeddings_df.loc[contig].values.reshape(1, -1)

        best_target = None
        best_score = -1.0

        # Find best matching bin
        for target_bin, centroid in updated_centroids.items():
            target_vec = centroid.reshape(1, -1)
            sim = cosine_similarity(contig_vec, target_vec)[0][0]

            if sim > best_score:
                best_score = sim
                best_target = target_bin

        # Check threshold and safety
        if best_target and best_score >= similarity_threshold:
            target_members = updated_bin_members[best_target]

            # SCG Safety Check
            # Only need to check if this contig has genes that conflict with the bin
            contig_genes = gene_mappings_cache.get(contig, {}).keys()
            if not contig_genes:
                # No core genes -> Safe to add (won't increase duplication count)
                is_safe = True
            else:
                # Check for conflicts
                current_dup, _ = get_bin_scg_stats(target_members, gene_mappings_cache)
                new_dup, _ = get_bin_scg_stats(
                    target_members + [contig], gene_mappings_cache
                )

                if (
                    new_dup - current_dup
                ) < max_duplication_increase and new_dup <= max_total_duplication:
                    is_safe = True
                else:
                    is_safe = False

            if is_safe:
                # Assign to bin
                logger.debug(
                    f"Rescued singleton {contig} -> {best_target} (Sim: {best_score:.3f})"
                )
                clusters_df.loc[clusters_df["contig"] == contig, "cluster"] = (
                    best_target
                )
                updated_bin_members[best_target].append(
                    contig
                )  # Update local list for subsequent checks
                rescued_singletons += 1

    logger.info(
        f"Singleton Rescue complete: Assigned {rescued_singletons} contigs to existing bins."
    )

    return clusters_df
